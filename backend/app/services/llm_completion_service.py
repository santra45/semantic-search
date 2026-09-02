"""
Generic single-shot LLM completion service.

Used for short structured calls — primarily intent classification from the
Magento chatbot's LLMClassifier — where we want central token tracking and
provider abstraction without dragging in the heavier rerank prompt-building
logic. Reuses MODEL_PRICING and the token-usage helpers from
llm_rerank_service so cost accounting stays consistent across endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic
from google import genai
from openai import OpenAI
from groq import Groq

from backend.app.services.llm_rerank_service import (
    MODEL_PRICING,
    estimate_cost,
    get_token_usage,
    log_gemini_request,
    log_gemini_response,
    make_http_client,
)
from backend.app.services import usage_service
from backend.app.utils.llm_logger import log_llm_interaction

logger = logging.getLogger("llm_completion")

DEFAULT_MODELS = {
    "gemini":    "gemini-2.5-flash",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.1-8b-instant",
}


def complete(
    prompt: str,
    *,
    json_mode: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.0,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client_id: str = "anonymous",
    query_type: str = "chat_intent",
) -> tuple[str, dict]:
    """
    Single-shot completion. Returns ``(text, usage)`` where ``usage`` is
    ``{"input": int, "output": int, "cost": float, "provider": str, "model": str}``.

    Writes one usage_events row per call with the supplied query_type as its
    call_type, so cost rolls up alongside the other LLM operations. Callers
    that want to surface per-call cost to a downstream client (e.g. the
    Magento chatbot's per-message billing column) can use the returned
    ``usage`` dict directly.
    """
    provider = (provider or "gemini").lower()
    model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["gemini"])

    response = None
    response_text = ""

    if provider == "gemini":
        log_gemini_request(model, prompt)
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        gen_config: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            # Gemini honours this on 1.5+ and silently ignores on older.
            gen_config["responseMimeType"] = "application/json"
        # Disable the thinking phase on the 2.5 Flash family — these single-shot
        # completions (decomposition, legacy classify) need no reasoning pass.
        # Typed ThinkingConfig (needs the modern google-genai SDK — see the pin
        # in requirements.txt); a camelCase dict is rejected as extra_forbidden.
        _m = (model or "").lower()
        if "2.5" in _m and "flash" in _m:
            gen_config["thinking_config"] = genai.types.ThinkingConfig(thinking_budget=0)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gen_config,
        )
        log_gemini_response(response)
        response_text = (response.text or "").strip()

    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI requires api_key")
        client = OpenAI(api_key=api_key, http_client=make_http_client())
        kwargs: dict = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        response_text = (response.choices[0].message.content or "").strip()

    elif provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic requires api_key")
        client = anthropic.Anthropic(api_key=api_key, http_client=make_http_client())
        # Anthropic doesn't have an explicit json_mode flag — caller's prompt
        # should already instruct the model. We just pass through.
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # content is a list of typed blocks; concatenate any text blocks.
        parts = []
        for block in (response.content or []):
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        response_text = ("".join(parts)).strip()

    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq requires api_key")
        client = Groq(api_key=api_key, http_client=make_http_client())
        groq_kwargs: dict = {
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if json_mode:
            groq_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**groq_kwargs)
        response_text = (response.choices[0].message.content or "").strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # ── Token + cost accounting (mirrors llm_rerank_service) ───────────────
    usage = get_token_usage(provider, response, prompt, response_text)
    cost = estimate_cost(model, usage)

    logger.info(f"🔢 {query_type} usage: {usage} cost: ${round(cost, 8)}")

    log_llm_interaction(
        provider=provider,
        model=model,
        purpose=query_type,
        prompt=prompt,
        response_text=response_text,
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        cost=cost,
        client_id=client_id,
    )

    # The tenant comes from the request context, not from `client_id`: this
    # service is handed a bare client_id, which cannot name the site,
    # subscription and product a usage_events row requires. track() reads what
    # the auth chokepoint bound for the request and opens its own short-lived
    # session, so nothing here grows a parameter.
    #
    # kind is always 'serve'. All three call_types that reach this function —
    # chat_intent and chat_rewrite from /magento/chatbot/classify, and
    # chat_query_decompose from query_decomposer — run while a shopper waits.
    # Nothing on an indexing path calls complete().
    #
    # billable stays False. This is one step of a turn, not the turn: the
    # answer that follows carries the single billable row, and flagging the
    # classification too would spend two of a merchant's monthly requests on
    # one question. Note that the query_decomposer path is reached from
    # /magento/search, which has no chokepoint at all, so those rows will log
    # NO CONTEXT and be refused until AI Search gets one.
    #
    # Swallowed but logged with the tenant and the amount: the completion is
    # already paid for by this point, so accounting must not be what fails the
    # request, and the log line is what keeps the spend recoverable.
    try:
        usage_service.track(
            query_type,
            provider,
            model,
            usage["input"],
            usage["output"],
            usage["input"]  * MODEL_PRICING.get(model, {}).get("input",  0),
            usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0),
            usage_service.KIND_SERVE,
        )
    except Exception as e:
        logger.warning(
            "usage not recorded for %s (client=%s %s/%s tokens in=%s out=%s): %s",
            query_type, client_id, provider, model,
            usage["input"], usage["output"], e,
        )

    return response_text, {
        "input":    int(usage["input"]),
        "output":   int(usage["output"]),
        "cost":     float(cost),
        "provider": provider,
        "model":    model,
    }
