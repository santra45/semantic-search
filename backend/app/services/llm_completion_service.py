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

from backend.app.services.llm_rerank_service import (
    MODEL_PRICING,
    estimate_cost,
    get_token_usage,
    log_gemini_request,
    log_gemini_response,
    make_http_client,
)
from backend.app.services.token_usage_service import track_usage
from backend.app.utils.llm_logger import log_llm_interaction

logger = logging.getLogger("llm_completion")

DEFAULT_MODELS = {
    "gemini":    "gemini-2.5-flash",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
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

    Tracks one row per call in token_usage_tracking with the supplied
    query_type so cost rolls up alongside the other LLM operations. Callers
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

    try:
        track_usage(
            client_id=client_id,
            query_type=query_type,
            llm_provider=provider,
            llm_model=model,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            input_cost=usage["input"]  * MODEL_PRICING.get(model, {}).get("input",  0),
            output_cost=usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0),
            request_text_length=len(prompt),
            response_text_length=len(response_text),
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to track usage for {query_type}: {e}")

    return response_text, {
        "input":    int(usage["input"]),
        "output":   int(usage["output"]),
        "cost":     float(cost),
        "provider": provider,
        "model":    model,
    }
