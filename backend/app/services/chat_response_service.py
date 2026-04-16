from __future__ import annotations

import json
import logging
from typing import Any

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

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL = "I could not confirm that from this store's information."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def build_grounded_prompt(
    message: str,
    sources: list[dict[str, Any]],
    conversation_history: list[dict[str, str]],
) -> str:
    history_lines = []
    for item in conversation_history[-8:]:
        role = item.get("role", "user").upper()
        content = item.get("content", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")

    rendered_sources = []
    for index, source in enumerate(sources, start=1):
        rendered_sources.append(
            json.dumps(
                {
                    "source_id": index,
                    "content_type": source.get("content_type"),
                    "title": source.get("title"),
                    "excerpt": source.get("excerpt"),
                    "permalink": source.get("permalink"),
                },
                ensure_ascii=True,
            )
        )

    return f"""
You are a grounded ecommerce support assistant.

Rules:
- Answer ONLY from the supplied store evidence.
- If the evidence does not support a claim, say exactly: "{DEFAULT_REFUSAL}"
- Keep the answer concise and factual.
- Never invent pricing, policy details, shipping claims, or availability.
- Return valid JSON only with this shape:
  {{
    "answer": "string",
    "follow_up_suggestions": ["string", "string"],
    "grounded": true
  }}

Conversation history:
{chr(10).join(history_lines) if history_lines else "No prior messages."}

Customer message:
{message}

Evidence:
{chr(10).join(rendered_sources) if rendered_sources else "No evidence."}
""".strip()


def _resolve_provider_defaults(llm_provider: str | None, llm_model: str | None) -> tuple[str, str]:
    provider = (llm_provider or "gemini").strip().lower()
    defaults = {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    return provider, (llm_model or defaults.get(provider, defaults["gemini"])).strip()


def _track_answer_usage(
    client_id: str,
    provider: str,
    model: str,
    usage: dict[str, Any],
    prompt: str,
    response_text: str,
) -> dict[str, Any]:
    cost = estimate_cost(model, usage)
    input_cost = usage["input"] * MODEL_PRICING.get(model, {}).get("input", 0)
    output_cost = usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0)

    try:
        track_usage(
            client_id=client_id,
            query_type="chat_answer",
            llm_provider=provider,
            llm_model=model,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            input_cost=input_cost,
            output_cost=output_cost,
            request_text_length=len(prompt),
            response_text_length=len(response_text),
        )
    except Exception as exc:
        logger.warning("Failed to track chat answer usage: %s", exc)

    return {
        "input_tokens": usage["input"],
        "output_tokens": usage["output"],
        "total_tokens": usage["total"],
        "cost": round(cost, 8),
        "llm_provider": provider,
        "llm_model": model,
    }


def generate_grounded_answer(
    message: str,
    sources: list[dict[str, Any]],
    conversation_history: list[dict[str, str]],
    llm_provider: str | None,
    llm_model: str | None,
    llm_api_key: str | None,
    client_id: str,
) -> dict[str, Any]:
    if not sources:
        return {
            "answer": DEFAULT_REFUSAL,
            "follow_up_suggestions": [],
            "grounded": False,
            "usage": {},
        }

    if not llm_api_key:
        raise ValueError("LLM API key is required for chatbot responses.")

    provider, model = _resolve_provider_defaults(llm_provider, llm_model)
    prompt = build_grounded_prompt(message, sources, conversation_history)

    response = None
    response_text = ""

    if provider == "gemini":
        log_gemini_request(model, prompt)
        client = genai.Client(api_key=llm_api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        log_gemini_response(response)
        response_text = response.text.strip()
    elif provider == "openai":
        client = OpenAI(api_key=llm_api_key, http_client=make_http_client())
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        response_text = response.choices[0].message.content.strip()
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=llm_api_key, http_client=make_http_client())
        response = client.messages.create(
            model=model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text.strip()
    else:
        raise ValueError(f"Unsupported llm provider: {provider}")

    usage = get_token_usage(provider, response, prompt, response_text)
    tracked_usage = _track_answer_usage(client_id, provider, model, usage, prompt, response_text)
    payload = _extract_json_object(response_text) or {}

    answer = str(payload.get("answer") or "").strip() or DEFAULT_REFUSAL
    follow_ups = payload.get("follow_up_suggestions") or []
    if not isinstance(follow_ups, list):
        follow_ups = []

    grounded = bool(payload.get("grounded", True)) and answer != DEFAULT_REFUSAL

    return {
        "answer": answer,
        "follow_up_suggestions": [str(item).strip() for item in follow_ups if str(item).strip()][:3],
        "grounded": grounded,
        "usage": tracked_usage,
        "raw_response": response_text,
    }
