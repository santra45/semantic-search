"""
Build a LangChain chat model per-request.

Resolution order for provider+model+api_key:
  1. Explicit overrides (from request — the Magento module forwards its configured
     provider, model, and the encrypted LLM API key decrypted by the caller).
  2. Environment defaults (CHAT_LLM_PROVIDER / CHAT_LLM_MODEL / GEMINI_API_KEY).
"""

from __future__ import annotations

from typing import Optional

from backend.app.config import GEMINI_API_KEY
from backend.app.magento.chatbot.services.config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL


def _normalize_provider(provider: Optional[str]) -> str:
    p = (provider or DEFAULT_LLM_PROVIDER or "google").strip().lower()
    if p in ("gemini", "google", "google-genai"):
        return "google"
    if p in ("openai", "gpt"):
        return "openai"
    if p in ("anthropic", "claude"):
        return "anthropic"
    return "google"


def build_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
):
    """Return a LangChain chat model. Imports are deferred so the backend boots without
    LangChain when no chat request has been served yet (helpful for minimal deployments
    that disable the chatbot)."""
    p = _normalize_provider(provider)

    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
        )

    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-sonnet-4-6",
            api_key=api_key,
            temperature=temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    resolved_model = model or DEFAULT_LLM_MODEL or "gemini-2.0-flash-lite"
    kwargs = {
        "model": resolved_model,
        "google_api_key": api_key or GEMINI_API_KEY,
        "temperature": temperature,
        "convert_system_message_to_human": True,
    }
    # Gemini 2.5 Flash defaults thinking ON — the reasoning pass adds several
    # seconds of dead air before the first streamed token on the answer model,
    # and neither routing nor answer generation needs it. thinking_budget=0
    # disables it. This rides langchain-google-genai's google-ai-generativelanguage
    # transport, independent of the separate google-genai SDK used by the raw
    # rerank client — so it is unaffected by that SDK's version. Applied only to
    # the 2.5 Flash family, which accepts a zero budget (Flash-Lite already
    # defaults off, so budget=0 is a harmless no-op there).
    _m = resolved_model.lower()
    if "2.5" in _m and "flash" in _m:
        kwargs["thinking_budget"] = 0
    return ChatGoogleGenerativeAI(**kwargs)
