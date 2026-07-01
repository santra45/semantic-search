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

    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_LLM_MODEL or "gemini-2.0-flash-lite",
        google_api_key=api_key or GEMINI_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
