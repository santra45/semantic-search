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
    if p in ("groq", "llama"):
        return "groq"
    return "google"


# Per-provider fallback when the caller names a provider but no model. Kept as
# data so resolve_provider_model() and build_llm() cannot disagree about what
# "the default" is — they used to be separate literals at each call site, and
# usage rows ended up attributed to a model that was never invoked.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "groq": "llama-3.3-70b-versatile",
}


def resolve_provider_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, str]:
    """The provider and model `build_llm` would actually construct.

    Call sites need this for usage accounting and cost lookup. Recording the
    request's raw `llm_model` instead means every request that leaves the model
    on "service default" is filed under whatever literal that call site happened
    to guess — priced at zero if that name isn't in the pricing table, and
    misattributed if the deployment sets CHAT_LLM_MODEL to something else.
    """
    p = _normalize_provider(provider)
    if p == "google":
        return p, model or DEFAULT_LLM_MODEL or "gemini-2.5-flash-lite"
    return p, model or _DEFAULT_MODELS[p]


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
    p, resolved_model = resolve_provider_model(provider, model)

    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
        )

    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
        )

    if p == "groq":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

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
