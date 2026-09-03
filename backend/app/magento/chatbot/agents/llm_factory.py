"""
Build a LangChain chat model per-request.

Resolution order for provider+model+api_key:
  1. Explicit overrides (from request — the Magento module forwards its configured
     provider, model, and the encrypted LLM API key decrypted by the caller).
  2. Environment defaults (CHAT_LLM_PROVIDER / CHAT_LLM_MODEL / GEMINI_API_KEY).
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.config import GEMINI_API_KEY
from backend.app.magento.chatbot.services.config import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

# The prefixes Google issues its API keys under. Published strings, not secrets,
# which is what lets this check exist at all — and it is worth having because
# the errors Google returns for a value that is not a key are unrecognisable
# from the outside. A blank key reaches Application Default Credentials and dies
# as "DefaultCredentialsError"; a non-empty non-key comes back as "Expected
# OAuth 2 access token ... ACCESS_TOKEN_TYPE_UNSUPPORTED", which reads like a
# broken integration rather than a mis-typed setting and sends whoever is on
# call into the transport layer.
#
# BOTH ENTRIES ARE LOAD-BEARING, and the second one was learned the hard way.
# "AIza" is the 39-character AI Studio key everyone recognises; the newer format
# is longer and shares none of that shape. A check that knew only about "AIza"
# fired this warning against a live merchant key that Google then accepted
# without complaint — a false alarm on the exact screen an operator reads while
# hunting a real auth failure, which is worse than no check at all.
#
# So when this list is wrong the cost is misdirection, not a broken request:
# nothing here gates the call. An unrecognised prefix means "this may be why the
# next line is a 401", never "do not try".
_GOOGLE_KEY_PREFIXES = ("AIza", "AQ.")


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

    # WHICH KEY IS ABOUT TO BE SPENT, AND DOES IT LOOK LIKE A KEY AT ALL.
    #
    # `api_key or GEMINI_API_KEY` is the third of the three silent fall-throughs
    # llm_key_service's module docstring enumerates, and it is the only one with
    # no log line of its own: a merchant whose blob stops opening keeps working
    # on the server's budget and nothing says so. Worse, a deployment whose
    # GEMINI_API_KEY is still the .env placeholder turns that safety net into a
    # trap — the fallback fires, the placeholder goes to Google, and the merchant
    # sees an authentication error naming a key they never configured.
    #
    # NEITHER BRANCH PRINTS KEY MATERIAL. Length, whether one of the published
    # prefixes above is present, and whether the value survives its own strip()
    # are enough to tell a missing key from a stale blob from a
    # pasted-with-newline key, and none of the three is a secret.
    key = api_key or GEMINI_API_KEY or ""
    source = "merchant (X-LLM-API-Key-Encrypted)" if api_key else "server env GEMINI_API_KEY"
    if not key:
        logger.error(
            "chat LLM: NO Google key at all — the merchant's blob did not open "
            "and GEMINI_API_KEY is unset. The request will fail inside "
            "google.auth as DefaultCredentialsError, which names neither cause."
        )
    elif not key.startswith(_GOOGLE_KEY_PREFIXES) or key != key.strip():
        logger.warning(
            "chat LLM: the key from %s does not match a Google API key prefix "
            "this build knows about (len=%d known_prefix=%s "
            "has_surrounding_whitespace=%s). If the next line is a 401, the key "
            "is the likeliest cause and Google's message will not say so. If it "
            "is not, Google has issued a format newer than %s and this warning "
            "is the thing that is wrong.",
            source, len(key), key.startswith(_GOOGLE_KEY_PREFIXES),
            key != key.strip(), ", ".join(_GOOGLE_KEY_PREFIXES),
        )
    elif not api_key:
        logger.warning(
            "chat LLM: falling back to the SERVER's GEMINI_API_KEY — this "
            "tenant's generation spend lands on our budget. Either they have "
            "configured no LLM key, or theirs no longer unwraps (see the "
            "llm_key_service warning that precedes this line if so)."
        )

    kwargs = {
        "model": resolved_model,
        "google_api_key": key,
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
