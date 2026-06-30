"""Shared Gemini helpers.

Single source of truth for the "can we disable this model's thinking phase?"
decision, used by every Gemini call site (the LangChain factory, the rerank
service, and the single-shot completion service) so the gate logic doesn't
drift between them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=1)
def _thinking_budget_supported() -> bool:
    """True when the installed google-genai SDK's ThinkingConfig accepts a
    `thinking_budget`. Older SDKs ship a ThinkingConfig WITHOUT that field and
    reject it as `extra_forbidden`, which would crash the call. Detected once."""
    try:
        from google import genai
        genai.types.ThinkingConfig(thinking_budget=0)
        return True
    except Exception:
        return False


def gemini_thinking_off() -> Optional[Any]:
    """A ``ThinkingConfig(thinking_budget=0)`` for the raw ``genai.Client`` paths
    (rerank, completion), or ``None`` when the installed SDK is too old to
    support it. Returning None lets callers disable thinking where the SDK
    supports it and degrade gracefully — thinking stays on, but no crash —
    where it doesn't (pin/upgrade ``google-genai`` to actually disable it
    there). Never raises."""
    if not _thinking_budget_supported():
        return None
    from google import genai
    return genai.types.ThinkingConfig(thinking_budget=0)


def thinking_can_be_disabled(model: Optional[str]) -> bool:
    """True when ``model`` accepts a zero thinking budget.

    Gemini 2.5 Flash and Flash-Lite let us turn the "thinking" (reasoning)
    phase fully off with ``thinking_budget=0``. Routing, reranking, and answer
    generation don't need a reasoning pass, and leaving thinking on the default
    (ON for 2.5 Flash) was adding seconds of dead time before the first token.

    Excluded on purpose:
      * 2.5 Pro — requires a non-zero budget and rejects 0.
      * 2.0 / 1.5 models — have no thinking phase and reject a thinking_config
        outright (a 400 from the API), which would break the call.

    So we only signal "disable" for the flash family, where budget=0 is valid.
    """
    m = (model or "").lower()
    return "2.5" in m and "flash" in m
