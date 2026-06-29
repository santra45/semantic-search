"""Shared Gemini helpers.

Single source of truth for the "can we disable this model's thinking phase?"
decision, used by every Gemini call site (the LangChain factory, the rerank
service, and the single-shot completion service) so the gate logic doesn't
drift between them.
"""

from __future__ import annotations

from typing import Optional


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
