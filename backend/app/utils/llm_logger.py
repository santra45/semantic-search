"""
Uniform LLM-call logger — every outbound call to Gemini / OpenAI / Anthropic
(or any future provider) should go through one of the two entry points:

    log_llm_interaction(provider, model, purpose, prompt, response_text,
                        input_tokens, output_tokens, cost, client_id, error)
        Fire once a call completes, with the numbers already in hand.

    with log_llm_call(provider=…, model=…, purpose=…, client_id=…, prompt=…) as ctx:
        result = ...
        ctx.record(response_text=..., input_tokens=..., output_tokens=..., cost=...)
        Context-manager variant — automatically records duration + catches
        exceptions + logs on exit.

Both produce the same readable block in logs/llm.log:

    ══════════════════════════════════════════════════════════════════════════
    🤖 LLM CALL  ✅ SUCCESS   gemini / gemini-2.5-flash   purpose=product_rerank
       Client: abc-123         Duration: 342 ms
       Tokens: in=520 out=42 total=562
       Cost:   $0.000156
    ──────────────────────────────────────────────────────────────────────────
    📤 PROMPT:
         You are an expert content recommendation assistant...
    ──────────────────────────────────────────────────────────────────────────
    📥 RESPONSE:
         ["123", "456", "789"]
    ══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from backend.app.utils.logging_config import llm_logger

MAX_PROMPT_CHARS = 4000
MAX_RESPONSE_CHARS = 4000
RULE = "=" * 78
INNER = "-" * 78


def log_llm_interaction(
    *,
    provider: str,
    model: str,
    purpose: str,
    prompt: str = "",
    response_text: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost: float = 0.0,
    client_id: str = "anonymous",
    duration_ms: Optional[int] = None,
    error: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Single-shot: write one complete LLM call entry to logs/llm.log."""
    total = (input_tokens or 0) + (output_tokens or 0)
    status = "❌ ERROR" if error else "✅ SUCCESS"

    header = (
        f"🤖 LLM CALL  {status}   {provider} / {model}   purpose={purpose}"
    )

    meta_lines = [
        f"   Client:   {client_id}",
    ]
    if duration_ms is not None:
        meta_lines[0] += f"        Duration: {duration_ms} ms"
    meta_lines.append(f"   Tokens:   in={input_tokens}  out={output_tokens}  total={total}")
    meta_lines.append(f"   Cost:     ${cost:.8f}")
    if extra:
        import json as _json
        meta_lines.append("   Extra:    " + _json.dumps(extra, ensure_ascii=False))

    prompt_block = _indent(_truncate(prompt, MAX_PROMPT_CHARS), "     ")
    if error:
        response_block = _indent(f"ERROR: {error}", "     ")
    else:
        response_block = _indent(_truncate(response_text, MAX_RESPONSE_CHARS), "     ")

    body = (
        "\n" + RULE + "\n"
        + header + "\n"
        + "\n".join(meta_lines) + "\n"
        + INNER + "\n"
        + "📤 PROMPT:\n" + prompt_block + "\n"
        + INNER + "\n"
        + "📥 RESPONSE:\n" + response_block + "\n"
        + RULE
    )

    if error:
        llm_logger.error(body)
    else:
        llm_logger.info(body)


class _CallContext:
    """Mutable bag the caller fills in during an LLM call."""
    __slots__ = ("response_text", "input_tokens", "output_tokens", "cost", "error", "extra")

    def __init__(self) -> None:
        self.response_text: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost: float = 0.0
        self.error: Optional[str] = None
        self.extra: Optional[dict] = None

    def record(
        self,
        *,
        response_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        extra: Optional[dict] = None,
    ) -> None:
        self.response_text = response_text
        self.input_tokens = int(input_tokens or 0)
        self.output_tokens = int(output_tokens or 0)
        self.cost = float(cost or 0.0)
        if extra:
            self.extra = extra


@contextmanager
def log_llm_call(
    *,
    provider: str,
    model: str,
    purpose: str,
    prompt: str = "",
    client_id: str = "anonymous",
):
    """Context-manager variant. Example:

        with log_llm_call(provider="gemini", model=m, purpose="chat_answer",
                          prompt=p, client_id=cid) as ctx:
            resp = client.generate(...)
            ctx.record(response_text=resp.text,
                       input_tokens=resp.usage.in,
                       output_tokens=resp.usage.out,
                       cost=estimated)
    """
    ctx = _CallContext()
    t0 = time.perf_counter()
    try:
        yield ctx
    except Exception as exc:
        ctx.error = str(exc)
        raise
    finally:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        log_llm_interaction(
            provider=provider,
            model=model,
            purpose=purpose,
            prompt=prompt,
            response_text=ctx.response_text,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            cost=ctx.cost,
            client_id=client_id,
            duration_ms=duration_ms,
            error=ctx.error,
            extra=ctx.extra,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


def _indent(text: str, prefix: str = "     ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines()) if text else (prefix + "(empty)")
