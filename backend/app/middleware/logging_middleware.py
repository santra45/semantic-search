"""
Starlette middleware that logs every HTTP request + response to logs/api.log.

Readability is the top priority:
  * One box per exchange, easy to eyeball on `tail -f`
  * Secrets (Authorization / API-key / creds headers) redacted
  * JSON bodies pretty-printed, oversized ones truncated with a marker
  * Binary / non-UTF8 bodies noted explicitly rather than dumped raw
"""

from __future__ import annotations

import json
import time
from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from backend.app.utils.logging_config import api_logger


SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-llm-api-key-encrypted",
    "x-magento-admin-creds-encrypted",
}

SKIP_PATHS: tuple[str, ...] = (
    "/api/health",        # liveness probe spam
    "/static/",           # static file noise
    "/favicon.ico",
)

MAX_BODY_CHARS = 4000
RULE = "=" * 78
INNER = "-" * 78


class APILoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path.startswith(p) for p in SKIP_PATHS):
            return await call_next(request)

        start = time.perf_counter()

        # ── Capture + re-inject request body ──────────────────────────────────
        req_body = await request.body()

        async def receive() -> Message:
            return {"type": "http.request", "body": req_body, "more_body": False}

        request = Request(request.scope, receive)
        request_block = self._format_request(request, req_body)

        # ── Pass to the app, capture response body ────────────────────────────
        response = await call_next(request)

        resp_chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            resp_chunks.append(chunk)
        resp_body = b"".join(resp_chunks)

        duration_ms = int((time.perf_counter() - start) * 1000)
        response_block = self._format_response(response, resp_body, duration_ms)

        api_logger.info("\n" + RULE + "\n" + request_block + "\n" + INNER + "\n" + response_block + "\n" + RULE)

        # Rebuild the response we've consumed.
        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    # ── Formatters ───────────────────────────────────────────────────────────

    def _format_request(self, request: Request, body: bytes) -> str:
        client_ip = request.client.host if request.client else "?"
        headers = _sanitize_headers(request.headers.items())
        query = dict(request.query_params) if request.query_params else {}

        lines = [
            f"📤 REQUEST  {request.method} {request.url.path}",
            f"   Client: {client_ip}",
        ]
        if query:
            lines.append(f"   Query: {_pretty(query)}")
        if headers:
            lines.append(f"   Headers: {_pretty(headers)}")
        lines.append(f"   Body:\n{_format_body(body)}")
        return "\n".join(lines)

    def _format_response(self, response: Response, body: bytes, duration_ms: int) -> str:
        status = response.status_code
        icon = "✅" if 200 <= status < 300 else ("⚠️ " if 300 <= status < 500 else "❌")
        lines = [
            f"{icon} RESPONSE  {status}  ({duration_ms} ms)",
            f"   Body:\n{_format_body(body)}",
        ]
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sanitize_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in pairs:
        if k.lower() in SENSITIVE_HEADERS:
            out[k] = "***REDACTED***"
        else:
            out[k] = v
    return out


def _format_body(body: bytes) -> str:
    if not body:
        return "     (empty)"
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return f"     (binary, {len(body)} bytes)"

    # Try pretty-printing JSON.
    stripped = text.strip()
    if stripped and stripped[0] in "{[":
        try:
            pretty = json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
            return _indent(pretty[:MAX_BODY_CHARS]) + ("\n     …[truncated]" if len(pretty) > MAX_BODY_CHARS else "")
        except json.JSONDecodeError:
            pass

    return _indent(text[:MAX_BODY_CHARS]) + ("\n     …[truncated]" if len(text) > MAX_BODY_CHARS else "")


def _pretty(payload: dict) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        return str(payload)


def _indent(text: str, prefix: str = "     ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
