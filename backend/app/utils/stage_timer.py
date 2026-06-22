"""Per-request sub-stage timing for the chatbot retrieve handlers.

Mirrors the Magento-side RequestTimer. The X-Request-Id header the Magento
ApiClient sets rides in on every chat-pipeline call, so the line emitted here
shares an id with the [Timeline] line in Magento's ai_assistant.log and the
browser-console line. Grep one id and you get the whole journey across all
three tiers, with this line breaking the backend's share of the time into its
individual stages (embed / qdrant / mmr / rerank).

Marks are DELTA-based: each mark() records the time elapsed since the previous
mark, so the line reads directly as "where did this endpoint's milliseconds
go" rather than as offsets from the start.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from starlette.requests import Request

from backend.app.utils.logging_config import api_logger


class StageTimer:
    """Lightweight delta timer; one per request handler invocation."""

    def __init__(self, label: str, request: Optional[Request] = None, rid: str = ""):
        self.label = label
        self.rid = (rid or _rid_from_request(request) or "-")[:64]
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._stages: List[Tuple[str, int]] = []

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under `name`."""
        now = time.perf_counter()
        self._stages.append((name, int((now - self._last) * 1000)))
        self._last = now

    def flush(self) -> None:
        """Emit the assembled per-stage line to logs/api.log."""
        total = int((time.perf_counter() - self._t0) * 1000)
        parts = " ".join(f"{name}={ms}ms" for name, ms in self._stages) or "(none)"
        api_logger.info(f"[Timeline] rid={self.rid} {self.label} total={total}ms | {parts}")


def _rid_from_request(request: Optional[Request]) -> str:
    if request is None:
        return ""
    try:
        return request.headers.get("x-request-id", "") or ""
    except Exception:
        return ""
