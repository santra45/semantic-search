"""Fail-soft query helpers and the counting rules the read API must obey.

Ported from routers/operator.py, which has run these against a live database for
months. Same shapes, same names, so a query can be moved between the two without
a rewrite while both consoles exist.

────────────────────────────────────────────────────────────────────────────
READS FAIL SOFT, WRITES FAIL LOUD.

A read that raises takes a whole dashboard down over one broken panel. A read
that returns [] degrades that panel and leaves the other nine standing. Writes
do the opposite — see admin/audit.py, where a failure rolls back and reports.

The trap this creates, and the reason _i_or_none exists: a fail-soft read that
returns 0 is indistinguishable from a genuine zero. Everywhere the difference
is knowable, absence must survive to the JSON as null, and the UI renders "—"
rather than "0". "This platform costs nothing" and "we could not measure what
this platform costs" are not the same sentence.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── The counting rules (ADMIN_CONSOLE_PLAN.md §3.3) ──────────────────────────
#
# Get these wrong and the charts do not break — they lie plausibly, which is
# worse. Written as constants so every endpoint spells them the same way.
#
#   REQUESTS  COUNT(*) WHERE billable = 1.  One customer action writes ONE
#             billable row plus several non-billable ones for embeddings,
#             reranks and tool calls. Counting every row inflates requests ~4x.
#
#   COST      SUM(total_cost) across EVERY row, billable or not. The
#             non-billable rows are where most of the money actually goes.
#
# So the two headline numbers come from different row sets over the same table,
# and any query that computes both with one WHERE clause has one of them wrong.
BILLABLE_REQUESTS = "SUM(CASE WHEN billable = 1 THEN 1 ELSE 0 END)"
ALL_ROWS_COST = "SUM(total_cost)"

# usage_ledger_read.LEDGER unions the v1 archive with usage_events and is the
# only honest source for cost/token totals spanning the migration. It
# deliberately does NOT project `billable` or `product_code` — v1 has no
# equivalent and projecting them would render "unknowable" as "zero".
#
# Consequence for every endpoint here: PER-PRODUCT and BILLABLE-REQUEST figures
# come from usage_events directly and cover the v2 era only; cost and token
# TOTALS come from LEDGER and cover both. Mixing them in one row without saying
# which is which produces a table where the columns describe different periods.
LEDGER_IS_V2_ONLY_FOR = ("billable", "product_code")


def rows(db: Session, sql: str, params: Optional[dict] = None) -> list[dict]:
    try:
        return [dict(r) for r in db.execute(text(sql), params or {}).mappings().all()]
    except Exception as exc:
        logger.warning("admin read failed: %s", exc)
        _safe_rollback(db)
        return []


def one(db: Session, sql: str, params: Optional[dict] = None) -> dict:
    result = rows(db, sql, params)
    return result[0] if result else {}


def scalar(db: Session, sql: str, params: Optional[dict] = None, default: Any = 0) -> Any:
    try:
        value = db.execute(text(sql), params or {}).scalar()
        return default if value is None else value
    except Exception as exc:
        logger.warning("admin scalar read failed: %s", exc)
        _safe_rollback(db)
        return default


def scalar_or_none(db: Session, sql: str, params: Optional[dict] = None) -> Any:
    """scalar() whose failure and whose NULL both stay None.

    For anything where 0 would be a claim rather than a count.
    """
    return scalar(db, sql, params, default=None)


def _safe_rollback(db: Session) -> None:
    """A failed statement poisons the session until it is rolled back.

    Without this the FIRST broken panel makes every LATER query on the same
    request fail too, turning one bad card into an empty dashboard — the exact
    failure fail-soft exists to prevent.
    """
    try:
        db.rollback()
    except Exception:
        pass


def table_exists(db: Session, *names: str) -> bool:
    """Do ALL of these tables exist?

    A presence probe rather than try/except around the real query, because the
    two answers differ: a swallowed 1146 and a genuine empty result look
    identical downstream, and a panel that cannot tell them apart renders
    confident zeros for a table that is not there.
    """
    if not names:
        return True
    found = scalar(
        db,
        """
        SELECT COUNT(*) FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN :names
        """.replace(":names", "(" + ",".join(f"'{n}'" for n in names) + ")"),
        {},
        default=0,
    )
    return int(found or 0) == len(names)


# ── Coercion ─────────────────────────────────────────────────────────────────

def iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def f(value: Any) -> float:
    try:
        return round(float(value or 0), 8)
    except (TypeError, ValueError):
        return 0.0


def i(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def i_or_none(value: Any) -> Optional[int]:
    """i(), except "nothing was found" stays None instead of becoming 0.

    i(None) -> 0 is right for a COUNT over a table that matched nothing. It is a
    lie for a lookup that could not run, or a period with no counter row: both
    mean NOT MEASURED, and rendering them as 0 is how a console spends a
    migration window reporting that the platform costs nothing.
    """
    return None if value is None else i(value)


def f_or_none(value: Any) -> Optional[float]:
    return None if value is None else f(value)


def clamp_days(days: Any, default: int = 30) -> int:
    try:
        days = int(days)
    except (TypeError, ValueError):
        return default
    return max(1, min(days, 365))


def clamp_limit(limit: Any, default: int = 50, ceiling: int = 500) -> int:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, ceiling))


def clamp_offset(offset: Any) -> int:
    try:
        return max(0, int(offset))
    except (TypeError, ValueError):
        return 0


# ── Sorting ──────────────────────────────────────────────────────────────────

def safe_sort(requested: Optional[str], allowed: dict[str, str], fallback: str) -> str:
    """Map a caller's sort key onto a SQL fragment from a fixed allowlist.

    An allowlist and not quoting, because an ORDER BY cannot be parameterised —
    the value is SQL, not data, so the only safe form is one this code wrote.
    An unknown key falls back silently rather than 400-ing: a stale bookmark
    with a renamed sort should show the list, not an error.
    """
    if requested and requested in allowed:
        return allowed[requested]
    return allowed[fallback]
