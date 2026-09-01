"""Operator analytics + cost console (Phase 4.4 + 4.5).

An ALL-TENANT ops dashboard for the SaaS provider — not a merchant-facing view.
It surfaces per-tenant usage, cost, quota, and chat-quality analytics pulled from
the usage ledger (per-request cost), `clients` / `license_keys` (the tenant
registry), the monthly quota counters, and the `chat_*` analytics tables.

Serving model:
  * GET /operator                — the HTML shell (no tenant data in it, so it's
                                   safe to serve openly; the JS logs in with the
                                   operator key and loads everything via fetch).
  * GET /api/operator/*          — the JSON data, GATED by the X-Operator-Key
                                   header (checked against AICHATBOT_OPERATOR_KEY).

Every data query is fail-soft: a missing/renamed table degrades that one section
to empty instead of 500-ing the whole console (some analytics tables are optional
per deployment).

FAIL-SOFT IS WHY THIS FILE WAS THE WORST OF THE FOUR
----------------------------------------------------
Eleven of the queries below named tables the v2 billing migration renamed —
nine `token_usage_tracking` and two `usage_logs`. Because _rows/_one/_scalar
swallow pymysql 1146 and return {} or [], none of them 500'd. They rendered
ZEROS instead: every cost, token count and request count on the all-tenant
overview, on every row of the tenant list, and on every tenant deep-dive read
0.00 with no visible error and eleven identical "operator query failed" WARNINGs
buried in the log. A console whose entire purpose is showing the operator what
the platform costs was confidently reporting that it costs nothing.

That is the failure this whole change exists to remove, so the fixes here are of
two kinds and it matters which is which:

  * The nine ledger reads now use usage_ledger_read.LEDGER — the frozen v1
    archive plus the live v2 `usage_events` as one table. Real numbers again.
  * The two quota reads and the chat-quality block CANNOT produce a real number
    today, so they stop producing a fake one. `search_count` is None rather than
    0 when neither ledger has a row for the month, and the chat block reports
    `available: false` when its tables do not exist on this deployment.

WHAT THE CONSOLE ACTUALLY SHOWS DURING THE DUAL-READ WINDOW
-----------------------------------------------------------
Pre-migration history, stopping at the archive's last row, and nothing since.
The v1 JWT keys carrying 100% of current traffic resolve no v2 context, so
usage_service.record() refuses every row and today's spend is in NEITHER ledger.
Each of the three endpoints therefore returns a `usage_source` block from
usage_ledger_read.provenance(); `usage_source.current` is false until the first
v2 licence is issued.

backend/app/templates/operator.html DOES NOT RENDER ANY OF THAT YET, and its
fmtNum() coerces null to "0" — so a None search_count still displays as a zero
in the browser. The JSON is now honest; the page is not. Updating the template
to branch on `usage_source.current` and to show "—" for a null quota is the
remaining half of this fix and it is in a file this change does not own.
"""

from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from backend.app.config import OPERATOR_KEY
from backend.app.services.database import get_db
from backend.app.services.usage_ledger_read import LEDGER, provenance

logger = logging.getLogger("operator_dashboard")

router = APIRouter()
_HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "operator.html")


# ── Auth ─────────────────────────────────────────────────────────────────────

def require_operator(
    x_operator_key: Optional[str] = Header(None, alias="X-Operator-Key")
) -> bool:
    """Gate every data endpoint. Locked by default: if no operator key is
    configured, the console is unreachable (403) rather than wide open."""
    if not OPERATOR_KEY:
        raise HTTPException(
            status_code=403,
            detail="Operator dashboard is not configured (set AICHATBOT_OPERATOR_KEY).",
        )
    if not x_operator_key or not hmac.compare_digest(str(x_operator_key), str(OPERATOR_KEY)):
        raise HTTPException(status_code=403, detail="Invalid operator key.")
    return True


# ── Small fail-soft query helpers ────────────────────────────────────────────

def _rows(db: Session, sql: str, params: dict) -> list[dict]:
    try:
        return [dict(r) for r in db.execute(text(sql), params).mappings().all()]
    except Exception as exc:
        logger.warning("operator query failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return []


def _one(db: Session, sql: str, params: dict) -> dict:
    rows = _rows(db, sql, params)
    return rows[0] if rows else {}


def _scalar(db: Session, sql: str, params: dict, default: Any = 0) -> Any:
    try:
        val = db.execute(text(sql), params).scalar()
        return default if val is None else val
    except Exception as exc:
        logger.warning("operator scalar query failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return default


def _clamp_days(days: int) -> int:
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 30
    return max(1, min(days, 365))


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _f(value: Any) -> float:
    try:
        return round(float(value or 0), 8)
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _i_or_none(value: Any) -> Optional[int]:
    """_i(), except that "nothing was found" stays None instead of becoming 0.

    _i(None) -> 0 is right for a COUNT over a table that exists and matched
    nothing. It is a lie for a lookup that could not run, or for a month with no
    counter row at all: both mean "not measured", and rendering them as 0 is how
    this console spent the whole migration reporting that the platform costs
    nothing. Anywhere the difference is knowable, the None survives to the JSON.
    """
    if value is None:
        return None
    return _i(value)


def _tables_exist(db: Session, *names: str) -> bool:
    """Do ALL of these tables exist in the current schema?

    A presence probe, not a try//except around the real query, because the two
    answers are different: a swallowed 1146 and a genuine empty result look
    identical downstream, and this console has already shipped one dashboard
    that could not tell them apart. One information_schema lookup per call is
    cheap and it is NOT cached — chat_conversations et al. are created lazily by
    conversation_service.ensure_chat_tables() the first time a shopper sends a
    message, so "absent" is a fact with a short shelf life.

    Returns False on a lookup failure, which folds "I could not find out" into
    "not available" — the caller's contract is only that True means it is safe to
    query, and a False sends it down the honest not-measured path either way.
    """
    if not names:
        return True
    try:
        found = db.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name IN :names"
            ).bindparams(bindparam("names", expanding=True)),
            {"names": list(names)},
        ).scalar()
        return int(found or 0) == len(set(names))
    except Exception as exc:
        logger.warning("operator table probe failed for %s: %s", names, exc)
        try:
            db.rollback()
        except Exception:
            pass
        return False


def _monthly_quota(db: Session, month: str, client_id: Optional[str] = None) -> dict[str, dict]:
    """This month's request quota per client, from whichever ledger has it.

    THE QUOTA MOVED LEVELS AND THE TWO NUMBERS ARE NOT THE SAME NUMBER. v1
    metered searches per CLIENT in usage_logs (now usage_logs_archive_v1,
    frozen). v2 meters billable requests per SUBSCRIPTION in usage_counters, so
    a client with two sites has two counter rows and this sums them to keep the
    per-client shape the console and its HTML already speak. That sum is the
    right figure for "what has this customer used this month" and the WRONG one
    for a quota decision, which is per-subscription — do not reuse it for one;
    usage_service.within_request_quota() is the entry point for that.

    Returns {client_id: {"count": int | None, "source": str}}. `count` is None
    when neither ledger holds a row for that client and month, and that None is
    load-bearing: it is the difference between "used nothing yet this month" and
    "this month is not being recorded anywhere", and during the dual-read window
    the second is the true one for every tenant on a v1 key.
    """
    out: dict[str, dict] = {}

    # v2 first: it is the live ledger, so where it has a row that row wins.
    v2_where = "uc.period = :month"
    params: dict[str, Any] = {"month": month}
    if client_id:
        v2_where += " AND s.client_id = :cid"
        params["cid"] = client_id
    for r in _rows(db, f"""
        SELECT s.client_id                     AS client_id,
               SUM(uc.billable_requests)       AS n
        FROM usage_counters uc
        JOIN subscriptions sub ON sub.id = uc.subscription_id
        JOIN sites s           ON s.id  = sub.site_id
        WHERE {v2_where}
        GROUP BY s.client_id
    """, params):
        out[r["client_id"]] = {"count": _i(r.get("n")), "source": "usage_counters"}

    # v1 archive fills only the gaps. It stops at the migration, so it can
    # describe a month that ran before the cutover and never a current one.
    v1_where = "month = :month"
    if client_id:
        v1_where += " AND client_id = :cid"
    for r in _rows(db, f"""
        SELECT client_id, search_count, ingest_count
        FROM usage_logs_archive_v1
        WHERE {v1_where}
    """, params):
        out.setdefault(r["client_id"], {
            "count": _i(r.get("search_count")),
            "ingest_count": _i(r.get("ingest_count")),
            "source": "usage_logs_archive_v1",
        })

    return out


# The answer for a client that neither ledger has a row for. Shared and READ
# ONLY - it is handed out as a .get() default at three sites, so mutating it
# would silently rewrite what "not measured" means for every tenant at once.
# Note the absent `ingest_count` key: .get("ingest_count") on this returns None,
# which is the honest answer, where a 0 here would claim the tenant ingested
# nothing this month.
_QUOTA_UNMEASURED = {"count": None, "source": None}


# ── HTML shell ───────────────────────────────────────────────────────────────

@router.get("/operator", response_class=HTMLResponse)
def operator_page():
    """The dashboard shell. Contains no tenant data — the JS authenticates with
    the operator key and loads everything from the gated /api/operator/* JSON.
    Served as a raw static file (not via Jinja) so the page's JS/CSS braces can
    never collide with template delimiters."""
    try:
        with open(_HTML_PATH, "r", encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="operator.html not found")


# ── All-tenant overview ──────────────────────────────────────────────────────

@router.get("/api/operator/overview")
def operator_overview(
    days: int = 30,
    _: bool = Depends(require_operator),
    db: Session = Depends(get_db),
):
    days = _clamp_days(days)
    since = datetime.utcnow() - timedelta(days=days)
    p = {"since": since}

    totals = _one(db, f"""
        SELECT COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests,
               COUNT(DISTINCT client_id) AS active_tenants
        FROM {LEDGER} u
        WHERE created_at >= :since
    """, p)

    total_tenants = _scalar(db, "SELECT COUNT(*) FROM clients", {}, 0)

    by_type = _rows(db, f"""
        SELECT query_type,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE created_at >= :since
        GROUP BY query_type
        ORDER BY cost DESC
    """, p)

    series = _rows(db, f"""
        SELECT DATE(created_at) AS d,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE created_at >= :since
        GROUP BY DATE(created_at)
        ORDER BY d
    """, p)

    # The join is safe across the union: every client_id column involved -
    # clients.id, the archive's and usage_events' - is utf8mb4_general_ci, and
    # the ledger forces that collation explicitly so a table recreated under the
    # server default cannot turn this into ERROR 1271 at runtime.
    top = _rows(db, f"""
        SELECT t.client_id,
               COALESCE(c.name, t.client_id) AS name,
               COALESCE(SUM(t.total_cost), 0) AS cost,
               COALESCE(SUM(t.total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} t
        LEFT JOIN clients c ON c.id = t.client_id
        WHERE t.created_at >= :since
        GROUP BY t.client_id, c.name
        ORDER BY cost DESC
        LIMIT 10
    """, p)

    return {
        "days": days,
        "totals": {
            "cost": _f(totals.get("cost")),
            "tokens": _i(totals.get("tokens")),
            "requests": _i(totals.get("requests")),
            "active_tenants": _i(totals.get("active_tenants")),
            "total_tenants": _i(total_tenants),
        },
        "by_type": [
            {"query_type": r.get("query_type") or "unknown",
             "cost": _f(r.get("cost")), "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in by_type
        ],
        "series": [
            {"date": str(r.get("d")), "cost": _f(r.get("cost")),
             "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in series
        ],
        "top_tenants": [
            {"client_id": r.get("client_id"), "name": r.get("name"),
             "cost": _f(r.get("cost")), "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in top
        ],
        "usage_source": provenance(db),
    }


# ── Per-tenant rollup list ───────────────────────────────────────────────────

@router.get("/api/operator/tenants")
def operator_tenants(
    days: int = 30,
    _: bool = Depends(require_operator),
    db: Session = Depends(get_db),
):
    days = _clamp_days(days)
    since = datetime.utcnow() - timedelta(days=days)
    month = datetime.utcnow().strftime("%Y-%m")

    # The quota join came out of the SQL. It used to be a fourth LEFT JOIN on
    # `usage_logs`, which no longer exists; its v2 replacement lives two joins
    # away (usage_counters -> subscriptions -> sites -> client), and bolting
    # that onto a query that already carries three derived tables would make one
    # statement nobody can read for a column the console renders as a bar. It is
    # one extra round trip, resolved per client below.
    quota = _monthly_quota(db, month)

    rows = _rows(db, f"""
        SELECT c.id, c.name, c.email, c.plan, c.is_active, c.created_at,
               lk.allowed_domain, lk.expires_at, lk.search_limit_per_month,
               COALESCE(t.cost, 0) AS cost,
               COALESCE(t.tokens, 0) AS tokens,
               COALESCE(t.requests, 0) AS requests,
               t.last_request
        FROM clients c
        LEFT JOIN (
            SELECT client_id,
                   SUM(total_cost) AS cost,
                   SUM(total_tokens) AS tokens,
                   COUNT(*) AS requests,
                   MAX(created_at) AS last_request
            FROM {LEDGER} l
            WHERE created_at >= :since
            GROUP BY client_id
        ) t ON t.client_id = c.id
        LEFT JOIN (
            SELECT client_id,
                   MAX(allowed_domain) AS allowed_domain,
                   MAX(expires_at) AS expires_at,
                   MAX(search_limit_per_month) AS search_limit_per_month
            FROM license_keys
            WHERE is_active = 1
            GROUP BY client_id
        ) lk ON lk.client_id = c.id
        ORDER BY cost DESC, c.created_at DESC
    """, {"since": since})

    tenants = [
        {
            "client_id": r.get("id"),
            "name": r.get("name"),
            "email": r.get("email"),
            "plan": r.get("plan"),
            "is_active": bool(r.get("is_active")),
            "created_at": _iso(r.get("created_at")),
            "domain": r.get("allowed_domain"),
            "expires_at": _iso(r.get("expires_at")),
            "cost": _f(r.get("cost")),
            "tokens": _i(r.get("tokens")),
            "requests": _i(r.get("requests")),
            "last_request": _iso(r.get("last_request")),
            # None, not 0, when no ledger holds this client's month. The old
            # COALESCE(u.search_count, 0) turned a renamed table into a tenant
            # list where every quota bar read 0 / 250 and looked healthy.
            "search_count": _i_or_none(
                quota.get(r.get("id"), _QUOTA_UNMEASURED)["count"]
            ),
            "search_count_source": quota.get(r.get("id"), _QUOTA_UNMEASURED)["source"],
            "search_limit": _i(r.get("search_limit_per_month")),
        }
        for r in rows
    ]
    return {"days": days, "month": month, "tenants": tenants, "usage_source": provenance(db)}


# ── Per-tenant deep dive ─────────────────────────────────────────────────────

@router.get("/api/operator/tenant/{client_id}")
def operator_tenant_detail(
    client_id: str,
    days: int = 30,
    _: bool = Depends(require_operator),
    db: Session = Depends(get_db),
):
    days = _clamp_days(days)
    since = datetime.utcnow() - timedelta(days=days)
    month = datetime.utcnow().strftime("%Y-%m")
    p = {"cid": client_id, "since": since}

    info = _one(db, """
        SELECT c.id, c.name, c.email, c.plan, c.is_active, c.created_at,
               lk.allowed_domain, lk.expires_at, lk.search_limit_per_month, lk.product_limit
        FROM clients c
        LEFT JOIN (
            SELECT client_id,
                   MAX(allowed_domain) AS allowed_domain,
                   MAX(expires_at) AS expires_at,
                   MAX(search_limit_per_month) AS search_limit_per_month,
                   MAX(product_limit) AS product_limit
            FROM license_keys WHERE is_active = 1 GROUP BY client_id
        ) lk ON lk.client_id = c.id
        WHERE c.id = :cid
    """, {"cid": client_id})
    if not info:
        raise HTTPException(status_code=404, detail="Tenant not found")

    kpis = _one(db, f"""
        SELECT COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE client_id = :cid AND created_at >= :since
    """, p)

    by_type = _rows(db, f"""
        SELECT query_type,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY query_type ORDER BY cost DESC
    """, p)

    by_model = _rows(db, f"""
        SELECT llm_provider, llm_model,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY llm_provider, llm_model ORDER BY cost DESC
    """, p)

    series = _rows(db, f"""
        SELECT DATE(created_at) AS d,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM {LEDGER} u
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY DATE(created_at) ORDER BY d
    """, p)

    # ingest_count has NO v2 equivalent in a counter. usage_logs.ingest_count
    # was per-client; v2 keeps indexed items on sites.indexed_items, which is a
    # CURRENT stock and not a monthly flow, so it cannot answer "how much did
    # this tenant ingest in September" and is deliberately not substituted for
    # it here. The archive answers it for months that ran before the cutover and
    # nothing answers it for months since - hence None rather than 0.
    usage = _monthly_quota(db, month, client_id=client_id).get(client_id, _QUOTA_UNMEASURED)

    return {
        "days": days,
        "tenant": {
            "client_id": info.get("id"),
            "name": info.get("name"),
            "email": info.get("email"),
            "plan": info.get("plan"),
            "is_active": bool(info.get("is_active")),
            "created_at": _iso(info.get("created_at")),
            "domain": info.get("allowed_domain"),
            "expires_at": _iso(info.get("expires_at")),
            "search_limit": _i(info.get("search_limit_per_month")),
            "product_limit": _i(info.get("product_limit")),
        },
        "kpis": {
            "cost": _f(kpis.get("cost")),
            "tokens": _i(kpis.get("tokens")),
            "requests": _i(kpis.get("requests")),
            "search_count": _i_or_none(usage.get("count")),
            "search_count_source": usage.get("source"),
            "ingest_count": _i_or_none(usage.get("ingest_count")),
        },
        "by_type": [
            {"query_type": r.get("query_type") or "unknown",
             "cost": _f(r.get("cost")), "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in by_type
        ],
        "by_model": [
            {"provider": r.get("llm_provider"), "model": r.get("llm_model"),
             "cost": _f(r.get("cost")), "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in by_model
        ],
        "series": [
            {"date": str(r.get("d")), "cost": _f(r.get("cost")),
             "tokens": _i(r.get("tokens")), "requests": _i(r.get("requests"))}
            for r in series
        ],
        "quality": _chat_quality(db, client_id, since),
        "usage_source": provenance(db),
    }


# The three tables the chat-quality block reads. They are created lazily by
# conversation_service.ensure_chat_tables() on the first /magento/chatbot/message
# call, so on a deployment that has served chat they hold the entire production
# conversation history, and on one that has not they are simply absent. Both are
# ordinary states; neither is an error, and the point of naming them here is that
# the block can tell them apart.
_CHAT_TABLES = ("chat_conversations", "chat_messages", "chat_feedback")


def _chat_quality(db: Session, client_id: str, since: datetime) -> dict:
    """Chat-quality analytics for a tenant — grounding rate, response time,
    feedback tallies, and the most-recent thumbs-down answers.

    Still fail-soft, but no longer fail-QUIET. The previous version leaned on
    _rows/_one swallowing the 1146 from three absent tables and returned its
    initialiser unchanged: 0 conversations, 0 messages, 0 thumbs-up, 0
    thumbs-down, 0.00 chat cost — a full set of numbers, none of them measured,
    rendered by operator.html as though a tenant had served chat traffic and got
    no feedback on any of it. On this database all three tables are absent, so
    that is what the console has been showing.

    It now probes for the tables first and, when they are not there, returns
    `available: false` with NULLS in place of the counts. A null is not much of a
    signal, but it is the truthful one, and it is the shape the template needs
    before it can render "—". The zeros stay only where they are earned: tables
    present, query ran, nothing matched.

    NOT a reason to delete these queries. See the comment above
    schema_v2.LEGACY_TABLES_TO_DROP — an earlier draft of it claimed the chat_*
    tables were empty husks, that claim was wrong, and it was the entire
    justification for a DROP that MySQL cannot roll back.
    """
    out: dict[str, Any] = {
        "available": True,
        "conversations": 0,
        "messages": 0,
        "avg_response_ms": 0,
        "grounded_rate": None,
        "chat_cost": 0.0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "top_disliked": [],
    }

    if not _tables_exist(db, *_CHAT_TABLES):
        logger.info(
            "operator: chat analytics unavailable for %s - one or more of %s "
            "does not exist on this deployment. Reporting nulls rather than "
            "zeros; the tables appear the first time a shopper sends a message.",
            client_id, ", ".join(_CHAT_TABLES),
        )
        out.update({
            "available": False,
            "conversations": None,
            "messages": None,
            "avg_response_ms": None,
            "chat_cost": None,
            "thumbs_up": None,
            "thumbs_down": None,
            "note": (
                "Chat analytics tables are not present on this deployment. "
                "These fields are NOT MEASURED, not zero."
            ),
        })
        return out

    p = {"cid": client_id, "since": since}

    conv = _one(db, """
        SELECT COUNT(*) AS c
        FROM chat_conversations
        WHERE client_id = :cid AND started_at >= :since
    """, p)
    out["conversations"] = _i(conv.get("c"))

    m = _one(db, """
        SELECT COUNT(*) AS msgs,
               AVG(NULLIF(m.response_time_ms, 0)) AS avg_ms,
               AVG(m.grounded) AS grounded_rate,
               COALESCE(SUM(m.cost), 0) AS chat_cost
        FROM chat_messages m
        JOIN chat_conversations c ON c.id = m.conversation_id
        WHERE c.client_id = :cid AND m.created_at >= :since AND m.role = 'assistant'
    """, p)
    out["messages"] = _i(m.get("msgs"))
    out["avg_response_ms"] = _i(m.get("avg_ms"))
    out["grounded_rate"] = None if m.get("grounded_rate") is None else round(float(m["grounded_rate"]) * 100, 1)
    out["chat_cost"] = _f(m.get("chat_cost"))

    for r in _rows(db, """
        SELECT f.feedback_type, COUNT(*) AS n
        FROM chat_feedback f
        JOIN chat_conversations c ON c.id = f.conversation_id
        WHERE c.client_id = :cid AND f.created_at >= :since
        GROUP BY f.feedback_type
    """, p):
        ft = (r.get("feedback_type") or "").lower()
        if "up" in ft:
            out["thumbs_up"] += _i(r.get("n"))
        elif "down" in ft:
            out["thumbs_down"] += _i(r.get("n"))

    out["top_disliked"] = [
        {
            "question": (r.get("message_text") or "")[:200],
            "answer": (r.get("response_text") or "")[:300],
            "at": _iso(r.get("created_at")),
        }
        for r in _rows(db, """
            SELECT m.message_text, m.response_text, f.created_at
            FROM chat_feedback f
            JOIN chat_messages m ON m.id = f.message_id
            JOIN chat_conversations c ON c.id = f.conversation_id
            WHERE c.client_id = :cid AND LOWER(f.feedback_type) LIKE '%down%'
            ORDER BY f.created_at DESC
            LIMIT 10
        """, {"cid": client_id})
    ]
    return out
