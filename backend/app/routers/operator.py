"""Operator analytics + cost console (Phase 4.4 + 4.5).

An ALL-TENANT ops dashboard for the SaaS provider — not a merchant-facing view.
It surfaces per-tenant usage, cost, quota, and chat-quality analytics pulled from
`token_usage_tracking` (the per-request cost log), `clients` / `license_keys`
(the tenant registry), `usage_logs` (monthly quota), and the `chat_*` analytics
tables.

Serving model:
  * GET /operator                — the HTML shell (no tenant data in it, so it's
                                   safe to serve openly; the JS logs in with the
                                   operator key and loads everything via fetch).
  * GET /api/operator/*          — the JSON data, GATED by the X-Operator-Key
                                   header (checked against AICHATBOT_OPERATOR_KEY).

Every data query is fail-soft: a missing/renamed table degrades that one section
to empty instead of 500-ing the whole console (some analytics tables are optional
per deployment).
"""

from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.config import OPERATOR_KEY
from backend.app.services.database import get_db

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

    totals = _one(db, """
        SELECT COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests,
               COUNT(DISTINCT client_id) AS active_tenants
        FROM token_usage_tracking
        WHERE created_at >= :since
    """, p)

    total_tenants = _scalar(db, "SELECT COUNT(*) FROM clients", {}, 0)

    by_type = _rows(db, """
        SELECT query_type,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE created_at >= :since
        GROUP BY query_type
        ORDER BY cost DESC
    """, p)

    series = _rows(db, """
        SELECT DATE(created_at) AS d,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE created_at >= :since
        GROUP BY DATE(created_at)
        ORDER BY d
    """, p)

    top = _rows(db, """
        SELECT t.client_id,
               COALESCE(c.name, t.client_id) AS name,
               COALESCE(SUM(t.total_cost), 0) AS cost,
               COALESCE(SUM(t.total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking t
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

    rows = _rows(db, """
        SELECT c.id, c.name, c.email, c.plan, c.is_active, c.created_at,
               lk.allowed_domain, lk.expires_at, lk.search_limit_per_month,
               COALESCE(t.cost, 0) AS cost,
               COALESCE(t.tokens, 0) AS tokens,
               COALESCE(t.requests, 0) AS requests,
               t.last_request,
               COALESCE(u.search_count, 0) AS search_count
        FROM clients c
        LEFT JOIN (
            SELECT client_id,
                   SUM(total_cost) AS cost,
                   SUM(total_tokens) AS tokens,
                   COUNT(*) AS requests,
                   MAX(created_at) AS last_request
            FROM token_usage_tracking
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
        LEFT JOIN (
            SELECT client_id, search_count
            FROM usage_logs
            WHERE month = :month
        ) u ON u.client_id = c.id
        ORDER BY cost DESC, c.created_at DESC
    """, {"since": since, "month": month})

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
            "search_count": _i(r.get("search_count")),
            "search_limit": _i(r.get("search_limit_per_month")),
        }
        for r in rows
    ]
    return {"days": days, "tenants": tenants}


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

    kpis = _one(db, """
        SELECT COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE client_id = :cid AND created_at >= :since
    """, p)

    by_type = _rows(db, """
        SELECT query_type,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY query_type ORDER BY cost DESC
    """, p)

    by_model = _rows(db, """
        SELECT llm_provider, llm_model,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY llm_provider, llm_model ORDER BY cost DESC
    """, p)

    series = _rows(db, """
        SELECT DATE(created_at) AS d,
               COALESCE(SUM(total_cost), 0) AS cost,
               COALESCE(SUM(total_tokens), 0) AS tokens,
               COUNT(*) AS requests
        FROM token_usage_tracking
        WHERE client_id = :cid AND created_at >= :since
        GROUP BY DATE(created_at) ORDER BY d
    """, p)

    usage = _one(db, """
        SELECT COALESCE(search_count, 0) AS search_count,
               COALESCE(ingest_count, 0) AS ingest_count
        FROM usage_logs WHERE client_id = :cid AND month = :month
    """, {"cid": client_id, "month": month})

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
            "search_count": _i(usage.get("search_count")),
            "ingest_count": _i(usage.get("ingest_count")),
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
    }


def _chat_quality(db: Session, client_id: str, since: datetime) -> dict:
    """Chat-quality analytics for a tenant — grounding rate, response time,
    feedback tallies, and the most-recent thumbs-down answers. Fully fail-soft:
    if the chat_* tables aren't present in this deployment, returns zeros/empties
    rather than breaking the tenant view."""
    out = {
        "conversations": 0,
        "messages": 0,
        "avg_response_ms": 0,
        "grounded_rate": None,
        "chat_cost": 0.0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "top_disliked": [],
    }
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
