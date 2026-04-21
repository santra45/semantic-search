"""
Admin dashboard endpoints — consumed by the Magento module's backend pages.

  GET  /api/magento/chatbot/agent/admin/chats
  GET  /api/magento/chatbot/agent/admin/chats/{session_id}
  GET  /api/magento/chatbot/agent/admin/analytics
  POST /api/magento/chatbot/agent/admin/cleanup        (manual retention purge)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.database import get_db

from backend.app.magento.chatbot.db.schema import ensure_agent_schema
from backend.app.magento.chatbot.routers.common import authorize_request
from backend.app.magento.chatbot.services import chat_history_service

router = APIRouter()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


@router.get("/magento/chatbot/agent/admin/chats")
def list_chats(
    request: Request,
    store_code: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None),
    guest_session_id: Optional[str] = Query(None),
    kind: Optional[str] = Query(None, regex=r"^(customer|guest|all)?$"),
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    ensure_agent_schema(db)

    only_logged_in: Optional[bool] = None
    if kind == "customer":
        only_logged_in = True
    elif kind == "guest":
        only_logged_in = False

    sessions = chat_history_service.list_sessions(
        db,
        client_id=license_data["client_id"],
        store_code=store_code,
        customer_id=customer_id,
        guest_session_id=guest_session_id,
        only_logged_in=only_logged_in,
        since=_parse_iso(since),
        until=_parse_iso(until),
        limit=limit,
        offset=offset,
    )

    # Keyword search filter (simple LIKE over the last assistant / user content)
    if search:
        like = f"%{search.strip()}%"
        keep_ids = db.execute(
            text(
                """
                SELECT DISTINCT session_id FROM agent_chat_messages
                WHERE client_id = :cid AND content LIKE :like
                """
            ),
            {"cid": license_data["client_id"], "like": like},
        ).fetchall()
        allowed = {row.session_id for row in keep_ids}
        sessions = [s for s in sessions if s["session_id"] in allowed]

    total_row = db.execute(
        text("SELECT COUNT(*) AS c FROM agent_chat_sessions WHERE client_id = :cid"),
        {"cid": license_data["client_id"]},
    ).fetchone()

    return {
        "sessions": sessions,
        "total": int(total_row.c or 0) if total_row else 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/magento/chatbot/agent/admin/chats/{session_id}")
def chat_transcript(
    session_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    session = chat_history_service.get_session(
        db, client_id=license_data["client_id"], session_id=session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = chat_history_service.get_session_messages(
        db, client_id=license_data["client_id"], session_id=session_id
    )
    return {"session": session, "messages": messages}


@router.get("/magento/chatbot/agent/admin/analytics")
def analytics(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    store_code: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Aggregate stats for Chart.js rendering in the Magento admin dashboard."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    ensure_agent_schema(db)

    since = datetime.utcnow() - timedelta(days=days)
    client_id = license_data["client_id"]

    params: dict = {"client_id": client_id, "since": since}
    store_clause = ""
    if store_code:
        store_clause = "AND s.store_code = :store_code"
        params["store_code"] = store_code

    # Totals
    totals_row = db.execute(
        text(
            f"""
            SELECT
                COUNT(DISTINCT s.id) AS session_count,
                COUNT(DISTINCT CASE WHEN s.is_customer_login = 1 THEN s.id END) AS customer_sessions,
                COUNT(DISTINCT CASE WHEN s.is_customer_login = 0 THEN s.id END) AS guest_sessions,
                COUNT(m.id) AS message_count,
                COALESCE(SUM(m.total_tokens), 0) AS total_tokens,
                COALESCE(SUM(m.total_cost), 0) AS total_cost,
                COALESCE(AVG(m.response_time_ms), 0) AS avg_response_ms
            FROM agent_chat_sessions s
            LEFT JOIN agent_chat_messages m ON m.session_id = s.id
            WHERE s.client_id = :client_id AND s.last_activity_at >= :since {store_clause}
            """
        ),
        params,
    ).fetchone()

    # Messages per day (time series)
    series_rows = db.execute(
        text(
            f"""
            SELECT DATE(m.created_at) AS day,
                   COUNT(*) AS messages,
                   COALESCE(SUM(m.total_tokens), 0) AS tokens,
                   COALESCE(SUM(m.total_cost), 0) AS cost
            FROM agent_chat_messages m
            JOIN agent_chat_sessions s ON s.id = m.session_id
            WHERE s.client_id = :client_id AND m.created_at >= :since {store_clause}
            GROUP BY DATE(m.created_at)
            ORDER BY day ASC
            """
        ),
        params,
    ).fetchall()

    # Agent usage breakdown
    agent_rows = db.execute(
        text(
            f"""
            SELECT m.agent_name AS agent, COUNT(*) AS c
            FROM agent_chat_messages m
            JOIN agent_chat_sessions s ON s.id = m.session_id
            WHERE s.client_id = :client_id AND m.created_at >= :since
              AND m.role = 'assistant' AND m.agent_name IS NOT NULL {store_clause}
            GROUP BY m.agent_name
            """
        ),
        params,
    ).fetchall()

    # Top user queries
    top_queries = db.execute(
        text(
            f"""
            SELECT SUBSTRING(m.content, 1, 120) AS q, COUNT(*) AS c
            FROM agent_chat_messages m
            JOIN agent_chat_sessions s ON s.id = m.session_id
            WHERE s.client_id = :client_id AND m.role = 'user'
              AND m.created_at >= :since {store_clause}
            GROUP BY q
            ORDER BY c DESC
            LIMIT 15
            """
        ),
        params,
    ).fetchall()

    # Per-store breakdown (independent of `store_code` filter above)
    per_store = db.execute(
        text(
            """
            SELECT store_code, COUNT(*) AS sessions, COALESCE(SUM(message_count), 0) AS messages
            FROM agent_chat_sessions
            WHERE client_id = :client_id AND last_activity_at >= :since
            GROUP BY store_code
            ORDER BY sessions DESC
            """
        ),
        {"client_id": client_id, "since": since},
    ).fetchall()

    return {
        "range_days": days,
        "store_code": store_code,
        "totals": {
            "sessions": int(totals_row.session_count or 0) if totals_row else 0,
            "customer_sessions": int(totals_row.customer_sessions or 0) if totals_row else 0,
            "guest_sessions": int(totals_row.guest_sessions or 0) if totals_row else 0,
            "messages": int(totals_row.message_count or 0) if totals_row else 0,
            "total_tokens": int(totals_row.total_tokens or 0) if totals_row else 0,
            "total_cost": float(totals_row.total_cost or 0) if totals_row else 0.0,
            "avg_response_ms": float(totals_row.avg_response_ms or 0) if totals_row else 0.0,
        },
        "series": [
            {
                "day": row.day.isoformat() if hasattr(row.day, "isoformat") else str(row.day),
                "messages": int(row.messages or 0),
                "tokens": int(row.tokens or 0),
                "cost": float(row.cost or 0),
            }
            for row in series_rows
        ],
        "agent_breakdown": [
            {"agent": row.agent or "unknown", "count": int(row.c or 0)} for row in agent_rows
        ],
        "top_queries": [
            {"query": row.q, "count": int(row.c or 0)} for row in top_queries
        ],
        "per_store": [
            {
                "store_code": row.store_code or "default",
                "sessions": int(row.sessions or 0),
                "messages": int(row.messages or 0),
            }
            for row in per_store
        ],
    }


class CleanupRequest(BaseModel):
    license_key: Optional[str] = None
    older_than_days: int = 90


@router.post("/magento/chatbot/agent/admin/cleanup")
def admin_cleanup(
    req: CleanupRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Manually delete chat history older than `older_than_days`. Called by the Magento cleanup cron
    when the admin enables scheduled purge."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )
    deleted = chat_history_service.purge_old_chats(
        db,
        client_id=license_data["client_id"],
        older_than_days=int(req.older_than_days),
    )
    return {"deleted_rows": deleted, "older_than_days": req.older_than_days}
