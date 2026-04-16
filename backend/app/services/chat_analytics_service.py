from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.conversation_service import ensure_chat_tables
from backend.app.services.qdrant_service import get_client_content_counts

CHAT_QUERY_TYPES = ("chat_answer", "chat_context", "chat_rewrite")


def get_dashboard_data(
    db: Session,
    client_id: str,
    domain: str,
    store_id: str | None = None,
) -> dict[str, Any]:
    ensure_chat_tables(db)

    message_where = [
        "c.client_id = :client_id",
        "m.role = 'assistant'",
    ]
    message_params: dict[str, Any] = {"client_id": client_id}
    if store_id:
        message_where.append("c.store_id = :store_id")
        message_params["store_id"] = store_id

    summary_row = db.execute(
        text(
            f"""
            SELECT
                COUNT(DISTINCT CASE WHEN DATE(c.started_at) = CURDATE() THEN c.id END) AS chats_today,
                COUNT(DISTINCT CASE WHEN DATE_FORMAT(c.started_at, '%Y-%m') = DATE_FORMAT(NOW(), '%Y-%m') THEN c.id END) AS chats_month,
                AVG(NULLIF(m.response_time_ms, 0)) AS avg_response_time_ms,
                SUM(m.cost) AS total_cost,
                SUM(CASE WHEN m.grounded = 0 THEN 1 ELSE 0 END) AS unresolved_chats
            FROM chat_conversations c
            LEFT JOIN chat_messages m ON m.conversation_id = c.id
            WHERE {' AND '.join(message_where)}
            """
        ),
        message_params,
    ).fetchone()

    content_counts = get_client_content_counts(client_id, domain)

    return {
        "chats_today": int(summary_row.chats_today or 0),
        "chats_month": int(summary_row.chats_month or 0),
        "unresolved_chats": int(summary_row.unresolved_chats or 0),
        "avg_response_time_ms": round(float(summary_row.avg_response_time_ms or 0), 2),
        "total_cost": round(float(summary_row.total_cost or 0), 6),
        "content_counts": content_counts,
    }


def get_usage_data(
    db: Session,
    client_id: str,
    days: int = 30,
) -> dict[str, Any]:
    ensure_chat_tables(db)

    start_date = datetime.utcnow() - timedelta(days=days)
    params = {
        "client_id": client_id,
        "start_date": start_date,
    }

    summary_row = db.execute(
        text(
            """
            SELECT
                COUNT(*) AS total_requests,
                SUM(total_tokens) AS total_tokens,
                SUM(total_cost) AS total_cost
            FROM token_usage_tracking
            WHERE client_id = :client_id
              AND query_type IN ('chat_answer', 'chat_context', 'chat_rewrite')
              AND created_at >= :start_date
            """
        ),
        params,
    ).fetchone()

    models = db.execute(
        text(
            """
            SELECT
                llm_provider,
                llm_model,
                query_type,
                COUNT(*) AS request_count,
                SUM(total_tokens) AS total_tokens,
                SUM(total_cost) AS total_cost
            FROM token_usage_tracking
            WHERE client_id = :client_id
              AND query_type IN ('chat_answer', 'chat_context', 'chat_rewrite')
              AND created_at >= :start_date
            GROUP BY llm_provider, llm_model, query_type
            ORDER BY total_cost DESC
            """
        ),
        params,
    ).fetchall()

    hourly = db.execute(
        text(
            """
            SELECT
                DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00') AS hour,
                SUM(total_cost) AS total_cost,
                SUM(total_tokens) AS total_tokens,
                COUNT(*) AS request_count
            FROM token_usage_tracking
            WHERE client_id = :client_id
              AND query_type IN ('chat_answer', 'chat_context', 'chat_rewrite')
              AND created_at >= :start_date
            GROUP BY DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00')
            ORDER BY hour ASC
            """
        ),
        params,
    ).fetchall()

    return {
        "period_days": days,
        "summary": {
            "total_requests": int(summary_row.total_requests or 0),
            "total_tokens": int(summary_row.total_tokens or 0),
            "total_cost": round(float(summary_row.total_cost or 0), 6),
        },
        "models": [
            {
                "llm_provider": row.llm_provider,
                "llm_model": row.llm_model,
                "query_type": row.query_type,
                "request_count": int(row.request_count or 0),
                "total_tokens": int(row.total_tokens or 0),
                "total_cost": round(float(row.total_cost or 0), 6),
            }
            for row in models
        ],
        "hourly": [
            {
                "hour": row.hour,
                "request_count": int(row.request_count or 0),
                "total_tokens": int(row.total_tokens or 0),
                "total_cost": round(float(row.total_cost or 0), 6),
            }
            for row in hourly
        ],
    }
