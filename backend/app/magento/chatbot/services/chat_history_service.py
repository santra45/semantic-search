"""
Persist agent chat sessions and messages in MySQL.

Every turn (user message, each agent-LLM response, each tool invocation) gets its
own row so dashboards can slice by role, agent, tool, customer, guest, date, or
store. The row also carries token counts — populated from LangChain's
`usage_metadata` on each AIMessage.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.magento.chatbot.db.schema import ensure_agent_schema


# ── Thread id helpers ────────────────────────────────────────────────────────

def build_thread_id(
    *,
    client_id: str,
    is_customer_login: bool,
    customer_id: Optional[str],
    guest_session_id: Optional[str],
    store_code: str,
) -> str:
    """Client-scoped thread ids so tenants never collide in the Postgres checkpointer."""
    if is_customer_login and customer_id:
        return f"{client_id}::customer::{customer_id}"
    if guest_session_id:
        return f"{client_id}::guest::{guest_session_id}"
    return f"{client_id}::anon::{store_code}"


# ── Session CRUD ─────────────────────────────────────────────────────────────

def start_or_update_session(
    db: Session,
    *,
    client_id: str,
    thread_id: str,
    store_code: str,
    customer_id: Optional[str],
    guest_session_id: Optional[str],
    is_customer_login: bool,
    quote_id: Optional[str],
) -> dict[str, Any]:
    ensure_agent_schema(db)

    row = db.execute(
        text(
            "SELECT * FROM agent_chat_sessions WHERE client_id = :client_id AND thread_id = :thread_id LIMIT 1"
        ),
        {"client_id": client_id, "thread_id": thread_id},
    ).fetchone()

    if row is None:
        session_id = str(uuid.uuid4())
        db.execute(
            text(
                """
                INSERT INTO agent_chat_sessions (
                    id, client_id, thread_id, store_code, customer_id, guest_session_id,
                    is_customer_login, quote_id, started_at, last_activity_at, message_count, status
                ) VALUES (
                    :id, :client_id, :thread_id, :store_code, :customer_id, :guest_session_id,
                    :is_login, :quote_id, NOW(), NOW(), 0, 'active'
                )
                """
            ),
            {
                "id": session_id,
                "client_id": client_id,
                "thread_id": thread_id,
                "store_code": store_code,
                "customer_id": customer_id,
                "guest_session_id": guest_session_id,
                "is_login": 1 if is_customer_login else 0,
                "quote_id": quote_id,
            },
        )
        db.commit()
        return _session_row(
            db.execute(
                text("SELECT * FROM agent_chat_sessions WHERE id = :id"),
                {"id": session_id},
            ).fetchone()
        )

    db.execute(
        text(
            """
            UPDATE agent_chat_sessions
            SET last_activity_at = NOW(),
                customer_id = COALESCE(:customer_id, customer_id),
                guest_session_id = COALESCE(:guest_session_id, guest_session_id),
                is_customer_login = :is_login,
                quote_id = COALESCE(:quote_id, quote_id),
                status = 'active'
            WHERE id = :id
            """
        ),
        {
            "id": row.id,
            "customer_id": customer_id,
            "guest_session_id": guest_session_id,
            "is_login": 1 if is_customer_login else 0,
            "quote_id": quote_id,
        },
    )
    db.commit()
    return _session_row(
        db.execute(
            text("SELECT * FROM agent_chat_sessions WHERE id = :id"),
            {"id": row.id},
        ).fetchone()
    )


def _session_row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "session_id": row.id,
        "client_id": row.client_id,
        "thread_id": row.thread_id,
        "store_code": row.store_code,
        "customer_id": row.customer_id,
        "guest_session_id": row.guest_session_id,
        "is_customer_login": bool(row.is_customer_login),
        "quote_id": row.quote_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "last_activity_at": row.last_activity_at.isoformat() if row.last_activity_at else None,
        "message_count": int(row.message_count or 0),
        "status": row.status,
    }


# ── Message append ───────────────────────────────────────────────────────────

def append_message(
    db: Session,
    *,
    session_id: str,
    client_id: str,
    thread_id: str,
    role: str,
    content: Optional[str],
    agent_name: Optional[str] = None,
    tool_calls: Optional[list[dict]] = None,
    tool_name: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_cost: float = 0.0,
    response_time_ms: int = 0,
) -> str:
    ensure_agent_schema(db)
    message_id = str(uuid.uuid4())
    total_tokens = int((input_tokens or 0) + (output_tokens or 0))
    db.execute(
        text(
            """
            INSERT INTO agent_chat_messages (
                id, session_id, client_id, thread_id, role, agent_name,
                content, tool_calls_json, tool_name,
                input_tokens, output_tokens, total_tokens, total_cost, response_time_ms
            ) VALUES (
                :id, :session_id, :client_id, :thread_id, :role, :agent_name,
                :content, :tool_calls_json, :tool_name,
                :input_tokens, :output_tokens, :total_tokens, :total_cost, :response_time_ms
            )
            """
        ),
        {
            "id": message_id,
            "session_id": session_id,
            "client_id": client_id,
            "thread_id": thread_id,
            "role": role,
            "agent_name": agent_name,
            "content": content,
            "tool_calls_json": json.dumps(tool_calls) if tool_calls else None,
            "tool_name": tool_name,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": total_tokens,
            "total_cost": float(total_cost or 0),
            "response_time_ms": int(response_time_ms or 0),
        },
    )
    db.execute(
        text(
            "UPDATE agent_chat_sessions SET message_count = message_count + 1, last_activity_at = NOW() WHERE id = :sid"
        ),
        {"sid": session_id},
    )
    db.commit()
    return message_id


# ── Guest → customer merge ───────────────────────────────────────────────────

def merge_guest_into_customer(
    db: Session,
    *,
    client_id: str,
    guest_thread_id: str,
    customer_thread_id: str,
    customer_id: str,
) -> int:
    """Reassign every guest_thread_id message/session to the customer's thread.
    Returns number of MySQL rows updated. Postgres checkpointer migration is
    handled separately by the orchestrator (migrate_thread_history)."""
    if not guest_thread_id or not customer_thread_id or guest_thread_id == customer_thread_id:
        return 0

    ensure_agent_schema(db)

    res1 = db.execute(
        text(
            """
            UPDATE agent_chat_messages
            SET thread_id = :new
            WHERE client_id = :client_id AND thread_id = :old
            """
        ),
        {"client_id": client_id, "old": guest_thread_id, "new": customer_thread_id},
    )
    res2 = db.execute(
        text(
            """
            UPDATE agent_chat_sessions
            SET thread_id = :new, customer_id = :customer_id, is_customer_login = 1
            WHERE client_id = :client_id AND thread_id = :old
            """
        ),
        {
            "client_id": client_id,
            "old": guest_thread_id,
            "new": customer_thread_id,
            "customer_id": customer_id,
        },
    )
    db.commit()
    return int((res1.rowcount or 0) + (res2.rowcount or 0))


# ── Retention cleanup ────────────────────────────────────────────────────────

def purge_old_chats(db: Session, *, client_id: str, older_than_days: int) -> int:
    if older_than_days <= 0:
        return 0
    ensure_agent_schema(db)

    deleted_msgs = db.execute(
        text(
            """
            DELETE m FROM agent_chat_messages m
            JOIN agent_chat_sessions s ON s.id = m.session_id
            WHERE s.client_id = :client_id
              AND s.last_activity_at < (NOW() - INTERVAL :days DAY)
            """
        ),
        {"client_id": client_id, "days": older_than_days},
    )
    deleted_sessions = db.execute(
        text(
            """
            DELETE FROM agent_chat_sessions
            WHERE client_id = :client_id
              AND last_activity_at < (NOW() - INTERVAL :days DAY)
            """
        ),
        {"client_id": client_id, "days": older_than_days},
    )
    db.commit()
    return int((deleted_msgs.rowcount or 0) + (deleted_sessions.rowcount or 0))


# ── Reads (for router + dashboard) ───────────────────────────────────────────

def list_sessions(
    db: Session,
    *,
    client_id: str,
    store_code: Optional[str] = None,
    customer_id: Optional[str] = None,
    guest_session_id: Optional[str] = None,
    only_logged_in: Optional[bool] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    ensure_agent_schema(db)
    conditions = ["client_id = :client_id"]
    params: dict[str, Any] = {"client_id": client_id, "limit": limit, "offset": offset}
    if store_code:
        conditions.append("store_code = :store_code")
        params["store_code"] = store_code
    if customer_id:
        conditions.append("customer_id = :customer_id")
        params["customer_id"] = customer_id
    if guest_session_id:
        conditions.append("guest_session_id = :guest_session_id")
        params["guest_session_id"] = guest_session_id
    if only_logged_in is True:
        conditions.append("is_customer_login = 1")
    elif only_logged_in is False:
        conditions.append("is_customer_login = 0")
    if since:
        conditions.append("last_activity_at >= :since")
        params["since"] = since
    if until:
        conditions.append("last_activity_at <= :until")
        params["until"] = until

    rows = db.execute(
        text(
            f"""
            SELECT * FROM agent_chat_sessions
            WHERE {" AND ".join(conditions)}
            ORDER BY last_activity_at DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).fetchall()
    return [_session_row(r) for r in rows]


def get_session_messages(
    db: Session,
    *,
    client_id: str,
    session_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    ensure_agent_schema(db)
    rows = db.execute(
        text(
            """
            SELECT * FROM agent_chat_messages
            WHERE client_id = :client_id AND session_id = :session_id
            ORDER BY created_at ASC
            LIMIT :limit
            """
        ),
        {"client_id": client_id, "session_id": session_id, "limit": limit},
    ).fetchall()
    return [
        {
            "message_id": r.id,
            "role": r.role,
            "agent_name": r.agent_name,
            "tool_name": r.tool_name,
            "content": r.content,
            "tool_calls": json.loads(r.tool_calls_json) if r.tool_calls_json else None,
            "input_tokens": int(r.input_tokens or 0),
            "output_tokens": int(r.output_tokens or 0),
            "total_tokens": int(r.total_tokens or 0),
            "total_cost": float(r.total_cost or 0),
            "response_time_ms": int(r.response_time_ms or 0),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_session(db: Session, *, client_id: str, session_id: str) -> Optional[dict[str, Any]]:
    ensure_agent_schema(db)
    row = db.execute(
        text(
            "SELECT * FROM agent_chat_sessions WHERE client_id = :client_id AND id = :session_id LIMIT 1"
        ),
        {"client_id": client_id, "session_id": session_id},
    ).fetchone()
    return _session_row(row) if row else None
