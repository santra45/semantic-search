from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def ensure_chat_tables(db: Session) -> None:
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS chat_conversations (
                id VARCHAR(64) PRIMARY KEY,
                client_id VARCHAR(64) NOT NULL,
                store_id VARCHAR(64) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                customer_id VARCHAR(64) NULL,
                started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_message_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(32) NOT NULL DEFAULT 'active',
                INDEX idx_chat_conversations_client_store (client_id, store_id),
                INDEX idx_chat_conversations_session (session_id)
            )
            """
        )
    )
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id VARCHAR(64) PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                role VARCHAR(32) NOT NULL,
                message_text MEDIUMTEXT NULL,
                response_text MEDIUMTEXT NULL,
                sources_json JSON NULL,
                tokens_json JSON NULL,
                cost DECIMAL(12, 6) NOT NULL DEFAULT 0,
                grounded TINYINT(1) NOT NULL DEFAULT 1,
                response_time_ms INT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_chat_messages_conversation (conversation_id, created_at)
            )
            """
        )
    )
    db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS chat_feedback (
                id VARCHAR(64) PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                message_id VARCHAR(64) NOT NULL,
                feedback_type VARCHAR(32) NOT NULL,
                resolution_flag TINYINT(1) NOT NULL DEFAULT 0,
                admin_notes TEXT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_chat_feedback_message (message_id)
            )
            """
        )
    )
    db.commit()


def _serialize(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True)


def _conversation_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "conversation_id": row.id,
        "client_id": row.client_id,
        "store_id": row.store_id,
        "session_id": row.session_id,
        "customer_id": row.customer_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "last_message_at": row.last_message_at.isoformat() if row.last_message_at else None,
        "status": row.status,
    }


def start_or_get_conversation(
    db: Session,
    client_id: str,
    store_id: str,
    session_id: str,
    customer_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    ensure_chat_tables(db)

    row = None
    if conversation_id:
        row = db.execute(
            text(
                """
                SELECT *
                FROM chat_conversations
                WHERE id = :conversation_id AND client_id = :client_id
                LIMIT 1
                """
            ),
            {"conversation_id": conversation_id, "client_id": client_id},
        ).fetchone()

    if row is None:
        row = db.execute(
            text(
                """
                SELECT *
                FROM chat_conversations
                WHERE client_id = :client_id
                  AND store_id = :store_id
                  AND session_id = :session_id
                  AND status = 'active'
                ORDER BY last_message_at DESC
                LIMIT 1
                """
            ),
            {
                "client_id": client_id,
                "store_id": store_id,
                "session_id": session_id,
            },
        ).fetchone()

    if row is None:
        conversation_id = conversation_id or str(uuid.uuid4())
        db.execute(
            text(
                """
                INSERT INTO chat_conversations (
                    id, client_id, store_id, session_id, customer_id, started_at, last_message_at, status
                ) VALUES (
                    :id, :client_id, :store_id, :session_id, :customer_id, NOW(), NOW(), 'active'
                )
                """
            ),
            {
                "id": conversation_id,
                "client_id": client_id,
                "store_id": store_id,
                "session_id": session_id,
                "customer_id": customer_id,
            },
        )
        db.commit()
        row = db.execute(
            text("SELECT * FROM chat_conversations WHERE id = :id LIMIT 1"),
            {"id": conversation_id},
        ).fetchone()
    elif customer_id and not row.customer_id:
        db.execute(
            text(
                """
                UPDATE chat_conversations
                SET customer_id = :customer_id
                WHERE id = :conversation_id
                """
            ),
            {"customer_id": customer_id, "conversation_id": row.id},
        )
        db.commit()
        row = db.execute(
            text("SELECT * FROM chat_conversations WHERE id = :id LIMIT 1"),
            {"id": row.id},
        ).fetchone()

    return _conversation_row_to_dict(row)


def append_turn(
    db: Session,
    conversation_id: str,
    message_text: str,
    response_text: str,
    sources: list[dict[str, Any]],
    usage: dict[str, Any],
    grounded: bool,
    response_time_ms: int = 0,
) -> dict[str, Any]:
    ensure_chat_tables(db)

    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())
    cost = float(usage.get("cost", 0) or 0)

    db.execute(
        text(
            """
            INSERT INTO chat_messages (
                id, conversation_id, role, message_text, response_text,
                sources_json, tokens_json, cost, grounded, response_time_ms
            ) VALUES (
                :id, :conversation_id, 'user', :message_text, NULL,
                NULL, NULL, 0, 1, 0
            )
            """
        ),
        {
            "id": user_message_id,
            "conversation_id": conversation_id,
            "message_text": message_text,
        },
    )

    db.execute(
        text(
            """
            INSERT INTO chat_messages (
                id, conversation_id, role, message_text, response_text,
                sources_json, tokens_json, cost, grounded, response_time_ms
            ) VALUES (
                :id, :conversation_id, 'assistant', NULL, :response_text,
                :sources_json, :tokens_json, :cost, :grounded, :response_time_ms
            )
            """
        ),
        {
            "id": assistant_message_id,
            "conversation_id": conversation_id,
            "response_text": response_text,
            "sources_json": _serialize(sources),
            "tokens_json": _serialize(usage),
            "cost": cost,
            "grounded": 1 if grounded else 0,
            "response_time_ms": response_time_ms,
        },
    )

    db.execute(
        text(
            """
            UPDATE chat_conversations
            SET last_message_at = NOW(), status = 'active'
            WHERE id = :conversation_id
            """
        ),
        {"conversation_id": conversation_id},
    )
    db.commit()

    return {
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "conversation_id": conversation_id,
    }


def get_history(
    db: Session,
    client_id: str,
    conversation_id: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    ensure_chat_tables(db)

    if not conversation_id and not session_id:
        raise ValueError("conversation_id or session_id is required")

    if not conversation_id and session_id:
        row = db.execute(
            text(
                """
                SELECT id
                FROM chat_conversations
                WHERE client_id = :client_id AND session_id = :session_id
                ORDER BY last_message_at DESC
                LIMIT 1
                """
            ),
            {"client_id": client_id, "session_id": session_id},
        ).fetchone()
        conversation_id = row.id if row else None

    if not conversation_id:
        return {"conversation_id": None, "messages": []}

    rows = db.execute(
        text(
            """
            SELECT *
            FROM chat_messages
            WHERE conversation_id = :conversation_id
            ORDER BY created_at ASC
            LIMIT :limit
            """
        ),
        {"conversation_id": conversation_id, "limit": limit},
    ).fetchall()

    messages = []
    for row in rows:
        messages.append(
            {
                "message_id": row.id,
                "role": row.role,
                "message_text": row.message_text,
                "response_text": row.response_text,
                "sources": json.loads(row.sources_json) if row.sources_json else [],
                "usage": json.loads(row.tokens_json) if row.tokens_json else {},
                "cost": float(row.cost or 0),
                "grounded": bool(row.grounded),
                "response_time_ms": int(row.response_time_ms or 0),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
        )

    return {"conversation_id": conversation_id, "messages": messages}


def list_conversations(
    db: Session,
    client_id: str,
    store_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    ensure_chat_tables(db)

    where = ["client_id = :client_id"]
    params: dict[str, Any] = {"client_id": client_id, "limit": limit}
    if store_id:
        where.append("store_id = :store_id")
        params["store_id"] = store_id

    rows = db.execute(
        text(
            f"""
            SELECT *
            FROM chat_conversations
            WHERE {' AND '.join(where)}
            ORDER BY last_message_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).fetchall()

    return [_conversation_row_to_dict(row) for row in rows]


def get_recent_context(
    db: Session,
    conversation_id: str,
    limit: int = 8,
) -> list[dict[str, str]]:
    ensure_chat_tables(db)

    rows = db.execute(
        text(
            """
            SELECT role, message_text, response_text
            FROM chat_messages
            WHERE conversation_id = :conversation_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
        ),
        {"conversation_id": conversation_id, "limit": limit},
    ).fetchall()

    context = []
    for row in reversed(rows):
        if row.role == "user" and row.message_text:
            context.append({"role": "user", "content": row.message_text})
        elif row.role == "assistant" and row.response_text:
            context.append({"role": "assistant", "content": row.response_text})
    return context


def reset_session(
    db: Session,
    client_id: str,
    session_id: str,
    store_id: str | None = None,
) -> int:
    ensure_chat_tables(db)

    conditions = ["client_id = :client_id", "session_id = :session_id", "status = 'active'"]
    params: dict[str, Any] = {"client_id": client_id, "session_id": session_id}
    if store_id:
        conditions.append("store_id = :store_id")
        params["store_id"] = store_id

    result = db.execute(
        text(
            f"""
            UPDATE chat_conversations
            SET status = 'closed', last_message_at = NOW()
            WHERE {' AND '.join(conditions)}
            """
        ),
        params,
    )
    db.commit()
    return int(result.rowcount or 0)
