"""Analytics behind the AI Chatbot's merchant dashboard.

  get_dashboard_data() -> GET /api/magento/chatbot/dashboard
  get_usage_data()     -> GET /api/magento/chatbot/usage

WHY get_usage_data() WAS RETURNING 500
--------------------------------------
Its three queries named `token_usage_tracking`, which the v2 billing migration
renamed to `token_usage_tracking_archive_v1`. Nothing here swallows, so every
call raised pymysql 1146 straight out of routers/chatbot.py:474 - HTTP 500 on a
live, authenticated, merchant-facing endpoint. This is the same defect as
magento/chatbot/routers/usage.py:50, in a second file; it was not in the review
that found that one, and it is reproducible with the same licence key.

They now read usage_ledger_read.LEDGER - the frozen v1 archive and the live v2
`usage_events` as one table. See that module for why this is a union and not a
repoint: usage_events is empty until v2 licences are issued, so a repoint swaps
a 500 for a merchant dashboard of confident zeros.

DURING THE DUAL-READ WINDOW these figures are pre-migration history that stops
at the archive's last row, because current traffic is recorded in neither
ledger. The response carries `usage_source` so a caller can tell an absent
number from a small one; nothing else in the shape changed.

THE CHAT-TYPE FILTER WAS ALSO WRONG, INDEPENDENTLY OF THE RENAME
----------------------------------------------------------------
The three queries filtered `query_type IN ('chat_answer', 'chat_context',
'chat_rewrite')` - hardcoded, three times, and NOT the CHAT_QUERY_TYPES constant
declared at the top of this module, which held the same three values and was
read by nothing. chat_context and chat_rewrite have never been written; the two
that have, chat_intent and chat_tool_call, were both absent from the list. On
the archive that is 29.04% of every chat-attributed rupee missing from a
merchant-facing cost panel. The list now lives once, in
usage_ledger_read.CHAT_CALL_TYPES, and is bound rather than interpolated.

THE chat_* TABLES ARE A DIFFERENT STORY - DO NOT "CLEAN THEM UP"
-----------------------------------------------------------------
get_dashboard_data() reads chat_conversations and chat_messages, which are
absent from this database and from init/01-schema.sql. That is NOT evidence they
are dead: ensure_chat_tables() runs CREATE TABLE IF NOT EXISTS for all three at
the top of every public function here and in conversation_service, and
routers/chatbot.py writes them on the live /magento/chatbot/message route. They
come into existence the first time a shopper sends a message, and on any
deployment where that has happened they hold the entire production conversation
history and nothing else does. See the long comment above
schema_v2.LEGACY_TABLES_TO_DROP, which was corrected once already for asserting
the opposite.

So a zero from get_dashboard_data() is honest in a way the token figures were
not: ensure_chat_tables() guarantees the tables exist by the time the SELECT
runs, so "0 conversations" means zero conversations rather than a missing table.
Left exactly as it was.

ONE THING THIS MODULE DOES NOT RECONCILE. get_dashboard_data().total_cost comes
from chat_messages.cost and get_usage_data().summary.total_cost comes from the
token ledger. They are two independent records of the same spend, written by
different code paths, and nothing checks them against each other. Fixing that
means picking one as authoritative, which is a decision about billing and not a
tidy-up.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from backend.app.services.conversation_service import ensure_chat_tables
from backend.app.services.qdrant_service import get_client_content_counts
from backend.app.services.usage_ledger_read import CHAT_CALL_TYPES, LEDGER, provenance

# Kept as this module's public name for the set, now sourced from the one place
# it is defined rather than being a fourth copy that drifts. It was previously a
# three-value tuple that no query read - the queries carried their own literal
# lists - which is how it stayed wrong without anybody noticing.
CHAT_QUERY_TYPES = CHAT_CALL_TYPES


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
        # From chat_messages.cost, NOT from the token ledger - see the module
        # docstring. These two numbers are not reconciled and can disagree.
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
        # Expanded by SQLAlchemy into one bind per value. Bound rather than
        # interpolated even though the tuple is a module constant: the moment
        # somebody makes the set configurable, an f-string here becomes an
        # injection point and nothing about the call site would look different.
        "chat_types": list(CHAT_CALL_TYPES),
    }
    chat_types = bindparam("chat_types", expanding=True)

    summary_row = db.execute(
        text(
            f"""
            SELECT
                COUNT(*) AS total_requests,
                SUM(total_tokens) AS total_tokens,
                SUM(total_cost) AS total_cost
            FROM {LEDGER} u
            WHERE client_id = :client_id
              AND query_type IN :chat_types
              AND created_at >= :start_date
            """
        ).bindparams(chat_types),
        params,
    ).fetchone()

    models = db.execute(
        text(
            f"""
            SELECT
                llm_provider,
                llm_model,
                query_type,
                COUNT(*) AS request_count,
                SUM(total_tokens) AS total_tokens,
                SUM(total_cost) AS total_cost
            FROM {LEDGER} u
            WHERE client_id = :client_id
              AND query_type IN :chat_types
              AND created_at >= :start_date
            GROUP BY llm_provider, llm_model, query_type
            ORDER BY total_cost DESC
            """
        ).bindparams(chat_types),
        params,
    ).fetchall()

    hourly = db.execute(
        text(
            f"""
            SELECT
                DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00') AS hour,
                SUM(total_cost) AS total_cost,
                SUM(total_tokens) AS total_tokens,
                COUNT(*) AS request_count
            FROM {LEDGER} u
            WHERE client_id = :client_id
              AND query_type IN :chat_types
              AND created_at >= :start_date
            GROUP BY DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00')
            ORDER BY hour ASC
            """
        ).bindparams(chat_types),
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
        # Added key, existing keys untouched. Without it a merchant on a short
        # window cannot tell "no chat spend" from "spend not recorded since the
        # migration", and those are the two possibilities today.
        "usage_source": provenance(db),
    }
