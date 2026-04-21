"""
Process-level AsyncPostgresSaver + connection pool.

The pool and saver survive across chat requests so LangGraph doesn't re-negotiate
Postgres or re-run setup on every invocation. Thread migration (guest → customer)
rewrites every `checkpoint*` table's thread_id column — ported from
magento_chatbot/agents/orchestrator.py::migrate_thread_history.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.app.magento.chatbot.services.config import (
    POSTGRES_CHECKPOINTER_ENABLED,
    postgres_connection_string,
)

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_pool: Optional[AsyncConnectionPool] = None
_saver: Optional[AsyncPostgresSaver] = None


async def get_saver() -> Optional[AsyncPostgresSaver]:
    """Return the shared AsyncPostgresSaver or None when the checkpointer is disabled."""
    if not POSTGRES_CHECKPOINTER_ENABLED:
        return None

    global _pool, _saver
    if _saver is not None:
        return _saver

    async with _lock:
        if _saver is not None:
            return _saver
        try:
            pool = AsyncConnectionPool(
                conninfo=postgres_connection_string(),
                max_size=20,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            _pool = pool
            _saver = saver
            logger.info("AsyncPostgresSaver ready")
            return _saver
        except Exception as exc:
            logger.exception("Postgres checkpointer unavailable: %s", exc)
            return None


async def migrate_thread_history(from_thread_id: str, to_thread_id: str) -> bool:
    """Reassign every checkpoint row from the guest thread to the customer thread.

    Returns True on success, False if skipped (destination non-empty, pool missing, etc.)."""
    if not POSTGRES_CHECKPOINTER_ENABLED:
        return False
    if not from_thread_id or not to_thread_id or from_thread_id == to_thread_id:
        return False

    await get_saver()
    if _pool is None:
        return False

    try:
        async with _pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(1) AS cnt FROM checkpoints WHERE thread_id = %s",
                    (to_thread_id,),
                )
                row = await cur.fetchone()
                if row and int(row.get("cnt", 0)) > 0:
                    logger.info(
                        "Skip migration %s -> %s: destination already has checkpoints",
                        from_thread_id,
                        to_thread_id,
                    )
                    return False

                await cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND column_name = 'thread_id'
                      AND table_name LIKE 'checkpoint%'
                    ORDER BY table_name
                    """
                )
                tables = await cur.fetchall() or []
                for t in tables:
                    table_name = t.get("table_name")
                    if not table_name:
                        continue
                    await cur.execute(
                        f"UPDATE {table_name} SET thread_id = %s WHERE thread_id = %s",
                        (to_thread_id, from_thread_id),
                    )
                return True
    except Exception as exc:
        logger.exception("Thread migration failed: %s", exc)
        return False
