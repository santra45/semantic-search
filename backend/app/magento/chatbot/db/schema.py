"""
Idempotent MySQL schema for the Magento multi-agent chatbot.

After the privacy refactor (chat history moved to Magento) this backend only
keeps two of its own tables:

  - client_magento_credentials : encrypted admin creds per client for REST token minting
  - agent_client_vocab         : per-client/per-store attribute & category vocabularies

`token_usage_tracking` (created elsewhere) still gets the billing columns used
by the agent endpoints — but no more customer_id / guest_session_id / thread_id.
Those joined the chat tables in the Magento DB.

Call ensure_agent_schema(db) on startup or on-demand from any router.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS client_magento_credentials (
        client_id VARCHAR(64) PRIMARY KEY,
        base_url VARCHAR(512) NOT NULL,
        admin_username_encrypted MEDIUMTEXT NOT NULL,
        admin_password_encrypted MEDIUMTEXT NOT NULL,
        api_version VARCHAR(16) NOT NULL DEFAULT 'V1',
        verify_ssl TINYINT(1) NOT NULL DEFAULT 1,
        default_store_code VARCHAR(64) NOT NULL DEFAULT 'default',
        last_token_minted_at DATETIME NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_client_vocab (
        client_id VARCHAR(64) NOT NULL,
        store_code VARCHAR(64) NOT NULL DEFAULT 'default',
        vocab_type VARCHAR(32) NOT NULL,
        vocab_json MEDIUMTEXT NOT NULL,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (client_id, store_code, vocab_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


_ENSURED = False


def ensure_agent_schema(db: Session) -> None:
    """Create the backend's own tables. Chat tables live on the Magento side."""
    global _ENSURED
    if _ENSURED:
        return

    for stmt in _STATEMENTS:
        db.execute(text(stmt))

    db.commit()
    _ENSURED = True
