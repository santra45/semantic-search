"""
Idempotent MySQL schema for the Magento multi-agent chatbot.

Extends the existing semantic-search MySQL with:
  - client_magento_credentials   : encrypted admin creds per client for token minting
  - agent_chat_sessions          : one row per (client, thread) — customer OR guest
  - agent_chat_messages          : every user/assistant/tool turn with agent + token cost
  - agent_client_vocab           : per-client attribute & category vocabularies (replaces flat JSON)
  - token_usage_tracking         : extended with customer_id / guest_session_id / thread_id / agent_name

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
    CREATE TABLE IF NOT EXISTS agent_chat_sessions (
        id VARCHAR(64) PRIMARY KEY,
        client_id VARCHAR(64) NOT NULL,
        thread_id VARCHAR(191) NOT NULL,
        store_code VARCHAR(64) NOT NULL DEFAULT 'default',
        customer_id VARCHAR(64) NULL,
        guest_session_id VARCHAR(191) NULL,
        is_customer_login TINYINT(1) NOT NULL DEFAULT 0,
        quote_id VARCHAR(191) NULL,
        started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_activity_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        message_count INT NOT NULL DEFAULT 0,
        status VARCHAR(32) NOT NULL DEFAULT 'active',
        UNIQUE KEY uq_agent_chat_thread (client_id, thread_id),
        INDEX idx_agent_chat_client_customer (client_id, customer_id),
        INDEX idx_agent_chat_client_guest (client_id, guest_session_id),
        INDEX idx_agent_chat_client_store (client_id, store_code),
        INDEX idx_agent_chat_last_activity (client_id, last_activity_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_chat_messages (
        id VARCHAR(64) PRIMARY KEY,
        session_id VARCHAR(64) NOT NULL,
        client_id VARCHAR(64) NOT NULL,
        thread_id VARCHAR(191) NOT NULL,
        role VARCHAR(32) NOT NULL,
        agent_name VARCHAR(64) NULL,
        content MEDIUMTEXT NULL,
        tool_calls_json JSON NULL,
        tool_name VARCHAR(128) NULL,
        input_tokens INT NOT NULL DEFAULT 0,
        output_tokens INT NOT NULL DEFAULT 0,
        total_tokens INT NOT NULL DEFAULT 0,
        total_cost DECIMAL(14, 8) NOT NULL DEFAULT 0,
        response_time_ms INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_agent_msg_session (session_id, created_at),
        INDEX idx_agent_msg_client_created (client_id, created_at),
        INDEX idx_agent_msg_thread (thread_id, created_at)
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

# token_usage_tracking gets extra columns. MySQL 8 supports "ADD COLUMN IF NOT EXISTS" only via
# information_schema check, so we detect-then-alter.
_TOKEN_USAGE_EXTENSIONS = [
    ("customer_id", "ALTER TABLE token_usage_tracking ADD COLUMN customer_id VARCHAR(64) NULL"),
    ("guest_session_id", "ALTER TABLE token_usage_tracking ADD COLUMN guest_session_id VARCHAR(191) NULL"),
    ("thread_id", "ALTER TABLE token_usage_tracking ADD COLUMN thread_id VARCHAR(191) NULL"),
    ("agent_name", "ALTER TABLE token_usage_tracking ADD COLUMN agent_name VARCHAR(64) NULL"),
]

_TOKEN_USAGE_INDEXES = [
    ("idx_tut_client_customer", "ALTER TABLE token_usage_tracking ADD INDEX idx_tut_client_customer (client_id, customer_id)"),
    ("idx_tut_client_guest", "ALTER TABLE token_usage_tracking ADD INDEX idx_tut_client_guest (client_id, guest_session_id)"),
    ("idx_tut_thread", "ALTER TABLE token_usage_tracking ADD INDEX idx_tut_thread (thread_id)"),
]


_ENSURED = False


def ensure_agent_schema(db: Session) -> None:
    """Create chatbot tables and extend token_usage_tracking in place. Idempotent + cached per-process."""
    global _ENSURED
    if _ENSURED:
        return

    for stmt in _STATEMENTS:
        db.execute(text(stmt))

    # token_usage_tracking may not exist yet on fresh installs — only extend if it does.
    exists = db.execute(text("""
        SELECT COUNT(*) AS c
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = 'token_usage_tracking'
    """)).fetchone()

    if exists and int(exists.c) > 0:
        for column_name, alter_sql in _TOKEN_USAGE_EXTENSIONS:
            present = db.execute(text("""
                SELECT COUNT(*) AS c
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'token_usage_tracking'
                  AND column_name = :col
            """), {"col": column_name}).fetchone()
            if not present or int(present.c) == 0:
                db.execute(text(alter_sql))

        for index_name, alter_sql in _TOKEN_USAGE_INDEXES:
            present = db.execute(text("""
                SELECT COUNT(*) AS c
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = 'token_usage_tracking'
                  AND index_name = :idx
            """), {"idx": index_name}).fetchone()
            if not present or int(present.c) == 0:
                try:
                    db.execute(text(alter_sql))
                except Exception:
                    pass

    db.commit()
    _ENSURED = True
