"""Chatbot-specific configuration pulled from env."""

import os


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot_state")
POSTGRES_USER = os.getenv("POSTGRES_USER", "chatbot")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
# Off by default — chat history lives on the Magento side for privacy reasons.
# Setting it to "true" would only be useful for debugging graph runs; it never
# persists anything the storefront should store.
POSTGRES_CHECKPOINTER_ENABLED = os.getenv("POSTGRES_CHECKPOINTER_ENABLED", "false").lower() == "true"

MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("MAX_CHAT_HISTORY_MESSAGES", "8"))

# Agent LLM defaults. Per-request overrides (from the Magento module) take precedence.
DEFAULT_LLM_PROVIDER = os.getenv("CHAT_LLM_PROVIDER", "google")
DEFAULT_LLM_MODEL = os.getenv("CHAT_LLM_MODEL", "gemini-2.0-flash-lite")

INTENT_ROUTER_ENABLED = os.getenv("INTENT_ROUTER_ENABLED", "true").lower() == "true"

ADMIN_TOKEN_TTL_SECONDS = int(os.getenv("MAGENTO_ADMIN_TOKEN_TTL", str(4 * 60 * 60)))


def postgres_connection_string() -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
