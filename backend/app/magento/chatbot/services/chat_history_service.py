"""
Deprecated — chat history now lives in the Magento module.

This stub keeps imports resolvable. Do not call these helpers from new code.
"""

from __future__ import annotations

from typing import Any


def build_thread_id(
    *,
    client_id: str,
    is_customer_login: bool,
    customer_id: Any = None,
    guest_session_id: Any = None,
    store_code: str = "default",
) -> str:
    """Back-compat helper. Thread id is still client-scoped so LangGraph trace ids don't collide across tenants."""
    if is_customer_login and customer_id:
        return f"{client_id}::customer::{customer_id}"
    if guest_session_id:
        return f"{client_id}::guest::{guest_session_id}"
    return f"{client_id}::anon::{store_code}"
