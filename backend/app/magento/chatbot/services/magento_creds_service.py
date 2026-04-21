"""
Per-client Magento admin credentials.

Flow:
  1. Magento module admin enters credentials in system.xml (Magento-encrypted).
  2. Module decrypts and re-encrypts with its license key (AES-256-CBC, same cipher used today for
     the LLM API key — see Helper/Encryption.php). The encrypted blob is sent in the
     `X-Magento-Admin-Creds-Encrypted` header on the first request from a fresh client.
  3. Backend decrypts with the license key (backend/app/services/llm_key_service.decrypt_key)
     and stores username + password separately in MySQL, each blob encrypted with
     a Fernet key derived from JWT_SECRET. Plaintext never hits disk.

Storage schema: see backend/app/magento/chatbot/db/schema.py (client_magento_credentials).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Optional

from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.magento.chatbot.db.schema import ensure_agent_schema
from backend.app.services.llm_key_service import decrypt_key


def _fernet() -> Fernet:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET is required to encrypt Magento admin credentials")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def _encrypt_at_rest(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def _decrypt_at_rest(value: str) -> str:
    return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")


def store_credentials_from_header(
    db: Session,
    client_id: str,
    license_key: str,
    encrypted_blob: str,
) -> bool:
    """Accept the per-request X-Magento-Admin-Creds-Encrypted header and persist credentials.

    The encrypted blob decrypts to JSON:
      {"base_url": "...", "username": "...", "password": "...",
       "api_version": "V1", "verify_ssl": true, "default_store_code": "default"}

    Returns True if stored (or already up-to-date), False on decryption failure.
    """
    if not encrypted_blob:
        return False

    try:
        plaintext = decrypt_key(encrypted_blob, license_key)
        if not plaintext:
            return False
        payload: dict[str, Any] = json.loads(plaintext)
    except Exception:
        return False

    base_url = (payload.get("base_url") or "").rstrip("/")
    username = payload.get("username") or ""
    password = payload.get("password") or ""
    if not base_url or not username or not password:
        return False

    ensure_agent_schema(db)

    db.execute(
        text(
            """
            INSERT INTO client_magento_credentials (
                client_id, base_url, admin_username_encrypted, admin_password_encrypted,
                api_version, verify_ssl, default_store_code
            ) VALUES (
                :client_id, :base_url, :username, :password,
                :api_version, :verify_ssl, :store_code
            )
            ON DUPLICATE KEY UPDATE
                base_url = VALUES(base_url),
                admin_username_encrypted = VALUES(admin_username_encrypted),
                admin_password_encrypted = VALUES(admin_password_encrypted),
                api_version = VALUES(api_version),
                verify_ssl = VALUES(verify_ssl),
                default_store_code = VALUES(default_store_code)
            """
        ),
        {
            "client_id": client_id,
            "base_url": base_url,
            "username": _encrypt_at_rest(username),
            "password": _encrypt_at_rest(password),
            "api_version": payload.get("api_version") or "V1",
            "verify_ssl": 1 if payload.get("verify_ssl", True) else 0,
            "store_code": payload.get("default_store_code") or "default",
        },
    )
    db.commit()
    return True


def get_credentials(db: Session, client_id: str) -> Optional[dict]:
    """Return plaintext creds for the client, or None if not provisioned."""
    ensure_agent_schema(db)

    row = db.execute(
        text(
            """
            SELECT base_url, admin_username_encrypted, admin_password_encrypted,
                   api_version, verify_ssl, default_store_code
            FROM client_magento_credentials
            WHERE client_id = :client_id
            LIMIT 1
            """
        ),
        {"client_id": client_id},
    ).fetchone()

    if not row:
        return None

    try:
        return {
            "base_url": row.base_url,
            "username": _decrypt_at_rest(row.admin_username_encrypted),
            "password": _decrypt_at_rest(row.admin_password_encrypted),
            "api_version": row.api_version or "V1",
            "verify_ssl": bool(row.verify_ssl),
            "default_store_code": row.default_store_code or "default",
        }
    except Exception:
        return None


def touch_last_mint(db: Session, client_id: str) -> None:
    db.execute(
        text(
            "UPDATE client_magento_credentials SET last_token_minted_at = :now WHERE client_id = :client_id"
        ),
        {"client_id": client_id, "now": datetime.utcnow()},
    )
    db.commit()
