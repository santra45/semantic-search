"""Auth for the WordPress Q&A routers.

Same tenant contract as the Magento side — Bearer license key resolves to a
client_id + licensed domain, and the Origin/Referer of the incoming request
must belong to that domain. What's absent is deliberate: WordPress has no
equivalent of `magento_creds_service` / `admin_token_service`, because this
module never calls back into the store's REST API. It reads Qdrant and it
answers. That makes this file a third the size of its Magento twin, and the
smaller surface is the point — there is no admin credential to leak here.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.license_service import (
    check_search_quota,
    extract_license_key_from_authorization,
    validate_license_key,
)
from backend.app.services.llm_key_service import decrypt_key


def resolve_license_key(
    authorization: Optional[str],
    request_license: Optional[str],
) -> Optional[str]:
    """Header wins over body. The plugin always sends the header; the body
    field exists so a merchant debugging with curl doesn't have to."""
    return extract_license_key_from_authorization(authorization) or request_license


def authorize_request(
    *,
    request: Request,
    db: Session,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
) -> dict:
    """Validate the caller and return their license data (client_id, domain,
    product_limit, …) with the resolved key folded back in.

    Raises 401 (no key), 403 (bad key / wrong domain) or 429 (over quota).
    """
    license_key = resolve_license_key(authorization, request_license)
    if not license_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(license_key, db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    DomainAuthorizer(db).validate_request(request, license_data, api_key=x_api_key)
    _enforce_search_quota(db, license_data)

    license_data["license_key"] = license_key
    return license_data


def _enforce_search_quota(db: Session, license_data: dict) -> None:
    """Reject over-quota tenants with 429.

    Shares the Magento side's env gate (AICHATBOT_QUOTA_ENFORCEMENT) on
    purpose: one switch should arm or disarm quota enforcement for the whole
    deployment, not one per platform. Off by default, and fails OPEN on any
    lookup error — a quota check must never be why a paying merchant's
    product pages stop answering.
    """
    if os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "0") != "1":
        return
    try:
        client_id = license_data.get("client_id")
        search_limit = int(
            license_data.get("search_limit_per_month")
            or license_data.get("search_limit")
            or 0
        )
        if not client_id or search_limit <= 0:
            return  # no usable limit configured → don't block
        within_quota = check_search_quota(db, str(client_id), search_limit)
    except Exception:
        return  # fail open — never block on a lookup/DB error
    if not within_quota:
        raise HTTPException(
            status_code=429,
            detail="Monthly usage limit reached. Please contact the store.",
        )


def decrypt_llm_key(encrypted: Optional[str], license_key: str) -> Optional[str]:
    """The plugin ships the merchant's LLM key still encrypted; the license key
    is the secret. Returns None on any failure so the caller falls back to the
    server's own key rather than 500ing on a corrupted option value."""
    if not encrypted:
        return None
    try:
        return decrypt_key(encrypted, license_key)
    except Exception:
        return None
