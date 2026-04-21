"""Shared helpers for the agent routers — auth, credential resolution, context building."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.license_service import (
    extract_license_key_from_authorization,
    validate_license_key,
)
from backend.app.services.llm_key_service import decrypt_key

from backend.app.magento.chatbot.agents.request_context import RequestContext
from backend.app.magento.chatbot.services import admin_token_service, magento_creds_service
from backend.app.magento.chatbot.services.magento_client import MagentoClient


def resolve_license_key(
    authorization: Optional[str],
    request_license: Optional[str],
) -> Optional[str]:
    return extract_license_key_from_authorization(authorization) or request_license


def authorize_request(
    *,
    request: Request,
    db: Session,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
) -> dict:
    license_key = resolve_license_key(authorization, request_license)
    if not license_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        license_data = validate_license_key(license_key, db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    DomainAuthorizer(db).validate_request(request, license_data, api_key=x_api_key)
    license_data["license_key"] = license_key
    return license_data


def maybe_persist_magento_creds(
    *,
    db: Session,
    client_id: str,
    license_key: str,
    encrypted_creds_header: Optional[str],
) -> None:
    """If the module included an admin-creds blob, decrypt + persist it (and reset the cached token)."""
    if not encrypted_creds_header:
        return
    if magento_creds_service.store_credentials_from_header(
        db, client_id, license_key, encrypted_creds_header
    ):
        admin_token_service.invalidate_token(client_id)


def decrypt_llm_key(encrypted: Optional[str], license_key: str) -> Optional[str]:
    if not encrypted:
        return None
    try:
        return decrypt_key(encrypted, license_key)
    except Exception:
        return None


async def build_request_context(
    *,
    db: Session,
    license_data: dict,
    store_code: str,
    customer_id: Optional[str],
    is_customer_login: bool,
    guest_session_id: Optional[str],
    quote_id: Optional[str],
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_api_key_encrypted: Optional[str],
) -> RequestContext:
    """Resolve credentials + mint token + construct a fully-wired RequestContext."""
    ctx = RequestContext(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        license_key=license_data["license_key"],
        store_code=store_code or "default",
        customer_id=int(customer_id) if customer_id and str(customer_id).isdigit() else None,
        is_customer_login=bool(is_customer_login),
        guest_session_id=guest_session_id,
        quote_id=quote_id or None,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=decrypt_llm_key(llm_api_key_encrypted, license_data["license_key"]),
    )

    creds = magento_creds_service.get_credentials(db, license_data["client_id"])
    if creds:
        token = await admin_token_service.get_or_mint_token(
            client_id=license_data["client_id"], creds=creds
        )
        if token:
            magento_creds_service.touch_last_mint(db, license_data["client_id"])
        ctx.magento_client = MagentoClient(
            client_id=license_data["client_id"],
            base_url=creds["base_url"],
            api_version=creds.get("api_version", "V1"),
            verify_ssl=bool(creds.get("verify_ssl", True)),
            store_code=ctx.store_code or creds.get("default_store_code", "default"),
            admin_token=token,
        )
    return ctx
