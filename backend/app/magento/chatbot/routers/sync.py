"""
Mixed-content sync for the agent chatbot.

  POST   /api/magento/chatbot/agent/sync/batch
  POST   /api/magento/chatbot/agent/sync/delete
  GET    /api/magento/chatbot/agent/sync/status

A single batch can carry a mix of `content_type` values (product, cms_page,
cms_block, widget, store_config). Each item is formatted via product_formatter,
embedded via the existing Gemini embedder, and upserted into the per-tenant
Qdrant collection. Attribute and category vocabularies are merged per-store.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.services.cache_service import invalidate_client_results
from backend.app.services.database import get_db
from backend.app.services.embedder import embed_document
from backend.app.services.license_service import increment_ingest_count
from backend.app.services.qdrant_service import (
    delete_content_item,
    get_client_content_counts,
    get_client_product_count,
    upsert_content_item,
)

from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
    maybe_persist_magento_creds,
)
from backend.app.magento.chatbot.services import vocab_service
from backend.app.magento.chatbot.services.product_formatter import format_item

router = APIRouter()

SUPPORTED_TYPES = {"product", "cms_page", "cms_block", "widget", "store_config"}


class SyncItem(BaseModel):
    entity_id: str
    content_type: str
    store_code: str = "default"
    # Bag of raw fields the formatter will inspect (no rigid schema — the Magento module
    # defines what to send per content type; see Model/Content/*ContentProvider.php).
    payload: dict[str, Any] = Field(default_factory=dict)


class SyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    items: list[SyncItem] = Field(default_factory=list)
    batch_number: int = 1
    total_batches: int = 1
    store_code: str = "default"


class SyncDeleteItem(BaseModel):
    entity_id: str
    content_type: str


class SyncDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    items: list[SyncDeleteItem] = Field(default_factory=list)


@router.post("/magento/chatbot/agent/sync/batch")
def sync_batch(
    req: SyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    x_magento_creds: Optional[str] = Header(None, alias="X-Magento-Admin-Creds-Encrypted"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )
    maybe_persist_magento_creds(
        db=db,
        client_id=license_data["client_id"],
        license_key=license_data["license_key"],
        encrypted_creds_header=x_magento_creds,
    )

    # Quota check against the *product* limit — non-product content is free.
    incoming_products = sum(1 for it in req.items if it.content_type == "product")
    if incoming_products:
        current = get_client_product_count(license_data["client_id"], license_data["domain"])
        if current + incoming_products > license_data["product_limit"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Product limit exceeded. Current: {current}, Incoming: {incoming_products}, "
                    f"Limit: {license_data['product_limit']}"
                ),
            )

    embedding_api_key = decrypt_llm_key(
        x_llm_api_key_encrypted or req.llm_api_key_encrypted, license_data["license_key"]
    )

    attribute_vocab_sink: dict[str, set[str]] = defaultdict(set)
    category_vocab_sink: dict[str, dict[str, str]] = {}

    success_ids: list[str] = []
    failed_ids: list[str] = []
    success_by_type: dict[str, int] = defaultdict(int)

    for item in req.items:
        if item.content_type not in SUPPORTED_TYPES:
            failed_ids.append(item.entity_id)
            continue

        try:
            text_for_embed, payload = format_item(
                item.content_type,
                item.payload,
                attribute_vocab_sink=attribute_vocab_sink if item.content_type == "product" else None,
                category_vocab_sink=category_vocab_sink if item.content_type == "product" else None,
            )
            payload["embedded_text"] = text_for_embed
            payload["store_code"] = item.store_code or req.store_code

            vector = embed_document(text_for_embed, embedding_api_key, license_data["client_id"])

            upsert_content_item(
                client_id=license_data["client_id"],
                domain=license_data["domain"],
                content_type=item.content_type,
                entity_id=item.entity_id,
                vector=vector,
                payload=payload,
            )
            success_ids.append(item.entity_id)
            success_by_type[item.content_type] += 1
        except Exception:
            failed_ids.append(item.entity_id)

    if attribute_vocab_sink:
        try:
            vocab_service.merge_attributes(
                db, license_data["client_id"], req.store_code, attribute_vocab_sink
            )
        except Exception:
            pass
    if category_vocab_sink:
        try:
            vocab_service.merge_categories(
                db, license_data["client_id"], req.store_code, category_vocab_sink
            )
        except Exception:
            pass

    if success_by_type.get("product"):
        increment_ingest_count(db, license_data["client_id"], count=success_by_type["product"])

    if req.batch_number >= req.total_batches:
        try:
            invalidate_client_results(license_data["client_id"])
        except Exception:
            pass

    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "by_type": dict(success_by_type),
        "batch_number": req.batch_number,
        "total_batches": req.total_batches,
        "is_last_batch": req.batch_number >= req.total_batches,
    }


@router.post("/magento/chatbot/agent/sync/delete")
def sync_delete(
    req: SyncDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    deleted = 0
    for item in req.items:
        if item.content_type not in SUPPORTED_TYPES:
            continue
        try:
            delete_content_item(
                client_id=license_data["client_id"],
                domain=license_data["domain"],
                content_type=item.content_type,
                entity_id=item.entity_id,
            )
            deleted += 1
        except Exception:
            pass

    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass
    return {"deleted": deleted}


@router.get("/magento/chatbot/agent/sync/status")
def sync_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    counts = get_client_content_counts(
        license_data["client_id"],
        license_data["domain"],
        list(SUPPORTED_TYPES),
    )
    return {
        "counts": counts,
        "total_indexed": sum(int(c) for c in counts.values()),
    }
