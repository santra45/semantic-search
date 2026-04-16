from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.services.chat_analytics_service import get_dashboard_data, get_usage_data
from backend.app.services.chat_response_service import DEFAULT_REFUSAL, generate_grounded_answer
from backend.app.services.chat_retrieval_service import DEFAULT_ALLOWED_TYPES, retrieve_evidence
from backend.app.services.content_ingest_service import delete_items, ingest_items
from backend.app.services.conversation_service import (
    append_turn,
    get_history,
    get_recent_context,
    list_conversations,
    reset_session,
    start_or_get_conversation,
)
from backend.app.services.database import get_db
from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.license_service import (
    extract_license_key_from_authorization,
    increment_ingest_count,
    validate_license_key,
)
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.qdrant_service import get_client_content_counts, get_client_product_count

router = APIRouter()


class ChatbotSyncItem(BaseModel):
    entity_id: str
    content_type: str
    title: str = ""
    content: str = ""
    summary: str = ""
    permalink: str = ""
    status: str = "active"
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    sku: str = ""
    name: str = ""
    categories: str = ""
    tags: str = ""
    description: str = ""
    short_description: str = ""
    price: float = 0
    regular_price: float = 0
    sale_price: float = 0
    currency: str = ""
    currency_symbol: str = ""
    on_sale: bool = False
    image_url: str = ""
    stock_status: str = "instock"
    average_rating: float = 0
    attributes: list[dict[str, Any]] = Field(default_factory=list)


class ChatbotSyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    items: list[ChatbotSyncItem] = Field(default_factory=list)
    batch_number: int = 1
    total_batches: int = 1
    llm_api_key_encrypted: Optional[str] = None


class ChatbotDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    items: list[ChatbotSyncItem] = Field(default_factory=list)


class ChatSessionStartRequest(BaseModel):
    license_key: Optional[str] = None
    session_id: str
    store_id: str
    customer_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatMessageRequest(BaseModel):
    license_key: Optional[str] = None
    session_id: str
    conversation_id: Optional[str] = None
    message: str
    store_id: str
    customer_id: Optional[str] = None
    allowed_content_types: list[str] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None


def resolve_headers(
    authorization: Optional[str],
    x_api_key: Optional[str],
    x_llm_api_key_encrypted: Optional[str],
    request_license: Optional[str],
    request_llm_key: Optional[str],
) -> dict[str, Optional[str]]:
    return {
        "license_key": extract_license_key_from_authorization(authorization) or request_license,
        "api_key": x_api_key,
        "llm_api_key_encrypted": x_llm_api_key_encrypted or request_llm_key,
    }


def validate_chatbot_request(
    request: Request,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
    db: Session,
) -> dict[str, Any]:
    headers = resolve_headers(authorization, x_api_key, None, request_license, None)
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])
    return license_data


@router.post("/magento/chatbot/sync/batch")
def chatbot_sync_batch(
    req: ChatbotSyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    headers = resolve_headers(
        authorization,
        x_api_key,
        x_llm_api_key_encrypted,
        req.license_key,
        req.llm_api_key_encrypted,
    )
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    incoming_product_count = len([
        item for item in req.items if item.content_type == "product"
    ])
    current_product_count = get_client_product_count(license_data["client_id"], license_data["domain"])
    if current_product_count + incoming_product_count > license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail="Product limit exceeded for this license.",
        )

    embedding_api_key = None
    if headers["llm_api_key_encrypted"]:
        try:
            embedding_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
        except Exception:
            embedding_api_key = None

    payload = ingest_items(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        items=[item.model_dump() for item in req.items],
        embedding_api_key=embedding_api_key,
    )

    if payload["success_count"]:
        increment_ingest_count(db, license_data["client_id"], count=payload["success_count"])

    return {
        **payload,
        "batch_number": req.batch_number,
        "total_batches": req.total_batches,
        "is_last_batch": req.batch_number >= req.total_batches,
    }


@router.post("/magento/chatbot/sync/delete")
def chatbot_sync_delete(
    req: ChatbotDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, req.license_key, db)
    return delete_items(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        items=[item.model_dump() for item in req.items],
    )


def _build_content_counts_response(client_id: str, domain: str) -> dict[str, Any]:
    counts = get_client_content_counts(client_id, domain, DEFAULT_ALLOWED_TYPES)
    product_ready = int(counts.get("product", 0)) > 0
    non_product_ready = any(
        int(count) > 0 for content_type, count in counts.items() if content_type != "product"
    )
    return {
        "counts": counts,
        "product_ready": product_ready,
        "non_product_ready": non_product_ready,
        "total_indexed": sum(int(count) for count in counts.values()),
    }


@router.get("/magento/chatbot/sync/status")
def chatbot_sync_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return _build_content_counts_response(license_data["client_id"], license_data["domain"])


@router.get("/magento/chatbot/sync/content-counts")
def chatbot_content_counts(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return _build_content_counts_response(license_data["client_id"], license_data["domain"])


@router.post("/magento/chatbot/session/start")
def chatbot_session_start(
    req: ChatSessionStartRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, req.license_key, db)
    conversation = start_or_get_conversation(
        db=db,
        client_id=license_data["client_id"],
        store_id=req.store_id,
        session_id=req.session_id,
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
    )
    history = get_history(
        db=db,
        client_id=license_data["client_id"],
        conversation_id=conversation["conversation_id"],
    )
    return {
        **conversation,
        "history": history["messages"],
    }


@router.post("/magento/chatbot/message")
def chatbot_message(
    req: ChatMessageRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    start_time = time.time()
    headers = resolve_headers(
        authorization,
        x_api_key,
        x_llm_api_key_encrypted,
        req.license_key,
        req.llm_api_key_encrypted,
    )
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation = start_or_get_conversation(
        db=db,
        client_id=license_data["client_id"],
        store_id=req.store_id,
        session_id=req.session_id,
        customer_id=req.customer_id,
        conversation_id=req.conversation_id,
    )

    decrypted_llm_key = None
    if headers["llm_api_key_encrypted"]:
        try:
            decrypted_llm_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
        except Exception:
            decrypted_llm_key = None

    evidence = retrieve_evidence(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        message=req.message.strip(),
        embedding_api_key=decrypted_llm_key,
        allowed_content_types=req.allowed_content_types or DEFAULT_ALLOWED_TYPES,
    )

    if evidence["grounded"]:
        context = get_recent_context(db, conversation["conversation_id"])
        answer_payload = generate_grounded_answer(
            message=req.message.strip(),
            sources=evidence["sources"],
            conversation_history=context,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_api_key=decrypted_llm_key,
            client_id=license_data["client_id"],
        )
    else:
        answer_payload = {
            "answer": DEFAULT_REFUSAL,
            "follow_up_suggestions": [],
            "grounded": False,
            "usage": {},
        }

    response_time_ms = int((time.time() - start_time) * 1000)
    turn = append_turn(
        db=db,
        conversation_id=conversation["conversation_id"],
        message_text=req.message.strip(),
        response_text=answer_payload["answer"],
        sources=evidence["sources"],
        usage=answer_payload.get("usage", {}),
        grounded=bool(answer_payload.get("grounded", False)),
        response_time_ms=response_time_ms,
    )

    return {
        "conversation_id": conversation["conversation_id"],
        "message_id": turn["assistant_message_id"],
        "answer": answer_payload["answer"],
        "sources": evidence["sources"],
        "follow_up_suggestions": answer_payload.get("follow_up_suggestions", []),
        "usage": answer_payload.get("usage", {}),
        "grounded": bool(answer_payload.get("grounded", False)),
        "retrieval_confidence": evidence["confidence"],
    }


@router.get("/magento/chatbot/history")
def chatbot_history(
    request: Request,
    conversation_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return get_history(
        db=db,
        client_id=license_data["client_id"],
        conversation_id=conversation_id,
        session_id=session_id,
    )


@router.get("/magento/chatbot/conversations")
def chatbot_conversations(
    request: Request,
    store_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return {
        "conversations": list_conversations(
            db=db,
            client_id=license_data["client_id"],
            store_id=store_id,
        )
    }


@router.get("/magento/chatbot/usage")
def chatbot_usage(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return get_usage_data(db, license_data["client_id"], days=days)


@router.get("/magento/chatbot/dashboard")
def chatbot_dashboard(
    request: Request,
    store_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, None, db)
    return get_dashboard_data(
        db=db,
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        store_id=store_id,
    )


@router.post("/magento/chatbot/reset")
def chatbot_reset(
    req: ChatSessionStartRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = validate_chatbot_request(request, authorization, x_api_key, req.license_key, db)
    closed = reset_session(
        db=db,
        client_id=license_data["client_id"],
        session_id=req.session_id,
        store_id=req.store_id,
    )
    return {"closed_conversations": closed}
