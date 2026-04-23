"""
Pure retrieval endpoints. The backend is a Qdrant proxy now — zero chat logic.

  POST /api/magento/chatbot/retrieve/products    — semantic / structured product search
  POST /api/magento/chatbot/retrieve/content     — CMS pages / blocks / widgets / store config
  POST /api/magento/chatbot/retrieve/answer      — optional RAG summary over provided sources

All three are license + domain guarded. They never see chat history and never
touch any identifiers beyond client_id (already anonymous).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
    get_collection_name,
    qdrant,
    search_content as qdrant_search_content,
    search_products as qdrant_search_products,
)
from backend.app.services.token_usage_service import TokenUsageTracker

from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
    maybe_persist_magento_creds,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class ProductRetrieveRequest(BaseModel):
    license_key: Optional[str] = None
    query: Optional[str] = None
    skus: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=lambda: ["product"])
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    only_in_stock: bool = False
    attribute_filters: dict[str, str] = Field(default_factory=dict)  # {"color": "red", "size": "m"}
    category_id: Optional[str] = None
    limit: int = 8
    rerank: bool = False  # admin-toggled — small LLM rerank of top-N

    @field_validator("attribute_filters", mode="before")
    @classmethod
    def _coerce_attribute_filters(cls, value):
        """PHP json-encodes empty arrays as `[]`, which doesn't match a `dict`
        schema. Accept `None`/`[]`/list-of-{name,value}-dicts and coerce."""
        if value in (None, "", [], {}):
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
        if isinstance(value, list):
            out: dict[str, str] = {}
            for entry in value:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("key") or entry.get("code")
                    val  = entry.get("value") or entry.get("option")
                    if name and val:
                        out[str(name)] = str(val)
            return out
        return {}

    @field_validator("skus", "content_types", mode="before")
    @classmethod
    def _coerce_string_list(cls, value):
        if value in (None, "", [], {}):
            return []
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v) for v in value if v not in (None, "")]
        return []

    @field_validator("min_price", "max_price", mode="before")
    @classmethod
    def _coerce_optional_float(cls, value):
        if value in (None, "", [], {}):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_optional_str(cls, value):
        if value in (None, "", [], {}):
            return None
        return str(value)


class ContentRetrieveRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    content_types: list[str] = Field(default_factory=lambda: ["cms_page", "cms_block"])
    limit: int = 5

    @field_validator("content_types", mode="before")
    @classmethod
    def _coerce_types(cls, value):
        if value in (None, "", [], {}):
            return ["cms_page", "cms_block"]
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v) for v in value if v not in (None, "")]
        return ["cms_page", "cms_block"]


class AnswerRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/magento/chatbot/retrieve/products")
def retrieve_products(
    req: ProductRetrieveRequest,
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

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    # ── Direct SKU lookup path (no embedding, no LLM) ────────────────────────
    if req.skus:
        hits = _lookup_by_skus(client_id, domain, req.skus)
        return {
            "results": hits,
            "count": len(hits),
            "mode": "sku",
        }

    # ── Semantic search path ─────────────────────────────────────────────────
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query or skus is required")

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    query_vector = embed_query(req.query.strip(), embedding_api_key, client_id)

    # Fetch over-broad so the post-filter has room to narrow down.
    raw_limit = max(req.limit, req.limit * 3 if req.attribute_filters or req.category_id else req.limit)

    hits = qdrant_search_products(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=raw_limit,
        min_price=req.min_price,
        max_price=req.max_price,
        only_in_stock=req.only_in_stock,
        content_types=req.content_types or ["product"],
    )

    # Attribute + category filtering — these are stored as payload booleans,
    # so we filter server-side here (Qdrant's FieldCondition also works but
    # the Python side is simpler for dynamic keys and lets us keep the
    # qdrant_service generic).
    if req.attribute_filters:
        for attr, value in req.attribute_filters.items():
            key = f"attr_{_slug(attr)}_{_slug(value)}"
            hits = [h for h in hits if h.get(key) is True]
    if req.category_id:
        key = f"cat_{req.category_id}"
        hits = [h for h in hits if h.get(key) is True]

    hits = hits[: req.limit]

    mode = "semantic"
    if req.rerank and hits:
        try:
            hits = _llm_rerank(
                req.query.strip(),
                hits,
                license_data=license_data,
                llm_api_key=embedding_api_key,
                db=db,
            )
            mode = "semantic+rerank"
        except Exception as exc:
            logger.warning("retrieve/products rerank failed: %s", exc)

    return {
        "results": hits,
        "count": len(hits),
        "mode": mode,
    }


@router.post("/magento/chatbot/retrieve/content")
def retrieve_content(
    req: ContentRetrieveRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    query_vector = embed_query(req.query.strip(), embedding_api_key, license_data["client_id"])

    hits = qdrant_search_content(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        query_vector=query_vector,
        limit=req.limit,
        content_types=req.content_types or ["cms_page", "cms_block"],
    )
    return {"results": hits, "count": len(hits)}


@router.post("/magento/chatbot/retrieve/answer")
def retrieve_answer(
    req: AnswerRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Optional RAG summary. Admin-toggled on the Magento side; if the
    Magento module never calls this endpoint the LLM is never invoked."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip() or not req.sources:
        raise HTTPException(status_code=400, detail="query and sources are required")

    api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])

    from backend.app.magento.chatbot.agents.llm_factory import build_llm

    llm = build_llm(
        provider=req.llm_provider,
        model=req.llm_model,
        api_key=api_key,
        temperature=0.2,
    )

    sources_blob = "\n\n".join(
        f"[{s.get('content_type') or 'source'}] "
        f"{s.get('title') or s.get('name') or s.get('identifier') or ''}\n"
        f"{(s.get('summary') or s.get('content') or s.get('description') or '')[:800]}"
        for s in req.sources[:6]
    )

    prompt = (
        "You are a concise store assistant. Answer the user's question "
        "using ONLY the sources below. If the sources don't answer it, say so honestly.\n\n"
        f"User question: {req.query.strip()}\n\n"
        f"Sources:\n{sources_blob}\n\n"
        "Answer in 2-4 sentences. No markdown headings. No lists unless the user asked for one."
    )

    from langchain_core.messages import HumanMessage

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        logger.warning("retrieve/answer LLM invoke failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM unavailable")

    answer_text = _extract_text(resp.content).strip()
    usage = getattr(resp, "usage_metadata", None) or {}
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=license_data["client_id"],
            query_type="chat_answer",
            llm_provider=req.llm_provider or "google",
            llm_model=req.llm_model or "gemini-2.0-flash-lite",
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            request_text_length=len(prompt),
            response_text_length=len(answer_text),
        )
    except Exception:
        pass

    return {
        "answer": answer_text,
        "grounded": True,
        "usage": {
            "input": int(usage.get("input_tokens", 0) or 0),
            "output": int(usage.get("output_tokens", 0) or 0),
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_text(content: Any) -> str:
    """LangChain's AIMessage.content can be a plain string OR a list of content
    blocks (Gemini returns the list shape, e.g. [{'type': 'text', 'text': '...'}]).
    Normalize both to a flat string so downstream code can always .strip() etc."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # {'type': 'text', 'text': '...'} or {'text': '...'}
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _slug(value: str) -> str:
    import re as _re

    s = str(value or "").strip().lower()
    s = _re.sub(r"%", " percent", s)
    s = _re.sub(r"[^a-z0-9]+", "_", s)
    s = _re.sub(r"_+", "_", s)
    return s.strip("_")


def _lookup_by_skus(client_id: str, domain: str, skus: list[str]) -> list[dict]:
    """Direct SKU lookup via Qdrant scroll with filter — no embedding required."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    collection = get_collection_name(client_id, domain)
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        if collection not in existing:
            return []

        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="sku", match=MatchAny(any=skus))]),
            limit=max(10, len(skus) * 2),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("SKU lookup failed: %s", exc)
        return []

    return [dict(p.payload or {}) for p in points]


def _llm_rerank(
    query: str,
    hits: list[dict],
    *,
    license_data: dict,
    llm_api_key: Optional[str],
    db: Session,
) -> list[dict]:
    """Delegate to the existing llm_rerank_service already used by the /magento/search endpoint."""
    from backend.app.services.llm_rerank_service import llm_rerank_products

    return llm_rerank_products(
        query,
        hits,
        len(hits),
        llm_provider=None,
        llm_model=None,
        llm_api_key=llm_api_key,
        client_id=license_data["client_id"],
    ) or hits
