"""Read endpoints for the WooCommerce product Q&A widget.

  POST /api/wordpress/productqa/retrieve/product   — the on-page product, by post ID
  POST /api/wordpress/productqa/retrieve/content   — merchant FAQ grounding
  POST /api/wordpress/productqa/retrieve/answer    — one grounded answer

All three are license + domain guarded. None of them see a customer identity:
the plugin keeps its transcripts in WordPress and sends only the question and
the product it's about.

Why a separate product endpoint from Magento's: Magento looks products up by
SKU, which is safe there because SKU is mandatory. WooCommerce makes SKU
optional and plenty of real stores leave it blank, so the only identifier
guaranteed to exist is the post ID. Looking up by post ID also removes an
entire failure mode — two products sharing a SKU is a data problem in
WooCommerce, not an error.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
    retrieve_content_by_entity_ids,
    search_content as qdrant_search_content,
)
from backend.app.services.token_usage_service import TokenUsageTracker
from backend.app.utils.llm_logger import log_llm_call
from backend.app.utils.stage_timer import StageTimer

from backend.app.wordpress.productqa.services.common import authorize_request, decrypt_llm_key
from backend.app.wordpress.productqa.services.prompts import build_answer_prompt, scrub_pii

logger = logging.getLogger(__name__)

router = APIRouter()

# WordPress has no store views. The field is carried through anyway because
# point ids and payload filters are shaped around it across every platform —
# dropping it here would make WordPress points the odd ones out for no gain.
DEFAULT_STORE_CODE = "default"


# ── Schemas ──────────────────────────────────────────────────────────────────


class ProductLookupRequest(BaseModel):
    license_key: Optional[str] = None
    # WordPress post ID. Accepts a single id or a list — the widget only ever
    # asks for one, but the list form costs nothing and makes the endpoint
    # usable for a future "compare" surface.
    product_id: Optional[str] = None
    product_ids: list[str] = Field(default_factory=list)
    store_code: str = DEFAULT_STORE_CODE

    @field_validator("product_id", mode="before")
    @classmethod
    def _coerce_id(cls, value):
        # WordPress will happily send an int; Qdrant payloads store entity_id
        # as a string, so normalise before the filter is built.
        if value in (None, "", [], {}):
            return None
        return str(value).strip()

    @field_validator("product_ids", mode="before")
    @classmethod
    def _coerce_ids(cls, value):
        if value in (None, "", [], {}):
            return []
        if isinstance(value, (str, int)):
            return [str(value).strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def resolved_ids(self) -> list[str]:
        ids = list(self.product_ids)
        if self.product_id and self.product_id not in ids:
            ids.insert(0, self.product_id)
        return ids


class ContentRetrieveRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    content_types: list[str] = Field(default_factory=lambda: ["faq"])
    limit: int = 3
    store_code: str = DEFAULT_STORE_CODE
    hybrid: bool = False

    @field_validator("content_types", mode="before")
    @classmethod
    def _coerce_types(cls, value):
        if value in (None, "", [], {}):
            return ["faq"]
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v) for v in value if v not in (None, "")]
        return ["faq"]


class AnswerRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    sources: list[dict] = Field(default_factory=list)
    instruction: Optional[str] = None
    contact: Optional[dict] = None
    store_code: str = DEFAULT_STORE_CODE
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    @field_validator("contact", mode="before")
    @classmethod
    def _coerce_contact(cls, value):
        return value if isinstance(value, dict) else None


# ── Product lookup ───────────────────────────────────────────────────────────


@router.post("/wordpress/productqa/retrieve/product")
def retrieve_product(
    req: ProductLookupRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Fetch indexed products by WordPress post ID.

    A filter-scroll, not a vector search: no embedding call, no similarity
    floor, no LLM. The widget already knows exactly which product the shopper
    is looking at, so ranking it against anything would only introduce a way
    to return the wrong one.

    Returns an empty list rather than a 404 when the product isn't indexed —
    "not synced yet" is a normal state right after install, and the plugin
    turns it into a friendly "can't pull details right now" rather than an
    error.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    ids = req.resolved_ids()
    if not ids:
        raise HTTPException(status_code=400, detail="product_id is required")

    hits = retrieve_content_by_entity_ids(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        entity_ids=ids,
        content_types=["product"],
        store_code=req.store_code or DEFAULT_STORE_CODE,
        limit=len(ids),
    )

    return {"results": hits, "count": len(hits), "mode": "product_id"}


# ── FAQ / content retrieval ──────────────────────────────────────────────────


@router.post("/wordpress/productqa/retrieve/content")
def retrieve_content(
    req: ContentRetrieveRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Semantic search over the merchant's FAQ entries.

    Unlike the product lookup this DOES embed — the shopper's question has to
    be matched against FAQ answers by meaning, since nobody asks "what is your
    returns policy" using the words the merchant used to write it.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    timer = StageTimer("wp/retrieve/content", request)

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    query_vector = embed_query(req.query.strip(), embedding_api_key, license_data["client_id"])

    # BM25 alongside the dense vector when the merchant has hybrid on. Policy
    # questions lean on exact terms ("warranty", "restocking fee") that
    # semantic search tends to rank below near-synonyms. Soft-fails to
    # dense-only rather than 500ing on a cold fastembed cache.
    sparse_query_vector = None
    if req.hybrid:
        try:
            from backend.app.services.sparse_embedder import embed_sparse_query
            sparse_query_vector = embed_sparse_query(req.query.strip())
        except Exception as exc:
            logger.warning("wp retrieve/content sparse-embed failed: %s — dense-only", exc)
    timer.mark("embed")

    hits = qdrant_search_content(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        query_vector=query_vector,
        limit=req.limit,
        content_types=req.content_types or ["faq"],
        store_code=req.store_code or DEFAULT_STORE_CODE,
        hybrid=req.hybrid,
        sparse_query_vector=sparse_query_vector,
    )
    timer.mark("qdrant")
    timer.flush()

    return {"results": hits[: req.limit], "count": len(hits[: req.limit])}


# ── Grounded answer ──────────────────────────────────────────────────────────


@router.post("/wordpress/productqa/retrieve/answer")
def retrieve_answer(
    req: AnswerRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Answer one product question from the sources the caller supplies.

    Single-shot: one prompt, one completion, no tool loop. A product page
    widget is a spinner the shopper is watching, and the plugin has already
    handed over everything relevant — a retrieval loop would buy nothing and
    cost seconds.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip() or not req.sources:
        raise HTTPException(status_code=400, detail="query and sources are required")

    api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    client_id = license_data["client_id"]

    from backend.app.magento.chatbot.agents.llm_factory import build_llm, resolve_provider_model

    # Low temperature: this answers questions of fact about a product the
    # merchant is legally responsible for describing accurately.
    llm = build_llm(
        provider=req.llm_provider,
        model=req.llm_model,
        api_key=api_key,
        temperature=0.2,
    )

    prompt = build_answer_prompt(
        query=req.query.strip(),
        sources=req.sources,
        instruction=req.instruction,
        contact=req.contact,
    )

    # Ask the factory what it built rather than re-deriving it. Most merchants
    # leave the model on "use the service default", and a locally-guessed
    # fallback here would file all of their usage under a model that was never
    # called — priced at zero, since that guess need not be in the pricing
    # table.
    provider_name, model_name = resolve_provider_model(req.llm_provider, req.llm_model)

    input_tokens = 0
    output_tokens = 0
    input_cost = 0.0
    output_cost = 0.0
    answer = ""

    with log_llm_call(
        provider=provider_name,
        model=model_name,
        purpose="wp_product_qa",
        prompt=prompt,
        client_id=client_id,
    ) as log_ctx:
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.warning("wp retrieve/answer LLM invoke failed: %s", exc)
            raise HTTPException(status_code=502, detail="LLM unavailable")

        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        answer = _extract_text(response.content).strip()

        from backend.app.services.llm_rerank_service import price_usage
        input_cost, output_cost = price_usage(model_name, input_tokens, output_tokens)

        log_ctx.record(
            response_text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=float(input_cost + output_cost),
            extra={"sources": len(req.sources or [])},
        )

    # Usage accounting must never be the reason an answer fails to reach the
    # shopper — the LLM call is already paid for by this point. Swallowed, but
    # not silently: a bare `except: pass` here is what hid `wp_product_qa`
    # being rejected as an unregistered query_type, and every answer went
    # unrecorded for as long as nobody thought to check.
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=client_id,
            query_type="wp_product_qa",
            llm_provider=provider_name,
            llm_model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=float(input_cost),
            output_cost=float(output_cost),
            request_text_length=len(prompt),
            response_text_length=len(answer),
        )
    except Exception as exc:
        logger.warning(
            "wp retrieve/answer usage not recorded for client %s (%s/%s): %s",
            client_id, provider_name, model_name, exc,
        )

    return {
        "answer": scrub_pii(answer),
        "grounded": True,
        "usage": {
            "input": input_tokens,
            "output": output_tokens,
            "cost": float(input_cost + output_cost),
            "provider": provider_name,
            "model": model_name,
        },
    }


def _extract_text(content) -> str:
    """LangChain returns either a plain string or a list of content blocks
    depending on the provider. Flatten both to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text") or ""))
        return "".join(chunks)
    return str(content or "")
