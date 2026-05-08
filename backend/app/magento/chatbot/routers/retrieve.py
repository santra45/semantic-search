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
from backend.app.utils.llm_logger import log_llm_call

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
    # When `rerank=True` the reranker uses these to pick the right provider/model
    # (otherwise falls back to the service's defaults, which may not match the
    # tenant's billing config and will lose cost tracking).
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

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
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
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
        _format_source_for_prompt(s) for s in req.sources[:6]
    )

    prompt = (
        "You are a concise store assistant. Answer the customer's question using ONLY the sources below.\n\n"
        "Rules:\n"
        " - If the sources don't contain the answer, say honestly in one sentence that you don't have that "
        "specific information and suggest the customer check the product page or contact support. Do not guess "
        "or invent details.\n"
        " - Keep the answer to 1-2 short sentences. For \"tell me about\" requests, you may use 2-3 sentences.\n"
        " - Put any concrete number, measurement, timeframe, or money value in **bold** (e.g. **30 days**, "
        "**$50**, **1.5 kg**).\n"
        " - Never invent SKUs, prices, dates, dimensions, or policy terms that aren't in the sources.\n"
        " - No markdown headings. No bulleted lists unless the customer explicitly asked for a list.\n"
        " - Plain prose only — write the way a helpful human store assistant would.\n\n"
        f"Customer question: {req.query.strip()}\n\n"
        f"Sources:\n{sources_blob}"
    )

    from langchain_core.messages import HumanMessage

    provider_name = (req.llm_provider or "google").lower()
    model_name    = req.llm_model or "gemini-2.0-flash-lite"

    with log_llm_call(
        provider=provider_name,
        model=model_name,
        purpose="chat_answer",
        prompt=prompt,
        client_id=license_data["client_id"],
    ) as _log_ctx:
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.warning("retrieve/answer LLM invoke failed: %s", exc)
            raise HTTPException(status_code=502, detail="LLM unavailable")

        answer_text = _extract_text(resp.content).strip()
        usage = getattr(resp, "usage_metadata", None) or {}
        input_tokens  = int(usage.get("input_tokens",  0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        # Cost is computed from the model pricing table — matches what the
        # /classify endpoint returns so the Magento per-message billing row
        # can sum cost across both call types without per-shape branches.
        # Split into input_cost / output_cost so the token_usage_tracking
        # row populates both component columns (the tracker computes
        # total_cost = input_cost + output_cost internally). The previous
        # version computed `cost` as a single value and never passed it to
        # the tracker — every chat_answer row ended up with zeroed cost
        # columns despite token counts being correct.
        from backend.app.services.llm_rerank_service import MODEL_PRICING
        pricing = MODEL_PRICING.get(model_name, {})
        input_cost  = input_tokens  * pricing.get("input",  0.0)
        output_cost = output_tokens * pricing.get("output", 0.0)
        cost = input_cost + output_cost

        _log_ctx.record(
            response_text=answer_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=float(cost),
            extra={"sources": len(req.sources or [])},
        )
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=license_data["client_id"],
            query_type="chat_answer",
            llm_provider=req.llm_provider or "google",
            llm_model=req.llm_model or "gemini-2.0-flash-lite",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=float(input_cost),
            output_cost=float(output_cost),
            request_text_length=len(prompt),
            response_text_length=len(answer_text),
        )
    except Exception:
        pass

    return {
        "answer": answer_text,
        "grounded": True,
        "usage": {
            "input":    input_tokens,
            "output":   output_tokens,
            "cost":     float(cost),
            "provider": provider_name,
            "model":    model_name,
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_source_for_prompt(s: dict) -> str:
    """Flatten one source into the text block the RAG summarizer sees.

    Per-content-type formatting because different shapes need different
    framing for the LLM:
      - product    → sku, variants, price, stock, attributes, description.
      - cms_page / cms_block → URL, heading, meta description, content body.
      - everything else → generic title + body fallback.
    """
    ct = (s.get("content_type") or "").lower()
    title = s.get("title") or s.get("name") or s.get("identifier") or s.get("sku") or ""

    if ct == "product" or s.get("sku") or s.get("type_id"):
        return _format_product_source(s, title)

    if ct in ("cms_page", "cms_block"):
        return _format_cms_source(s, ct, title)

    body = (s.get("summary") or s.get("content") or s.get("description") or "")[:800]
    return f"[{ct or 'source'}] {title}\n{body}"


def _format_cms_source(s: dict, ct: str, title: str) -> str:
    """Lay out a CMS page / block with all its useful framing context.

    The LLM benefits from seeing the page's display heading and URL — both
    let it write a more specific answer ("see our **Return Policy** page
    for full details"). meta_description is added explicitly so it doesn't
    get truncated when the body is long.

    Body cap is 4000 chars per source. With up to 6 sources passed to the
    summariser, the prompt body block can grow to ~24000 chars — roughly
    6000 tokens, comfortable inside any modern LLM's context window. If
    you find the LLM losing the early sources to "lost in the middle",
    drop this back to 2500-3000.
    """
    parts: list[str] = [f"[{ct}] {title}"]

    heading = (s.get("content_heading") or "").strip()
    if heading and heading.lower() != str(title).lower():
        parts.append(f"Heading: {heading}")

    permalink = (s.get("permalink") or "").strip()
    if permalink:
        parts.append(f"URL: {permalink}")

    meta_desc = (s.get("meta_description") or "").strip()
    if meta_desc:
        parts.append(f"Summary: {meta_desc}")

    keywords = (s.get("meta_keywords") or "").strip()
    if keywords:
        parts.append(f"Keywords: {keywords}")

    body = s.get("content") or s.get("summary") or ""
    if body:
        parts.append("")
        parts.append(str(body)[:4000])

    return "\n".join(parts)


def _format_product_source(s: dict, title: str) -> str:
    parts: list[str] = [f"[product] {title}"]
    sku = s.get("sku")
    if sku:
        parts.append(f"SKU: {sku}")
    type_id = s.get("type_id")
    if type_id:
        parts.append(f"Type: {type_id}")
    if s.get("stock_status"):
        parts.append(f"Stock: {s.get('stock_status')}")

    price = s.get("price")
    if price:
        currency = s.get("currency") or ""
        parts.append(f"Price: {price} {currency}".strip())

    categories = s.get("categories")
    if categories:
        parts.append(f"Categories: {categories}")

    brand = s.get("brand")
    if brand:
        parts.append(f"Brand: {brand}")

    # Attributes block — the LLM needs these to answer "what's the weight",
    # "what's it made of", "who makes it", and any custom-attribute question
    # the merchant exposed (battery life, dimensions, screen size etc.). The
    # values are stored as flat top-level keys on the payload (color="Red",
    # weight="1.5", etc.) by format_product. We iterate everything that
    # ISN'T a known structural field, isn't a filter-flag boolean, and is a
    # short scalar — that gives us the attribute set without a hard-coded
    # whitelist that would miss merchant-defined attributes.
    attribute_lines = _extract_attribute_lines(s)
    if attribute_lines:
        parts.append("Attributes:\n" + "\n".join(attribute_lines))

    # The critical bit: variant attributes + children so the LLM can
    # answer "what sizes does this come in".
    variant_attrs = s.get("variant_attributes") or {}
    if isinstance(variant_attrs, dict) and variant_attrs:
        lines = []
        for attr_code, values in variant_attrs.items():
            if not values:
                continue
            vals = values if isinstance(values, list) else [values]
            lines.append(f"  - {attr_code}: {', '.join(str(v) for v in vals)}")
        if lines:
            parts.append("Available variants:\n" + "\n".join(lines))

    children = s.get("children") or []
    if isinstance(children, list) and children:
        # Show up to 20 child SKUs with their attributes and stock.
        child_lines = []
        for ch in children[:20]:
            if not isinstance(ch, dict):
                continue
            attrs = ch.get("attributes") or {}
            attr_bits = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            stock = ch.get("stock_status") or ""
            price_c = ch.get("price") or ""
            child_lines.append(
                f"  - {ch.get('sku', '')}  {attr_bits}  {price_c}  {stock}".strip()
            )
        if child_lines:
            parts.append("Child SKUs:\n" + "\n".join(child_lines))

    desc = (s.get("description") or s.get("short_description") or s.get("summary") or "")
    if desc:
        parts.append(f"Description: {str(desc)[:1500]}")

    return "\n".join(parts)


# Structural fields on a product source — they're surfaced explicitly above,
# so we skip them when listing free-form attributes. Anything NOT in this set
# and not prefixed `attr_` / `cat_` is treated as a merchant attribute the
# LLM should see.
_KNOWN_PRODUCT_FIELDS = frozenset({
    "sku", "name", "title", "summary", "description", "short_description",
    "permalink", "image_url", "price", "currency", "currency_symbol",
    "regular_price", "sale_price", "on_sale", "average_rating",
    "categories", "category_paths", "category_ids", "tags",
    "stock_status", "type_id", "is_configurable", "has_variants",
    "variant_attributes", "children", "child_skus",
    "content_type", "entity_id", "client_id", "store_code", "embedded_text",
    "score", "product_id", "page_id", "post_id", "value", "label", "key",
    "identifier", "status", "meta_description", "updated_at",
    "brand", "gender",   # already surfaced explicitly
})


def _extract_attribute_lines(s: dict) -> list[str]:
    """Pull merchant attributes (weight, material, custom fields) out of the
    payload and render them as a bulleted list. Skips structural fields,
    filter booleans, and anything that isn't a short scalar so we don't dump
    JSON blobs or 2000-char descriptions into the prompt twice."""
    lines: list[str] = []
    for key, value in s.items():
        if key in _KNOWN_PRODUCT_FIELDS:
            continue
        if key.startswith("attr_") or key.startswith("cat_"):
            continue
        if value in (None, "", [], {}):
            continue
        # Only short scalars — long text is the description path.
        if isinstance(value, (int, float)):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text or len(text) > 200:
                continue
        else:
            continue
        label = key.replace("_", " ").strip().title()
        lines.append(f"  - {label}: {text}")
    return lines


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
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    db: Session,
) -> list[dict]:
    """Delegate to the existing llm_rerank_service already used by the /magento/search endpoint."""
    from backend.app.services.llm_rerank_service import llm_rerank_products

    return llm_rerank_products(
        query,
        hits,
        len(hits),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        client_id=license_data["client_id"],
    ) or hits
