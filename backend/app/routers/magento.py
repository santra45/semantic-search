from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.app.services.cache_service import get_cached_embedding, get_cached_results, set_cached_embedding, set_cached_results, invalidate_client_results
from backend.app.services.database import get_db
from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.embedder import embed_query, embed_document
from backend.app.services.license_service import validate_license_key, check_search_quota, increment_search_count, log_search, increment_ingest_count, extract_license_key_from_authorization
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.llm_rerank_service import llm_rerank_products
from backend.app.services.mmr import apply_mmr, strip_vector
from backend.app.services.qdrant_service import search_products, upsert_product, delete_product, get_client_product_count
from backend.app.utils.slug import slug as _slug
# Magento has richer product structure (configurables, variants, super-attrs)
# than the generic WooCommerce format. Reuse the chatbot's Magento-aware
# formatter so sync produces a payload with attr_*/cat_* filter keys, variant
# rollups, and full category paths for downstream filtering and reranking.
from backend.app.magento.chatbot.services.product_formatter import format_product
import json
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()


class MagentoSearchRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    limit: int = 10
    # Legacy flag from the pre-structured-filter AISearch client. The
    # structured filters below now arrive pre-extracted from the Magento
    # vocab matchers, so server-side intent analysis is no longer used.
    # Kept for wire back-compat — ignored by the handler.
    enable_intent: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None

    # ── Structured filters (mirror of ProductRetrieveRequest) ─────────────────
    # Extracted Magento-side by the AISearch vocab matchers (attribute / brand /
    # category / currency) and applied as native Qdrant FieldConditions BEFORE
    # semantic ranking. Brand is merged into attribute_filters under the
    # merchant's brand attribute code on the Magento side, so it rides the same
    # attr_<code>_<slug> boolean key every synced product point already carries.
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    only_in_stock: bool = False
    attribute_filters: Dict[str, str] = Field(default_factory=dict)  # {"color": "red"}
    category_id: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None
    store_code: Optional[str] = None

    # ── Full-parity retrieval toggles (mirror of ProductRetrieveRequest) ──────
    rerank: bool = False
    rerank_limit: Optional[int] = None
    mmr: bool = False
    mmr_lambda: float = 0.5
    query_decomposition: bool = False
    filter_compat_mode: bool = False
    # Hybrid (BM25 + dense) is accepted for wire-compat but intentionally NEVER
    # honored — mirrors the AIChatbot on-hold state (sparse fusion degrades
    # e-commerce / brand queries; structured filters are the right mechanism).
    # The dense-only path always runs regardless of this flag.
    hybrid: bool = False

    @field_validator("sort_by", mode="before")
    @classmethod
    def _coerce_sort_by(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        return v if v in {"price", "name", "rating", "newest"} else None

    @field_validator("sort_order", mode="before")
    @classmethod
    def _coerce_sort_order(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        return v if v in {"asc", "desc"} else None

    @field_validator("mmr_lambda", mode="before")
    @classmethod
    def _coerce_mmr_lambda(cls, value):
        if value in (None, "", [], {}):
            return 0.5
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @field_validator("attribute_filters", mode="before")
    @classmethod
    def _coerce_attribute_filters(cls, value):
        """PHP json-encodes empty arrays as `[]`, which doesn't match a dict
        schema. Accept None / [] / dict / list-of-{name,value} and coerce."""
        if value in (None, "", [], {}):
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
        if isinstance(value, list):
            out: Dict[str, str] = {}
            for entry in value:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("key") or entry.get("code")
                    val = entry.get("value") or entry.get("option")
                    if name and val:
                        out[str(name)] = str(val)
            return out
        return {}

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_category_id(cls, value):
        """Empty strings and PHP-flavoured falsy values become None, not the
        literal string — a category_id="" would otherwise build a cat_ filter
        that matches nothing and strips every product."""
        if value in (None, "", [], {}, 0, "0"):
            return None
        return str(value)

    @field_validator("min_price", "max_price", mode="before")
    @classmethod
    def _coerce_optional_float(cls, value):
        if value in (None, "", [], {}):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class MagentoChild(BaseModel):
    """One configurable variant. attributes is {code: label}."""
    product_id: str = ""
    sku: str = ""
    name: str = ""
    price: float = 0
    regular_price: float = 0
    stock_status: str = "instock"
    attributes: dict = Field(default_factory=dict)


class MagentoProduct(BaseModel):
    """Single Magento product.

    All new fields are optional so older Magento module versions that send the
    legacy shape (string `categories`, no `children`, etc.) still validate.
    """
    product_id: str
    name: str
    sku: str = ""
    brand: str = ""
    gender: str = ""
    # Accepts legacy "Cat1 > Sub, Cat2" string OR new [{id,name,path},...] list.
    categories: Any = Field(default_factory=list)
    tags: Any = ""
    description: str = ""
    short_description: str = ""
    price: float = 0
    regular_price: float = 0
    sale_price: float = 0
    currency: str = ""
    currency_symbol: str = ""
    on_sale: bool = False
    permalink: str = ""
    image_url: str = ""
    stock_status: str = "instock"
    average_rating: float = 0
    attributes: list = Field(default_factory=list)
    # Configurable / variant data (empty for simple products)
    type_id: str = "simple"
    is_configurable: bool = False
    has_variants: bool = False
    children: List[MagentoChild] = Field(default_factory=list)
    variant_attributes: dict = Field(default_factory=dict)


class MagentoSyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    products: List[MagentoProduct]
    batch_number: int = 1
    total_batches: int = 1
    llm_api_key_encrypted: Optional[str] = None


class MagentoDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    product_id: str


def resolve_headers(
    authorization: Optional[str],
    x_api_key: Optional[str],
    x_llm_api_key_encrypted: Optional[str],
    request_license: Optional[str],
    request_llm_key: Optional[str],
):
    return {
        "license_key": extract_license_key_from_authorization(authorization) or request_license,
        "api_key": x_api_key,
        "llm_api_key_encrypted": x_llm_api_key_encrypted or request_llm_key,
    }


def _build_cache_key(query: str, req: "MagentoSearchRequest") -> str:
    """Results-cache key that folds the full structured-filter + toggle set in,
    so two searches with the same text but different filters / sort / parity
    options cache separately instead of colliding. cache_service.make_key
    SHA-256-hashes whatever we return, so an arbitrarily long JSON string is a
    fine key component."""
    parts = {
        "q": query,
        "limit": req.limit,
        "min": req.min_price,
        "max": req.max_price,
        "stock": req.only_in_stock,
        "attrs": sorted((req.attribute_filters or {}).items()),
        "cat": req.category_id,
        "sort": [req.sort_by, req.sort_order],
        "mmr": [req.mmr, req.mmr_lambda],
        "decomp": req.query_decomposition,
        "rerank": [req.rerank, req.rerank_limit],
        "store": req.store_code,
    }
    return json.dumps(parts, sort_keys=True, default=str)


def _apply_sort(hits: List[dict], sort_by: str, sort_order: str) -> List[dict]:
    """Apply a customer-requested sort to the (already-filtered) candidate pool.
    Items with no usable sort value drop to the end so they never beat properly
    valued items for the top slots. Unknown sort_by falls through unchanged.

    Isolated copy of retrieve.py's _apply_sort — kept here so the search and
    chatbot routers stay fully decoupled."""
    reverse = sort_order == "desc"

    if sort_by == "price":
        def key(h: dict):
            raw = h.get("price")
            try:
                v = float(raw)
                if v <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                return (1, 0.0)
            return (0, v)
        return sorted(hits, key=key, reverse=reverse)

    if sort_by == "name":
        def key_name(h: dict):
            name = str(h.get("name") or h.get("title") or "").strip().lower()
            return (0 if name else 1, name)
        return sorted(hits, key=key_name, reverse=reverse)

    if sort_by == "rating":
        def key_rating(h: dict):
            try:
                v = float(h.get("average_rating") or 0)
            except (TypeError, ValueError):
                v = 0.0
            return (0 if v > 0 else 1, v)
        return sorted(hits, key=key_rating, reverse=reverse)

    if sort_by == "newest":
        def key_newest(h: dict):
            ts = str(h.get("updated_at") or h.get("created_at") or "").strip()
            return (0 if ts else 1, ts)
        return sorted(hits, key=key_newest, reverse=not (sort_order == "asc"))

    return hits


@router.post("/magento/search")
async def magento_search(
    req: MagentoSearchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db)
):
    start_time = time.time()

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

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

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    if not check_search_quota(db, client_id, license_data["search_limit"]):
        raise HTTPException(status_code=429, detail="Monthly search limit reached. Please upgrade your plan.")

    # Structured filters (attribute_filters / category_id / price / sort / stock)
    # arrive pre-extracted from the Magento-side vocab matchers and are applied
    # below as native Qdrant FieldConditions BEFORE semantic ranking — the same
    # pipeline the chatbot's /retrieve/products runs, kept as an isolated copy
    # here so search and chatbot stay decoupled and AISearch keeps its own
    # quota / cache / search_logs.
    query_text = req.query.strip()
    cache_query = query_text.lower()
    cache_key = _build_cache_key(cache_query, req)

    embedding_api_key = None
    if headers["llm_api_key_encrypted"]:
        try:
            embedding_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
        except Exception:
            embedding_api_key = None

    cached_results = get_cached_results(f"{client_id}_{domain}", cache_key)
    if cached_results is not None:
        response_time = int((time.time() - start_time) * 1000)
        increment_search_count(db, client_id)
        log_search(db, client_id, query_text, len(cached_results), response_time, cached=True)
        return {
            "query": req.query,
            "count": len(cached_results),
            "cached": True,
            "results": cached_results,
            "mode": "cached",
        }

    # ── Query decomposition (optional, admin-toggled) ─────────────────────────
    # Splits a compositional query into 2-3 sub-queries, each embedded and fused
    # server-side via Qdrant RRF. Heuristic-gated inside maybe_decompose, so the
    # LLM cost is only paid for genuinely compound queries. Soft-fails to the
    # single-query path on any error.
    sub_queries: List[str] = [query_text]
    if req.query_decomposition:
        try:
            from backend.app.services.query_decomposer import maybe_decompose
            sub_queries = maybe_decompose(
                query_text,
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
                api_key=embedding_api_key,
                client_id=client_id,
            )
        except Exception as exc:
            logger.warning("magento/search decomposition failed: %s — single-vector path", exc)
            sub_queries = [query_text]
    decomposed = len(sub_queries) > 1

    # Embed. The primary vector keeps the existing case-normalised embedding
    # cache; decomposed sub-queries embed fresh (matches retrieve/products).
    query_vector = get_cached_embedding(cache_query)
    if query_vector is None:
        query_vector = embed_query(sub_queries[0], embedding_api_key, client_id)
        set_cached_embedding(cache_query, query_vector)
    query_vectors: Optional[List[List[float]]] = None
    if decomposed:
        query_vectors = [query_vector]
        for sq in sub_queries[1:]:
            query_vectors.append(embed_query(sq, embedding_api_key, client_id))

    # ── Fan-out sizing (mirror of retrieve/products) ──────────────────────────
    # Structured pre-filters need no oversample (every returned hit already
    # matches). Sort + MMR DO need a wider pool to work on.
    fan_out = 1
    if req.sort_by:
        fan_out = max(fan_out, 5)
    mmr_active = req.mmr and not req.sort_by
    if mmr_active:
        fan_out = max(fan_out, 5)
    raw_limit = max(req.limit, req.limit * fan_out)

    had_structured_filters = bool(req.attribute_filters or req.category_id)

    results = search_products(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=raw_limit,
        min_price=req.min_price,
        max_price=req.max_price,
        only_in_stock=req.only_in_stock,
        content_types=["product"],
        store_code=req.store_code,
        # Hybrid stays OFF — sparse fusion degrades e-commerce / brand queries;
        # the structured filters below are the correct mechanism (mirrors the
        # AIChatbot on-hold state).
        hybrid=False,
        sparse_query_vector=None,
        with_vectors=mmr_active,
        query_vectors=query_vectors,
        attribute_filters=req.attribute_filters or None,
        category_id=req.category_id,
    )

    # Compat fallback — when structured filters return nothing AND the admin
    # opted in, retry semantic-only then Python post-filter the broader pool.
    # Transition aid for tenants whose points predate a payload-key change; in
    # steady state this never fires.
    if not results and had_structured_filters and req.filter_compat_mode:
        logger.warning(
            "[magento/search] compat fallback — structured filter returned 0 hits; "
            "retrying without filters. attribute_filters=%s category_id=%s",
            req.attribute_filters, req.category_id,
        )
        results = search_products(
            client_id=client_id,
            domain=domain,
            query_vector=query_vector,
            limit=raw_limit,
            min_price=req.min_price,
            max_price=req.max_price,
            only_in_stock=req.only_in_stock,
            content_types=["product"],
            store_code=req.store_code,
            hybrid=False,
            sparse_query_vector=None,
            with_vectors=mmr_active,
            query_vectors=query_vectors,
        )
        if req.attribute_filters:
            for attr, value in req.attribute_filters.items():
                key = f"attr_{_slug(attr)}_{_slug(value)}"
                results = [h for h in results if h.get(key) is True]
        if req.category_id:
            key = f"cat_{req.category_id}"
            results = [h for h in results if h.get(key) is True]

    # ── Customer-requested sort (operates on the on-topic candidate pool) ─────
    if req.sort_by:
        results = _apply_sort(results, req.sort_by, req.sort_order or "asc")

    # ── MMR diversification — BEFORE rerank, skipped when sort_by is set ──────
    base_mode = "semantic+decomp" if decomposed else "semantic"
    mode = base_mode
    if mmr_active and len(results) > req.limit:
        try:
            results = apply_mmr(
                query_vector=query_vector,
                candidates=results,
                lambda_val=req.mmr_lambda,
                k=req.limit,
            )
            mode = base_mode + "+mmr"
        except Exception as exc:
            logger.warning("magento/search MMR failed: %s — vector order", exc)
            results = results[: req.limit]
    else:
        results = results[: req.limit]
    # Always drop the dense vectors search_products stamped on (with_vectors)
    # before they reach the wire — Magento has no consumer for 3072-dim arrays.
    strip_vector(results)

    # ── Optional LLM rerank (admin-toggled) ───────────────────────────────────
    if req.rerank and results:
        if req.rerank_limit and 0 < req.rerank_limit < len(results):
            results = results[: req.rerank_limit]
        try:
            reranked = llm_rerank_products(
                query_text,
                results,
                len(results),
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
                llm_api_key=embedding_api_key,
                client_id=client_id,
            )
            if reranked:
                logger.info(
                    "[magento/search rerank] client=%s candidates=%d returned=%d",
                    client_id, len(results), len(reranked),
                )
                results = reranked
                mode = mode + "+rerank"
        except Exception as exc:
            logger.warning("magento/search rerank failed: %s", exc)

    set_cached_results(f"{client_id}_{domain}", cache_key, results)

    response_time = int((time.time() - start_time) * 1000)
    increment_search_count(db, client_id)
    log_search(db, client_id, query_text, len(results), response_time, cached=False)

    return {
        "query": req.query,
        "count": len(results),
        "cached": False,
        "results": results,
        "mode": mode,
    }


@router.post("/magento/sync/batch")
def magento_sync_batch(
    req: MagentoSyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db)
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

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    current_count = get_client_product_count(client_id, domain)
    incoming_count = len(req.products)
    if current_count + incoming_count > license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Product limit exceeded. Current: {current_count}, Incoming: {incoming_count}, Limit: {license_data['product_limit']}",
        )

    embedding_api_key = None
    if headers["llm_api_key_encrypted"]:
        try:
            embedding_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
        except Exception:
            embedding_api_key = None

    success_ids = []
    failed_ids = []

    for product in req.products:
        try:
            p = product.model_dump()
            # format_product returns (embedding_text, qdrant_payload). The
            # payload includes attr_{code}_{value}=True, cat_{id}=True, and
            # rollups (variant_attributes, children, child_skus) the search
            # filters and the popup rely on.
            text, payload = format_product(p)
            vector = embed_document(text, embedding_api_key, client_id)
            payload["embedded_text"] = text
            upsert_product(client_id, domain, product.product_id, vector, payload)
            success_ids.append(product.product_id)
        except Exception:
            failed_ids.append(product.product_id)

    if success_ids:
        increment_ingest_count(db, client_id, count=len(success_ids))

    if req.batch_number >= req.total_batches:
        invalidate_client_results(client_id)

    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "batch_number": req.batch_number,
        "total_batches": req.total_batches,
        "is_last_batch": req.batch_number >= req.total_batches,
    }


@router.get("/magento/sync/quota")
def magento_sync_quota(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    headers = resolve_headers(authorization, x_api_key, None, None, None)
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    current_count = get_client_product_count(client_id, domain)
    product_limit = license_data["product_limit"]

    return {
        "current_count": current_count,
        "product_limit": product_limit,
        "remaining": product_limit - current_count,
        "exceeded": current_count > product_limit
    }


@router.post("/magento/sync/delete")
def magento_sync_delete(
    req: MagentoDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    headers = resolve_headers(authorization, x_api_key, None, req.license_key, None)
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    delete_product(license_data["client_id"], req.product_id)
    invalidate_client_results(license_data["client_id"])

    return {"deleted": True, "product_id": req.product_id}
