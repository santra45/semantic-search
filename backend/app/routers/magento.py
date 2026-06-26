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

    # ── Multi-brand-attribute OR (mirror of ProductRetrieveRequest) ───────────
    # SINGLE configured brand attribute → Magento still folds brand into
    # attribute_filters[<brand_code>] (the historical path, unchanged, brand
    # stays None here). MULTIPLE configured brand attributes (admin enters them
    # comma-separated) → Magento sends the brand VALUE here + the list of codes,
    # and the value matches whichever brand attribute holds it (OR group built
    # by qdrant_service._build_content_filter — already shared, no re-sync).
    brand: Optional[str] = None
    brand_attribute_codes: List[str] = Field(default_factory=list)

    # ── Pagination ("show more") (mirror of ProductRetrieveRequest) ───────────
    # page_size > 0 opts into paginated mode (AISearch sends it = the display
    # result_limit); offset is the page start. Page 1 (offset 0) is byte-
    # identical to the pre-pagination first slice. Other callers leave
    # page_size = 0 and keep the legacy single-window behaviour.
    offset: int = 0
    page_size: int = 0

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

    @field_validator("offset", "page_size", mode="before")
    @classmethod
    def _coerce_non_negative_int(cls, value):
        """PHP may send '' / null for an absent pagination field; coerce to 0."""
        if value in (None, "", [], {}):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @field_validator("brand_attribute_codes", mode="before")
    @classmethod
    def _coerce_brand_codes(cls, value):
        """Accept None / '' / a single string / a list; return a clean str list."""
        if value in (None, "", [], {}):
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []


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
        "brand": req.brand,
        "brand_codes": sorted(req.brand_attribute_codes or []),
        "sort": [req.sort_by, req.sort_order],
        "mmr": [req.mmr, req.mmr_lambda],
        "decomp": req.query_decomposition,
        "rerank": [req.rerank, req.rerank_limit],
        "store": req.store_code,
        # Pagination: each page caches separately so "show more" (offset>0)
        # never returns page 1's window.
        "offset": req.offset,
        "page_size": req.page_size,
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

    cached = get_cached_results(f"{client_id}_{domain}", cache_key)
    if cached is not None:
        # Back-compat: entries written before pagination stored a bare list
        # (results only). Newer entries store a dict carrying the page's
        # pagination meta so a cache-hit "show more" still knows has_more.
        if isinstance(cached, list):
            cached = {"results": cached}
        cached_results = cached.get("results", [])
        response_time = int((time.time() - start_time) * 1000)
        increment_search_count(db, client_id)
        log_search(db, client_id, query_text, len(cached_results), response_time, cached=True)
        return {
            "query": req.query,
            "count": len(cached_results),
            "cached": True,
            "results": cached_results,
            "mode": "cached",
            "has_more": bool(cached.get("has_more", False)),
            "next_offset": int(cached.get("next_offset", 0)),
            "pool_total": int(cached.get("pool_total", len(cached_results))),
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
    # Paginated ("show more") → fetch a deep pool to page over, even when
    # MMR / sort wouldn't otherwise oversample.
    paginated = req.page_size > 0
    if paginated:
        fan_out = max(fan_out, 5)
    raw_limit = max(req.limit, req.limit * fan_out)

    # Normalise the brand-OR inputs: only active when BOTH a value and at least
    # one attribute code are present (the multi-brand-attribute path). A single
    # configured brand code rides attribute_filters[<code>] Magento-side, so
    # brand stays None here.
    brand_value = (req.brand or "").strip() or None
    brand_codes = [c.strip() for c in (req.brand_attribute_codes or []) if c and c.strip()]
    if not brand_value or not brand_codes:
        brand_value, brand_codes = None, []

    had_structured_filters = bool(req.attribute_filters or req.category_id or brand_value)

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
        # Multi-brand-attribute OR (2026-06-03): match the brand value against
        # ANY of these attributes' boolean keys. None for the single-code path.
        brand=brand_value,
        brand_attribute_codes=brand_codes or None,
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
        # Dual-read both payload shapes: the new indexed lists
        # (`attribute_facets` / `category_ids`) on reshaped points AND the
        # legacy `attr_<code>_<value>` / `cat_<id>` booleans on points synced
        # before the reshape — so this legacy route's post-filter works across a
        # mid-rollout collection and after a full re-sync.
        if req.attribute_filters:
            for attr, value in req.attribute_filters.items():
                token = f"{_slug(attr)}:{_slug(value)}"
                legacy_key = f"attr_{_slug(attr)}_{_slug(value)}"
                results = [
                    h for h in results
                    if token in (h.get("attribute_facets") or [])
                    or h.get(legacy_key) is True
                ]
        if req.category_id:
            cid = str(req.category_id)
            legacy_key = f"cat_{req.category_id}"
            results = [
                h for h in results
                if cid in [str(c) for c in (h.get("category_ids") or [])]
                or h.get(legacy_key) is True
            ]

    # ── Customer-requested sort (operates on the on-topic candidate pool) ─────
    if req.sort_by:
        results = _apply_sort(results, req.sort_by, req.sort_order or "asc")

    # ── Order the candidate pool ──────────────────────────────────────────────
    # MMR runs BEFORE rerank and is skipped when sort_by is set. Paginated mode
    # MMR-orders the FULL pool (k=len) so later "show more" pages window over a
    # stable, fully-ranked list. Greedy MMR's first picks are identical to
    # k=req.limit, so page 1 (offset=0) is unchanged. Non-paginated callers keep
    # the historical k=req.limit slice.
    base_mode = "semantic+decomp" if decomposed else "semantic"
    mode = base_mode
    if mmr_active and len(results) > (1 if paginated else req.limit):
        try:
            results = apply_mmr(
                query_vector=query_vector,
                candidates=results,
                lambda_val=req.mmr_lambda,
                k=len(results) if paginated else req.limit,
            )
            mode = base_mode + "+mmr"
        except Exception as exc:
            logger.warning("magento/search MMR failed: %s — vector order", exc)
            if not paginated:
                results = results[: req.limit]
    elif not paginated:
        results = results[: req.limit]
    # Always drop the dense vectors search_products stamped on (with_vectors)
    # before they reach the wire — Magento has no consumer for 3072-dim arrays.
    strip_vector(results)

    # ── Pagination window ("show more products") ──────────────────────────────
    pool_total = len(results)
    has_more = False
    next_offset = 0
    if paginated:
        offset = max(0, req.offset)
        # Rerank ON: window a rerank_limit-sized slice (Magento shows page_size
        # of it); stride = rerank_limit. Rerank OFF: the window IS the page;
        # stride = page_size. Advancing by the rerank window (not the display
        # page) means no overlap between pages.
        stride = req.rerank_limit if (req.rerank and req.rerank_limit and req.rerank_limit > 0) else req.page_size
        stride = max(1, stride)
        next_offset = offset + stride
        has_more = next_offset < pool_total
        # Relevance-floor gate — stop "show more" once the next window has no
        # genuinely-relevant items left, so we don't paginate into noise.
        # Skipped for explicit sorts (the customer asked for price/etc. order).
        # Uses the original cosine score MMR preserved (rerank hasn't reshaped
        # these yet).
        if has_more and not req.sort_by:
            _pagination_floor = 0.40
            nxt = results[next_offset : next_offset + stride]
            best = max((float(h.get("score") or 0) for h in nxt), default=0.0)
            if best < _pagination_floor:
                has_more = False
        results = results[offset : offset + stride]

    # ── Optional LLM rerank (admin-toggled) ───────────────────────────────────
    if req.rerank and results:
        # Trim the rerank pool (non-paginated only — paginated already windowed
        # to the rerank stride above).
        if (not paginated) and req.rerank_limit and 0 < req.rerank_limit < len(results):
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

    # Cache the page as a dict so a cache-hit "show more" recovers has_more /
    # next_offset (older bare-list entries are still read — see the hit branch).
    set_cached_results(f"{client_id}_{domain}", cache_key, {
        "results": results,
        "has_more": has_more,
        "next_offset": next_offset,
        "pool_total": pool_total,
    })

    response_time = int((time.time() - start_time) * 1000)
    increment_search_count(db, client_id)
    log_search(db, client_id, query_text, len(results), response_time, cached=False)

    return {
        "query": req.query,
        "count": len(results),
        "cached": False,
        "results": results,
        "mode": mode,
        "has_more": has_more,
        "next_offset": next_offset,
        "pool_total": pool_total,
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
