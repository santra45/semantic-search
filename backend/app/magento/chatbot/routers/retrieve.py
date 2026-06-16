"""
Pure retrieval endpoints. The backend is a Qdrant proxy now — zero chat logic.

  POST /api/magento/chatbot/retrieve/products    — semantic / structured product search
  POST /api/magento/chatbot/retrieve/content     — CMS pages / blocks / widgets / store config
  POST /api/magento/chatbot/retrieve/answer      — optional RAG summary over provided sources

All three are license + domain guarded. They never see chat history and never
touch any identifiers beyond client_id (already anonymous).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
    get_collection_name,
    qdrant,
    retrieve_content_by_entity_ids as qdrant_retrieve_content_by_entity_ids,
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
    # Brand routing. SINGLE configured brand attribute → Magento still folds
    # brand into `attribute_filters[<brand_code>]` (the historical path,
    # unchanged). MULTIPLE configured brand attributes (e.g. "brand" +
    # "pump_brand", admin enters them comma-separated) → Magento sends the
    # brand VALUE here plus the list of brand attribute codes, and
    # _build_content_filter ORs across the per-attribute boolean keys
    # (attr_<code>_<slug(value)>) so the value matches whichever attribute
    # actually holds it. No re-sync — those attr_* keys are already on the
    # synced points for every attribute. Empty string / list coerced to None.
    brand: Optional[str] = None
    brand_attribute_codes: list[str] = Field(default_factory=list)
    # Note: brand was briefly a top-level field in the structured-filter
    # rebuild (2026-05-22 morning) but the SINGLE-code path is routed through
    # `attribute_filters` — Magento merges entities['brand'] into
    # attribute_filters[<brand_attribute_code>] before sending. This means
    # the Qdrant filter matches the SAME `attr_<code>_<slug>` boolean key
    # that the sync pipeline has always been writing, which works on
    # existing product points without re-sync. See plan.md "Structured
    # filter rebuild" entry for full rationale.
    # Compat fallback for the post-filter → pre-filter rebuild (2026-05-22+).
    # When True AND the structured filters (attribute / category / brand)
    # produce zero hits, the handler retries WITHOUT those filters and
    # applies the legacy Python post-filter on the broader semantic pool.
    # Lets tenants who haven't re-synced after the rebuild keep getting
    # results until their points carry the new payload shape. Admin
    # toggles via aichatbot/llm/filter_compat_mode. Default False — once
    # the tenant has fully re-synced, this should stay off.
    filter_compat_mode: bool = False
    limit: int = 8
    rerank: bool = False  # admin-toggled — small LLM rerank of top-N
    # Cap on how many candidates flow into the LLM reranker after Qdrant
    # search + MMR. Rerank cost scales linearly with this number, so it's
    # the main lever once rerank is on. When None / 0 the handler reranks
    # everything that survives MMR (legacy behaviour). The internal
    # llm_rerank_service still applies a 25-item safety ceiling on top of
    # this (prompt-size protection) — see llm_rerank_service.py:410.
    # Magento threads this through from aichatbot/llm/rerank_limit.
    rerank_limit: Optional[int] = None
    # When `rerank=True` the reranker uses these to pick the right provider/model
    # (otherwise falls back to the service's defaults, which may not match the
    # tenant's billing config and will lose cost tracking).
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # Phase 2.2 — admin-toggled hybrid search. When True the handler
    # generates a BM25 sparse vector alongside the dense query vector
    # and asks Qdrant to fuse the two via RRF. When False the existing
    # dense-only path runs unchanged. Magento threads this through from
    # the `aichatbot/llm/hybrid_search_enabled` config flag.
    hybrid: bool = False
    # Phase 2.3 — admin-toggled MMR (Maximal Marginal Relevance) over
    # the post-retrieval candidate pool, BEFORE the LLM reranker (2.1).
    # Prevents N near-duplicate results (same product in 6 colour
    # variants, or 5 chunks of the same long policy) from dominating
    # the top-K and starving the reranker of variety. Disabled by
    # default and skipped automatically when sort_by is set (customer
    # explicitly wants a specific order — don't second-guess them).
    mmr: bool = False
    # MMR lambda — relevance/diversity tradeoff. 1.0 = pure relevance
    # (equivalent to MMR off), 0.0 = pure diversity. 0.5 (default) is
    # balanced. Admin can tune via `aichatbot/llm/mmr_lambda`. Values
    # outside [0,1] are clamped server-side rather than 422'd, so a
    # config typo never breaks a customer query.
    mmr_lambda: float = 0.5
    # Phase 3.3 — admin-toggled query decomposition. When True the
    # handler runs the query through query_decomposer.maybe_decompose
    # before embedding; if it returns 2-3 sub-queries each is embedded
    # and the resulting candidate sets are fused server-side by Qdrant
    # via RRF. Gated by an internal heuristic too — simple short
    # queries skip the LLM call regardless of this flag.
    query_decomposition: bool = False
    # Customer's current store view. When set, retrieval is scoped to
    # points tagged with this store_code (set by the Magento side at
    # sync time from store->getCode()). Absent / empty means "no store
    # filter" — used by legacy single-store deployments that pre-date
    # per-store sync.
    store_code: Optional[str] = None
    # Sort intent extracted by the PHP classifier from phrases like
    # "cheapest", "most expensive", "lowest priced". The Qdrant search
    # itself returns by semantic similarity (vector cosine); we fetch a
    # larger relevance-ordered pool, then apply this sort in Python over
    # the on-topic candidates, then slice back to `limit`. So semantic
    # relevance still gates which products appear, but the order honours
    # the customer's stated preference.
    #
    # sort_by   — currently "price" only. Future-proofed for "name",
    #             "rating", "newest" etc. Unknown values fall back to
    #             vector relevance (no resort) instead of erroring.
    # sort_order — "asc" | "desc". Defaults to "asc" when sort_by is set
    #              and order is omitted.
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None

    @field_validator("sort_by", mode="before")
    @classmethod
    def _coerce_sort_by(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        # Whitelist known sort keys; silently drop unknowns rather than
        # 422-ing so a forward-incompatible classifier output never
        # breaks the retrieval call.
        return v if v in {"price", "name", "rating", "newest"} else None

    @field_validator("mmr_lambda", mode="before")
    @classmethod
    def _coerce_mmr_lambda(cls, value):
        # Clamp into [0, 1]. Bad values (non-numeric, out of range)
        # collapse to the default 0.5 instead of 422-ing — same posture
        # as sort_by above. A merchant's typo in admin config field
        # should never DOS the chatbot.
        if value in (None, "", [], {}):
            return 0.5
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @field_validator("sort_order", mode="before")
    @classmethod
    def _coerce_sort_order(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        return v if v in {"asc", "desc"} else None

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

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_optional_string(cls, value):
        """Empty strings and PHP-flavoured falsy values should be None, not
        passed through as the literal string. Without this, a Magento payload
        with category_id="" would build a FieldCondition for `cat_` and filter
        away every product on the planet."""
        if value in (None, "", [], {}, 0, "0"):
            return None
        return value

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
    # See ProductRetrieveRequest.store_code — same semantics.
    store_code: Optional[str] = None
    # Phase 2.2 — same admin toggle as products. Particularly valuable
    # for CMS / store-config queries where customers often use exact
    # keywords ("warranty", "tax registration") that semantic search
    # ranks below near-synonyms.
    hybrid: bool = False
    # Phase 2.3 — same MMR knobs as ProductRetrieveRequest. Especially
    # useful for CMS content where chunking (1.3) can put 2-3 paragraphs
    # of the same long page in the top-K; MMR pushes a chunk from a
    # different page up to break the near-duplicate streak.
    mmr: bool = False
    mmr_lambda: float = 0.5
    # Phase 3.3 — same admin toggle as ProductRetrieveRequest. Useful
    # for CMS / policy queries that compound multiple concepts ("shipping
    # cost to canada with express delivery", "returns policy for
    # international orders") — the dense single-vector path tends to
    # match the dominant concept and miss the others.
    query_decomposition: bool = False

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


class ContentByIdsRequest(BaseModel):
    """Vector-less retrieval by entity_id list.

    Powers ProductSearchAgent's category-context call: the agent tallies
    the displayed products' `category_ids`, picks every category tied for
    the top frequency, and asks here for those specific entity_ids. This
    replaces the prior `/retrieve/content` call on the raw customer
    query, which used to surface categories unrelated to the products
    being shown.
    """

    license_key: Optional[str] = None
    entity_ids: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=lambda: ["category"])
    store_code: Optional[str] = None
    limit: int = 5

    @field_validator("entity_ids", "content_types", mode="before")
    @classmethod
    def _coerce_str_list(cls, value):
        if value in (None, "", [], {}):
            return []
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []


class AnswerRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    active_retrieval: bool = False
    store_code: Optional[str] = None
    # Phase 3.2 — when active retrieval invokes its retrieve_more_*
    # tools, they should honor the merchant's INITIAL retrieval
    # quality settings. `hybrid` here is the same toggle as
    # ProductRetrieveRequest.hybrid (admin: aichatbot/llm/hybrid_search_enabled).
    # MMR is intentionally NOT propagated — active-retrieval queries
    # are already narrow refinements and diversification would push
    # results toward unrelated content (see retrieval_tools.py).
    hybrid: bool = False
    # Optional store contact details forwarded by the Magento agents
    # (PolicyAgent / StoreInfoAgent / GenericChatAgent). When set, the
    # prompt instructs the LLM to surface phone/email in any refusal so
    # the customer always has a path forward. Pulled live from Magento's
    # general/store_information/* config — universal across clients.
    #
    # Shape: {"phone": "+44 ...", "email": "support@..."}
    contact: dict[str, str] = Field(default_factory=dict)
    # Optional last-few-turns of conversation history for multi-turn
    # follow-up queries ("what's the price of next-day delivery" after a
    # product question about delivery options). Each item:
    #   {"role": "user"|"assistant", "content": "..."}
    # Capped at 6 turns by the prompt builder — older context is ignored.
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # Purpose switches the prompt template:
    #
    #   "answer"   (default) — strict grounded Q&A. Refuses when evidence
    #                          doesn't support a claim. Used by PolicyAgent,
    #                          StoreInfoAgent, GenericChatAgent for policy /
    #                          info questions where confident-but-wrong
    #                          answers are the failure mode to avoid.
    #
    #   "preamble"           — confirmation framing. Used by
    #                          ProductSearchAgent's NL answer-line. Vector
    #                          search has ALREADY filtered to relevant
    #                          products; the LLM's job is to acknowledge
    #                          the customer's request in one sentence
    #                          using their phrasing, NOT to second-guess
    #                          whether the evidence supports the claim.
    #                          Skipping the strict refusal rule is the
    #                          whole point.
    #
    #   "comparison"         — comparative framing. Used by GenericChatAgent's
    #                          "X vs Y" branch. Asks the LLM to contrast
    #                          both sides using product evidence per
    #                          operand.
    purpose: Optional[str] = "answer"
    # Free-form extra instruction the caller wants surfaced in the
    # prompt. ProductSearchAgent uses this to pass framing hints
    # ("results are sorted cheapest first", "the customer described
    # their situation as advisory"). Distinct from `query` so the LLM
    # doesn't confuse the instruction with the customer's question.
    instruction: Optional[str] = None

    @field_validator("contact", mode="before")
    @classmethod
    def _coerce_contact(cls, value):
        if value in (None, "", [], {}):
            return {}
        if isinstance(value, dict):
            out: dict[str, str] = {}
            for key in ("phone", "email"):
                v = value.get(key)
                if v is not None and str(v).strip() != "":
                    out[key] = str(v).strip()
            return out
        return {}

    @field_validator("conversation_history", mode="before")
    @classmethod
    def _coerce_history(cls, value):
        if value in (None, "", [], {}):
            return []
        if not isinstance(value, list):
            return []
        out: list[dict[str, str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            content = str(entry.get("content") or "").strip()
            if role in ("user", "assistant") and content != "":
                out.append({"role": role, "content": content})
        return out[-6:]


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
        hits = _lookup_by_skus(client_id, domain, req.skus, store_code=req.store_code)
        return {
            "results": hits,
            "count": len(hits),
            "mode": "sku",
        }

    # ── Semantic search path ─────────────────────────────────────────────────
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query or skus is required")

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])

    # Phase 3.3 — query decomposition. Runs BEFORE embedding so each
    # sub-query gets its own dense vector + its own RRF prefetch slot.
    # The decomposer's heuristic gate skips simple queries even when
    # the admin toggle is on, so the LLM cost is only paid for
    # genuinely compositional questions. Soft-fall to [original_query]
    # on any error — same posture as hybrid / mmr graceful degradation.
    sub_queries: list[str] = [req.query.strip()]
    if req.query_decomposition:
        try:
            from backend.app.services.query_decomposer import maybe_decompose
            sub_queries = maybe_decompose(
                req.query.strip(),
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
                api_key=embedding_api_key,
                client_id=client_id,
            )
        except Exception as exc:
            logger.warning("retrieve/products decomposition failed: %s — single-vector path", exc)
            sub_queries = [req.query.strip()]
    decomposed = len(sub_queries) > 1

    # Embed each sub-query. Single-query case is exactly one embed call
    # — same cost as pre-3.3. Multi-sub-query case pays N embed calls
    # (~$0.0001 each for Gemini, sub-millisecond latency each).
    query_vector = embed_query(sub_queries[0], embedding_api_key, client_id)
    query_vectors: Optional[list[list[float]]] = None
    if decomposed:
        query_vectors = [query_vector]
        for sq in sub_queries[1:]:
            query_vectors.append(embed_query(sq, embedding_api_key, client_id))

    # Phase 2.2 — when the admin has hybrid on, also generate a BM25
    # sparse query vector. Soft-fail on import / inference errors so a
    # malformed model cache or missing fastembed install degrades to
    # dense-only instead of 500ing the request.
    #
    # Sparse vector is built from the ORIGINAL combined query even when
    # decomposition splits the dense side — BM25 keyword signal is more
    # useful when grounded in what the customer actually typed than in
    # per-concept fragments that lose cross-concept term co-occurrence.
    sparse_query_vector = None
    if req.hybrid:
        try:
            from backend.app.services.sparse_embedder import embed_sparse_query
            sparse_query_vector = embed_sparse_query(req.query.strip())
        except Exception as exc:
            logger.warning("retrieve/products hybrid sparse-embed failed: %s — dense-only", exc)

    # Fetch over-broad so the post-filter has room to narrow down. When a
    # sort intent is present we widen even further — we want enough
    # on-topic candidates that the price-sorted slice doesn't just show
    # the same N closest-by-vector items. Universal: every catalog with
    # > limit relevant items benefits from the wider pool.
    #
    # Phase 2.3: MMR also benefits from oversampling — diversity is
    # meaningless if there are only `limit` candidates to choose from.
    # Bump fan_out to ≥5 when MMR is on (and not pre-empted by sort_by;
    # MMR is skipped in that branch — see the apply step below).
    #
    # Structured filter rebuild (2026-05-22+): the historical fan_out=3
    # for attribute_filters / category_id was compensation for the
    # post-Qdrant Python filter dropping a fraction of hits. With those
    # filters now pushed into the Qdrant query as native FieldConditions,
    # every returned hit already matches — no oversample needed for
    # filter selectivity. Sort + MMR fan_outs still apply because they
    # need extra candidate variety to do their job.
    fan_out = 1
    if req.sort_by:
        fan_out = max(fan_out, 5)
    mmr_active = req.mmr and not req.sort_by
    if mmr_active:
        fan_out = max(fan_out, 5)
    raw_limit = max(req.limit, req.limit * fan_out)

    # Normalise the brand-OR inputs: only active when BOTH a value and at
    # least one attribute code are present (the multi-brand-attribute path).
    brand_value = (req.brand or "").strip() or None
    brand_codes = [c.strip() for c in (req.brand_attribute_codes or []) if c and c.strip()]
    if not brand_value or not brand_codes:
        brand_value, brand_codes = None, []

    had_structured_filters = bool(
        req.attribute_filters or req.category_id or brand_value
    )

    hits = qdrant_search_products(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=raw_limit,
        min_price=req.min_price,
        max_price=req.max_price,
        only_in_stock=req.only_in_stock,
        content_types=req.content_types or ["product"],
        store_code=req.store_code,
        hybrid=req.hybrid,
        sparse_query_vector=sparse_query_vector,
        # Only ask qdrant to return vectors when MMR will actually use
        # them — saves wire bytes on every dense-only-without-MMR call.
        with_vectors=mmr_active,
        # Phase 3.3 — non-None only when decomposition produced 2+
        # sub-queries. qdrant_service collapses 1-element lists back
        # to the single-vector path internally.
        query_vectors=query_vectors,
        # Structured filter rebuild (2026-05-22+) — attribute_filters
        # (which now carries brand under the merchant's configured brand
        # attribute code, merged Magento-side in ProductSearchAgent) and
        # category_id land as native Qdrant FieldConditions composed by
        # _build_content_filter alongside price + stock + store_code, so
        # semantic ranking computes over the already-filtered candidate
        # set instead of post-fetch Python filtering after the fact.
        attribute_filters=req.attribute_filters or None,
        category_id=req.category_id,
        # Multi-brand-attribute OR (2026-06-03): match the brand value
        # against ANY of these attributes' boolean keys.
        brand=brand_value,
        brand_attribute_codes=brand_codes or None,
    )

    # Compat fallback (Phase: structured filter rebuild — 2026-05-22+).
    # When the pre-filter Qdrant query returns nothing AND the request
    # carried structured filters AND the admin has opted into compat
    # mode, re-run WITHOUT those filters and apply the legacy Python
    # post-filter on the broader semantic pool.
    #
    # The post-filter checks payload booleans the sync used to write;
    # for tenants who synced before the rebuild the booleans were set
    # then too, so this fallback still finds matches. For tenants whose
    # data is fully up-to-date this branch never fires (the pre-filter
    # found their results on the first try).
    #
    # A WARNING-level log line surfaces every compat-fallback hit so
    # operators can spot tenants that need a re-sync. The flag is
    # documented as a transition aid only — plan to remove after
    # telemetry shows ~zero usage.
    if not hits and had_structured_filters and req.filter_compat_mode:
        logger.warning(
            "[retrieve/products] compat fallback engaged — structured filter "
            "returned 0 hits; retrying without filters + post-filter. "
            "Tenant likely needs a re-sync. attribute_filters=%s category_id=%s",
            req.attribute_filters, req.category_id,
        )
        hits = qdrant_search_products(
            client_id=client_id,
            domain=domain,
            query_vector=query_vector,
            limit=raw_limit,
            min_price=req.min_price,
            max_price=req.max_price,
            only_in_stock=req.only_in_stock,
            content_types=req.content_types or ["product"],
            store_code=req.store_code,
            hybrid=req.hybrid,
            sparse_query_vector=sparse_query_vector,
            with_vectors=mmr_active,
            query_vectors=query_vectors,
            # Structured filters dropped — falling back to semantic-only.
        )
        # Legacy Python post-filter — kept here only on the compat path.
        # Equivalent to the pre-rebuild logic but applied to a wider
        # raw_limit pool. Brand has already been merged into
        # attribute_filters[brand_code] Magento-side, so the same loop
        # handles it. Returns empty when the merchant's products genuinely
        # don't carry the boolean flags either (in which case the tenant
        # must re-sync — no software fallback fixes that).
        if req.attribute_filters:
            for attr, value in req.attribute_filters.items():
                key = f"attr_{_slug(attr)}_{_slug(value)}"
                hits = [h for h in hits if h.get(key) is True]
        if req.category_id:
            key = f"cat_{req.category_id}"
            hits = [h for h in hits if h.get(key) is True]

    # ── Apply customer-requested sort on the on-topic candidate pool ─────
    # Runs AFTER attribute/category narrowing so the sort operates on
    # items that already match what the customer asked for. Vector
    # relevance gated which products made it this far; the sort decides
    # the final ordering. Stable sort preserves relevance order within
    # equal sort keys, so a tie on price (rare in practice) keeps the
    # most-relevant item on top.
    if req.sort_by:
        hits = _apply_sort(hits, req.sort_by, req.sort_order or "asc")

    # ── Phase 2.3: MMR diversification BEFORE the LLM reranker ──────────
    # Skipped when sort_by is set (the customer asked for a specific
    # ordering — don't second-guess them) and when there's not enough
    # candidate headroom for MMR to do anything meaningful (≤ limit
    # items in the pool; MMR can't diversify what's already the full
    # result set).
    # Phase 3.3 — mode string surfaces "+decomp" when decomposition
    # actually produced multiple sub-queries (not just when the toggle
    # is on — the heuristic gate may have skipped the LLM call). Lets
    # ops grep response.mode in api.log to count decomposed turns
    # without parsing structured fields.
    base_mode = "semantic+decomp" if decomposed else "semantic"
    mode = base_mode
    from backend.app.services.mmr import apply_mmr, strip_vector
    if mmr_active and len(hits) > req.limit:
        try:
            hits = apply_mmr(
                query_vector=query_vector,
                candidates=hits,
                lambda_val=req.mmr_lambda,
                k=req.limit,
            )
            mode = base_mode + "+mmr"
        except Exception as exc:
            # MMR failures fall back to vector-similarity order. Never
            # let a numpy hiccup deprive the customer of results.
            logger.warning("retrieve/products MMR failed: %s — falling back to vector order", exc)
            hits = hits[: req.limit]
    else:
        hits = hits[: req.limit]
    # Always strip dense vectors before returning. Earlier the strip
    # only ran inside the MMR branch — when the candidate pool was
    # smaller than `limit` (very common for narrow catalogues) the
    # else-branch leaked 3072-float arrays per hit to Magento, which
    # then forwarded them to /retrieve/answer/stream. ~50KB per hit
    # for no consumer benefit.
    strip_vector(hits)

    if req.rerank and hits:
        # Trim the rerank pool. By this point `hits` has been shrunk to
        # req.limit by MMR (or the explicit slice). When the admin set
        # rerank_limit < req.limit they want the LLM to only score the
        # top-N most relevant candidates rather than the full display
        # pool — the cost lever lives here, BEFORE the LLM call. Skipped
        # cleanly when rerank_limit is unset / 0 / already >= len(hits).
        if req.rerank_limit and 0 < req.rerank_limit < len(hits):
            logger.debug(
                "retrieve/products rerank pool trimmed from %d to %d (rerank_limit)",
                len(hits), req.rerank_limit,
            )
            hits = hits[: req.rerank_limit]
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
            # Always append — mode is built from earlier stages
            # (semantic, +decomp, +mmr) and "+rerank" is the terminal
            # suffix. The old "mode != 'semantic'" guard pre-dated
            # the +decomp prefix and now misclassifies.
            mode = mode + "+rerank"
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

    # Phase 3.3 — query decomposition for CMS / policy queries. Same
    # heuristic gate as retrieve_products; especially useful for
    # compound policy questions ("shipping cost to canada with express",
    # "returns policy for international orders") that mix multiple
    # semantic concepts.
    sub_queries: list[str] = [req.query.strip()]
    if req.query_decomposition:
        try:
            from backend.app.services.query_decomposer import maybe_decompose
            # ContentRetrieveRequest deliberately doesn't carry
            # llm_provider/llm_model — the decomposer falls back to
            # the backend default (gemini-2.5-flash via env GEMINI_API_KEY)
            # if these are None. Threading them would require extending
            # the schema + ApiClient for a marginal benefit.
            sub_queries = maybe_decompose(
                req.query.strip(),
                llm_provider=None,
                llm_model=None,
                api_key=embedding_api_key,
                client_id=license_data["client_id"],
            )
        except Exception as exc:
            logger.warning("retrieve/content decomposition failed: %s — single-vector path", exc)
            sub_queries = [req.query.strip()]
    decomposed = len(sub_queries) > 1

    query_vector = embed_query(sub_queries[0], embedding_api_key, license_data["client_id"])
    query_vectors: Optional[list[list[float]]] = None
    if decomposed:
        query_vectors = [query_vector]
        for sq in sub_queries[1:]:
            query_vectors.append(embed_query(sq, embedding_api_key, license_data["client_id"]))

    # Phase 2.2 — sparse query vector for hybrid mode. Same soft-fail
    # pattern as retrieve_products so a CMS query never gets bricked by
    # a BM25 cold-start glitch. Built from the ORIGINAL combined query
    # even when decomposition is active (see retrieve_products for the
    # cross-concept BM25 rationale).
    sparse_query_vector = None
    if req.hybrid:
        try:
            from backend.app.services.sparse_embedder import embed_sparse_query
            sparse_query_vector = embed_sparse_query(req.query.strip())
        except Exception as exc:
            logger.warning("retrieve/content hybrid sparse-embed failed: %s — dense-only", exc)

    # Phase 2.3 — MMR over-sampling. CMS retrieval doesn't have the
    # fan_out logic the products handler uses, so we apply the
    # equivalent inline: when MMR will run, oversample 5× so it has a
    # real pool to diversify across. Otherwise stick with req.limit.
    if req.mmr:
        fetch_limit = max(req.limit * 5, 30)
    else:
        fetch_limit = req.limit

    hits = qdrant_search_content(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        query_vector=query_vector,
        limit=fetch_limit,
        content_types=req.content_types or ["cms_page", "cms_block"],
        store_code=req.store_code,
        hybrid=req.hybrid,
        sparse_query_vector=sparse_query_vector,
        with_vectors=req.mmr,
        query_vectors=query_vectors,
    )

    # Apply MMR (when on) then strip the internal `_dense_vector` field
    # whether MMR ran or not — search_content stamped it on every hit
    # if with_vectors was True, and Magento agents don't need 3072-dim
    # arrays riding through the JSON response.
    if req.mmr and len(hits) > req.limit:
        from backend.app.services.mmr import apply_mmr, strip_vector
        try:
            hits = apply_mmr(
                query_vector=query_vector,
                candidates=hits,
                lambda_val=req.mmr_lambda,
                k=req.limit,
            )
        except Exception as exc:
            logger.warning("retrieve/content MMR failed: %s — falling back to vector order", exc)
            hits = hits[: req.limit]
        finally:
            strip_vector(hits)
    elif req.mmr:
        # Pool ≤ limit — nothing for MMR to do, but still need to drop
        # the vectors we asked qdrant for.
        from backend.app.services.mmr import strip_vector
        hits = hits[: req.limit]
        strip_vector(hits)
    else:
        hits = hits[: req.limit]

    return {"results": hits, "count": len(hits)}


@router.post("/magento/chatbot/retrieve/content_by_ids")
def retrieve_content_by_ids(
    req: ContentByIdsRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Scroll-based retrieval keyed on entity_id — no embedding, no vector search.

    Used by ProductSearchAgent to fetch the categories the retrieved
    products actually belong to (top entries from a frequency tally of
    product.category_ids), avoiding the "categories don't match products"
    mismatch that the prior parallel /retrieve/content call on the
    customer query produced.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.entity_ids:
        return {"results": [], "count": 0}

    hits = qdrant_retrieve_content_by_entity_ids(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        entity_ids=req.entity_ids,
        content_types=req.content_types or ["category"],
        store_code=req.store_code,
        limit=req.limit,
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

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    # Phase 3.2 — active-retrieval tools are built per-request via the
    # factory in retrieval_tools.py so the LLM-facing signatures stay
    # minimal (`query`, `limit`) while the tool bodies have closure
    # access to client_id / domain / api_key / store_code / hybrid.
    # Empty tuple when active_retrieval is off so we don't pay the
    # closure-build cost or pass tools to the LLM that it won't use.
    from backend.app.magento.chatbot.agents.retrieval_tools import (
        make_retrieval_tools,
        MAX_ACTIVE_RETRIEVAL_ITERATIONS,
    )
    if req.active_retrieval:
        tools, tool_map = make_retrieval_tools(
            client_id=client_id,
            domain=domain,
            api_key=api_key,
            store_code=req.store_code,
            hybrid=req.hybrid,
            source_formatter=_format_source_for_prompt,
        )
    else:
        tools, tool_map = [], {}

    prompt = _build_answer_prompt(
        query=req.query.strip(),
        sources=req.sources,
        contact=req.contact,
        conversation_history=req.conversation_history,
        purpose=req.purpose or "answer",
        instruction=req.instruction,
        active_retrieval=req.active_retrieval,
    )

    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    provider_name = (req.llm_provider or "google").lower()
    model_name    = req.llm_model or "gemini-2.0-flash-lite"

    messages = [HumanMessage(content=prompt)]
    input_tokens = 0
    output_tokens = 0
    iterations = 0
    final_answer = ""
    last_resp = None

    llm_to_use = llm.bind_tools(tools) if req.active_retrieval and tools else llm

    with log_llm_call(
        provider=provider_name,
        model=model_name,
        purpose="chat_answer",
        prompt=prompt,
        client_id=client_id,
    ) as _log_ctx:
        while iterations < MAX_ACTIVE_RETRIEVAL_ITERATIONS:
            try:
                resp = llm_to_use.invoke(messages)
            except Exception as exc:
                logger.warning("retrieve/answer LLM invoke failed: %s", exc)
                if iterations == 0:
                    raise HTTPException(status_code=502, detail="LLM unavailable")
                break

            last_resp = resp
            usage = getattr(resp, "usage_metadata", None) or {}
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)

            if req.active_retrieval and hasattr(resp, "tool_calls") and resp.tool_calls:
                messages.append(resp)
                for tool_call in resp.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    t_id = tool_call["id"]

                    if t_name in tool_map:
                        try:
                            tool_result = tool_map[t_name].invoke(t_args)
                        except Exception as err:
                            tool_result = f"Error executing tool: {err}"
                    else:
                        tool_result = f"Error: Tool '{t_name}' not found."

                    messages.append(ToolMessage(content=tool_result, tool_call_id=t_id))
                iterations += 1
            else:
                final_answer = _extract_text(resp.content).strip()
                break
        else:
            if last_resp and hasattr(last_resp, "tool_calls") and last_resp.tool_calls:
                try:
                    resp = llm.invoke(messages)
                    usage = getattr(resp, "usage_metadata", None) or {}
                    input_tokens += int(usage.get("input_tokens", 0) or 0)
                    output_tokens += int(usage.get("output_tokens", 0) or 0)
                    final_answer = _extract_text(resp.content).strip()
                except Exception as exc:
                    logger.warning("retrieve/answer final forced LLM invoke failed: %s", exc)
                    final_answer = "I'm sorry, I was unable to complete the search due to an internal error."
            else:
                final_answer = _extract_text(last_resp.content).strip() if last_resp else ""

        # Cost calculation
        from backend.app.services.llm_rerank_service import MODEL_PRICING
        pricing = MODEL_PRICING.get(model_name, {})
        input_cost  = input_tokens  * pricing.get("input",  0.0)
        output_cost = output_tokens * pricing.get("output", 0.0)
        cost = input_cost + output_cost

        _log_ctx.record(
            response_text=final_answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=float(cost),
            extra={"sources": len(req.sources or []), "active_retrieval_iterations": iterations},
        )
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=client_id,
            query_type="chat_answer",
            llm_provider=req.llm_provider or "google",
            llm_model=req.llm_model or "gemini-2.0-flash-lite",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=float(input_cost),
            output_cost=float(output_cost),
            request_text_length=len(prompt),
            response_text_length=len(final_answer),
        )
    except Exception:
        pass

    return {
        "answer": final_answer,
        "grounded": True,
        "usage": {
            "input":    input_tokens,
            "output":   output_tokens,
            "cost":     float(cost),
            "provider": provider_name,
            "model":    model_name,
        },
    }


@router.post("/magento/chatbot/retrieve/answer/stream")
def retrieve_answer_stream(
    req: AnswerRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Streaming variant of /retrieve/answer.

    Returns NDJSON (newline-delimited JSON), one event per line:
        {"event": "token", "text": "..."}    — incremental chunk
        {"event": "done",  "answer": "...", "usage": {...}}    — terminal

    NDJSON over chunked transfer encoding rather than SSE because:
      - PHP can read it as plain chunks via cURL WRITEFUNCTION (the Magento
        proxy in Controller/Ajax/Stream.php) without needing an SSE parser.
      - Frontend can use fetch + ReadableStream.getReader without
        EventSource — fetch is what we already use for everything else.
      - Restart-friendly: each line is a complete JSON object so a partial
        chunk just means "wait for more bytes".

    Behaviour parity with /retrieve/answer: same auth, same prompt, same
    cost computation, same TokenUsageTracker write at the end. The only
    difference is the response shape.
    """
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

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    # Phase 3.2 — same per-request tool factory as the one-shot
    # /retrieve/answer handler. Building once outside the generator
    # means the tools' closures + the (optional) sparse-embedder
    # import happen on the request thread, not inside the streaming
    # generator (where exceptions are harder to surface cleanly).
    from backend.app.magento.chatbot.agents.retrieval_tools import (
        make_retrieval_tools,
        MAX_ACTIVE_RETRIEVAL_ITERATIONS,
    )
    if req.active_retrieval:
        tools, tool_map = make_retrieval_tools(
            client_id=client_id,
            domain=domain,
            api_key=api_key,
            store_code=req.store_code,
            hybrid=req.hybrid,
            source_formatter=_format_source_for_prompt,
        )
    else:
        tools, tool_map = [], {}

    # Same prompt as the one-shot endpoint — keeps answer style consistent
    # whether the merchant has streaming on or off.
    prompt = _build_answer_prompt(
        query=req.query.strip(),
        sources=req.sources,
        contact=req.contact,
        conversation_history=req.conversation_history,
        purpose=req.purpose or "answer",
        instruction=req.instruction,
        active_retrieval=req.active_retrieval,
    )

    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    provider_name = (req.llm_provider or "google").lower()
    model_name    = req.llm_model or "gemini-2.0-flash-lite"

    def event_stream():
        # Accumulate as we go so we can compute total cost + write the
        # tracker row once at the end. The langchain stream() yields
        # AIMessageChunk objects; the FINAL chunk usually carries
        # usage_metadata, but we max-merge across chunks to be safe.
        full_answer: list[str] = []
        in_tokens  = 0
        out_tokens = 0

        messages = [HumanMessage(content=prompt)]

        if req.active_retrieval and tools:
            llm_to_use = llm.bind_tools(tools)
            iterations = 0
            last_resp = None

            # Tool-call loop runs NON-streaming — each round is a
            # structured tool_calls response, not natural-language
            # tokens worth streaming. Once the LLM stops calling tools
            # (or we hit the iteration cap), the loop exits and the
            # final answer is streamed via llm.stream(messages) below.
            while iterations < MAX_ACTIVE_RETRIEVAL_ITERATIONS:
                try:
                    resp = llm_to_use.invoke(messages)
                except Exception as exc:
                    logger.warning("retrieve/answer/stream tool loop LLM invoke failed: %s", exc)
                    yield json.dumps({"event": "error", "message": "LLM unavailable"}) + "\n"
                    return

                last_resp = resp
                usage = getattr(resp, "usage_metadata", None) or {}
                in_tokens += int(usage.get("input_tokens", 0) or 0)
                out_tokens += int(usage.get("output_tokens", 0) or 0)

                if hasattr(resp, "tool_calls") and resp.tool_calls:
                    messages.append(resp)
                    for tool_call in resp.tool_calls:
                        t_name = tool_call["name"]
                        t_args = tool_call["args"]
                        t_id = tool_call["id"]

                        if t_name in tool_map:
                            try:
                                tool_result = tool_map[t_name].invoke(t_args)
                            except Exception as err:
                                tool_result = f"Error executing tool: {err}"
                        else:
                            tool_result = f"Error: Tool '{t_name}' not found."

                        messages.append(ToolMessage(content=tool_result, tool_call_id=t_id))
                    iterations += 1
                else:
                    break

        # Now stream the final response turn chunk by chunk
        stream_in_tokens = 0
        stream_out_tokens = 0
        try:
            for chunk in llm.stream(messages):
                token_text = _extract_text(getattr(chunk, "content", "")) or ""
                if token_text:
                    full_answer.append(token_text)
                    yield json.dumps({"event": "token", "text": token_text}) + "\n"
                meta = getattr(chunk, "usage_metadata", None) or {}
                if meta:
                    stream_in_tokens = max(stream_in_tokens, int(meta.get("input_tokens", 0) or 0))
                    stream_out_tokens = max(stream_out_tokens, int(meta.get("output_tokens", 0) or 0))
        except Exception as exc:
            logger.warning("retrieve/answer/stream LLM stream failed: %s", exc)
            yield json.dumps({"event": "error", "message": "LLM unavailable"}) + "\n"
            return

        in_tokens += stream_in_tokens
        out_tokens += stream_out_tokens
        answer_text = "".join(full_answer).strip()

        # Cost calculation — split into input/output components so the
        # tracker row matches what /retrieve/answer writes (same fix from
        # the cost-tracking bug we patched a few turns ago).
        from backend.app.services.llm_rerank_service import MODEL_PRICING
        pricing = MODEL_PRICING.get(model_name, {})
        input_cost  = in_tokens  * pricing.get("input",  0.0)
        output_cost = out_tokens * pricing.get("output", 0.0)
        cost = input_cost + output_cost

        try:
            TokenUsageTracker(db).create_usage_record(
                client_id=client_id,
                query_type="chat_answer",
                llm_provider=provider_name,
                llm_model=model_name,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                input_cost=float(input_cost),
                output_cost=float(output_cost),
                request_text_length=len(prompt),
                response_text_length=len(answer_text),
            )
        except Exception:
            pass

        yield json.dumps({
            "event":  "done",
            "answer": answer_text,
            "usage": {
                "input":    in_tokens,
                "output":   out_tokens,
                "cost":     float(cost),
                "provider": provider_name,
                "model":    model_name,
            },
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_answer_prompt(
    query: str,
    sources: list[dict],
    contact: Optional[dict] = None,
    conversation_history: Optional[list[dict]] = None,
    purpose: str = "answer",
    instruction: Optional[str] = None,
    active_retrieval: bool = False,
) -> str:
    """Single source of truth for the /retrieve/answer prompt — shared
    between the one-shot and streaming endpoints so style and rules stay
    in sync.

    Three modes, switched by `purpose`:

      "answer" (default) — strict grounded Q&A. Refuses when the
        evidence doesn't support a claim. Right behaviour for policy /
        info questions where confident-but-wrong is the failure mode.

      "preamble" — confirmation framing for ProductSearchAgent's
        natural-language answer line. Vector search has ALREADY filtered
        to relevant products; the prompt's job is to acknowledge the
        customer's request in one sentence, NOT to second-guess whether
        the evidence supports the claim. The strict refusal rule is
        DROPPED entirely — if the LLM hesitates here, it produces "I
        don't have specific information about indoor water features"
        while looking at four indoor water features, which is exactly
        the failure the senior reported.

      "comparison" — used by GenericChatAgent's comparative branch.
        Contrasts both sides explicitly.

    Across all modes:
      - **Conversation context** (last 6 turns) for pronoun resolution.
      - **Contact details on refusal** in "answer" mode only — preamble
        doesn't refuse.
      - **Direction-of-flow rule** for damage scenarios.
    """
    # Source formatting — most purposes use the full per-content-type
    # formatters. For purpose=category_info we ask for COMPACT product
    # sources so the supporting product samples don't drown out the
    # category description; the LLM only needs name + SKU + price +
    # short description to mention 1-2 illustratively, not the full
    # 1500-char-per-product dump.
    _compact_products = purpose in ("category_info", "product_qa")
    sources_blob = "\n\n".join(
        _format_source_for_prompt(s, compact_products=_compact_products)
        for s in (sources or [])[:6]
    )

    history_block = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:
            role = (turn.get("role") or "").strip().lower()
            text = (turn.get("content") or "").strip()
            if not text or role not in ("user", "assistant"):
                continue
            history_lines.append(
                f"{'CUSTOMER' if role == 'user' else 'ASSISTANT'}: {text}"
            )
        if history_lines:
            history_block = (
                "\n\nConversation so far (most recent last) — use to resolve "
                "pronouns and follow-ups, but don't quote it verbatim:\n"
                + "\n".join(history_lines)
            )

    instruction_block = ""
    if instruction:
        instruction_block = f"\n\nAdditional framing instruction: {instruction.strip()}"
        
    # FAQ-link rule (2026-06-09). Surfaced ONLY when a faq source is actually
    # present, so neither the rule nor its tokens touch non-FAQ answers. Tells
    # the model to turn a URL the merchant put inside a FAQ answer into a
    # clickable markdown link — the widget's renderMarkdown only linkifies
    # [text](url) syntax, so a bare URL would otherwise render as plain,
    # unclickable text. Scoped to [faq] sources so links never leak from other
    # content types (cms_page links etc. stay as they are today).
    faq_link_rule = ""
    if any((s.get("content_type") or "").lower() == "faq" for s in (sources or [])):
        faq_link_rule = (
            " - **Links in FAQ answers.** When you use a fact from a `[faq]` source "
            "that contains a URL (a tracking page, returns form, guide, etc.), "
            "surface that link to the customer as a clickable **markdown link** with "
            "short descriptive text — e.g. `[track your order](https://example.com/"
            "track)` — never a bare URL and never a raw HTML `<a>` tag. Only do this "
            "for URLs that actually appear in a `[faq]` source; never invent a link "
            "and never surface URLs from non-FAQ sources.\n"
        )

    if purpose == "preamble":
        # CONFIRMATION mode — used by ProductSearchAgent's NL answer-line.
        # Drops the strict refusal rule. The cards render below the
        # message regardless, so the prompt's job is a warm, informative
        # acknowledgment that engages with the category description and
        # the variety on offer — not a flat one-liner. The earlier
        # 1-2-sentence cap left rich category + product evidence
        # unused; the customer asked, we have the data, we should use it.
        return (
            "You are writing the lead-in for a product search result page. "
            "Vector search has ALREADY confirmed these products match the customer's "
            "request — your job is to acknowledge that match warmly and informatively. "
            "Do NOT second-guess whether the evidence supports the match.\n\n"
            "Tone & length:\n"
            " - 2 to 4 sentences. Substantive but not verbose.\n"
            " - Match the customer's own phrasing for the topic (\"garden planters\" "
            "not \"plant containers\"; \"indoor water features\" not \"interior "
            "fountains\").\n"
            " - For yes/no questions, start with \"Yes\" or \"Yes — \".\n"
            " - For \"recommend\" / \"suggest\" / \"help me choose\" questions, "
            "frame as advice (\"For a small garden, these compact options would "
            "suit you well:\").\n"
            " - Plain prose only — no markdown headings, no bullets, no numbered "
            "lists.\n\n"
            "What to include (drawing from the evidence below):\n"
            " - When a matching CATEGORY collection is present in the evidence, "
            "name it and say a sentence about what it covers (use its description "
            "as a guide — paraphrase, don't quote verbatim).\n"
            " - When the matched products show variety, describe that variety as "
            "a SET: price range (\"from £20 up to around £150\"), common "
            "materials / styles / finishes, or notable options (e.g. \"including "
            "both solar-lit and mains-powered choices\"). Pull these details "
            "from the matched products listed below.\n"
            " - End with a brief invitation to look at the options below.\n\n"
            "What to avoid:\n"
            " - Do NOT list individual product names or SKUs — cards render below "
            "this sentence and the customer will see every product visually.\n"
            " - Do NOT refuse or claim insufficient information. The match is real.\n"
            " - Do NOT invent details not present in the evidence (no fake prices, "
            "no fake brand names).\n"
            " - Do NOT call attention to your sources (\"according to the data\", "
            "\"based on what I see\") — speak as the store, not as an assistant.\n"
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nMatched content (categories first, then products — describe as "
            f"a SET, never list individual product names):\n{sources_blob}"
        )

    if purpose == "category_info":
        # CATEGORY OVERVIEW mode (structured filter rebuild 2026-05-22+).
        # Used by CategoryInfoAgent when the customer asks ABOUT a
        # category itself rather than asking to see its products.
        #
        # Sources arrive in one of two shapes, dispatched by the agent
        # based on the matched category's children_count:
        #
        # (A) PARENT shape — the matched category is an umbrella that
        #     contains sub-categories. Sources blob:
        #       [0] = the parent category itself (description, meta, breadcrumb)
        #       [1..N] = 3-5 direct child categories, each a category
        #                source with name + summary + permalink
        #     The customer wants ORIENTATION — what's IN this section?
        #     so the LLM describes the parent AND names 2-3 sub-collections
        #     to give a sense of structure.
        #
        # (B) LEAF shape — the matched category has no children, IT IS
        #     the product collection. Sources blob:
        #       [0] = the leaf category itself
        #       [1..N] = 3-5 representative products with full descriptions
        #     The LLM describes the leaf AND names 1-2 products
        #     illustratively to anchor the overview in concrete examples.
        #
        # Both shapes share the same rules below; the LLM distinguishes
        # by content_type on each supporting source ([category] vs
        # [product]).
        return (
            "You are describing a product category to a customer. The first "
            "source below is the category itself (its merchant-authored "
            "description, name, breadcrumb, meta description). The remaining "
            "sources are either SUB-CATEGORIES (if the matched category is "
            "a navigation umbrella with children) or REPRESENTATIVE PRODUCTS "
            "(if the matched category is a leaf containing actual products). "
            "Look at the content_type tag on each supporting source to tell "
            "them apart — `[category]` = sub-category, `[product]` = a "
            "product example.\n\n"
            "Goal: write a clear 2-4 sentence overview of what the category "
            "covers, what's inside it, and (when you can tell) who it would "
            "suit. Use the merchant's description as your source of truth — "
            "paraphrase it naturally, don't quote verbatim.\n\n"
            "How to use the supporting sources:\n"
            " - When SUB-CATEGORIES are present (parent-shape response): "
            "name 2-3 of them in your overview to give the customer a sense "
            "of structure. Use the merchant's category names exactly. "
            "Example: \"Our **Solar Water Features** collection covers "
            "outdoor pieces that run on solar power. It includes "
            "sub-collections for **Solar Fountains**, **Solar Bird Baths**, "
            "and **Solar Pump Systems**, ranging from compact standalone "
            "pieces to larger installations.\"\n"
            " - When PRODUCT examples are present (leaf-shape response): "
            "name 1-2 illustratively to anchor the overview. Use the "
            "product's actual name. Example: \"Our **Solar Fountains** "
            "include pieces like the **Tranquillity Sphere** and "
            "**Mini Cascade Bowl**, both self-contained solar units that "
            "don't need mains plumbing.\"\n"
            " - Do NOT enumerate every sub-category or every product, and "
            "do NOT list SKUs — the supporting items render as cards "
            "beneath this text and the customer will see them visually.\n\n"
            "Rules:\n"
            " - Plain prose. No markdown headings. No bulleted or numbered "
            "lists unless the customer explicitly asked.\n"
            " - Bold concrete category names, sub-collection names, product "
            "names, sizes, prices, and quantities (**Solar Fountains**, "
            "**Tranquillity Sphere**, **45 cm**, **£199**, **3-year "
            "warranty**).\n"
            " - Don't refuse — the merchant has clearly defined this "
            "category; produce an overview from what's given even if the "
            "description is short.\n"
            " - Don't invent facts beyond what the sources say. If the "
            "description doesn't cover something the customer asked about, "
            "acknowledge that the cards below show the actual range.\n"
            " - Speak as the store (\"our Solar Fountains collection\"), "
            "not as an external assistant (\"according to the data\")."
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nCategory + supporting items (sub-categories OR products — "
              f"check content_type on each):\n{sources_blob}"
        )

    if purpose == "product_qa":
        # PRODUCT Q&A mode (post 2026-05-25). Used by ProductQAAgent when
        # the customer asks a question ABOUT products in general rather
        # than browsing them. Examples: "are your products waterproof?",
        # "do you have anything organic?", "what's the typical warranty?",
        # "is this safe for kids?".
        #
        # Sources blob shape:
        #   1..N (cms_page / cms_block) — PRIMARY. Merchant-curated
        #        FAQ-style content (Materials & Care, Safety, Sizing,
        #        Warranty pages, sometimes inline FAQ blocks). This is
        #        where catalogue-wide product-attribute answers actually
        #        live.
        #   N+1..M (product) — SECONDARY. Compact product format
        #        (name + SKU + price + short description). Used as
        #        concrete examples the LLM can name inline ("such as our
        #        X and Y"), not as the primary answer source.
        #
        # Display contract:
        #   The ProductQAAgent puts ONLY the CMS sources into
        #   data.sources so the frontend renders content cards + the
        #   citation strip. Products reach the LLM via this prompt for
        #   grounding but never render as product cards — the customer
        #   asked a question, not a browse.
        return (
            "You are answering a Q&A-style question about the store's "
            "products in general. The customer asked a question (yes/no, "
            "attribute, care, safety, sizing, warranty, compatibility) and "
            "wants a direct text answer — NOT a product listing.\n\n"
            "Sources below come in two flavors — check the content_type "
            "tag on each:\n"
            "  - `[cms_page]` / `[cms_block]` (PRIMARY) — merchant-curated "
            "FAQ-style content. These hold the authoritative answer to "
            "catalogue-wide product questions (Materials & Care, Safety, "
            "Sizing, Warranty, etc.). Prefer these when synthesising the "
            "answer body.\n"
            "  - `[product]` (SECONDARY) — concrete product examples. Use "
            "them to name 1-2 actual products inline as evidence (\"such "
            "as our Tranquillity Sphere and Mini Cascade Bowl\"), NOT as "
            "the primary answer source.\n\n"
            "Goal: write a clear 3-5 sentence answer that directly "
            "addresses the customer's question, drawing from the CMS "
            "content. Longer (up to 7-8 sentences) is fine for complex "
            "questions; shorter is fine for clean yes/no answers. The "
            "customer will not see product cards beneath this reply, so "
            "the text must stand on its own.\n\n"
            "Rules:\n"
            " - Plain prose. No markdown headings. No bulleted or numbered "
            "lists unless the customer explicitly asked.\n"
            " - Bold concrete facts: materials, dimensions, prices, "
            "timeframes, warranty terms (**600D Oxford cloth**, **3-year "
            "warranty**, **30 days**, **£199**).\n"
            " - When the answer is yes/no, lead with \"Yes\" / \"No\" / "
            "\"Mostly yes — \" / \"It depends — \" so the customer gets "
            "the conclusion before the explanation.\n"
            " - Don't invent specs, warranty terms, materials, or product "
            "names that aren't in the sources. If the sources don't fully "
            "cover the question, share what they DO cover and acknowledge "
            "the gap (\"the curated info doesn't spell out X — happy to "
            "check with our team\"). Don't refuse outright.\n"
            " - Speak as the store (\"our outdoor covers are...\"), not as "
            "an external assistant (\"the documentation says...\").\n"
            " - Do NOT include phone numbers or email addresses inline — "
            "the interface renders contact options as separate clickable "
            "chips below your message."
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nFAQ-style content (PRIMARY) + product examples (SECONDARY):\n{sources_blob}"
        )

    if purpose == "order_answer":
        # ORDER-GROUNDED ANSWER mode (2026-06-09). Used by OrderAgent for
        # free-text questions/requests about a logged-in customer's OWN
        # order — status ("did it ship?", "where's my order?"), contents
        # ("what did I buy in order X?"), or change/modify requests
        # ("change my shipping address"). The single `[order]` source
        # carries that order's facts, scoped to non-PII (no street address,
        # no payment, no customer name). NEVER refuse — the customer is
        # authenticated and we have their order in hand.
        return (
            "You are a store assistant answering a logged-in customer's "
            "question about their OWN order. The source below ([order]) "
            "contains that order's facts: order number, status, items, "
            "totals, shipping method, tracking, and whether it can still "
            "be cancelled or has already shipped. Answer using ONLY these "
            "facts.\n\n"
            "Rules:\n"
            " - Lead with the direct answer. Always name the order number "
            "(**#1000064522**) and its current **status**.\n"
            " - ORDER CONTENTS (\"what did I buy?\"): list the items "
            "naturally in prose (\"You ordered **2x Solar Fountain** and "
            "**1x Pump Kit**\"). Don't dump SKUs.\n"
            " - STATUS / tracking: state the status and, if present, the "
            "tracking carrier + number. If it hasn't shipped, say so "
            "plainly. NEVER invent a delivery date that isn't in the "
            "facts.\n"
            " - CHANGE / MODIFY requests (shipping address, items, "
            "delivery): a customer CANNOT change a placed order "
            "themselves. Say so clearly, then — if it has NOT shipped — "
            "tell them the team can help and to reach out; if it HAS "
            "shipped, explain it's too late to change and to contact "
            "support about options. NEVER promise the change will happen.\n"
            " - Bold concrete values (order number, status, prices, "
            "quantities, tracking numbers).\n"
            " - Plain prose, 2-4 sentences. No markdown headings, no lists "
            "unless asked.\n"
            " - Do NOT put phone numbers or emails inline — the interface "
            "renders clickable contact chips below your message.\n"
            " - NEVER say you have no information about the order — the "
            "facts are right here. If the exact thing asked isn't in the "
            "facts, give the relevant status and point them to support."
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nThe customer's order:\n{sources_blob}"
        )

    if purpose == "purchase_history":
        # PURCHASE-HISTORY / "your usuals" mode (2026-06-09). Used by
        # PurchaseHistoryAgent. Sources are the customer's most-bought
        # products (name + how many separate orders included it + total
        # quantity) — catalog data + counts, no PII. Warmly summarise
        # their buying habits and invite a reorder; the product list +
        # "Buy again" actions render separately.
        return (
            "You are a store assistant telling a logged-in customer what "
            "they buy most often, from their own purchase history. The "
            "sources below are their top products, each with how many "
            "separate orders included it and the total quantity bought.\n\n"
            "Rules:\n"
            " - Open with a short, friendly summary naming their top 1-3 "
            "products and roughly how often (\"You order **Solar "
            "Fountain** most — it's been in **5** of your orders\").\n"
            " - 2-3 sentences, plain prose. The product list + one-tap "
            "'Buy again' actions render below your text, so don't "
            "enumerate everything or list SKUs.\n"
            " - Invite them to reorder (\"want me to add your usuals to "
            "your cart?\").\n"
            " - Bold product names and counts. Don't invent products or "
            "numbers beyond the sources."
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nThe customer's most-bought products:\n{sources_blob}"
        )

    # Default "answer" purpose + "comparison" (same prompt, comparison
    # gets an extra hint via `instruction`).
    #
    # NOTE: there used to be a `contact_line` directive here that asked
    # the LLM to mention phone/email inline whenever it refused. That
    # was the right idea at the time but it conflicts with the chip
    # rendering UX — the frontend now shows clickable tel:/mailto:
    # chips below every chatbot reply, so the LLM dumping the same
    # numbers into its text body produced an ugly "duplicate contact"
    # experience AND triggered refusals in cases where the evidence
    # actually contained the answer (the LLM saw the "include contact"
    # instruction and concluded it should refuse to use it). We rely on
    # the chip channel for contact handoff; the LLM should produce
    # clean answers using only the source content.
    active_retrieval_rule = ""
    if active_retrieval and purpose == "answer":
        active_retrieval_rule = (
            " - **Active Retrieval Tools.** If the initial sources provided are insufficient, incomplete, or lack critical facts needed to answer the question, do NOT refuse or say you don't know yet. Instead, use the active retrieval tools (`retrieve_more_content` or `retrieve_more_products`) to query for more information. Only give up if, after executing your tool-calling step(s), the information remains unavailable.\n"
        )

    return (
        "You are a concise store assistant. Answer the customer's question using ONLY the sources below.\n\n"
        "Rules:\n"
        + active_retrieval_rule +
        " - If the sources don't contain the answer, say honestly in one sentence that you don't have "
        "that specific information. Do NOT include phone numbers or email addresses in your reply — the "
        "interface renders contact options as separate clickable chips below your message; mentioning "
        "them inline duplicates that and makes the reply feel cluttered.\n"
        " - Keep the answer to 1-2 short sentences. For \"tell me about\" requests, you may use 2-3 sentences.\n"
        " - Put any concrete number, measurement, timeframe, or money value in **bold** (e.g. **30 days**, "
        "**$50**, **1.5 kg**).\n"
        " - Never invent SKUs, prices, dates, dimensions, or policy terms that aren't in the sources.\n"
        " - No markdown headings. No bulleted lists unless the customer explicitly asked for a list.\n"
        " - Plain prose only — write the way a helpful human store assistant would.\n"
        + faq_link_rule +
        " - **Direction of flow matters.** If the customer describes RECEIVING a damaged, broken, or "
        "defective product, and the evidence only covers RETURNING such items (e.g. 'damaged-on-return' "
        "or 'sender's responsibility'), do NOT apply that evidence to the customer's situation. Say plainly "
        "that the policy in evidence is for customer-initiated returns, and that received-damaged claims "
        "need to be raised with the support team separately.\n"
        " - **Comparative questions** (\"difference between X and Y\", \"X vs Y\"): contrast both sides "
        "using only what the sources say about each. If the evidence covers only one side, name what "
        "you do know and acknowledge the gap on the other.\n"
        " - **When the question is about store info** (address, hours, phone, email, payment options, "
        "shipping methods, etc.) and a `[store_config: ...]` source is present, USE the facts in that "
        "source verbatim. The composite contains the literal values from the store's admin config — they "
        "are authoritative. Do not say you don't have the information when a store_config source is "
        "provided that contains the answer.\n"
        " - **FACTUAL EXTRACTION across messy sources.** Merchants frequently store key facts (address, "
        "phone, hours, payment methods, policy text) inside CMS blocks with unrelated names (`footer_v2`, "
        "`sirena_footer`, `info_main`) or as inline HTML in pages whose identifier is cryptic (`pp` for "
        "privacy policy, `home` for an FAQ-shaped page). When the customer asks for a SPECIFIC fact "
        "(e.g. \"what's the store address?\", \"do you accept PayPal?\", \"when are you open?\"), scan "
        "the BODY of every source — including blocks whose title looks unrelated — for that fact. If a "
        "block contains an address-shaped sequence, a phone number, opening hours, etc., that's likely "
        "where the merchant has put the answer. Pull it out and present it cleanly. Do NOT refuse on the "
        "basis that the source's title doesn't match the question; the merchant's content hygiene is "
        "their problem, not the customer's."
        + instruction_block
        + "\n\n"
        f"Customer question: {query}"
        + history_block
        + f"\n\nSources:\n{sources_blob}"
    )


def _format_source_for_prompt(s: dict, compact_products: bool = False) -> str:
    """Flatten one source into the text block the RAG summarizer sees.

    Per-content-type formatting because different shapes need different
    framing for the LLM:
      - product             → sku, variants, price, stock, attributes, description.
                              When `compact_products=True` (purpose=category_info),
                              uses _format_product_source_compact instead:
                              name + sku + price + 240-char short.
      - cms_page / cms_block → URL, heading, meta description, content body.
      - store_config         → the full composite body (contact_info, shipping_options,
                               payment_options, store_identity, tax_info, store_rules,
                               locale_info, social_links). Address/phone/hours live
                               in `contact_info`; truncating these at 800 chars meant
                               the LLM saw the anchors but not the actual facts.
      - category             → breadcrumb + URL + meta_description + FULL description
                               (15000-char cap). Skips the generic fallback that used
                               to read `summary` first and cap at 800, which gave
                               the LLM a meta_description stub instead of the
                               merchant-authored description.
      - everything else      → generic title + body fallback.

    When the source carries a `comparison_side` tag (set by
    GenericChatAgent's comparative branch), the side is prepended so the
    LLM can cleanly attribute facts to the right operand.

    `compact_products` is set by _build_answer_prompt when purpose is
    category_info — the LLM is describing a category and the supporting
    products are illustrative anchors, not the primary subject, so the
    full per-product dump would drown out the category description.
    """
    ct = (s.get("content_type") or "").lower()
    title = s.get("title") or s.get("name") or s.get("identifier") or s.get("sku") or s.get("label") or ""
    side = (s.get("comparison_side") or "").strip()
    side_prefix = f"[COMPARE-SIDE: {side}] " if side else ""

    if ct == "product" or s.get("sku") or s.get("type_id"):
        # Compact form for category_info supporting sources — the LLM
        # only needs name + SKU + price + short description to mention
        # 1-2 products illustratively. Full _format_product_source dumps
        # ~50 lines per product (all attributes, all variants, child
        # SKUs, 1500-char description) which drowns out the category
        # description it's supposed to be building the answer from.
        if compact_products:
            return side_prefix + _format_product_source_compact(s, title)
        return side_prefix + _format_product_source(s, title)

    if ct in ("cms_page", "cms_block"):
        return side_prefix + _format_cms_source(s, ct, title)

    if ct == "store_config":
        return side_prefix + _format_store_config_source(s, title)

    if ct == "category":
        return side_prefix + _format_category_source(s, title)

    if ct == "faq":
        return side_prefix + _format_faq_source(s, title)

    body = (s.get("summary") or s.get("content") or s.get("description") or "")[:800]
    return f"{side_prefix}[{ct or 'source'}] {title}\n{body}"

def _format_faq_source(s: dict, title: str) -> str:
    """Format a merchant-authored FAQ source for the prompt.

    Unlike the generic fallback (which reads the 300-char `summary` first),
    this uses the FULL answer body so any link the merchant put inside the FAQ
    answer survives into the prompt intact — the answer-purpose FAQ-link rule
    then turns that URL into a clickable markdown link in the reply. FAQ
    answers are short, so a 4000-char cap is plenty.
    """
    body = (s.get("content") or s.get("summary") or s.get("description") or "")[:4000]
    return f"[faq] {title}\n{body}"


def _format_category_source(s: dict, title: str) -> str:
    """Format a category source for the LLM prompt.

    The previous version routed categories through the generic-fallback
    branch which read `summary` first (= merchant's short meta_description,
    typically 1-2 sentences) and capped the body at 800 chars. The LLM
    saw a stub and wrote answers off the supporting products instead of
    the category description — exactly the bug CategoryInfoAgent was
    designed to avoid.

    This formatter prioritises the FULL merchant-authored description
    (passed as `description` / `content` by CategoryInfoAgent.loadCategorySource,
    or by CategoryContentProvider when the category was retrieved from
    Qdrant in another path) and keeps meta_description / breadcrumb /
    URL as structured framing fields the LLM can use to anchor its
    answer.

    Body cap of 15000 chars matches what CMS pages get — merchant
    category descriptions can be substantial (SEO-heavy landing pages)
    and we don't want to truncate them mid-sentence the way the 800-char
    fallback did.

    Falls back gracefully to summary / meta_description when the full
    description is empty — this happens for CHILD categories that
    CategoryInfoAgent slims down on purpose (enumerating sub-collections
    only needs name + 1-line summary, not full description per child).
    """
    parts: list[str] = [f"[category] {title}"]

    breadcrumb = (s.get("breadcrumb") or "").strip()
    if breadcrumb:
        parts.append(f"Breadcrumb: {breadcrumb}")

    permalink = (s.get("permalink") or "").strip()
    if permalink:
        parts.append(f"URL: {permalink}")

    # Show meta_description as a separate "Summary" line when it's
    # present AND different from the body description. Gives the LLM
    # the merchant's curated 1-line take alongside the longer body.
    meta_desc = (s.get("meta_description") or "").strip()
    body = (s.get("description") or s.get("content") or "").strip()
    if meta_desc and meta_desc != body:
        parts.append(f"Summary: {meta_desc}")

    if body:
        parts.append("")
        parts.append(str(body)[:15000])
    elif meta_desc:
        # No body — fall back to meta_description as the body itself
        # (already shown as Summary above, but it's the only content
        # we have so include it here too in case the LLM only reads
        # the body section).
        parts.append("")
        parts.append(meta_desc)
    else:
        # Last-ditch fallback for child categories that only carry a
        # short summary. Keeps the source non-empty so the LLM knows
        # the category exists even when description is unavailable.
        summary = (s.get("summary") or "").strip()
        if summary:
            parts.append("")
            parts.append(summary)

    return "\n".join(parts)


def _format_product_source_compact(s: dict, title: str) -> str:
    """Compact product source for category_info supporting context.

    Used when the LLM is describing a category and the products are
    illustrative anchors, not the primary subject. ~4 lines per product:
    name, SKU, price, short description. The full _format_product_source
    is correct for ProductSearchAgent / ProductDetailAgent (the customer
    asked about products and wants every detail) but wrong here — we
    just need the LLM to be able to name 1-2 products.

    Drops: type_id, stock_status, categories, brand, attributes,
    variant_attributes, children, child_skus, long description. All of
    those are still on the source dict (Qdrant returned them, the
    Magento side passed them through) — they just don't go INTO the
    LLM prompt. The product cards rendered below the answer text still
    have them.
    """
    parts: list[str] = [f"[product] {title}"]

    sku = s.get("sku")
    if sku:
        parts.append(f"SKU: {sku}")

    price = s.get("price")
    if price:
        currency = s.get("currency") or ""
        parts.append(f"Price: {price} {currency}".strip())

    # Short description, capped tight. Prefer the explicit short_description
    # field, then summary, then the first 200 chars of full description.
    short = (
        s.get("short_description")
        or s.get("summary")
        or ""
    )
    if not short:
        long_desc = (s.get("description") or "")
        short = str(long_desc)[:200]
    if short:
        parts.append(f"Short: {str(short)[:240]}")

    return "\n".join(parts)


def _format_store_config_source(s: dict, title: str) -> str:
    """Lay out a store_config composite for the LLM.

    StoreConfigContentProvider on the Magento side packs each composite
    (contact_info, shipping_options, payment_options, store_identity,
    tax_info, store_rules, locale_info, social_links) into a single
    payload with:

      - `label`   — human title (e.g. "Store contact information")
      - `content` — anchors + facts ("Phone number: ...; Email: ...;
                    Mailing address: street, city, postcode, country;
                    Business hours: ...")
      - `value`   — facts only, no anchors
      - `summary` — first 300 chars of facts, used for card snippets

    Previously the generic fallback read `summary` first and capped at
    800 chars — so the LLM saw 300 chars of facts and lost half the
    composite. That's why "what's the store address" got refused even
    when the address WAS indexed: the address was past the 300-char cap.

    Now we explicitly read `value` (facts only) preferring it over
    `content` (which leads with anchor text the LLM doesn't need to
    see), cap at 6000 chars (composites rarely exceed 2000), and tag
    the source with its composite key so the LLM knows which kind of
    info it's looking at.
    """
    key = str(s.get("key") or s.get("entity_id") or "").strip() or "store_config"

    # `value` is "facts only" (no anchor preamble). Prefer it because
    # the anchors are retrieval-only signal — at answer time they just
    # take up token budget. Fall back to `content` (which contains
    # anchors + facts) when `value` is empty (old payloads from before
    # the composite split).
    body = (
        s.get("value")
        or s.get("content")
        or s.get("summary")
        or ""
    )
    body = str(body)

    label = title or s.get("label") or key.replace("_", " ").title()

    parts = [f"[store_config: {key}] {label}"]
    if body:
        parts.append(body)
    return "\n".join(parts)


def _format_cms_source(s: dict, ct: str, title: str) -> str:
    """Lay out a CMS page / block with all its useful framing context.

    The LLM benefits from seeing the page's display heading and URL — both
    let it write a more specific answer ("see our **Return Policy** page
    for full details"). meta_description is added explicitly so it doesn't
    get truncated when the body is long.

    Body cap: 15000 chars per source. CMS policy pages routinely run
    several thousand words (return policy, warranty terms, FAQ pages);
    the old 4000-char cap chopped them mid-sentence and the LLM could
    only see the start. With Gemini's 1M-token context window plus
    typical 6 sources × ~3700 tokens per source = ~22000 tokens, this
    fits comfortably while letting full policy content reach the model.
    If you find the LLM losing the early sources to "lost in the
    middle", drop this back to 8000-10000.
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
        parts.append(str(body))

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
    """Compat shim — delegates to the canonical slug() in app.utils.slug.

    Kept as a private name in this module so the compat-fallback block
    (post-filter retry) keeps working without touching its many literal
    f-string interpolations. New code should `from backend.app.utils.slug
    import slug` directly. Same algorithm by definition since this just
    forwards to it.
    """
    from backend.app.utils.slug import slug as _shared_slug
    return _shared_slug(value)


def _apply_sort(hits: list[dict], sort_by: str, sort_order: str) -> list[dict]:
    """Apply a customer-requested sort to the (already-filtered) candidate
    pool. Handles missing / non-numeric payload fields gracefully — items
    with no sortable value drop to the end so they don't compete with
    properly-priced items for the top slots.

    sort_by   — "price" | "name" | "rating" | "newest". Anything else
                falls through (returns hits unchanged) rather than
                erroring, so a forward-compatible classifier output
                doesn't break retrieval if it sends a sort the backend
                hasn't learned yet.
    sort_order — "asc" | "desc". Defaults to ascending for unknown values.
    """
    reverse = sort_order == "desc"

    if sort_by == "price":
        def key(h: dict) -> tuple[int, float]:
            raw = h.get("price")
            try:
                v = float(raw)
                if v <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                # No usable price → push to the back (group=1) regardless
                # of asc/desc so the top of the list always shows items
                # with real prices.
                return (1, 0.0)
            return (0, v)
        return sorted(hits, key=key, reverse=reverse)

    if sort_by == "name":
        def key_name(h: dict) -> tuple[int, str]:
            name = str(h.get("name") or h.get("title") or "").strip().lower()
            return (0 if name else 1, name)
        return sorted(hits, key=key_name, reverse=reverse)

    if sort_by == "rating":
        def key_rating(h: dict) -> tuple[int, float]:
            try:
                v = float(h.get("average_rating") or 0)
            except (TypeError, ValueError):
                v = 0.0
            return (0 if v > 0 else 1, v)
        return sorted(hits, key=key_rating, reverse=reverse)

    if sort_by == "newest":
        def key_newest(h: dict) -> tuple[int, str]:
            ts = str(h.get("updated_at") or h.get("created_at") or "").strip()
            return (0 if ts else 1, ts)
        # "newest" naturally implies descending date order — flip when
        # asc is explicitly requested.
        return sorted(hits, key=key_newest, reverse=not (sort_order == "asc"))

    return hits


def _lookup_by_skus(
    client_id: str,
    domain: str,
    skus: list[str],
    store_code: Optional[str] = None,
) -> list[dict]:
    """Direct SKU lookup via Qdrant scroll with filter — no embedding required.

    When `store_code` is provided we also filter on it so multi-store
    deployments return the customer's store-view variant of each SKU.
    Without it (legacy single-store) we get whichever variant happens to
    score first, which for single-store collections is fine (only one
    variant exists anyway).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    collection = get_collection_name(client_id, domain)
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        if collection not in existing:
            return []

        must = [FieldCondition(key="sku", match=MatchAny(any=skus))]
        if store_code:
            must.append(FieldCondition(key="store_code", match=MatchValue(value=store_code)))

        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=must),
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
    """Delegate to the existing llm_rerank_service already used by the /magento/search endpoint.

    Phase 2.1a (LLM Reranker Tuning) — wrapper-level latency telemetry.
    The inner service times the LLM call itself; we add a wrapper-level
    timer here so we can tell the two apart:

      * `inner ms`  — pure LLM round-trip (logged by llm_rerank_service)
      * `wrapper ms` — LLM + payload reshaping + (future) cache lookups
                       + provider client init overhead

    When we later add the LRU cache in 2.1d, cache hits will show as
    near-zero wrapper-ms with no corresponding inner-ms entry — that's
    how we'll measure cache hit rate without a separate counter.
    """
    import time
    from backend.app.services.llm_rerank_service import llm_rerank_products

    t0 = time.perf_counter()
    result = llm_rerank_products(
        query,
        hits,
        len(hits),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        client_id=license_data["client_id"],
    ) or hits
    wrapper_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "[rerank] client=%s provider=%s model=%s candidates=%d returned=%d wrapper_ms=%d",
        license_data.get("client_id", "?"),
        llm_provider or "gemini",
        llm_model or "(default)",
        len(hits),
        len(result),
        wrapper_ms,
    )
    return result
