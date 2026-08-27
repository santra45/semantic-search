"""
Active-retrieval tool factory (Phase 3.2).

When the LLM is answering a customer question and finds the initial
sources insufficient, it can invoke these tools to fetch more sources
rather than refusing. Pattern matches what /retrieve/answer's prompt
instructs the LLM to do when the active-retrieval flag is on.

Why a factory and not module-level `@tool` decorators:
    The tools need closures over per-request state (client_id, domain,
    decrypted LLM/embedding API key, store_code, merchant's hybrid
    setting). Module-level tools would have to either accept all that
    state as args (which the LLM would see and might pass garbage to),
    or read from a thread-local (more magic, more failure modes).
    The per-request factory pattern keeps the LLM-visible surface area
    small — the LLM only sees `query` and `limit` — while internally
    the tool body has everything it needs.

Tool/file separation rationale:
    Phase 3.1's `tools.py` holds 12 *agent-selector* tools (no-op bodies,
    schemas only — Magento dispatches the corresponding agent). This
    file holds *executor* tools that actually hit Qdrant inside the
    /retrieve/answer LLM loop. Different lifecycle, different bind
    sites — separate file keeps each concern clear.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from langchain_core.tools import tool

# Imports kept at module level — these modules are import-side-effect-
# free, so paying their cost once at app boot beats paying it on every
# request that hits make_retrieval_tools().
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
    scroll_content as qdrant_scroll,
    search_content as qdrant_search_content,
    search_products as qdrant_search_products,
)

# Hard ceiling on active-retrieval LLM iterations. Single source of
# truth — both the one-shot and streaming handlers reference it. Bump
# in one place if observed logs show the LLM frequently needs more
# rounds before producing an answer.
MAX_ACTIVE_RETRIEVAL_ITERATIONS = 3

# Per-call result-count limits. Defaults match what the original inline
# tools used; max caps prevent the LLM from asking for absurd counts
# that would balloon prompt size + cost on the next iteration.
_CONTENT_LIMIT_DEFAULT = 3
_CONTENT_LIMIT_MAX     = 8
_PRODUCT_LIMIT_DEFAULT = 5
_PRODUCT_LIMIT_MAX     = 10

# How many candidates the relaxation pass pulls before ranking them by
# distance from what the customer asked for. Wider than the answer size
# because the nearest value is frequently not the nearest vector match.
# The whole point of this approach is that the answer is already in the
# merchant's own words, sitting in the payload. Matching happens over that
# text directly -- nothing is extracted at sync time, nothing is stored, and
# there is no schema to drift out of step with the catalogue.
#
# The text is cached per collection because a customer asking three questions
# in a row should not re-scroll the catalogue three times. Short TTL so an
# edited product shows up quickly.
_TEXT_CACHE: dict[tuple[str, str, str], tuple[float, list[tuple[dict, str]]]] = {}
_TEXT_TTL_SECONDS = 300
_TEXT_FIELDS = ("name", "short_description", "description", "merchant_info")
_MAX_TEXT_CHARS = 8000
# What the matcher reads plus what a rendered product needs. Excludes
# `embedded_text`, which is most of the payload and only duplicates the
# description fields already listed here.
_WANTED_FIELDS = [
    "sku", "entity_id", "name", "short_description", "description",
    "merchant_info", "price", "regular_price", "currency", "image_url",
    "permalink", "stock_status", "type_id", "brand", "categories",
    "variant_attributes", "average_rating", "content_type", "children",
]

# Whitespace is not meaningful inside a specification. The same catalogue
# writes "90 l/min" and "90l/min", and matching strictly finds barely half of
# them -- measured on a live 722-product store: 6 hits strict, 12 loose.
_SQUASH = re.compile(r"\s+")


def _squash(text: str) -> str:
    """Lowercase and strip ALL whitespace, so spacing cannot hide a match."""
    return _SQUASH.sub("", (text or "").lower())


def _product_text(client_id: str, domain: str, store_code: Optional[str]):
    """Every product's own words, as (payload, all-text, name-only) triples.

    The name is kept separately so a product that IS the thing searched for
    can outrank one that merely mentions it. Descriptions compare against
    rivals constantly -- a Cube 70's copy name-checks the Panther -- and
    without that split the comparison outranks the product.
    """
    key = (client_id, domain, store_code or "")
    hit = _TEXT_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TEXT_TTL_SECONDS:
        return hit[1]

    rows: list[tuple[dict, str]] = []
    for payload in qdrant_scroll(
        client_id=client_id,
        domain=domain,
        content_type="product",
        store_code=store_code,
        payload_fields=_WANTED_FIELDS,
    ):
        blob = " ".join(str(payload.get(f) or "") for f in _TEXT_FIELDS)[:_MAX_TEXT_CHARS]
        if blob.strip():
            rows.append((payload, _squash(blob), _squash(str(payload.get("name") or ""))))

    _TEXT_CACHE[key] = (time.time(), rows)
    return rows


# A number followed by a short word -- "50 l/min", "150 micron", "24 v". Used
# ONLY to tell a customer what forms the store writes its measurements in when
# their term finds nothing. Nothing is stored and nothing is converted; this
# reports the merchant's own wording back.
_MEASURE_FORM = re.compile(r"\d+(?:[.,]\d+)?\s*([a-z][a-z/.\-]{0,7})\b", re.I)
_FORM_STOPWORDS = {
    "x", "and", "or", "the", "a", "of", "in", "to", "for", "with", "per",
    "pack", "piece", "pieces", "year", "years", "day", "days", "month",
    "months", "off", "is", "no", "up", "each", "set", "pc", "pcs",
}


def _measurement_forms(rows, limit: int = 8) -> list[str]:
    """The unit spellings this store actually uses, most common first."""
    counts: dict[str, int] = {}
    for payload, _ in rows:
        blob = " ".join(str(payload.get(f) or "") for f in _TEXT_FIELDS)
        for match in _MEASURE_FORM.finditer(blob):
            token = match.group(1).lower().strip(".-")
            if len(token) < 1 or token in _FORM_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [tok for tok, n in ranked[:limit] if n >= 5]


def make_retrieval_tools(
    *,
    client_id: str,
    domain: str,
    api_key: str,
    embed_api_key: Optional[str] = None,
    embed_model: Optional[str] = None,
    store_code: Optional[str] = None,
    hybrid: bool = False,
    source_formatter: Callable[[dict[str, Any]], str],
    product_sink: Optional[list[dict[str, Any]]] = None,
):
    """Build a fresh pair of active-retrieval tools for ONE /retrieve/answer call.

    Returns:
        (tools_list, tool_name_to_callable_map)

    Each tool is a closure over the per-request context:
      * client_id + domain — tenant + collection scoping
      * api_key            — the LLM key, used for anything that calls
                             a completion model
      * embed_api_key      — the tenant's separate embedding key. None
                             falls back to api_key, which is what every
                             install predating the embedding config
                             sends.
      * embed_model        — embedding model id, already validated by
                             embedding_key_service. None means the
                             server default.
      * store_code         — store-view scoping for the search filter
                             (None = no store filter, used by legacy
                             single-store deployments)
      * hybrid             — when True AND fastembed is installed, the
                             tool uses Qdrant's RRF fusion over dense
                             + BM25 sparse. Falls back to dense-only
                             silently if sparse_embedder isn't
                             available (e.g. fastembed missing in dev).
      * source_formatter   — callable that turns one Qdrant hit dict
                             into the text block the LLM will see.
                             Passed in to avoid a circular import on
                             retrieve._format_source_for_prompt.

    MMR is INTENTIONALLY OFF for active-retrieval queries: by the time
    the LLM is calling these tools, it has already narrowed in on a
    specific need ("warranty coverage", "shipping to Canada"). Adding
    MMR diversification at that point would push results toward
    unrelated content and defeat the refinement.
    """
    # Pre-resolve sparse embedder once when hybrid is requested. If
    # fastembed isn't installed we silently turn hybrid off — the tools
    # still work, just dense-only. Single import attempt here is cheaper
    # than catching the failure inside every tool invocation.
    sparse_embed_query = None
    if hybrid:
        try:
            from backend.app.services.sparse_embedder import embed_sparse_query
            sparse_embed_query = embed_sparse_query
        except Exception:
            hybrid = False  # graceful degrade — never break the answer path

    def _maybe_sparse(query_text: str):
        """Generate sparse vector when hybrid is on; None otherwise.

        Returns None on any sparse-embed failure so the qdrant search
        falls back to dense-only for THIS call (the rest of the
        request keeps trying sparse — failures are per-query, not
        per-handler-lifetime).
        """
        if not hybrid or sparse_embed_query is None:
            return None
        try:
            return sparse_embed_query(query_text)
        except Exception:
            return None

    @tool
    def retrieve_more_content(query: str, limit: int = _CONTENT_LIMIT_DEFAULT) -> str:
        """Search the store's CMS pages, blocks, FAQ entries, and active
        promotions for additional information matching the query. Use
        this when the initial sources provided don't cover what the
        customer asked about, but you suspect the store DOES have the
        information somewhere — e.g. policy details that live on a
        page you haven't seen, or FAQ entries.

        Args:
            query: Refined search phrase. Make it more specific or use
                different keywords than the customer's original
                question; you've already seen the initial sources, so
                avoid querying for the same thing.
            limit: How many additional source snippets to fetch.
                Default 3, max 8.
        """
        try:
            limit = max(1, min(int(limit), _CONTENT_LIMIT_MAX))
            text = query.strip()
            if not text:
                return "No query provided."
            q_vec = embed_query(text, embed_api_key or api_key, client_id, model=embed_model)
            sparse_vec = _maybe_sparse(text)
            hits = qdrant_search_content(
                client_id=client_id,
                domain=domain,
                query_vector=q_vec,
                limit=limit,
                content_types=["cms_page", "cms_block", "faq", "promotion"],
                store_code=store_code,
                hybrid=hybrid and sparse_vec is not None,
                sparse_query_vector=sparse_vec,
                with_vectors=False,
            )
            if not hits:
                return "No additional content found for that query."
            return "\n\n".join(source_formatter(h) for h in hits)
        except Exception as exc:
            # Returning an error string (not raising) lets the LLM read
            # it and decide whether to retry with a different query or
            # give up gracefully — matches the soft-fail pattern the
            # rest of the active-retrieval loop relies on.
            return f"Error performing content search: {exc}"

    @tool
    def retrieve_more_products(
        query: str,
        limit: int = _PRODUCT_LIMIT_DEFAULT,
        category_id: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        attribute_filters: Optional[dict[str, str]] = None,
    ) -> str:
        """Search the store's product catalogue by MEANING, for open-ended
        browsing -- "what do you have in red?", "something for cold
        weather", "a quieter model".

        For a pure value question ('do you have 10 GPM?', 'anything in
        size L?'), prefer `find_products_listing`: embeddings compare
        meaning not magnitude, so "10 GPM" and "8 GPM" look almost
        identical to this tool and it will return either.

        EXCEPTION -- COMPOUND QUERIES. When the customer combines the
        value with a product TYPE, category, colour, or descriptor
        ('20 GPM nozzle', 'red waterproof jacket', 'wireless mouse
        with USB-C'), use THIS tool with the compound phrase as
        `query`. A literal-match search on the value alone (via
        `find_products_listing`) will return anything mentioning the
        value in any product type; semantic search here can rank
        products that match BOTH the value AND the type/descriptor.

        Args:
            query: Refined product search phrase.
            limit: How many products to fetch. Default 5, max 10.
            category_id: Optional numeric category id to filter to
                products in that category only. Use when the customer
                or the conversation context names a category.
            min_price / max_price: Optional price bounds. Use when the
                customer mentions a budget (e.g. "under £50").
            attribute_filters: Optional dict like {"color": "red",
                "size": "m"}. Use when the customer mentions specific
                product attributes the catalogue exposes. Match the
                attribute code and value casing exactly. For brand-style
                filters, use the merchant's brand attribute code as the
                key (commonly "brand", sometimes "manufacturer" or
                "vendor") — e.g. {"brand": "Altico"}.
        """
        try:
            limit = max(1, min(int(limit), _PRODUCT_LIMIT_MAX))
            text = query.strip()
            if not text:
                return "No query provided."
            q_vec = embed_query(text, embed_api_key or api_key, client_id, model=embed_model)
            sparse_vec = _maybe_sparse(text)
            hits = qdrant_search_products(
                client_id=client_id,
                domain=domain,
                query_vector=q_vec,
                limit=limit,
                content_types=["product"],
                store_code=store_code,
                hybrid=hybrid and sparse_vec is not None,
                sparse_query_vector=sparse_vec,
                with_vectors=False,
                # Structured filter rebuild (2026-05-22+) — active-retrieval
                # tool now respects the same FieldCondition pre-filtering as
                # the primary /retrieve/products handler. Brand is routed
                # through attribute_filters[<brand_code>] just like the
                # main pipeline, so the LLM uses the existing attribute
                # filter rather than a separate brand arg.
                attribute_filters=attribute_filters or None,
                category_id=category_id,
                min_price=min_price,
                max_price=max_price,
            )
            if not hits:
                return "No additional products found for that query."
            # Same sink as the literal-match tool. A product found by meaning
            # is no less buyable than one found by wording.
            if product_sink is not None:
                product_sink.extend(hits)
            return "\n\n".join(source_formatter(h) for h in hits)
        except Exception as exc:
            return f"Error performing product search: {exc}"


    @tool
    def find_products_listing(
        search_term: str,
        also_try: Optional[list[str]] = None,
        limit: int = _PRODUCT_LIMIT_DEFAULT,
    ) -> str:
        """Find products whose listed details literally contain a value the
        customer named -- "10 GPM", "50 l/min", "2 inch", "150 micron", "24V".

        Use this the moment a customer asks for a specific figure the product
        in front of them does not have. Ordinary product search matches
        wording, so asking it for "10 GPM" returns whatever *reads* like a
        pump; this matches the merchant's own written specification instead.

        Compound phrases will NOT match. This is LITERAL substring matching;
        no product's text contains the exact string '20 GPM nozzle' back-to
        -back even if 20 GPM nozzles are in the catalogue. For a compound
        query that combines a value with a product type or descriptor, use
        `retrieve_more_products` instead -- its semantic search can rank
        products that match BOTH the value AND the descriptor.

        The header line of this tool's result reports both the match count
        and the truncation: 'N products list "X". Showing K of N.' If K is
        less than N, the tail is not visible to you -- do not make claims
        about what the tail does or does not contain; narrow the query and
        call this tool again, or route to `retrieve_more_products`.

        Report the store's figures exactly as the merchant wrote them. Do NOT
        convert between units -- if the store lists l/min, answer in l/min.
        Converting invents a number the merchant never published.

        Args:
            search_term: The value as the customer expressed it, e.g. "10 GPM".
                Whitespace does not matter; case does not matter.
            also_try: Other ways the SAME value might be written in the
                catalogue -- ["10gpm", "10 g.p.m"]. Spelling variants only.
                Never a converted figure in another unit.
            limit: How many products to return. Default 5, max 10.
        """
        try:
            limit = max(1, min(int(limit), _PRODUCT_LIMIT_MAX))
            terms = [search_term] + list(also_try or [])
            needles = [(t, _squash(t)) for t in terms if t and _squash(t)]
            if not needles:
                return "No search term provided."

            rows = _product_text(client_id, domain, store_code)
            if not rows:
                return "This store has no product data to search."

            seen, by_name, by_body = set(), [], []
            for payload, blob, name_blob in rows:
                for original, needle in needles:
                    if needle not in blob:
                        continue
                    sku = str(payload.get("sku") or payload.get("entity_id") or id(payload))
                    if sku not in seen:
                        seen.add(sku)
                        # A hit in the NAME means this product is the thing
                        # asked for. A hit in the body may only mean its copy
                        # mentions it -- often to compare against it.
                        (by_name if needle in name_blob else by_body).append((payload, original))
                    break
            matched = by_name + by_body

            # Hand the matched products to the caller as DATA, not just as
            # text for the model to paraphrase. The widget renders them as
            # cards from these payloads, so the price on screen is the price
            # in the catalogue -- the model never gets the chance to retype
            # it, and therefore never gets the chance to retype it wrongly.
            if product_sink is not None:
                for payload, _ in matched[:limit]:
                    product_sink.append(payload)

            if matched:
                head = (
                    f"{len(matched)} product(s) list \"{search_term}\". "
                    f"Quote these figures exactly as written -- do not convert them."
                )
                body = "\n\n".join(source_formatter(p) for p, _ in matched[:limit])
                if len(matched) > limit:
                    head += f" Showing {limit} of {len(matched)}."
                return head + "\n\n" + body

            # Nothing matched. A bare "not found" is a poor answer when the
            # store plainly does publish this kind of figure -- just in its own
            # units. Telling the customer which forms the merchant uses is
            # still only reporting what the merchant wrote; it converts
            # nothing and invents nothing.
            forms = _measurement_forms(rows)
            if forms:
                return (
                    f"NOT LISTED: no product states \"{search_term}\". This store writes its "
                    f"measurements as: {', '.join(forms)}. Tell the customer plainly that this "
                    f"exact figure is not listed, mention which of those units the store uses "
                    f"for what they asked about, and invite them to ask again that way. Do NOT "
                    f"convert their figure yourself and do NOT present a different number as "
                    f"though it answered them."
                )
            return (
                f"NOT LISTED: no product states \"{search_term}\", and this store does not "
                f"publish figures of that kind. Say so plainly and offer to put the customer "
                f"in touch with the store."
            )

        except Exception as exc:
            return f"Error searching product listings: {exc}"

    tools = [retrieve_more_content, retrieve_more_products, find_products_listing]
    tool_map = {t.name: t for t in tools}
    return tools, tool_map
