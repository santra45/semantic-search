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

from typing import Any, Callable, Optional

from langchain_core.tools import tool

# Imports kept at module level — these modules are import-side-effect-
# free, so paying their cost once at app boot beats paying it on every
# request that hits make_retrieval_tools().
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
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


def make_retrieval_tools(
    *,
    client_id: str,
    domain: str,
    api_key: str,
    store_code: Optional[str] = None,
    hybrid: bool = False,
    source_formatter: Callable[[dict[str, Any]], str],
):
    """Build a fresh pair of active-retrieval tools for ONE /retrieve/answer call.

    Returns:
        (tools_list, tool_name_to_callable_map)

    Each tool is a closure over the per-request context:
      * client_id + domain — tenant + collection scoping
      * api_key            — Gemini embedding key (same key the LLM
                             call uses; this matches the existing
                             convention in retrieve.py / classify.py)
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
            q_vec = embed_query(text, api_key, client_id)
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
    def retrieve_more_products(query: str, limit: int = _PRODUCT_LIMIT_DEFAULT) -> str:
        """Search the store's product catalogue for additional products
        matching the query. Use this when the customer's question would
        benefit from concrete product evidence that wasn't in the
        initial sources — e.g. they asked "what do you have in red?"
        but the initial sources are all CMS pages.

        Args:
            query: Refined product search phrase.
            limit: How many products to fetch. Default 5, max 10.
        """
        try:
            limit = max(1, min(int(limit), _PRODUCT_LIMIT_MAX))
            text = query.strip()
            if not text:
                return "No query provided."
            q_vec = embed_query(text, api_key, client_id)
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
            )
            if not hits:
                return "No additional products found for that query."
            return "\n\n".join(source_formatter(h) for h in hits)
        except Exception as exc:
            return f"Error performing product search: {exc}"

    tools = [retrieve_more_content, retrieve_more_products]
    tool_map = {t.name: t for t in tools}
    return tools, tool_map
