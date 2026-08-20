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

# How many candidates the relaxation pass pulls before ranking them by
# distance from what the customer asked for. Wider than the answer size
# because the nearest value is frequently not the nearest vector match.
_RELAX_POOL = 30


def _bare(text: str) -> str:
    """Key reduced to alphanumerics, for tolerant matching."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _decimals(value: float) -> int:
    """How many decimal places a value was written to."""
    text = f"{value:g}"
    return len(text.split(".")[1]) if "." in text else 0


def _rank_by_distance(
    hits: list[dict[str, Any]],
    key: str,
    unit: str,
    target: float,
) -> list[tuple[dict[str, Any], float]]:
    """Order candidates by how close their value sits to the one asked for.

    A product carrying several values for the key (its description and its
    spec summary disagreeing) is judged on whichever is closest -- the most
    generous reading, which is the right one when we are already telling the
    customer these are near misses rather than matches.
    """
    ranked: list[tuple[dict[str, Any], float]] = []
    for hit in hits:
        best: Optional[float] = None
        for spec in hit.get("specs") or []:
            if not isinstance(spec, dict) or spec.get("key") != key:
                continue
            if unit and (spec.get("unit") or "") != unit:
                continue
            num = spec.get("num")
            if not isinstance(num, (int, float)):
                continue
            if best is None or abs(num - target) < abs(best - target):
                best = float(num)
        if best is not None:
            ranked.append((hit, best))
    ranked.sort(key=lambda pair: abs(pair[1] - target))
    return ranked


def make_retrieval_tools(
    *,
    client_id: str,
    domain: str,
    api_key: str,
    store_code: Optional[str] = None,
    hybrid: bool = False,
    source_formatter: Callable[[dict[str, Any]], str],
    spec_vocabulary: Optional[dict[str, Any]] = None,
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
    def retrieve_more_products(
        query: str,
        limit: int = _PRODUCT_LIMIT_DEFAULT,
        category_id: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        attribute_filters: Optional[dict[str, str]] = None,
    ) -> str:
        """Search the store's product catalogue for additional products
        matching the query. Use this when the customer's question would
        benefit from concrete product evidence that wasn't in the
        initial sources — e.g. they asked "what do you have in red?"
        but the initial sources are all CMS pages.

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
            return "\n\n".join(source_formatter(h) for h in hits)
        except Exception as exc:
            return f"Error performing product search: {exc}"


    @tool
    def find_products_by_spec(
        spec_key: str,
        operator: str = ">=",
        value: Optional[float] = None,
        value_max: Optional[float] = None,
        unit: Optional[str] = None,
        query_hint: str = "",
        limit: int = _PRODUCT_LIMIT_DEFAULT,
    ) -> str:
        """Find products by a MEASURED specification value.

        Use this whenever the customer asks for a specification the product
        in front of them does not meet -- "what if I wanted 10 GPM?", "do you
        have anything lighter than 20 lbs?", "something rated above 100 psi".
        This searches by the NUMBER, which ordinary product search cannot do.

        Args:
            spec_key: Which specification, from the list of available keys
                below. Must be one of them.
            operator: ">=", "<=", "=", "between", or "any" (any product that
                states this specification at all).
            value: The number the customer asked for.
            value_max: Upper bound, for "between" only.
            unit: The unit as listed for that key below. Values are compared
                only within the same unit, so passing the wrong one finds
                nothing.
            query_hint: A short phrase describing the kind of product, e.g.
                "DEF dispenser". Keeps results in the right family instead of
                returning anything in the catalogue that meets the number.
            limit: How many products to return. Default 5, max 10.
        """
        try:
            limit = max(1, min(int(limit), _PRODUCT_LIMIT_MAX))
            key = (spec_key or "").strip().lower().replace(" ", "_")
            if not key:
                return "No spec_key provided."

            known = spec_vocabulary or {}
            if known and key not in known:
                # The model routinely drops or adds a separator -- 'flowrate'
                # for 'flow_rate'. Comparing on alphanumerics alone resolves
                # that silently instead of returning an error the customer
                # would experience as the assistant simply not knowing.
                stripped = _bare(key)
                exact = [k for k in known if _bare(k) == stripped]
                if len(exact) == 1:
                    key = exact[0]
                else:
                    near = [k for k in known
                            if stripped and (stripped in _bare(k) or _bare(k) in stripped)]
                    suggestion = f" Did you mean: {', '.join(near[:5])}?" if near else ""
                    return (
                        f"This store does not record a specification called '{key}'."
                        f"{suggestion} Available: {', '.join(sorted(known)[:25])}"
                    )

            resolved_unit = (unit or "").strip().lower()
            if not resolved_unit and key in known:
                resolved_unit = str((known.get(key) or {}).get("unit") or "")

            op = (operator or ">=").strip().lower()
            pred: dict[str, Any] = {"key": key, "unit": resolved_unit or None}
            if op == ">=" and value is not None:
                pred["gte"] = float(value)
            elif op == "<=" and value is not None:
                pred["lte"] = float(value)
            elif op == "between" and value is not None and value_max is not None:
                pred["gte"], pred["lte"] = float(value), float(value_max)
            elif op == "=" and value is not None:
                # Exact float equality would miss 9.0 against a stored 9.
                # Half a unit of the written precision is the same tolerance
                # the extractor uses to decide two values agree.
                tol = 0.5 * (10.0 ** -_decimals(float(value)))
                pred["gte"], pred["lte"] = float(value) - tol, float(value) + tol

            text = (query_hint or "").strip() or key.replace("_", " ")
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
                spec_filters=[pred],
            )
            if hits:
                return "\n\n".join(source_formatter(h) for h in hits)

            # ── Relaxation ───────────────────────────────────────────────
            # Nothing meets the bound. Rather than report a dead end, drop
            # the numeric condition, keep the key, and rank what the store
            # DOES carry by distance from what was asked for.
            #
            # The banner is the mechanism, not a suggestion: it states the
            # requirement was not met, so the answer cannot present these as
            # though they satisfied it. Leaving that to the model's judgement
            # is exactly how "answering around a gap" happens.
            if value is None:
                return "No products found with that specification."

            pool = qdrant_search_products(
                client_id=client_id,
                domain=domain,
                query_vector=q_vec,
                limit=_RELAX_POOL,
                content_types=["product"],
                store_code=store_code,
                hybrid=hybrid and sparse_vec is not None,
                sparse_query_vector=sparse_vec,
                with_vectors=False,
                spec_filters=[{"key": key, "unit": resolved_unit or None}],
            )
            nearest = _rank_by_distance(pool, key, resolved_unit, float(value))
            if not nearest:
                return (
                    f"NOT AVAILABLE: no product in this store states a {key.replace('_', ' ')} "
                    f"at all. Say plainly that this specification is not listed, and offer to "
                    f"put the customer in touch rather than substituting a different figure."
                )

            unit_label = f" {resolved_unit}" if resolved_unit else ""
            shown = [h for h, _ in nearest[:limit]]
            actual = ", ".join(f"{v:g}{unit_label}" for _, v in nearest[:limit])
            banner = (
                f"RELAXED SEARCH -- NOTHING MEETS {key.replace('_', ' ')} "
                f"{op} {value:g}{unit_label}. The products below are the CLOSEST this store "
                f"carries ({actual}), not matches. Your reply MUST say the exact requirement "
                f"could not be met before presenting them."
            )
            return banner + "\n\n" + "\n\n".join(source_formatter(h) for h in shown)

        except Exception as exc:
            return f"Error performing specification search: {exc}"

    tools = [retrieve_more_content, retrieve_more_products, find_products_by_spec]
    # The LLM only knows which specifications it may filter on because the
    # description tells it, and that list is built per request from what this
    # store actually has. No hardcoded spec vocabulary anywhere -- a pump
    # catalogue advertises flow_rate, a furniture one seat_height, same code.
    if spec_vocabulary:
        lines = []
        for k, meta in list(spec_vocabulary.items())[:60]:
            meta = meta if isinstance(meta, dict) else {}
            unit = str(meta.get("unit") or "").strip()
            count = int(meta.get("count") or 0)
            lines.append(f"  {k}{f' ({unit})' if unit else ''}"
                         f"{f' - {count} products' if count else ''}")
        find_products_by_spec.description = (
            find_products_by_spec.description.rstrip()
            + "\n\nSpecifications available in THIS store:\n"
            + "\n".join(lines)
        )
    else:
        # Nothing extracted yet (no sync since the feature shipped). Say so
        # rather than leaving the model to guess key names that cannot match.
        find_products_by_spec.description = (
            find_products_by_spec.description.rstrip()
            + "\n\nNOTE: this store has no extracted specifications yet, so this "
              "tool will find nothing. Use retrieve_more_products instead."
        )

    tool_map = {t.name: t for t in tools}
    return tools, tool_map
