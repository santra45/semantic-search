import re
import uuid
from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    PayloadField,
    PointStruct,
    Prefetch,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.app.config import EMBED_DIM, QDRANT_HOST, QDRANT_PORT

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

KNOWN_CONTENT_TYPES = [
    "product",
    "cms_page",
    "cms_block",
    "widget",
    "review",
    "policy",
    "faq",
    "store_config",
    "promotion",
    "page",
    "post",
]

# Content types whose body is split into multiple points instead of one.
# When a chunked type is upserted we delete-by-filter on the parent
# (content_type + entity_id + store_code) before inserting the new chunks,
# which atomically:
#   * removes any legacy single-point version (pre-chunking) for this entity
#   * removes any old chunks left over when chunk count shrinks (5 chunks → 3)
#
# Adding a new chunkable type later: just add it here AND make sure
# product_formatter has a matching format_<type>_chunkable function.
CHUNKABLE_CONTENT_TYPES = {"cms_page", "cms_block"}


# ── Hybrid-search named-vector slots (Phase 2.2) ────────────────────────────
#
# Every collection created post-2.2 carries two named vectors:
#   * DENSE_VECTOR_NAME  — the existing Gemini embedding, 3072-dim cosine
#   * SPARSE_VECTOR_NAME — BM25 sparse vector from fastembed for exact-token
#                          / keyword matches that pure semantic misses
#
# At query time the admin toggle (`aichatbot/llm/hybrid_search_enabled`)
# decides whether search uses dense-only (single vector query) or hybrid
# (prefetch from both, fuse via Qdrant's native RRF). Writes always
# populate both slots — BM25 generation is ~5 ms CPU and free, not worth
# making conditional.
#
# Constants are kept here (not embedded in helpers) so any future feature
# that needs to address a specific vector slot (re-embed scripts, vector
# inspection tooling) has a single source of truth.
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse_bm25"


def get_collection_name(client_id: str, domain: str) -> str:
    client_safe = re.sub(r"[^a-zA-Z0-9]", "_", client_id)
    domain_safe = re.sub(r"[^a-zA-Z0-9]", "_", domain)
    return f"products_{domain_safe}_{client_safe}"


def ensure_collection_exists(client_id: str, domain: str) -> str:
    """Create the per-tenant collection if absent.

    Phase 2.2: collections are now born with NAMED vectors:
      * `dense`       — the existing Gemini cosine-similarity vector
      * `sparse_bm25` — BM25 sparse vector for exact-token signal

    This is a breaking shape change relative to the pre-2.2 unnamed
    single-vector layout. Existing collections from before this deploy
    will NOT be migrated automatically — drop them manually and run a
    full re-sync. The decision was made deliberately while the system
    is still in dev (see plan.md Phase 2.2). For production a
    blue/green migration script would replace this no-op behaviour.
    """
    collection_name = get_collection_name(client_id, domain)
    existing = {collection.name for collection in qdrant.get_collections().collections}
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            # Dense slot keeps the same dimensions + distance as the
            # legacy unnamed vector — only the name (and therefore the
            # upsert / query shape) is new.
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                # Default SparseVectorParams — index type is auto-picked
                # by Qdrant (HNSW-on-inverted-index in 1.17). No tuning
                # knobs needed until we have benchmark evidence.
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
    return collection_name


def _collection_exists(client_id: str, domain: str) -> bool:
    collection_name = get_collection_name(client_id, domain)
    existing = {collection.name for collection in qdrant.get_collections().collections}
    return collection_name in existing


def build_point_id(
    client_id: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str] = None,
) -> str:
    """Deterministic Qdrant point id.

    `store_code` is folded into the hash so the same product/page indexed
    for two store views gets two separate Qdrant points — one for each
    locale's content. Without it, the second sync would overwrite the
    first, leaving only one store-view's localised text in the index.

    Backwards-compat: when `store_code` is None or 'default' the id
    collapses to the legacy shape so existing single-store points are not
    invalidated by deploying this change. Pre-existing collections keep
    matching the same UUIDs they always did until a fresh per-store sync
    repopulates them under per-store ids.

    For products specifically we keep the legacy two-segment shape
    (`client-entity`) when there's no store code — that's been the
    canonical shape since day one and changing it would orphan every
    existing single-store product point.
    """
    has_store = store_code not in (None, "", "default")

    if content_type == "product":
        if has_store:
            source = f"{client_id}-product-{entity_id}-{store_code}"
        else:
            source = f"{client_id}-{entity_id}"
    else:
        if has_store:
            source = f"{client_id}-{content_type}-{entity_id}-{store_code}"
        else:
            source = f"{client_id}-{content_type}-{entity_id}"

    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source))


def build_chunk_point_id(
    client_id: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str],
    chunk_index: int,
) -> str:
    """Deterministic Qdrant point id for one chunk of a chunkable entity.

    Shape: `{client}-{type}-{entity}-{store}-chunk-{N}`. Different from
    build_point_id() — the chunked id always carries store_code (defaults
    to "default") and the chunk suffix, so two chunks of the same page
    never collide and chunk N of page X resolves to a stable UUID across
    re-syncs (in-place upsert, no orphan).

    Why a separate function: keeping build_point_id() unchanged means the
    legacy non-chunked path (products, store_config, etc.) keeps the
    exact same UUIDs it had before this feature shipped, so no existing
    client needs to re-sync those types.
    """
    sc = store_code or "default"
    source = f"{client_id}-{content_type}-{entity_id}-{sc}-chunk-{int(chunk_index)}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source))


def _build_entity_filter(
    client_id: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str],
) -> Filter:
    """Filter matching every point belonging to one entity in one store.

    Used for bulk-delete-by-entity (chunked upsert cleanup, chunked
    delete). Matches BOTH legacy single-point variants AND new chunks
    because both write the same payload fields (client_id, content_type,
    entity_id, store_code).

    Including client_id on the filter is defence-in-depth — every
    collection is already per-tenant, but explicit beats implicit when
    the cost is one extra condition.
    """
    must = [
        FieldCondition(key="client_id",    match=MatchValue(value=str(client_id))),
        FieldCondition(key="content_type", match=MatchValue(value=str(content_type))),
        FieldCondition(key="entity_id",    match=MatchValue(value=str(entity_id))),
    ]
    if store_code:
        must.append(FieldCondition(key="store_code", match=MatchValue(value=str(store_code))))
    return Filter(must=must)


def _type_specific_id_key(content_type: str) -> str:
    return {
        "product": "product_id",
        "page": "page_id",
        "post": "post_id",
    }.get(content_type, "entity_id")


def _build_content_filter(
    content_types: Optional[Iterable[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    only_in_stock: bool = False,
    store_code: Optional[str] = None,
) -> Optional[Filter]:
    must_conditions = []
    content_types = [content_type for content_type in (content_types or []) if content_type]

    if content_types:
        if len(content_types) == 1:
            must_conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=content_types[0]))
            )
        else:
            must_conditions.append(
                FieldCondition(key="content_type", match=MatchAny(any=content_types))
            )

    if min_price is not None or max_price is not None:
        must_conditions.append(
            FieldCondition(
                key="price",
                range=Range(gte=min_price, lte=max_price),
            )
        )

    if only_in_stock:
        must_conditions.append(
            FieldCondition(key="stock_status", match=MatchValue(value="instock"))
        )

    # store_code filter — scopes retrieval to a single store view so a
    # French shopper doesn't pull English-indexed content from a sister
    # store. Empty / None means "no store filter" — used by the legacy
    # single-store retrievals that pre-date multi-store sync.
    if store_code:
        must_conditions.append(
            FieldCondition(key="store_code", match=MatchValue(value=store_code))
        )

    return Filter(must=must_conditions) if must_conditions else None


def _format_hit(hit: Any) -> dict[str, Any]:
    payload = dict(hit.payload or {})
    content_type = payload.get("content_type", "product")
    entity_id_key = _type_specific_id_key(content_type)
    # `label` covers store_config rows (whose human title lives there, not in
    # `title`/`name`); leaving it out caused store_config search results to
    # arrive at the RAG summarizer with an empty title and snippet, and the
    # LLM correctly refused with the "I don't see that in our policies" line.
    title = (
        payload.get("title")
        or payload.get("name")
        or payload.get("label")
        or payload.get("question")
        or payload.get("identifier")
        or ""
    )
    # `value` is the store_config payload's actual answer text (e.g. the
    # phone number, the address). Same reason: without it the snippet is
    # empty for any /retrieve/content hit on store info.
    snippet = (
        payload.get("summary")
        or payload.get("excerpt")
        or payload.get("content")
        or payload.get("description")
        or payload.get("value")
        or payload.get("short_description")
        or ""
    )

    result = {
        **payload,
        "content_type": content_type,
        "entity_id": str(payload.get(entity_id_key) or payload.get("entity_id") or ""),
        "score": round(float(hit.score or 0), 4),
        "title": title,
        "summary": snippet[:500],
    }

    if entity_id_key not in result and result["entity_id"]:
        result[entity_id_key] = result["entity_id"]

    # Phase 2.3 — when the caller asked qdrant for vectors (search_content's
    # `with_vectors=True` path, used by MMR), pluck the dense slot onto the
    # hit under a private `_dense_vector` key. The MMR helper consumes it;
    # retrieve.py strips it before responding to Magento so the wire
    # payload doesn't carry 3072-dim floats for no consumer.
    vec = getattr(hit, "vector", None)
    if vec is not None:
        dense = None
        if isinstance(vec, dict):
            # Named-vector collection (post-2.2). DENSE_VECTOR_NAME is the
            # slot MMR cares about — sparse cosine has different metric
            # behaviour and isn't the right input for MMR's similarity.
            dense = vec.get(DENSE_VECTOR_NAME)
        elif isinstance(vec, list):
            # Legacy unnamed-vector collection (pre-2.2). Single list IS
            # the dense vector.
            dense = vec
        if dense:
            result["_dense_vector"] = list(dense)

    return result


def product_exists(client_id: str, domain: str, product_id: str) -> bool:
    return content_item_exists(client_id, domain, "product", product_id)


def content_item_exists(client_id: str, domain: str, content_type: str, entity_id: str) -> bool:
    if not _collection_exists(client_id, domain):
        return False

    collection_name = get_collection_name(client_id, domain)
    point_id = build_point_id(client_id, content_type, entity_id)
    result = qdrant.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=False,
        with_vectors=False,
    )
    return len(result) > 0


def search_content(
    client_id: str,
    domain: str,
    query_vector: list[float],
    limit: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    only_in_stock: bool = False,
    content_types: Optional[list[str]] = None,
    store_code: Optional[str] = None,
    dedupe_by_parent: bool = True,
    max_per_parent: int = 2,
    hybrid: bool = False,
    sparse_query_vector: Optional[SparseVector] = None,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Vector search over the per-tenant collection.

    Two modes (Phase 2.2):

      * **Dense-only** (default, `hybrid=False`) — single-vector query
        against the `dense` slot. Behaves identically to pre-2.2 search
        except for the `using=DENSE_VECTOR_NAME` selector that the
        named-vector schema now requires.

      * **Hybrid** (`hybrid=True`) — Qdrant Query API with two
        prefetches: dense (semantic) and sparse_bm25 (exact-token).
        Results are fused server-side via Reciprocal Rank Fusion
        (Fusion.RRF). Caller must pass `sparse_query_vector` —
        generated by sparse_embedder.embed_sparse_query(). If sparse
        vector is missing while hybrid=True, we degrade gracefully to
        dense-only with a logged warning rather than 500ing the
        request.

    Dedup-by-parent semantics:

      When `dedupe_by_parent` is on (default), we fetch up to 3x `limit`
      raw candidates and keep at most `max_per_parent` hits per
      (content_type, parent_entity_id, store_code). This prevents one
      verbose chunked page from monopolising the top-K — a 20-paragraph
      Return Policy chunked into 20 points would otherwise return 20
      copies of itself when the query matches the page strongly.

      For non-chunked content (products, store_config) the dedup key is
      always unique per point (no two products share the same entity_id
      with the same store_code), so the dedup pass is a no-op.

      `max_per_parent=2` allows compound questions to surface two
      relevant paragraphs of the same page (e.g. "returns AND warranty"
      both matching the policy page).
    """
    collection_name = ensure_collection_exists(client_id, domain)
    query_filter = _build_content_filter(
        content_types=content_types,
        min_price=min_price,
        max_price=max_price,
        only_in_stock=only_in_stock,
        store_code=store_code,
    )

    # Over-fetch when dedup is on so the post-filter pass has enough
    # candidates to fill `limit` even when several top hits collapse
    # into the same parent.
    fetch_limit = min(limit * 3, 50) if dedupe_by_parent else limit

    # Phase 2.3 — when MMR is going to run downstream the caller asks
    # for vectors back on each hit. We only need the DENSE slot (sparse
    # cosine doesn't behave like dense cosine for MMR's similarity term).
    # Specifying a list of names is the qdrant-client knob that fetches
    # only those vectors — cheaper than `with_vectors=True` which would
    # also drag back the sparse_bm25 indices/values.
    vectors_arg: Any = [DENSE_VECTOR_NAME] if with_vectors else False

    if hybrid and sparse_query_vector is not None:
        # Hybrid path — Qdrant fuses the two prefetch result sets via
        # Reciprocal Rank Fusion. Each prefetch carries the SAME filter
        # so content_type / store_code / price / stock all narrow both
        # sides identically before fusion. Per-side limit matches the
        # post-dedup fetch budget so RRF has plenty of headroom.
        result = qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=fetch_limit,
                ),
                Prefetch(
                    query=sparse_query_vector,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=fetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
            with_vectors=vectors_arg,
        )
    else:
        if hybrid and sparse_query_vector is None:
            # Caller asked for hybrid but didn't (or couldn't) generate
            # the sparse vector. Don't 500 — silently fall back to
            # dense-only so the chatbot stays alive. The retrieve.py
            # caller logs the degradation.
            pass
        # Dense-only path — explicit `using=` is REQUIRED post-2.2
        # because the collection now has named vectors. Omitting it
        # would raise "vector name is required".
        result = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=vectors_arg,
        )

    hits = [_format_hit(hit) for hit in result.points]
    if not dedupe_by_parent:
        return hits[:limit]
    return _dedupe_by_parent(hits, limit, max_per_parent=max_per_parent)


def _dedupe_by_parent(hits: list[dict[str, Any]], limit: int, max_per_parent: int = 2) -> list[dict[str, Any]]:
    """Keep up to *max_per_parent* hits per (content_type, parent, store).

    Iterates hits in score order (qdrant returns highest first), keeps the
    first N hits per parent group, stops when *limit* total survivors are
    collected. Stable: input order is preserved among kept hits.
    """
    if limit <= 0:
        return []
    counts: dict[tuple[str, str, str], int] = {}
    out: list[dict[str, Any]] = []
    for hit in hits:
        parent = hit.get("parent_entity_id") or hit.get("entity_id") or ""
        key = (
            str(hit.get("content_type") or ""),
            str(parent),
            str(hit.get("store_code") or ""),
        )
        if counts.get(key, 0) >= max_per_parent:
            continue
        out.append(hit)
        counts[key] = counts.get(key, 0) + 1
        if len(out) >= limit:
            break
    return out


def search_products(
    client_id: str,
    domain: str,
    query_vector: list[float],
    limit: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    only_in_stock: bool = False,
    content_types: Optional[list[str]] = None,
    store_code: Optional[str] = None,
    hybrid: bool = False,
    sparse_query_vector: Optional[SparseVector] = None,
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    # Phase 2.2: hybrid + sparse args passed through to search_content
    # so the products endpoint participates in BM25 + dense fusion when
    # the admin toggle is on.
    # Phase 2.3: with_vectors plumbs the dense-vector return through so
    # the retrieve handler can run MMR diversification on the candidate
    # pool before the LLM reranker sees it.
    return search_content(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=limit,
        min_price=min_price,
        max_price=max_price,
        only_in_stock=only_in_stock,
        content_types=content_types or ["product"],
        store_code=store_code,
        hybrid=hybrid,
        sparse_query_vector=sparse_query_vector,
        with_vectors=with_vectors,
    )


def upsert_content_item(
    client_id: str,
    domain: str,
    content_type: str,
    entity_id: str,
    vector: list[float],
    payload: dict[str, Any],
    store_code: Optional[str] = None,
    sparse_vector: Optional[SparseVector] = None,
) -> None:
    """Upsert a single non-chunked entity into the per-tenant collection.

    Phase 2.2: collections now use named vectors. `vector` lands in the
    `dense` slot; `sparse_vector` (when provided) lands in `sparse_bm25`.
    Callers that don't pass sparse get a dense-only point — search still
    works because the hybrid toggle decides whether the sparse slot is
    queried, not whether it's populated. (Skipping sparse on individual
    points just means those points contribute zero to the BM25 ranking
    for any query — they'd still be reachable via the dense prefetch.)
    """
    collection_name = ensure_collection_exists(client_id, domain)
    # store_code is part of the point id when present so per-store-view
    # variants of the same entity get separate points (different
    # embeddings, different localised text). See build_point_id.
    point_id = build_point_id(client_id, content_type, entity_id, store_code)
    type_key = _type_specific_id_key(content_type)

    normalized_payload = {
        **payload,
        "client_id": client_id,
        "content_type": content_type,
        "entity_id": str(entity_id),
        type_key: str(entity_id),
    }
    # Mirror store_code onto the payload too so the filter at retrieval
    # time can match on it. The Magento side already passes store_code
    # via SyncItem.store_code; this defaults it onto the payload if the
    # caller didn't explicitly include it.
    if store_code and "store_code" not in normalized_payload:
        normalized_payload["store_code"] = store_code

    if content_type == "product":
        normalized_payload.setdefault("product_id", str(entity_id))

    # Named-vector dict: keys MUST match the names declared in
    # ensure_collection_exists or Qdrant rejects the upsert.
    vector_dict: dict[str, Any] = {DENSE_VECTOR_NAME: vector}
    if sparse_vector is not None:
        vector_dict[SPARSE_VECTOR_NAME] = sparse_vector

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=vector_dict, payload=normalized_payload)],
    )


def upsert_chunked_content_item(
    client_id: str,
    domain: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str],
    chunks: list[dict[str, Any]],
    base_payload: dict[str, Any],
) -> int:
    """Replace every existing point for this entity with the given chunks.

    Atomicity model (delete-first):

      1. Filter-delete by (client_id, content_type, entity_id, store_code).
         This wipes any legacy single-point variant AND any prior chunks
         from an earlier sync — including the case where the chunk count
         shrunk (5 chunks → 3), which a naive upsert-with-deterministic-ids
         would leave with two orphans.

      2. Upsert the N new chunks with deterministic ids
         (build_chunk_point_id). Subsequent re-syncs of the same content
         hit the same uuid5s, so concurrent reads see exactly one version
         of each chunk index.

      There is a brief window between (1) and (2) where the entity has no
      points indexed. We chose this over "upsert-then-clean-orphans"
      because the alternative would produce DUPLICATE retrieval hits
      during the window (worse than briefly missing the entity).

    Each chunk dict must carry {vector: list[float], content: str,
    chunk_index: int}. base_payload supplies the shared per-entity fields
    (title, identifier, permalink, summary, …). Returns the number of
    points upserted.
    """
    if not chunks:
        # Defensive: chunker always returns >= 1 chunk, but if a caller
        # somehow passes an empty list we still want to clear any stale
        # points rather than leave them.
        delete_content_items_by_entity(client_id, domain, content_type, entity_id, store_code)
        return 0

    collection_name = ensure_collection_exists(client_id, domain)
    total_chunks = len(chunks)
    type_key = _type_specific_id_key(content_type)

    # Step 1 — bulk-delete every point currently associated with this entity.
    # Bypasses build_point_id entirely (filter on payload fields, not point
    # ids), so it catches legacy single-point variants whose id used a
    # different naming scheme.
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=_build_entity_filter(client_id, content_type, entity_id, store_code)
        ),
    )

    # Step 2 — build + upsert the N new chunk points.
    #
    # Phase 2.2: each chunk's vector is now a named dict {dense, sparse_bm25}.
    # The sparse component is optional per-chunk — if the caller doesn't
    # provide one, only dense is written (point still reachable via the
    # dense prefetch in hybrid search). In practice the sync router
    # always provides both, but keeping the field optional means tests
    # and one-off scripts don't have to fake a sparse vector.
    points: list[PointStruct] = []
    for chunk in chunks:
        idx = int(chunk["chunk_index"])
        chunk_payload = {
            **base_payload,
            "client_id":        client_id,
            "content_type":     content_type,
            "entity_id":        str(entity_id),
            type_key:           str(entity_id),
            "parent_entity_id": str(entity_id),
            "chunk_index":      idx,
            "total_chunks":     total_chunks,
            # Per-chunk fields. Content is the chunk body only — see the
            # chunkable formatter for why we don't carry the full page
            # text on every chunk.
            "content":          str(chunk.get("content") or ""),
            "embedded_text":    str(chunk.get("embedded_text") or ""),
        }
        if store_code and "store_code" not in chunk_payload:
            chunk_payload["store_code"] = store_code

        chunk_vector: dict[str, Any] = {DENSE_VECTOR_NAME: chunk["vector"]}
        if chunk.get("sparse_vector") is not None:
            chunk_vector[SPARSE_VECTOR_NAME] = chunk["sparse_vector"]

        points.append(
            PointStruct(
                id=build_chunk_point_id(client_id, content_type, entity_id, store_code, idx),
                vector=chunk_vector,
                payload=chunk_payload,
            )
        )

    qdrant.upsert(collection_name=collection_name, points=points)
    return len(points)


def delete_content_items_by_entity(
    client_id: str,
    domain: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str] = None,
) -> None:
    """Filter-based bulk delete — wipes every point for (entity, store).

    Used by:
      * upsert_chunked_content_item — clear before re-insert
      * delete_content_item — when content_type is chunkable
      * the /sync/delete handler — when content_type is chunkable

    Idempotent: if no matching points exist (entity already gone) the
    delete is a no-op. Filter-based so legacy single-point variants
    AND new chunks are both removed in one call.
    """
    if not _collection_exists(client_id, domain):
        return
    collection_name = get_collection_name(client_id, domain)
    qdrant.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=_build_entity_filter(client_id, content_type, entity_id, store_code)
        ),
    )


def upsert_product(client_id: str, domain: str, product_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    upsert_content_item(client_id, domain, "product", product_id, vector, payload)


def upsert_page(client_id: str, domain: str, page_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    upsert_content_item(client_id, domain, "page", page_id, vector, payload)


def upsert_post(client_id: str, domain: str, post_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    upsert_content_item(client_id, domain, "post", post_id, vector, payload)


def delete_content_item(
    client_id: str,
    domain: str,
    content_type: str,
    entity_id: str,
    store_code: Optional[str] = None,
) -> None:
    """Delete a single point by (client, content_type, entity_id) — and,
    when given, also scoped by store_code so the per-store variant is
    targeted specifically. When store_code is omitted the legacy point id
    is used so callers that delete from a single-store install before
    multi-store sync was deployed continue to work.

    For sites running multi-store sync the delete observer should pass
    store_code explicitly (else the wrong-store-view variant survives).

    Chunkable types route through delete_content_items_by_entity instead
    so every chunk of the entity is removed in one filter-based call.
    Falling back to a by-id delete would only wipe one chunk's UUID and
    leave the other N-1 stranded.
    """
    if content_type in CHUNKABLE_CONTENT_TYPES:
        delete_content_items_by_entity(client_id, domain, content_type, entity_id, store_code)
        return
    collection_name = ensure_collection_exists(client_id, domain)
    point_id = build_point_id(client_id, content_type, entity_id, store_code)
    qdrant.delete(collection_name=collection_name, points_selector=[point_id])


def delete_product(client_id: str, domain: str, product_id: str) -> None:
    delete_content_item(client_id, domain, "product", product_id)


def delete_page(client_id: str, domain: str, page_id: str) -> None:
    delete_content_item(client_id, domain, "page", page_id)


def delete_post(client_id: str, domain: str, post_id: str) -> None:
    delete_content_item(client_id, domain, "post", post_id)


def count_content_type(client_id: str, domain: str, content_type: str) -> int:
    """Count how many *logical entities* of *content_type* are indexed.

    For non-chunkable types (product, store_config, promotion, widget) this
    is just a raw point count — one point per entity.

    For chunkable types (cms_page, cms_block) one logical entity expands to
    N Qdrant points after the chunking refactor. The admin KPI strip needs
    to show entity-count ("50 pages indexed"), not point-count ("250 chunks"),
    so we restrict the filter to:

        chunk_index = 0   ← the first chunk of every new chunked entity
      OR chunk_index missing  ← legacy single-point entities synced before
                                 the chunking deploy

    Both conditions are needed during the rollout window: stores that
    haven't run a full re-sync since the chunking deploy still have the
    legacy points without a `chunk_index` field. Counting them too keeps
    the KPI honest for mixed-state collections.
    """
    if not _collection_exists(client_id, domain):
        return 0

    collection_name = get_collection_name(client_id, domain)

    if content_type in CHUNKABLE_CONTENT_TYPES:
        count_filter = Filter(
            must=[
                FieldCondition(key="content_type", match=MatchValue(value=content_type)),
            ],
            # Qdrant semantics: `must AND (any of should)`. So this resolves
            # to: content_type matches AND (chunk_index == 0 OR chunk_index
            # is empty/missing). IsEmptyCondition matches null, missing
            # field, and empty array — exactly the legacy-point shape.
            should=[
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                IsEmptyCondition(is_empty=PayloadField(key="chunk_index")),
            ],
        )
    else:
        count_filter = _build_content_filter(content_types=[content_type])

    result = qdrant.count(
        collection_name=collection_name,
        count_filter=count_filter,
        exact=True,
    )
    return int(result.count or 0)


def get_client_content_counts(
    client_id: str,
    domain: str,
    content_types: Optional[list[str]] = None,
) -> dict[str, int]:
    if not _collection_exists(client_id, domain):
        types = content_types or KNOWN_CONTENT_TYPES
        return {content_type: 0 for content_type in types}

    resolved_types = content_types or KNOWN_CONTENT_TYPES
    return {
        content_type: count_content_type(client_id, domain, content_type)
        for content_type in resolved_types
    }


def get_client_product_count(client_id: str, domain: str) -> int:
    return count_content_type(client_id, domain, "product")


def get_total_collection_count(client_id: str, domain: str) -> int:
    if not _collection_exists(client_id, domain):
        return 0

    collection_name = get_collection_name(client_id, domain)
    result = qdrant.count(collection_name=collection_name, exact=True)
    return int(result.count or 0)


def delete_client_collection(client_id: str, domain: str) -> bool:
    collection_name = get_collection_name(client_id, domain)
    existing = {collection.name for collection in qdrant.get_collections().collections}
    if collection_name not in existing:
        return False

    qdrant.delete_collection(collection_name=collection_name)
    return True