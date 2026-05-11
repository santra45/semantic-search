import re
import uuid
from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
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


def get_collection_name(client_id: str, domain: str) -> str:
    client_safe = re.sub(r"[^a-zA-Z0-9]", "_", client_id)
    domain_safe = re.sub(r"[^a-zA-Z0-9]", "_", domain)
    return f"products_{domain_safe}_{client_safe}"


def ensure_collection_exists(client_id: str, domain: str) -> str:
    collection_name = get_collection_name(client_id, domain)
    existing = {collection.name for collection in qdrant.get_collections().collections}
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
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
) -> list[dict[str, Any]]:
    collection_name = ensure_collection_exists(client_id, domain)
    query_filter = _build_content_filter(
        content_types=content_types,
        min_price=min_price,
        max_price=max_price,
        only_in_stock=only_in_stock,
        store_code=store_code,
    )

    result = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    return [_format_hit(hit) for hit in result.points]


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
) -> list[dict[str, Any]]:
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
    )


def upsert_content_item(
    client_id: str,
    domain: str,
    content_type: str,
    entity_id: str,
    vector: list[float],
    payload: dict[str, Any],
    store_code: Optional[str] = None,
) -> None:
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

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=vector, payload=normalized_payload)],
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
    """
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
    if not _collection_exists(client_id, domain):
        return 0

    collection_name = get_collection_name(client_id, domain)
    result = qdrant.count(
        collection_name=collection_name,
        count_filter=_build_content_filter(content_types=[content_type]),
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