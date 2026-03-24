from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range,
    PointStruct, VectorParams, Distance
)
from backend.app.config import QDRANT_HOST, QDRANT_PORT, EMBED_DIM
import uuid
import re

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ─── Collection Naming ─────────────────────────────────────────────────────────

def get_collection_name(client_id: str) -> str:
    """
    Derive a safe Qdrant collection name from a client_id.
    Format: products_<sanitized_client_id>
    Only alphanumerics and underscores are kept; everything else becomes '_'.
    """
    safe = re.sub(r"[^a-zA-Z0-9]", "_", client_id)
    return f"products_{safe}"


# ─── Collection Bootstrap ──────────────────────────────────────────────────────

def ensure_collection_exists(client_id: str) -> str:
    """
    Create the client's Qdrant collection if it does not exist yet.
    Returns the collection name so callers can chain it.
    """
    coll = get_collection_name(client_id)
    existing = {c.name for c in qdrant.get_collections().collections}
    if coll not in existing:
        qdrant.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Created Qdrant collection: {coll}")
    return coll


# ─── Core Operations ───────────────────────────────────────────────────────────

def product_exists(client_id: str, product_id: str) -> bool:
    coll = ensure_collection_exists(client_id)
    point_uuid = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{client_id}-{product_id}"
    ))

    result = qdrant.retrieve(
        collection_name=coll,
        ids=[point_uuid],
        with_payload=False,
        with_vectors=False
    )

    return len(result) > 0


def search_products(
    client_id: str,
    query_vector: list,
    limit: int = 10,
    min_price: float = None,
    max_price: float = None,
    only_in_stock: bool = False
) -> list:

    coll = ensure_collection_exists(client_id)

    # Build dynamic filters (no client_id filter needed — collection IS the client)
    must_conditions = []

    # Add Price Range if provided
    if min_price is not None or max_price is not None:
        must_conditions.append(
            FieldCondition(
                key="price",
                range=Range(
                    gte=min_price,  # Greater than or equal
                    lte=max_price   # Less than or equal
                )
            )
        )

    # Add Stock Filter
    if only_in_stock:
        must_conditions.append(
            FieldCondition(key="stock_status", match=MatchValue(value="instock"))
        )

    # Execute Query
    query_filter = Filter(must=must_conditions) if must_conditions else None

    result = qdrant.query_points(
        collection_name=coll,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )

    # ─── Fixed fields always returned ──────────────────────────────────────────
    FIXED_KEYS = {
        "product_id", "name", "price", "permalink", "image_url",
        "stock_status", "categories", "client_id",
    }

    # Format results — fixed fields + tags + all dynamic attributes
    results = []
    for hit in result.points:
        p = hit.payload

        entry = {
            "product_id":   p.get("product_id"),
            "name":         p.get("name"),
            "price":        p.get("price"),
            "permalink":    p.get("permalink"),
            "image_url":    p.get("image_url"),
            "stock_status": p.get("stock_status"),
            "categories":   p.get("categories"),
            "tags":         p.get("tags", ""),   # ← needed for rerank
            "score":        round(hit.score, 4),
        }

        # Attach every dynamic attribute (gender, color, material, size…)
        for key, val in p.items():
            if key not in FIXED_KEYS and key not in entry:
                entry[key] = val

        results.append(entry)

    return results


def upsert_product(client_id: str, product_id: str, vector: list, payload: dict):
    coll = ensure_collection_exists(client_id)

    point_uuid = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{client_id}-{product_id}"
    ))

    payload["client_id"]  = client_id
    payload["product_id"] = str(product_id)

    qdrant.upsert(
        collection_name=coll,
        points=[
            PointStruct(
                id=point_uuid,
                vector=vector,
                payload=payload
            )
        ]
    )


def delete_product(client_id: str, product_id: str):
    coll = ensure_collection_exists(client_id)

    point_uuid = str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{client_id}-{product_id}"
    ))

    qdrant.delete(
        collection_name=coll,
        points_selector=[point_uuid]
    )


def get_client_product_count(client_id: str) -> int:
    """Count how many products are indexed for a client."""
    coll = get_collection_name(client_id)
    existing = {c.name for c in qdrant.get_collections().collections}
    if coll not in existing:
        return 0  # Collection doesn't exist yet → 0 products

    result = qdrant.count(collection_name=coll)
    return result.count


# ─── Admin / Lifecycle ─────────────────────────────────────────────────────────

def delete_client_collection(client_id: str) -> bool:
    """
    Permanently delete the entire Qdrant collection for a client.
    Use with caution — all vectors and payloads are destroyed.
    Returns True if deleted, False if the collection did not exist.
    """
    coll = get_collection_name(client_id)
    existing = {c.name for c in qdrant.get_collections().collections}
    if coll not in existing:
        return False

    qdrant.delete_collection(collection_name=coll)
    print(f"🗑️  Deleted Qdrant collection: {coll}")
    return True