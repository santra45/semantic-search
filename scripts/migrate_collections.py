from qdrant_client import QdrantClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.app.config import QDRANT_HOST, QDRANT_PORT
from backend.app.services.qdrant_service import (
    ensure_collection_exists,
    get_collection_name,
)

OLD_COLL = "local_shared_products_BGE"  # or whatever your QDRANT_COLLECTION was
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
offset = None
while True:
    records, offset = qdrant.scroll(
        collection_name=OLD_COLL,
        with_payload=True,
        with_vectors=True,
        limit=100,
        offset=offset,
    )
    if not records:
        break
    for r in records:
        client_id = r.payload.get("client_id")
        if not client_id:
            continue
        coll = ensure_collection_exists(client_id)
        from qdrant_client.models import PointStruct

        qdrant.upsert(
            collection_name=coll,
            points=[PointStruct(id=r.id, vector=r.vector, payload=r.payload)],
        )
    if offset is None:
        break
print("✅ Migration complete")
