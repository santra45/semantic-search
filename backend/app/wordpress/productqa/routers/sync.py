"""Ingest endpoints for the WooCommerce product Q&A plugin.

  POST /api/wordpress/productqa/sync/batch    — upsert a batch of items
  POST /api/wordpress/productqa/sync/delete   — remove single items
  GET  /api/wordpress/productqa/sync/status   — indexed counts
  POST /api/wordpress/productqa/sync/purge    — drop one content_type

Four content types: `product`, `faq`, and the store's own site content —
`page` and `post`. Products are single points; the other three chunk, because
a merchant's returns policy is routinely long enough that a single vector lets
the opening sentence dominate and buries the clause that actually answers the
question.

Everything writes into the same per-tenant collection the rest of the platform
uses (`products_{domain}_{client_id}`), with the same point-id scheme. That is
deliberate — a merchant's license is the tenant boundary, not the plugin.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.services.cache_service import invalidate_client_results, r as redis_client
from backend.app.services.database import get_db
from backend.app.services.embedder import embed_document
from backend.app.services.license_service import increment_ingest_count
from backend.app.services.qdrant_service import (
    CHUNKABLE_CONTENT_TYPES,
    delete_by_content_type,
    delete_content_item,
    get_client_content_counts,
    get_client_product_count,
    upsert_chunked_content_item,
    upsert_content_item,
)
from backend.app.services.sparse_embedder import embed_sparse_document

# These two are platform-agnostic — chunk_text(text, size, overlap) and
# merge_*(db, client_id, store_code, values) contain nothing Magento-specific.
# They live under that package for historical reasons only (same as
# llm_factory, imported by the retrieve router). Importing them is reuse of
# shared infrastructure, not coupling to Magento; if this ever needs to change
# for WordPress, the right move is to promote them to backend.app.services
# rather than fork a copy here.
from backend.app.magento.chatbot.services.text_chunker import chunk_text
from backend.app.magento.chatbot.services import vocab_service

from backend.app.wordpress.productqa.services.common import authorize_request, decrypt_llm_key

# The product formatter lives one level up, under app/wordpress/services/,
# because it is shared with the search plugin's sync endpoint — both write to
# the same Qdrant point for the same product, so both must produce the same
# bytes. See that module's header.
from backend.app.wordpress.services.product_formatter import (
    SITE_CONTENT_TYPES,
    SUPPORTED_TYPES,
    build_product_point,
    format_faq_chunkable,
    format_item,
    format_site_content_chunkable,
)

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_STORE_CODE = "default"

# How long an identical re-post of the same item is treated as already done.
# Long enough to absorb WordPress firing `save_post` several times for one
# editor click (WooCommerce saves the product, then its variations, then meta),
# short enough that a real second edit ten seconds later still goes through.
_DEDUP_TTL_SECONDS = 10


# ── Schemas ──────────────────────────────────────────────────────────────────


class SyncItem(BaseModel):
    entity_id: str
    content_type: str
    store_code: str = DEFAULT_STORE_CODE
    # No rigid schema: the plugin decides what to send per content type and the
    # formatter reads what it finds. Locking this down would mean a schema
    # migration every time a merchant's theme exposes a new product field.
    payload: dict[str, Any] = Field(default_factory=dict)


class SyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    items: list[SyncItem] = Field(default_factory=list)
    batch_number: int = 1
    total_batches: int = 1
    store_code: str = DEFAULT_STORE_CODE


class SyncDeleteItem(BaseModel):
    entity_id: str
    content_type: str
    store_code: Optional[str] = None


class SyncDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    items: list[SyncDeleteItem] = Field(default_factory=list)


class SyncPurgeRequest(BaseModel):
    license_key: Optional[str] = None
    content_type: str
    store_code: Optional[str] = None


# ── Dedup guard ──────────────────────────────────────────────────────────────


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    """Short stable hash of the payload.

    Hashes the WHOLE payload deliberately: if anything about the product
    actually changed the hash changes and the sync proceeds. Only byte-identical
    re-posts collapse.
    """
    try:
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(payload)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


def _claim_sync_slot(client_id: str, content_type: str, entity_id: str, fingerprint: str) -> bool:
    """True when this caller now owns the slot; False when the identical item
    was seen within the TTL.

    Fails OPEN on any Redis error — double-syncing costs an embedding call,
    silently dropping a real product update costs the merchant an answer that's
    wrong for as long as nobody notices.
    """
    key = f"wp_sync_dedup:{client_id}:{content_type}:{entity_id}:{fingerprint}"
    try:
        return bool(redis_client.set(key, "1", nx=True, ex=_DEDUP_TTL_SECONDS))
    except Exception as exc:
        logger.warning("wp sync dedup check failed (proceeding without): %s", exc)
        return True


# ── Batch ────────────────────────────────────────────────────────────────────


@router.post("/wordpress/productqa/sync/batch")
def sync_batch(
    req: SyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    # Set by full-sync runs. A full sync is authoritative: if the same product
    # turns up twice we WANT the second one embedded, because it reflects the
    # latest state. Counting it as "success, skipped" is how an index ends up
    # reporting more products than it holds.
    x_full_sync: Optional[str] = Header(None, alias="X-Full-Sync"),
    db: Session = Depends(get_db),
):
    bypass_dedup = (x_full_sync or "").strip().lower() in ("1", "true", "yes")

    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    # Quota is counted against products only. FAQ entries, pages and posts are
    # free: the plan limit is `get_client_product_count`, and folding a store's
    # own help content into that total would stop a merchant sitting near their
    # limit from indexing the delivery page that answers half their questions.
    incoming_products = sum(1 for item in req.items if item.content_type == "product")
    if incoming_products:
        current = get_client_product_count(license_data["client_id"], license_data["domain"])
        if current + incoming_products > license_data["product_limit"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Product limit exceeded. Current: {current}, "
                    f"Incoming: {incoming_products}, Limit: {license_data['product_limit']}"
                ),
            )

    embedding_api_key = decrypt_llm_key(
        x_llm_api_key_encrypted or req.llm_api_key_encrypted, license_data["license_key"]
    )

    attribute_vocab_sink: dict[str, set[str]] = defaultdict(set)
    category_vocab_sink: dict[str, dict[str, str]] = {}

    success_ids: list[str] = []
    failed_ids: list[str] = []
    success_by_type: dict[str, int] = defaultdict(int)

    for item in req.items:
        if item.content_type not in SUPPORTED_TYPES:
            failed_ids.append(item.entity_id)
            continue

        if not bypass_dedup:
            fingerprint = _payload_fingerprint(item.payload)
            if not _claim_sync_slot(
                license_data["client_id"], item.content_type, item.entity_id, fingerprint
            ):
                logger.info(
                    "wp sync_batch dedup-skipped %s/%s (fingerprint %s) — identical "
                    "re-post within %ds. Set X-Full-Sync: 1 to bypass.",
                    item.content_type, item.entity_id, fingerprint, _DEDUP_TTL_SECONDS,
                )
                success_ids.append(item.entity_id)
                success_by_type[item.content_type] += 1
                continue

        try:
            store_code = item.store_code or req.store_code or DEFAULT_STORE_CODE

            if item.content_type in CHUNKABLE_CONTENT_TYPES:
                _process_chunkable_item(item, store_code, embedding_api_key, license_data)
            else:
                if item.content_type == "product":
                    # Shared with routers/sync.py — the search plugin's endpoint
                    # assembles the identical point through the same call, which
                    # is what makes it safe for either plugin to write last.
                    text_for_embed, payload = build_product_point(
                        item.payload,
                        store_code=store_code,
                        attribute_vocab_sink=attribute_vocab_sink,
                        category_vocab_sink=category_vocab_sink,
                    )
                else:
                    text_for_embed, payload = format_item(item.content_type, item.payload)
                    payload["embedded_text"] = text_for_embed
                    payload["store_code"] = store_code

                vector = embed_document(text_for_embed, embedding_api_key, license_data["client_id"])
                try:
                    sparse_vector = embed_sparse_document(text_for_embed)
                except Exception as exc:
                    logger.warning(
                        "wp sparse embed failed for %s/%s: %s — proceeding dense-only",
                        item.content_type, item.entity_id, exc,
                    )
                    sparse_vector = None

                upsert_content_item(
                    client_id=license_data["client_id"],
                    domain=license_data["domain"],
                    content_type=item.content_type,
                    entity_id=item.entity_id,
                    vector=vector,
                    payload=payload,
                    store_code=store_code,
                    sparse_vector=sparse_vector,
                )

            # Counted per ITEM, not per point — a FAQ entry that chunks into
            # six points is still one thing the merchant's progress bar should
            # tick once for.
            success_ids.append(item.entity_id)
            success_by_type[item.content_type] += 1
        except Exception as exc:
            logger.warning(
                "wp sync failed for %s/%s: %s", item.content_type, item.entity_id, exc
            )
            failed_ids.append(item.entity_id)

    # Vocabulary merges are best-effort: they improve future query routing but
    # a failure here must not fail a batch that actually indexed.
    if attribute_vocab_sink:
        try:
            vocab_service.merge_attributes(
                db, license_data["client_id"], req.store_code, attribute_vocab_sink
            )
        except Exception:
            pass
    if category_vocab_sink:
        try:
            vocab_service.merge_categories(
                db, license_data["client_id"], req.store_code, category_vocab_sink
            )
        except Exception:
            pass

    if success_by_type.get("product"):
        increment_ingest_count(db, license_data["client_id"], count=success_by_type["product"])

    if req.batch_number >= req.total_batches:
        try:
            invalidate_client_results(license_data["client_id"])
        except Exception:
            pass

    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "by_type": dict(success_by_type),
        "batch_number": req.batch_number,
        "total_batches": req.total_batches,
        "is_last_batch": req.batch_number >= req.total_batches,
    }


def _process_chunkable_item(
    item: SyncItem,
    store_code: str,
    embedding_api_key: Optional[str],
    license_data: dict[str, Any],
) -> int:
    """Embed and upsert one FAQ entry, page or post as N chunks.

    The header (the FAQ's question heading, or the page's title and URL) is
    prepended to every chunk's embedding text. That repetition is the whole
    trick: chunk 4 of a long shipping policy still carries "FAQ: How long does
    delivery take" in its vector, so it's findable even though its own text
    never says "delivery".

    Each chunk's payload carries only ITS body, so when retrieval lands on
    chunk 4 the prompt gets that paragraph rather than the whole policy.
    """
    if item.content_type == "faq":
        header, body, base_payload = format_faq_chunkable(item.payload)
    elif item.content_type in SITE_CONTENT_TYPES:
        header, body, base_payload = format_site_content_chunkable(
            item.content_type, item.payload
        )
        # A page whose body cleans down to nothing — one built entirely out of
        # page-builder shortcodes that render to markup, say — has nothing to
        # embed. chunk_text returns [""] for empty input, so without this the
        # embedder would be handed an empty string and the failure would be
        # logged as an embedding error rather than as what it is.
        if not body.strip():
            raise ValueError(
                f"{item.content_type} {item.entity_id} has no readable text after cleaning"
            )
    else:
        raise ValueError(f"non-chunkable content_type routed to chunked path: {item.content_type}")

    base_payload["store_code"] = store_code

    chunk_records: list[dict[str, Any]] = []
    for idx, chunk_body in enumerate(
        chunk_text(body)
    ):
        embed_text = f"{header}\nContent: {chunk_body}" if header else chunk_body
        vector = embed_document(embed_text, embedding_api_key, license_data["client_id"])
        try:
            sparse_vector = embed_sparse_document(embed_text)
        except Exception as exc:
            logger.warning(
                "wp sparse embed failed for chunk faq/%s idx=%d: %s — dense-only",
                item.entity_id, idx, exc,
            )
            sparse_vector = None
        chunk_records.append({
            "vector": vector,
            "sparse_vector": sparse_vector,
            "content": chunk_body,
            "chunk_index": idx,
            "embedded_text": embed_text,
        })

    return upsert_chunked_content_item(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        content_type=item.content_type,
        entity_id=item.entity_id,
        store_code=store_code,
        chunks=chunk_records,
        base_payload=base_payload,
    )


# ── Delete / status / purge ──────────────────────────────────────────────────


@router.post("/wordpress/productqa/sync/delete")
def sync_delete(
    req: SyncDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Remove points for the given entities.

    Per-item failures are swallowed: this is called from the WordPress delete
    hook, and a product that was never indexed in the first place must not
    make deleting it look like it failed.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    deleted = 0
    for item in req.items:
        if item.content_type not in SUPPORTED_TYPES:
            continue
        try:
            delete_content_item(
                client_id=license_data["client_id"],
                domain=license_data["domain"],
                content_type=item.content_type,
                entity_id=item.entity_id,
                store_code=item.store_code,
            )
            deleted += 1
        except Exception as exc:
            logger.warning(
                "wp delete failed for %s/%s: %s", item.content_type, item.entity_id, exc
            )

    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass

    return {"deleted": deleted}


@router.get("/wordpress/productqa/sync/status")
def sync_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Indexed counts and plan headroom, for the plugin dashboard.

    Reports only the types this module writes. The tenant's collection may
    hold cms_page / category points written by another of the merchant's
    integrations; surfacing those here would make the WooCommerce dashboard
    claim credit for content it knows nothing about.

    Quota rides along on this response rather than living at its own endpoint
    so the plugin can answer "will this sync fit?" in one round trip, and so
    the WordPress side never has to call a Magento-named route to find out.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )

    counts = get_client_content_counts(
        license_data["client_id"],
        license_data["domain"],
        sorted(SUPPORTED_TYPES),
    )

    # Counts the WHOLE tenant's products, not just this plugin's — the plan
    # limit applies to the collection, and a merchant syncing from two places
    # needs to see the total they're actually consuming.
    product_count = get_client_product_count(license_data["client_id"], license_data["domain"])
    product_limit = int(license_data.get("product_limit") or 0)

    return {
        "counts": counts,
        "total_indexed": sum(int(c) for c in counts.values()),
        "quota": {
            "current_count": product_count,
            "product_limit": product_limit,
            "remaining": max(0, product_limit - product_count) if product_limit else None,
            "exceeded": bool(product_limit and product_count > product_limit),
        },
    }


@router.post("/wordpress/productqa/sync/purge")
def sync_purge_content_type(
    req: SyncPurgeRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Drop every point of one content_type.

    Exists for the FAQ replace-in-place flow: FAQ entity ids are positional,
    so deleting the third of five entries would otherwise leave a stale point
    #5 orphaned forever. Purge-then-write is the only way to make renames,
    reorders and deletions all correct.

    Idempotent — purging an empty type is a no-op, not an error.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    content_type = (req.content_type or "").strip().lower()
    if content_type not in SUPPORTED_TYPES:
        return {
            "success": False,
            "deleted_count": 0,
            "message": (
                f"Unknown content_type '{content_type}'. "
                f"Supported: {sorted(SUPPORTED_TYPES)}"
            ),
        }

    try:
        deleted = delete_by_content_type(
            client_id=license_data["client_id"],
            domain=license_data["domain"],
            content_type=content_type,
            store_code=req.store_code,
        )
    except Exception as exc:
        logger.exception("[wp sync/purge] failed for %s: %s", content_type, exc)
        return {"success": False, "deleted_count": 0, "message": f"Purge failed: {exc}"}

    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass

    return {
        "success": True,
        "deleted_count": int(deleted),
        "message": f"Purged ~{deleted} {content_type} points.",
    }
