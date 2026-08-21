"""
Mixed-content sync for the agent chatbot.

  POST   /api/magento/chatbot/agent/sync/batch
  POST   /api/magento/chatbot/agent/sync/delete
  GET    /api/magento/chatbot/agent/sync/status

A single batch can carry a mix of `content_type` values (product, cms_page,
cms_block, widget, store_config). Each item is formatted via product_formatter,
embedded via the existing Gemini embedder, and upserted into the per-tenant
Qdrant collection. Attribute and category vocabularies are merged per-store.
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
    delete_client_collection,
    delete_content_item,
    get_client_content_counts,
    get_client_product_count,
    upsert_chunked_content_item,
    upsert_content_item,
)
# Phase 2.2 — BM25 sparse vectors written alongside the existing dense
# Gemini vectors so hybrid search has something to fuse with on the read
# side. Generation is ~5 ms CPU per text, no per-call $ cost. Imported
# at module level (and not inside the loop) so the BM25 model loads
# exactly once on first sync rather than once per item.
from backend.app.services.sparse_embedder import embed_sparse_document

from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
    maybe_persist_magento_creds,
)
from backend.app.magento.chatbot.services import vocab_service
from backend.app.magento.chatbot.services.product_formatter import (
    format_category_chunkable,
    format_cms_block_chunkable,
    format_cms_page_chunkable,
    format_faq_chunkable,
    format_item,
)
from backend.app.magento.chatbot.services.text_chunker import chunk_text

# How big each chunk's *body* gets, and how much context is carried across
# adjacent chunk boundaries. 500 chars (~80-100 words) lets a single chunk
# semantically centre on one paragraph or short section. 200-char overlap
# preserves cross-paragraph context for queries that straddle a section
# break.
_CHUNK_TARGET_SIZE = 500
_CHUNK_OVERLAP     = 200

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_TYPES = {"product", "cms_page", "cms_block", "category", "widget", "store_config", "promotion", "faq"}

# Window for the "did we just embed this exact payload?" guard. Long enough
# to absorb any near-simultaneous fires from multiple Magento modules sharing
# a license key (observers, crons, manual sync). Short enough that a real
# admin edit a few seconds later still goes through.
_DEDUP_TTL_SECONDS = 10


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    """Stable short hash of the payload for dedup keys.

    Deliberately hash the full payload — if the actual content changed, the
    hash changes, the new sync goes through. Only IDENTICAL re-posts collapse.
    """
    try:
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        blob = repr(payload)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]

def _repair_mojibake_str(s: str) -> str:
    """Reverse a cp1252/Latin-1 misdecode of UTF-8 bytes (e.g. ℃ → „ƒ).

    Strict round-trip: if the string doesn't cleanly re-encode as cp1252
    and re-decode as UTF-8, it's either already clean or not this
    corruption pattern, so we return it untouched. This makes the repair
    a no-op on well-formed UTF-8 (accented brand names, real °C glyphs).
    """
    if not s:
        return s
    try:
        return s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _repair_payload_mojibake(value: Any) -> Any:
    """Recursively repair mojibake in every string within a payload
    (nested dicts and lists included). Non-string scalars pass through.
    """
    if isinstance(value, str):
        return _repair_mojibake_str(value)
    if isinstance(value, dict):
        return {k: _repair_payload_mojibake(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_repair_payload_mojibake(v) for v in value]
    return value


def _claim_sync_slot(client_id: str, content_type: str, entity_id: str, fingerprint: str) -> bool:
    """Atomic 'is this sync already in flight or just done?' check via Redis SETNX.

    Returns True if THIS caller now owns the slot (proceed with embed/upsert).
    Returns False if the same (client, content_type, entity, payload-hash) was
    seen in the last _DEDUP_TTL_SECONDS — caller should skip the embed and
    treat the item as already-synced.

    Falls open on Redis failure: if the cache is down, we'd rather double-sync
    than silently lose a real update.
    """
    key = f"sync_dedup:{client_id}:{content_type}:{entity_id}:{fingerprint}"
    try:
        # nx=True → only set if absent. Returns True on success.
        return bool(redis_client.set(key, "1", nx=True, ex=_DEDUP_TTL_SECONDS))
    except Exception as exc:
        logger.warning("sync dedup check failed (proceeding without): %s", exc)
        return True


class SyncItem(BaseModel):
    entity_id: str
    content_type: str
    store_code: str = "default"
    # Bag of raw fields the formatter will inspect (no rigid schema — the Magento module
    # defines what to send per content type; see Model/Content/*ContentProvider.php).
    payload: dict[str, Any] = Field(default_factory=dict)


class SyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    items: list[SyncItem] = Field(default_factory=list)
    batch_number: int = 1
    total_batches: int = 1
    store_code: str = "default"


class SyncDeleteItem(BaseModel):
    entity_id: str
    content_type: str
    # Optional: when present the per-store variant is targeted; when
    # absent the legacy single-store point is targeted (back-compat with
    # old callers that don't know about per-store sync yet).
    store_code: Optional[str] = None


class SyncDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    items: list[SyncDeleteItem] = Field(default_factory=list)


def _process_chunkable_item(
    item: "SyncItem",
    store_code: str,
    embedding_api_key: Optional[str],
    license_data: dict[str, Any],
) -> int:
    """Embed + upsert one CMS-style item as N chunks.

    Returns the number of chunks written. Raises on embedding failure
    (caught by the outer loop's per-item try/except so one bad page
    doesn't tank the whole batch).

    The chunkable formatter splits the item into (header, body, base_payload):
      - header  — title + metadata + factual anchors, repeated on every
                  chunk's embedding so the vector stays grounded in the
                  parent page.
      - body    — the raw content, what gets sliced into chunks.
      - base_payload — per-page metadata (title, identifier, permalink, …)
                       copied onto every chunk's payload.

    Each chunk's `content` in payload is just THAT chunk's body — when
    retrieval matches chunk 3 of a long policy, the LLM sees that
    paragraph specifically, not the whole page. The page-level title /
    permalink / summary still ride along on every chunk so the chat card
    UI and citation strip can identify the parent page.
    """
    if item.content_type == "cms_page":
        header, body, base_payload = format_cms_page_chunkable(item.payload)
    elif item.content_type == "cms_block":
        header, body, base_payload = format_cms_block_chunkable(item.payload)
    elif item.content_type == "category":
        header, body, base_payload = format_category_chunkable(item.payload)
    elif item.content_type == "faq":
        header, body, base_payload = format_faq_chunkable(item.payload)
    else:
        # Defensive: outer loop should only route chunkable types here.
        raise ValueError(f"non-chunkable content_type routed to chunked path: {item.content_type}")

    base_payload["store_code"] = store_code

    body_chunks = chunk_text(body, target_size=_CHUNK_TARGET_SIZE, overlap=_CHUNK_OVERLAP)

    chunk_records: list[dict[str, Any]] = []
    for idx, chunk_body in enumerate(body_chunks):
        # Embed text = header + chunk body. Header repetition is the
        # whole point of the chunked formatter — keeps every chunk's
        # vector grounded in the page's title + meta + anchors.
        embed_text = f"{header}\nContent: {chunk_body}" if header else chunk_body
        vector = embed_document(embed_text, embedding_api_key, license_data["client_id"])
        # Phase 2.2 — sparse BM25 vector generated from the SAME embed
        # text the dense vector saw. Fast (~5 ms CPU) and free; cached
        # nothing because each chunk is unique. Failure is non-fatal —
        # if BM25 generation crashes we log and continue with dense-
        # only for this chunk (point still reachable, just no BM25
        # contribution at search time).
        try:
            sparse_vector = embed_sparse_document(embed_text)
        except Exception as exc:
            logger.warning(
                "sparse embed failed for chunk %s/%s idx=%d: %s — proceeding dense-only",
                item.content_type, item.entity_id, idx, exc,
            )
            sparse_vector = None
        chunk_records.append({
            "vector":        vector,
            "sparse_vector": sparse_vector,
            "content":       chunk_body,
            "chunk_index":   idx,
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


@router.post("/magento/chatbot/agent/sync/batch")
def sync_batch(
    req: SyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    x_magento_creds: Optional[str] = Header(None, alias="X-Magento-Admin-Creds-Encrypted"),
    # When the Magento side sets this header (truthy value), skip the
    # Redis dedup guard. Full-sync batches are authoritative — if we get
    # the same product twice in a sync run, we WANT it embedded twice
    # (the second one wins and reflects the latest state), not "counted
    # as success but silently skipped". The dedup safety-net is only
    # useful for the realtime observer path where multiple Magento
    # modules might fire on the same product save in quick succession.
    x_full_sync: Optional[str] = Header(None, alias="X-Full-Sync"),
    db: Session = Depends(get_db),
):
    bypass_dedup = (x_full_sync or "").strip().lower() in ("1", "true", "yes")
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

    # Quota check against the *product* limit — non-product content is free.
    incoming_products = sum(1 for it in req.items if it.content_type == "product")
    if incoming_products:
        current = get_client_product_count(license_data["client_id"], license_data["domain"])
        if current + incoming_products > license_data["product_limit"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Product limit exceeded. Current: {current}, Incoming: {incoming_products}, "
                    f"Limit: {license_data['product_limit']}"
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

        # Safety net for the case where two Magento modules both fire on the
        # same product save under the same license. The PHP-side ownership
        # helper covers the normal case; this catches the race window during
        # deploys and any future custom integrations posting the same item.
        #
        # Bypassed entirely during full sync (X-Full-Sync header set) —
        # full-sync batches are authoritative and a duplicate should be
        # re-embedded with the latest state, not counted-as-success-and-
        # skipped. The dedup-as-success behaviour was the source of the
        # demo-day "Magento says 852 processed but Qdrant has 285" mystery.
        if not bypass_dedup:
            fingerprint = _payload_fingerprint(item.payload)
            if not _claim_sync_slot(
                license_data["client_id"], item.content_type, item.entity_id, fingerprint
            ):
                logger.info(
                    "sync_batch dedup-skipped %s/%s (fingerprint %s) — "
                    "duplicate post within %ds. Set X-Full-Sync: 1 to bypass.",
                    item.content_type, item.entity_id, fingerprint, _DEDUP_TTL_SECONDS,
                )
                success_ids.append(item.entity_id)
                success_by_type[item.content_type] += 1
                continue

        try:
            # store_code stamped onto payload + passed to upsert. When the
            # Magento side runs per-store sync each batch carries the
            # specific store_code; we use it both as the payload tag (so
            # retrieve can filter) AND as part of the point id (so two
            # store-views of the same entity don't overwrite each other).
            store_code = item.store_code or req.store_code
            
            # Repair upstream UTF-8/cp1252 mojibake (e.g. ℃ → „ƒ) before
            # formatting + embedding, so neither the stored payload nor
            # the dense/sparse vectors carry corrupted text. Runs once
            # over the raw payload for both chunkable and single-point
            # paths. No-op on already-clean UTF-8.
            item.payload = _repair_payload_mojibake(item.payload)

            if item.content_type in CHUNKABLE_CONTENT_TYPES:
                # Chunked path — N points per item. Each chunk's embedding
                # gets the header prepended so it stays grounded in the
                # parent page's title/identifier/anchors. upsert_chunked_
                # content_item filter-deletes first so old chunks (or
                # legacy single points pre-chunking) can't leave orphans.
                _process_chunkable_item(
                    item, store_code, embedding_api_key, license_data,
                )
            else:
                # Single-point path — unchanged from pre-chunking behaviour
                # so products / store_config / promotion / widget keep
                # their existing deterministic UUIDs and need no resync.
                #
                # Phase 2.2: also generate a sparse BM25 vector from the
                # same embed text so hybrid search has signal on these
                # types too. fastembed BM25 is ~5 ms CPU; soft-fails to
                # None on error and the point upserts dense-only.
                text_for_embed, payload = format_item(
                    item.content_type,
                    item.payload,
                    attribute_vocab_sink=attribute_vocab_sink if item.content_type == "product" else None,
                    category_vocab_sink=category_vocab_sink if item.content_type == "product" else None,
                )
                payload["embedded_text"] = text_for_embed
                payload["store_code"]    = store_code

                vector = embed_document(text_for_embed, embedding_api_key, license_data["client_id"])
                try:
                    sparse_vector = embed_sparse_document(text_for_embed)
                except Exception as exc:
                    logger.warning(
                        "sparse embed failed for %s/%s: %s — proceeding dense-only",
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
            # Item-level success — N chunks count as one item processed so
            # the Magento progress bar (items shipped vs items acked)
            # stays sane. Per-point counts would over-report by 5-10x for
            # chunked content.
            success_ids.append(item.entity_id)
            success_by_type[item.content_type] += 1
        except Exception:
            failed_ids.append(item.entity_id)

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


@router.post("/magento/chatbot/agent/sync/delete")
def sync_delete(
    req: SyncDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
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
        except Exception:
            pass

    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass
    return {"deleted": deleted}


@router.get("/magento/chatbot/agent/sync/status")
def sync_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    counts = get_client_content_counts(
        license_data["client_id"],
        license_data["domain"],
        list(SUPPORTED_TYPES),
    )
    return {
        "counts": counts,
        "total_indexed": sum(int(c) for c in counts.values()),
    }


# ── Purge / reset endpoints ──────────────────────────────────────────────────
#
# Used by the admin "Manage Indexed Content" dashboard section. Two flavors:
#
#   * /sync/purge            — delete every point of a given content_type
#                              from the tenant's collection (optionally
#                              scoped to a store_code). Used when the
#                              merchant wants to drop a stale slice of
#                              the index after changing sync settings —
#                              e.g. after switching product-sync-scope
#                              from ALL → VISIBLE_IN_STOCK, the
#                              out-of-stock products synced earlier need
#                              explicit removal.
#
#   * /sync/purge/collection — drop the whole per-tenant collection.
#                              Next sync recreates it from scratch with
#                              the current named-vector schema (handy when
#                              upgrading the schema or recovering from a
#                              corrupt index).
#
# Both authorize via the standard license_key + x-api-key headers — same
# auth posture as every other sync endpoint. Idempotent: purging an
# already-empty collection / content_type is a no-op.


class SyncPurgeRequest(BaseModel):
    """Selective purge of one content_type. store_code optional — when
    set, only the per-store-variant points are removed; when absent,
    every point of that content_type across all stores in the
    collection is removed."""
    license_key: Optional[str] = None
    content_type: str
    store_code: Optional[str] = None


@router.post("/magento/chatbot/agent/sync/purge")
def sync_purge_content_type(
    req: SyncPurgeRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
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
        logger.exception("[sync/purge] failed for %s: %s", content_type, exc)
        return {
            "success": False,
            "deleted_count": 0,
            "message": f"Purge failed: {exc}",
        }

    # Drop cached retrieval results — the just-purged content shouldn't
    # keep surfacing from Redis-cached answers in the next 10s window.
    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass

    scope_note = f" (store_code={req.store_code})" if req.store_code else ""
    return {
        "success":       True,
        "deleted_count": int(deleted),
        "message":       f"Purged ~{deleted} {content_type} points{scope_note}.",
    }


@router.post("/magento/chatbot/agent/sync/purge/collection")
def sync_purge_collection(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Drop the tenant's entire Qdrant collection. Destructive. Next
    sync recreates the collection from scratch via
    ensure_collection_exists, which is the right behaviour when the
    named-vector schema needs to be reset (e.g. after a Phase 2.2-style
    schema migration) OR when the merchant wants a clean slate.

    No body — the license_key from the auth header is the only input
    needed. Returns {success, dropped: bool, message}.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )

    try:
        dropped = delete_client_collection(
            client_id=license_data["client_id"],
            domain=license_data["domain"],
        )
    except Exception as exc:
        logger.exception("[sync/purge/collection] failed: %s", exc)
        return {
            "success": False,
            "dropped": False,
            "message": f"Collection drop failed: {exc}",
        }

    try:
        invalidate_client_results(license_data["client_id"])
    except Exception:
        pass

    if dropped:
        return {
            "success": True,
            "dropped": True,
            "message": "Collection dropped. Run a full sync to rebuild the index.",
        }
    return {
        "success": True,
        "dropped": False,
        "message": "Collection did not exist — nothing to drop.",
    }
