import hmac
import hashlib
import base64
import os
import json
import anyio
from fastapi import APIRouter, Request, HTTPException, Header, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.services.embedder import embed_document
from backend.app.services.qdrant_service import (
    upsert_product,
    delete_product,
    get_client_product_count,
    product_exists,
    retrieve_content_by_entity_ids,
)
from backend.app.services.cache_service import invalidate_client_results
from backend.app.services.license_service import increment_ingest_count, validate_license_key, get_client_license
from backend.app.services.database import get_db
from backend.app.services.product_service import build_product_text, extract_payload  # ← import
from backend.app.services.llm_key_service import decrypt_key
from backend.app.wordpress.services.product_formatter import LEGACY_MANAGED_BY, MANAGED_BY

router    = APIRouter()


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    expected = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature)

def _has_full_woo_payload(client_id: str, domain: str, product_id: str) -> bool:
    """Was this product's point written by the full WooCommerce formatter?

    Every WooCommerce writer resolves the same per-tenant collection from the
    same licence, and a product's point id is derived from
    (client_id, content_type, product_id) — so they all write to the SAME
    record. Both plugins' PHP syncs now produce identical bytes, so between
    those two the write order stopped mattering.

    This webhook is the exception, and it cannot be brought into line. It runs
    off WooCommerce's REST product object, which carries no variation detail,
    no merchant notes, and attribute display labels with no taxonomy code — so
    the attributes it does carry key the payload differently. There is no way
    to write this path that produces the same point.

    Overwriting the full record with this one is a silent downgrade: search
    keeps working, and "what sizes does this come in" starts answering "I don't
    have that information" with nothing logged anywhere to explain it. So when
    the marker is present, this webhook stands down and lets the plugin's own
    realtime sync — which posts the full payload — do the update instead.

    Fails OPEN. If the lookup errors, indexing something is better than
    indexing nothing: a store must never stop syncing because a Qdrant read
    hiccuped.
    """
    try:
        hits = retrieve_content_by_entity_ids(
            client_id=client_id,
            domain=domain,
            entity_ids=[product_id],
            content_types=["product"],
            limit=1,
        )
    except Exception as exc:
        print(f"⚠️  payload check failed for {product_id}: {exc} — proceeding")
        return False

    # Both values accepted: points written before the two formatters were
    # merged carry the old marker, and they are just as full as the new ones.
    return bool(hits) and str(hits[0].get("managed_by") or "") in (MANAGED_BY, LEGACY_MANAGED_BY)


def process_upsert(product: dict, action: str, client_id: str, db: Session, license_data: dict = None, llm_api_key_encrypted: str = None) -> dict:
    """
    Shared logic for created + updated webhooks.
    Both do the same thing — embed and upsert.
    """
    product_id = str(product["id"])

    if product.get("type") == "variation":
        return {"status": "skipped", "reason": "variation"}

    if product.get("status") != "publish":
        delete_product(client_id, product_id)
        invalidate_client_results(client_id)
        print(f"🗑️  Webhook [{action}]: removed product {product_id}")
        return {"status": "removed", "product_id": product_id}

    # CRITICAL: Check product limit before indexing
    if not license_data:
        license_data = get_client_license(db, client_id)
    
    current_count = get_client_product_count(client_id, license_data["domain"])

    exists = product_exists(client_id, license_data["domain"], product_id)

    # Stand down if the point already holds the full WooCommerce payload — see
    # the helper above. Checked before the embed so a store that stands down
    # isn't paying for embeddings it then throws away.
    if exists and _has_full_woo_payload(client_id, license_data["domain"], product_id):
        print(f"⏭️  Webhook [{action}]: {product_id} holds the full payload — not overwriting")
        return {"status": "skipped", "reason": "full_payload_present", "product_id": product_id}

    # Only block NEW products
    if not exists and current_count >= license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Product limit exceeded. Current: {current_count}, Limit: {license_data['product_limit']}"
        )
    # For updates, check if we're adding a new product or updating existing
    # If product doesn't exist in vector store, count it as new
    # if current_count >= license_data["product_limit"]:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Product limit exceeded. Current: {current_count}, Limit: {license_data['product_limit']}"
    #     )

    # Decrypt embedding API key if provided
    if llm_api_key_encrypted:
        try:
            print(f"Decrypting embedding API key for license {license_data['license_key']}")
            print(f"LLM API key encrypted: {llm_api_key_encrypted}")
            embedding_api_key = decrypt_key(llm_api_key_encrypted, license_data["license_key"])   
        except Exception as e:
            print(f"❌ Embedding API key decryption failed: {e}")
            embedding_api_key = None
    else:
        print(f"Embedding API key not provided, using default")
        embedding_api_key = None

    # Uses product_service — raw WooCommerce format with nested categories/tags/attributes
    text    = build_product_text(product)
    vector  = embed_document(text, embedding_api_key, client_id)
    payload = extract_payload(product)
    payload["embedded_text"] = text

    upsert_product(client_id, license_data["domain"], product_id, vector, payload)
    invalidate_client_results(client_id)
    if not exists:
        increment_ingest_count(db, client_id, count=1)
    print(f"✅ Webhook [{action}]: indexed {product_id} - {product.get('name')}")

    return {"status": action, "product_id": product_id}

async def parse_webhook_body(request: Request) -> tuple:
    """
    Read body once. Return (raw_bytes, parsed_json_or_none).
    Handles WooCommerce ping (form-encoded) and real webhook (JSON).
    """
    body         = await request.body()
    content_type = request.headers.get("content-type", "")

    # WooCommerce ping — not JSON, just acknowledge it
    if "application/json" not in content_type:
        return body, None

    if not body:
        return body, None

    try:
        return body, json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/webhook/product-created")
def product_created(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    llm_api_key: Optional[str] = Query(None),  # ← reads ?llm_api_key= from URL
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
     # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    # Ping request — just acknowledge
    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        return process_upsert(
            product=product,
            action="created",
            client_id=client_id,
            db=db,
            llm_api_key_encrypted=llm_api_key
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions to preserve status codes
        raise e
    except Exception as e:
        print(f"❌ Webhook [created] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/product-updated")
def product_updated(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    llm_api_key: Optional[str] = Query(None),  # ← reads ?llm_api_key= from URL
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
    # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        return process_upsert(
            product=product,
            action="updated",
            client_id=client_id,
            db=db,
            llm_api_key_encrypted=llm_api_key
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions to preserve status codes
        raise e
    except Exception as e:
        print(f"❌ Webhook [updated] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/product-deleted")
def product_deleted(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
    # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    product_id = str(product.get("id", ""))
    if not product_id:
        return {"status": "skipped", "reason": "no product id"}

    try:
        # Get license data to retrieve domain
        license_data = get_client_license(db, client_id)
        delete_product(client_id, license_data["domain"], product_id)
        invalidate_client_results(client_id)
        print(f"🗑️  Webhook [deleted]: removed product {product_id}")
        return {"status": "deleted", "product_id": product_id}
    except Exception as e:
        print(f"❌ Webhook [deleted] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
