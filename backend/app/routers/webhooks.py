"""
WooCommerce webhook receiver for semantic-search-woo. DEPRECATED — this whole
module is decommissioned by the v2 schema. Do not build on it.

WHAT IS HERE AND WHAT FIRES IT
------------------------------
Three endpoints, all registered under /api by main.py, all POSTed by
WooCommerce ITSELF rather than by our plugin's PHP. SSW_Webhook_Manager
(semantic-search-woo/includes/class-webhook-manager.php) creates them through
the store's own WooCommerce REST API when the merchant saves a licence key,
and bakes the tenant identity and both provider keys into the delivery URL as
query parameters, because a WooCommerce webhook has no other way to carry
them:

  POST /api/webhook/product-created   topic product.created
  POST /api/webhook/product-updated   topic product.updated
  POST /api/webhook/product-deleted   topic product.deleted

Each one authenticates the same way: `?client_id=` names the tenant, and the
X-WC-Webhook-Signature header is HMAC-SHA256 of the raw body under a shared
secret the plugin generated and pushed to /api/register-webhook-secret, which
stored it on `clients.webhook_secret`. Note what that means — the caller
proves it knows the webhook secret, NOT that it holds a licence key. Nothing
on this path ever presents one.

WHERE THE PLAINTEXT LICENCE KEY COMES FROM, AND WHY THAT KILLS THIS FILE
-----------------------------------------------------------------------
Both provider keys arrive here encrypted, and llm_key_service.decrypt_key
derives its AES key as sha256(license_key) — so unwrapping them needs the
merchant's plaintext licence key. Since the caller is WooCommerce and not our
plugin, there is nothing on the wire to take it from, so this file goes and
reads it out of the database instead:

  * process_upsert()  -> get_client_license(db, client_id)["license_key"]
                         handed to resolve_embedding_key(). The KEK. This is
                         the dependency the whole decommission is about.
  * process_upsert()  -> the same dict's ["domain"] and ["product_limit"] for
                         the Qdrant collection and the catalogue ceiling.
  * product_deleted() -> get_client_license(db, client_id)["domain"], only to
                         name the collection.

schema_v2 stores `licences.key_hash` and no plaintext column at all, so after
the migration get_client_license has nothing to select and this path has no
way to obtain a KEK. It is not fixable from the server side: a hash cannot be
un-hashed, and no amount of rework here changes the fact that WooCommerce's
own webhook does not carry a credential we can derive one from.

WHAT REPLACES IT
----------------
The push method ai-product-qa-woo already uses: the PLUGIN hooks WordPress
actions itself and calls the API with `Authorization: Bearer <license_key>` on
every request (AIPQA_API_Client::build_headers, ai-product-qa-woo/includes/
class-api-client.php). The plaintext key is then always presented, the KEK is
always in scope, and the server resolves identity from the key it was handed
rather than from a client_id in a query string. semantic-search-woo already
has half of this — SSW_Sync::register_hooks() pushes product upserts through
/api/sync/batch — so the switch is finishing the job, not starting it.
docs/webhook-migration.md is the plan, file by file.

UNTIL THEN
----------
This module stays wired up because the deployed plugin still points at it. Add
nothing to it, and route no new work through it.
"""

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
from backend.app.services.embedding_key_service import (
    resolve_embedding_key,
    resolve_embedding_model,
)
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


def process_upsert(product: dict, action: str, client_id: str, db: Session, license_data: dict = None, llm_api_key_encrypted: str = None, embedding_api_key_encrypted: str = None, embedding_provider: str = None, embedding_model: str = None) -> dict:
    """
    Shared logic for created + updated webhooks.
    Both do the same thing — embed and upsert.
    """
    product_id = str(product["id"])

    if product.get("type") == "variation":
        return {"status": "skipped", "reason": "variation"}

    # Resolved BEFORE the unpublish branch, not after it. Removing a point
    # needs the domain that names the collection, and the domain only exists on
    # license_data — see the fix immediately below.
    #
    # get_client_license is DEPRECATED and this is the call that keeps it
    # alive: license_data["license_key"] is the plaintext key, which is the KEK
    # for the encrypted provider keys further down. Read the module header
    # before touching this.
    if not license_data:
        license_data = get_client_license(db, client_id)

    if product.get("status") != "publish":
        # qdrant_service.delete_product is (client_id, domain, product_id).
        # This was called with two arguments, so every unpublished product
        # raised TypeError, the endpoint's blanket `except Exception` turned it
        # into a 500, and the point was never removed — a product pulled out of
        # `publish` kept answering shoppers from the index indefinitely.
        #
        # Nothing else covered it: SSW_Sync::sync_single() in
        # semantic-search-woo deliberately stands down for a non-published
        # product on the stated grounds that "removal is the product-updated
        # webhook's job". It was not doing that job. The same two-argument call
        # is still live at routers/ingest.py and routers/magento.py, which are
        # somebody else's files.
        delete_product(client_id, license_data["domain"], product_id)
        invalidate_client_results(client_id)
        print(f"🗑️  Webhook [{action}]: removed product {product_id}")
        return {"status": "removed", "product_id": product_id}

    # CRITICAL: Check product limit before indexing
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

    # The merchant's embedding key if they configured one, else their LLM
    # key. The raw blobs are no longer printed — a webhook fires on every
    # product save, and this was writing an encrypted key into the log on
    # each one.
    #
    # THIS IS THE LINE THAT CANNOT BE PORTED. The third argument is the
    # merchant's plaintext licence key, read out of license_keys.license_key by
    # get_client_license above, and llm_key_service.decrypt_key turns it into
    # the AES key as sha256(license_key). v2 stores only a SHA-256 hash, so
    # there is no plaintext left to read and no way to derive the KEK on a
    # request that never presented a key. Every other caller of
    # resolve_embedding_key takes the key from the request — routers/sync.py
    # and routers/ingest.py from the body, the WordPress namespace from the
    # Authorization header — and that is what this path has to become.
    embedding_api_key = resolve_embedding_key(
        embedding_api_key_encrypted,
        llm_api_key_encrypted,
        license_data["license_key"],
    )
    embedding_model = resolve_embedding_model(embedding_model, embedding_provider)

    # Uses product_service — raw WooCommerce format with nested categories/tags/attributes
    text    = build_product_text(product)
    vector  = embed_document(text, embedding_api_key, client_id, model=embedding_model)
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
    embedding_api_key: Optional[str] = Query(None),  # ← reads ?embedding_api_key= from URL
    embedding_provider: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
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
            llm_api_key_encrypted=llm_api_key,
            embedding_api_key_encrypted=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
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
    embedding_api_key: Optional[str] = Query(None),  # ← reads ?embedding_api_key= from URL
    embedding_provider: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
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
            llm_api_key_encrypted=llm_api_key,
            embedding_api_key_encrypted=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
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
        # Get license data to retrieve domain. DEPRECATED call — this one wants
        # nothing but the domain, so under v2 it becomes
        # resolve_key(db, presented_key)["collection_name"] once the plugin
        # presents a key. It cannot simply be repointed at the new schema while
        # the caller is WooCommerce, because WooCommerce presents no key.
        license_data = get_client_license(db, client_id)
        delete_product(client_id, license_data["domain"], product_id)
        invalidate_client_results(client_id)
        print(f"🗑️  Webhook [deleted]: removed product {product_id}")
        return {"status": "deleted", "product_id": product_id}
    except Exception as e:
        print(f"❌ Webhook [deleted] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
