from fastapi import APIRouter, HTTPException, Depends, Request, Header, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
from backend.app.services.embedder import embed_document
from backend.app.services.qdrant_service import upsert_product, get_client_product_count
from backend.app.services.license_service import validate_license_key, increment_ingest_count, extract_license_key_from_authorization
from backend.app.services.database import get_db
from backend.app.services.cache_service import invalidate_client_results
from backend.app.services.product_service import build_product_text, extract_payload
import time
from urllib.parse import urlparse

router = APIRouter()


class SyncProduct(BaseModel):
    product_id:        str
    name:              str
    categories:        str = ""
    tags:              str = ""
    description:       str = ""
    short_description: str = ""
    price:             float = 0
    regular_price:     float = 0
    sale_price:        float = 0
    currency:          str = ""
    currency_symbol:   str = ""
    on_sale:           bool = False
    permalink:         str = ""
    image_url:         str = ""
    stock_status:      str = "instock"
    average_rating:    float = 0
    attributes:        list = Field(default_factory=list)


class SyncBatchRequest(BaseModel):
    license_key:   str
    products:      List[SyncProduct]
    batch_number:  int = 1
    total_batches: int = 1


class SyncBatchResponse(BaseModel):
    success_count: int
    failed_count:  int
    failed_ids:    List[str]
    batch_number:  int
    total_batches: int
    is_last_batch: bool


@router.post("/sync/batch", response_model=SyncBatchResponse)
def sync_batch(req: SyncBatchRequest, request: Request, db: Session = Depends(get_db)):
    try:
        license_data = validate_license_key(req.license_key, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    client_id   = license_data["client_id"]
    
    # CRITICAL: Enforce domain authorization
    origin = request.headers.get("origin") or request.headers.get("referer")
    allowed_domain = license_data.get("domain")

    if allowed_domain and origin:
        hostname = urlparse(origin).hostname

        if allowed_domain and hostname not in [allowed_domain, "127.0.0.1"]:
            raise HTTPException(
                status_code=403,
                detail=f"Domain not authorized. License valid for: {allowed_domain}"
            )
    
    # CRITICAL: Check total indexed count + incoming count against plan limit
    current_count = get_client_product_count(client_id)
    incoming_count = len(req.products)
    total_after_ingest = current_count + incoming_count
    
    if total_after_ingest > license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Product limit exceeded. Current: {current_count}, Incoming: {incoming_count}, Limit: {license_data['product_limit']}"
        )
    
    success_ids = []
    failed_ids  = []

    for product in req.products:
        try:
            p = product.model_dump()

            # product_service handles flat string format from plugin
            text    = build_product_text(p)

            if len(success_ids) > 0:
                time.sleep(0.5)

            vector  = embed_document(text)
            payload = extract_payload(p)
            payload["embedded_text"] = text

            upsert_product(client_id, product.product_id, vector, payload)
            success_ids.append(product.product_id)

        except Exception as e:
            print(f"❌ Sync failed for {product.product_id}: {e}")
            failed_ids.append(product.product_id)

    if success_ids:
        increment_ingest_count(db, client_id, count=len(success_ids))

    is_last_batch = req.batch_number >= req.total_batches
    if is_last_batch:
        invalidate_client_results(client_id)

    return SyncBatchResponse(
        success_count=len(success_ids),
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
        batch_number=req.batch_number,
        total_batches=req.total_batches,
        is_last_batch=is_last_batch
    )


@router.get("/sync/status")
def sync_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    token = extract_license_key_from_authorization(authorization) or license_key
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(token, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # CRITICAL: Enforce domain authorization
    origin = request.headers.get("origin") or request.headers.get("referer")
    allowed_domain = license_data.get("domain")

    if allowed_domain and origin:
        hostname = urlparse(origin).hostname

        if allowed_domain and hostname not in [allowed_domain, "127.0.0.1"]:
            raise HTTPException(
                status_code=403,
                detail=f"Domain not authorized. License valid for: {allowed_domain}"
            )
    
    count = get_client_product_count(license_data["client_id"])

    return {
        "client_id":     license_data["client_id"],
        "indexed_count": count,
        "plan":          license_data["plan"],
        "product_limit": license_data["product_limit"]
    }