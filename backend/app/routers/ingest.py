from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.orm import Session
from backend.app.services.embedder import embed_document
from backend.app.services.qdrant_service import upsert_product, delete_product, get_client_product_count
from backend.app.services.license_service import validate_license_key, increment_ingest_count
from backend.app.services.database import get_db
from backend.app.services.cache_service import invalidate_client_results
from backend.app.services.product_service import build_product_text, extract_payload
from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.embedding_key_service import (
    resolve_embedding_key,
    resolve_embedding_model,
)
from urllib.parse import urlparse

router = APIRouter()


class Product(BaseModel):
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
    attributes:        list = Field(default_factory=list)   # ← safe mutable default


class IngestRequest(BaseModel):
    license_key: str
    products:    List[Product]
    llm_api_key_encrypted: str = None
    # The tenant's separate embedding key, encrypted under the license key
    # exactly like llm_api_key_encrypted. Absent = fall back to the LLM key,
    # which is what every install predating the embedding config sends.
    embedding_api_key_encrypted: str = None
    embedding_provider: str = None
    embedding_model: str = None


class DeleteRequest(BaseModel):
    license_key: str
    product_id:  str


@router.post("/ingest")
def ingest(req: IngestRequest, request: Request, db: Session = Depends(get_db)):
    try:
        license_data = validate_license_key(req.license_key, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    client_id = license_data["client_id"]
    domain    = license_data["domain"]
    license_key = req.license_key

    # The embedding key if the merchant configured one, else the LLM key.
    embedding_api_key = resolve_embedding_key(
        req.embedding_api_key_encrypted,
        req.llm_api_key_encrypted,
        license_key,
    )
    embedding_model = resolve_embedding_model(req.embedding_model, req.embedding_provider)

    # CRITICAL: Enforce secure domain authorization
    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data)

    # CRITICAL: Check total indexed count + incoming count against plan limit
    current_count = get_client_product_count(client_id, domain)
    incoming_count = len(req.products)
    total_after_ingest = current_count + incoming_count
    
    if total_after_ingest > license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Product limit exceeded. Current: {current_count}, Incoming: {incoming_count}, Limit: {license_data['product_limit']}"
        )

    success = []
    failed  = []

    for product in req.products:
        try:
            p       = product.model_dump()
            text    = build_product_text(p)
            vector  = embed_document(text, embedding_api_key, client_id, model=embedding_model)
            payload = extract_payload(p)
            payload["embedded_text"] = text
            upsert_product(client_id, domain, product.product_id, vector, payload)
            success.append(product.product_id)
        except Exception as e:
            failed.append({"product_id": product.product_id, "error": str(e)})

    if success:
        increment_ingest_count(db, client_id, count=len(success))

    return {
        "client_id":     client_id,
        "success_count": len(success),
        "failed_count":  len(failed),
        "failed":        failed
    }


@router.post("/ingest/delete")
def delete(req: DeleteRequest, request: Request, db: Session = Depends(get_db)):
    try:
        license_data = validate_license_key(req.license_key, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        # CRITICAL: Enforce secure domain authorization
        authorizer = DomainAuthorizer(db)
        authorizer.validate_request(request, license_data)

        delete_product(license_data["client_id"], req.product_id)
        return {"deleted": True, "product_id": req.product_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))