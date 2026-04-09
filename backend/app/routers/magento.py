from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session

from backend.app.services.cache_service import get_cached_embedding, get_cached_results, set_cached_embedding, set_cached_results, invalidate_client_results
from backend.app.services.database import get_db
from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.embedder import embed_query, embed_document
from backend.app.services.intent_service import analyze_intent
from backend.app.services.license_service import validate_license_key, check_search_quota, increment_search_count, log_search, increment_ingest_count, extract_license_key_from_authorization
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.llm_rerank_service import llm_rerank_products, should_use_llm_reranking
from backend.app.services.product_service import build_product_text, extract_payload
from backend.app.services.qdrant_service import search_products, upsert_product, delete_product, get_client_product_count
from backend.app.services.rerank_service import extract_keywords, filter_and_rerank
import time

router = APIRouter()


class MagentoSearchRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    limit: int = 10
    enable_intent: bool = False
    llm_provider: str = None
    llm_model: str = None
    llm_api_key_encrypted: Optional[str] = None


class MagentoProduct(BaseModel):
    product_id: str
    name: str
    categories: str = ""
    tags: str = ""
    description: str = ""
    short_description: str = ""
    price: float = 0
    regular_price: float = 0
    sale_price: float = 0
    currency: str = ""
    currency_symbol: str = ""
    on_sale: bool = False
    permalink: str = ""
    image_url: str = ""
    stock_status: str = "instock"
    average_rating: float = 0
    attributes: list = Field(default_factory=list)


class MagentoSyncBatchRequest(BaseModel):
    license_key: Optional[str] = None
    products: List[MagentoProduct]
    batch_number: int = 1
    total_batches: int = 1
    llm_api_key_encrypted: Optional[str] = None


class MagentoDeleteRequest(BaseModel):
    license_key: Optional[str] = None
    product_id: str


def resolve_headers(
    authorization: Optional[str],
    x_api_key: Optional[str],
    x_llm_api_key_encrypted: Optional[str],
    request_license: Optional[str],
    request_llm_key: Optional[str],
):
    return {
        "license_key": extract_license_key_from_authorization(authorization) or request_license,
        "api_key": x_api_key,
        "llm_api_key_encrypted": x_llm_api_key_encrypted or request_llm_key,
    }


@router.post("/magento/search")
async def magento_search(
    req: MagentoSearchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db)
):
    start_time = time.time()

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    headers = resolve_headers(
        authorization,
        x_api_key,
        x_llm_api_key_encrypted,
        req.license_key,
        req.llm_api_key_encrypted,
    )
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    if not check_search_quota(db, client_id, license_data["search_limit"]):
        raise HTTPException(status_code=429, detail="Monthly search limit reached. Please upgrade your plan.")

    query = req.query.strip().lower()

    # Initialize intent analysis variables
    min_price = None
    max_price = None
    only_in_stock = False

    # Perform intent analysis if enabled
    if req.enable_intent:
        try:
            intent = analyze_intent(
                query,
                req.llm_provider,
                req.llm_model,
                headers["llm_api_key_encrypted"],
                headers["license_key"],
            )
            if intent:
                min_price = intent.get('min_price')
                max_price = intent.get('max_price')
                only_in_stock = intent.get('in_stock', False)
        except Exception as e:
            # Log error but continue without intent filtering
            print(f"Intent analysis failed: {e}")

    cached_results = get_cached_results(f"{client_id}_{domain}", query)
    if cached_results is not None:
        response_time = int((time.time() - start_time) * 1000)
        increment_search_count(db, client_id)
        log_search(db, client_id, query, len(cached_results), response_time, cached=True)
        return {"query": req.query, "count": len(cached_results), "cached": True, "results": cached_results}

    query_vector = get_cached_embedding(query)
    if query_vector is None:
        embedding_api_key = None
        if headers["llm_api_key_encrypted"]:
            try:
                embedding_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
            except Exception:
                embedding_api_key = None
        query_vector = embed_query(query, embedding_api_key, client_id)
        set_cached_embedding(query, query_vector)

    fetch_limit = req.limit * 5
    results = search_products(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=fetch_limit,
        min_price=min_price,
        max_price=max_price,
        only_in_stock=only_in_stock,
    )

    # Step 5b — keyword post-filter & re-rank
    # Runs on RAW customer query (req.query), no LLM needed.
    # extract_keywords() detects gender, color, material signals.
    # filter_and_rerank() removes wrong-gender products and boosts
    # soft-signal matches, then slices back to original limit.
    keywords = extract_keywords(req.query)
    print(
        f"🔑 Keywords: gender={keywords['gender']} colors={keywords['colors']} materials={keywords['materials']}"
    )
    results = filter_and_rerank(results, keywords, req.limit)
    print(f"Keyword filtering took: {time.time() - start_time}")

    # Step 5c — LLM re-ranking for complex queries
    # Uses Gemini to analyze semantic relevance and filter out irrelevant products
    if should_use_llm_reranking(req.query, results):
        print(f"🤖 Applying LLM re-ranking for query: '{req.query}'")
        if headers["llm_api_key_encrypted"]:
            try:
                llm_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
            except Exception as e:
                print(f"❌ Decryption failed: {e}")
                llm_api_key = None
        else:
            print(f"API key not getting from DB")
            llm_api_key = None
        llm_results = llm_rerank_products(
            req.query, 
            results, 
            req.limit,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            llm_api_key=llm_api_key,
            client_id=client_id,
        )
        print(f"LLM re-ranking took: {time.time() - start_time}")
        if llm_results is not None:
            print(f"🤖 LLM re-ranked {len(results)} → {len(llm_results)} products")
            results = llm_results
        else:
            print(f"🤖 LLM re-ranking failed, using filtered results")
    else:
        print(f"⚡ Skipping LLM re-ranking for simple query: '{req.query}'")

    set_cached_results(f"{client_id}_{domain}", query, results)

    response_time = int((time.time() - start_time) * 1000)
    increment_search_count(db, client_id)
    log_search(db, client_id, query, len(results), response_time, cached=False)

    return {"query": req.query, "count": len(results), "cached": False, "results": results}


@router.post("/magento/sync/batch")
def magento_sync_batch(
    req: MagentoSyncBatchRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db)
):
    headers = resolve_headers(
        authorization,
        x_api_key,
        x_llm_api_key_encrypted,
        req.license_key,
        req.llm_api_key_encrypted,
    )
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    current_count = get_client_product_count(client_id, domain)
    incoming_count = len(req.products)
    if current_count + incoming_count > license_data["product_limit"]:
        raise HTTPException(
            status_code=400,
            detail=f"Product limit exceeded. Current: {current_count}, Incoming: {incoming_count}, Limit: {license_data['product_limit']}",
        )

    embedding_api_key = None
    if headers["llm_api_key_encrypted"]:
        try:
            embedding_api_key = decrypt_key(headers["llm_api_key_encrypted"], headers["license_key"])
        except Exception:
            embedding_api_key = None

    success_ids = []
    failed_ids = []

    for product in req.products:
        try:
            p = product.model_dump()
            text = build_product_text(p)
            vector = embed_document(text, embedding_api_key, client_id)
            payload = extract_payload(p)
            payload["embedded_text"] = text
            upsert_product(client_id, domain, product.product_id, vector, payload)
            success_ids.append(product.product_id)
        except Exception:
            failed_ids.append(product.product_id)

    if success_ids:
        increment_ingest_count(db, client_id, count=len(success_ids))

    if req.batch_number >= req.total_batches:
        invalidate_client_results(client_id)

    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "batch_number": req.batch_number,
        "total_batches": req.total_batches,
        "is_last_batch": req.batch_number >= req.total_batches,
    }


@router.get("/magento/sync/quota")
def magento_sync_quota(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    headers = resolve_headers(authorization, x_api_key, None, None, None)
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    current_count = get_client_product_count(client_id, domain)
    product_limit = license_data["product_limit"]

    return {
        "current_count": current_count,
        "product_limit": product_limit,
        "remaining": product_limit - current_count,
        "exceeded": current_count > product_limit
    }


@router.post("/magento/sync/delete")
def magento_sync_delete(
    req: MagentoDeleteRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    headers = resolve_headers(authorization, x_api_key, None, req.license_key, None)
    if not headers["license_key"]:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        license_data = validate_license_key(headers["license_key"], db)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    authorizer = DomainAuthorizer(db)
    authorizer.validate_request(request, license_data, api_key=headers["api_key"])

    delete_product(license_data["client_id"], req.product_id)
    invalidate_client_results(license_data["client_id"])

    return {"deleted": True, "product_id": req.product_id}
