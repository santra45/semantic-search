from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import search_products
from backend.app.services.cache_service import (get_cached_embedding, set_cached_embedding,
    get_cached_results,   set_cached_results)
from backend.app.services.wordpress_service import search_wordpress_fallback, should_trigger_fallback
from backend.app.services.intent_service import analyze_intent
import time
from backend.app.services.license_service import (
    validate_license_key,
    increment_search_count,
    check_search_quota,
    log_search
)
from backend.app.services.database import get_db
from urllib.parse import urlparse

router = APIRouter()

class SearchRequest(BaseModel):
    license_key: str
    query: str
    limit: int = 10


@router.post("/search")
async def search(req: SearchRequest, request: Request, db: Session = Depends(get_db)):
    start_time = time.time()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 — validate license key
    try:
        license_data = validate_license_key(req.license_key, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    client_id = license_data["client_id"]

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
    
    # Step 2 — check monthly quota
    if not check_search_quota(db, client_id, license_data["search_limit"]):
        raise HTTPException(
            status_code=429,
            detail="Monthly search limit reached. Please upgrade your plan."
        )

    query = req.query.strip().lower()

    # Step 3 — check results cache
    cached_results = get_cached_results(client_id, query)
    if cached_results is not None:
        print(f"⚡ Cache HIT (results): '{query}'")
        response_time = int((time.time() - start_time) * 1000)
        # CRITICAL: Count cached searches toward quota to prevent bypass
        increment_search_count(db, client_id)
        log_search(db, client_id, query, len(cached_results), response_time, cached=True)
        return {
            "query":   req.query,
            "count":   len(cached_results),
            "cached":  True,
            "results": cached_results
        }

    # ─── NEW: INTENT ANALYSIS ───────────────────────────────────────────────
    # We do this BEFORE embedding because we need to know IF we need filters.
    # Note: You might want to cache this too if Gemini gets expensive/slow.
    intent = analyze_intent(query)
    print(f"🧠 Intent Extracted: {intent.clean_query} | Max: {intent.max_price}")
    # ────────────────────────────────────────────────────────────────────────

    # Step 4 — check embedding cache
    query_vector = get_cached_embedding(intent.clean_query)
    if query_vector is not None:
        print(f"⚡ Cache HIT (embedding): '{intent.clean_query}'")
    else:
        print(f"🌐 Cache MISS: '{intent.clean_query}' — calling Gemini")
        query_vector = embed_query(intent.clean_query)
        set_cached_embedding(query, query_vector)

    # Step 5 — search Qdrant
    results = results = search_products(
        client_id=client_id,
        query_vector=query_vector,
        limit=req.limit,
        min_price=intent.min_price,
        max_price=intent.max_price,
        only_in_stock=intent.only_in_stock
    )

    # Step 5a — check if fallback should be triggered
    if should_trigger_fallback(results):
        print(f"🔄 Triggering WordPress fallback for query: '{query}' (max score: {max(r['score'] for r in results) if results else 0})")
        
        # Try WordPress fallback search
        fallback_results = await search_wordpress_fallback(
            client_id=client_id,
            query=req.query,
            license_key=req.license_key,
            limit=req.limit
        )
        
        if fallback_results:
            print(f"✅ WordPress fallback returned {len(fallback_results)} results")
            results = fallback_results
        else:
            print(f"❌ WordPress fallback returned no results, using empty result set")
            results = []

    # Step 6 — cache results
    set_cached_results(client_id, query, results)

    # Step 7 — track usage
    response_time = int((time.time() - start_time) * 1000)
    increment_search_count(db, client_id)
    log_search(db, client_id, query, len(results), response_time, cached=False)

    return {
        "query":   req.query,
        "count":   len(results),
        "cached":  False,
        "results": results
    }
