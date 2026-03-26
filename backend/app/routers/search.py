from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import search_products
from backend.app.services.cache_service import (get_cached_embedding, set_cached_embedding, get_cached_results, set_cached_results)
from backend.app.services.wordpress_service import (search_wordpress_fallback, should_trigger_fallback)
from backend.app.services.intent_service import analyze_intent
from backend.app.services.rerank_service import extract_keywords, filter_and_rerank
from backend.app.services.llm_rerank_service import llm_rerank_products, should_use_llm_reranking
from backend.app.services.llm_key_service import decrypt_key
import time
from backend.app.services.license_service import (validate_license_key, increment_search_count, check_search_quota, log_search)
from backend.app.services.database import get_db
from urllib.parse import urlparse

router = APIRouter()


class SearchRequest(BaseModel):
    license_key: str
    query: str
    limit: int = 10
    enable_intent: bool = False
    llm_provider: str = None
    llm_model: str = None
    llm_api_key_encrypted: str = None


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
    domain = license_data["domain"]
    license_key = req.license_key

    # CRITICAL: Enforce domain authorization
    origin = request.headers.get("origin") or request.headers.get("referer")
    allowed_domain = license_data.get("domain")

    if allowed_domain and origin:
        hostname = urlparse(origin).hostname

        if allowed_domain and hostname not in [allowed_domain, "127.0.0.1"]:
            raise HTTPException(
                status_code=403,
                detail=f"Domain not authorized. License valid for: {allowed_domain}",
            )
    print(f"domain validation took: {time.time() - start_time}")
    # Step 2 — check monthly quota
    if not check_search_quota(db, client_id, license_data["search_limit"]):
        raise HTTPException(
            status_code=429,
            detail="Monthly search limit reached. Please upgrade your plan.",
        )

    query = req.query.strip().lower()
    print(f"Search quota took: {time.time() - start_time}")

    # Step 3 — check results cache
    cached_results = get_cached_results(f"{client_id}_{domain}", query)
    if cached_results is not None:
        print(f"⚡ Cache HIT (results): '{query}'")
        response_time = int((time.time() - start_time) * 1000)
        # CRITICAL: Count cached searches toward quota to prevent bypass
        increment_search_count(db, client_id)
        log_search(
            db, client_id, query, len(cached_results), response_time, cached=True
        )
        return {
            "query": req.query,
            "count": len(cached_results),
            "cached": True,
            "results": cached_results,
        }
    print(f"Cache check took: {time.time() - start_time}")

    # ─── INTENT ANALYSIS (conditional) ────────────────────────────────────────
    if req.enable_intent:
        # We do this BEFORE embedding because we need to know IF we need filters.
        # Note: You might want to cache this too if Gemini gets expensive/slow.
        intent = analyze_intent(query)
        print(f"🧠 Intent Extracted: {intent.clean_query} | Max: {intent.max_price}")
        clean_query = intent.clean_query
        min_price = intent.min_price
        max_price = intent.max_price
        only_in_stock = intent.only_in_stock
    else:
        # Use original query without intent analysis
        intent = None
        clean_query = query
        min_price = None
        max_price = None
        only_in_stock = False
    # ────────────────────────────────────────────────────────────────────────

    # Step 4 — check embedding cache
    query_vector = get_cached_embedding(clean_query)
    if query_vector is not None:
        print(f"⚡ Cache HIT (embedding): '{clean_query}'")
    else:
        print(f"🌐 Cache MISS: '{clean_query}' — calling Gemini")
        
        # Decrypt embedding API key if provided
        if req.llm_api_key_encrypted:
            try:
                embedding_api_key = decrypt_key(req.llm_api_key_encrypted, license_key)   
            except Exception as e:
                print(f"❌ Embedding API key decryption failed: {e}")
                embedding_api_key = None
        else:
            print(f"Embedding API key not provided, using default")
            embedding_api_key = None
            
        query_vector = embed_query(clean_query, embedding_api_key)
        set_cached_embedding(query, query_vector)

    # Step 5 — search Qdrant
    # Fetch 2× the requested limit so that after keyword filtering
    # we still have enough candidates to fill the requested result count.
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
    # Runs on the RAW customer query (req.query), no LLM needed.
    # extract_keywords() detects gender, color, material signals.
    # filter_and_rerank() removes wrong-gender products and boosts
    # soft-signal matches, then slices back to the original limit.
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
        if req.llm_api_key_encrypted:
            try:
                llm_api_key = decrypt_key(req.llm_api_key_encrypted, license_key)   
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
            llm_api_key=llm_api_key
        )
        print(f"LLM re-ranking took: {time.time() - start_time}")
        if llm_results is not None:
            print(f"🤖 LLM re-ranked {len(results)} → {len(llm_results)} products")
            results = llm_results
        else:
            print(f"🤖 LLM re-ranking failed, using filtered results")
    else:
        print(f"⚡ Skipping LLM re-ranking for simple query: '{req.query}'")

    # Step 5a — check if fallback should be triggered
    #if should_trigger_fallback(results):
    #    print(f"🔄 Triggering WordPress fallback for query: '{req.query}' (max score: {max(r['score'] #for r in results) if results else 0})")
    #    
    #    # Try WordPress fallback search
    #    fallback_results = await search_wordpress_fallback(
    #        client_id=client_id,
    #        query=req.query,
    #        license_key=req.license_key,
    #        limit=req.limit
    #    )
    #
    #    if fallback_results:
    #        print(f"✅ WordPress fallback returned {len(fallback_results)} results")
    #        results = fallback_results
    #    else:
    #        print(f"❌ WordPress fallback returned no results, using empty result set")
    #        results = []

    # Step 6 — cache results
    set_cached_results(f"{client_id}_{domain}", query, results)

    # Step 7 — track usage
    response_time = int((time.time() - start_time) * 1000)
    increment_search_count(db, client_id)
    log_search(db, client_id, query, len(results), response_time, cached=False)

    return {
        "query": req.query,
        "count": len(results),
        "cached": False,
        "results": results,
    }
