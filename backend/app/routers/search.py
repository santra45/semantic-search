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
from backend.app.services.embedding_key_service import (
    resolve_embedding_key,
    resolve_embedding_model,
)
import time
from backend.app.services.license_service import (increment_search_count, log_search)
from backend.app.services import request_auth
from backend.app.services.database import get_db

router = APIRouter()

# Derived from the plugins, not invented: `semantic-search-woo` is the only
# thing that builds POST /api/search. Its Magento counterpart is a separate
# route (/api/magento/search, guarded by magento.py's own _SEARCH_PRODUCTS),
# which is why this set names one product rather than both search modules.
_SEARCH_PRODUCTS = frozenset({"woo_search"})


class SearchRequest(BaseModel):
    license_key: str
    query: str
    limit: int = 10
    enable_intent: bool = False
    llm_provider: str = None
    llm_model: str = None
    llm_api_key_encrypted: str = None
    # The tenant's separate embedding key, encrypted under the license key
    # exactly like llm_api_key_encrypted. Absent = fall back to the LLM key,
    # which is what every install predating the embedding config sends.
    embedding_api_key_encrypted: str = None
    embedding_provider: str = None
    embedding_model: str = None
    content_types: list = None  # None = all types, ['product'], ['page'], ['post'], or ['product', 'page', 'post']


@router.post("/search")
async def search(req: SearchRequest, request: Request, db: Session = Depends(get_db)):
    start_time = time.time()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 — authenticate, authorise, and bind the tenant context.
    #
    # This one call replaces three things that used to sit inline here: a
    # validate_license_key that resolved no v2 key, a hand-rolled Origin check,
    # and a check_search_quota reading usage_logs_archive_v1 — a table the v2
    # migration froze, so it metered a count that can no longer grow.
    #
    # The hand-rolled domain check is not merely duplicated by DomainAuthorizer,
    # it was weaker than it: it compared the Origin hostname against the licensed
    # domain alone, where _get_all_valid_domains() also accepts the www/apex
    # counterpart and the configured subdomains. Every caller is server-to-server
    # PHP with no Origin header, so both versions pass the same traffic today;
    # the difference only shows up the day a browser calls this endpoint.
    license_data = request_auth.authorize_request(
        request=request,
        db=db,
        authorization=None,          # this endpoint takes the key in the body
        x_api_key=None,
        request_license=req.license_key,
        allowed_products=_SEARCH_PRODUCTS,
    )

    client_id = license_data["client_id"]
    domain = license_data["domain"]
    license_key = req.license_key

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
    query_vector = get_cached_embedding(query)
    if query_vector is not None:
        print(f"⚡ Cache HIT (embedding): '{query}'")
    else:
        print(f"🌐 Cache MISS: '{query}' — calling Gemini")
        
        # The embedding key if the merchant configured one, else the LLM key.
        embedding_api_key = resolve_embedding_key(
            req.embedding_api_key_encrypted,
            req.llm_api_key_encrypted,
            license_key,
        )
        embedding_model = resolve_embedding_model(req.embedding_model, req.embedding_provider)
        query_vector = embed_query(query, embedding_api_key, client_id, model=embedding_model)
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
        content_types=req.content_types,
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
