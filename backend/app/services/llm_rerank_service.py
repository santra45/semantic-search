import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Tuple

import httpx
import tiktoken

from google import genai
from openai import OpenAI
import anthropic
from backend.app.services.token_usage_service import track_usage
from backend.app.utils.gemini import thinking_can_be_disabled
from backend.app.utils.llm_logger import log_llm_interaction

# ---------------------------
# Logger Setup
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm_logger")


# ---------------------------
# HTTP Event Hooks (OpenAI & Anthropic)
# ---------------------------
def log_request(request: httpx.Request):
    body = request.content.decode("utf-8", errors="replace")
    headers = dict(request.headers)
    if "authorization" in headers:
        headers["authorization"] = "Bearer sk-***MASKED***"
    if "x-api-key" in headers:
        headers["x-api-key"] = "***MASKED***"

    try:
        body_formatted = json.dumps(json.loads(body), indent=2)
    except Exception:
        body_formatted = body

    logger.debug(
        f"\n{'='*60}\n"
        f"📤 REQUEST\n"
        f"  → {request.method} {request.url}\n"
        f"  Headers: {json.dumps(headers, indent=2)}\n"
        f"  Body:\n{body_formatted}\n"
        f"{'='*60}"
    )


def log_response(response: httpx.Response):
    response.read()  # Ensure body is fully loaded
    try:
        body_formatted = json.dumps(json.loads(response.text), indent=2)
    except Exception:
        body_formatted = response.text

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 RESPONSE\n"
        f"  ← Status: {response.status_code}\n"
        f"  Body:\n{body_formatted}\n"
        f"{'='*60}"
    )


def make_http_client() -> httpx.Client:
    return httpx.Client(
        event_hooks={
            "request": [log_request],
            "response": [log_response],
        }
    )


# ---------------------------
# Gemini Logger (manual)
# ---------------------------
def log_gemini_request(model: str, prompt: str):
    logger.debug(
        f"\n{'='*60}\n"
        f"📤 GEMINI REQUEST\n"
        f"  Model: {model}\n"
        f"  Prompt:\n{prompt}\n"
        f"{'='*60}"
    )


def log_gemini_response(response):
    try:
        text = response.text
    except Exception:
        text = str(response)

    try:
        metadata = str(response.usage_metadata)
    except Exception:
        metadata = "N/A"

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 GEMINI RESPONSE\n"
        f"  Text:\n{text}\n"
        f"  Usage Metadata: {metadata}\n"
        f"{'='*60}"
    )


# ---------------------------
# Helper: Extract JSON array
# ---------------------------
def extract_json_array(text: str):
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


# ---------------------------
# Helper: Estimate tokens (fallback)
# ---------------------------
def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    token_count = len(tokens)
    return token_count


# ---------------------------
# Helper: Get token usage
# ---------------------------
def get_token_usage(provider, response, prompt, response_text):
    try:
        if provider == "openai":
            usage = response.usage
            return {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            }

        elif provider == "anthropic":
            usage = response.usage
            return {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.input_tokens + usage.output_tokens,
            }

        elif provider == "gemini":
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata = response.usage_metadata
                input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
                total_tokens = getattr(metadata, "total_token_count", 0) or 0

                # candidates_token_count is None in some Gemini responses,
                # so derive output from total - input, fallback to estimating from text
                candidates = getattr(metadata, "candidates_token_count", None)
                if candidates is not None:
                    output_tokens = candidates
                elif total_tokens > input_tokens:
                    output_tokens = total_tokens - input_tokens  # derived
                elif output_tokens == 0 and response_text:
                    output_tokens = estimate_tokens(response_text)
                else:
                    output_tokens = estimate_tokens(response_text)  # last resort fallback

                return {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,  # recalculate proper total
                }
            else:
                return {
                    "input": estimate_tokens(prompt),
                    "output": estimate_tokens(response_text),
                    "total": estimate_tokens(prompt + response_text),
                }

    except Exception as e:
        logger.warning(f"⚠️ Token usage extraction failed: {e}")
        return {
            "input": estimate_tokens(prompt),
            "output": estimate_tokens(response_text),
            "total": estimate_tokens(prompt + response_text),
        }


# ---------------------------
# Optional: Cost calculation
# ---------------------------
MODEL_PRICING = {
    # ── Gemini ────────────────────────────────────────────────────────────────
    "gemini-3.1-pro-preview":    {"input": 0.000002,     "output": 0.000012},
    "gemini-2.5-pro":            {"input": 0.00000125,   "output": 0.00001},
    "gemini-2.5-flash":          {"input": 0.0000003,    "output": 0.0000025},
    "gemini-2.5-flash-lite":     {"input": 0.0000001,    "output": 0.0000004},
    "gemma-3-27b-it":            {"input": 0.00000008,   "output": 0.00000016},
    "gemma-4-31b-it":            {"input": 0.00000013,   "output": 0.00000038},

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-5.5":                   {"input": 0.000005,     "output": 0.00003},
    "gpt-5.4-mini":              {"input": 0.00000075,   "output": 0.0000045},
    "gpt-5.4-nano":              {"input": 0.0000002,    "output": 0.00000125},
    "gpt-5.2":                   {"input": 0.00000175,   "output": 0.000014},

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-opus-4-6":           {"input": 0.000005,     "output": 0.000025},
    "claude-sonnet-4-6":         {"input": 0.000003,     "output": 0.000015},
    "claude-haiku-4-5-20251001": {"input": 0.000001,     "output": 0.000005},
    "claude-3-5-sonnet-20241022":{"input": 0.000003,     "output": 0.000015},
}

def estimate_cost(model: str, usage: Dict) -> float:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    return (
        usage["input"] * pricing["input"] +
        usage["output"] * pricing["output"]
    )


# ---------------------------
# In-process LRU+TTL cache for rerank results  (Phase 2.1d)
# ---------------------------
#
# Repeat queries inside a short window — the same shopper hitting "return
# policy" twice in 30s, or two parallel page-loads of an FAQ widget —
# would otherwise re-pay the LLM cost. A 60s TTL collapses those into
# one LLM call.
#
# Scope:
#   * Module-local. Lives inside a single uvicorn worker; multiple workers
#     = multiple cache instances (acceptable — SMB query volumes make
#     cross-worker dedup not worth the Redis round-trip).
#   * Tenant-isolated: client_id is part of every key, so one tenant can
#     never see another tenant's reranked list.
#   * Failure-aware: only SUCCESSFUL reranks are cached. If the LLM call
#     fails the fallback (unranked top-N) is returned but NOT cached, so
#     the next request gets a real retry.
#
# Eviction: LRU at `maxsize` entries, TTL at `ttl_seconds`. Lazy expiry
# on get; bulk eviction on set when over capacity.
#
# Concurrency: a single threading.Lock guards both OrderedDict mutations
# and TTL checks. Uvicorn's default async worker still runs request
# handlers in a thread pool for sync endpoints (which retrieve/products
# is), so the lock is necessary.

class _RerankCache:
    """Tiny thread-safe LRU+TTL cache. ~30 lines vs adding cachetools."""

    __slots__ = ("_maxsize", "_ttl", "_data", "_lock", "_hits", "_misses")

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 60):
        self._maxsize = int(maxsize)
        self._ttl = float(ttl_seconds)
        # OrderedDict gives us O(1) move_to_end for LRU semantics.
        # Value: (expires_at, payload_list).
        self._data: "OrderedDict[Tuple, Tuple[float, List[Dict[str, Any]]]]" = OrderedDict()
        self._lock = threading.Lock()
        # Stats counters kept for occasional log dumps — cheap to maintain.
        self._hits = 0
        self._misses = 0

    def get(self, key: Tuple) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                # Lazy expiry — entry is past TTL, treat as miss + evict.
                del self._data[key]
                self._misses += 1
                return None
            # Move to end so LRU eviction targets genuinely cold entries.
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: Tuple, value: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._data[key] = (time.time() + self._ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                # popitem(last=False) drops the oldest — true LRU order
                # because every get() promotes its hit to the end.
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size":   len(self._data),
                "hits":   self._hits,
                "misses": self._misses,
            }


# Singleton — sized for SMB tenant patterns (a chatbot session generates
# ~5-20 reranked queries; 1000 entries holds ~50-200 active sessions
# simultaneously, well above realistic concurrency for the target tier).
_RERANK_CACHE = _RerankCache(maxsize=1000, ttl_seconds=60)


def _item_identity(item: Dict[str, Any]) -> str:
    """Stable per-item identifier used in the rerank cache key.

    Mirrors the type-specific id resolution the rerank prompt itself
    uses (product_id / page_id / post_id / entity_id) so the cache key
    reflects exactly which candidates would have been embedded in the
    prompt — different candidate sets correctly miss the cache.
    """
    ct = item.get("content_type") or ("product" if item.get("sku") else "")
    eid = (
        item.get("product_id")
        or item.get("page_id")
        or item.get("post_id")
        or item.get("entity_id")
        or item.get("id")
        or ""
    )
    return f"{ct}:{eid}"


def _rerank_cache_key(client_id: str, query: str, content: List[Dict[str, Any]]) -> Tuple:
    """Compose the cache key.

    Normalisation:
      * client_id forced to str (defensive — never let a non-string slip in)
      * query trimmed, lower-cased, whitespace collapsed so casing/spacing
        variation doesn't multiply entries
      * candidate ids taken IN ORDER for the first 25 items, matching the
        `content[:25]` cap inside the rerank prompt. Same first-25 in
        same order → same key. Different items (or different ordering
        from Qdrant) → cache miss → fresh rerank.
    """
    norm_q = " ".join((query or "").lower().split())
    ids = tuple(_item_identity(item) for item in content[:25])
    return (str(client_id), norm_q, ids)


# ---------------------------
# Main Function
# ---------------------------
def llm_rerank_content(
    query: str,
    content: List[Dict],
    limit: int = 10,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    client_id: str = "anonymous"
) -> List[Dict]:

    provider = llm_provider or "gemini"

    if provider == "gemini":
        model = llm_model or "gemini-1.5-flash"
    elif provider == "openai":
        model = llm_model or "gpt-4o-mini"
    elif provider == "anthropic":
        model = llm_model or "claude-3-5-haiku-20241022"
    else:
        model = "gemini-1.5-flash"

    api_key = llm_api_key

    if not content:
        return []

    if not api_key:
        logger.warning("⚠️ No API key provided. Returning top results without reranking.")
        return content[:limit]

    # ── Cache lookup (Phase 2.1d) ────────────────────────────────────────
    # Sits between the cheap-guards and the LLM call. Skipping the cache
    # for empty content / missing API key is intentional — those paths
    # don't represent a real "rerank result" we'd want to memoise.
    #
    # Skipping when query is whitespace too: nothing useful to cache and
    # the LLM would fail validation downstream.
    cache_key = _rerank_cache_key(client_id, query, content) if (query or "").strip() else None
    if cache_key is not None:
        cached = _RERANK_CACHE.get(cache_key)
        if cached is not None:
            # Hit — re-slice to the caller's `limit` (the cached value
            # was stored at whatever limit the FIRST caller requested,
            # which may differ from this caller).
            logger.info(
                "[rerank-cache] hit  client=%s candidates=%d returned=%d",
                client_id, len(content), min(len(cached), limit),
            )
            return cached[:limit]
        logger.info(
            "[rerank-cache] miss client=%s candidates=%d",
            client_id, len(content),
        )

    content_summaries = []
    content_map = {}

    # Process mixed content types (products, pages, posts).
    # The [:25] is a safety ceiling on prompt size — the LLM call serialises
    # one summary per item and we don't want a 100-item query to blow the
    # context window. The upstream caller (retrieve.py rerank_limit) is the
    # real knob; this just protects against a misconfigured caller passing
    # a huge list. If you raise rerank_limit above 25, raise this in lock-
    # step or items 26+ will be silently dropped from the rerank pool.
    for item in content[:25]:
        content_type = item.get("content_type", "product")
        
        # Generate unique ID based on content type
        if content_type == "product":
            item_id = str(item.get("product_id") or item.get("id"))
        elif content_type == "page":
            item_id = f"page_{item.get('page_id')}"
        elif content_type == "post":
            item_id = f"post_{item.get('post_id')}"
        else:
            continue
        
        if not item_id:
            continue

        # Build summary based on content type
        if content_type == "product":
            summary = {
                "id": item_id,
                "type": "product",
                "name": item.get("name", ""),
                "category": item.get("categories", ""),
                "price": item.get("price", 0),
            }
        elif content_type == "page":
            summary = {
                "id": item_id,
                "type": "page",
                "title": item.get("title", ""),
                "excerpt": item.get("excerpt", "")[:200],  # Truncate excerpt
            }
        elif content_type == "post":
            summary = {
                "id": item_id,
                "type": "post",
                "title": item.get("title", ""),
                "excerpt": item.get("excerpt", "")[:200],
                "categories": item.get("categories", ""),
                "tags": item.get("tags", ""),
            }
        else:
            continue

        content_summaries.append(summary)
        content_map[item_id] = item

    prompt = f"""
    You are an expert content recommendation assistant for an e-commerce website.

    Customer query:
    {query}

    Available content (products, pages, blog posts):
    {json.dumps(content_summaries)}

    Task:
    - Select ONLY relevant content (products, pages, or posts)
    - For products: ignore wrong category/gender
    - For pages/posts: match the topic/theme of the query
    - Prefer exact matches over partial
    - Rank by relevance (best first)
    - Include a mix of content types if relevant

    Return ONLY a JSON array of content IDs as strings.
    Do NOT include objects, scores, reasons, or any other fields — just the bare IDs.
    Example: ["123", "page_456", "post_789"] or []
    """

    # Telemetry — wall-clock from "about to call the LLM" to "got a response
    # back". Logged via `duration_ms` on the llm_interaction record so the
    # ops team can answer "is rerank our P95 culprit?" without speculation.
    # Phase 2.1a (LLM Reranker Tuning).
    rerank_t0 = time.perf_counter()
    response_text = ""
    response = None

    try:
        # ---------------------------
        # GEMINI
        # ---------------------------
        if provider == "gemini":
            log_gemini_request(model, prompt)
            client = genai.Client(api_key=api_key)
            # Reranking a short candidate list needs no reasoning pass — turn
            # the thinking phase off on the flash family (was costing ~2s).
            # Typed config: the SDK's GenerateContentConfig rejects a camelCase
            # dict (extra_forbidden on thinkingBudget), so use the real objects.
            gen_config = (
                genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
                if thinking_can_be_disabled(model) else None
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            log_gemini_response(response)
            response_text = response.text.strip()

        # ---------------------------
        # OPENAI
        # ---------------------------
        elif provider == "openai":
            client = OpenAI(
                api_key=api_key,
                http_client=make_http_client(),
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            response_text = response.choices[0].message.content.strip()

        # ---------------------------
        # ANTHROPIC
        # ---------------------------
        elif provider == "anthropic":
            client = anthropic.Anthropic(
                api_key=api_key,
                http_client=make_http_client(),
            )
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text.strip()

        else:
            return content[:limit]

        # Stop the clock as soon as the LLM call returns — anything below
        # this point (token accounting, JSON parsing, payload reshaping) is
        # our overhead, not the LLM's, and we'd rather not blame the LLM
        # for slow Python.
        duration_ms = int((time.perf_counter() - rerank_t0) * 1000)

        # ---------------------------
        # TOKEN USAGE & COST
        # ---------------------------
        usage = get_token_usage(provider, response, prompt, response_text)
        cost = estimate_cost(model, usage)

        logger.info(f"🔢 Token Usage: {usage}")
        logger.info(f"💰 Estimated Cost: ${round(cost, 8)}")
        logger.info(f"⏱  Rerank LLM latency: {duration_ms} ms ({provider} / {model})")

        log_llm_interaction(
            provider=provider,
            model=model,
            purpose="product_rerank",
            prompt=prompt,
            response_text=response_text,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            cost=cost,
            client_id=client_id,
            duration_ms=duration_ms,
            # Extra metadata makes the log entry self-describing — when
            # grepping logs/llm.log for slow rerank calls we can see how
            # many candidates were ranked without re-reading the prompt.
            extra={
                "items":      len(content_summaries),
                "candidates": len(content),
                "limit":      limit,
            },
        )

        # Track token usage
        try:
            track_usage(
                client_id=client_id,
                query_type="product_rerank",
                llm_provider=provider,
                llm_model=model,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                input_cost=usage["input"] * MODEL_PRICING.get(model, {}).get("input", 0),
                output_cost=usage["output"] * MODEL_PRICING.get(model, {}).get("output", 0),
                request_text_length=len(prompt),
                response_text_length=len(response_text)
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to track token usage: {e}")

        # ---------------------------
        # Parse & Return
        # ---------------------------
        json_text = extract_json_array(response_text)

        if json_text:
            relevant_ids = json.loads(json_text)
            relevant_content = []
            for entry in relevant_ids:
                # Some models embellish the response with {"id": "...", "score": ...}
                # even when the prompt asks for bare IDs — accept either shape.
                if isinstance(entry, dict):
                    item_id = entry.get("id") or entry.get("product_id") or entry.get("page_id") or entry.get("post_id")
                else:
                    item_id = entry
                if item_id is None:
                    continue
                item_id_str = str(item_id)
                if item_id_str in content_map:
                    relevant_content.append(content_map[item_id_str])

            if relevant_content:
                # Cache the FULL reranked list (not the limit-sliced one)
                # so a later caller asking for a larger limit on the same
                # (client, query, candidates) tuple still benefits from
                # the LLM work we just paid for.
                #
                # Only successful, non-empty reranks are cached. Empty
                # results stay uncached on purpose — gives transient LLM
                # parse failures a free retry path on the next call.
                if cache_key is not None:
                    _RERANK_CACHE.set(cache_key, relevant_content)
                return relevant_content[:limit]

        logger.warning("⚠️ No relevant content found after reranking.")
        return []

    except Exception as e:
        # Time even on failure — slow failures (timeouts especially) are
        # the most useful latency data we can collect.
        duration_ms = int((time.perf_counter() - rerank_t0) * 1000)
        logger.error(f"❌ Error during LLM reranking after {duration_ms} ms: {str(e)}", exc_info=True)
        try:
            log_llm_interaction(
                provider=provider,
                model=model,
                purpose="product_rerank",
                prompt=prompt,
                client_id=client_id,
                duration_ms=duration_ms,
                error=str(e),
                extra={"candidates": len(content), "limit": limit},
            )
        except Exception:
            pass
        return content[:limit]


# ---------------------------
# Smart Trigger
# ---------------------------
def should_use_llm_reranking(query: str, content: List[Dict]) -> bool:
    simple_indicators = ["shirt", "pants", "dress", "shoes", "bag", "watch"]
    query_lower = query.lower()

    if any(indicator in query_lower for indicator in simple_indicators) and len(query.split()) <= 2:
        return False

    if len(content) > 5 or len(query.split()) > 3:
        return True

    complex_indicators = [
        "for", "with", "that", "which",
        "under", "over", "between",
        "size", "color", "material",
    ]

    if any(indicator in query_lower for indicator in complex_indicators):
        return True

    return False


# ---------------------------
# Backward Compatibility Wrapper
# ---------------------------
def llm_rerank_products(
    query: str,
    products: List[Dict],
    limit: int = 10,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    client_id: str = "anonymous"
) -> List[Dict]:
    """Backward compatibility wrapper for llm_rerank_content."""
    return llm_rerank_content(
        query=query,
        content=products,
        limit=limit,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        client_id=client_id
    )

