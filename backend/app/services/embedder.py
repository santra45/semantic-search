import logging
import tiktoken
from google import genai

from backend.app.config import EMBED_MODEL
from backend.app.services.cache_service import get_cached_embedding, set_cached_embedding
from backend.app.services.token_usage_service import track_usage
from backend.app.utils.llm_logger import log_llm_interaction

logger = logging.getLogger("embed_logger")


# ── Pricing (per-token) ─────────────────────────────────────────────────────

EMBED_PRICING = {
    "gemini-embedding-001": {"input": 0.00000015},  # $0.15 per 1M tokens
}


def estimate_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def get_embed_token_count(response, text: str) -> int:
    try:
        if hasattr(response, "metadata") and response.metadata:
            token_count = getattr(response.metadata, "token_count", None)
            if token_count:
                return int(token_count)
    except Exception as e:
        logger.warning(f"Could not extract token count from response metadata: {e}")
    return estimate_tokens(text)


def estimate_embed_cost(model: str, token_count: int) -> float:
    pricing = EMBED_PRICING.get(model)
    return token_count * pricing["input"] if pricing else 0.0


def get_client(api_key: str = None):
    return genai.Client(api_key=api_key)


# ── Embed ───────────────────────────────────────────────────────────────────


def _embed(
    text: str,
    task_type: str,
    query_type: str,
    api_key: str,
    client_id: str,
    model: str = None,
) -> list[float]:
    # `model` is the tenant's configured embedding model, already validated
    # against the supported set by embedding_key_service. None means they
    # never set one, which is every install predating the embedding config.
    model       = model or EMBED_MODEL
    client      = get_client(api_key)
    result = client.models.embed_content(
        model=model,
        contents=text,
        config={"task_type": task_type},
    )

    token_count = get_embed_token_count(result, text)
    cost        = estimate_embed_cost(model, token_count)
    dims        = len(result.embeddings[0].values)

    log_llm_interaction(
        provider="google",
        model=model,
        purpose=query_type,
        prompt=text,
        response_text=f"<embedding vector: {dims} dims>",
        input_tokens=token_count,
        output_tokens=0,
        cost=cost,
        client_id=client_id,
        extra={"task_type": task_type, "dims": dims},
    )

    try:
        track_usage(
            client_id=client_id,
            query_type=query_type,
            llm_provider="google",
            llm_model=model,
            input_tokens=token_count,
            output_tokens=0,
            input_cost=cost,
            output_cost=0.0,
            request_text_length=len(text),
            response_text_length=0,
        )
    except Exception as e:
        logger.warning(f"Failed to track token usage: {e}")

    return result.embeddings[0].values


# Query embeddings are cached (24h) keyed by model + task + text. Two wins:
#   * within one chat turn the same query is embedded once for /retrieve/content
#     and again for /retrieve/products — the second call is now a cache hit;
#   * the FAQ last-resort lookup on the refusal path reuses the primary query's
#     vector instead of re-embedding.
# Namespaced so it can never collide with the legacy search.py / magento.py
# callers (which cache under the bare-text key, possibly a different model).
# Documents are deliberately NOT cached — they embed once at sync time and
# caching every chunk would bloat Redis for no reuse.
# Built per-call rather than once at import, because the model is now a
# per-tenant value. Two tenants on different embedding models must not share
# a cache entry — the vectors are not interchangeable.
def _query_cache_ns(model: str) -> str:
    return f"{model}:RETRIEVAL_QUERY"


def embed_query(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_search",
    model: str = None,
) -> list[float]:
    ns = _query_cache_ns(model or EMBED_MODEL)
    cached = get_cached_embedding(text, ns)
    if cached is not None:
        return cached
    vector = _embed(text, "RETRIEVAL_QUERY", query_type, api_key, client_id, model)
    try:
        set_cached_embedding(text, vector, ns)
    except Exception as exc:  # a cache-write hiccup must never break embedding
        logger.warning(f"embedding cache write failed: {exc}")
    return vector


def embed_document(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_document",
    model: str = None,
) -> list[float]:
    return _embed(text, "RETRIEVAL_DOCUMENT", query_type, api_key, client_id, model)
