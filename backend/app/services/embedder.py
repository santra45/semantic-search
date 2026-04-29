import logging
import tiktoken
from google import genai

from backend.app.config import EMBED_MODEL
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
) -> list[float]:
    client = get_client(api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={"task_type": task_type},
    )

    token_count = get_embed_token_count(result, text)
    cost        = estimate_embed_cost(EMBED_MODEL, token_count)
    dims        = len(result.embeddings[0].values)

    log_llm_interaction(
        provider="google",
        model=EMBED_MODEL,
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
            llm_model=EMBED_MODEL,
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


def embed_query(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_search",
) -> list[float]:
    return _embed(text, "RETRIEVAL_QUERY", query_type, api_key, client_id)


def embed_document(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_document",
) -> list[float]:
    return _embed(text, "RETRIEVAL_DOCUMENT", query_type, api_key, client_id)


# Gemini's documented per-call input cap. Anything larger gets split into
# multiple requests below. 100 is the published ceiling for embedding-001
# at time of writing; we keep a 5-item safety buffer.
_MAX_EMBED_BATCH_SIZE = 95


def embed_documents_batch(
    texts: list[str],
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_document",
) -> list[list[float]]:
    """
    Embed many documents in a single Gemini API call. The whole list counts
    as ONE request against the per-minute and per-day quotas, so 50 product
    texts = 1 RPD instead of 50 RPD on the free tier.

    Returns vectors in the same order as the input texts. If an input list
    exceeds the per-call cap, it's split into chunks and the chunks are
    re-stitched in order. Each chunk is one API call.

    For per-document task tracking we still record one usage row per *call*
    with summed tokens — the dashboard's "request count" therefore reflects
    API calls, not product count, which is the honest accounting (you're
    actually spending one request).
    """
    if not texts:
        return []

    # Split into chunks if necessary so we never exceed the per-call cap.
    if len(texts) > _MAX_EMBED_BATCH_SIZE:
        out: list[list[float]] = []
        for i in range(0, len(texts), _MAX_EMBED_BATCH_SIZE):
            chunk = texts[i : i + _MAX_EMBED_BATCH_SIZE]
            out.extend(_embed_documents_chunk(chunk, api_key, client_id, query_type))
        return out

    return _embed_documents_chunk(texts, api_key, client_id, query_type)


def _embed_documents_chunk(
    texts: list[str],
    api_key: str,
    client_id: str,
    query_type: str,
) -> list[list[float]]:
    client = get_client(api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )

    vectors = [e.values for e in result.embeddings]
    if len(vectors) != len(texts):
        # Defensive: Gemini SDK guarantees alignment, but if anything ever
        # changes upstream we want to surface it loudly rather than silently
        # mis-pair vectors with their source items in the upsert phase.
        raise RuntimeError(
            f"Batch embed returned {len(vectors)} vectors for {len(texts)} inputs"
        )

    # Token + cost accounting. One DB row per API call, summed across the
    # batch. Total tokens and total cost come out identical to the per-item
    # path; only "request count" decreases (which is the whole point).
    total_tokens = sum(estimate_tokens(t) for t in texts)
    cost = estimate_embed_cost(EMBED_MODEL, total_tokens)
    dims = len(vectors[0]) if vectors else 0

    log_llm_interaction(
        provider="google",
        model=EMBED_MODEL,
        purpose=query_type,
        prompt=f"<batch of {len(texts)} documents>",
        response_text=f"<batch of {len(vectors)} embeddings, {dims} dims each>",
        input_tokens=total_tokens,
        output_tokens=0,
        cost=cost,
        client_id=client_id,
        extra={
            "task_type": "RETRIEVAL_DOCUMENT",
            "dims": dims,
            "batch_size": len(texts),
        },
    )

    try:
        track_usage(
            client_id=client_id,
            query_type=query_type,
            llm_provider="google",
            llm_model=EMBED_MODEL,
            input_tokens=total_tokens,
            output_tokens=0,
            input_cost=cost,
            output_cost=0.0,
            request_text_length=sum(len(t) for t in texts),
            response_text_length=0,
        )
    except Exception as e:
        logger.warning(f"Failed to track batch token usage: {e}")

    return vectors
