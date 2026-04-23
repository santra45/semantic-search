import logging
import tiktoken
from google import genai

from backend.app.config import EMBED_MODEL
from backend.app.services.token_usage_service import track_usage
from backend.app.utils.llm_logger import log_llm_interaction

logger = logging.getLogger("embed_logger")


# ── Pricing (per-token) ─────────────────────────────────────────────────────

EMBED_PRICING = {
    "gemini-embedding-001": {"input": 0.000000149},  # $0.149 per 1M tokens
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
