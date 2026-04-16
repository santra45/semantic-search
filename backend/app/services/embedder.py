import json
import logging
import tiktoken
from google import genai
from backend.app.config import GEMINI_API_KEY, EMBED_MODEL
from backend.app.services.token_usage_service import track_usage

# ---------------------------
# Logger Setup
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("embed_logger")


# ---------------------------
# Pricing Config
# ---------------------------
EMBED_PRICING = {
    "gemini-embedding-001": {"input": 0.000000149},  # $0.149 per 1M tokens
}


# ---------------------------
# Helper: Estimate tokens (fallback)
# ---------------------------
def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ---------------------------
# Helper: Get token count from embedding response
# ---------------------------
def get_embed_token_count(response, text: str) -> int:
    try:
        # Try to extract from response metadata if available
        if hasattr(response, "metadata") and response.metadata:
            token_count = getattr(response.metadata, "token_count", None)
            if token_count:
                return token_count
    except Exception as e:
        logger.warning(f"⚠️ Could not extract token count from response metadata: {e}")

    # Fallback: estimate using tiktoken
    return estimate_tokens(text)


# ---------------------------
# Helper: Estimate embedding cost
# ---------------------------
def estimate_embed_cost(model: str, token_count: int) -> float:
    pricing = EMBED_PRICING.get(model)
    if not pricing:
        return 0.0
    return token_count * pricing["input"]


# ---------------------------
# Client Init
# ---------------------------
def get_client(api_key: str = None):
    """Initialize Gemini client with provided API key or fallback to config."""
    return genai.Client(api_key=api_key)


# ---------------------------
# Embed Query
# ---------------------------
def embed_query(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_search",
) -> list[float]:
    """
    Embed a search query using the new SDK.
    Logs token usage and estimated cost.
    """
    client = get_client(api_key)

    logger.debug(
        f"\n{'='*60}\n"
        f"📤 GEMINI EMBED REQUEST\n"
        f"  Model: {EMBED_MODEL}\n"
        f"  Task Type: RETRIEVAL_QUERY\n"
        f"  Text (truncated): {text[:200]}{'...' if len(text) > 200 else ''}\n"
        f"{'='*60}"
    )

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"},
    )

    token_count = get_embed_token_count(result, text)
    cost = estimate_embed_cost(EMBED_MODEL, token_count)

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 GEMINI EMBED RESPONSE\n"
        f"  Task Type: RETRIEVAL_QUERY\n"
        f"  Embedding Dims: {len(result.embeddings[0].values)}\n"
        f"{'='*60}"
    )
    logger.info(f"🔢 Token Usage (query): {{ \"input\": {token_count}, \"total\": {token_count} }}")
    logger.info(f"💰 Estimated Cost (query): ${round(cost, 8)}")
    
    # Track token usage
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
            response_text_length=0
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to track token usage: {e}")

    return result.embeddings[0].values


# ---------------------------
# Embed Document
# ---------------------------
def embed_document(
    text: str,
    api_key: str = None,
    client_id: str = "anonymous",
    query_type: str = "embed_document",
) -> list[float]:
    """
    Embed a product document for indexing.
    Logs token usage and estimated cost.
    """
    client = get_client(api_key)

    logger.debug(
        f"\n{'='*60}\n"
        f"📤 GEMINI EMBED REQUEST\n"
        f"  Model: {EMBED_MODEL}\n"
        f"  Task Type: RETRIEVAL_DOCUMENT\n"
        f"  Text (truncated): {text[:200]}{'...' if len(text) > 200 else ''}\n"
        f"{'='*60}"
    )

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )

    token_count = get_embed_token_count(result, text)
    cost = estimate_embed_cost(EMBED_MODEL, token_count)

    logger.debug(
        f"\n{'='*60}\n"
        f"📥 GEMINI EMBED RESPONSE\n"
        f"  Task Type: RETRIEVAL_DOCUMENT\n"
        f"  Embedding Dims: {len(result.embeddings[0].values)}\n"
        f"{'='*60}"
    )
    logger.info(f"🔢 Token Usage (document): {{ \"input\": {token_count}, \"total\": {token_count} }}")
    logger.info(f"💰 Estimated Cost (document): ${round(cost, 8)}")
    
    # Track token usage
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
            response_text_length=0
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to track token usage: {e}")

    return result.embeddings[0].values