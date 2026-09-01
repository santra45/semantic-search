import logging
import tiktoken
from google import genai

from backend.app.config import EMBED_MODEL
from backend.app.services import usage_service
from backend.app.services.cache_service import get_cached_embedding, set_cached_embedding
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


# ── Indexing spend vs serving spend ─────────────────────────────────────────
#
# usage_events.kind separates the two, and this is the only write site in the
# system where both are possible, so the decision has to be made here rather
# than by a lookup table on call_type.
#
# Derived from task_type and NOT from query_type, deliberately. query_type is
# free text chosen by the caller: embed_query() defaults it to 'embed_search'
# but chat_retrieval_service passes 'chat_context', and any future caller can
# pass anything at all. task_type is set by the two wrappers below and by
# nothing else — RETRIEVAL_DOCUMENT is reachable only through embed_document(),
# which is only ever called on an indexing path. Branching on the caller's
# label instead would mean one novel query_type filing a catalogue sync under
# shopper traffic, which is exactly the mixing `kind` exists to prevent and
# which no test would catch.
_SYNC_TASK_TYPE = "RETRIEVAL_DOCUMENT"


def _kind_for(task_type: str) -> str:
    return usage_service.KIND_SYNC if task_type == _SYNC_TASK_TYPE else usage_service.KIND_SERVE


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

    # The tenant comes from the request context, not from `client_id`.
    #
    # This function receives a bare client_id and nothing else, and a client_id
    # cannot name a site, a subscription or a product — so it cannot produce a
    # usage_events row at all. The alternative was threading a ctx argument down
    # to embed_query()/embed_document(), which have 25 call sites between them;
    # the sites that diff missed would go on writing unattributable rows with
    # nothing to make them visible. usage_service.track() reads what the auth
    # chokepoint bound for this request and opens its own short-lived session,
    # so this call site keeps the signature it has always had.
    #
    # billable stays False on every embedding. Exactly one row per
    # customer-visible action carries it — chat_answer for the Magento chatbot,
    # wp_product_qa for the Woo widget — and a single turn embeds the query
    # once for /retrieve/products and again for /retrieve/content, plus once
    # more per active-retrieval tool call. Flagging them would bill a merchant
    # three or four requests for one shopper question.
    #
    # Swallowed but never silent: an accounting failure must not be the reason
    # an embedding call fails, since the vector is already paid for by here.
    # The line carries the tenant, the model and the amount so the spend is
    # recoverable from the log even when the row is not.
    try:
        usage_service.track(
            query_type,
            "google",
            model,
            token_count,
            0,
            cost,
            0.0,
            _kind_for(task_type),
        )
    except Exception as e:
        logger.warning(
            "usage not recorded for %s (client=%s model=%s tokens=%d cost=%s): %s",
            query_type, client_id, model, token_count, cost, e,
        )

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
