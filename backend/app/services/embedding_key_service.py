"""Which key — and which model — pays for embeddings.

Until now embeddings were billed to whatever key the tenant configured for
chat completions: every router decrypted `X-LLM-API-Key-Encrypted` into a
variable literally named `embedding_api_key` and handed it to the embedder.
That works, but it conflates two spends that tenants want separated — chat
tokens are priced per-provider and change with the model they pick, while
embedding tokens are a flat Google cost incurred by syncs they don't
directly trigger.

The modules and plugins now ship a second key field, so this module resolves
the pair:

  * `X-Embedding-API-Key-Encrypted` if the tenant configured one,
  * otherwise the existing LLM key, which is what every deployed install
    sends today and must keep working untouched.

Both arrive encrypted under the tenant's license key and are decrypted here,
in memory, exactly like the LLM key.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.config import EMBED_MODEL
from backend.app.services.llm_key_service import decrypt_key

logger = logging.getLogger(__name__)


# ── Provider / model allowlists ─────────────────────────────────────────────
#
# Deliberately narrow. An embedding model is not a free choice the way a chat
# model is: the vectors already written to Qdrant have gemini-embedding-001's
# dimensionality, and a collection holds one shape. Honouring an arbitrary
# model id would either 400 on every upsert or, worse, silently write vectors
# that never match anything at query time.
#
# This is why the admin dropdowns ship locked rather than open. When a second
# embedding provider is supported, it is added here and the dropdowns unlock
# — the modules and plugins already put both values on the wire, so nothing
# downstream of this dict has to change.

DEFAULT_EMBEDDING_PROVIDER = "google"

# `gemini` is an accepted spelling of `google`: the Magento modules store
# `google`, the WooCommerce plugins spell the LLM equivalent `gemini`, and a
# merchant running both should not have to know the difference.
PROVIDER_ALIASES = {
    "google": "google",
    "gemini": "google",
}

SUPPORTED_EMBEDDING_MODELS = {
    "google": {"gemini-embedding-001"},
}


def resolve_embedding_key(
    embedding_encrypted: Optional[str],
    llm_encrypted: Optional[str],
    license_key: str,
) -> Optional[str]:
    """The tenant's embedding key, or the LLM key, or None.

    None means "use whatever the server has", which is the pre-existing
    behaviour for installs that configured neither. A key that fails to
    decrypt is treated as absent rather than fatal: a corrupted option value
    in someone's admin should degrade to the server key, not 500 every sync.
    """
    for encrypted, source in ((embedding_encrypted, "embedding"), (llm_encrypted, "llm")):
        if not encrypted:
            continue
        try:
            key = decrypt_key(encrypted, license_key)
        except Exception as exc:
            logger.warning("%s key failed to decrypt, skipping: %s", source, exc)
            continue
        if key:
            return key
    return None


def resolve_embedding_provider(requested: Optional[str] = None) -> str:
    """Normalised provider name, falling back to the default on anything
    unrecognised."""
    if not requested:
        return DEFAULT_EMBEDDING_PROVIDER
    provider = PROVIDER_ALIASES.get(requested.strip().lower())
    if provider:
        return provider
    logger.warning(
        "unsupported embedding provider %r requested, falling back to %s",
        requested,
        DEFAULT_EMBEDDING_PROVIDER,
    )
    return DEFAULT_EMBEDDING_PROVIDER


def resolve_embedding_model(
    requested: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """The model to embed with, validated against the provider's allowlist.

    Anything unsupported falls back to the server default, loudly — a tenant
    whose config asks for something we can't honour should show up in the
    logs, not in a corrupted collection.
    """
    provider = resolve_embedding_provider(provider)
    if not requested:
        return EMBED_MODEL
    model = requested.strip()
    if model in SUPPORTED_EMBEDDING_MODELS.get(provider, ()):
        return model
    logger.warning(
        "unsupported embedding model %r for provider %s, falling back to %s",
        model,
        provider,
        EMBED_MODEL,
    )
    return EMBED_MODEL
