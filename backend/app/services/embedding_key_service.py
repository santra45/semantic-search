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
in memory, exactly like the LLM key. That licence key is the key-encryption
key, which means reissuing it orphans both blobs at once — see the re-wrap
requirement in llm_key_service._kek(), and resolve_embedding_key() below for
what that costs us when it happens.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.config import EMBED_MODEL
from backend.app.services.llm_key_service import (
    decrypt_key,
    licence_for_log,
    tenant_for_log,
)

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

    TREATED AS ABSENT, NO LONGER TREATED AS UNREMARKABLE. Falling back is
    harmless only when the tenant configured nothing. When they DID configure a
    key and it would not open, the identical fallback moves their embedding
    spend onto OUR Google budget: embedder.get_client(None) builds
    genai.Client(api_key=None), which reads the server key out of the
    environment and succeeds. No layer above this one can separate the two
    cases — a router sees None either way — so they are separated here, and
    only the second one logs.

    The partial case is covered too and needs no line of its own: a tenant
    whose embedding blob is orphaned but whose LLM blob still opens gets the
    LLM key back (correct — that is the documented fallback order) and
    decrypt_key has already logged which of the two failed and why.
    """
    # Whether the tenant has anything stored at all. This is the entire
    # difference between "never configured a key", which is most installs and
    # must stay quiet, and "configured one we cannot open", which is money.
    configured = False

    for encrypted, source in ((embedding_encrypted, "embedding"), (llm_encrypted, "llm")):
        if not encrypted:
            continue
        configured = True
        try:
            key = decrypt_key(encrypted, license_key, purpose=f"{source} key")
        except Exception as exc:
            # decrypt_key does not raise, by contract: it logs and returns
            # None. This stays as belt and braces for the day that changes, and
            # is NOT the failure path — a key that would not open arrives below
            # as a falsy `key`, which is what the fallback warning keys off.
            # Reading this except block as the failure handler is what let the
            # silent fallback survive: it has never once executed.
            logger.warning(
                "%s key: decrypt_key raised instead of returning None (%s: %s). "
                "%s",
                source,
                type(exc).__name__,
                exc,
                tenant_for_log(),
            )
            continue
        if key:
            return key

    if configured:
        # Carries its own tenant and KEK fields rather than saying "see the
        # line above". Under concurrency there is no line above: uvicorn
        # interleaves requests and the decrypt_key warning for THIS tenant can
        # be several other tenants' lines back.
        logger.warning(
            "embedding key: FALLING BACK TO THE SERVER KEY - this tenant has "
            "at least one stored provider key and none of them could be "
            "unwrapped, so these embeddings are billed to OUR Google account "
            "instead of theirs. %s %s. llm_key_service.decrypt_key logged "
            "which key failed and why.",
            tenant_for_log(),
            licence_for_log(license_key),
        )

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
