"""Resolving the tenant's web-search key.

ONE FIELD. NO FALLBACK CHAIN. THAT IS THE WHOLE DESIGN.
-------------------------------------------------------
`embedding_key_service.resolve_embedding_key()` falls back — embedding key, then
LLM key, then the server's own. That is correct there: an embedding call has to
happen for sync to work at all, and a tenant who configured nothing has always
been served on the server key.

This module deliberately does NOT do that, and the reason is money rather than
taste. Web grounding is reached from `Controller/Ajax/Ask` and
`Controller/Ajax/Prime`, which are public storefront endpoints with **no CSRF, no
rate limit, and no length cap on the question**, while backend quota enforcement
is still env-gated off (`AICHATBOT_QUOTA_ENFORCEMENT`, see request_auth.py). A
fallback chain would attach an uncapped, per-call third-party cost to a URL
anyone can POST to in a loop.

There is also a subtler reason not to reach for the embedding key specifically.
Every tenant HAS one and it is guaranteed to be a Google key, because the
embedding provider dropdown is locked to google/gemini-embedding-001 — so it
looks like the obvious answer. Using it would put *generation* spend on the key
the merchant set up for *embeddings*: the bill lands on the right vendor account
under the wrong heading, every dashboard that separates indexing spend from
serving spend quietly starts lying, and nobody notices for a month. The two-key
separation shipped in 2026-08 exists precisely to prevent that.

So: no dedicated key means the feature is OFF for that tenant. Silently, cheaply,
and with a log line saying so.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.services.llm_key_service import decrypt_key

logger = logging.getLogger(__name__)


def resolve_web_search_key(
    web_search_encrypted: Optional[str],
    license_key: str,
) -> Optional[str]:
    """The tenant's own web-search API key, or None.

    None always means "web grounding is off for this request". Callers must treat
    it as a feature flag, never as "use the server's key".

    A key that will not decrypt is reported at WARNING and treated as absent. It
    is worth its own line because it is not the same event as never configuring
    one: the merchant believes the feature is on, and it is not.
    """
    if not web_search_encrypted:
        # Not configured. The overwhelmingly common case — this feature ships
        # OFF — so it stays quiet. A log line here would drown the warning below
        # within a day and stop meaning anything.
        return None

    key = decrypt_key(web_search_encrypted, license_key, purpose="web search key")
    if not key:
        logger.warning(
            "web search key: a key IS configured for this tenant but could not be "
            "unwrapped, so web grounding is disabled for this request. The most "
            "common cause is a reissued licence key — the licence is the KEK, so "
            "reissuing it invalidates every blob wrapped under the old one and the "
            "merchant must re-save the field. llm_key_service.decrypt_key logged "
            "which key failed and why."
        )
        return None

    return key
