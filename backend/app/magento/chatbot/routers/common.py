"""Shared helpers for the agent routers — auth, credential resolution, context building."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.services import (
    auth_cache,
    licensing_service,
    request_context,
    usage_service,
)
# Aliased because `license_key` is the name of the local holding the presented
# key in every function below, and a module shadowed by a local is the kind of
# NameError that only fires on the branch nobody exercised.
from backend.app.services import license_key as license_key_format
from backend.app.services.domain_auth_service import DomainAuthorizer
from backend.app.services.license_service import (
    check_search_quota,
    extract_license_key_from_authorization,
    validate_license_key,
)
from backend.app.services.llm_key_service import decrypt_key

# Two different modules are called request_context. This one is the agents'
# per-request value object (Magento client, store code, quote id) and only its
# CLASS is imported, so it does not shadow the services module bound above —
# which is the ambient tenant context the usage writers read. Read the services
# module's docstring before renaming either.
from backend.app.magento.chatbot.agents.request_context import RequestContext
from backend.app.magento.chatbot.services import admin_token_service, magento_creds_service
from backend.app.magento.chatbot.services.magento_client import MagentoClient


def resolve_license_key(
    authorization: Optional[str],
    request_license: Optional[str],
) -> Optional[str]:
    return extract_license_key_from_authorization(authorization) or request_license


# ── v2 licensing: the dual-read window ───────────────────────────────────────
#
# EVERYTHING FROM HERE TO authorize_request() IS DUPLICATED, deliberately and
# verbatim in behaviour, in backend/app/wordpress/productqa/services/common.py.
# The two chokepoints have always been twins — resolve_license_key,
# _enforce_search_quota and decrypt_llm_key are already copied between them —
# and the WordPress file's docstring makes a point of not importing anything
# from the Magento package. Change one, change the other in the same commit.

# Which v2 alias fills which v1 contract key. RENAMED, never replaced:
# license_data is read at roughly a hundred sites across this package and the
# WordPress twin, and the WHOLE dict is additionally handed by value to
# DomainAuthorizer.validate_request, _llm_rerank and _process_chunkable_item —
# so a v1 name that quietly stops being produced breaks callers that no per-key
# grep will surface. Adding keys is safe. Dropping one is not.
#
# The other six contract keys (client_id, domain, plan, client_name,
# product_code, platform) already spell identically in both shapes and so are
# carried over by the plain dict copy in _license_data_from_v2_context().
_V2_TO_V1_KEYS = {
    "licence_id": "license_id",              # v2 uses British spelling throughout
    "catalogue_limit": "product_limit",      # per-SITE under v2, was per-licence
    "request_limit": "search_limit",         # per-SUBSCRIPTION, was per-licence
    "licence_expires_at": "license_expires",
}

# Proof at IMPORT that every source name above still exists in a resolved
# context, in the shape usage_service._context_subset() uses. Without it a
# rename in licensing_service._context_from_row() would not fail here — this
# file would simply stop producing product_limit, and the first symptom would
# be a KeyError in a sync router on a live request days later. An application
# that refuses to boot is a five-minute problem; a chokepoint that silently
# drops a contract key is the failure this whole rewrite exists to end.
_UNKNOWN_V2_FIELDS = sorted(set(_V2_TO_V1_KEYS) - licensing_service.CONTEXT_FIELDS)
if _UNKNOWN_V2_FIELDS:
    raise ImportError(
        "The Magento chokepoint maps " + ", ".join(_UNKNOWN_V2_FIELDS) + " out of "
        "the resolved licence context onto the v1 license_data contract, but "
        "licensing_service.CONTEXT_FIELDS no longer contains "
        + ("them" if len(_UNKNOWN_V2_FIELDS) > 1 else "it") + ". "
        "_context_from_row() changed shape — update _V2_TO_V1_KEYS here AND in "
        "backend/app/wordpress/productqa/services/common.py rather than putting "
        "the old name back. The context now carries: "
        + ", ".join(sorted(licensing_service.CONTEXT_FIELDS)) + "."
    )

# Stamped onto every license_data so a log line, the dashboard and any future
# migration report can tell which resolver answered without re-deriving it from
# the key. Today this reads "v1" on 100% of traffic: the licences table is
# empty, and a v1 JWT cannot be hashed forward into a v2 key.
AUTH_PATH_V2 = "v2"
AUTH_PATH_V1 = "v1"


def _resolve_v2_context(db: Session, license_key: str) -> Optional[dict]:
    """Resolve an opaque v2 licence key — cache first — or None.

    None means "not a v2 key, or not authorised", and the caller must fall
    through to the v1 JWT path. Those two are deliberately not distinguished:
    resolve_key() already returns None for both, and a caller that could tell
    them apart would be a probe oracle.

    looks_valid() before anything else is what makes the dual read free. A v1
    JWT fails the shape and checksum test without a hash, a Redis round trip or
    a query, so during the migration window — when every key in the database is
    still a JWT — this function costs one regex-ish string check per request.

    Exceptions are NOT swallowed. A key that passed looks_valid() is a genuine
    v2 key, and if MySQL or the context shape is broken the honest answer is a
    500. Falling through to validate_license_key() would decode a non-JWT and
    raise ValueError, which the caller turns into "403 Invalid license key" —
    telling a paying merchant their key is bad when the truth is that our
    database is down.
    """
    if not license_key_format.looks_valid(license_key):
        return None

    key_hash = license_key_format.hash_key(license_key)

    # auth_cache handles its own failures and answers None for "go ask MySQL",
    # so a dead Redis degrades to the pre-cache behaviour rather than a 500.
    cached = auth_cache.get(key_hash)
    if cached is not None:
        return cached

    context = licensing_service.resolve_key(db, license_key)
    if context is not None:
        # Populate only on success. Caching a refusal would keep a merchant
        # locked out for the TTL after the admin who fixed their subscription
        # has already told them it works now.
        auth_cache.put(key_hash, context)
    return context


def _license_data_from_v2_context(context: dict) -> dict:
    """A resolved v2 context, wearing the v1 license_data contract as well.

    Both shapes, one dict. The v1 keys exist because ~110 call sites read them;
    the v2 keys ride along because usage_service.record() needs six of them
    (client_id, site_id, subscription_id, product_code, platform, key_owner)
    and would otherwise refuse the row as unattributable.

    Note what changes underneath a caller that reads `domain`: v1 took it from
    a JWT claim frozen at issue time and never once selected
    license_keys.allowed_domain, so editing a tenant's domain in the database
    has had no effect at all. v2 supplies sites.domain, a live row, and
    DomainAuthorizer gates every request on it. A tenant whose stored domain
    has drifted from their issued token gets a different verdict the moment
    their key resolves through here — diff the two before cutting anyone over.
    """
    license_data = dict(context)
    for v2_name, v1_name in _V2_TO_V1_KEYS.items():
        license_data[v1_name] = context.get(v2_name)
    return license_data


def _interaction_id_for(request: Request) -> str:
    """The id every usage row of this turn will share.

    Header first. One shopper turn is three or four HTTP requests — tool-call,
    retrieve products, retrieve content, answer — each with its own call to
    this chokepoint, so minting per call would split a turn across four ids and
    orphan the retrieval spend from the answer that paid for it. AIChatbot's
    RequestTimer already stamps one per-turn id as X-Request-Id on every
    backend call it makes, and it is already capped at the 64 characters the
    column allows.

    The other modules (AIProductQA, AI Search, both Woo plugins) do not send
    the header yet, so for them the mint below really is one id per HTTP
    request. That is honest grouping of what we can see, not per-turn grouping;
    fixing it is a plugin-side change.
    """
    header = request.headers.get(request_context.INTERACTION_ID_HEADER)
    return (
        request_context.interaction_id_from_header(header)
        or usage_service.new_interaction_id()
    )


def authorize_request(
    *,
    request: Request,
    db: Session,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
) -> dict:
    """Authenticate the caller, bind their tenant context, return license_data.

    Dual read, not a switchover: an opaque v2 key resolves through
    licensing_service.resolve_key(); anything else falls back to the v1 JWT
    decoder exactly as before. Both paths produce the same dict contract plus
    an `auth_path` discriminator.

    MUST STAY A PLAIN CALL INSIDE THE HANDLER BODY. Refactoring this into a
    FastAPI Depends() looks tidier and silently breaks every usage row:
    measured in the container, a ContextVar bound inside a sync dependency is
    invisible in the handler, because each sync dependency is its own
    run_in_threadpool dispatch with its own context copy that is discarded on
    return. bind_context() below only reaches the write sites because this runs
    in the same context as the code that calls it.

    Raises 401 (no key), 403 (bad key / wrong domain) or 429 (over quota).
    """
    license_key = resolve_license_key(authorization, request_license)
    if not license_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    context = _resolve_v2_context(db, license_key)
    if context is not None:
        license_data = _license_data_from_v2_context(context)
        license_data["auth_path"] = AUTH_PATH_V2
    else:
        try:
            license_data = validate_license_key(license_key, db)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        license_data["auth_path"] = AUTH_PATH_V1

    DomainAuthorizer(db).validate_request(request, license_data, api_key=x_api_key)
    _enforce_search_quota(db, license_data)

    # Both paths. The v2 context stores only a hash and a display prefix, but
    # llm_key_service derives its AES key as sha256(license_key), so the
    # PLAINTEXT presented key is the KEK for every merchant-supplied LLM and
    # embedding key and is read at ~20 sites downstream. Dropping it here does
    # not raise — decrypt_llm_key() returns None on failure — it silently falls
    # back to the server's own key and nobody notices for a month.
    license_data["license_key"] = license_key
    license_data["interaction_id"] = _interaction_id_for(request)

    # Bound AFTER the 403/429 gates so a refused request never leaves a tenant
    # identity behind, and bound with the SAME dict that is returned so a shared
    # service reading it out of band sees exactly what a handler holding the
    # local sees. Nothing to reset: see request_context's module docstring for
    # the measurements showing the binding cannot outlive the request.
    # Bound ONLY on the v2 path. A v1 JWT produces a perfectly truthy dict that
    # carries none of the six tenant identifiers, and record() tested the
    # context for truthiness rather than for shape - so a v1 request skipped the
    # NO CONTEXT branch, reached _tenant_fields(), and raised. Measured: twelve
    # real authenticated requests burned real embedding tokens and wrote zero
    # rows to usage_events, with the loss reported as CALLER BUG at ERROR
    # against innocent call sites.
    #
    # During the dual-read window a v1 request genuinely HAS no v2 identity.
    # Leaving the scope empty is the honest representation of that, and it is
    # the state record() already knows how to describe.
    if license_data.get("auth_path") == AUTH_PATH_V2:
        request_context.bind_context(license_data)
    return license_data


def _enforce_search_quota(db: Session, license_data: dict) -> None:
    """Reject over-quota tenants with 429 instead of only tracking usage.

    Env-gated OFF by default (AICHATBOT_QUOTA_ENFORCEMENT=1 to arm): a wrong
    limit or a stale usage row would take a whole storefront down, so this
    ships disabled until the plan + usage data is verified in the target env.
    Fails OPEN on any lookup error — a quota check must never be the reason a
    paying merchant's bot goes dark.

    TWO COUNTERS, PICKED BY WHICH RESOLVER ANSWERED. The v2 context names its
    allowance `request_limit` and meters it per SUBSCRIPTION in usage_counters;
    v1 named it `search_limit` and metered it per CLIENT in usage_logs. Reading
    a v2 context through the v1 branch is not a smaller check, it is no check
    at all: neither v1 name is present, the int() falls to 0, and the
    `search_limit <= 0` early exit returns — no log, no exception, no 429 ever,
    with the env flag set to 1 and the whole thing still looking armed.

    Do not arm this yet either way. usage_counters is only populated once the
    write sites are wired, and license_service.check_search_quota() reads
    usage_logs, which the v2 migration renamed to usage_logs_archive_v1 — so
    the v1 branch below is already a guaranteed no-op that the outer `except`
    swallows without a word.
    """
    if os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "0") != "1":
        return

    subscription_id = license_data.get("subscription_id")
    if subscription_id:
        # v2: one primary-key read of usage_counters. within_request_quota()
        # already fails open internally and hands back (ok, used, limit)
        # precisely so the refusal can name both numbers — "over your plan
        # limit" without them generates a support ticket every time. The outer
        # try stays as belt and braces for an import-time or argument error the
        # function itself cannot catch.
        try:
            ok, used, limit = usage_service.within_request_quota(
                db,
                str(subscription_id),
                int(license_data.get("request_limit") or 0),
            )
        except Exception:
            return  # fail open — never block on a lookup/DB error
        if not ok:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Monthly usage limit reached ({used}/{limit}). "
                    "Please contact the store."
                ),
            )
        return

    # v1 JWT path — behaviour untouched. Known dead weight, left in place so
    # this change carries no v1 behaviour risk: `search_limit_per_month` has
    # never once fired, because validate_license_key() renames that column to
    # `search_limit` on the way out, and check_search_quota() reads a table the
    # migration archived. Delete both when the v1 path itself is retired.
    try:
        client_id = license_data.get("client_id")
        search_limit = int(
            license_data.get("search_limit_per_month")
            or license_data.get("search_limit")
            or 0
        )
        if not client_id or search_limit <= 0:
            return  # no usable limit configured → don't block
        within_quota = check_search_quota(db, str(client_id), search_limit)
    except Exception:
        return  # fail open — never block on a lookup/DB error
    if not within_quota:
        raise HTTPException(
            status_code=429,
            detail="Monthly usage limit reached. Please contact the store.",
        )


def maybe_persist_magento_creds(
    *,
    db: Session,
    client_id: str,
    license_key: str,
    encrypted_creds_header: Optional[str],
) -> None:
    """If the module included an admin-creds blob, decrypt + persist it (and reset the cached token)."""
    if not encrypted_creds_header:
        return
    if magento_creds_service.store_credentials_from_header(
        db, client_id, license_key, encrypted_creds_header
    ):
        admin_token_service.invalidate_token(client_id)


def decrypt_llm_key(encrypted: Optional[str], license_key: str) -> Optional[str]:
    if not encrypted:
        return None
    try:
        return decrypt_key(encrypted, license_key)
    except Exception:
        return None


async def build_request_context(
    *,
    db: Session,
    license_data: dict,
    store_code: str,
    customer_id: Optional[str],
    is_customer_login: bool,
    guest_session_id: Optional[str],
    quote_id: Optional[str],
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_api_key_encrypted: Optional[str],
) -> RequestContext:
    """Resolve credentials + mint token + construct a fully-wired RequestContext."""
    ctx = RequestContext(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        license_key=license_data["license_key"],
        store_code=store_code or "default",
        customer_id=int(customer_id) if customer_id and str(customer_id).isdigit() else None,
        is_customer_login=bool(is_customer_login),
        guest_session_id=guest_session_id,
        quote_id=quote_id or None,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=decrypt_llm_key(llm_api_key_encrypted, license_data["license_key"]),
    )

    creds = magento_creds_service.get_credentials(db, license_data["client_id"])
    if creds:
        token = await admin_token_service.get_or_mint_token(
            client_id=license_data["client_id"], creds=creds
        )
        if token:
            magento_creds_service.touch_last_mint(db, license_data["client_id"])
        ctx.magento_client = MagentoClient(
            client_id=license_data["client_id"],
            base_url=creds["base_url"],
            api_version=creds.get("api_version", "V1"),
            verify_ssl=bool(creds.get("verify_ssl", True)),
            store_code=ctx.store_code or creds.get("default_store_code", "default"),
            admin_token=token,
        )
    return ctx
