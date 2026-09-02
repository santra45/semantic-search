"""The one authorisation chokepoint. Every authenticated request passes here.

WHY THIS MODULE EXISTS
----------------------
It did not, and that was the problem. The dual-read window was implemented
twice — once in backend/app/magento/chatbot/routers/common.py and once in
backend/app/wordpress/productqa/services/common.py — and both files carried a
comment instructing the next reader to change one whenever they changed the
other. Measured before this module was extracted: nine of the ten shared
functions were byte-identical once docstrings were stripped, and the tenth
(_assert_product_allowed) differed only in where it got its allowed-product
set from. That is not two implementations, it is one implementation and a copy
waiting to drift.

The generic routers forced the issue. /api/search, /api/magento/search,
/api/sync/*, /api/dashboard/*, /api/token-usage/me/* and
/api/register-webhook-secret serve BOTH platforms, so they cannot import
either product's chokepoint without a package importing its sibling. They
authenticated inline against validate_license_key() instead and therefore got
no v2 resolution, no product gate and no usage context — which is exactly the
gap this module closes.

WHAT A CALLER MUST DECIDE
-------------------------
Only one thing: *allowed_products*, the set of product codes permitted to call
this endpoint. It is a REQUIRED keyword with no default, because the failure
mode of forgetting it is silent — an endpoint outside the product gate is how
a one-product key ends up driving every product, and that bypass is the whole
reason the gate exists. Passing None is allowed and means "no mapping", which
fails closed for v2 keys.

Callers resolve their own set because they know their own shape:

  - the Magento chatbot chokepoint keys it per route, since three products
    share its endpoints
  - the WordPress ProductQA chokepoint uses one constant for its whole package
  - the generic routers state theirs literally, since no table can key an
    endpoint two platforms share

MUST STAY A PLAIN CALL INSIDE THE HANDLER BODY. Refactoring authorize_request
into a FastAPI Depends() looks tidier and silently breaks every usage row:
measured in the container, a ContextVar bound inside a sync dependency is
invisible in the handler, because each sync dependency is its own
run_in_threadpool dispatch with its own context copy that is discarded on
return. bind_context() only reaches the write sites because it runs in the
same context as the code that calls it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.services import (
    auth_cache,
    catalog,
    licensing_service,
    request_context,
    tenancy_service,
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

logger = logging.getLogger(__name__)


# Stamped onto every license_data so a log line, the dashboard and any future
# migration report can tell which resolver answered without re-deriving it from
# the key.
AUTH_PATH_V2 = "v2"
AUTH_PATH_V1 = "v1"


# Which v2 alias fills which v1 contract key. RENAMED, never replaced:
# license_data is read at roughly a hundred sites across both product packages
# and the generic routers, and the WHOLE dict is additionally handed by value
# to DomainAuthorizer.validate_request, _llm_rerank and _process_chunkable_item
# — so a v1 name that quietly stops being produced breaks callers that no
# per-key grep will surface. Adding keys is safe. Dropping one is not.
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
# module would simply stop producing product_limit, and the first symptom would
# be a KeyError in a sync router on a live request days later. An application
# that refuses to boot is a five-minute problem; a chokepoint that silently
# drops a contract key is the failure this whole rewrite exists to end.
_UNKNOWN_V2_FIELDS = sorted(set(_V2_TO_V1_KEYS) - licensing_service.CONTEXT_FIELDS)
if _UNKNOWN_V2_FIELDS:
    raise ImportError(
        "The shared chokepoint maps " + ", ".join(_UNKNOWN_V2_FIELDS) + " out of "
        "the resolved licence context onto the v1 license_data contract, but "
        "licensing_service.CONTEXT_FIELDS no longer contains "
        + ("them" if len(_UNKNOWN_V2_FIELDS) > 1 else "it") + ". "
        "_context_from_row() changed shape — update _V2_TO_V1_KEYS here rather "
        "than putting the old name back. The context now carries: "
        + ", ".join(sorted(licensing_service.CONTEXT_FIELDS)) + "."
    )


def resolve_license_key(
    authorization: Optional[str],
    request_license: Optional[str],
) -> Optional[str]:
    """Header wins over body. Every module sends the header; the body field
    exists so a merchant debugging with curl doesn't have to."""
    return extract_license_key_from_authorization(authorization) or request_license


def decrypt_llm_key(encrypted: Optional[str], license_key: str) -> Optional[str]:
    """The module ships the merchant's LLM key still encrypted; the license key
    is the secret. Returns None on any failure so the caller falls back to the
    server's own key rather than 500ing on a corrupted option value."""
    if not encrypted:
        return None
    try:
        return decrypt_key(encrypted, license_key)
    except Exception:
        return None


def _resolve_v2_context(db: Session, license_key: str) -> Optional[dict]:
    """Resolve an opaque v2 licence key — cache first — or None.

    None means "not a v2 key, or not authorised", and the caller must fall
    through to the v1 JWT path. Those two are deliberately not distinguished:
    resolve_key() already returns None for both, and a caller that could tell
    them apart would be a probe oracle.

    looks_valid() before anything else is what makes the dual read free. A v1
    JWT fails the shape and checksum test without a hash, a Redis round trip or
    a query, so during the migration window — when almost every key in the
    database is still a JWT — this function costs one string check per request.

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


def _assert_collection_name_agrees(context: dict) -> None:
    """Refuse a v2 context whose stored collection name is not the derived one.

    THE VALUE IS CARRIED AND READ BY NOBODY. _license_data_from_v2_context()
    copies `collection_name` out of the resolved context, and all ten Qdrant
    call sites still derive the name themselves from (client_id, domain).
    licensing_service._context_from_row() says why the value is stored and
    "never recomputed from `domain`": get_collection_name maps
    shop.example.com, shop-example-com and shop_example_com onto the same
    string, live collections were named from an UNNORMALISED host, and
    sites.domain is normalised by tenancy_service.normalise_domain() —
    lowercased, no www, no port. So the two can differ, and when they do, every
    read goes to a collection that does not exist. Qdrant answers a missing
    collection with zero results rather than an error: the store simply goes
    quiet, with nothing in the logs, days after whatever caused the drift.

    WHY AN ASSERTION HERE RATHER THAN AT ISSUANCE. Issuance only covers licences
    minted after the check lands; this is the one place EVERY v2 request passes,
    so it covers the ones already out in the field too.

    WHY IT REFUSES RATHER THAN LOGS AND CONTINUES. On a mismatch we cannot tell
    from here which of the two names holds the points — the stored one if the
    collection predates domain normalisation, the derived one if the store has
    re-synced since, because a sync derives the name exactly as a read does.
    That is not a decision to make silently on a shopper's request.

    500 rather than 403, deliberately: the caller's key is perfectly good and
    telling a paying merchant it is not would be a lie. _resolve_v2_context()
    already settled this same question the same way.
    """
    stored = context.get("collection_name")
    derived = tenancy_service.derive_collection_name(
        str(context.get("client_id") or ""),
        str(context.get("domain") or ""),
    )
    if stored and stored == derived:
        return

    # Both names, at ERROR, because the remediation is "go and look at which of
    # these two collections actually holds points" and neither name is
    # recoverable from the other.
    logger.error(
        "licence resolved with a collection name nothing will read: "
        "sites.collection_name=%r but every Qdrant call site derives %r from "
        "client_id=%s domain=%r. Reads against the derived name return zero "
        "results rather than an error, so this request is refused instead of "
        "answered emptily. Check which collection holds the store's points and "
        "re-seed sites.collection_name from it.",
        stored,
        derived,
        context.get("client_id"),
        context.get("domain"),
    )
    raise HTTPException(
        status_code=500,
        detail=(
            "This store's search index is misconfigured on our side. "
            "Please contact support — no change is needed to your license key."
        ),
    )


def product_label(code: str) -> str:
    """The customer-facing name plus the code: AI Chatbot (magento_chatbot).

    The name alone is ambiguous (two products are called "AI Product Q&A", one
    per platform) and the code alone means nothing to the merchant reading the
    403 in their admin log. Both, or the refusal generates a support ticket.
    """
    product = catalog.get_product(code)
    return f"{product['name']} ({code})" if product else code


def _assert_product_allowed(
    request: Request,
    license_data: dict,
    allowed_products: Optional[Union[str, frozenset, set]],
) -> None:
    """403 unless the licence's product is one this endpoint serves.

    SKIPPED ENTIRELY ON THE v1 PATH, and that is not laziness. v1 keys predate
    per-product licensing: license_service.create_license_key() takes
    product_code as an OPTIONAL argument and all four local keys carry NULL, so
    the honest reading of a v1 key is "all products" and enforcing against it
    would 403 tenants who did nothing wrong. Worse, the v1 keys that DO name a
    product were issued when one key was expected to serve all three Magento
    modules on a store — turning that into a refusal would take working
    storefronts down to fix an accounting problem that only exists on the v2
    path, where record() reads the product off the context.

    The `product_code is None` guard below is therefore belt and braces for the
    same shape arriving through v2, not a duplicate of the auth_path test.
    """
    if license_data.get("auth_path") != AUTH_PATH_V2:
        return

    product_code = license_data.get("product_code")
    if not product_code:
        return

    if allowed_products is None:
        allowed = None
    elif isinstance(allowed_products, str):
        allowed = frozenset({allowed_products})
    else:
        allowed = frozenset(allowed_products)

    if allowed is None:
        # FAIL CLOSED. An endpoint with no mapping is an endpoint outside the
        # authorisation gate, which is precisely the bug this function exists
        # to close — so it is refused for v2 keys until someone maps it, rather
        # than quietly inheriting the old anything-goes behaviour.
        logger.error(
            "no product mapping for %r — refusing a v2 licence (%s) on an "
            "endpoint outside the product gate. Give the call site an explicit "
            "allowed_products set; an unmapped endpoint is how a one-product "
            "key ends up driving every product.",
            request.url.path,
            product_code,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"This license key is for {product_label(str(product_code))} "
                f"and is not authorised for {request.url.path}."
            ),
        )

    if product_code not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"This license key is for {product_label(str(product_code))}, "
                f"which does not include {request.url.path}. That "
                "endpoint is served by: "
                + ", ".join(product_label(code) for code in sorted(allowed))
                + ". Use the license key issued for that module."
            ),
        )


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

    Do not arm this yet either way. usage_counters is only populated once every
    write site is wired, and license_service.check_search_quota() reads
    usage_logs, which the v2 migration renamed to usage_logs_archive_v1.
    """
    if os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "0") != "1":
        return

    subscription_id = license_data.get("subscription_id")
    if subscription_id:
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
    allowed_products: Optional[Union[str, frozenset, set]],
) -> dict:
    """Authenticate the caller, bind their tenant context, return license_data.

    Dual read, not a switchover: an opaque v2 key resolves through
    licensing_service.resolve_key(); anything else falls back to the v1 JWT
    decoder exactly as before. Both paths produce the same dict contract plus
    an `auth_path` discriminator.

    *allowed_products* is required and has no default — see the module
    docstring. None means "this endpoint has no mapping" and fails closed for
    v2 keys.

    Raises 401 (no key), 403 (bad key / wrong domain / wrong product) or
    429 (over quota).
    """
    license_key = resolve_license_key(authorization, request_license)
    if not license_key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    context = _resolve_v2_context(db, license_key)
    if context is not None:
        # Before the context is copied into a hundred call sites' license_data:
        # the collection name it carries has to be the one they will derive.
        _assert_collection_name_agrees(context)
        license_data = _license_data_from_v2_context(context)
        license_data["auth_path"] = AUTH_PATH_V2
    else:
        try:
            license_data = validate_license_key(license_key, db)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        license_data["auth_path"] = AUTH_PATH_V1

    DomainAuthorizer(db).validate_request(request, license_data, api_key=x_api_key)
    # Which STORE may call, then which PRODUCT may call. Both are 403s and the
    # order between them is not arbitrary: the domain gate is the older, more
    # specific answer ("this key is not for this shop"), and a caller who fails
    # it should hear that rather than a message about product codes. Both run
    # before the quota check so a refused request never spends a counter read,
    # and before bind_context() so it never leaves a tenant identity behind.
    _assert_product_allowed(request, license_data, allowed_products)
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
    # local sees.
    #
    # Bound ONLY on the v2 path. A v1 JWT produces a perfectly truthy dict that
    # carries none of the six tenant identifiers, and record() tested the
    # context for truthiness rather than for shape — so a v1 request skipped the
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
