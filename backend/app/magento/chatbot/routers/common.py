"""Shared helpers for the agent routers — auth, credential resolution, context building."""

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

# Two different modules are called request_context. This one is the agents'
# per-request value object (Magento client, store code, quote id) and only its
# CLASS is imported, so it does not shadow the services module bound above —
# which is the ambient tenant context the usage writers read. Read the services
# module's docstring before renaming either.
from backend.app.magento.chatbot.agents.request_context import RequestContext
from backend.app.magento.chatbot.services import admin_token_service, magento_creds_service
from backend.app.magento.chatbot.services.magento_client import MagentoClient

logger = logging.getLogger(__name__)


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
#
# ONE EXCEPTION, AND IT IS NOT AN OVERSIGHT: the product authorisation table
# below. This chokepoint fronts three products through shared endpoints and so
# needs a per-route mapping; the WordPress twin fronts exactly one product
# across its whole package and so carries a single constant instead. Both files
# say so at the top of that section. Everything else here still has to move in
# lockstep.

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


# ── Which product may call which endpoint ────────────────────────────────────
#
# THE BYPASS THIS CLOSES. A licence names exactly one product, and until this
# table existed nothing compared that product against the endpoint being
# called. Measured: a real v2 licence for product_code=magento_chatbot was
# presented to the WordPress ProductQA chokepoint and answered HTTP 200 with
# auth_path=v2, and the usage_events row it produced read product_code=
# magento_chatbot, platform=magento — written from a WordPress request.
# usage_service.record() takes product_code and subscription_id straight off
# the resolved context and never off the request, so a merchant holding the
# cheapest module's key could drive every other module, book its cost against
# the wrong product, and draw billable requests out of the wrong
# subscription's usage_counters row. The per-product billing this rewrite
# exists to enable would be reporting fiction.
#
# PER ENDPOINT, NOT PER CHOKEPOINT. catalog.py says why in as many words:
# "Three of the Magento modules share backend endpoints (AIChatbot, AIProductQA
# and AISearch all call /magento/chatbot/agent/sync/*), which is exactly why
# the product identity has to travel on the license key: the route the request
# arrived on cannot tell them apart." The route cannot identify the product on
# its own — but it does narrow the set, and narrowing three products to one is
# the difference between billing fiction and billing.
#
# DERIVED FROM THE MODULES, NOT INVENTED. Every entry below is the set of
# Czargroup PHP modules that actually builds that URL. Re-derive it with:
#
#   grep -rno "api/magento/chatbot/[a-zA-Z0-9_/-]*" Czargroup/
#
# which today reports Czargroup/AIChatbot/Model/ApiClient.php (+ its
# Model/Agent/ToolCallClassifier.php), Czargroup/AIProductQA/Model/ApiClient.php
# and Czargroup/AISearch/Model/ApiClient.php. If a module starts calling an
# endpoint it did not call before, the new caller 403s until it is added here —
# which is the loud direction of failure, and the only one available: this
# table cannot be validated against the mounted routes at import time, because
# the routers import this module and the reverse import would be a cycle.

# The whole platform, taken from the catalogue rather than typed out, so that
# adding a fourth Magento product widens the shared sync surface in the same
# edit that adds it to catalog.PRODUCTS.
_MAGENTO_PRODUCTS = frozenset(
    product["code"] for product in catalog.products_for_platform("magento")
)

# Every Magento module writes into the ONE Qdrant collection its store shares,
# so the whole /agent/sync/* family is granted to the whole platform. Both
# catalog.py and Czargroup/AISearch/Helper/IndexOwnership.php describe the
# family — "ultimately POST to the same /api/magento/chatbot/agent/sync/*
# endpoints" — rather than individual routes, and IndexOwnership hands the
# whole family to whichever module wins the ownership contest. Granting only
# the two routes AISearch's ApiClient happens to build this month would 403 a
# store the moment ownership moved. Over-granting inside this family costs
# nothing that matters: a magento_search key driving a sync IS a magento_search
# sync, and record() attributes it to magento_search correctly.
_SYNC_FAMILY = _MAGENTO_PRODUCTS

# The retrieval surface AIProductQA shares with AIChatbot: it asks for matching
# products, matching content, and an answer built from them. AISearch is
# absent on purpose and is not an omission — its read path is
# POST /api/magento/search in backend/app/routers/search.py, which
# authenticates inline and never reaches this chokepoint at all.
_ANSWERING_PRODUCTS = frozenset({"magento_chatbot", "magento_product_qa"})

# The conversational surface, which only AIChatbot has. AIProductQA answers one
# question about one product on its own page: no tool-calling turn, no intent
# classifier, no token stream, and no usage panel of its own (its dashboard
# reads /api/token-usage/me/*, not this package).
_CHATBOT_ONLY = frozenset({"magento_chatbot"})

_ROUTE_PRODUCTS: dict[str, frozenset[str]] = {
    "/magento/chatbot/agent/sync/batch": _SYNC_FAMILY,
    "/magento/chatbot/agent/sync/delete": _SYNC_FAMILY,
    "/magento/chatbot/agent/sync/status": _SYNC_FAMILY,
    "/magento/chatbot/agent/sync/purge": _SYNC_FAMILY,
    "/magento/chatbot/agent/sync/purge/collection": _SYNC_FAMILY,
    "/magento/chatbot/retrieve/products": _ANSWERING_PRODUCTS,
    "/magento/chatbot/retrieve/content": _ANSWERING_PRODUCTS,
    "/magento/chatbot/retrieve/answer": _ANSWERING_PRODUCTS,
    "/magento/chatbot/retrieve/content_by_ids": _CHATBOT_ONLY,
    "/magento/chatbot/retrieve/answer/stream": _CHATBOT_ONLY,
    "/magento/chatbot/classify": _CHATBOT_ONLY,
    "/magento/chatbot/agent/tool-call": _CHATBOT_ONLY,
    "/magento/chatbot/usage/stats": _CHATBOT_ONLY,
}

# The prefix every route above shares, and the anchor the lookup cuts on.
# main.py mounts these routers with prefix="/api", so the path a request
# carries is /api/magento/chatbot/... — searching for this root rather than
# stripping a hard-coded "/api" means a re-mount under a different prefix
# cannot silently stop matching, i.e. cannot silently stop authorising.
_ROUTE_ROOT = "/magento/chatbot/"

# Proof at IMPORT that every code named above is one the catalogue actually
# sells, in the same spirit as the _V2_TO_V1_KEYS guard above. A product code
# that no longer exists would not raise anywhere: it would simply never match
# license_data['product_code'], and this table would 403 the module it was
# written to allow. PRODUCT CODES ARE PERMANENT (catalog.py says so), so this
# can only fire on a typo or on a code that was renamed against that rule —
# both of which are cheaper to find at boot than in a merchant's support
# ticket.
_UNKNOWN_PRODUCT_CODES = sorted(
    {code for codes in _ROUTE_PRODUCTS.values() for code in codes}
    - set(catalog.PRODUCTS)
)
if _UNKNOWN_PRODUCT_CODES:
    raise ImportError(
        "The Magento chokepoint authorises endpoints for "
        + ", ".join(_UNKNOWN_PRODUCT_CODES) + ", which catalog.PRODUCTS does "
        "not define. Product codes are permanent by contract — do not rename "
        "one to match this table; fix the table. The catalogue sells: "
        + ", ".join(sorted(catalog.PRODUCTS)) + "."
    )


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


def _assert_collection_name_agrees(context: dict) -> None:
    """Refuse a v2 context whose stored collection name is not the derived one.

    THE VALUE IS CARRIED AND READ BY NOBODY. _license_data_from_v2_context()
    copies `collection_name` out of the resolved context, and all ten Qdrant
    call sites still derive the name themselves from (client_id, domain) —
    qdrant_service.py at 140/256/583/1166/1207/1314/1365/1371, retrieve.py:3059
    and tenancy_service.py:218. licensing_service._context_from_row() says why
    the value is stored and "never recomputed from `domain`": get_collection_name
    maps shop.example.com, shop-example-com and shop_example_com onto the same
    string, live collections were named from an UNNORMALISED host, and
    sites.domain is normalised by tenancy_service.normalise_domain() —
    lowercased, no www, no port. So the two can differ, and when they do, every
    read goes to a collection that does not exist. Qdrant answers a missing
    collection with zero results rather than an error: the store simply goes
    quiet, with nothing in the logs, days after whatever caused the drift.

    WHY AN ASSERTION HERE RATHER THAN AT ISSUANCE. Issuance only covers licences
    minted after the check lands; this is the one place EVERY v2 request passes,
    so it covers the ones already out in the field too. It is also the only
    option that does not require editing the ten call sites, none of which can
    be made to prefer the stored value one at a time without the half-migrated
    state being worse than either end of it.

    WHY IT REFUSES RATHER THAN LOGS AND CONTINUES. On a mismatch we cannot tell
    from here which of the two names holds the points — the stored one if the
    collection predates domain normalisation, the derived one if the store has
    re-synced since, because a sync derives the name exactly as a read does.
    That is not a decision to make silently on a shopper's request. Serving it
    means picking `derived` by accident and finding out never; refusing means a
    human looks at Qdrant, sees which name has points, and re-seeds
    sites.collection_name from it (or renames the collection).

    500 rather than 403, deliberately: the caller's key is perfectly good and
    telling a paying merchant it is not would be a lie. _resolve_v2_context()
    already settled this same question the same way — "if MySQL or the context
    shape is broken the honest answer is a 500".

    Currently latent and expected to stay that way: all four local sites have
    sites.collection_name identical to get_collection_name(client_id, domain),
    and the column is NOT NULL, so an empty value is itself a broken row.
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


def _route_key(request: Request) -> Optional[str]:
    """The declared route path with the mount prefix cut off, or None.

    scope["route"] first: FastAPI writes the matched APIRoute there
    (fastapi/routing.py sets child_scope["route"] = self), so this is the
    framework's own answer to "which endpoint is running" rather than a second
    guess at it from the URL, and it is immune to a trailing slash, a
    percent-encoded segment or a redirect. request.url.path is the fallback for
    a caller that is not a mounted route — a test holding a hand-built Request,
    say — and returning None for anything with no recognisable route is what
    makes an unmapped endpoint fail closed rather than unasserted.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    cut = path.find(_ROUTE_ROOT)
    return path[cut:] if cut != -1 else None


def _product_label(code: str) -> str:
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
    expected_product: Optional[Union[str, frozenset, set]] = None,
) -> None:
    """403 unless the licence's product is one this endpoint serves.

    *expected_product* lets a caller state the product itself — a single code
    or a set of them — for an entry point that _ROUTE_PRODUCTS cannot key,
    which is anything not mounted under _ROUTE_ROOT. No router passes it today;
    every endpoint on this chokepoint is in the table.

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

    route_key = _route_key(request)

    if expected_product is None:
        allowed = _ROUTE_PRODUCTS.get(route_key) if route_key else None
    elif isinstance(expected_product, str):
        allowed = frozenset({expected_product})
    else:
        allowed = frozenset(expected_product)

    if allowed is None:
        # FAIL CLOSED. An endpoint missing from _ROUTE_PRODUCTS is an endpoint
        # outside the authorisation gate, which is precisely the bug this
        # function exists to close — so a new route is refused for v2 keys
        # until someone maps it, rather than quietly inheriting the old
        # anything-goes behaviour. Costs nothing today: with licences=0 no
        # request reaches this branch at all, and v1 returned above.
        logger.error(
            "no product mapping for %r — refusing a v2 licence (%s) on an "
            "endpoint outside the product gate. Add the route to "
            "_ROUTE_PRODUCTS in this module (and in the WordPress twin if it "
            "belongs there); an unmapped endpoint is how a one-product key "
            "ends up driving every product.",
            route_key or request.url.path,
            product_code,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"This license key is for {_product_label(str(product_code))} "
                f"and is not authorised for {request.url.path}."
            ),
        )

    if product_code not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"This license key is for {_product_label(str(product_code))}, "
                f"which does not include {request.url.path}. That "
                "endpoint is served by: "
                + ", ".join(_product_label(code) for code in sorted(allowed))
                + ". Use the license key issued for that module."
            ),
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
    expected_product: Optional[Union[str, frozenset, set]] = None,
) -> dict:
    """Authenticate the caller, bind their tenant context, return license_data.

    Dual read, not a switchover: an opaque v2 key resolves through
    licensing_service.resolve_key(); anything else falls back to the v1 JWT
    decoder exactly as before. Both paths produce the same dict contract plus
    an `auth_path` discriminator.

    *expected_product* overrides the _ROUTE_PRODUCTS lookup for a caller whose
    endpoint this module cannot key by path. Leave it unset in these routers —
    the table already covers every route mounted under _ROUTE_ROOT, and a
    router that declares its own product while the table says something else is
    two answers to one question.

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
    _assert_product_allowed(request, license_data, expected_product)
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
