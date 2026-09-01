"""Shared helpers for the agent routers — auth, credential resolution, context building."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from backend.app.services import catalog, request_auth

# Re-exported so this package's five routers keep importing their auth helpers
# from one place. They do not need to know the implementation moved.
from backend.app.services.request_auth import (  # noqa: F401
    AUTH_PATH_V1,
    AUTH_PATH_V2,
    decrypt_llm_key,
    resolve_license_key,
)

# Two different modules are called request_context. This one is the agents'
# per-request value object (Magento client, store code, quote id) and only its
# CLASS is imported, so it does not shadow the services module bound above —
# which is the ambient tenant context the usage writers read. Read the services
# module's docstring before renaming either.
from backend.app.magento.chatbot.agents.request_context import RequestContext
from backend.app.magento.chatbot.services import admin_token_service, magento_creds_service
from backend.app.magento.chatbot.services.magento_client import MagentoClient

logger = logging.getLogger(__name__)


# ── v2 licensing: the dual-read window ───────────────────────────────────────
#
# THE DUAL READ ITSELF NO LONGER LIVES HERE. It used to, alongside a verbatim
# copy of itself in backend/app/wordpress/productqa/services/common.py and a
# comment in each file telling the reader to change one whenever they changed
# the other. Nine of those ten functions were byte-identical once docstrings
# were stripped. They are now backend/app/services/request_auth.py, which the
# generic routers can also reach — they serve both platforms and so could never
# import either product package.
#
# WHAT STAYS HERE IS THE ONE GENUINELY LOCAL ANSWER: which product may call
# which endpoint. This chokepoint fronts three products through shared
# endpoints and so needs a per-route mapping; the WordPress package fronts
# exactly one product and carries a single constant instead.


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


def authorize_request(
    *,
    request: Request,
    db: Session,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
) -> dict:
    """Authenticate a Magento chatbot caller. See request_auth.authorize_request.

    The only thing this adds is the answer to "which product may call THIS
    endpoint", looked up per route because three products share these URLs.

    An endpoint missing from _ROUTE_PRODUCTS resolves to None, which
    request_auth fails closed on for v2 keys rather than letting it inherit the
    old anything-goes behaviour. That is the whole reason the lookup passes its
    miss through instead of substituting a default: a route nobody mapped is a
    route outside the gate, and it should be loud.

    Deliberately no expected_product override. The table already covers every
    route mounted under _ROUTE_ROOT, and a router that declares its own product
    while the table says something else is two answers to one question.

    MUST STAY A PLAIN CALL INSIDE THE HANDLER BODY, not a Depends(). See the
    request_auth module docstring for the measurement.
    """
    route_key = _route_key(request)
    return request_auth.authorize_request(
        request=request,
        db=db,
        authorization=authorization,
        x_api_key=x_api_key,
        request_license=request_license,
        allowed_products=_ROUTE_PRODUCTS.get(route_key) if route_key else None,
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
