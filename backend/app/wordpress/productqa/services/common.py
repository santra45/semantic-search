"""Auth for the WordPress Q&A routers.

Same tenant contract as the Magento side — Bearer license key resolves to a
client_id + licensed domain, and the Origin/Referer of the incoming request
must belong to that domain. What's absent is deliberate: WordPress has no
equivalent of `magento_creds_service` / `admin_token_service`, because this
module never calls back into the store's REST API. It reads Qdrant and it
answers. There is no admin credential to leak here.

This file used to carry its own copy of the entire dual-read window — ten
functions that were byte-identical to the Magento twin's once docstrings were
stripped, with a comment at the top of each file telling the reader to change
one whenever they changed the other. That copy is gone; the implementation
lives in backend/app/services/request_auth.py and this module supplies only the
one thing that is genuinely local: which product may call these endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from backend.app.services import catalog, request_auth

# Re-exported so this package's routers keep importing their auth helpers from
# one place. They do not need to know the implementation moved.
from backend.app.services.request_auth import (  # noqa: F401
    AUTH_PATH_V1,
    AUTH_PATH_V2,
    decrypt_llm_key,
    resolve_license_key,
)

logger = logging.getLogger(__name__)


# ── Which product may call these endpoints ───────────────────────────────────
#
# THE BYPASS THIS CLOSES, AND IT WAS MEASURED THROUGH THIS EXACT FILE. A licence
# names exactly one product, and until this check existed nothing compared that
# product against the endpoint being called. A real v2 licence for
# product_code=magento_chatbot was presented HERE and answered HTTP 200 with
# auth_path=v2, and the usage_events row it produced read
# product_code=magento_chatbot, platform=magento — written from a WordPress
# request. usage_service.record() takes product_code and subscription_id
# straight off the resolved context and never off the request, so a merchant
# holding the cheapest module's key could drive every other module, book its
# cost against the wrong product, and draw billable requests out of the wrong
# subscription's usage_counters row.
#
# ONE PRODUCT, NOT A TABLE. Unlike the Magento twin — which fronts three
# products through shared /magento/chatbot/* endpoints and therefore needs a
# per-route mapping — every route in this package belongs to the same product.
# The `ai-product-qa-woo` plugin is the only thing that builds these URLs
# (grep -rno "api/wordpress/productqa/[a-zA-Z0-9_/-]*" ai-product-qa-woo/ lists
# all seven of them, and no other plugin lists any). The other WooCommerce
# product, `woo_search` / semantic-search-woo, talks to /api/search,
# /api/sync/* and /api/webhook/* — different routers, which now authorise
# through the same shared chokepoint and state `woo_search` for themselves.
#
# Stating it as one constant rather than a table of identical values is the
# point: a route added to this package tomorrow is still woo_product_qa by
# construction, so there is no table to forget to update and no route that can
# slip outside the gate.
WOO_PRODUCTQA_PRODUCTS = frozenset({"woo_product_qa"})

# Proof at IMPORT that the code above is one the catalogue actually sells, in
# the same spirit as request_auth's _V2_TO_V1_KEYS guard. A code that no longer
# exists would not raise anywhere: it would simply never match
# license_data['product_code'], and this gate would 403 the one plugin it was
# written to allow. PRODUCT CODES ARE PERMANENT (catalog.py says so), so this
# can only fire on a typo or on a rename made against that rule.
_UNKNOWN_PRODUCT_CODES = sorted(WOO_PRODUCTQA_PRODUCTS - set(catalog.PRODUCTS))
if _UNKNOWN_PRODUCT_CODES:
    raise ImportError(
        "The WordPress chokepoint authorises "
        + ", ".join(_UNKNOWN_PRODUCT_CODES) + ", which catalog.PRODUCTS does "
        "not define. Product codes are permanent by contract — do not rename "
        "one to match this module; fix this module. The catalogue sells: "
        + ", ".join(sorted(catalog.PRODUCTS)) + "."
    )


def authorize_request(
    *,
    request: Request,
    db: Session,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_license: Optional[str],
) -> dict:
    """Authenticate a WordPress Q&A caller. See request_auth.authorize_request.

    Deliberately takes no allowed-products argument. Every route in this
    package is woo_product_qa, so there is nothing for a caller to decide and
    nothing for a caller to get wrong — which is the same reason the constant
    above is a constant and not a table.

    MUST STAY A PLAIN CALL INSIDE THE HANDLER BODY, not a Depends(). See the
    request_auth module docstring for the measurement.
    """
    return request_auth.authorize_request(
        request=request,
        db=db,
        authorization=authorization,
        x_api_key=x_api_key,
        request_license=request_license,
        allowed_products=WOO_PRODUCTQA_PRODUCTS,
    )
