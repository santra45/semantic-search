"""
Single source of truth for what we sell: platforms, products, and plans.

Why this exists as its own module rather than inline dicts in the onboarding
router: the same catalogue has to be agreed on by four places that must never
drift apart —

  * the onboarding UI (renders the platform/product pickers),
  * license issuance (writes `platform` + `product_code` onto the key),
  * license validation (returns them so request handlers know what was bought),
  * usage accounting (stamps them onto every token_usage_tracking row).

If any two of those disagree about what `woo_product_qa` means, keys get issued
that nothing will honour, or usage lands under a product code the dashboard
doesn't know how to group. Keeping the list in one importable place makes that
class of bug impossible rather than merely unlikely.

PRODUCT CODES ARE PERMANENT. They are written into issued JWTs and into billing
history. Renaming one orphans every key and every usage row carrying the old
value. Add new codes; never repurpose an existing one.
"""

from __future__ import annotations

from typing import Optional

# ── Platforms ────────────────────────────────────────────────────────────────
#
# The e-commerce host a product installs into. Kept separate from the product
# code (rather than parsed out of its prefix) because the two answer different
# reporting questions — "how much does Magento cost us to serve" versus "how
# much does Product Q&A cost us to serve" — and a prefix convention would tie
# those together the first time a product ships on a second platform.

PLATFORMS: dict[str, dict] = {
    "magento": {
        "code": "magento",
        "name": "Magento 2",
        "blurb": "Installed with Composer, enabled from the CLI.",
        # Drives which install instructions the onboarding page shows once a
        # key has been issued. Magento and WooCommerce have completely
        # different install rituals and showing the wrong one is the fastest
        # way to generate a support ticket.
        "install_style": "composer",
        "accent": "#C7502A",
    },
    "woocommerce": {
        "code": "woocommerce",
        "name": "WooCommerce",
        "blurb": "Installed as a WordPress plugin ZIP.",
        "install_style": "wp_zip",
        "accent": "#7F54B3",
    },
}


# ── Products ─────────────────────────────────────────────────────────────────
#
# One entry per thing a customer can buy and install. Three of the Magento
# modules share backend endpoints (AIChatbot, AIProductQA and AISearch all call
# /magento/chatbot/agent/sync/*), which is exactly why the product identity has
# to travel on the license key: the route the request arrived on cannot tell
# them apart.

PRODUCTS: dict[str, dict] = {
    "magento_chatbot": {
        "code": "magento_chatbot",
        "platform": "magento",
        "name": "AI Chatbot",
        "tagline": "Conversational shopping assistant for the storefront.",
        "description": (
            "Answers product questions, searches the catalogue, looks up orders "
            "and edits the cart — grounded in your own catalogue and CMS content."
        ),
        "artifact": "czargroup/module-ai-chatbot",
        "module_name": "Czargroup_AIChatbot",
        # Whether this product indexes the catalogue itself. Shared-index
        # products read whatever another product on the same store already
        # synced, so onboarding can tell the customer they don't need to
        # re-sync — and so billing knows not to charge them twice for it.
        "indexes_catalogue": True,
    },
    "magento_product_qa": {
        "code": "magento_product_qa",
        "platform": "magento",
        "name": "AI Product Q&A",
        "tagline": "Per-product “Ask about this product” on the product page.",
        "description": (
            "A question box on every product page, answered from that product's "
            "own specs, attributes and description — never invented."
        ),
        "artifact": "czargroup/module-ai-product-qa",
        "module_name": "Czargroup_AIProductQA",
        "indexes_catalogue": True,
    },
    "magento_search": {
        "code": "magento_search",
        "platform": "magento",
        "name": "AI Search",
        "tagline": "Semantic search that reads intent, not keywords.",
        "description": (
            "Replaces catalogue search so “something warm for a toddler” finds "
            "the right products without matching a single word in the title."
        ),
        "artifact": "czargroup/module-ai-search",
        "module_name": "Czargroup_AISearch",
        "indexes_catalogue": True,
    },
    "woo_product_qa": {
        "code": "woo_product_qa",
        "platform": "woocommerce",
        "name": "AI Product Q&A",
        "tagline": "Per-product question box for WooCommerce.",
        "description": (
            "Shoppers ask about a product on its own page and get an answer drawn "
            "from that product's data and your site content."
        ),
        "artifact": "ai-product-qa-woo.zip",
        "module_name": "ai-product-qa-woo",
        "indexes_catalogue": True,
    },
    "woo_search": {
        "code": "woo_search",
        "platform": "woocommerce",
        "name": "Semantic Search",
        "tagline": "Natural-language product search for WooCommerce.",
        "description": (
            "Swaps WooCommerce's keyword search for semantic retrieval, so shoppers "
            "describe what they want instead of guessing your product titles."
        ),
        "artifact": "semantic-search-woo.zip",
        "module_name": "semantic-search-woo",
        "indexes_catalogue": True,
    },
}


# ── Plans ────────────────────────────────────────────────────────────────────
#
# Mirrors PLAN_LIMITS in license_service, which stays the authority for what
# gets written onto a key. The copy here is presentation only — the numbers are
# duplicated deliberately so the marketing surface can't silently disagree with
# what a key actually grants; assert_plans_match() below is what keeps them
# honest.

PLANS: dict[str, dict] = {
    "starter": {
        "code": "starter",
        "name": "Starter",
        "price": "$0",
        "period": "per month",
        "product_limit": 500,
        "search_limit_per_month": 10_000,
        "features": [
            "Up to 500 catalogue items",
            "10,000 requests a month",
            "Usage dashboard",
            "Email support",
        ],
    },
    "growth": {
        "code": "growth",
        "name": "Growth",
        "price": "$29",
        "period": "per month",
        "product_limit": 5_000,
        "search_limit_per_month": 100_000,
        "features": [
            "Up to 5,000 catalogue items",
            "100,000 requests a month",
            "Usage dashboard",
            "Priority support",
        ],
    },
    "pro": {
        "code": "pro",
        "name": "Pro",
        "price": "$99",
        "period": "per month",
        "product_limit": 25_000,
        "search_limit_per_month": 500_000,
        "features": [
            "Up to 25,000 catalogue items",
            "500,000 requests a month",
            "Usage dashboard",
            "Dedicated support",
        ],
    },
}


DEFAULT_PLAN = "starter"


# ── Lookups ──────────────────────────────────────────────────────────────────

def is_valid_platform(code: Optional[str]) -> bool:
    return code in PLATFORMS


def is_valid_product(code: Optional[str]) -> bool:
    return code in PRODUCTS


def is_valid_plan(code: Optional[str]) -> bool:
    return code in PLANS


def get_product(code: str) -> Optional[dict]:
    return PRODUCTS.get(code)


def get_platform(code: str) -> Optional[dict]:
    return PLATFORMS.get(code)


def products_for_platform(platform_code: str) -> list[dict]:
    """Every product installable on *platform_code*, in catalogue order."""
    return [p for p in PRODUCTS.values() if p["platform"] == platform_code]


def platform_of(product_code: str) -> Optional[str]:
    """The platform a product belongs to, or None if the code is unknown.

    Callers use this to derive `platform` rather than trusting a client-supplied
    value — the product code is the thing the customer actually chose, and the
    platform follows from it. Accepting both independently would let a request
    claim a (platform, product) pair that doesn't exist.
    """
    product = PRODUCTS.get(product_code)
    return product["platform"] if product else None


def validate_selection(product_code: str, platform_code: Optional[str] = None) -> str:
    """Check a product code and return the platform it implies.

    Raises ValueError with a message safe to show a customer. If *platform_code*
    is supplied it must agree with the product's real platform; a mismatch means
    the form was tampered with or the UI is out of date, and either way issuing
    the key would produce a credential nothing honours.
    """
    if not is_valid_product(product_code):
        raise ValueError(
            f"Unknown product '{product_code}'. Pick one of: "
            + ", ".join(sorted(PRODUCTS))
        )

    real_platform = PRODUCTS[product_code]["platform"]

    if platform_code and platform_code != real_platform:
        raise ValueError(
            f"{PRODUCTS[product_code]['name']} installs on "
            f"{PLATFORMS[real_platform]['name']}, not "
            f"{PLATFORMS.get(platform_code, {}).get('name', platform_code)}."
        )

    return real_platform


def public_catalog() -> dict:
    """The catalogue as the onboarding page consumes it.

    Shaped for the client rather than mirroring the internal dicts: products are
    nested under their platform so the UI can filter without a join, and the
    plan list is ordered cheapest-first so the renderer doesn't have to know the
    tier ordering.
    """
    return {
        "platforms": [
            {
                **platform,
                "products": [
                    {
                        "code": p["code"],
                        "name": p["name"],
                        "tagline": p["tagline"],
                        "description": p["description"],
                        "artifact": p["artifact"],
                        "module_name": p["module_name"],
                    }
                    for p in products_for_platform(platform["code"])
                ],
            }
            for platform in PLATFORMS.values()
        ],
        "plans": [PLANS[code] for code in ("starter", "growth", "pro")],
        "default_plan": DEFAULT_PLAN,
    }


def assert_plans_match(plan_limits: dict) -> None:
    """Fail loudly if the advertised plans drift from the issued ones.

    Called at import time by license_service. The failure mode this prevents is
    quiet and expensive: the pricing table promises 5,000 products, the key
    grants 500, and nobody notices until a customer's sync starts getting
    rejected halfway through.
    """
    for code, plan in PLANS.items():
        limits = plan_limits.get(code)
        if limits is None:
            raise RuntimeError(
                f"catalog.PLANS advertises plan '{code}' but license_service "
                f"has no limits for it — a key issued on this plan would fall "
                f"back to starter limits."
            )
        if (limits["product_limit"] != plan["product_limit"]
                or limits["search_limit_per_month"] != plan["search_limit_per_month"]):
            raise RuntimeError(
                f"Plan '{code}' disagrees between catalog and license_service: "
                f"advertised {plan['product_limit']}/{plan['search_limit_per_month']}, "
                f"issued {limits['product_limit']}/{limits['search_limit_per_month']}."
            )
