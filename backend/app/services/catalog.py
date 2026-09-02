"""
Single source of truth for what we sell: platforms, products, and plans.

Why this exists as its own module rather than inline dicts in the onboarding
router: the same catalogue has to be agreed on by four places that must never
drift apart —

  * the onboarding UI (renders the platform/product/plan pickers),
  * licence issuance (the subscription a minted key hangs off names a product),
  * licence resolution (returns them so request handlers know what was bought),
  * usage accounting (stamps them onto every usage_events row).

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
        "key_segment": "mchat",
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
        "key_segment": "mpqa",
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
        "key_segment": "msrch",
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
        "key_segment": "wpqa",
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
        "key_segment": "wsrch",
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


# ── Plans: two ladders, not one ──────────────────────────────────────────────
#
# There are two ladders because there are two scopes, and the reason is a fact
# about the vector store rather than a pricing preference.
#
# Every module installed on one store shares ONE Qdrant collection, named per
# (client, domain). A store running AIChatbot, AIProductQA and AISearch syncs
# its catalogue once and all three read the same points. So:
#
#   * catalogue size is consumed once per STORE   -> INDEX_PLANS,  sites.index_plan
#   * licences and request quota are per MODULE   -> MODULE_PLANS, subscriptions.plan
#
# These cannot collapse back into one ladder. The tempting shortcut is to drop
# INDEX_PLANS and derive a site's catalogue ceiling from its subscriptions —
# the max of their plans, say. That breaks on cancellation: a store on three
# modules cancels whichever one happened to carry the biggest plan, the derived
# ceiling drops below the number of items already sitting in the collection,
# and there is no clean resolution. Nothing can be un-indexed to get back under
# the line; refusing every sync bricks a store still paying for two modules;
# silently keeping the old ceiling means the number on the dashboard is a lie.
# A site owning its own index_plan has none of that — cancelling a module
# changes what that module may serve and nothing about what the store may hold.
#
# The independence runs the other way too, which is why neither ladder gates
# the other: a store on `free` can perfectly reasonably buy a `pro` licence for
# one module. 500 catalogue items answered 500,000 times a month is an ordinary
# shape for a small store with a lot of traffic.
#
# These dicts are now presentation AND authority. The onboarding page renders
# straight out of them, and onboarding writes the numeric limit from here onto
# the row it creates. That is the point: the old arrangement had catalog.PLANS
# advertising numbers while license_service.PLAN_LIMITS granted them, which
# needed an import-time assert_plans_match() to stop the two drifting. One dict
# cannot disagree with itself.
#
# The first line of each `features` list restates that plan's numeric limit for
# the pricing page. Change the number and change the copy with it: the import-
# time guard below checks structure, not prose, and cannot catch a stale bullet
# promising 5,000 items on a rung that grants 500.


# Bought once per SITE. Gates how much catalogue the store's shared collection
# may hold, counted in logical entities (a configurable product and its CMS
# pages are entities, not vector points).

INDEX_PLANS: dict[str, dict] = {
    "free": {
        "code": "free",
        "name": "Free",
        "price": "$0",
        "period": "per month",
        "catalogue_limit": 500,
        "features": [
            "Up to 500 catalogue items",
            "Shared by every module on the store",
            "Catalogue and site content",
        ],
    },
    "small": {
        "code": "small",
        "name": "Small",
        "price": "$19",
        "period": "per month",
        "catalogue_limit": 5_000,
        "features": [
            "Up to 5,000 catalogue items",
            "Shared by every module on the store",
            "Catalogue and site content",
        ],
    },
    "medium": {
        "code": "medium",
        "name": "Medium",
        "price": "$49",
        "period": "per month",
        "catalogue_limit": 25_000,
        "features": [
            "Up to 25,000 catalogue items",
            "Shared by every module on the store",
            "Incremental sync on catalogue changes",
        ],
    },
    "large": {
        "code": "large",
        "name": "Large",
        "price": "$149",
        "period": "per month",
        "catalogue_limit": 100_000,
        "features": [
            "Up to 100,000 catalogue items",
            "Shared by every module on the store",
            "Incremental sync on catalogue changes",
        ],
    },
}


# Bought per SUBSCRIPTION, i.e. once per (site x product). Gates billable
# requests per calendar month. `billable` is true on exactly one usage_events
# row per customer-visible action, so this counts answers a shopper saw — not
# the several LLM calls that may have gone into producing one.
#
# The three sellable rungs carry the same request numbers the single old ladder
# advertised. Deliberate: the module half of what customers were already quoted
# is unchanged, and only the catalogue half moved out into INDEX_PLANS.

MODULE_PLANS: dict[str, dict] = {
    "trial": {
        "code": "trial",
        "name": "Trial",
        "price": "$0",
        "period": "while you evaluate",
        "request_limit": 250,
        # A trial is a real subscription row with a small allowance, never the
        # absence of one. An absent subscription would mean a licence that
        # resolves to nothing, and every resolver would need a second code path
        # for "authorised but unmetered" — which is precisely the path that
        # forgets to write a usage row.
        #
        # Not sellable, so it never renders on the pricing page: it is the
        # status a subscription starts in, not a rung anyone picks. Showing it
        # beside Starter would only invite "why would I pay for Starter", and
        # Starter is the same module with forty times the allowance.
        "selectable": False,
        "features": [
            "250 requests to try it on your own catalogue",
            "Every feature of the paid plans",
            "Upgrade in place — the same key keeps working",
        ],
    },
    "starter": {
        "code": "starter",
        "name": "Starter",
        "price": "$0",
        "period": "per month",
        "request_limit": 10_000,
        "features": [
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
        "request_limit": 100_000,
        "features": [
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
        "request_limit": 500_000,
        "features": [
            "500,000 requests a month",
            "Usage dashboard",
            "Dedicated support",
        ],
    },
}


# Ladder order, cheapest rung first, and simultaneously the sellable set.
# Written out rather than relying on dict insertion order because it carries a
# second meaning: a rung absent from here is one the pricing page never renders
# and no customer can choose. `trial` is absent on purpose. A new rung added to
# the dict above and forgotten here would exist as a legal plan value that
# reaches the database only via a hand-written UPDATE, with nobody able to
# explain where it came from — assert_plan_ladders_sane() makes that an import
# failure instead.

INDEX_PLAN_ORDER = ("free", "small", "medium", "large")
MODULE_PLAN_ORDER = ("starter", "growth", "pro")


DEFAULT_INDEX_PLAN = "free"

# The rung the pricing page pre-selects, not what a subscription starts on.
# A subscription created without an explicit purchase starts at status='trial'
# on TRIAL_MODULE_PLAN; DEFAULT_MODULE_PLAN is only the radio button that comes
# up checked.
DEFAULT_MODULE_PLAN = "starter"

TRIAL_MODULE_PLAN = "trial"


# The onboarding form asks for catalogue size as a range, because a merchant
# knows roughly how many products they sell and does not know which rung that
# buys. This is the only place the two vocabularies meet.
#
# Keys are the option values in templates/onboarding.html — change one and the
# other must change with it, which assert_plan_ladders_sane() enforces for the
# values but cannot for the keys. A range that lands on the rung BELOW its own
# upper bound would cap a store short of the catalogue it was told it could
# index, so each range maps to the rung whose catalogue_limit covers its top.
CATALOGUE_SIZE_INDEX_PLAN: dict[str, str] = {
    "1-500": "free",         # 500     <= free.catalogue_limit
    "501-5000": "small",     # 5,000   <= small.catalogue_limit
    "5001-25000": "medium",  # 25,000  <= medium.catalogue_limit
    "25000+": "large",       # 100,000 <= large.catalogue_limit
}


def index_plan_for_catalogue_size(value: Optional[str]) -> str:
    """The INDEX_PLANS rung a stated catalogue size buys.

    Falls back to DEFAULT_INDEX_PLAN when the field was left blank or carries a
    value the form never offered. Falling back to the SMALLEST rung is
    deliberate and is the safe direction: a store that under-declares gets a
    ceiling it hits and a clear upgrade path, where over-granting hands out
    100,000 items of index on an unverified self-declaration.
    """
    return CATALOGUE_SIZE_INDEX_PLAN.get((value or "").strip(), DEFAULT_INDEX_PLAN)


# ── Lookups ──────────────────────────────────────────────────────────────────

def is_valid_platform(code: Optional[str]) -> bool:
    return code in PLATFORMS


def is_valid_product(code: Optional[str]) -> bool:
    return code in PRODUCTS


# No is_valid_plan(). There is no such thing as "a plan" any more — asking
# whether 'pro' is valid has two different answers depending on whether it is
# going onto sites.index_plan or subscriptions.plan, and a single validator
# would happily let a form post 'large' into a subscription.

def is_valid_index_plan(code: Optional[str]) -> bool:
    return code in INDEX_PLANS


def is_valid_module_plan(code: Optional[str]) -> bool:
    return code in MODULE_PLANS


def catalogue_limit_for(index_plan: str) -> int:
    """Items the site's shared collection may hold on *index_plan*.

    Raises on an unknown rung rather than falling back to the smallest. The old
    onboarding did `if not is_valid_plan(plan): plan = DEFAULT_PLAN`, which is
    fine for coercing a tampered radio button but catastrophic here: this value
    is written to sites.catalogue_limit and then enforced against every sync, so
    guessing means either a customer who paid for 100,000 items being cut off at
    500, or the reverse. Validate the form value with is_valid_index_plan()
    first; by the time you are asking for the number, guessing is not an option.
    """
    plan = INDEX_PLANS.get(index_plan)
    if plan is None:
        raise ValueError(
            f"Unknown index plan '{index_plan}'. Expected one of: "
            + ", ".join(INDEX_PLAN_ORDER)
        )
    return plan["catalogue_limit"]


def request_limit_for(module_plan: str) -> int:
    """Billable requests a month on *module_plan*. Written to
    subscriptions.request_limit. Raises on an unknown rung, for the same reason
    catalogue_limit_for() does.

    Accepts 'trial' — it is a real rung with a real number, just not a sellable
    one, and the code path that opens a trial subscription needs its limit from
    the same place everything else gets one.
    """
    plan = MODULE_PLANS.get(module_plan)
    if plan is None:
        raise ValueError(
            f"Unknown module plan '{module_plan}'. Expected one of: "
            + ", ".join(MODULE_PLAN_ORDER) + f", or '{TRIAL_MODULE_PLAN}'."
        )
    return plan["request_limit"]


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
    nested under their platform so the UI can filter without a join, and each
    plan list is ordered cheapest-first so the renderer doesn't have to know the
    tier ordering.

    The single `plans` / `default_plan` pair this used to return is gone, and
    the page now has two pickers to render: how big the store's catalogue is
    (once, whatever it buys), and how much traffic this module handles (per
    module). `trial` is filtered out of module_plans by INDEX/MODULE_PLAN_ORDER
    — nothing here is a rung the customer cannot buy.
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
        "index_plans": [INDEX_PLANS[code] for code in INDEX_PLAN_ORDER],
        "module_plans": [MODULE_PLANS[code] for code in MODULE_PLAN_ORDER],
        "default_index_plan": DEFAULT_INDEX_PLAN,
        "default_module_plan": DEFAULT_MODULE_PLAN,
    }


def assert_plan_ladders_sane() -> None:
    """Every rung renderable, reachable, and above the one below it.

    Replaces assert_plans_match(), which existed to catch drift between this
    module and license_service.PLAN_LIMITS. There is no second copy of the
    numbers any more, so that whole failure mode is gone; what is left are the
    ways one dict can be wrong on its own, all of which are quiet:

      * A rung whose 'code' disagrees with its dict key. The radio button posts
        the code, everything else looks the plan up by key, and the mismatch
        surfaces as a customer being put on a plan they did not pick.
      * A sellable rung missing from its *_ORDER tuple: a legal plan value that
        no page renders and no customer can choose.
      * A rung missing a display field. The onboarding renderer reads name,
        price, period and features unguarded, so this is a JavaScript exception
        in the browser and an empty pricing section — a failure that never
        reaches the server logs.
      * Limits that do not strictly ascend. One transposed digit and an upgrade
        buys less than the rung below it, and the page renders that happily.

    Called at import time, at the bottom of this module. It reads only literals
    defined a few lines up, so it costs microseconds and can only fail if this
    file is internally inconsistent — which is exactly when you want to find
    out, rather than when a customer clicks Buy.
    """
    ladders = (
        ("INDEX_PLANS", "INDEX_PLAN_ORDER", INDEX_PLANS, INDEX_PLAN_ORDER, "catalogue_limit"),
        ("MODULE_PLANS", "MODULE_PLAN_ORDER", MODULE_PLANS, MODULE_PLAN_ORDER, "request_limit"),
    )

    for plans_name, order_name, plans, order, limit_field in ladders:
        for code, plan in plans.items():
            if plan.get("code") != code:
                raise RuntimeError(
                    f"{plans_name}['{code}'] carries code '{plan.get('code')}' — "
                    f"the pricing page posts the code and everything else looks "
                    f"up the key, so these must be the same string."
                )
            for field in ("name", "price", "period", "features", limit_field):
                if not plan.get(field):
                    raise RuntimeError(
                        f"{plans_name}['{code}'] has no '{field}'. The onboarding "
                        f"page reads it unguarded and would render nothing."
                    )
            if code not in order and plan.get("selectable", True):
                raise RuntimeError(
                    f"{plans_name} defines sellable rung '{code}' but {order_name} "
                    f"does not list it — no customer could ever choose it."
                )

        for code in order:
            if code not in plans:
                raise RuntimeError(
                    f"{order_name} names '{code}', which {plans_name} does not define."
                )

        limits = [plans[code][limit_field] for code in order]
        if any(nxt <= prev for prev, nxt in zip(limits, limits[1:])):
            raise RuntimeError(
                f"{plans_name} {limit_field} values do not ascend: {limits}. "
                f"An upgrade would buy less than the rung below it."
            )

    # The defaults are what a request that named no plan lands on, so a typo in
    # one is a KeyError deep inside signup rather than here.
    if DEFAULT_INDEX_PLAN not in INDEX_PLANS:
        raise RuntimeError(f"DEFAULT_INDEX_PLAN '{DEFAULT_INDEX_PLAN}' is not an INDEX_PLANS rung.")
    if DEFAULT_MODULE_PLAN not in MODULE_PLANS:
        raise RuntimeError(f"DEFAULT_MODULE_PLAN '{DEFAULT_MODULE_PLAN}' is not a MODULE_PLANS rung.")
    if TRIAL_MODULE_PLAN not in MODULE_PLANS:
        raise RuntimeError(f"TRIAL_MODULE_PLAN '{TRIAL_MODULE_PLAN}' is not a MODULE_PLANS rung.")

    # A range mapping to a rung that does not exist would send every store that
    # picked it to a KeyError inside signup instead of to a plan.
    for size_range, code in CATALOGUE_SIZE_INDEX_PLAN.items():
        if code not in INDEX_PLANS:
            raise RuntimeError(
                f"CATALOGUE_SIZE_INDEX_PLAN['{size_range}'] names '{code}', "
                f"which INDEX_PLANS does not define."
            )


def assert_key_segments_unique() -> None:
    """Every product needs its own key segment.

    The segment exists so a customer pasting three keys into three module
    configs can tell them apart at a glance. Two products sharing one defeats
    that, and it is the kind of thing a copy-pasted catalogue entry does
    quietly. Called at import time by license_key consumers.
    """
    seen: dict[str, str] = {}
    for product in PRODUCTS.values():
        seg = product.get("key_segment")
        if not seg:
            raise RuntimeError(f"Product '{product['code']}' has no key_segment.")
        if seg in seen:
            raise RuntimeError(
                f"Products '{seen[seg]}' and '{product['code']}' both use key "
                f"segment '{seg}' — keys for them would be indistinguishable."
            )
        seen[seg] = product["code"]


# Self-checked at import. assert_plans_match() used to be invoked by
# license_service, which the schema rewrite deletes; leaving its replacement to
# be called by some future consumer would make it dead code from the day it was
# written, and a guard nobody runs is worse than no guard because it reads like
# one that does.
assert_plan_ladders_sane()

# The same argument applies here, and this one had already gone wrong: the
# segment guard was written, and then nothing called it. A duplicate segment is
# what a copy-pasted PRODUCTS entry produces, and it stays invisible until two
# of a customer's keys are indistinguishable in the one place the segment
# exists to help - the module config screen they are pasting into.
assert_key_segments_unique()
