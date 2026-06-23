"""
Tool registry for the tool-calling LLM intent router.

Each tool maps 1:1 to a Magento agent. The LLM picks exactly one tool per
turn; the Magento side maps the tool name back to its intent + entities and
dispatches. The bodies are no-ops — these are SCHEMAS only (LangChain reads
the signature + docstring; we never execute the LLM's choice).

Docstrings are kept terse on purpose: every tool's schema is bound into the
routing prompt on every call, so the description states only what THIS tool
is for. The cross-cutting rules — "call exactly one tool", the
QUESTION-vs-BROWSE distinction, "prefer specific over generic" — live once in
the router's system prompt, not repeated per tool.

Naming: verb-first (`search_products`, `cancel_order`) so the LLM sees the
action and the PHP side maps by exact tool name.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.tools import tool


# ── Catalog & content tools ────────────────────────────────────────────────


@tool
def search_products(
    query: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    consultative: bool = False,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> str:
    """Browse/find products to look at and buy (imperative intent — the
    customer wants a list of product cards). Examples: "show me X", "find a
    red X under £50", "cheapest stainless steel fountain", "products from
    <brand>". For a QUESTION about products ("are these waterproof?") use
    answer_product_question instead.

    Args:
        query: The product search phrase as a noun phrase (e.g. "modern
            water features").
        min_price: Lower price bound, when stated.
        max_price: Upper price bound, when stated.
        sort_by: "price", "name", "rating", or "newest" when an ordering is
            requested ("cheapest", "newest"); omit otherwise.
        sort_order: "asc" or "desc"; defaults to "asc" when sort_by is set.
        consultative: True for exploratory asks ("I have a small garden, what
            would you suggest").
        brand: Brand name when mentioned (canonical label; Magento validates
            it). Omit when none.
        category: Category name when mentioned (Magento resolves it to an id).
            Omit when none.
        attributes: {attribute_code: value} when specific attributes are
            mentioned, e.g. {"color": "red"}. Omit when none.
    """


@tool
def get_product_detail(
    skus: List[str],
    query: Optional[str] = None,
) -> str:
    """Get details on ONE specific product the customer named (by product
    name or SKU) — stock, variants, price, dimensions, materials, etc.

    Args:
        skus: SKU code(s) or product name(s) as written by the customer —
            pass their wording verbatim (e.g. "WJ01-XS", "Kelkay Aqua
            Cascade"). The downstream agent resolves names / SKUs / typos
            against the catalog; don't resolve them yourself.
        query: The original question ("is it in stock?", "what dimensions?")
            so the agent has the customer's phrasing.
    """


@tool
def answer_product_question(query: str) -> str:
    """Answer a QUESTION about the store's products in general — properties,
    materials, care, safety, usage, compatibility, sizing, or warranty
    (inquisitive intent, satisfied by a text answer with no product cards).
    Examples: "are your products waterproof?", "is this safe for kids?",
    "how do I clean these?".

    Use search_products to BROWSE/buy, get_product_detail for ONE named
    product, get_category_info to describe a category.

    Args:
        query: The customer's question verbatim (don't rewrite to a noun
            phrase).
    """


@tool
def get_category_info(category: str) -> str:
    """Describe a product CATEGORY itself — what it covers, what's in it, who
    it's for (descriptive/topical intent, not browsing). Examples: "tell me
    about <category>", "what is the <category> collection", "what kinds of
    things are in <category>". To BROWSE products in it ("show me <category>",
    "cheapest <category>") use search_products.

    Args:
        category: The category name as mentioned; Magento resolves it to an id
            via CategoryVocabulary. Prefer the canonical name from the
            "Detected matches" hint when one was provided.
    """


@tool
def get_brand_info(brand: str) -> str:
    """Describe a BRAND / manufacturer itself — what it makes, what its range
    covers, what it's known for (descriptive intent, not browsing). Examples:
    "tell me about <brand>", "who makes <brand>", "is <brand> any good?". To
    BROWSE its products ("show me <brand> products") use search_products.

    Args:
        brand: The brand name as mentioned; Magento validates it against
            BrandVocabulary. Prefer the canonical name from the "Detected
            matches" hint when one was provided.
    """


@tool
def get_store_policy(query: str) -> str:
    """Answer store POLICY questions and ORDER PROBLEMS — returns, refunds,
    exchanges, shipping, warranty, cancellations, delivery timeframes, and
    what to do when something is wrong with an order. Backed by CMS / FAQ
    content; works for guests (no login needed).

    Use THIS tool — not get_order_info — for order PROBLEMS even when the
    customer mentions their order:
      - missing items / parts didn't arrive
      - order arrived damaged / broken / faulty
      - received the wrong / incorrect item
      - order hasn't arrived / is late
      - how to return an item or get a refund

    Args:
        query: The policy or order-problem question, kept close to verbatim
            so the downstream RAG search hits the right CMS / FAQ page.
    """


@tool
def get_store_info(query: str) -> str:
    """Answer questions about the store itself — opening hours, address,
    contact methods, accepted payment methods, store / company details.

    Args:
        query: The customer's question about store info.
    """


# ── Cart tools (split by action — gives the LLM a clearer routing signal) ──


@tool
def view_cart() -> str:
    """Show the current cart contents. Use for "what's in my cart", "show my
    basket", "view cart". No arguments — reads the session's active cart.
    """


@tool
def manage_cart(
    action: str,
    skus: Optional[List[str]] = None,
    quantities: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
) -> str:
    """Add, remove, update, or clear items in the cart (modify cart contents).

    Args:
        action: "add", "remove", "update", or "clear".
        skus: SKU(s) the action applies to (for add / remove / update).
        quantities: Quantity per SKU for "add" / "update" (defaults to 1).
        item_ids: Cart-line item id(s) when remove / update targets a line
            instead of a SKU; the PHP side accepts either.
    """


# ── Order tools (view vs cancel split — cancel is destructive, separate) ─


@tool
def view_orders(order_ids: Optional[List[str]] = None) -> str:
    """Show the customer's order history, or a specific order when named. Use
    for "show my orders", "where's my order #1234", "track my purchase".

    Args:
        order_ids: Specific order id(s) to show; empty = recent-orders summary.
    """


@tool
def cancel_order(order_ids: List[str], action: str = "cancel") -> str:
    """Initiate or confirm cancellation of a specific order. Two-step:
    "cancel" shows a confirmation card, "cancel_confirm" actually cancels.

    Args:
        order_ids: The order(s) to cancel (required).
        action: "cancel" (default, shows the confirmation) or "cancel_confirm"
            once the customer has explicitly confirmed.
    """


@tool
def get_order_info(order_ids: Optional[List[str]] = None, query: str = "") -> str:
    """Answer a factual QUESTION about the customer's own order's DATA, or a
    request to CHANGE it — status / delivery ("has my order shipped?"),
    contents ("what did I buy in #X?"), or changes ("change the shipping
    address"). Requires login.

    NOT for order PROBLEMS (missing / damaged / wrong item / didn't arrive /
    returns / refunds) — those go to get_store_policy even when an order is
    mentioned. Use view_orders to just SEE orders, cancel_order to cancel.

    Args:
        order_ids: Order number(s) named, if any; empty = most recent order.
        query: The customer's question / request, verbatim.
    """


@tool
def get_purchase_history() -> str:
    """Show what the customer buys most often, from their own order history —
    "your usuals" / "buy again". Examples: "what do I usually buy?", "what
    have I bought before?", "reorder my usuals". Requires login. No arguments.
    """


# ── Profile + wishlist tools (auth-gated; PHP returns an auth card for guests) ─


@tool
def view_profile(profile_field: str = "all") -> str:
    """Show the customer's saved profile info. Use for "what's my email",
    "show my address", "my account details".

    Args:
        profile_field: "email", "name", "phone", "address", or "all".
    """


@tool
def manage_wishlist(
    action: str,
    skus: Optional[List[str]] = None,
) -> str:
    """View, add to, or remove from the saved-for-later list (wishlist /
    favourites). Use for "save this for later", "add to favourites", "show my
    wishlist", "remove from saved items". Never for cart actions.

    Args:
        action: "view", "add", or "remove".
        skus: SKU(s) for "add" / "remove"; omit for "view".
    """


# ── Conversational tools (greetings + catch-all) ───────────────────────────


@tool
def greet(query: str) -> str:
    """Respond to greetings, thanks, and goodbyes only — "hi", "hello",
    "thanks", "bye". Anything asking for info or an action does NOT use this.

    Args:
        query: The customer's greeting verbatim.
    """


@tool
def general_chat(
    query: str,
    comparative: bool = False,
    compare_terms: Optional[List[str]] = None,
) -> str:
    """Fallback for queries no specific tool fits — generic / off-topic chat,
    or broad comparisons ("difference between solar and mains-powered water
    features").

    Args:
        query: The customer's question verbatim.
        comparative: True for comparisons ("difference between X and Y",
            "X vs Y", "which is better").
        compare_terms: The things being compared when comparative is True,
            e.g. ["solar", "mains-powered"].
    """


# Single source of truth — both `bind_tools` on the LLM side and the PHP
# tool->intent mapper consume this list. Order is a tie-breaker nudge when two
# tools could match: answer_product_question sits before search_products so
# the Q&A tool wins an ambiguous "question vs browse" call; greet stays last.
ALL_TOOLS = [
    greet,
    view_cart,
    manage_cart,
    view_orders,
    cancel_order,
    get_order_info,
    get_purchase_history,
    view_profile,
    manage_wishlist,
    get_store_policy,
    get_store_info,
    get_category_info,
    get_brand_info,
    answer_product_question,
    search_products,
    get_product_detail,
    general_chat,
]

# Convenience: tool name → tool object. Used by the orchestrator to look up
# the schema when the LLM returns a tool_call by name.
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}
