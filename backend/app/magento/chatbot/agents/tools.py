"""
Tool registry for the Phase 3.1 tool-calling LLM intent router.

Each tool here corresponds 1:1 with a Magento agent. The LLM picks
exactly one tool per turn; the Magento side maps the picked tool name
back onto its intent + entities and dispatches as before. The tool
bodies are intentionally no-ops — these are SCHEMAS only. We never
execute Python code from the LLM's choice; the Magento agents own
execution.

Why a function with @tool instead of Pydantic BaseModel classes:
    LangChain's bind_tools() accepts either, but @tool-decorated
    functions let us:
      * keep the docstring (= LLM's tool description)
      * use snake_case identifiers (which match our INTENT_* naming +
        the PHP-side tool→intent map without translation)
      * declare args with Python type hints + Field metadata in one
        place
    The function body is irrelevant; LangChain extracts the signature
    + docstring for the schema sent to the LLM.

Tool naming convention:
    Verb-first (`search_products`, `view_cart`, `cancel_order`) so the
    LLM sees the action it's invoking and the Magento side can map by
    exact tool name. Routing-by-description is the actual brittleness
    fix versus the regex-based heuristic — the docstrings describe
    when to use each tool in natural language, not in regex.
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
    """Search the product catalog. Use this when the customer wants to find or
    browse products: "show me X", "do you have Y", "I'm looking for Z",
    "what's the cheapest A", "any red shoes under £50", "products like B",
    "products from Altico", "anything in the Solar Fountains category".

    Args:
        query: The customer's product search phrase (cleaned to the noun
            phrase — e.g. for "show me modern water features under £50",
            query is "modern water features").
        min_price: Lower bound when the customer specified one (e.g.
            "over £20", "above 30 dollars").
        max_price: Upper bound when the customer specified one (e.g.
            "under £50", "less than 100").
        sort_by: One of "price", "name", "rating", "newest" when the
            customer asks for an ordering ("cheapest", "most expensive",
            "newest first"). Omit when no ordering is requested.
        sort_order: "asc" or "desc". Defaults to "asc" when sort_by is
            set and the customer didn't specify direction.
        consultative: True when the query is exploratory rather than a
            specific product search — "I have a small garden, what
            would you suggest", "I need a gift for a 10-year-old".
        brand: The brand name when the customer mentions one (e.g.
            "Altico", "Acme Garden"). Pass the canonical brand label as
            the merchant uses it; the Magento side validates against the
            known brand vocabulary before applying it as a structured
            filter. Omit when no brand is mentioned.
        category: The category name when the customer mentions one
            (e.g. "Solar Fountains", "Outdoor Lighting"). Pass the
            category name verbatim; the Magento side resolves it to a
            numeric category id via the category vocabulary. Omit when
            no category is mentioned.
        attributes: Dict of {attribute_code: value} when the customer
            mentions specific product attributes — colour, size, material,
            or any merchant-defined attribute. e.g. {"color": "red",
            "material": "stainless steel"}. Match the attribute code and
            value casing exactly as they appear in the catalogue. Omit
            when no attributes are mentioned.
    """


@tool
def get_product_detail(
    skus: List[str],
    query: Optional[str] = None,
) -> str:
    """Get detailed information about a specific product the customer has
    named or whose SKU appears in their message. Use when the customer
    references one product by name or SKU and wants more info, stock
    status, variants, or price.

    Args:
        skus: SKU code(s) extracted from the message, or recognised
            product names that resolved to known SKUs.
        query: The original natural-language question (e.g. "is it in
            stock?", "what colours come in?") so the agent's RAG step
            has the customer's actual phrasing.
    """


@tool
def get_category_info(category: str) -> str:
    """Get a descriptive overview of a product CATEGORY itself — what
    the category covers, what kinds of products belong in it, who it's
    typically for. Use when the customer asks ABOUT a category rather
    than asking to SEE products in it.

    Pick this tool when the customer's intent is descriptive / topical:
      • "tell me about <category>"
      • "what is the <category> collection"
      • "describe the <category> section"
      • "what kinds of things are in <category>"
      • "what does <category> mean / cover"
      • "are <category> any good for X?"

    Pick `search_products` instead when the customer wants to BROWSE
    or FIND products:
      • "show me <category>"
      • "products in <category>"
      • "what do you have in <category>"
      • "cheapest <category>"

    Args:
        category: The category name as the customer mentioned it.
            The Magento side resolves the name to a numeric category id
            via the per-merchant CategoryVocabulary. Pass the canonical
            name from the "Detected matches" hint in your system prompt
            if one was provided.
    """


@tool
def get_store_policy(query: str) -> str:
    """Answer questions about store policies — returns, refunds, shipping,
    warranty, exchanges, cancellations, delivery timeframes, tracking.

    Args:
        query: The customer's policy question, kept close to verbatim
            so the downstream RAG search hits the relevant CMS page.
    """


@tool
def get_store_info(query: str) -> str:
    """Answer questions about the store itself — opening hours, physical
    address, contact methods, accepted payment methods, store name,
    company details.

    Args:
        query: The customer's question about store info.
    """


# ── Cart tools (split by action — gives the LLM clearer routing signal) ──


@tool
def view_cart() -> str:
    """Show the customer the current contents of their cart. Use when the
    customer asks "what's in my cart", "show my basket", "view cart".

    No arguments — the agent reads the active cart from the customer's
    session.
    """


@tool
def manage_cart(
    action: str,
    skus: Optional[List[str]] = None,
    quantities: Optional[List[int]] = None,
    item_ids: Optional[List[int]] = None,
) -> str:
    """Add, remove, update, or clear items in the customer's cart. Use
    when the customer wants to MODIFY the cart contents.

    Args:
        action: One of "add", "remove", "update", "clear".
        skus: SKU(s) the action applies to (required for add / remove /
            update; ignored for clear).
        quantities: Quantity per SKU when "add" or "update". Defaults
            to 1 if omitted.
        item_ids: Cart-line item ids when "remove" or "update" targets
            a specific line rather than a SKU. The PHP side will accept
            either skus or item_ids and resolve at dispatch time.
    """


# ── Order tools (view vs cancel split — cancel is destructive, separate) ─


@tool
def view_orders(order_ids: Optional[List[str]] = None) -> str:
    """Show the customer's order history, or a specific order when they
    name one. Use when the customer asks "show my orders", "where's my
    order #1234", "track my purchase".

    Args:
        order_ids: Specific order id(s) to show. Empty list = recent
            orders summary.
    """


@tool
def cancel_order(order_ids: List[str], action: str = "cancel") -> str:
    """Initiate or confirm cancellation of a specific order. Two-step flow
    by design — the first call shows a confirmation card, the second
    call (action="cancel_confirm") actually cancels.

    Args:
        order_ids: The order(s) to cancel. Required.
        action: "cancel" to show the confirmation card (default),
            "cancel_confirm" when the customer has explicitly confirmed
            they want to cancel (clicked a confirm chip or wrote
            "yes cancel").
    """


# ── Profile + wishlist tools (auth-gated; PHP side returns auth card if
# the customer is a guest) ──────────────────────────────────────────────────


@tool
def view_profile(profile_field: str = "all") -> str:
    """Show the customer's saved profile information. Use when the customer
    asks "what's my email", "show my address", "my account details".

    Args:
        profile_field: One of "email", "name", "phone", "address",
            "all". "all" returns a summary card; specific fields return
            just that detail.
    """


@tool
def manage_wishlist(
    action: str,
    skus: Optional[List[str]] = None,
) -> str:
    """View, add, remove or save items for later.
        ONLY use for:
        - wishlist
        - save for later
        - save this item
        - favorites
        - favourite
        - heart this
        - bookmark
        - remember this item
        Never use for cart actions.
        Examples:
        "save this for later"
        "add this to favourites"
        "show my wishlist"
        "remove from saved items"
        Args:
        ...
        """


# ── Conversational tools (greetings + catch-all) ───────────────────────────


@tool
def greet(query: str) -> str:
    """Respond to greetings, thank-yous, and goodbyes. Use ONLY for short
    social pleasantries: "hi", "hello", "thanks", "bye", "good morning",
    "appreciate it". Anything that asks for product info, policy info,
    or any other action should NOT use this tool.

    Args:
        query: The customer's greeting verbatim.
    """


@tool
def general_chat(
    query: str,
    comparative: bool = False,
    compare_terms: Optional[List[str]] = None,
) -> str:
    """Fallback for queries that don't match any of the more specific
    tools — generic questions, off-topic chat, broad comparisons, or
    requests we can't confidently route. Also use for comparative
    queries that span topics, like "what's the difference between solar
    and mains-powered water features".

    Args:
        query: The customer's question verbatim.
        comparative: True when the customer is asking for a comparison
            ("difference between X and Y", "X vs Y", "which is better").
        compare_terms: The two-or-more things being compared, when
            comparative is True (e.g. ["solar", "mains-powered"]).
    """


# Single source of truth — both `bind_tools` on the LLM side and the PHP
# tool→intent mapper consume this list. Order matters: when two tools
# could plausibly match, ordering in this list nudges LLM preference
# (search before detail; view before manage; greet last as a no-op).
ALL_TOOLS = [
    greet,
    view_cart,
    manage_cart,
    view_orders,
    cancel_order,
    view_profile,
    manage_wishlist,
    get_store_policy,
    get_store_info,
    get_category_info,
    search_products,
    get_product_detail,
    general_chat,
]

# Convenience: tool name → tool object. Used by the orchestrator to
# look up the schema when the LLM returns a tool_call by name.
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}
