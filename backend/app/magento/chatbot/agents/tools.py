"""
LangChain tools for the 4 agents. Each agent builder returns a fresh list of
@tool-decorated coroutines closing over the current RequestContext.

Product agent:  find_products, get_product_details_by_skus, search_by_category,
                get_product_categories, search_content
Profile agent:  get_customer_profile, get_customer_email, get_customer_name,
                get_customer_phone, get_customer_addresses
Order agent:    get_customer_orders, get_last_order, get_order_details,
                get_order_status, get_shipping_status
Cart agent:     add_to_cart, add_product_by_name, view_cart, remove_from_cart,
                update_cart_quantity, update_cart_quantity_by_sku,
                get_checkout_url, clear_cart
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

from langchain_core.tools import tool

from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import search_content as qdrant_search_content
from backend.app.services.qdrant_service import search_products as qdrant_search_products

from backend.app.magento.chatbot.agents.request_context import RequestContext

logger = logging.getLogger(__name__)

LOGIN_REQUIRED_MESSAGE = (
    "🔒 **Login Required**\n\n"
    "Please sign in to access this information.\n\n"
    "Once logged in I can show your profile, orders, and saved addresses."
)


# ── Qdrant helpers ───────────────────────────────────────────────────────────


def _embed(ctx: RequestContext, text: str) -> list[float]:
    return embed_query(text, ctx.llm_api_key, ctx.client_id)


def _format_product_bullets(items: list[dict], currency: str = "$") -> str:
    if not items:
        return "No products matched your search."
    lines = [f"I found {len(items)} products:"]
    for i, item in enumerate(items, 1):
        name = item.get("name") or "Unknown"
        sku = item.get("sku") or ""
        price = item.get("price") or 0
        stock = item.get("stock_status") or "instock"
        try:
            price_str = f"{currency}{float(price):,.2f}"
        except Exception:
            price_str = str(price)
        stock_label = "In stock" if str(stock).lower() == "instock" else "Out of stock"
        lines.append(f"{i}. **{name}** — {price_str} · {stock_label}" + (f"  `{sku}`" if sku else ""))
    return "\n".join(lines)


def _clean_query(query: str) -> tuple[str, Optional[float], Optional[float]]:
    """Extract price range phrases and return (clean_query, min_price, max_price)."""
    q = query.lower()
    min_p: Optional[float] = None
    max_p: Optional[float] = None

    m = re.search(r"(?:above|over|more\s+th[ae]n|greater\s+th[ae]n|>\s*)\$?\s*(\d+)", q)
    if m:
        min_p = float(m.group(1))
    m = re.search(r"(?:under|below|less\s+th[ae]n|<\s*)\$?\s*(\d+)", q)
    if m:
        max_p = float(m.group(1))
    m = re.search(r"between\s+\$?\s*(\d+)\s*\$?\s+(?:and|to|-)\s+\$?\s*(\d+)\s*\$?", q)
    if m:
        min_p = float(m.group(1))
        max_p = float(m.group(2))

    cleaned = re.sub(
        r"(?:price\s+(?:above|over|under|below|between|greater|less)[^.?!]*|"
        r"(?:above|over|under|below)\s+\$?\d+[^.?!]*|"
        r"between\s+\$?\d+[^.?!]*)",
        "",
        q,
    ).strip()
    return (cleaned or "product"), min_p, max_p


# ── Product agent ────────────────────────────────────────────────────────────


def product_tools(ctx: RequestContext) -> List:
    @tool
    async def find_products(query: str, max_results: int = 10) -> str:
        """Find products by description, attributes, or price. Returns a formatted product list."""
        cleaned, min_p, max_p = _clean_query(query)
        try:
            vector = _embed(ctx, cleaned)
        except Exception as exc:
            logger.exception("embed failed: %s", exc)
            return "I couldn't search right now — please try again."

        hits = qdrant_search_products(
            client_id=ctx.client_id,
            domain=ctx.domain,
            query_vector=vector,
            limit=max_results,
            min_price=min_p,
            max_price=max_p,
        )
        return _format_product_bullets(hits)

    @tool
    async def get_product_details_by_skus(skus: str) -> str:
        """Look up live price + stock for a comma-separated list of SKUs."""
        sku_list = [s.strip() for s in skus.split(",") if s.strip()]
        if not sku_list or not ctx.magento_client:
            return "No SKUs provided."
        items = await ctx.magento_client.get_products_by_skus(sku_list)
        if not items:
            return f"No live details available for: {skus}"
        currency = await ctx.magento_client.get_currency_symbol()
        return _format_product_bullets(
            [
                {
                    "name": it.get("name"),
                    "sku": it.get("sku"),
                    "price": it.get("price"),
                    "stock_status": (it.get("extension_attributes", {}).get("stock_item", {}).get("is_in_stock") and "instock") or "outofstock",
                }
                for it in items
            ],
            currency=currency,
        )

    @tool
    async def search_content(query: str, content_type: str = "cms_page", max_results: int = 5) -> str:
        """Search CMS pages, blocks, widgets, FAQs, or policies. Use for store policy, return, shipping, about-us questions."""
        try:
            vector = _embed(ctx, query)
        except Exception:
            return "I couldn't search right now — please try again."
        hits = qdrant_search_content(
            client_id=ctx.client_id,
            domain=ctx.domain,
            query_vector=vector,
            limit=max_results,
            content_types=[content_type],
        )
        if not hits:
            return f"No {content_type} entries matched that."
        lines = [f"Found {len(hits)} {content_type} result(s):"]
        for h in hits:
            title = h.get("title") or h.get("identifier") or "(untitled)"
            summary = (h.get("summary") or h.get("content") or "")[:400]
            lines.append(f"- **{title}** — {summary}")
        return "\n".join(lines)

    @tool
    async def get_product_categories() -> str:
        """List the store's product categories with IDs."""
        if not ctx.magento_client:
            return "Categories unavailable."
        cats = await ctx.magento_client.get_categories()
        if not cats:
            return "Categories unavailable."

        def walk(node: dict, depth: int = 0) -> list[str]:
            out = []
            indent = "  " * depth
            out.append(f"{indent}📁 {node.get('name', '?')} (ID: {node.get('id')})")
            for child in node.get("children_data", []) or []:
                out.extend(walk(child, depth + 1))
            return out

        return "\n".join(walk(cats))

    @tool
    async def search_by_category(category_id: int, max_results: int = 10) -> str:
        """Find the best products in a specific category by id."""
        hits = qdrant_search_products(
            client_id=ctx.client_id,
            domain=ctx.domain,
            query_vector=_embed(ctx, f"category {category_id} products"),
            limit=max_results,
        )
        filtered = [h for h in hits if h.get(f"cat_{category_id}") is True]
        return _format_product_bullets(filtered or hits)

    return [find_products, get_product_details_by_skus, search_content, get_product_categories, search_by_category]


# ── Profile agent ────────────────────────────────────────────────────────────


def profile_tools(ctx: RequestContext) -> List:
    @tool
    async def get_customer_profile(customer_id: int = 0) -> str:
        """Return the logged-in customer's full profile."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        customer = await ctx.magento_client.get_customer_by_id(cid)
        if not customer:
            return "Profile unavailable right now."
        firstname = customer.get("firstname", "")
        lastname = customer.get("lastname", "")
        email = customer.get("email", "")
        addresses = customer.get("addresses", []) or []
        phone = ""
        for addr in addresses:
            if addr.get("default_billing") or addr.get("default_shipping"):
                phone = addr.get("telephone") or ""
                if phone:
                    break
        if not phone and addresses:
            phone = addresses[0].get("telephone") or ""
        return (
            f"## Your Profile\n"
            f"- **Name:** {firstname} {lastname}\n"
            f"- **Email:** {email}\n"
            f"- **Phone:** {phone or 'Not saved'}\n"
            f"- **Saved addresses:** {len(addresses)}"
        )

    @tool
    async def get_customer_email(customer_id: int = 0) -> str:
        """Return the customer's email address."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        customer = await ctx.magento_client.get_customer_by_id(cid)
        return f"📧 **Your Email:** {customer.get('email', 'Not available')}" if customer else LOGIN_REQUIRED_MESSAGE

    @tool
    async def get_customer_name(customer_id: int = 0) -> str:
        """Return the customer's full name."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        c = await ctx.magento_client.get_customer_by_id(cid)
        if not c:
            return LOGIN_REQUIRED_MESSAGE
        return f"👤 **Your Name:** {c.get('firstname', '')} {c.get('lastname', '')}".strip()

    @tool
    async def get_customer_phone(customer_id: int = 0) -> str:
        """Return the customer's phone number from their default address."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        c = await ctx.magento_client.get_customer_by_id(cid)
        if not c:
            return LOGIN_REQUIRED_MESSAGE
        for addr in c.get("addresses", []) or []:
            if addr.get("default_billing") or addr.get("default_shipping"):
                phone = addr.get("telephone")
                if phone:
                    return f"📞 **Your Phone:** {phone}"
        addresses = c.get("addresses", []) or []
        if addresses and addresses[0].get("telephone"):
            return f"📞 **Your Phone:** {addresses[0]['telephone']}"
        return "📞 No phone number saved. Add one from your account settings."

    @tool
    async def get_customer_addresses(customer_id: int = 0) -> str:
        """Return the customer's saved addresses."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        c = await ctx.magento_client.get_customer_by_id(cid)
        if not c:
            return LOGIN_REQUIRED_MESSAGE
        addresses = c.get("addresses", []) or []
        if not addresses:
            return "📍 No saved addresses."
        lines = ["## 📍 My Addresses"]
        for i, addr in enumerate(addresses, 1):
            name = f"{addr.get('firstname', '')} {addr.get('lastname', '')}".strip()
            street = ", ".join(addr.get("street", []) or [])
            city = addr.get("city", "")
            region = addr.get("region") or {}
            region_name = region.get("region") if isinstance(region, dict) else region
            lines.append(
                f"{i}. **{name}** — {street}, {city}, {region_name}, {addr.get('postcode', '')}, {addr.get('country_id', '')}"
            )
        return "\n".join(lines)

    return [get_customer_profile, get_customer_email, get_customer_name, get_customer_phone, get_customer_addresses]


# ── Order agent ──────────────────────────────────────────────────────────────


def _status_label(status: str) -> str:
    return {
        "pending": "🟡 Pending",
        "processing": "🔵 Processing",
        "complete": "🟢 Delivered",
        "canceled": "🔴 Canceled",
        "closed": "⚫ Closed",
        "holded": "🟠 On Hold",
        "shipped": "🚚 Shipped",
    }.get((status or "").lower(), status.capitalize() if status else "Unknown")


def _fmt_date(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return iso


def order_tools(ctx: RequestContext) -> List:
    @tool
    async def get_customer_orders(customer_id: int = 0, limit: int = 5) -> str:
        """List the customer's recent orders."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        currency = await ctx.magento_client.get_currency_symbol()
        orders = await ctx.magento_client.get_customer_orders(cid, page_size=limit)
        if not orders:
            return "📦 You don't have any orders yet."
        lines = ["## Your Recent Orders"]
        for o in orders:
            oid = o.get("increment_id", "?")
            lines.append(
                f"- **#{oid}** · {_fmt_date(o.get('created_at', ''))} · "
                f"{_status_label(o.get('status', ''))} · {currency}{float(o.get('grand_total', 0)):,.2f}"
            )
        lines.append("\nSay _order details #ORDER_NUMBER_ for the full breakdown.")
        return "\n".join(lines)

    @tool
    async def get_last_order(customer_id: int = 0) -> str:
        """Return the customer's most recent order summary."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        currency = await ctx.magento_client.get_currency_symbol()
        orders = await ctx.magento_client.get_customer_orders(cid, page_size=1)
        if not orders:
            return "📦 You don't have any orders yet."
        o = orders[0]
        return (
            f"## Last Order #{o.get('increment_id', '?')}\n"
            f"- {_fmt_date(o.get('created_at', ''))}\n"
            f"- {_status_label(o.get('status', ''))}\n"
            f"- Total: {currency}{float(o.get('grand_total', 0)):,.2f}"
        )

    @tool
    async def get_order_details(order_number: str, customer_id: int = 0) -> str:
        """Show full details (items, addresses, totals) for a specific order number."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        order_num = str(order_number or "").strip().lstrip("#")
        if not order_num.isdigit():
            return "Please provide a valid order number."
        currency = await ctx.magento_client.get_currency_symbol()
        order = await ctx.magento_client.get_order_by_increment_id(order_num, cid)
        if not order:
            return f"❌ Order #{order_num} not found."
        lines = [
            f"## 📦 Order #{order.get('increment_id')} · {_status_label(order.get('status', ''))}",
            f"Placed on {_fmt_date(order.get('created_at', ''))}",
            "",
            "**Items**",
        ]
        for item in order.get("items", []) or []:
            name = item.get("name", "?")
            qty = int(item.get("qty_ordered", 0) or 0)
            price = float(item.get("price", 0) or 0)
            row = float(item.get("row_total", price * qty) or (price * qty))
            lines.append(f"- {name} · x{qty} · {currency}{price:,.2f} = {currency}{row:,.2f}")
        lines += [
            "",
            f"**Total:** {currency}{float(order.get('grand_total', 0)):,.2f}",
        ]
        return "\n".join(lines)

    @tool
    async def get_order_status(order_number: str, customer_id: int = 0) -> str:
        """Return just the status of an order."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        order_num = str(order_number or "").strip().lstrip("#")
        if not order_num.isdigit():
            return "Please provide a valid order number."
        order = await ctx.magento_client.get_order_by_increment_id(order_num, cid)
        if not order:
            return f"❌ Order #{order_num} not found."
        return f"Order **#{order.get('increment_id')}** — {_status_label(order.get('status', ''))}"

    @tool
    async def get_shipping_status(order_number: str, customer_id: int = 0) -> str:
        """Return shipping / tracking progress for an order."""
        cid = ctx.effective_customer_id
        if not cid or not ctx.magento_client:
            return LOGIN_REQUIRED_MESSAGE
        order_num = str(order_number or "").strip().lstrip("#")
        if not order_num.isdigit():
            return "Please provide a valid order number."
        order = await ctx.magento_client.get_order_by_increment_id(order_num, cid)
        if not order:
            return f"❌ Order #{order_num} not found."
        desc = order.get("shipping_description") or "N/A"
        return (
            f"🚚 **Order #{order.get('increment_id')}** — {_status_label(order.get('status', ''))}\n"
            f"Shipping method: {desc}"
        )

    return [get_customer_orders, get_last_order, get_order_details, get_order_status, get_shipping_status]


# ── Cart agent ───────────────────────────────────────────────────────────────


def _need_cart(ctx: RequestContext) -> Optional[str]:
    if not ctx.quote_id or not ctx.magento_client:
        return "I can't access your cart right now — please open the shop so a cart is created, then try again."
    return None


def cart_tools(ctx: RequestContext) -> List:
    from backend.app.magento.chatbot.services.magento_client import MagentoClient

    @tool
    async def add_to_cart(sku: str, qty: int = 1) -> str:
        """Add a product to the cart by SKU."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        result = await ctx.magento_client.add_to_cart(ctx.quote_id, sku, qty, is_guest=is_guest)
        if result.get("success"):
            return f"✅ Added **{sku}** to your cart. Say _my cart_ to review it."
        return f"❌ Couldn't add {sku}: {result.get('error_message') or 'unknown error'}"

    @tool
    async def add_product_by_name(product_name: str, color: Optional[str] = None, size: Optional[str] = None, qty: int = 1) -> str:
        """Search for a product by name (optionally color/size) and add it to the cart."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        query = product_name
        if color:
            query += f" {color}"
        try:
            vec = _embed(ctx, query)
        except Exception:
            return "Search unavailable right now."
        hits = qdrant_search_products(
            client_id=ctx.client_id,
            domain=ctx.domain,
            query_vector=vec,
            limit=20,
        )
        if not hits:
            return f"No products matched '{product_name}'."
        chosen = None
        for h in hits:
            sku = str(h.get("sku") or "").upper()
            name = str(h.get("name") or "").lower()
            if color and color.lower() not in name and color.lower() not in sku.lower():
                continue
            if size and size.upper() not in sku.split("-"):
                continue
            chosen = h
            break
        chosen = chosen or hits[0]
        sku = chosen.get("sku") or ""
        name = chosen.get("name") or product_name
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        result = await ctx.magento_client.add_to_cart(ctx.quote_id, sku, qty, is_guest=is_guest)
        if result.get("success"):
            return f"✅ Added **{name}** to your cart."
        return f"❌ Couldn't add {name}: {result.get('error_message') or 'unknown error'}"

    @tool
    async def view_cart() -> str:
        """Show the current cart contents and totals."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        cart = await ctx.magento_client.get_cart(ctx.quote_id, is_guest=is_guest)
        if not cart or not cart.get("items"):
            return "🛒 Your cart is empty."
        totals = await ctx.magento_client.get_cart_totals(ctx.quote_id, is_guest=is_guest) or {}
        currency = await ctx.magento_client.get_currency_symbol()
        lines = ["## 🛒 Your Cart"]
        for item in cart["items"]:
            qty = float(item.get("qty", 1) or 1)
            price = float(item.get("price", 0) or 0)
            subtotal = qty * price
            lines.append(
                f"- **{item.get('name', '?')}** · x{int(qty)} · {currency}{price:,.2f} = {currency}{subtotal:,.2f}"
            )
        grand = float(totals.get("grand_total", 0) or 0)
        lines.append(f"\n**Grand total:** {currency}{grand:,.2f}")
        lines.append(f"\n👉 [Proceed to checkout]({ctx.magento_client.get_checkout_url()})")
        return "\n".join(lines)

    @tool
    async def remove_from_cart(item_id: Optional[int] = None, product_name: Optional[str] = None) -> str:
        """Remove an item from the cart. Pass item_id or product_name."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")

        if not item_id and product_name:
            cart = await ctx.magento_client.get_cart(ctx.quote_id, is_guest=is_guest)
            for it in (cart or {}).get("items", []) or []:
                if product_name.lower() in str(it.get("name") or "").lower() or product_name.lower() in str(it.get("sku") or "").lower():
                    item_id = int(it.get("item_id") or 0)
                    break

        if not item_id:
            return f"❌ Couldn't find '{product_name}' in your cart."

        ok = await ctx.magento_client.remove_from_cart(ctx.quote_id, int(item_id), is_guest=is_guest)
        return "✅ Removed from cart." if ok else "❌ Failed to remove that item."

    @tool
    async def update_cart_quantity(item_id: int, qty: int) -> str:
        """Update the quantity of a cart item by item_id. qty=0 removes it."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        if qty <= 0:
            ok = await ctx.magento_client.remove_from_cart(ctx.quote_id, int(item_id), is_guest=is_guest)
            return "✅ Removed from cart." if ok else "❌ Couldn't remove that item."
        ok = await ctx.magento_client.update_cart_item(ctx.quote_id, int(item_id), int(qty), is_guest=is_guest)
        return f"✅ Quantity updated to {qty}." if ok else "❌ Couldn't update quantity."

    @tool
    async def update_cart_quantity_by_sku(sku: str, qty: int) -> str:
        """Update the quantity of a cart item by SKU."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        cart = await ctx.magento_client.get_cart(ctx.quote_id, is_guest=is_guest)
        target = None
        for it in (cart or {}).get("items", []) or []:
            if str(it.get("sku", "")).lower() == sku.lower():
                target = int(it.get("item_id") or 0)
                break
        if not target:
            return f"❌ {sku} isn't in your cart."
        if qty <= 0:
            ok = await ctx.magento_client.remove_from_cart(ctx.quote_id, target, is_guest=is_guest)
            return "✅ Removed from cart." if ok else "❌ Couldn't remove that item."
        ok = await ctx.magento_client.update_cart_item(ctx.quote_id, target, int(qty), is_guest=is_guest)
        return f"✅ Quantity updated to {qty}." if ok else "❌ Couldn't update quantity."

    @tool
    async def get_checkout_url() -> str:
        """Return the checkout URL."""
        if not ctx.magento_client:
            return "Checkout unavailable."
        return f"🛒 [Proceed to checkout]({ctx.magento_client.get_checkout_url()})"

    @tool
    async def clear_cart() -> str:
        """Remove every item from the cart."""
        missing = _need_cart(ctx)
        if missing:
            return missing
        is_guest = MagentoClient.is_guest_cart_id(ctx.quote_id or "")
        cart = await ctx.magento_client.get_cart(ctx.quote_id, is_guest=is_guest)
        items = (cart or {}).get("items", []) or []
        if not items:
            return "🛒 Your cart is already empty."
        removed = 0
        for it in items:
            item_id = int(it.get("item_id") or 0)
            if not item_id:
                continue
            if await ctx.magento_client.remove_from_cart(ctx.quote_id, item_id, is_guest=is_guest):
                removed += 1
        return f"✅ Cart cleared — removed {removed} item(s)."

    return [
        add_to_cart, add_product_by_name, view_cart, remove_from_cart,
        update_cart_quantity, update_cart_quantity_by_sku, get_checkout_url, clear_cart,
    ]


# ── Export a build_all_tools helper (for pre-binding) ────────────────────────

def build_tools_by_agent(ctx: RequestContext) -> dict[str, List]:
    return {
        "product": product_tools(ctx),
        "profile": profile_tools(ctx),
        "order":   order_tools(ctx),
        "cart":    cart_tools(ctx),
    }


# Tools whose result should short-circuit the LLM and be returned directly.
DIRECT_RETURN_TOOLS = {
    "find_products", "get_product_details_by_skus", "search_content",
    "get_product_categories", "search_by_category",
    "get_customer_profile", "get_customer_email", "get_customer_name",
    "get_customer_phone", "get_customer_addresses",
    "get_customer_orders", "get_last_order", "get_order_details",
    "get_order_status", "get_shipping_status",
    "add_to_cart", "add_product_by_name", "view_cart", "remove_from_cart",
    "update_cart_quantity", "update_cart_quantity_by_sku",
    "get_checkout_url", "clear_cart",
}
