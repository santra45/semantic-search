import re
from typing import Any
import html


# ─── HTML ──────────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


# ─── Universal Value Expansion ─────────────────────────────────────────────────
# Only truly universal short codes that appear across many store types.
# Everything else passes through unchanged — Gemini understands it natively.

COMMON_EXPANSIONS = {
    # Clothing / apparel sizes
    "XS":     "XS extra small",
    "S":      "S small",
    "M":      "M medium",
    "L":      "L large",
    "XL":     "XL extra large",
    "XXL":    "XXL double extra large",
    "XXXL":   "XXXL triple extra large",
    "3XL":    "3XL triple extra large",
    "4XL":    "4XL quadruple extra large",

    # Age ranges — universal across toys, clothes, education
    "0-6M":   "age 0 to 6 months infant",
    "6-12M":  "age 6 to 12 months infant",
    "1-2Y":   "age 1 to 2 years toddler",
    "2-3Y":   "age 2 to 3 years toddler",
    "3-4Y":   "age 3 to 4 years",
    "4-5Y":   "age 4 to 5 years",
    "5-6Y":   "age 5 to 6 years",
    "6-7Y":   "age 6 to 7 years",
    "7-8Y":   "age 7 to 8 years",
    "8-9Y":   "age 8 to 9 years",
    "9-10Y":  "age 9 to 10 years",
    "10-11Y": "age 10 to 11 years",
    "11-12Y": "age 11 to 12 years",
    "12-13Y": "age 12 to 13 years",
    "13-14Y": "age 13 to 14 years",
}


def expand_value(val: str) -> str:
    """
    Expand known short codes to fuller descriptions.
    Everything not in the map passes through unchanged.
    Gemini handles domain-specific values natively.
    """
    return COMMON_EXPANSIONS.get(val.strip(), val.strip())


def expand_options(options: list[str]) -> str:
    """Expand a list of attribute options and join them."""
    return ", ".join(expand_value(o) for o in options if o.strip())


# ─── Attribute Extraction ──────────────────────────────────────────────────────

def extract_attributes(attributes: list) -> dict:
    """
    Universal attribute handler.

    Works with both formats:
    - WooCommerce webhook: [{"name": "Color", "options": ["Red", "Blue"]}]
    - Plugin sync flat:    [{"name": "Color", "options": ["Red", "Blue"]}]

    No special-casing of attribute names.
    Every attribute is treated the same way.
    """
    text_parts = []
    attr_map   = {}

    for attr in attributes:
        attr_name = attr.get("name", "").strip()
        options   = attr.get("options", [])

        # Handle both list and comma-separated string formats
        if isinstance(options, str):
            options = [o.strip() for o in options.split(",") if o.strip()]

        if not attr_name or not options:
            continue

        # Payload key: lowercase, spaces → underscores
        payload_key = attr_name.lower().replace(" ", "_")

        # Store raw values in payload for filtering/display
        attr_map[payload_key] = ", ".join(o.strip() for o in options)

        # Expand values for embedding text
        expanded = expand_options(options)

        # Fallthrough — index every attribute as-is
        # "Size: S small, M medium" or "RAM: 8GB" or "Material: Oak Wood"
        text_parts.append(f"{attr_name}: {expanded}")

    return {"text_parts": text_parts, "attr_map": attr_map}


# ─── Category + Tag Helpers ────────────────────────────────────────────────────

def resolve_list_or_string(value, key: str = "name") -> str:
    """
    Handles both formats:
    - Raw webhook list: [{"id": 1, "name": "Electronics"}, ...]
    - Plugin sync string: "Electronics, Laptops"
    """
    if isinstance(value, list):
        return ", ".join([
            item[key] if isinstance(item, dict) else str(item)
            for item in value
            if item
        ])
    return str(value) if value else ""


def resolve_image_url(images) -> str:
    """Extract image URL from both raw and flat formats."""
    if isinstance(images, list) and images:
        first = images[0]
        return first["src"] if isinstance(first, dict) else str(first)
    if isinstance(images, str):
        return images
    return ""


# ─── Build Embedding Text ─────────────────────────────────────────────────────

def build_product_text(p: dict) -> str:
    """
    Universal product text builder.

    Strategy — The Universal Adapter:
    1. Trust category + tag names as provided by store owner
    2. Loop all attributes generically with value expansion
    3. Fallthrough — unknown attributes indexed as-is
    4. Natural text structure, most important fields first
    5. No domain-specific hardcoding anywhere

    Works for: fashion, electronics, books, furniture,
               food, jewelry, toys, beauty, sports — any store type.
    """
    parts = []

    # ── 1. Product name (highest weight — appears first) ───────────────────
    name = p.get("name", "").strip()
    if name:
        parts.append(f"Product: {name}")

    # ── 2. Categories (store owner already categorized correctly) ──────────
    cats_str = resolve_list_or_string(p.get("categories", []))
    if cats_str:
        parts.append(f"Category: {cats_str}")

    # ── 3. Tags (store owner's own keywords — trust them) ──────────────────
    tags_str = resolve_list_or_string(p.get("tags", []))
    if tags_str:
        parts.append(f"Tags: {tags_str}")

    # ── 4. All attributes — universal handler, no special cases ───────────
    attributes = p.get("attributes", [])
    if attributes:
        attr_data = extract_attributes(attributes)
        parts.extend(attr_data["text_parts"])

    # ── 5. Short description — store owner's own summary ──────────────────
    short = strip_html(p.get("short_description", "")).strip()
    if short:
        parts.append(f"Summary: {short}")

    # ── 6. Full description — capped at 600 chars ─────────────────────────
    desc = strip_html(p.get("description", "")).strip()[:600]
    if desc:
        parts.append(f"Description: {desc}")

    # ── 7. Price (useful for price-based queries) ──────────────────────────
    price = float(p.get("price", 0))

    if price > 0:
        if price.is_integer():
            price_str = str(int(price))
        else:
            price_str = str(price)

        currency_symbol = html.unescape(p.get("currency_symbol", ""))
        parts.append(f"Price: {currency_symbol}{price_str}")

    return "\n".join(parts)


# ─── Extract Qdrant Payload ────────────────────────────────────────────────────

def extract_payload(p: dict) -> dict:
    """
    Builds the metadata payload stored alongside the vector in Qdrant.
    Universal — works for any store type.
    All dynamic attributes stored as flat key-value pairs.
    """
    cats_str  = resolve_list_or_string(p.get("categories", []))
    tags_str  = resolve_list_or_string(p.get("tags", []))
    image_url = resolve_image_url(p.get("images", p.get("image_url", "")))
    price     = str(p.get("price", "0") or "0")

    # All attributes stored dynamically
    attributes = p.get("attributes", [])
    attr_map   = extract_attributes(attributes)["attr_map"] if attributes else {}

    return {
        "name":           p.get("name", ""),
        "permalink":      p.get("permalink", ""),
        "price":          float(price),
        "currency":       p.get("currency", ""),
        "regular_price":  float(p.get("regular_price") or price or "0"),
        "sale_price":     float(p.get("sale_price")    or "0"),
        "on_sale":        bool(p.get("on_sale", False)),
        "categories":     cats_str,
        "tags":           tags_str,
        "image_url":      image_url,
        "stock_status":   p.get("stock_status", "instock"),
        "average_rating": float(p.get("average_rating") or "0"),
        **attr_map   # size, color, ram, storage, material — whatever the store has
    }