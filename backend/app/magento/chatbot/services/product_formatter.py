"""
Rich content formatter for the multi-agent chatbot.

Design goals:
  * One function per content type (product, cms_page, cms_block, widget, store_config).
  * Emits (embedding_text, qdrant_payload) tuples.
  * Qdrant payload includes the flattened filter keys the agents rely on:
      attr_{name}_{value} = True   (e.g. attr_color_red)
      cat_{id}             = True  (e.g. cat_37)
    so the product agent can build structured Qdrant filters from vocabulary hits.
  * HTML strips via BeautifulSoup where available, else a regex fallback.

Merges:
  * magento_chatbot/services/vector_store.py  (attr_*/cat_* metadata, HTML structuring)
  * semantic-search/backend/app/services/product_service.py  (shortcode expansion, price bucket)
"""

from __future__ import annotations

import html as html_mod
import re
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:  # pragma: no cover
    _HAS_BS4 = False


# ── HTML helpers ─────────────────────────────────────────────────────────────


def _strip_html_simple(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "").strip()


def html_to_structured_text(html: str) -> str:
    """Preserve semantic structure (paragraphs, bullets, table rows) when stripping HTML."""
    if not html:
        return ""
    if not _HAS_BS4:
        return _strip_html_simple(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    parts: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            parts.append(text)
    for ul in soup.find_all(["ul", "ol"]):
        for li in ul.find_all("li"):
            text = li.get_text(" ", strip=True)
            if text:
                parts.append(f"- {text}")
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) == 2:
                parts.append(f"{cells[0]}: {cells[1]}")
            elif cells:
                parts.append(" | ".join(cells))

    if not parts:
        return soup.get_text(" ", strip=True)
    return "\n".join(parts)


# ── Token normalization ──────────────────────────────────────────────────────

def normalize_token(value: Any) -> str:
    """Lowercase + collapse non-alphanumerics into single underscores."""
    s = str(value or "").strip().lower()
    s = re.sub(r"%", " percent", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# ── Shortcode expansion (apparel sizes, child ages) ──────────────────────────

_COMMON_EXPANSIONS = {
    "XS": "XS extra small", "S": "S small", "M": "M medium", "L": "L large",
    "XL": "XL extra large", "XXL": "XXL double extra large",
    "XXXL": "XXXL triple extra large", "3XL": "3XL triple extra large",
    "4XL": "4XL quadruple extra large",
    "0-6M": "age 0 to 6 months infant", "6-12M": "age 6 to 12 months infant",
    "1-2Y": "age 1 to 2 years toddler", "2-3Y": "age 2 to 3 years toddler",
    "3-4Y": "age 3 to 4 years", "4-5Y": "age 4 to 5 years",
    "5-6Y": "age 5 to 6 years", "6-7Y": "age 6 to 7 years",
    "7-8Y": "age 7 to 8 years", "8-9Y": "age 8 to 9 years",
    "9-10Y": "age 9 to 10 years",
}


def _expand(value: str) -> str:
    return _COMMON_EXPANSIONS.get((value or "").strip(), (value or "").strip())


def _expand_options(options: Iterable[str]) -> str:
    return ", ".join(_expand(o) for o in options if str(o or "").strip())


def _price_bucket(price: float) -> str:
    if price <= 0:
        return ""
    if price < 500:
        return "very cheap budget"
    if price < 2000:
        return "budget affordable"
    if price < 10000:
        return "mid range"
    if price < 50000:
        return "premium"
    return "luxury high end"


# ── Attribute expansion ──────────────────────────────────────────────────────

def _iter_attributes(attributes: Any) -> list[tuple[str, list[str]]]:
    """Handle both WooCommerce `[{name, options}]` and flat `{code: value}` shapes."""
    out: list[tuple[str, list[str]]] = []
    if isinstance(attributes, list):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            name = (attr.get("name") or attr.get("attribute_code") or "").strip()
            opts = attr.get("options") or attr.get("value")
            if isinstance(opts, str):
                opts = [o.strip() for o in opts.split(",") if o.strip()]
            if isinstance(opts, (int, float)):
                opts = [str(opts)]
            if not name or not opts:
                continue
            out.append((name, [str(o) for o in opts if str(o).strip()]))
    elif isinstance(attributes, dict):
        for code, val in attributes.items():
            if val in (None, "", []):
                continue
            if isinstance(val, list):
                opts = [str(v) for v in val if str(v).strip()]
            elif isinstance(val, str):
                opts = [v.strip() for v in val.split(",") if v.strip()]
            else:
                opts = [str(val)]
            if code and opts:
                out.append((code, opts))
    return out


def _resolve_list_or_string(value: Any, key: str = "name") -> tuple[str, list[tuple[str, str]]]:
    """Return (joined_name_string, [(id,name)...]) from either list of dicts or comma-string."""
    pairs: list[tuple[str, str]] = []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get(key) or "").strip()
                cid = str(item.get("id") or "").strip()
                if name:
                    names.append(name)
                if cid:
                    pairs.append((cid, name))
            elif item:
                names.append(str(item))
        return ", ".join(names), pairs
    if isinstance(value, str):
        return value, []
    return "", []


def _resolve_image(value: Any) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        return first.get("src") if isinstance(first, dict) else str(first)
    if isinstance(value, str):
        return value
    return ""


# ── Product ──────────────────────────────────────────────────────────────────

def format_product(
    product: Dict[str, Any],
    *,
    attribute_vocab_sink: Optional[Dict[str, set[str]]] = None,
    category_vocab_sink: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (embedding_text, qdrant_payload) for a single product.

    If vocab sinks are provided, attribute values and category ids get collected — used
    by the sync router to persist a per-client vocabulary to MySQL after a batch.
    """

    parts: list[str] = []
    payload: Dict[str, Any] = {}

    sku = str(product.get("sku") or "").strip()
    if sku:
        parts.append(f"SKU: {sku}")
    name = str(product.get("name") or "").strip()
    if name:
        parts.append(f"Product: {name}")
    brand = str(product.get("brand") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}")
    gender = str(product.get("gender") or "").strip()
    if gender:
        parts.append(f"Gender: {gender}")

    cats_str, cat_pairs = _resolve_list_or_string(product.get("categories"))
    if cats_str:
        parts.append(f"Category: {cats_str}")
    tags_str, _ = _resolve_list_or_string(product.get("tags"))
    if tags_str:
        parts.append(f"Tags: {tags_str}")

    attr_map: Dict[str, str] = {}
    for attr_name, options in _iter_attributes(product.get("attributes") or []):
        key = normalize_token(attr_name)
        expanded = _expand_options(options)
        parts.append(f"{attr_name}: {expanded}")
        attr_map[key] = ", ".join(options)

        for raw_value in options:
            value_key = normalize_token(raw_value)
            if key and value_key:
                payload[f"attr_{key}_{value_key}"] = True
                if attribute_vocab_sink is not None and value_key != "none":
                    attribute_vocab_sink.setdefault(key, set()).add(value_key)

    for cid, cname in cat_pairs:
        if cid:
            payload[f"cat_{cid}"] = True
            if category_vocab_sink is not None and cname:
                category_vocab_sink[cid] = {"id": cid, "name": normalize_token(cname)}

    short = html_to_structured_text(product.get("short_description") or "")
    if short:
        parts.append(f"Summary: {short}")
    long_desc = html_to_structured_text(product.get("description") or "")
    if long_desc:
        parts.append(f"Description: {long_desc[:600]}")

    raw_price = product.get("price")
    try:
        price_val = float(raw_price) if raw_price not in (None, "") else 0.0
    except (TypeError, ValueError):
        price_val = 0.0
    if price_val > 0:
        currency = product.get("currency") or ""
        symbol = html_mod.unescape(product.get("currency_symbol") or "")
        price_str = str(int(price_val)) if price_val.is_integer() else str(price_val)
        bucket = _price_bucket(price_val)
        parts.append(
            f"Price: {symbol}{price_str} {currency}. Budget level: {bucket}"
        )

    image_url = _resolve_image(product.get("images") or product.get("image_url") or "")

    payload.update(
        {
            "sku": sku,
            "brand": brand,
            "gender": gender,
            "name": name,
            "permalink": product.get("permalink") or "",
            "price": price_val,
            "currency": product.get("currency") or "",
            "regular_price": float(product.get("regular_price") or price_val or 0),
            "sale_price": float(product.get("sale_price") or 0),
            "on_sale": bool(product.get("on_sale", False)),
            "categories": cats_str,
            "tags": tags_str,
            "image_url": image_url,
            "stock_status": product.get("stock_status") or "instock",
            "average_rating": float(product.get("average_rating") or 0),
            **attr_map,
        }
    )

    return "\n".join(parts), payload


# ── CMS page / block / widget / store config ─────────────────────────────────

def format_cms_page(page: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    title = str(page.get("title") or page.get("name") or "").strip()
    identifier = str(page.get("identifier") or "").strip()
    content = html_to_structured_text(page.get("content") or "")
    meta_description = str(page.get("meta_description") or "").strip()

    parts = [f"CMS Page: {title}"] if title else []
    if identifier:
        parts.append(f"URL Key: {identifier}")
    if meta_description:
        parts.append(f"Summary: {meta_description}")
    if content:
        parts.append(f"Content: {content[:1500]}")

    payload = {
        "title": title,
        "identifier": identifier,
        "content": content[:2000],
        "meta_description": meta_description,
        "permalink": page.get("permalink") or "",
        "status": page.get("status") or "active",
    }
    return "\n".join(parts), payload


def format_cms_block(block: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    title = str(block.get("title") or block.get("name") or "").strip()
    identifier = str(block.get("identifier") or "").strip()
    content = html_to_structured_text(block.get("content") or "")

    parts = [f"CMS Block: {title}"] if title else []
    if identifier:
        parts.append(f"Identifier: {identifier}")
    if content:
        parts.append(f"Content: {content[:1200]}")

    payload = {
        "title": title,
        "identifier": identifier,
        "content": content[:2000],
        "status": block.get("status") or "active",
    }
    return "\n".join(parts), payload


def format_widget(widget: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    title = str(widget.get("title") or widget.get("instance_type") or "").strip()
    instance_type = str(widget.get("instance_type") or widget.get("type") or "").strip()
    description = str(widget.get("description") or "").strip()
    content = html_to_structured_text(widget.get("content") or "")

    parts = [f"Widget: {title or instance_type}"]
    if instance_type:
        parts.append(f"Type: {instance_type}")
    if description:
        parts.append(f"Purpose: {description}")
    if content:
        parts.append(f"Body: {content[:1000]}")

    payload = {
        "title": title,
        "instance_type": instance_type,
        "description": description,
        "content": content[:1500],
    }
    return "\n".join(parts), payload


def format_store_config(info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """For store metadata: hours, address, contact info, shipping policies, etc."""
    key = str(info.get("key") or info.get("identifier") or "").strip()
    label = str(info.get("label") or info.get("title") or key).strip()
    value = info.get("value") or info.get("content") or ""
    if isinstance(value, (dict, list)):
        value = html_mod.unescape(str(value))
    value = html_to_structured_text(str(value))

    parts = [f"Store Info: {label}"]
    if key:
        parts.append(f"Key: {key}")
    if value:
        parts.append(f"Value: {value[:1200]}")

    payload = {
        "key": key,
        "label": label,
        "value": value[:1500],
    }
    return "\n".join(parts), payload


# ── Dispatcher ───────────────────────────────────────────────────────────────

def format_item(
    content_type: str,
    item: Dict[str, Any],
    *,
    attribute_vocab_sink: Optional[Dict[str, set[str]]] = None,
    category_vocab_sink: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if content_type == "product":
        return format_product(
            item,
            attribute_vocab_sink=attribute_vocab_sink,
            category_vocab_sink=category_vocab_sink,
        )
    if content_type == "cms_page":
        return format_cms_page(item)
    if content_type == "cms_block":
        return format_cms_block(item)
    if content_type == "widget":
        return format_widget(item)
    if content_type == "store_config":
        return format_store_config(item)
    # Unknown type — treat as a generic blob, don't fail the whole batch.
    title = str(item.get("title") or item.get("name") or item.get("identifier") or "")
    content = html_to_structured_text(item.get("content") or item.get("description") or "")
    return (
        f"{content_type}: {title}\n{content[:1500]}",
        {"title": title, "content": content[:1500], "content_type": content_type},
    )
