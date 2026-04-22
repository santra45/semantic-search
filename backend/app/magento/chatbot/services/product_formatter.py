"""
Rich content formatter for the multi-agent chatbot.

Design goals:
  * One function per content type (product, cms_page, cms_block, widget, store_config).
  * Emits (embedding_text, qdrant_payload) tuples.
  * Qdrant payload includes the flattened filter keys the agents rely on:
      attr_{name}_{value} = True   (e.g. attr_color_red)
      cat_{id}             = True  (e.g. cat_37)
    so the product agent can build structured Qdrant filters from vocabulary hits.
  * HTML strips via BeautifulSoup where available, plus a final regex pass so
    residual tags / entities never reach the embedder or the admin UI.

Merges:
  * magento_chatbot/services/vector_store.py  (attr_*/cat_* metadata, variant handling, HTML structuring)
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


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def _final_clean(text: str) -> str:
    """Guarantee no HTML tags or raw entities survive.

    Even when BeautifulSoup runs first we defensively:
      * re-strip any `<...>` pattern (handles malformed markup BS4 left behind)
      * decode HTML entities (`&amp;`, `&nbsp;`, etc.)
      * collapse whitespace
    """
    if not text:
        return ""
    text = html_mod.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _strip_html_simple(value: str) -> str:
    return _final_clean(value or "")


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
        parts.append(soup.get_text(" ", strip=True))
    return _final_clean("\n".join(parts))


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


# ── Gender detection from category names ────────────────────────────────────

_GENDER_PATTERNS = [
    ("women", re.compile(r"\b(women|womens|women[' ]?s|ladies|lady|female)\b", re.I)),
    ("men",   re.compile(r"\b(men|mens|men[' ]?s|male|gentlemen)\b", re.I)),
    ("girls", re.compile(r"\b(girls?|girls[' ]?)\b", re.I)),
    ("boys",  re.compile(r"\b(boys?|boys[' ]?)\b", re.I)),
    ("kids",  re.compile(r"\b(kids?|children|child|infant|baby|babies|toddler)\b", re.I)),
    ("unisex", re.compile(r"\bunisex\b", re.I)),
]


def _infer_gender(category_names: list[str], existing_gender: str = "") -> str:
    """Pick the most specific gender signal from the product's category names."""
    if existing_gender:
        return existing_gender
    blob = " ".join(category_names or [])
    if not blob.strip():
        return ""
    for label, pattern in _GENDER_PATTERNS:
        if pattern.search(blob):
            return label
    return ""


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


def _resolve_categories(product: Dict[str, Any]) -> tuple[str, list[tuple[str, str]], list[str]]:
    """Return (joined_names_for_embed, [(id, name)...], [name_strings]) from the product.

    Accepts every shape the Magento module (and the WooCommerce legacy ingest) may send:
      * [{id, name}, ...]
      * ["1", "2", "3"]        (IDs only — names remain empty but cat_{id} still indexed)
      * "Cats > Subcat"         (already-joined string)
      * product['metadata']['categories'] with any of the above
    """

    raw = product.get("categories")
    if raw in (None, "", []):
        meta = product.get("metadata") or {}
        if isinstance(meta, dict):
            raw = meta.get("categories")

    pairs: list[tuple[str, str]] = []
    names: list[str] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cid = str(item.get("id") or "").strip()
                cname = str(item.get("name") or "").strip()
                if cid:
                    pairs.append((cid, cname))
                if cname:
                    names.append(cname)
            elif item not in (None, ""):
                token = str(item).strip()
                if token.isdigit():
                    pairs.append((token, ""))
                else:
                    names.append(token)
    elif isinstance(raw, str) and raw.strip():
        for token in re.split(r"[,>]", raw):
            token = token.strip()
            if token:
                if token.isdigit():
                    pairs.append((token, ""))
                else:
                    names.append(token)

    joined = ", ".join(names)
    return joined, pairs, names


def _resolve_tags(value: Any) -> str:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or ""
                if name:
                    out.append(str(name))
            elif item:
                out.append(str(item))
        return ", ".join(out)
    if isinstance(value, str):
        return value
    return ""


def _resolve_image(value: Any) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        return first.get("src") if isinstance(first, dict) else str(first)
    if isinstance(value, str):
        return value
    return ""


# ── Variant (configurable children) handling ────────────────────────────────


def _pull_children(product: Dict[str, Any]) -> list[dict]:
    """Find the configurable-children list wherever the provider stashed it."""
    for key in ("children", "configurable_children", "variants"):
        val = product.get(key)
        if isinstance(val, list) and val:
            return val

    meta = product.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("children", "configurable_children", "variants"):
            val = meta.get(key)
            if isinstance(val, list) and val:
                return val
    return []


def _aggregate_variant_attrs(children: list[dict]) -> dict[str, list[str]]:
    """{attribute_code: [distinct_values...]} across all child products."""
    agg: dict[str, set[str]] = {}
    skip = {"sku", "name", "price", "regular_price", "stock", "stock_status", "image", "image_url"}
    for child in children:
        attributes = child.get("attributes") if isinstance(child, dict) else None
        if isinstance(attributes, dict):
            for code, value in attributes.items():
                if not code or code in skip or value in (None, "", []):
                    continue
                agg.setdefault(code, set()).add(str(value))
        elif isinstance(child, dict):
            for code, value in child.items():
                if code in skip or value in (None, "", []):
                    continue
                if not isinstance(value, (str, int, float, bool)):
                    continue
                agg.setdefault(code, set()).add(str(value))
    return {k: sorted(v) for k, v in agg.items()}


def _clean_children_for_payload(children: list[dict]) -> list[dict]:
    """Keep child records small enough to store as Qdrant payload."""
    cleaned: list[dict] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        attributes = child.get("attributes")
        if isinstance(attributes, dict):
            attrs = {
                str(k): str(v)
                for k, v in attributes.items()
                if v not in (None, "", []) and isinstance(v, (str, int, float, bool))
            }
        else:
            attrs = {}
        cleaned.append({
            "sku": str(child.get("sku") or ""),
            "name": str(child.get("name") or ""),
            "price": float(child.get("price") or 0),
            "regular_price": float(child.get("regular_price") or child.get("price") or 0),
            "stock_status": str(child.get("stock_status") or "instock"),
            "attributes": attrs,
        })
    return cleaned


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
    name = str(product.get("name") or product.get("title") or "").strip()
    if name:
        parts.append(f"Product: {name}")
    brand = str(product.get("brand") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}")

    # Categories — accept list of {id,name}, list of IDs, or comma string.
    cats_str, cat_pairs, cat_names = _resolve_categories(product)
    if cats_str:
        parts.append(f"Category: {cats_str}")

    # Gender — explicit field wins; otherwise infer from category names.
    gender = _infer_gender(cat_names, str(product.get("gender") or "").strip())
    if gender:
        parts.append(f"Gender: {gender}")

    tags_str = _resolve_tags(product.get("tags"))
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
        if not cid:
            continue
        payload[f"cat_{cid}"] = True
        if category_vocab_sink is not None and cname:
            category_vocab_sink[cid] = {"id": cid, "name": normalize_token(cname)}

    # HTML stripping — always route through html_to_structured_text, then _final_clean
    # in case the upstream store already stripped tags but left entities/stray markers.
    short = html_to_structured_text(product.get("short_description") or product.get("summary") or "")
    if short:
        parts.append(f"Summary: {short}")
    long_desc = html_to_structured_text(product.get("description") or product.get("content") or "")
    if long_desc:
        parts.append(f"Description: {long_desc[:600]}")

    # ── Product type + configurable children ─────────────────────────────────
    meta = product.get("metadata") if isinstance(product.get("metadata"), dict) else {}
    type_id = str(
        product.get("type_id")
        or product.get("product_type")
        or (meta.get("type_id") if isinstance(meta, dict) else "")
        or "simple"
    ).strip().lower()
    children = _pull_children(product)
    variant_attrs = _aggregate_variant_attrs(children) if children else {}
    has_variants = bool(children)
    is_configurable = type_id == "configurable" or has_variants

    if type_id:
        parts.append(f"Product type: {type_id}")

    # Flatten variant attributes into filter keys + embed a readable summary.
    # Also seed attr_map so the parent product inherits human-readable top-level
    # fields (`color: "Red, Blue, Green"`, `size: "S, M, L"`) — otherwise the
    # parent row ended up with no attribute fields even though its children did.
    if variant_attrs:
        summary_chunks: list[str] = []
        child_skus: list[str] = []
        for attr_code, values in variant_attrs.items():
            key = normalize_token(attr_code)
            if not key:
                continue
            # "Color: Red, Blue, Green"
            readable = ", ".join(v for v in values if v)
            summary_chunks.append(f"{attr_code.replace('_', ' ').title()}: {readable}")
            if readable and key not in attr_map:
                attr_map[key] = readable
            for raw_value in values:
                value_key = normalize_token(raw_value)
                if value_key:
                    payload[f"attr_{key}_{value_key}"] = True
                    if attribute_vocab_sink is not None and value_key != "none":
                        attribute_vocab_sink.setdefault(key, set()).add(value_key)
        for ch in children:
            if isinstance(ch, dict) and ch.get("sku"):
                child_skus.append(str(ch["sku"]))

        if summary_chunks:
            parts.append("Available variants: " + " | ".join(summary_chunks))
        if child_skus:
            parts.append("Variant SKUs: " + ", ".join(child_skus[:60]))

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
            "category_ids": [cid for cid, _ in cat_pairs],
            "tags": tags_str,
            "image_url": image_url,
            "stock_status": product.get("stock_status") or "instock",
            "average_rating": float(product.get("average_rating") or 0),
            # Product-type metadata
            "type_id": type_id,
            "is_configurable": is_configurable,
            "has_variants": has_variants,
            "variant_attributes": variant_attrs,           # { color: [Red, Blue], size: [S, M] }
            "children": _clean_children_for_payload(children),  # [{sku, name, price, ...}, ...]
            "child_skus": ",".join(
                str(c.get("sku")) for c in children if isinstance(c, dict) and c.get("sku")
            ),
            **attr_map,
        }
    )

    # Final safety pass — make sure no HTML leaks into the embedded text.
    embedded_text = _final_clean("\n".join(p for p in parts if p))
    return embedded_text, payload


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
    return _final_clean("\n".join(parts)), payload


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
    return _final_clean("\n".join(parts)), payload


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
    return _final_clean("\n".join(parts)), payload


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
    return _final_clean("\n".join(parts)), payload


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
        _final_clean(f"{content_type}: {title}\n{content[:1500]}"),
        {"title": title, "content": content[:1500], "content_type": content_type},
    )
