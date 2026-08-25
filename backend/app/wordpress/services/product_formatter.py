"""WooCommerce content formatter — (embedding_text, qdrant_payload) pairs.

Four content types: `product`, `faq`, and the store's own site content —
`page` and `post`.

ONE FORMATTER, TWO CALLERS. Both WooCommerce plugins run their products
through this module:

  * `semantic-search-woo`  → POST /api/sync/batch                    (routers/sync.py)
  * `ai-product-qa-woo`    → POST /api/wordpress/productqa/sync/batch

They index the same catalogue into the same per-tenant collection, and a
product's point id is derived from (client_id, product_id) — so for any given
product the two write to the SAME record and whichever runs last wins. That
was a silent downgrade while the two formatters differed: the search payload
carried no attributes, variations or merchant notes, so a shopper's "what
sizes does this come in" began failing hours after a scheduled sync with
nothing logged anywhere.

Identical bytes make the race harmless. Use `build_product_point` from both
callers rather than reassembling the payload at each one — the two extra keys
it sets (`embedded_text`, `store_code`) are part of what has to match, and
they are the kind of thing that drifts when it is written out twice.

THE PAYLOAD CONTRACT IS NOT LOCAL. The shared Qdrant helpers in
`backend.app.services.qdrant_service` build their filters against specific
payload keys — `sku`, `price`, `stock_status`, `store_code`, `content_type`,
`attribute_facets`, `category_ids`. Rename or drop one of those here and
filtering silently returns the wrong subset rather than erroring. The key
names below are therefore fixed by that contract, even where a more natural
WooCommerce name exists (a WC "variation" is stored under `children`).

Slugging goes through the canonical `backend.app.utils.slug` for the same
reason: sync-time payload keys must match runtime filter keys exactly. That
one function is shared with every other platform on purpose — a WordPress-local
copy that drifted by one character would break attribute filtering with no
error anywhere.
"""

from __future__ import annotations

import html as html_mod
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from backend.app.utils.slug import slug

try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:  # pragma: no cover
    _HAS_BS4 = False


# Stamped onto every product this module writes, and read by the WooCommerce
# webhook handler so it can decline to overwrite a full point with a partial
# one. See the `managed_by` block in format_product below.
#
# The value names the FORMAT, not the plugin. Both WooCommerce plugins write
# it, because both now produce the full payload — a marker that named one of
# them would have to differ between the two, and the payloads have to match.
MANAGED_BY = "woo_full"

# Points written by the product Q&A plugin before the two formatters were
# merged carry this instead. Readers accept both; nothing writes it any more.
LEGACY_MANAGED_BY = "aipqa"

# Store code every WooCommerce point is written under. WordPress has no
# store-view concept, and `default` is the value build_point_id() collapses to
# the legacy single-store id — so this is not an arbitrary string, it is what
# keeps existing points addressable.
DEFAULT_STORE_CODE = "default"


# ── HTML + shortcode cleaning ────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")

# WordPress shortcodes — `[product_page id="99"]`, `[contact-form-7 ...]`,
# `[/vc_column]`. The WordPress analogue of Magento's `{{block ...}}`
# directives, and the same hazard: the plugin renders what it can via
# do_shortcode() before sending, but a shortcode from a deactivated plugin
# doesn't render, it just sits there as literal text. Embedding
# "[vc_row][vc_column width=1/2]" teaches the vector nothing and pollutes
# every answer that quotes the description.
#
# Deliberately narrow: requires a letter/underscore/slash start so ordinary
# bracketed prose ("[see fig. 2]", "[sic]") survives, and refuses to span
# newlines so an unclosed bracket can't eat the rest of a description.
_SHORTCODE_RE = re.compile(r"\[/?[a-zA-Z_][^\]\n]*\]")


def _final_clean(text: str) -> str:
    """Guarantee no HTML tags, raw entities, or shortcodes survive."""
    if not text:
        return ""
    text = html_mod.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _SHORTCODE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def html_to_structured_text(html: str) -> str:
    """Strip HTML while preserving the structure that carries meaning.

    Product descriptions in WooCommerce are overwhelmingly spec tables and
    bullet lists — flattening them with a plain tag-strip turns
    "<tr><td>Material</td><td>Rattan</td></tr>" into "Material Rattan", which
    reads as two unrelated words to the embedder. Rendering it as
    "Material: Rattan" keeps the pairing intact, and that pairing is what
    answers "what's it made of".
    """
    if not html:
        return ""

    html = _SHORTCODE_RE.sub(" ", html)

    if not _HAS_BS4:
        return _final_clean(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    parts: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            parts.append(text)
    for group in soup.find_all(["ul", "ol"]):
        for li in group.find_all("li"):
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


# ── Shortcode expansion (apparel sizes, child ages) ──────────────────────────
#
# "M" on its own is a near-useless embedding token — it collides with every
# other single letter in vector space. Expanding it to "M medium" gives the
# embedder a real word to anchor on, so "do you have this in medium?" matches.

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
    """Coarse affordability band, embedded as words so "something cheap"
    has something to match against."""
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


# ── Gender detection ─────────────────────────────────────────────────────────

_GENDER_PATTERNS = [
    ("women", re.compile(r"\b(women|womens|women[' ]?s|ladies|lady|female)\b", re.I)),
    ("men",   re.compile(r"\b(men|mens|men[' ]?s|male|gentlemen)\b", re.I)),
    ("girls", re.compile(r"\b(girls?|girls[' ]?)\b", re.I)),
    ("boys",  re.compile(r"\b(boys?|boys[' ]?)\b", re.I)),
    ("kids",  re.compile(r"\b(kids?|children|child|infant|baby|babies|toddler)\b", re.I)),
    ("unisex", re.compile(r"\bunisex\b", re.I)),
]


def _infer_gender(category_paths: list[str], existing_gender: str = "") -> str:
    """Read gender off the full category breadcrumbs when the merchant hasn't
    set it explicitly. Full paths, not leaf names — a product filed under
    "Clothing > Women > Hoodies" is women's even though the leaf says Hoodies.
    """
    if existing_gender:
        return existing_gender
    blob = " ".join(category_paths or [])
    if not blob.strip():
        return ""
    for label, pattern in _GENDER_PATTERNS:
        if pattern.search(blob):
            return label
    return ""


# ── Attributes ───────────────────────────────────────────────────────────────

def _iter_attributes(attributes: Any) -> list[tuple[str, str, list[str]]]:
    """Normalise WooCommerce attributes into (display_name, code, options).

    The plugin sends the list shape — `[{name, code, options: [...]}, ...]` —
    built from WC_Product::get_attributes(). The flat-dict branch is a
    tolerance for anyone posting to this endpoint by hand.

    `code` is the stable key (pa_colour), `name` the merchant's label
    (Colour). Keying the payload by code matters: two global attributes can
    share a display label, and keying by label would silently merge them.
    """
    out: list[tuple[str, str, list[str]]] = []

    if isinstance(attributes, list):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            name = (attr.get("name") or attr.get("code") or "").strip()
            code = (attr.get("code") or attr.get("taxonomy") or name).strip()
            opts = attr.get("options") or attr.get("value")
            if isinstance(opts, str):
                opts = [o.strip() for o in opts.split(",") if o.strip()]
            if isinstance(opts, (int, float)):
                opts = [str(opts)]
            if not name or not opts:
                continue
            out.append((name, code, [str(o) for o in opts if str(o).strip()]))

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
                out.append((code, code, opts))

    return out


def _resolve_categories(
    product: Dict[str, Any],
) -> tuple[str, list[tuple[str, str, str]], list[str]]:
    """Parse product_cat data into (joined_paths, triples, path_strings).

    Preferred shape from the plugin is `[{id, name, path}, ...]` where `path`
    is the full ancestor breadcrumb. The looser shapes are accepted because
    they cost three lines and save a support ticket when someone posts to this
    endpoint from a script.
    """
    raw = product.get("categories")

    triples: list[tuple[str, str, str]] = []   # (id, leaf_name, full_path)
    path_strings: list[str] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cid = str(item.get("id") or "").strip()
                leaf_name = str(item.get("name") or "").strip()
                full_path = str(item.get("path") or "").strip()
                if not full_path and leaf_name:
                    full_path = leaf_name
                if cid:
                    triples.append((cid, leaf_name, full_path))
                if full_path:
                    path_strings.append(full_path)
            elif item not in (None, ""):
                token = str(item).strip()
                if token.isdigit():
                    triples.append((token, "", ""))
                else:
                    path_strings.append(token)

    elif isinstance(raw, str) and raw.strip():
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                triples.append((token, "", ""))
            else:
                path_strings.append(token)

    return " | ".join(p for p in path_strings if p), triples, path_strings


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


# ── Custom fields ────────────────────────────────────────────────────────────
#
# Merchant-defined product fields — Advanced Custom Fields on most WooCommerce
# stores. Fabric composition, tog rating, wattage, care instructions: the
# specifics shoppers ask about, which live neither in the description nor in a
# WooCommerce attribute.
#
# They are NOT folded into `attributes`, even though that would be less code and
# would reach the prompt for free. Attributes feed two things custom fields have
# no business in: `attribute_facets`, which drives exact-match filtering, and the
# attribute vocabulary the query parser learns from. Both want short enumerable
# option values — "blue", "cotton", "XL". A paragraph of care instructions in
# that vocabulary degrades query parsing for every search the tenant runs, and
# facets keyed on a sentence match nothing by definition.
#
# Nothing here is truncated — not the value, not the label, not the count. A
# 1,200-char cap cut real merchant content off mid-sentence, and half a warranty
# clause is worse than none: the model can't tell it stopped early, so it answers
# from the fragment as confidently as it would from the whole thing.
#
# The embedding string isn't bounded here either, and never was — `description`
# and `merchant_info` go into it whole, so a wordy product reaches
# gemini-embedding-001's 2048-token ceiling with no custom fields at all
# (measured: ~1,900 tokens from description alone). Custom fields are one
# contributor among several, so a bound on them was never the thing standing
# between this and the ceiling. If one is wanted, it belongs on the assembled
# string in `format_product`, not on any single contributor to it.


def _resolve_custom_fields(value: Any) -> list[Dict[str, str]]:
    """Normalise the plugin's `custom_fields` into `[{label, value}, ...]`.

    Values go through the same HTML reader as descriptions: a wysiwyg field
    holds real markup, and a spec table in one is exactly the shape that reader
    turns into "Material: Rattan" rather than two adjacent words.
    """
    if isinstance(value, dict):
        # Tolerated for anyone posting to this endpoint by hand.
        value = [{"label": k, "value": v} for k, v in value.items()]
    if not isinstance(value, list):
        return []

    out: list[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label") or item.get("key") or "").strip()
        raw = item.get("value")
        if isinstance(raw, bool):
            raw = "Yes" if raw else "No"
        elif isinstance(raw, (list, tuple)):
            raw = ", ".join(str(v).strip() for v in raw if str(v).strip())

        text = html_to_structured_text(str(raw)) if raw not in (None, "") else ""
        if not label or not text:
            continue

        out.append({
            "label": label,
            "value": text,
        })

    return out


def _custom_field_embed_lines(fields: list[Dict[str, str]]) -> list[str]:
    """The custom-field slice of `searchable_text`, sent whole.

    Values go in under the merchant's own label and at full length; see the
    note above for why no bound lives at this layer.
    """
    lines: list[str] = []

    for field in fields:
        lines.append(f"{field['label']}: {field['value']}")

    return lines


def _resolve_image(value: Any) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        return first.get("src") if isinstance(first, dict) else str(first)
    if isinstance(value, str):
        return value
    return ""


# ── Variations ───────────────────────────────────────────────────────────────
#
# WooCommerce calls them variations; the payload key is `children` because
# that's what the shared retrieval formatter reads. Renaming it here would
# make every variable product look like a simple one at answer time.

def _pull_children(product: Dict[str, Any]) -> list[dict]:
    for key in ("children", "variations", "variants"):
        val = product.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def _aggregate_variant_attrs(children: list[dict]) -> dict[str, list[str]]:
    """{attribute_code: [distinct values]} across all variations.

    This is what answers "what sizes does this come in" — the parent product
    carries no size at all in WooCommerce, only its variations do.
    """
    agg: dict[str, set[str]] = {}
    skip = {"sku", "name", "price", "regular_price", "stock", "stock_status", "image", "image_url"}
    for child in children:
        if not isinstance(child, dict):
            continue
        attributes = child.get("attributes")
        if isinstance(attributes, dict):
            for code, value in attributes.items():
                if not code or code in skip or value in (None, "", []):
                    continue
                agg.setdefault(code, set()).add(str(value))
        else:
            for code, value in child.items():
                if code in skip or value in (None, "", []):
                    continue
                if not isinstance(value, (str, int, float, bool)):
                    continue
                agg.setdefault(code, set()).add(str(value))
    return {k: sorted(v) for k, v in agg.items()}


def _clean_children_for_payload(children: list[dict]) -> list[dict]:
    """Trim variations to what the answer prompt actually renders. A 60-variant
    product would otherwise carry a payload larger than its own description."""
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
            "price": _to_float(child.get("price")),
            "regular_price": _to_float(child.get("regular_price") or child.get("price")),
            "stock_status": str(child.get("stock_status") or "instock"),
            "attributes": attrs,
        })
    return cleaned


def _to_float(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Product ──────────────────────────────────────────────────────────────────

def format_product(
    product: Dict[str, Any],
    *,
    attribute_vocab_sink: Optional[Dict[str, set[str]]] = None,
    category_vocab_sink: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (embedding_text, qdrant_payload) for one WooCommerce product.

    The two outputs serve different readers and are shaped differently on
    purpose. `embedding_text` is prose for the embedder — expanded sizes,
    price bands, category breadcrumbs, everything that helps a fuzzy question
    find this product. `payload` is structured data for the answer prompt and
    the filters — exact values, no expansion.
    """
    parts: list[str] = []
    payload: Dict[str, Any] = {}
    facet_tokens: list[str] = []

    sku = str(product.get("sku") or "").strip()
    if sku:
        parts.append(f"SKU: {sku}")
    name = str(product.get("name") or product.get("title") or "").strip()
    if name:
        parts.append(f"Product: {name}")
    brand = str(product.get("brand") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}")

    cats_str, cat_triples, path_strings = _resolve_categories(product)
    if cats_str:
        parts.append(f"Category: {cats_str}")

    gender = _infer_gender(path_strings, str(product.get("gender") or "").strip())
    if gender:
        parts.append(f"Gender: {gender}")

    tags_str = _resolve_tags(product.get("tags"))
    if tags_str:
        parts.append(f"Tags: {tags_str}")

    # attr_map is keyed by attribute code and merged into the payload flat, so
    # `pa_colour` becomes a top-level payload field. That flatness is what
    # `_extract_attribute_lines` on the read side walks to build the
    # "Attributes:" block in the prompt.
    attr_map: Dict[str, str] = {}
    for attr_name, attr_code, options in _iter_attributes(product.get("attributes") or []):
        key = slug(attr_code)
        parts.append(f"{attr_name}: {_expand_options(options)}")
        attr_map[key] = ", ".join(options)

        for raw_value in options:
            value_key = slug(raw_value)
            if key and value_key:
                facet_tokens.append(f"{key}:{value_key}")
                if attribute_vocab_sink is not None and value_key != "none":
                    attribute_vocab_sink.setdefault(key, set()).add(value_key)

    for cid, leaf_name, full_path in cat_triples:
        if not cid or category_vocab_sink is None:
            continue
        display_name = leaf_name or (full_path.split(">")[-1].strip() if full_path else "")
        if display_name:
            category_vocab_sink[cid] = {
                "id": cid,
                "name": slug(display_name),
                "path": full_path,
            }

    short = html_to_structured_text(product.get("short_description") or product.get("excerpt") or "")
    if short:
        parts.append(f"Summary: {short}")
    long_desc = html_to_structured_text(product.get("description") or product.get("content") or "")
    if long_desc:
        parts.append(f"Description: {long_desc}")

    # Merchant-authored notes for this specific product, from the plugin's
    # hidden post meta. Authoritative — the answer prompt is told to weight it
    # above its own inference from the description.
    merchant_info = html_to_structured_text(product.get("merchant_info") or "")
    if merchant_info:
        parts.append(f"Merchant note: {merchant_info}")

    # Custom fields are embedded under the merchant's OWN label — "Tog rating"
    # rather than `tog_rating` — because the label is the wording their shoppers
    # read on the product page, and therefore the wording they ask questions in.
    custom_fields = _resolve_custom_fields(product.get("custom_fields"))
    parts.extend(_custom_field_embed_lines(custom_fields))

    # WooCommerce product types: simple | variable | grouped | external.
    # Kept verbatim rather than mapped onto Magento's vocabulary — the value
    # is rendered into the prompt, and telling a Woo merchant's assistant the
    # product is "configurable" would be a word from the wrong platform.
    type_id = str(product.get("type_id") or product.get("product_type") or "simple").strip().lower()
    children = _pull_children(product)
    variant_attrs = _aggregate_variant_attrs(children) if children else {}
    has_variants = bool(children)
    is_configurable = type_id == "variable" or has_variants

    if type_id:
        parts.append(f"Product type: {type_id}")

    if variant_attrs:
        summary_chunks: list[str] = []
        child_skus: list[str] = []
        for attr_code, values in variant_attrs.items():
            key = slug(attr_code)
            if not key:
                continue
            readable = ", ".join(v for v in values if v)
            summary_chunks.append(f"{attr_code.replace('_', ' ').title()}: {readable}")
            # Only fill a gap — never overwrite a parent-level attribute with
            # the aggregate of its variations.
            if readable and key not in attr_map:
                attr_map[key] = readable
            for raw_value in values:
                value_key = slug(raw_value)
                if value_key:
                    facet_tokens.append(f"{key}:{value_key}")
                    if attribute_vocab_sink is not None and value_key != "none":
                        attribute_vocab_sink.setdefault(key, set()).add(value_key)

        for ch in children:
            if isinstance(ch, dict) and ch.get("sku"):
                child_skus.append(str(ch["sku"]))

        if summary_chunks:
            parts.append("Available variants: " + " | ".join(summary_chunks))
        if child_skus:
            parts.append("Variant SKUs: " + ", ".join(child_skus[:60]))

    price_val = _to_float(product.get("price"))
    if price_val > 0:
        currency = product.get("currency") or ""
        symbol = html_mod.unescape(product.get("currency_symbol") or "")
        price_str = str(int(price_val)) if price_val.is_integer() else str(price_val)
        parts.append(f"Price: {symbol}{price_str} {currency}. Budget level: {_price_bucket(price_val)}")

    # Weight is a readable spec, never a facet — it's continuous, and exact-match
    # faceting on "1.5" would be useless.
    weight_val = _to_float(product.get("weight"))
    weight_unit = str(product.get("weight_unit") or "kg").strip() if weight_val > 0 else ""
    if weight_val > 0:
        weight_str = str(int(weight_val)) if weight_val.is_integer() else str(weight_val)
        parts.append(f"Weight: {weight_str} {weight_unit}")

    dimensions = str(product.get("dimensions") or "").strip()
    if dimensions:
        parts.append(f"Dimensions: {dimensions}")

    payload.update(
        {
            "sku": sku,
            "brand": brand,
            "gender": gender,
            "name": name,
            "permalink": product.get("permalink") or "",
            "price": price_val,
            "currency": product.get("currency") or "",
            "regular_price": _to_float(product.get("regular_price")) or price_val,
            "sale_price": _to_float(product.get("sale_price")),
            "on_sale": bool(product.get("on_sale", False)),
            "weight": weight_val if weight_val > 0 else None,
            "weight_unit": weight_unit or None,
            "dimensions": dimensions,
            "categories": cats_str,
            "category_paths": path_strings,
            "category_ids": [cid for cid, _, _ in cat_triples],
            "tags": tags_str,
            "image_url": _resolve_image(product.get("images") or product.get("image_url") or ""),
            "stock_status": product.get("stock_status") or "instock",
            "average_rating": _to_float(product.get("average_rating")),
            "type_id": type_id,
            "is_configurable": is_configurable,
            "has_variants": has_variants,
            "variant_attributes": variant_attrs,
            "short_description": short[:600],
            "description": long_desc[:2000],
            "children": _clean_children_for_payload(children),
            "child_skus": ",".join(
                str(c.get("sku")) for c in children if isinstance(c, dict) and c.get("sku")
            ),
            **attr_map,
        }
    )

    # Set AFTER the update() so a merchant attribute literally named
    # "attribute_facets" or "merchant_info" can't clobber either one via
    # **attr_map. Unlikely, but the failure would be invisible.
    payload["attribute_facets"] = sorted(set(facet_tokens))
    payload["merchant_info"] = merchant_info[:4000]
    payload["custom_fields"] = custom_fields

    # Format marker.
    #
    # Both plugins' PHP syncs now produce this payload, so between them the
    # write order no longer matters. What remains is the third writer: the
    # search plugin's WooCommerce webhooks, which post straight to
    # /api/webhook/* on every product save without passing through either
    # plugin's PHP. Same licence, same collection, same point id — but WooCommerce's
    # REST product object carries no variation detail, no merchant notes and no
    # attribute taxonomy codes, so that path CANNOT produce these bytes however
    # it is written.
    #
    # The webhook handler reads this field and declines to overwrite a point
    # that carries it. See routers/webhooks.py::_has_full_woo_payload.
    payload["managed_by"] = MANAGED_BY

    return _final_clean("\n".join(p for p in parts if p)), payload


def build_product_point(
    product: Dict[str, Any],
    *,
    store_code: str = DEFAULT_STORE_CODE,
    attribute_vocab_sink: Optional[Dict[str, set[str]]] = None,
    category_vocab_sink: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """(embedding_text, payload) exactly as it is stored, for both sync routers.

    Thin, and that is the point. `format_product` produces the content; the two
    keys added here are the rest of what a stored point contains, and they are
    added in one place so the two callers cannot drift on them.

    `store_code` is written onto the payload explicitly rather than left to
    `upsert_content_item` to default, because that helper only fills it in when
    the caller passes one — so a caller that omitted it would store a point
    with no store_code at all, and a payload diff against the other caller's
    point would show a missing key for no reason a reader could see.
    """
    text, payload = format_product(
        product,
        attribute_vocab_sink=attribute_vocab_sink,
        category_vocab_sink=category_vocab_sink,
    )
    payload["embedded_text"] = text
    payload["store_code"] = store_code or DEFAULT_STORE_CODE
    return text, payload


# ── FAQ ──────────────────────────────────────────────────────────────────────

def format_faq_chunkable(faq: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """(header, body, base_payload) for one merchant FAQ entry.

    The header is repeated onto every chunk's embedding by the sync router, so
    a chunk from the middle of a long returns policy still knows it's about
    returns. `content` is deliberately absent from the payload — the router
    sets it per chunk.
    """
    title = str(faq.get("title") or "").strip()
    content = html_to_structured_text(faq.get("content") or "")

    header_text = _final_clean(f"FAQ: {title}") if title else ""

    return header_text, content, {
        "title": title,
        "summary": content[:300],
        "status": "active",
    }


def format_faq(faq: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Single-point variant, for the non-chunked fallback path."""
    header, body, payload = format_faq_chunkable(faq)
    payload["content"] = body
    text = f"{header}\nContent: {body}" if header else body
    return _final_clean(text), payload


# ── Site content: pages and posts ────────────────────────────────────────────
#
# The store's own pages are where the answers to "how long does delivery take",
# "can I return this" and "is there a warranty" actually live. None of that is
# in product data, and a widget that can only read product data answers those
# questions with an honest but useless "contact the store".
#
# Chunked, for the same reason cms_page is on the Magento side: a shipping
# policy runs to several screens, and one vector over the whole thing lets the
# opening paragraph decide what the page matches. The clause that answers the
# question is three paragraphs down.
#
# The type label the shopper-facing prompt sees is deliberately different per
# type. A `page` is store policy the assistant may state as fact; a `post` is
# editorial that may be years old, and telling the model which it is reading is
# the difference between quoting a current returns window and quoting one from
# a 2019 blog announcement.

_SITE_CONTENT_LABELS = {
    "page": "Page",
    "post": "Blog post",
}


def format_site_content_chunkable(
    content_type: str, item: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    """(header, body, base_payload) for one WordPress page or post.

    The header repeats onto every chunk's embedding, so chunk 4 of a long
    delivery page still carries "Page: Shipping & Delivery" in its vector and
    stays findable even though its own paragraph never says "shipping".

    `content` is absent from base_payload on purpose — the sync router sets it
    per chunk. Copying the whole body onto every chunk would hand the prompt
    the entire page whichever paragraph matched, which is the thing chunking
    exists to avoid.
    """
    label = _SITE_CONTENT_LABELS.get(content_type, "Page")

    title = str(item.get("title") or item.get("name") or "").strip()
    slug_value = str(item.get("slug") or item.get("identifier") or "").strip()
    # From whichever SEO plugin the store runs — Yoast, Rank Math, SEOPress.
    # The plugin reads them; this module doesn't care which wrote them.
    meta_description = str(item.get("meta_description") or "").strip()
    meta_keywords = str(item.get("meta_keywords") or "").strip()
    excerpt = html_to_structured_text(item.get("excerpt") or "")
    content = html_to_structured_text(item.get("content") or "")
    categories = _resolve_tags(item.get("categories"))
    tags = _resolve_tags(item.get("tags"))

    header_parts: list[str] = []
    if title:
        header_parts.append(f"{label}: {title}")
    if slug_value and slug_value.lower() != title.lower():
        header_parts.append(f"URL: {slug_value}")
    if meta_description:
        header_parts.append(f"Description: {meta_description}")
    if meta_keywords:
        header_parts.append(f"Keywords: {meta_keywords}")
    if categories:
        header_parts.append(f"Categories: {categories}")
    if tags:
        header_parts.append(f"Tags: {tags}")
    # Only when it says something the body doesn't open with — a hand-written
    # excerpt is a curated summary worth embedding, an auto-generated one is
    # just the first 55 words of the content repeated back.
    if excerpt and not content.startswith(excerpt[:80]):
        header_parts.append(f"Summary: {excerpt}")

    header_text = _final_clean("\n".join(header_parts))

    summary = meta_description or excerpt or content[:300]

    base_payload = {
        "title": title,
        "identifier": slug_value,
        "meta_description": meta_description,
        "meta_keywords": meta_keywords,
        "excerpt": excerpt[:600],
        "summary": summary[:600],
        "permalink": str(item.get("permalink") or ""),
        "author": str(item.get("author") or ""),
        "date": str(item.get("date") or ""),
        "modified": str(item.get("modified") or ""),
        "categories": categories,
        "tags": tags,
        "status": str(item.get("status") or "publish"),
    }

    return header_text, content, base_payload


def format_site_content(content_type: str, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Single-point variant, for the non-chunked fallback path."""
    header, body, payload = format_site_content_chunkable(content_type, item)
    payload["content"] = body
    text = f"{header}\nContent: {body}" if header else body
    return _final_clean(text), payload


# ── Dispatch ─────────────────────────────────────────────────────────────────

SUPPORTED_TYPES = {"product", "faq", "page", "post"}

# The subset that is the merchant's own site content rather than catalogue
# data. Callers use this to decide what a "site content" action covers without
# hard-coding the pair in four places.
SITE_CONTENT_TYPES = ("page", "post")


def format_item(
    content_type: str,
    item: Dict[str, Any],
    *,
    attribute_vocab_sink: Optional[Dict[str, set[str]]] = None,
    category_vocab_sink: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Route to the right formatter. Only the types this module syncs are
    supported — the router rejects anything else before reaching here, so an
    unknown type arriving is a bug rather than a merchant misconfiguration."""
    if content_type == "product":
        return format_product(
            item,
            attribute_vocab_sink=attribute_vocab_sink,
            category_vocab_sink=category_vocab_sink,
        )
    if content_type == "faq":
        return format_faq(item)
    if content_type in SITE_CONTENT_TYPES:
        return format_site_content(content_type, item)
    raise ValueError(f"Unsupported content_type for WordPress sync: {content_type!r}")
