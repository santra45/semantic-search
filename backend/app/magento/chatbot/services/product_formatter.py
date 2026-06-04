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
# Magento template directives — `{{block class="..."}}`, `{{widget type="..."}}`,
# `{{config path="..."}}`, etc. The Magento providers now render these
# server-side via FilterProvider before the content is shipped, but we keep
# this stripper as a safety net for two cases:
#   1. Older deployments still running the unpatched provider.
#   2. Directives that fail to render (deleted block class, missing template)
#      and fall through to the raw text.
# Either way, raw `{{...}}` syntax should never reach the embedder.
_MAGENTO_DIRECTIVE_RE = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)


def _final_clean(text: str) -> str:
    """Guarantee no HTML tags, raw entities, or Magento directives survive."""
    if not text:
        return ""
    text = html_mod.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _MAGENTO_DIRECTIVE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _strip_html_simple(value: str) -> str:
    return _final_clean(value or "")


def html_to_structured_text(html: str) -> str:
    """Preserve semantic structure (paragraphs, bullets, table rows) when stripping HTML."""
    if not html:
        return ""

    # Strip Magento directive syntax BEFORE parsing as HTML — BS4 treats
    # `{{...}}` as plain text and would let it leak through to the embedder.
    # This complements the Magento-side FilterProvider rendering: we render
    # what we can server-side, then drop any literal directive that survived.
    html = _MAGENTO_DIRECTIVE_RE.sub(" ", html)

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

from backend.app.utils.slug import slug as _shared_slug


def normalize_token(value: Any) -> str:
    """Lowercase + collapse non-alphanumerics into single underscores.

    Thin wrapper around the canonical slug() in backend.app.utils.slug —
    name kept for back-compat with existing call sites. New code should
    import slug() directly. Same algorithm: sync-time payload keys MUST
    match runtime filter keys or attribute / category filter lookups
    silently miss.
    """
    return _shared_slug(value)


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
    """Pick the most specific gender signal from the product's full category paths.

    Receives full path strings like "Default Category > Women > Tops > Hoodies"
    so every ancestor node contributes to the match — not just the leaf name.
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


# ── Attribute expansion ──────────────────────────────────────────────────────

def _iter_attributes(attributes: Any) -> list[tuple[str, str, list[str]]]:
    """Handle both WooCommerce `[{name, options}]` and flat `{code: value}` shapes.

    Returns (display_name, attribute_code, options) triples so the caller can
    use `code` as the stable dict key and `name` for the human-readable label.
    """
    out: list[tuple[str, str, list[str]]] = []
    if isinstance(attributes, list):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            name = (attr.get("name") or attr.get("attribute_code") or "").strip()
            # Prefer `code` as the stable key; fall back to normalised name.
            code = (attr.get("code") or attr.get("attribute_code") or name).strip()
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
                # For dict-shape, name and code are the same key.
                out.append((code, code, opts))
    return out


def _resolve_categories(
    product: Dict[str, Any],
) -> tuple[str, list[tuple[str, str, str]], list[str]]:
    """Parse category data from the product dict.

    Returns:
        joined_paths_for_embed  – full paths joined for the embedding line,
                                  e.g. "Default Category > Women > Tops > Hoodies"
        triples                 – [(id, leaf_name, full_path), ...]
        path_strings            – [full_path, ...] used for gender inference

    Accepts every shape the Magento module may send:
      * [{id, name, path}, ...]   ← NEW preferred shape from updated PHP
      * [{id, name}, ...]         ← legacy (no path field)
      * ["1", "2", ...]           ← IDs only
      * "Cats > Subcat"           ← pre-joined string
      * product['metadata']['categories'] with any of the above
    """
    raw = product.get("categories")
    if raw in (None, "", []):
        meta = product.get("metadata") or {}
        if isinstance(meta, dict):
            raw = meta.get("categories")

    triples: list[tuple[str, str, str]] = []   # (id, leaf_name, full_path)
    path_strings: list[str] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cid       = str(item.get("id") or "").strip()
                leaf_name = str(item.get("name") or "").strip()
                # `path` is the full breadcrumb, e.g. "Default Category > Women > Tops > Hoodies"
                full_path = str(item.get("path") or "").strip()

                # Fall back: if no path sent, use leaf name as the path
                # (keeps backward-compat with pre-update PHP versions)
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
        # Pre-joined string — treat each segment as its own path
        for token in re.split(r"[,]", raw):
            token = token.strip()
            if token:
                if token.isdigit():
                    triples.append((token, "", ""))
                else:
                    path_strings.append(token)

    # Build the embedding string from full paths (richer than leaf names alone)
    joined = " | ".join(p for p in path_strings if p)
    return joined, triples, path_strings


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
    """Return (embedding_text, qdrant_payload) for a single product."""

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

    # ── Categories ────────────────────────────────────────────────────────────
    # cats_str    → full paths joined for embedding ("Default Cat > Women > Tops | Default Cat > Women > Sale")
    # triples     → [(id, leaf_name, full_path), ...]
    # path_strings→ [full_path, ...] — fed to gender inference so every ancestor
    #               node contributes ("Women" is visible even if leaf is "Hoodies")
    cats_str, cat_triples, path_strings = _resolve_categories(product)
    if cats_str:
        parts.append(f"Category: {cats_str}")

    # Gender — explicit field wins; otherwise infer from full category paths.
    gender = _infer_gender(path_strings, str(product.get("gender") or "").strip())
    if gender:
        parts.append(f"Gender: {gender}")

    tags_str = _resolve_tags(product.get("tags"))
    if tags_str:
        parts.append(f"Tags: {tags_str}")

    # attr_map keyed by attribute `code` (stable identifier) rather than display
    # name so duplicate frontend labels don't collide in the payload.
    attr_map: Dict[str, str] = {}
    for attr_name, attr_code, options in _iter_attributes(product.get("attributes") or []):
        key = normalize_token(attr_code)
        expanded = _expand_options(options)
        parts.append(f"{attr_name}: {expanded}")
        attr_map[key] = ", ".join(options)

        for raw_value in options:
            value_key = normalize_token(raw_value)
            if key and value_key:
                payload[f"attr_{key}_{value_key}"] = True
                if attribute_vocab_sink is not None and value_key != "none":
                    attribute_vocab_sink.setdefault(key, set()).add(value_key)

    # Emit cat_{id} filter keys + feed category vocab sink.
    # Now uses the triple (id, leaf_name, full_path) so the vocab sink stores
    # the leaf name while the full path is what the embedder sees.
    for cid, leaf_name, full_path in cat_triples:
        if not cid:
            continue
        payload[f"cat_{cid}"] = True
        if category_vocab_sink is not None:
            # Prefer leaf_name for the display label; fall back to last segment of path.
            display_name = leaf_name or (full_path.split(">")[-1].strip() if full_path else "")
            if display_name:
                category_vocab_sink[cid] = {
                    "id": cid,
                    "name": normalize_token(display_name),
                    "path": full_path,          # ← store full path so downstream can re-infer
                }

    # HTML stripping. Both fields are also surfaced into the payload below so
    # downstream consumers (RAG summariser, ProductDetailAgent's fallback,
    # `_format_product_source` in retrieve.py) can read them after retrieval.
    # Without that, a product hit comes back with name+price+stock but no
    # actual descriptive text — same shape-mismatch bug that left store_config
    # snippets empty on the prompt side.
    short = html_to_structured_text(product.get("short_description") or product.get("summary") or "")
    if short:
        parts.append(f"Summary: {short}")
    long_desc = html_to_structured_text(product.get("description") or product.get("content") or "")
    if long_desc:
        # Increased from 600 → 2000 chars so product descriptions are not silently truncated.
        parts.append(f"Description: {long_desc}")

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

    if variant_attrs:
        summary_chunks: list[str] = []
        child_skus: list[str] = []
        for attr_code, values in variant_attrs.items():
            key = normalize_token(attr_code)
            if not key:
                continue
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

    # ── Build payload ─────────────────────────────────────────────────────────
    # `categories` stored as full path strings so any future re-indexing or
    # agent that reads the payload can still infer gender without re-fetching.
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
            # Store full paths in the payload so the agent can display breadcrumbs.
            "categories": cats_str,
            "category_paths": path_strings,                    # ← NEW: list of full paths
            "category_ids": [cid for cid, _, _ in cat_triples],
            "tags": tags_str,
            "image_url": image_url,
            "stock_status": product.get("stock_status") or "instock",
            "average_rating": float(product.get("average_rating") or 0),
            "type_id": type_id,
            "is_configurable": is_configurable,
            "has_variants": has_variants,
            "variant_attributes": variant_attrs,
            # Plain-text descriptions, capped so they don't blow up the payload.
            # `short_description` doubles as a snippet for product cards; the
            # full `description` powers the RAG summariser and the
            # "tell me about this product" fallback path.
            "short_description": short[:600],
            "description": long_desc[:2000],
            "children": _clean_children_for_payload(children),
            "child_skus": ",".join(
                str(c.get("sku")) for c in children if isinstance(c, dict) and c.get("sku")
            ),
            **attr_map,
        }
    )

    embedded_text = _final_clean("\n".join(p for p in parts if p))
    return embedded_text, payload


# ── CMS page / block / widget / store config ─────────────────────────────────

def format_cms_page(page: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Build (embedding_text, qdrant_payload) for one CMS page.

    The embedding text now includes every retrieval-relevant field the
    Magento provider ships — URL key, content heading, SEO title, SEO
    description, keywords, then content. This matters because:

      - meta_keywords is where merchants put synonyms ("returns, refunds,
        exchanges, money back") that customers actually use in queries but
        rarely appear in the page body verbatim.
      - meta_description is curated 1-2 sentence prose, perfect for both
        embedding signal and as a card snippet.
      - content_heading is the on-page display heading, often more
        descriptive than the URL-friendly title.

    Length budget (rationale — see also the cms_block formatter below):

      - Embedding text capped at 6000 chars (~1500 tokens). Sits well under
        Gemini embedding-001's 2048-token ceiling, leaving room for the
        title / heading / keywords lines that prefix the body. Any longer
        and the front of the page dominates the embedding so heavily that
        a query about the LAST paragraph of a policy stops matching.
        Beyond ~6000 chars, chunking (one Qdrant point per section) is the
        right answer rather than further bumping this cap.

      - Payload `content` capped at 15000 chars. This is what the LLM reads
        via _format_cms_source. Keeping more here than the embedding sees
        means a page can be indexed by its first 6000 chars, but once
        retrieved the LLM gets enough context to answer about later
        sections too. 15000 chars is comfortable for one source inside a
        prompt that ships up to 6 sources.

      - Pages longer than ~15000 chars: the tail is silently dropped from
        the payload too. If your store routinely has pages that long,
        chunking — a separate Qdrant point per logical section, sharing
        title/permalink — is the next refactor.
    """
    title            = str(page.get("title") or page.get("name") or "").strip()
    identifier       = str(page.get("identifier") or "").strip()
    content_heading  = str(page.get("content_heading") or "").strip()
    meta_title       = str(page.get("meta_title") or "").strip()
    meta_description = str(page.get("meta_description") or "").strip()
    meta_keywords    = str(page.get("meta_keywords") or "").strip()
    content          = html_to_structured_text(page.get("content") or "")

    # Snippet that ends up on the customer-facing card. The Magento provider
    # already computes this with the same priority (meta_description first,
    # then content[:300]) but we recompute defensively in case the provider
    # didn't run the new logic — old payloads still in flight during a
    # rolling deploy will fall through correctly.
    summary = str(page.get("summary") or "").strip()
    if not summary:
        summary = meta_description or content[:300]

    parts: list[str] = []
    if title:
        parts.append(f"CMS Page: {title}")
    if content_heading and content_heading.lower() != title.lower():
        parts.append(f"Heading: {content_heading}")
    if meta_title and meta_title.lower() not in {title.lower(), content_heading.lower()}:
        parts.append(f"SEO Title: {meta_title}")
    if identifier:
        parts.append(f"URL Key: {identifier}")
    if meta_description:
        parts.append(f"Description: {meta_description}")
    if meta_keywords:
        parts.append(f"Keywords: {meta_keywords}")

    # Same factual-anchor detection as cms_block — merchants sometimes
    # put the store address or contact info on a page like `about-us`
    # or `home` rather than a footer block, and pages with cryptic
    # identifiers (`pp` for privacy policy, `info_main` for a catch-all
    # info page) need the same help being retrievable for queries
    # about the facts they happen to contain.
    anchors = _detect_factual_anchors(content)
    if anchors:
        parts.append(f"Indexing hints: {anchors}")

    if content:
        parts.append(f"Content: {content}")

    payload = {
        "title":            title,
        "content_heading":  content_heading,
        "meta_title":       meta_title,
        "meta_description": meta_description,
        "meta_keywords":    meta_keywords,
        "identifier":       identifier,
        # 15000-char payload cap — what the LLM gets to read at retrieval
        # time. Bigger than the embedding cap on purpose: the embedding
        # only needs enough to hit a similarity match, the LLM needs enough
        # to actually answer detail questions about later parts of the page.
        "content":          content,
        "summary":          summary[:600],
        "permalink":        str(page.get("permalink") or ""),
        "status":           str(page.get("status") or "active"),
    }
    return _final_clean("\n".join(parts)), payload


# ─── Factual-content anchor detection ──────────────────────────────────────
#
# Merchants don't always file information where it "belongs". The store's
# address often lives inside a footer CMS block named `sirena_footer` or
# `footer_v2`, not the Magento general/store_information/* fields. The
# return policy lives mixed into a page called `info_main` rather than a
# dedicated `return-policy` page. Vector search can't find these for
# queries like "what's the store address" because the block's title and
# identifier carry zero address-related signal — only the body bytes do,
# and they have to fight for attention against blocks with better names.
#
# These helpers scan the cleaned content for factual patterns (address-
# shaped sequences, phone numbers, emails, hours-of-operation phrases,
# policy keywords, payment-method mentions, VAT/tax identifiers). When
# they fire, the formatter prepends explicit "this block contains X"
# anchor sentences to the EMBEDDING text. The block becomes retrievable
# for the natural-language queries customers actually use, regardless of
# the merchant's identifier-naming hygiene. The PAYLOAD body is
# untouched — the LLM still sees only the merchant's real content.
#
# Patterns kept deliberately broad (UK + US + generic numeric formats)
# so the same code works across the geographies a Magento install can
# serve. False positives are cheap (extra anchor text doesn't break
# anything); false negatives are the failure mode we're fighting.

_RE_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b")
_RE_US_ZIP     = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_RE_GENERIC_POSTAL = re.compile(r"\b\d{4,6}\b")
_RE_ADDRESS_WORDS = re.compile(
    r"\b(?:unit|suite|floor|street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|"
    r"way|drive|dr\.?|plaza|park|estate|industrial|business)\b",
    re.IGNORECASE,
)
_RE_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s\-\.]?)?(?:\(?\d{2,5}\)?[\s\-\.]?){2,4}\d{2,5}"
)
_RE_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.IGNORECASE)
_RE_HOURS = re.compile(
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\s*(?:[\-–to]+|–)\s*"
    r"(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:[\-–to]+|until)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    re.IGNORECASE,
)
_RE_POLICY_KEYWORDS = re.compile(
    r"\b(?:returns?|refunds?|warranty|guarantee|exchange|cancellation|"
    r"delivery|shipping|dispatch|track(?:ing)?|policy|policies)\b",
    re.IGNORECASE,
)
_RE_PAYMENT_KEYWORDS = re.compile(
    r"\b(?:visa|mastercard|paypal|stripe|klarna|clearpay|afterpay|amex|"
    r"american\s+express|maestro|discover|apple\s+pay|google\s+pay|"
    r"bank\s+transfer|cash\s+on\s+delivery|cod\b)\b",
    re.IGNORECASE,
)
_RE_TAX_KEYWORDS = re.compile(
    r"\b(?:vat\b|gst\b|sales\s+tax|company\s+reg(?:istration)?(?:\s+(?:number|no))?"
    r"|reg\.?\s*no\.?)\b",
    re.IGNORECASE,
)


def _detect_factual_anchors(content: str) -> str:
    """Scan messy CMS content for factual patterns and emit anchor
    sentences that semantically describe what the block contains.

    The anchors are prepended to the embedding text (not the payload), so
    they help vector search find this block for queries about those facts
    without changing what the LLM ultimately reads in its prompt.

    Returns a single string (sentences space-joined) ready to drop into
    the formatter's `parts` list. Empty when nothing factual is detected
    — most decorative banners and CTA blocks fall through unchanged.
    """
    if not content:
        return ""

    anchors: list[str] = []

    # Address: requires BOTH a postal-code-shaped sequence AND an
    # address-word ("Unit", "Street", "Avenue", etc.). Either alone has
    # too many false positives (a random 4-digit number; the word
    # "street" mentioned in marketing copy). The combination is a strong
    # signal.
    has_postcode = (
        _RE_UK_POSTCODE.search(content) is not None
        or _RE_US_ZIP.search(content) is not None
    )
    if has_postcode and _RE_ADDRESS_WORDS.search(content):
        anchors.append(
            "This block contains the store's physical and mailing address. "
            "Where the store is located. Postal address. Office address."
        )

    if _RE_PHONE.search(content):
        anchors.append(
            "This block contains a phone number. "
            "Customer service phone. Contact phone number. Tel."
        )

    if _RE_EMAIL.search(content):
        anchors.append(
            "This block contains an email address. "
            "Contact email. Customer support email. Get in touch."
        )

    if _RE_HOURS.search(content):
        anchors.append(
            "This block contains opening or business hours. "
            "When the store is open. Trading hours."
        )

    if _RE_POLICY_KEYWORDS.search(content):
        anchors.append(
            "This block references store policies — "
            "returns, refunds, warranty, shipping, delivery, cancellation."
        )

    if _RE_PAYMENT_KEYWORDS.search(content):
        anchors.append(
            "This block mentions accepted payment methods — "
            "credit / debit cards, PayPal, bank transfer, cash on delivery."
        )

    if _RE_TAX_KEYWORDS.search(content):
        anchors.append(
            "This block contains VAT / tax / company registration details."
        )

    return " ".join(anchors)


def format_cms_block(block: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Build (embedding_text, payload) for one CMS block.

    Length budget — half the cms_page caps because blocks are usually
    short reusable snippets (footer copy, banner text, contact-info widgets)
    rather than full articles. Bumping these too high just dilutes the
    embedding without buying retrieval quality.

      - Embedding text capped at 3000 chars (~750 tokens).
      - Payload content capped at 8000 chars — the LLM gets the full block
        text even for unusually long blocks (FAQ blocks, multi-paragraph
        legal disclaimers).

    Factual anchors (see _detect_factual_anchors above) are prepended to
    the embedding text when address / phone / email / hours / policy /
    payment / tax patterns are detected in the body. This is what makes
    a footer block named `sirena_footer` retrievable for "what's the
    store address" queries even when the merchant's content hygiene is
    nonexistent.
    """
    title = str(block.get("title") or block.get("name") or "").strip()
    identifier = str(block.get("identifier") or "").strip()
    content = html_to_structured_text(block.get("content") or "")

    parts = [f"CMS Block: {title}"] if title else []
    if identifier:
        parts.append(f"Identifier: {identifier}")

    # Factual-content anchors first, BEFORE the raw content — they're
    # short and high-signal, so leading with them gives the embedding
    # the right semantic centre of mass when the rest of the body is
    # marketing copy or HTML cruft.
    anchors = _detect_factual_anchors(content)
    if anchors:
        parts.append(f"Indexing hints: {anchors}")

    if content:
        parts.append(f"Content: {content}")

    payload = {
        "title": title,
        "identifier": identifier,
        "content": content,
        "summary": content[:300],
        "status": block.get("status") or "active",
    }
    return _final_clean("\n".join(parts)), payload


def format_promotion(promo: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Build (embedding_text, payload) for one active promotion / coupon rule.

    The Magento provider already does most of the prose construction (title +
    discount phrase + free shipping + description + coupon + validity, in
    that order, joined into `content`). Here we just normalise the payload
    and build a slightly more retrieval-friendly embedding text that
    repeats key signal tokens — "promotion", "discount", "coupon",
    "promo code" — so customer queries using any of those phrasings hit
    the right point even when the rule's name doesn't include them.
    """
    title         = str(promo.get("name") or promo.get("title") or "Promotion").strip()
    description   = str(promo.get("description") or "").strip()
    discount_text = str(promo.get("discount_text") or "").strip()
    coupon_code   = str(promo.get("coupon_code") or "").strip()
    free_shipping = bool(promo.get("free_shipping", False))
    from_date     = str(promo.get("from_date") or "").strip()
    to_date       = str(promo.get("to_date") or "").strip()
    body          = str(promo.get("content") or "").strip()

    # Embedding text — lead with retrieval-boosting category words so a
    # customer asking "any active discounts" / "promo codes" / "current
    # deals" matches even rules whose name is a marketing slogan rather
    # than a product-shape word.
    parts = [
        f"Promotion: {title}",
        "Active discount, deal, sale, offer, special, coupon code.",
    ]
    if discount_text:
        parts.append(f"Discount: {discount_text}")
    if free_shipping:
        parts.append("Free shipping included.")
    if description:
        parts.append(f"Description: {description}")
    if coupon_code:
        parts.append(f"Coupon code: {coupon_code}")
    if from_date or to_date:
        when = []
        if from_date: when.append(f"from {from_date}")
        if to_date:   when.append(f"until {to_date}")
        parts.append("Valid " + " ".join(when))

    payload = {
        "title":         title,
        "name":          title,
        "description":   description,
        "discount_text": discount_text,
        "coupon_code":   coupon_code,
        "free_shipping": free_shipping,
        "from_date":     from_date,
        "to_date":       to_date,
        "content":       body or " ".join(parts),
        "summary":       (title + (" — " + discount_text if discount_text else ""))[:300],
        "permalink":     str(promo.get("permalink") or ""),
        "status":        str(promo.get("status") or "active"),
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
    """Build (embedding_text, payload) for a store_config composite snippet.

    Adapted to the composite-snippet rewrite on the Magento side. Each row
    now arrives as a rich blob with semantic anchors + factual data baked
    into `content`, rather than a single bare label/value pair. The key
    things the formatter does:

      - Drops the old "Key: store_phone" line that used to sit between
        label and value. The snake_case identifier was admin metadata —
        zero retrieval value, just diluted the embedding.
      - Uses `content` as the primary body (composite anchors + facts).
        Falls back to `value` for legacy per-field rows from older syncs
        so this formatter stays compatible with un-migrated points until
        a re-sync.
      - Surfaces both `value` and `content` in the payload — _format_hit
        and frontend cards each have their own preference. Storing both
        is cheap and avoids per-consumer translation.

    Length budget: composite content can be 1-3KB after anchors + facts.
    Caps mirror cms_block — generous enough for any single composite
    without bloating the embedding window.
    """
    key   = str(info.get("key") or info.get("identifier") or "").strip()
    label = str(info.get("label") or info.get("title") or key).strip()

    # Content is the new composite body (anchors + facts). Value is the
    # legacy short-form field — for new rows it carries the facts only
    # (for chat-card display); for legacy per-field rows from old syncs
    # it carries the bare value.
    raw_content = info.get("content") or info.get("value") or ""
    raw_value   = info.get("value")   or info.get("content") or ""
    if isinstance(raw_content, (dict, list)):
        raw_content = html_mod.unescape(str(raw_content))
    if isinstance(raw_value, (dict, list)):
        raw_value = html_mod.unescape(str(raw_value))

    content = html_to_structured_text(str(raw_content))
    value   = html_to_structured_text(str(raw_value))

    # Embedding text — label as a one-line header, then the composite body.
    # Composite bodies already lead with semantic anchors so retrieval
    # catches the variety of customer phrasings; we don't add more here.
    parts: list[str] = []
    if label:
        parts.append(label + ".")
    if content:
        parts.append(content)

    # Summary is the short form the chat card shows. It comes from the
    # Magento composite (first 300 chars of facts only — without the
    # semantic anchors). Falls back to the value field for legacy per-row
    # syncs that don't carry summary.
    raw_summary = str(info.get("summary") or value or "")
    summary = html_to_structured_text(raw_summary)[:300]

    # Payload key order matters for _format_hit's snippet fallback chain
    # (summary → excerpt → content → description → value → short_description).
    # We want chat cards to show the short fact-only summary, NOT the full
    # composite body with semantic anchors at the front. Setting summary
    # explicitly here puts the right thing first in the chain.
    payload = {
        "key":     key,
        "label":   label,
        "summary": summary,
        "value":   value,     # short form for card body fallback
        "content": content,  # full composite body for the LLM
    }
    return _final_clean("\n".join(parts)), payload


# ── Chunkable variants (CMS pages + blocks) ─────────────────────────────────
#
# The chunked path splits *body* across N Qdrant points while repeating
# *header* (title, identifier, SEO metadata, factual anchors) on each chunk's
# embedding. That keeps every chunk's vector grounded in the page it came
# from — without the header repeat, a paragraph 3000 chars into a "Returns
# Policy" page would embed as just its own words, with nothing tying it back
# to the policy concept for retrieval purposes.
#
# Returns: (header_text, body_text, base_payload). The sync router:
#   1. chunks body_text into N chunks
#   2. for each chunk i, embeds (header_text + "\nContent: " + chunk_i)
#   3. upserts N points sharing base_payload but with per-chunk:
#        - point id    = uuid5("{client}-{type}-{entity}-{store}-chunk-{i}")
#        - content     = the chunk body (not the full page)
#        - chunk_index = i
#        - total_chunks = N
#        - parent_entity_id = entity_id
#
# Length budgets here are *body* budgets — the header is short (~200-400
# chars) and adds little to each chunk. Bumping body chunk size much past
# 500 chars defeats the chunking goal (the first sentence of a long chunk
# dominates the embedding again).

def format_cms_page_chunkable(page: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Chunkable variant of format_cms_page. See module-level note above."""
    title            = str(page.get("title") or page.get("name") or "").strip()
    identifier       = str(page.get("identifier") or "").strip()
    content_heading  = str(page.get("content_heading") or "").strip()
    meta_title       = str(page.get("meta_title") or "").strip()
    meta_description = str(page.get("meta_description") or "").strip()
    meta_keywords    = str(page.get("meta_keywords") or "").strip()
    content          = html_to_structured_text(page.get("content") or "")

    summary = str(page.get("summary") or "").strip()
    if not summary:
        summary = meta_description or content[:300]

    # Header — repeated verbatim on every chunk's embedding. Keep it tight
    # (no full content here); the goal is to identify which page each chunk
    # belongs to, not duplicate the body.
    header_parts: list[str] = []
    if title:
        header_parts.append(f"CMS Page: {title}")
    if content_heading and content_heading.lower() != title.lower():
        header_parts.append(f"Heading: {content_heading}")
    if meta_title and meta_title.lower() not in {title.lower(), content_heading.lower()}:
        header_parts.append(f"SEO Title: {meta_title}")
    if identifier:
        header_parts.append(f"URL Key: {identifier}")
    if meta_description:
        header_parts.append(f"Description: {meta_description}")
    if meta_keywords:
        header_parts.append(f"Keywords: {meta_keywords}")
    # Factual anchors run on the FULL body once (anchors fire on content
    # signals — postcode, phone, hours — that may live anywhere in the
    # page). Attach to the header so every chunk benefits, not just the
    # chunk that happens to contain the anchored text.
    anchors = _detect_factual_anchors(content)
    if anchors:
        header_parts.append(f"Indexing hints: {anchors}")

    header_text = _final_clean("\n".join(header_parts))

    # base_payload is the per-page metadata copied onto every chunk. The
    # sync router overlays chunk-specific fields (content, chunk_index,
    # total_chunks, parent_entity_id, summary).
    base_payload = {
        "title":            title,
        "content_heading":  content_heading,
        "meta_title":       meta_title,
        "meta_description": meta_description,
        "meta_keywords":    meta_keywords,
        "identifier":       identifier,
        # Note: `content` is intentionally OMITTED from base_payload —
        # the sync router sets it per chunk to the chunk body. Letting
        # the full page content leak through would defeat the
        # "matched paragraph" semantic the chunking promises.
        "summary":          summary[:600],
        "permalink":        str(page.get("permalink") or ""),
        "status":           str(page.get("status") or "active"),
    }
    return header_text, content, base_payload


def format_cms_block_chunkable(block: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Chunkable variant of format_cms_block. See module-level note above."""
    title      = str(block.get("title") or block.get("name") or "").strip()
    identifier = str(block.get("identifier") or "").strip()
    content    = html_to_structured_text(block.get("content") or "")

    header_parts: list[str] = []
    if title:
        header_parts.append(f"CMS Block: {title}")
    if identifier:
        header_parts.append(f"Identifier: {identifier}")
    anchors = _detect_factual_anchors(content)
    if anchors:
        header_parts.append(f"Indexing hints: {anchors}")

    header_text = _final_clean("\n".join(header_parts))

    base_payload = {
        "title":      title,
        "identifier": identifier,
        # `content` deliberately omitted — set per chunk by the sync router.
        "summary":    content[:300],
        "status":     block.get("status") or "active",
    }
    return header_text, content, base_payload


# ── FAQ (merchant-authored knowledge) ────────────────────────────────────────
#
# FAQ entries are the merchant's catch-all for questions the catalog + CMS
# don't answer. Authored in the admin as one textarea, parsed Magento-side
# into (title, content) entries by the `#`-heading convention — each `#`
# line is a title, the lines under it are that entry's answer. They reach
# the chatbot ONLY as a last resort: the three CMS-style agents
# (policy_faq / store_info / general) consult faq only when their primary
# sources retrieve nothing relevant.
#
# Chunkable like cms_block — short answers stay one chunk; a long answer is
# sliced with the title repeated as the header anchor on every chunk. The
# title is the whole point: a bare paragraph has nothing distinctive to
# embed on, so the merchant's title (ideally phrased as the customer's
# question) is what a query actually matches.


def format_faq_chunkable(faq: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Chunkable variant for one parsed FAQ entry — returns (header, body, payload).

    `title` is the merchant's heading line; `content` is the answer body.
    The title becomes the header repeated on every chunk's embedding (see
    sync._process_chunkable_item) — the semantic anchor for short answers.
    """
    title   = str(faq.get("title") or "").strip()
    content = html_to_structured_text(faq.get("content") or "")

    header_parts: list[str] = []
    if title:
        header_parts.append(f"FAQ: {title}")
    anchors = _detect_factual_anchors(content)
    if anchors:
        header_parts.append(f"Indexing hints: {anchors}")

    header_text = _final_clean("\n".join(header_parts))

    base_payload = {
        "title":   title,
        # `content` deliberately omitted — set per chunk by the sync router.
        "summary": content[:300],
        "status":  "active",
    }
    return header_text, content, base_payload


def format_faq(faq: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Single-point variant for the format_item fallback path (faq normally
    chunks via format_faq_chunkable). Mirrors format_category's wrapper.
    """
    header, body, payload = format_faq_chunkable(faq)
    payload["content"] = body
    text = f"{header}\nContent: {body}" if header else body
    return _final_clean(text), payload


# ── Categories ──────────────────────────────────────────────────────────────
#
# Magento catalog categories carry merchant-curated taxonomy data that
# the chatbot needs for browse-style queries: "what departments do you
# have?", "tell me about your modern water features collection",
# "do you have anything for kids?". Without this content type the bot
# can describe individual products (via /retrieve/products) but has no
# way to talk about the *groupings* the customer sees on the storefront.
#
# Chunking rationale: SEO-heavy stores routinely have 1-3k char category
# landing pages. Single-vector embedding lets the intro paragraph
# dominate while later sections (sub-collections, sizing guides,
# "what to look for" content) score poorly even when they're the answer.
# Same problem cms_page solved via chunking in Phase 1.3 — categories
# join the chunkable list (see CHUNKABLE_CONTENT_TYPES in
# qdrant_service.py).


def format_category(category: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Build (embedding_text, payload) for one catalog category.

    Kept as a single-output function so the format_item dispatcher's
    fallback path (non-chunked retrievals, scripts that don't know
    about the chunkable formatters) still works. The chunked path
    goes through format_category_chunkable below.
    """
    text, _, payload = _format_category_parts(category)
    # Non-chunked path needs `content` in the payload too — the chunked
    # path sets content per-chunk, but the single-shot path expects
    # the LLM to see the full body.
    payload["content"] = text
    return _final_clean(text), payload


def format_category_chunkable(category: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Chunkable variant — returns (header, body, payload).

    The chunked sync (sync._process_chunkable_item) takes these three
    pieces, chunks `body` into ~500-char chunks with overlap, and
    upserts each chunk as its own Qdrant point sharing the payload
    plus chunk-specific {chunk_index, parent_entity_id, total_chunks}.

    Header carries identity (name, breadcrumb, URL key, meta) so every
    chunk's embedding stays grounded in the category it came from.
    Body is the description text only — that's what gets sliced.
    """
    return _format_category_parts(category)


def _format_category_parts(category: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Shared field extraction. Single source of truth so the
    non-chunked and chunked formatters produce identical content,
    just sliced differently.
    """
    name             = str(category.get("name") or "").strip()
    identifier       = str(category.get("identifier") or category.get("url_key") or "").strip()
    meta_title       = str(category.get("meta_title") or "").strip()
    meta_description = str(category.get("meta_description") or "").strip()
    meta_keywords    = str(category.get("meta_keywords") or "").strip()
    breadcrumb       = str(category.get("breadcrumb") or "").strip()
    description      = html_to_structured_text(category.get("description") or "")

    # Snippet — what the customer sees on the chat card. Prefer the
    # merchant's curated meta_description; fall back to the first 300
    # chars of the description body when no meta is set.
    summary = str(category.get("summary") or "").strip()
    if not summary:
        summary = meta_description or description[:300]

    # Header — repeated on every chunk's embedding so each chunk stays
    # grounded in the category it belongs to. Without this, a chunk
    # from the middle of a long category page would embed as just its
    # own words, losing the cross-concept signal that ties it back to
    # the parent category for retrieval.
    header_parts: list[str] = []
    if name:
        # Mark as a category explicitly so the embedding doesn't get
        # confused with a similarly-named product. Categories are
        # taxonomy nodes, not items for sale — making that explicit
        # in the embed text helps disambiguation on
        # "do you have a Solar category?" type queries.
        header_parts.append(f"Category: {name}")
    if breadcrumb and breadcrumb.lower() != name.lower():
        # "Outdoor > Water Features > Solar" gives the LLM the taxonomy
        # context — answers like "Solar is part of our Water Features
        # range" require knowing where in the tree it sits.
        header_parts.append(f"Path: {breadcrumb}")
    if meta_title and meta_title.lower() != name.lower():
        header_parts.append(f"SEO Title: {meta_title}")
    if identifier:
        header_parts.append(f"URL Key: {identifier}")
    if meta_description and meta_description.lower() != summary.lower():
        header_parts.append(f"Description: {meta_description}")
    if meta_keywords:
        # meta_keywords often carries customer-phrased synonyms
        # ("outdoor, garden, patio, deck") that don't appear in the
        # category name or description but matter for retrieval.
        header_parts.append(f"Keywords: {meta_keywords}")

    # Same factual-anchor detection that CMS uses — catches the
    # ambiguous-naming case where a category description happens to
    # contain address / phone / hours info (rare but cheap to detect).
    anchors = _detect_factual_anchors(description)
    if anchors:
        header_parts.append(f"Indexing hints: {anchors}")

    header_text = _final_clean("\n".join(header_parts))

    # base_payload omits `content` on purpose — see the chunked-formatter
    # contract in format_cms_page_chunkable for the rationale. The
    # non-chunked wrapper (format_category) sets content from text.
    base_payload: Dict[str, Any] = {
        "name":              name,
        "title":             name,           # alias — frontend renders s.title
        "identifier":        identifier,
        "url_key":           identifier,
        "meta_title":        meta_title,
        "meta_description":  meta_description,
        "meta_keywords":     meta_keywords,
        "breadcrumb":        breadcrumb,
        "summary":           summary[:600],
        "permalink":         str(category.get("permalink") or ""),
        "image_url":         str(category.get("image_url") or ""),
        "parent_id":         category.get("parent_id"),
        "level":             category.get("level"),
        "status":            "active" if category.get("is_active", True) else "inactive",
    }

    return header_text, description, base_payload


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
    if content_type == "category":
        return format_category(item)
    if content_type == "widget":
        return format_widget(item)
    if content_type == "store_config":
        return format_store_config(item)
    if content_type == "promotion":
        return format_promotion(item)
    if content_type == "faq":
        return format_faq(item)
    title = str(item.get("title") or item.get("name") or item.get("identifier") or "")
    content = html_to_structured_text(item.get("content") or item.get("description") or "")
    return (
        _final_clean(f"{content_type}: {title}\n{content[:1500]}"),
        {"title": title, "content": content[:1500], "content_type": content_type},
    )