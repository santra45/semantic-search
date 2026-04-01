"""
rerank_service.py
─────────────────
Post-search keyword filtering & re-ranking.

Pipeline:
  raw customer query
        │
        ▼
  extract_keywords()  ← pure Python dict/regex, NO LLM
        │  returns gender, colors, materials detected in query
        ▼
  filter_and_rerank() ← runs on Qdrant result list
        │  • BLOCKS products whose payload text contains the
        │    *opposite* gender words when gender is explicit
        │  • BOOSTS products whose payload text mentions
        │    queried colors / materials (soft score)
        ▼
  cleaned, re-scored results (sliced to original limit)

Graceful fallback: if blocking removes ALL products,
the top-3 by original Qdrant vector score are returned
so the UI never shows an empty result set.
"""

import re
from typing import Optional

# ─── Keyword Dictionaries ──────────────────────────────────────────────────────

# Gender → synonyms that mean this gender in a customer query
GENDER_QUERY_SYNONYMS: dict[str, list[str]] = {
    "Men": [
        # core
        "men", "man", "male", "males", "mens",

        # formal / retail
        "gent", "gents", "gentleman", "gentlemen",
        "menswear", "menswears",

        # casual / slang
        "guy", "guys", "dude", "dudes", "bro", "bros",
        "boy", "boys", "lad", "lads",

        # roles
        "husband", "father", "dad", "dads", "boyfriend",

        # pronouns
        "him", "his", "he",

        # descriptive
        "masculine", "manly", "for him",

        # common typos
        "menss", "manz", "mal", "mail"
    ],

    "Women": [
        # core
        "women", "woman", "female", "females", "womens",

        # formal / retail
        "lady", "ladies", "womenswear", "womenswears",

        # casual / slang
        "girl", "girls", "gal", "gals", "chick", "chicks",
        "diva", "queens",

        # roles
        "wife", "mother", "mom", "moms", "girlfriend",

        # pronouns
        "her", "hers", "she",

        # descriptive
        "feminine", "ladylike", "for her",

        # life-stage specific
        "maternity", "pregnancy", "pregnant", "nursing",

        # common typos
        "womans", "wemens", "femail"
    ],

    "Kids": [
        # core
        "kid", "kids", "child", "children",

        # age groups
        "baby", "babies", "infant", "infants",
        "toddler", "toddlers", "newborn", "newborns",

        # casual
        "little one", "little ones", "youngster", "youngsters",

        # gendered kids
        "boy", "boys", "girl", "girls",

        # retail categories
        "youth", "junior", "jr", "preteen", "teen", "teenager",

        # usage context
        "school", "schoolwear", "playwear", "daycare",

        # family phrasing
        "son", "daughter", "baby boy", "baby girl",

        # common typos
        "childern", "babby", "todler"
    ],
}
# Gender → words in *product text* that signal this gender
# (product names, categories, tags, attribute values)
GENDER_PRODUCT_WORDS: dict[str, list[str]] = {
    "Men": [
        # core
        "men", "mens", "man", "male", "males",

        # retail
        "menswear", "mens clothing", "mens fashion",

        # casual labels
        "gents", "guy", "boys",

        # relationships
        "boyfriend", "husband",

        # descriptive
        "masculine", "for him", "his",

        # category hints
        "mens shirt", "mens shoes", "mens jeans"
    ],

    "Women": [
        # core
        "women", "womens", "woman", "female", "females",

        # retail
        "womenswear", "womens clothing", "ladieswear",

        # casual labels
        "ladies", "girl", "girls",

        # relationships
        "girlfriend", "wife",

        # descriptive
        "feminine", "for her", "her",

        # life-stage
        "maternity", "nursing", "feeding",

        # category hints
        "womens dress", "womens shoes", "womens top"
    ],

    "Kids": [
        # core
        "kid", "kids", "child", "children",

        # age groups
        "baby", "babies", "infant", "toddler",

        # retail
        "kidswear", "babywear", "childrenwear",

        # gendered kids
        "boys", "girls",

        # school context
        "school", "schoolwear", "uniform",

        # usage
        "playwear", "sleepwear",

        # family context
        "for kids", "for baby", "for children"
    ],
}

# Build opposite-gender lookup: "Men" → Women product words, etc.
GENDER_OPPOSITES: dict[str, list[str]] = {
    gender: [
        word
        for other_gender, words in GENDER_PRODUCT_WORDS.items()
        if other_gender != gender
        for word in words
    ]
    for gender in GENDER_PRODUCT_WORDS
}

COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "grey", "gray",
    "pink", "purple", "violet", "orange", "brown", "maroon", "beige",
    "cream", "navy", "cyan", "magenta", "gold", "silver", "teal",
    "indigo", "khaki", "olive", "coral", "turquoise",
]

MATERIALS = [
    "cotton", "polyester", "silk", "wool", "linen", "denim", "leather",
    "nylon", "rayon", "spandex", "fleece", "velvet", "satin", "chiffon",
    "georgette", "lycra", "jersey", "crepe", "net", "lace", "tweed",
    "cashmere", "suede", "canvas", "synthetic", "blended",
]

# ─── Stopwords to strip before soft-token matching ────────────────────────────

STOPWORDS = {
    "i", "want", "need", "looking", "for", "a", "an", "the", "some",
    "any", "please", "can", "you", "show", "me", "find", "get", "buy",
    "am", "is", "are", "my", "something", "good", "nice", "best",
    "cheap", "expensive", "new", "old", "like", "love", "prefer",
    "give", "suggest", "recommend",
}


# ─── Keyword Extraction ────────────────────────────────────────────────────────

def extract_keywords(query: str) -> dict:
    """
    Extract structured signals from a raw customer query.
    Completely pure Python — zero external calls.

    Returns:
        {
            "gender":    "Men" | "Women" | None,
            "colors":    ["red", "black", ...],
            "materials": ["cotton", ...],
            "tokens":    ["tshirt", "casual", ...]   ← leftover meaningful words
        }
    """
    # Normalise: lowercase, collapse whitespace, remove punctuation
    text = re.sub(r"[^\w\s]", " ", query.lower())
    words = text.split()

    detected_gender: Optional[str] = None
    detected_colors: list[str] = []
    detected_materials: list[str] = []
    meaningful_tokens: list[str] = []

    for word in words:
        # 1. Gender (blocking signal)
        matched_gender = None
        for gender, synonyms in GENDER_QUERY_SYNONYMS.items():
            if word in synonyms:
                matched_gender = gender
                break
        if matched_gender:
            # Last gender word wins (handles "for men and women" edge-case
            # by not setting detected_gender, user gets unfiltered results)
            if detected_gender and detected_gender != matched_gender:
                # Conflicting genders — treat as no gender filter
                detected_gender = None
            else:
                detected_gender = matched_gender
            continue  # don't add gender words to soft tokens

        # 2. Color (soft signal)
        if word in COLORS:
            detected_colors.append(word)
            meaningful_tokens.append(word)
            continue

        # 3. Material (soft signal)
        if word in MATERIALS:
            detected_materials.append(word)
            meaningful_tokens.append(word)
            continue

        # 4. Stopwords — skip
        if word in STOPWORDS or len(word) <= 2:
            continue

        meaningful_tokens.append(word)

    return {
        "gender":    detected_gender,
        "colors":    detected_colors,
        "materials": detected_materials,
        "tokens":    meaningful_tokens,
    }


# ─── Product Text Builder ─────────────────────────────────────────────────────

def _build_product_text(product: dict) -> str:
    """
    Concatenate all text fields from a Qdrant result payload into
    one lowercase string for keyword matching.
    Includes name, categories, tags, and every dynamic attribute value.
    """
    parts = [
        product.get("name", ""),
        product.get("categories", ""),
        product.get("tags", ""),
    ]

    # Dynamic attribute values stored at top level of payload
    # e.g. "gender": "Men", "color": "Red, Blue", "size": "M, L"
    skip_keys = {
        "name", "categories", "tags", "product_id", "price", "permalink",
        "image_url", "stock_status", "score", "currency", "regular_price",
        "sale_price", "on_sale", "average_rating", "sku", "brand",
        "client_id",
    }
    for key, val in product.items():
        if key not in skip_keys and isinstance(val, (str, int, float)):
            parts.append(str(val))

    return " ".join(parts).lower()


# ─── Soft Scorer ──────────────────────────────────────────────────────────────

def _soft_score(product_text: str, keywords: dict) -> float:
    """
    Returns a 0.0–1.0 score for how well the product
    matches soft (non-blocking) signals: colors and materials.
    """
    soft_targets = keywords["colors"] + keywords["materials"]
    if not soft_targets:
        return 0.0
    hits = sum(1 for t in soft_targets if t in product_text)
    return hits / len(soft_targets)


# ─── Gender Blocker ───────────────────────────────────────────────────────────

def _is_wrong_gender(product_text: str, required_gender: str) -> bool:
    """
    Returns True if the product is clearly the *opposite* gender to
    what the customer asked for.

    Logic: a product is "wrong gender" only when it contains one or more
    *opposite-gender* words AND contains NO words for the *required* gender
    nor any "unisex" markers.

    Unisex markers mean we don't block the product.
    """
    unisex_markers = ["unisex", "kids", "children", "child", "toddler", "baby", "infant"]
    if any(m in product_text for m in unisex_markers):
        return False

    opposite_words = GENDER_OPPOSITES.get(required_gender, [])
    required_words = GENDER_PRODUCT_WORDS.get(required_gender, [])

    has_opposite = any(
        re.search(rf"\b{re.escape(w)}\b", product_text)
        for w in opposite_words
    )
    has_required = any(
        re.search(rf"\b{re.escape(w)}\b", product_text)
        for w in required_words
    )

    # Block only if opposite is explicit and required gender is absent
    return has_opposite and not has_required


# ─── Main Filter + Re-rank ────────────────────────────────────────────────────

def filter_and_rerank(results: list, keywords: dict, original_limit: int) -> list:
    """
    Filter and re-rank a list of Qdrant result dicts using extracted keywords.

    Args:
        results:        List of product dicts from qdrant_service.search_products()
        keywords:       Output of extract_keywords()
        original_limit: How many results the caller originally asked for

    Returns:
        Filtered + re-scored list, at most original_limit items.
        Falls back to top-3 by vector score if blocking removes everything.
    """
    if not results or not any(keywords.values()):
        # Nothing to filter — return as-is (sliced)
        return results[:original_limit]

    required_gender = keywords.get("gender")
    blocked: list[dict] = []
    passed: list[dict] = []

    for product in results:
        product_text = _build_product_text(product)

        # ── Gender blocking ───────────────────────────────────────────────────
        if required_gender and _is_wrong_gender(product_text, required_gender):
            blocked.append(product)
            print(
                f"  🚫 Blocked [{product.get('name', '?')}] "
                f"— wrong gender (needed: {required_gender})"
            )
            continue

        # ── Soft score ────────────────────────────────────────────────────────
        soft = _soft_score(product_text, keywords)
        qdrant_score = product.get("score", 0.0)

        # Weighted final score: vector similarity 70%, soft keyword match 30%
        final_score = (qdrant_score * 0.7) + (soft * 0.3)

        product["score"] = round(final_score, 4)
        passed.append(product)

    print(
        f"  🔎 Rerank: {len(passed)} passed, "
        f"{len(blocked)} blocked, "
        f"gender_filter={required_gender}"
    )

    # ── Graceful fallback ─────────────────────────────────────────────────────
    if not passed:
        print("  ⚠️  All products blocked — falling back to top-3 by vector score")
        fallback = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:3]
        return fallback

    # ── Sort by final score, return up to original_limit ─────────────────────
    passed.sort(key=lambda r: r.get("score", 0), reverse=True)
    return passed[:25]
