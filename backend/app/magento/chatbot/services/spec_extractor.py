"""
Product specification extraction (Phase 1 of spec-aware catalog escalation).

WHY THIS EXISTS
    Customers ask comparative questions — "do you have anything that does
    10 GPM?" — but merchants write specifications as prose: "Flow rate:
    8 GPM" buried in a description paragraph. A vector cannot compare, and
    the attribute filter is exact string equality, so today the only field
    in the whole stack that supports a numeric range is `price`.

    This module reads specs out of the full product record at sync time and
    writes them to the payload as structured values, so at query time a
    comparison is an ordinary Qdrant filter — exactly the way price already
    works.

THE DIVISION OF LABOUR
    The model reads; code decides. Specifically:

      * the model is good at reading messy free text, and is the ONLY thing
        that can QUALIFY a key — telling a delivery hose from a suction
        hose in "20FT Delivery Hose, 5FT Suction Hose". A pattern matcher
        sees two conflicting lengths; they are different hoses;
      * the model is measurably worse at arithmetic, so it never converts
        anything. It reports the number and the unit exactly as written;
      * the model can fabricate, so every spec must carry a verbatim quote,
        and that quote is checked against the record here. A spec whose
        source text cannot be found never reaches the index.

DELIBERATELY NOT DONE: UNIT CONVERSION
    There is no gpm↔lpm table and no ft↔in arithmetic. Values are compared
    only within the same unit token. A store that mixes units for one spec
    loses some recall; it never returns a wrong answer. Mixed units are
    largely a multi-source problem and each store is indexed alone, so this
    has not been observed in practice — and when it is, the vocabulary
    makes it visible (one key carrying two units) and a single conversion
    can be added with evidence rather than guessed at up front.

MODEL CHOICE
    2.5 Flash Lite, not Google's newer 3.5 Flash Lite. Extraction is
    output-heavy — a spec list is far longer than the record that produced
    it — and 3.5 Flash Lite bills output at $2.50/M against 2.5's $0.40,
    so it costs roughly 5x more to index the same catalog. Qualification
    was re-checked on 2.5 before switching: delivery and suction hoses
    still separate correctly, and both real conflicts on F00201A1J are
    still found.

See prototypes/spec_extractor/ for the validation run this is derived from:
851 specs over two live catalogs, zero fabrications, zero qualification
errors.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Extraction pins its OWN model rather than borrowing the shared chat
# default. Two reasons: a merchant switching their answer model should not
# silently change how their catalog is indexed, and this reads the entire
# catalog once, so the cheapest capable model is the right one regardless of
# what the conversational side is set to.
DEFAULT_MODEL = os.getenv("SPEC_EXTRACTION_MODEL", "gemini-2.5-flash-lite")
DEFAULT_PROVIDER = os.getenv("SPEC_EXTRACTION_PROVIDER", "google")

# ── Prompt ───────────────────────────────────────────────────────────────

_PROMPT = """You are extracting product specifications from a merchant's catalog record.

The record below is presented as labelled blocks. Extract EVERY specification that is explicitly stated anywhere in it.

Rules:

1. QUALIFY EVERY KEY. If the record distinguishes two things of the same type, their keys MUST differ. A record that mentions both a delivery hose and a suction hose yields `delivery_hose_length` and `suction_hose_length` — never a shared `hose_length`. Read what the measurement belongs to, not merely what unit it carries.

2. QUOTE VERBATIM. `quote` must be copied character-for-character from the record. Do not tidy it, expand it, fix its spelling or repair broken characters. If you cannot quote it exactly, do not report it at all.

3. REPORT REPEATS. If the same specification appears in more than one block, emit it once per block, each with its own quote and its own field. Do not merge them and do not choose between them even when they disagree — disagreement is precisely what this is looking for.

4. NEVER CONVERT OR CALCULATE. Report the number and the unit exactly as written. `9 GPM` stays `9` and `GPM`. `4 ft` stays `4` and `ft`.

5. GIVE A CLEAN UNIT. Report the unit of measurement alone. Compound dimension strings label the axis as well as the unit — in `50H x 76L x 36Wcm` the unit is `cm`, not `Hcm` or `H`. Strip the axis letter; never invent a unit that was not written.

6. NEVER INFER. Only what the text states. No specification implied by the product category, no typical value for this kind of product, no range you assume to be reasonable.

Keys are lower_snake_case. Use `value_num` plus `unit` for measurements; use `value_text` for non-numeric specifications such as a material, a fluid type or a standard. Never populate both.
{vocabulary}
Product record:
---
{record}
---"""

_VOCAB_BLOCK = """
Keys already in use in this store — REUSE one whenever it means the same thing, so the catalog stays filterable. Invent a new key only for a specification genuinely not in this list:
{keys}
"""

# `title` is load-bearing, not decoration. LangChain implements structured
# output by declaring a tool whose FUNCTION NAME is derived from this field.
# Without it the name it generates is empty, and Gemini rejects the whole
# request:
#
#   400 tools[0].function_declarations[0].name: Invalid function name.
#   Must start with a letter or an underscore...
#
# which fails extraction for EVERY product. Some library versions invent a
# fallback name and some do not, so this cannot be left to the library --
# it worked in development and failed on the server for exactly that reason.
_SCHEMA: dict[str, Any] = {
    "title": "extract_product_specifications",
    "description": "Specifications stated in the product record.",
    "type": "object",
    "properties": {
        "specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string"},
                    "value_num": {"type": "number", "nullable": True},
                    "value_text": {"type": "string", "nullable": True},
                    "unit": {"type": "string", "nullable": True},
                    "quote": {"type": "string"},
                    "field": {"type": "string"},
                },
                "required": ["key", "label", "quote", "field"],
            },
        }
    },
    "required": ["specs"],
}

# Fields scanned, in descending order of how reliably merchants keep them
# current. This is also the survivorship ranking used when one value has to
# lead: an explicit attribute beats merchant notes, which beat the curated
# spec summary, which beats the description — the description being the copy
# nobody ever goes back and re-edits, and the source of the stale 8 GPM on
# the dispenser that started all this.
_SOURCE_FIELDS: tuple[str, ...] = (
    "name",
    "merchant_info",
    "short_description",
    "description",
)

# Merchants pad spec tables with pointers instead of values. They are not
# specifications, and left in they contradict every real value on the key.
_PLACEHOLDERS = frozenset({
    "seeitemdescription", "seedescription", "seebelow", "seeimage",
    "na", "none", "varies", "various", "asshown", "refertoimage", "tbc",
})

# Text already corrupted upstream. Sync repairs most mojibake before we see
# it, but anything left is noise about the merchant's encoding rather than
# about their specifications, so it is stored and shown but never compared.
_DAMAGE_MARK = "�"

_MAX_SPECS_PER_PRODUCT = 60
_MAX_RECORD_CHARS = 8000

_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^a-z0-9]+")
_KEY_CLEAN = re.compile(r"[^a-z0-9]+")
_UNIT_CLEAN = re.compile(r"[^a-z0-9/'\"µ.]+")


# ── Small helpers ────────────────────────────────────────────────────────


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\xa0", " ")).strip().lower()


def _norm_alnum(text: str) -> str:
    return _ALNUM.sub("", (text or "").lower())


def _canonical_key(key: str) -> str:
    return _KEY_CLEAN.sub("_", (key or "").strip().lower()).strip("_")[:64]


def _canonical_unit(unit: str) -> str:
    """Normalise the unit TOKEN. Never converts — see the module docstring."""
    return _UNIT_CLEAN.sub("", (unit or "").strip().lower())[:16]


def _written_decimals(value: float) -> int:
    text = f"{value:g}"
    return len(text.split(".")[1]) if "." in text else 0


def _half_width(value: float) -> float:
    """Half the rounding interval implied by how the value was written.

    A merchant writing 26.4 GPM in one field and 26 GPM in another has
    rounded, not contradicted themselves; one writing 9 and 8 has. Deriving
    the slack from the written precision gets both right without tuning a
    percentage against a sample.
    """
    return 0.5 * (10.0 ** -_written_decimals(value))


def build_record_text(payload: dict[str, Any]) -> str:
    """The record as the model sees it — and as quotes are verified against.

    These MUST stay the same string. An earlier version verified against
    field values only, so every quote of an attribute line was rejected as
    fabricated because the key half was missing from the haystack.
    """
    blocks: list[str] = []
    for field in _SOURCE_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            blocks.append(f"[{field}]\n{value.strip()}")

    facets = payload.get("attribute_facets")
    if isinstance(facets, list) and facets:
        lines = []
        for token in facets:
            if isinstance(token, str) and ":" in token:
                code, _, value = token.partition(":")
                if code.strip() and value.strip():
                    lines.append(f"{code.strip()}: {value.strip().replace('-', ' ')}")
        if lines:
            blocks.append("[attributes]\n" + "\n".join(lines))

    return "\n\n".join(blocks)[:_MAX_RECORD_CHARS]


def _vocabulary_block(vocabulary: Optional[dict[str, Any]]) -> str:
    """Render the store's known spec keys so the model reuses them.

    Without this, keys drift — one product yields `flow_rate`, the next
    `flowrate`, the next `flow` — and a filter on one silently misses the
    rest. Showing the extractor what already exists is what makes the
    vocabulary converge instead of fragment.
    """
    if not vocabulary:
        return "\n"
    entries = []
    for key in sorted(vocabulary)[:120]:
        meta = vocabulary.get(key) or {}
        unit = (meta.get("unit") or "").strip() if isinstance(meta, dict) else ""
        entries.append(f"  {key} ({unit})" if unit else f"  {key}")
    if not entries:
        return "\n"
    return _VOCAB_BLOCK.format(keys="\n".join(entries))


# ── Extraction ───────────────────────────────────────────────────────────


def _call_model(
    record: str,
    vocabulary: Optional[dict[str, Any]],
    *,
    api_key: str,
    provider: Optional[str],
    model: Optional[str],
) -> tuple[list[dict], int, int]:
    """One schema-forced call. Imports deferred so boot never pays for it.

    `include_raw` keeps the underlying message alongside the parsed object,
    which is the only way to read token usage off a structured-output call
    — and sync needs that to report what a full catalog pass costs.
    """
    from backend.app.magento.chatbot.agents.llm_factory import build_llm

    llm = build_llm(
        provider=provider or DEFAULT_PROVIDER,
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        temperature=0.0,
    )
    structured = llm.with_structured_output(_SCHEMA, include_raw=True)
    prompt = _PROMPT.format(vocabulary=_vocabulary_block(vocabulary), record=record)

    response = structured.invoke(prompt)

    raw_message = response.get("raw") if isinstance(response, dict) else None
    usage = getattr(raw_message, "usage_metadata", None) or {}
    tin = int(usage.get("input_tokens", 0) or 0)
    tout = int(usage.get("output_tokens", 0) or 0)

    parsed = response.get("parsed") if isinstance(response, dict) else response
    if parsed is None and raw_message is not None:
        # Schema enforcement declined for this response — fall back to
        # reading JSON out of the text rather than losing the whole product.
        text = raw_message.content
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", (text or "").strip(), flags=re.I)
        try:
            parsed = json.loads(text) if text else None
        except ValueError:
            parsed = None

    specs = parsed.get("specs") if isinstance(parsed, dict) else None
    return [s for s in (specs or []) if isinstance(s, dict)], tin, tout


def _verify(quote: str, haystack: str) -> str:
    """`exact`, `loose`, or `` — a spec survives only if its quote is real.

    `loose` means the model tidied whitespace or punctuation while quoting:
    sloppy, but not invented. Anything matching neither never existed in
    the record, and is dropped before it can reach an index or a customer.
    """
    quote = (quote or "").strip()
    if not quote:
        return ""
    if _norm_ws(quote) and _norm_ws(quote) in _norm_ws(haystack):
        return "exact"
    if _norm_alnum(quote) and _norm_alnum(quote) in _norm_alnum(haystack):
        return "loose"
    return ""


def _conflicted(specs: list[dict]) -> bool:
    """Do these values for one key genuinely disagree?

    Numbers are compared only WITHIN the same unit token — never across, so
    an unrecognised or mixed unit can cost recall but can never produce a
    wrong comparison. Text values conflict only when neither contains the
    other: "HDPE" beside "HDPE, PP" is a product made of two materials, and
    "1-3 years" beside "1-3 years old" is one fact told twice.
    """
    sound = [s for s in specs if not s.get("damaged")]

    by_unit: dict[str, list[dict]] = {}
    for spec in sound:
        if spec.get("num") is not None:
            by_unit.setdefault(spec.get("unit") or "", []).append(spec)

    for group in by_unit.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                gap = abs(float(a["num"]) - float(b["num"]))
                if gap >= _half_width(float(a["num"])) + _half_width(float(b["num"])):
                    return True

    if by_unit:
        return False

    texts = sorted({_norm_alnum(s.get("text") or "") for s in sound} - {""}, key=len)
    for i, short in enumerate(texts):
        if not any(short in longer for longer in texts[i + 1:]):
            return len(texts) > 1
    return False


def extract(
    payload: dict[str, Any],
    *,
    api_key: str,
    vocabulary: Optional[dict[str, Any]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Read every specification out of one product record.

    Returns a dict ready to merge onto the Qdrant payload:

        specs           list of {key,label,num,text,unit,quote,field,damaged}
        spec_keys       distinct keys, for the keyword index
        spec_conflicts  keys whose values genuinely disagree
        vocab           {key: {unit, label}} for the vocabulary sink
        usage           {input, output} tokens
        dropped         count discarded as unverified or placeholder

    Never raises: a product that fails extraction still syncs, just without
    specs. It remains findable semantically — it simply cannot be filtered
    on until the next sync picks it up.
    """
    empty = {"specs": [], "spec_keys": [], "spec_conflicts": [],
             "vocab": {}, "usage": {"input": 0, "output": 0}, "dropped": 0}

    record = build_record_text(payload)
    if not record.strip():
        return empty
    # A record with no numerals cannot hold a numeric spec. Text-only specs
    # (material, thread type) are worth having too, but not worth a model
    # call on a record that is pure marketing copy.
    if not any(ch.isdigit() for ch in record):
        return empty

    try:
        raw_specs, tin, tout = _call_model(
            record, vocabulary, api_key=api_key, provider=provider, model=model
        )
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("spec extraction failed for sku=%s: %s",
                       payload.get("sku") or payload.get("entity_id"), exc)
        return empty

    specs: list[dict[str, Any]] = []
    dropped = 0

    for raw in raw_specs[:_MAX_SPECS_PER_PRODUCT]:
        key = _canonical_key(str(raw.get("key") or ""))
        if not key:
            dropped += 1
            continue

        num = raw.get("value_num")
        num = float(num) if isinstance(num, (int, float)) and not isinstance(num, bool) else None
        text = str(raw.get("value_text")).strip() if raw.get("value_text") else None
        quote = str(raw.get("quote") or "")

        # Placeholders are pointers, not values.
        if num is None and _norm_alnum(text or "") in _PLACEHOLDERS:
            dropped += 1
            continue
        # Nothing physical carries a value this large; a parse went wrong.
        if num is not None and not (0 <= abs(num) < 1e7):
            dropped += 1
            continue

        verified = _verify(quote, record)
        if not verified:
            dropped += 1
            continue

        specs.append({
            "key": key,
            "label": str(raw.get("label") or "").strip()[:64],
            "num": num,
            "text": text[:120] if text else None,
            "unit": _canonical_unit(str(raw.get("unit") or "")),
            "quote": quote[:200],
            "field": str(raw.get("field") or "").strip().strip("[]")[:32],
            "damaged": _DAMAGE_MARK in quote or _DAMAGE_MARK in (text or ""),
        })

    grouped: dict[str, list[dict]] = {}
    for spec in specs:
        grouped.setdefault(spec["key"], []).append(spec)

    conflicts = sorted(k for k, group in grouped.items() if _conflicted(group))

    vocab: dict[str, dict[str, str]] = {}
    for key, group in grouped.items():
        units = [s["unit"] for s in group if s["unit"]]
        labels = [s["label"] for s in group if s["label"]]
        vocab[key] = {
            "unit": max(set(units), key=units.count) if units else "",
            "label": labels[0] if labels else key.replace("_", " ").title(),
        }

    return {
        "specs": specs,
        "spec_keys": sorted(grouped),
        "spec_conflicts": conflicts,
        "vocab": vocab,
        "usage": {"input": tin, "output": tout},
        "dropped": dropped,
    }
