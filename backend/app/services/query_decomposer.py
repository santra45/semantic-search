"""
Query decomposition (Phase 3.3).

Runs as a preprocessor inside `/retrieve/products` and `/retrieve/content`
when the merchant has the admin toggle on. Splits a single compositional
customer query into 1-3 semantic sub-queries; the caller then embeds each
and asks Qdrant to fuse the resulting candidate sets via RRF (the same
fusion primitive 2.2 already uses for hybrid BM25 + dense).

Why this exists:
    For "cheapest stainless steel water feature under £50 for a small
    garden", 3.1's tool-call has already extracted the structured filters
    (max_price=50, sort_by=price asc). What remains is the *semantic*
    string "stainless steel water feature for small garden" — which
    contains three distinct concepts that, blended into one embedding,
    typically rank below products that match the dominant concept and
    have weak signal on the others. Splitting into sub-queries and
    fusing the per-concept top-Ks surfaces the products that match
    multiple concepts decently rather than the ones that match one
    concept dominantly.

When NOT to decompose:
    * Admin toggle off (default).
    * Simple short queries — gated by the heuristic below. A 4-word
      "show me red shoes" query has nothing to decompose; firing an
      LLM call would waste tokens.

Architecture position:
    * BELOW 3.1 (tool-call) — operates on the cleaned-up query string
      the classifier extracted, doesn't compete with the args
      extractor.
    * ABOVE 2.3 (MMR) — produces a wider candidate pool that MMR then
      diversifies.
    * Independent of 3.2 (active retrieval) — 3.3 fires once per
      initial retrieval; 3.2 fires inside the answer LLM if initial
      sources are weak.

Cost / latency:
    * One small JSON-mode LLM call when the heuristic fires
      (~$0.0001-0.0003 typical, 200-500ms latency).
    * Tracked under `chat_query_decompose` so admin dashboard sees it
      separately from the answer LLM call (`chat_answer`) and the
      classifier (`chat_intent` / `chat_tool_call`).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from backend.app.services.llm_completion_service import complete

logger = logging.getLogger(__name__)

# Hard ceiling on the number of sub-queries the decomposer can return.
# Mirrors the value the LLM is instructed to honour in the prompt; we
# truncate post-parse defensively in case the LLM ignores the cap.
# Three is the standard sweet spot in RAG literature — enough breadth
# for a 3-concept query, not so many that the fused candidate pool
# becomes noisy.
MAX_SUB_QUERIES = 3

# Heuristic gate parameters. Surface for tuning without changing the
# function signature — admin doesn't need a knob for these (the toggle
# is binary; the gate just decides which queries are worth paying for).
_MIN_WORDS_FOR_DECOMPOSITION = 6
_MIN_CONJUNCTIONS = 2

# Conjunction signals that suggest a compositional query. Case-insensitive
# whole-word match — "for" inside "force" or "and" inside "android" don't
# trigger. Lowercased lookup set built once at import.
_CONJUNCTION_SIGNALS = frozenset({
    "and", "or", "with", "for", "under", "above",
    "between", "vs", "versus", "plus",
})

# Regex extracts contiguous alphanumeric runs (treats £/$ symbols as
# separators); plenty good enough for word-counting purposes.
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def maybe_decompose(
    query: str,
    *,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    client_id: str = "anonymous",
) -> list[str]:
    """Return a list of 1-MAX_SUB_QUERIES sub-queries for *query*.

    Always returns a non-empty list. Single-element list when:
      * the heuristic gate decides the query isn't compositional, OR
      * the LLM call fails / returns garbage (soft-fall to original).

    Caller pattern:
        sub_queries = maybe_decompose(req.query, ...)
        if len(sub_queries) == 1:
            # Single-vector retrieve path — same as before 3.3
            vector = embed_query(sub_queries[0])
            hits = search_products(query_vector=vector, ...)
        else:
            # Multi-vector path — qdrant fuses via Fusion.RRF
            vectors = [embed_query(q) for q in sub_queries]
            hits = search_products(query_vectors=vectors, ...)
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return [""]

    if not _is_compositional(cleaned):
        return [cleaned]

    sub_queries = _decompose_via_llm(
        cleaned,
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        client_id=client_id,
    )
    if not sub_queries:
        # LLM call failed or returned nothing usable. Fall back to the
        # original query so retrieval still happens — we'd rather take
        # the un-decomposed search than fail the customer turn.
        return [cleaned]

    # Truncate defensively + ensure non-empty.
    out = [q.strip() for q in sub_queries if q and q.strip()][:MAX_SUB_QUERIES]
    return out or [cleaned]


def _is_compositional(query: str) -> bool:
    """Heuristic gate. Two independent signals — either fires:

      1. Word count > _MIN_WORDS_FOR_DECOMPOSITION. Long queries are
         usually compositional ("cheapest stainless steel water feature
         under £50 for a small garden" — 11 words).
      2. >= _MIN_CONJUNCTIONS conjunction-signal tokens. Catches short
         but explicitly-compound queries ("red and blue and green
         shoes" — 6 words, 3 conjunctions).

    Conservative on purpose — false negatives (missing a compositional
    query) just degrade to single-vector retrieval, which is the
    pre-3.3 baseline. False positives waste an LLM call on a query
    that wouldn't have benefited.
    """
    words = _WORD_PATTERN.findall(query.lower())
    if len(words) > _MIN_WORDS_FOR_DECOMPOSITION:
        return True
    conjunctions = sum(1 for w in words if w in _CONJUNCTION_SIGNALS)
    return conjunctions >= _MIN_CONJUNCTIONS


# Prompt kept tight — the LLM only needs to do one thing, and a longer
# prompt would (a) cost more on every decomposition and (b) give the
# LLM more chances to follow a tangent. Reusing the JSON-mode flag on
# llm_completion_service.complete() means provider-side schema
# validation handles malformed responses — Gemini/OpenAI/Anthropic all
# enforce json shape when the flag is set.
_DECOMPOSE_PROMPT_TEMPLATE = """\
You are a query splitter for an e-commerce search engine.

Split the user's search query into 1 to {max_sub_queries} concise
semantic sub-queries. Each sub-query should capture ONE distinct
concept the user mentioned, written as a short noun phrase suitable
for vector search.

Rules:
  * Keep each sub-query under 8 words.
  * Do NOT include price filters, sort directives, or quantity numbers
    in the sub-queries — those are handled separately by the search
    engine. Examples to strip: "under £50", "cheapest", "newest",
    "less than 30".
  * If the query is genuinely about ONE concept, return one sub-query
    only (paraphrased / cleaned if useful, otherwise verbatim).
  * Never return zero sub-queries.

Return ONLY a JSON array of strings, nothing else.

User query: {query}
"""


def _decompose_via_llm(
    query: str,
    *,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    api_key: Optional[str],
    client_id: str,
) -> list[str]:
    """Single LLM call. Returns a list of sub-queries on success or an
    empty list on any failure path (network blip, malformed JSON,
    missing key). Caller treats empty as "fall back to original".
    """
    if not api_key:
        # No LLM API key configured — same posture as LLMClassifier
        # when isAvailable() returns false. Skip silently.
        return []

    prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(
        max_sub_queries=MAX_SUB_QUERIES,
        query=query.replace("\n", " "),
    )

    try:
        text, _usage = complete(
            prompt,
            json_mode=True,
            # Short response cap — three short noun phrases plus JSON
            # structure fits in 150 tokens easily; capping here prevents
            # a misbehaving model from generating an essay.
            max_tokens=200,
            temperature=0.0,
            provider=(llm_provider or "google"),
            model=llm_model,
            api_key=api_key,
            client_id=client_id,
            query_type="chat_query_decompose",
        )
    except ValueError as ve:
        logger.debug("[query_decomposer] config error, skipping decomposition: %s", ve)
        return []
    except Exception as exc:
        logger.warning("[query_decomposer] LLM call failed: %s", exc)
        return []

    return _parse_response(text)


def _parse_response(text: str) -> list[str]:
    """Tolerant JSON-array parser. Accepts a bare array, an array wrapped
    in markdown code fences, or an object with a `sub_queries` /
    `queries` key (some models like to add structure even when asked
    for a bare array).
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # Strip markdown fences when the model wraps despite json_mode being
    # on — empirically rare with native JSON-mode providers but cheap to
    # handle defensively.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Last-ditch: find the first [..] block. JSON-mode usually
        # prevents this branch from firing.
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        for key in ("sub_queries", "queries", "results"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            return []

    if not isinstance(parsed, list):
        return []

    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            # Some models nest each sub-query under a "query" / "text" key.
            for key in ("query", "text", "value"):
                if isinstance(item.get(key), str):
                    out.append(item[key].strip())
                    break
    return [q for q in out if q]
