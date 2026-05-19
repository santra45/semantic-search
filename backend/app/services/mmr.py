"""
Maximal Marginal Relevance (MMR) diversification (Phase 2.3).

Sits BETWEEN Qdrant retrieval (dense / hybrid) and the LLM reranker (2.1)
in the retrieve pipeline. Purpose: when the top-K candidates are
near-duplicates (e.g. 8 black hoodies that differ only by size), the
reranker has no breadth to work with and the customer sees a wall of
visually identical results. MMR re-orders the candidate pool to balance
relevance to the query with dissimilarity to already-picked items, so
the rerank sees a *diverse* set of relevant items.

Algorithm (Carbonell & Goldstein, 1998):

    MMR = argmax_i [ λ · sim(q, dᵢ)  −  (1 − λ) · max_{dⱼ ∈ S} sim(dᵢ, dⱼ) ]

where:
    q          — query vector
    dᵢ         — candidate document vector
    S          — already-selected set
    sim(.,.)   — cosine similarity
    λ ∈ [0,1]  — tradeoff knob (1 = pure relevance, 0 = pure diversity)

Typical practical values for `λ`:
    0.7-0.8 — light diversification, mostly relevance-driven
    0.5     — balanced (admin-default)
    0.3-0.4 — aggressive diversification, may sacrifice relevance

Complexity: O(N·K) where N = candidate count, K = output count. For
typical N=30, K=10, total dot-products ≈ 30·10 = 300 — trivial. Uses
numpy for the per-pair similarity to keep cold-path overhead low.

This module is intentionally Qdrant-agnostic — it takes a list of dicts
with a `_dense_vector` key and returns a re-ordered subset. That lets
us unit-test it without a Qdrant fixture and lets the retrieve handler
decide whether/when to apply it.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Key used by qdrant_service._format_hit to stash the dense vector on
# each hit when `with_vectors=True` is passed at search time. Keeping it
# `_`-prefixed signals "internal — strip before returning to the wire"
# (retrieve.py does the strip).
VECTOR_KEY = "_dense_vector"


def apply_mmr(
    query_vector: List[float],
    candidates: List[dict[str, Any]],
    lambda_val: float = 0.5,
    k: int = 10,
    vector_key: str = VECTOR_KEY,
) -> List[dict[str, Any]]:
    """Re-order *candidates* into a relevance/diversity-balanced top-K.

    Always returns at most `k` items in the order MMR picked them
    (most-MMR-optimal first). Candidates missing a vector are still
    considered but with zero relevance + zero diversity contribution —
    they end up near the bottom of the MMR ordering, never disappearing
    silently.

    No-op short-circuits:
      * empty input        → []
      * k ≤ 0              → []
      * λ ≥ 0.999          → pure relevance, equivalent to top-K by
                              query similarity. Skips the pairwise pass.
      * len(candidates) ≤ k → return as-is (nothing to diversify).
    """
    if not candidates or k <= 0:
        return []

    # Clamp lambda to the legal range; admin can put anything in the
    # config field and we'd rather degrade gracefully than 500 a
    # customer query because of a typo.
    lambda_val = max(0.0, min(1.0, float(lambda_val)))

    if len(candidates) <= k:
        # Not enough candidates to diversify; return what we have.
        # Caller may still want them in input order (which is already
        # relevance-sorted by Qdrant).
        return list(candidates)

    # Stage 1: compute the relevance vector q⋅dᵢ for every candidate.
    # Done once up front — used by every iteration of the selection loop.
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = float(np.linalg.norm(q)) or 1.0  # avoid div-by-zero

    doc_vecs: list[np.ndarray | None] = []
    rel_scores: list[float] = []
    for c in candidates:
        raw = c.get(vector_key)
        if not raw:
            doc_vecs.append(None)
            rel_scores.append(0.0)
            continue
        v = np.asarray(raw, dtype=np.float32)
        v_norm = float(np.linalg.norm(v)) or 1.0
        doc_vecs.append(v)
        rel_scores.append(float(np.dot(q, v) / (q_norm * v_norm)))

    # Fast path: pure relevance (λ ≈ 1) is just sorted-by-similarity.
    # Skip the O(NK) loop entirely.
    if lambda_val >= 0.999:
        order = sorted(range(len(candidates)), key=lambda i: rel_scores[i], reverse=True)
        return [candidates[i] for i in order[:k]]

    # Stage 2: greedy MMR selection.
    # `max_sim[i]` tracks the maximum cosine of candidate i against any
    # already-selected document. Re-using it across iterations is the
    # standard MMR optimisation — each iteration only needs to update
    # max_sim with similarities to the *newest* selection.
    max_sim = [0.0] * len(candidates)
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while len(selected) < k and remaining:
        best_i = -1
        best_score = -float("inf")
        for i in remaining:
            score = lambda_val * rel_scores[i] - (1.0 - lambda_val) * max_sim[i]
            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0:
            break

        selected.append(best_i)
        remaining.discard(best_i)

        # Update each remaining candidate's max-similarity-to-selected
        # using the freshly-picked one. Skip when either side has no
        # vector — those contribute zero diversity penalty.
        new_v = doc_vecs[best_i]
        if new_v is None:
            continue
        new_norm = float(np.linalg.norm(new_v)) or 1.0
        for j in remaining:
            jv = doc_vecs[j]
            if jv is None:
                continue
            jv_norm = float(np.linalg.norm(jv)) or 1.0
            sim = float(np.dot(jv, new_v) / (jv_norm * new_norm))
            if sim > max_sim[j]:
                max_sim[j] = sim

    return [candidates[i] for i in selected]


def strip_vector(hits: list[dict[str, Any]], vector_key: str = VECTOR_KEY) -> list[dict[str, Any]]:
    """Remove the `_dense_vector` field from each hit before returning
    to the wire. 3072-dim float lists × N hits would balloon the
    response body for no consumer benefit (Magento agents never read
    raw vectors). Mutates in place AND returns the list for chaining.
    """
    for h in hits:
        h.pop(vector_key, None)
    return hits
