"""
BM25 sparse-vector generation for hybrid search (Phase 2.2).

Pairs with the existing dense `embedder.py` (Gemini-backed). Where the
dense embedder turns text into a 3072-dim continuous vector capturing
semantic similarity, this module turns the same text into a sparse
BM25 vector capturing exact-token signal — the thing semantic search
keeps missing for SKU lookups, exact product names, brand searches,
and short policy keywords.

Both vectors live on the SAME Qdrant point under different named slots
(`dense` and `sparse_bm25`). At query time Qdrant's Query API prefetches
candidates from each and fuses them via Reciprocal Rank Fusion, which
is implemented server-side — no manual RRF in Python.

Cost / latency:
    * Model: Qdrant/bm25 (default fastembed BM25). ~30 MB ONNX file,
      downloaded once on first import (cached under ~/.cache/fastembed
      by default; overridable via FASTEMBED_CACHE_DIR env var).
    * CPU-only, no GPU. Typical inference 1-5 ms per text after warmup.
    * Zero per-call $ cost — runs entirely in-process.

Singleton pattern: a single SparseTextEmbedding instance is loaded
lazily on first use and reused for the process lifetime. fastembed's
embed() returns an iterator of sparse-vector objects; we always pass a
single-text list and pull the first result.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)

# Model name — Qdrant's reference BM25. Fastembed exposes others (e.g.
# SPLADE++) but BM25 is the right default for general-purpose product +
# CMS corpora: zero training, fast, well-understood IDF behaviour.
_BM25_MODEL_NAME = "Qdrant/bm25"

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy singleton — first call loads the ONNX model, subsequent
    calls reuse it. Threaded gate prevents two parallel first-callers
    from both downloading.

    Importing fastembed at module import time would cost ~30 MB of
    download on backend startup even for clients who never enable
    hybrid search. Lazy load defers that to first query.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        # Import inside the lock so cold-start cost shows up exactly
        # once in logs and we don't fan out parallel ONNX init.
        from fastembed import SparseTextEmbedding

        logger.info("[sparse_embedder] loading %s (one-time download on first run) …", _BM25_MODEL_NAME)
        _model = SparseTextEmbedding(model_name=_BM25_MODEL_NAME)
        logger.info("[sparse_embedder] model ready")
        return _model


def embed_sparse_document(text: str) -> SparseVector:
    """Generate a BM25 sparse vector for a *document* (corpus side).

    BM25 IDF is computed against the model's pre-built corpus
    statistics — fastembed's Qdrant/bm25 ships with statistics tuned
    for general English. For domain-specific tuning (e.g. fashion
    vocabulary, technical SKUs) a tenant-trained sparse model could
    swap in here; deferred to Phase 4 (operational maturity) when we
    have signal that BM25 quality is the bottleneck.

    Returns a qdrant_client.models.SparseVector ready to attach to a
    PointStruct's vector dict under the "sparse_bm25" slot.
    """
    model = _get_model()
    # embed() returns an iterator of fastembed SparseEmbedding objects.
    # We pass a 1-element list, take the first result, and convert into
    # the qdrant_client SparseVector shape Qdrant expects on the wire.
    fe_sparse = next(iter(model.embed([text or ""])))
    return _to_qdrant_sparse(fe_sparse)


def embed_sparse_query(text: str) -> SparseVector:
    """Generate a BM25 sparse vector for a *query* (read side).

    fastembed's BM25 uses the same model for queries and documents.
    Kept as a separate function for API symmetry with embedder.py
    (embed_query / embed_document) and so a future swap to a
    query-tuned sparse model (e.g. SPLADE-Q) only touches the query
    path.
    """
    model = _get_model()
    # query_embed() exists on newer fastembed BM25 wrappers; fall back
    # to embed() if not present (older fastembed versions). Both return
    # the same shape for BM25.
    method = getattr(model, "query_embed", None) or model.embed
    fe_sparse = next(iter(method([text or ""])))
    return _to_qdrant_sparse(fe_sparse)


def _to_qdrant_sparse(fe_sparse) -> SparseVector:
    """Convert fastembed's SparseEmbedding into qdrant_client.SparseVector.

    fastembed returns numpy arrays for `.indices` and `.values`; Qdrant
    needs plain Python lists on the wire. We do the conversion here so
    callers stay ignorant of the internal format.
    """
    indices = fe_sparse.indices
    values = fe_sparse.values
    # numpy → list. Defensive against fastembed already returning lists
    # (older versions did).
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return SparseVector(indices=list(indices), values=list(values))


def is_available() -> bool:
    """Cheap probe used by callers that want to gracefully skip sparse
    generation if fastembed isn't installed (e.g. running tests without
    the heavy ML dep). Returns True only when the model can be loaded.
    """
    try:
        _get_model()
        return True
    except Exception as exc:
        logger.warning("[sparse_embedder] unavailable: %s", exc)
        return False
