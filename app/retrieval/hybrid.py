"""
app/retrieval/hybrid.py — Hybrid retrieval: dense + BM25 fused with RRF.

Reciprocal Rank Fusion (RRF) with k=60 is used to merge dense and sparse
ranked lists. Top-K=6 chunks are returned after fusion.

Optional cross-encoder reranking is compiled out by default to save ~400 ms
and avoid downloading the model; enable via ENABLE_RERANKER=true env var.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.bm25 import get_bm25_index
from app.retrieval.vectorstore import get_vectorstore

log = logging.getLogger(__name__)
_cfg = get_settings()

_RRF_K = 60


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def _rrf_fuse(
    dense_docs: List[Document],
    sparse_hits: List[tuple[str, float]],
    top_k: int,
) -> List[Document]:
    """
    Merge two ranked lists into one using Reciprocal Rank Fusion.
    Returns up to top_k Document objects.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Dense ranking contribution
    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:200]  # stable key
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        doc_map[key] = doc

    # Sparse (BM25) ranking contribution
    for rank, (text, _bm25_score) in enumerate(sparse_hits):
        key = text[:200]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        if key not in doc_map:
            doc_map[key] = Document(page_content=text, metadata={"source": "bm25"})

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked[:top_k]]


def hybrid_retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """
    Perform hybrid retrieval for a single query.
    1. Dense: Qdrant top-8
    2. Sparse: BM25 top-8
    3. Fuse with RRF, return top-K (default from config)
    """
    k = top_k or _cfg.retrieval_top_k
    dense_k = 8
    sparse_k = 8

    # ── Dense retrieval ───────────────────────────────────────────────────────
    try:
        vs = get_vectorstore()
        dense_docs: List[Document] = vs.similarity_search(query, k=dense_k)
    except Exception as e:
        log.warning("Dense retrieval failed: %s", e)
        dense_docs = []

    # ── Sparse retrieval ──────────────────────────────────────────────────────
    try:
        bm25 = get_bm25_index()
        sparse_hits = bm25.search(query, top_k=sparse_k)
    except Exception as e:
        log.warning("BM25 retrieval failed: %s", e)
        sparse_hits = []

    # ── Fuse ──────────────────────────────────────────────────────────────────
    fused = _rrf_fuse(dense_docs, sparse_hits, top_k=k)
    log.debug(
        "Hybrid retrieve: dense=%d, sparse=%d, fused=%d (k=%d)",
        len(dense_docs),
        len(sparse_hits),
        len(fused),
        k,
    )
    return fused
