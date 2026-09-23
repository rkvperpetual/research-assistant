"""
app/retrieval/bm25.py — In-memory BM25 index for keyword retrieval.

Rebuilt from Qdrant on startup so it stays in sync after restarts.
Thread-safety: a simple lock guards mutations; reads are safe after init.
"""
from __future__ import annotations

import logging
import threading
from typing import List

log = logging.getLogger(__name__)

_lock = threading.Lock()


class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Stores corpus text alongside so we can return Document-like results.
    """

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._index = None  # BM25Okapi instance, built lazily

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            if self._corpus:
                tokenized = [self._tokenize(t) for t in self._corpus]
                self._index = BM25Okapi(tokenized)
            else:
                self._index = None
        except ImportError:
            log.warning("rank_bm25 not installed; BM25 retrieval disabled")

    def add_documents(self, texts: List[str]) -> None:
        with _lock:
            self._corpus.extend(texts)
            self._rebuild()

    def search(self, query: str, top_k: int = 8) -> List[tuple[str, float]]:
        """Return list of (text, score) tuples sorted by BM25 score desc."""
        with _lock:
            if self._index is None or not self._corpus:
                return []
            tokens = self._tokenize(query)
            scores = self._index.get_scores(tokens)
            ranked = sorted(
                zip(self._corpus, scores), key=lambda x: x[1], reverse=True
            )
            return [(text, float(score)) for text, score in ranked[:top_k] if score > 0]

    def corpus_size(self) -> int:
        return len(self._corpus)


_singleton: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _singleton
    if _singleton is None:
        _singleton = BM25Index()
    return _singleton


def bootstrap_bm25_from_qdrant() -> None:
    """
    Scroll all chunks from Qdrant and build the BM25 index.
    Called once during app startup lifespan.
    """
    try:
        from app.retrieval.vectorstore import get_qdrant_client
        from app.config import get_settings

        cfg = get_settings()
        client = get_qdrant_client()

        records, _ = client.scroll(
            collection_name=cfg.qdrant_collection,
            with_payload=True,
            with_vectors=False,
            limit=50_000,
        )

        texts: list[str] = []
        for rec in records:
            payload = rec.payload or {}
            # LangChain stores page_content under "page_content" key in Qdrant
            text = payload.get("page_content", "")
            if text:
                texts.append(text)

        if texts:
            idx = get_bm25_index()
            idx.add_documents(texts)
            log.info("BM25 index bootstrapped with %d documents", len(texts))
        else:
            log.info("No documents in Qdrant yet; BM25 index will be empty")

    except Exception as e:
        log.warning("BM25 bootstrap failed (non-fatal): %s", e)
