"""
app/retrieval/vectorstore.py — Qdrant client, collection bootstrap, and upsert.

Embedding via OpenRouter → nvidia/nv-embedqa-e5-v5 (1024-dim).
Batches of 64 chunks per embed call to stay within API limits.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.core.llm import get_embeddings

log = logging.getLogger(__name__)
_cfg = get_settings()

_client: QdrantClient | None = None
_vectorstore: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=_cfg.qdrant_url,
            api_key=_cfg.qdrant_api_key or None,
            timeout=60,
        )
    return _client


def bootstrap_collection() -> None:
    """
    Ensure the Qdrant collection exists with the correct vector dimension.
    If the collection exists but has a different dimension (e.g. after switching
    embedding models), it is deleted and recreated.
    """
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if _cfg.qdrant_collection in existing:
        # Check if dimensions match
        info = client.get_collection(_cfg.qdrant_collection)
        # vectors_config can be a dict or a VectorParams object
        current_dim = None
        vc = info.config.params.vectors
        if hasattr(vc, "size"):
            current_dim = vc.size
        elif isinstance(vc, dict) and "" in vc:
            current_dim = vc[""].size

        if current_dim and current_dim != _cfg.embedding_dim:
            log.warning(
                "Collection '%s' has dim=%d but config expects dim=%d — recreating",
                _cfg.qdrant_collection, current_dim, _cfg.embedding_dim,
            )
            client.delete_collection(_cfg.qdrant_collection)
        else:
            log.info("Qdrant collection '%s' already exists (dim=%s)", _cfg.qdrant_collection, current_dim)
            return

    log.info("Creating Qdrant collection '%s' (dim=%d)", _cfg.qdrant_collection, _cfg.embedding_dim)
    client.create_collection(
        collection_name=_cfg.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=_cfg.embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=_cfg.qdrant_collection,
            embedding=get_embeddings(),
        )
    return _vectorstore


def upsert_chunks(
    chunks: List[Document],
    batch_size: int = 64,
    on_batch_complete: Callable[[int], None] | None = None,
) -> None:
    """
    Embed and upsert chunks to Qdrant in batches.
    Assigns a UUID point_id to each chunk.
    """
    vs = get_vectorstore()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vs.add_documents(batch)
        if on_batch_complete:
            on_batch_complete(len(batch))
        log.debug("Upserted batch %d–%d", i, i + len(batch))


def scroll_all_documents() -> list[dict]:
    """
    Return a summary list of all unique doc_ids with their filename and chunk count.
    """
    client = get_qdrant_client()
    records, _ = client.scroll(
        collection_name=_cfg.qdrant_collection,
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )

    docs: dict[str, dict] = {}
    for rec in records:
        payload = rec.payload or {}
        # LangChain QdrantVectorStore stores metadata under "metadata" key
        meta = payload.get("metadata", payload)
        doc_id = meta.get("doc_id", "unknown")
        filename = meta.get("filename", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "source_type": meta.get("source_type", "unknown"),
                "chunk_count": 0,
                "ingested_at": meta.get("ingested_at", ""),
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


def delete_document(doc_id: str) -> int:
    """Delete all points for a doc_id. Returns number of points deleted."""
    client = get_qdrant_client()
    # Scroll to count first
    records, _ = client.scroll(
        collection_name=_cfg.qdrant_collection,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.doc_id", match=qmodels.MatchValue(value=doc_id))]
        ),
        with_payload=False,
        with_vectors=False,
        limit=10_000,
    )
    ids = [r.id for r in records]
    if ids:
        client.delete(
            collection_name=_cfg.qdrant_collection,
            points_selector=qmodels.PointIdsList(points=ids),
        )
    return len(ids)
