"""
app/ingestion/pipeline.py — End-to-end ingestion orchestrator.

parse → chunk → embed → upsert (Qdrant)
Also updates the global BM25 index.

This module is called from the FastAPI BackgroundTask so it runs
asynchronously and the HTTP request returns immediately.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.detect import detect_file_type
from app.ingestion.parsers import parse_file, parse_url
from app.retrieval.vectorstore import get_vectorstore, upsert_chunks
from app.retrieval.bm25 import get_bm25_index

log = logging.getLogger(__name__)
_cfg = get_settings()

# ─── In-memory job tracker ────────────────────────────────────────────────────
# {job_id: {"status": "processing|done|failed", "filename": ..., "chunks": int, "error": ...}}
_jobs: Dict[str, dict] = {}

# ─── Seen SHA-256s (loaded from Qdrant on startup) ───────────────────────────
_seen_hashes: set[str] = set()


def register_seen_hash(sha: str) -> None:
    _seen_hashes.add(sha)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def register_job(job_id: str, filename: str) -> None:
    """Make a queued ingestion job visible to status polling immediately."""
    _jobs[job_id] = {
        "status": "processing",
        "filename": filename,
        "chunks": 0,
        "chunks_indexed": 0,
        "progress": 0,
        "stage": "queued",
        "error": None,
        "started_at": time.time(),
    }


def list_jobs() -> list[dict]:
    return [{"job_id": k, **v} for k, v in _jobs.items()]


def _update_job(job_id: str, **values: object) -> None:
    """Update visible ingestion progress without replacing existing job data."""
    _jobs[job_id].update(values)


# ─── Core pipeline ────────────────────────────────────────────────────────────

def ingest_file(path: Path, job_id: str) -> None:
    """
    Full ingestion pipeline for a single file.
    Updates _jobs[job_id] with status and chunk count.
    """
    if job_id not in _jobs:
        register_job(job_id, path.name)
    try:
        _update_job(job_id, stage="checking duplicate", progress=5)
        sha = _sha256(path)
        if sha in _seen_hashes:
            log.info("Duplicate file skipped: %s", path.name)
            _update_job(job_id, status="duplicate", stage="already indexed", progress=100)
            return

        _update_job(job_id, stage="detecting file type", progress=10)
        file_type = detect_file_type(path)
        log.info("Detected %r → %s", path.name, file_type)

        _update_job(job_id, stage="extracting text", progress=25)
        raw_docs: List[Document] = parse_file(path, file_type)
        if not raw_docs:
            raise ValueError(f"No content extracted from {path.name}")

        doc_id = str(uuid.uuid4())

        # Attach sha + ingested_at to all raw docs
        ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for d in raw_docs:
            d.metadata.update({"sha256": sha, "ingested_at": ingested_at, "doc_id": doc_id})

        _update_job(job_id, stage="splitting into chunks", progress=45)
        chunks = chunk_documents(raw_docs, doc_id)
        _update_job(job_id, stage="indexing embeddings", progress=55, chunks=len(chunks))

        # Embed + upsert in batches of 64
        def report_indexed(count: int) -> None:
            indexed = _jobs[job_id].get("chunks_indexed", 0) + count
            total = len(chunks)
            progress = 55 + int(40 * indexed / total) if total else 95
            _update_job(job_id, chunks_indexed=indexed, progress=progress)

        upsert_chunks(chunks, on_batch_complete=report_indexed)

        # Update BM25 index
        bm25 = get_bm25_index()
        bm25.add_documents([c.page_content for c in chunks])

        _seen_hashes.add(sha)
        _jobs[job_id].update(
            {
                "status": "done",
                "chunks": len(chunks),
                "chunks_indexed": len(chunks),
                "progress": 100,
                "stage": "complete",
                "doc_id": doc_id,
                "file_type": file_type,
                "finished_at": time.time(),
            }
        )
        log.info("Ingested %s → %d chunks (doc_id=%s)", path.name, len(chunks), doc_id)

    except Exception as e:
        log.exception("Ingestion failed for %s", path.name)
        _update_job(job_id, status="failed", stage="failed", error=str(e))


def ingest_url(url: str, job_id: str) -> None:
    """Ingest a web page URL."""
    if job_id not in _jobs:
        register_job(job_id, url)
    try:
        _update_job(job_id, stage="fetching web page", progress=15)
        raw_docs = parse_url(url)
        if not raw_docs:
            raise ValueError(f"No content extracted from {url}")

        doc_id = str(uuid.uuid4())
        ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for d in raw_docs:
            d.metadata.update({"ingested_at": ingested_at, "doc_id": doc_id})

        _update_job(job_id, stage="splitting into chunks", progress=45)
        chunks = chunk_documents(raw_docs, doc_id)
        _update_job(job_id, stage="indexing embeddings", progress=55, chunks=len(chunks))

        def report_indexed(count: int) -> None:
            indexed = _jobs[job_id].get("chunks_indexed", 0) + count
            total = len(chunks)
            progress = 55 + int(40 * indexed / total) if total else 95
            _update_job(job_id, chunks_indexed=indexed, progress=progress)

        upsert_chunks(chunks, on_batch_complete=report_indexed)

        bm25 = get_bm25_index()
        bm25.add_documents([c.page_content for c in chunks])

        _jobs[job_id].update(
            {
                "status": "done",
                "chunks": len(chunks),
                "chunks_indexed": len(chunks),
                "progress": 100,
                "stage": "complete",
                "doc_id": doc_id,
                "file_type": "html_url",
                "finished_at": time.time(),
            }
        )
        log.info("Ingested URL %s → %d chunks", url, len(chunks))
    except Exception as e:
        log.exception("URL ingestion failed: %s", url)
        _update_job(job_id, status="failed", stage="failed", error=str(e))
