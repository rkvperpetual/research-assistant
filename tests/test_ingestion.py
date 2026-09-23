"""
tests/test_ingestion.py — Unit tests for the ingestion pipeline.

Tests run without a live Qdrant instance by mocking the upsert step.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_temp(content: bytes, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)


# ─── detect.py ───────────────────────────────────────────────────────────────

class TestDetect:
    def test_pdf_by_magic(self):
        from app.ingestion.detect import detect_file_type
        p = _write_temp(b"%PDF-1.4 fake pdf content", ".pdf")
        assert detect_file_type(p) == "pdf"

    def test_png_by_magic(self):
        from app.ingestion.detect import detect_file_type
        p = _write_temp(b"\x89PNG\r\n\x1a\n fake", ".png")
        assert detect_file_type(p) == "png"

    def test_txt_by_extension(self):
        from app.ingestion.detect import detect_file_type
        p = _write_temp(b"Hello world", ".txt")
        assert detect_file_type(p) == "txt"

    def test_md_by_extension(self):
        from app.ingestion.detect import detect_file_type
        p = _write_temp(b"# Heading", ".md")
        assert detect_file_type(p) == "md"

    def test_html_by_magic(self):
        from app.ingestion.detect import detect_file_type
        p = _write_temp(b"<!DOCTYPE html><html></html>", ".html")
        assert detect_file_type(p) == "html"


# ─── parsers.py ──────────────────────────────────────────────────────────────

class TestParsers:
    def test_parse_txt(self):
        from app.ingestion.parsers import parse_text
        p = _write_temp(b"This is a test document.\nWith multiple lines.", ".txt")
        docs = parse_text(p)
        assert docs, "Expected at least one Document"
        assert "test document" in docs[0].page_content

    def test_parse_md(self):
        from app.ingestion.parsers import parse_text
        p = _write_temp(b"# Title\n\nSome content here.", ".md")
        docs = parse_text(p)
        assert docs
        assert "Title" in docs[0].page_content

    def test_parse_html(self):
        from app.ingestion.parsers import parse_html
        html = b"<html><body><nav>skip</nav><p>Main content here</p></body></html>"
        p = _write_temp(html, ".html")
        docs = parse_html(p)
        assert docs
        assert "Main content" in docs[0].page_content
        # Nav should be stripped
        assert "skip" not in docs[0].page_content

    def test_metadata_shape(self):
        from app.ingestion.parsers import parse_text
        p = _write_temp(b"Content", ".txt")
        docs = parse_text(p)
        meta = docs[0].metadata
        assert "filename" in meta
        assert "source_type" in meta


# ─── chunker.py ──────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_size_respected(self):
        from app.ingestion.chunker import chunk_documents
        from langchain_core.documents import Document

        # Create a doc larger than chunk_size
        text = "word " * 500  # ~2500 chars
        docs = [Document(page_content=text, metadata={"filename": "test.txt", "source_type": "txt"})]
        chunks = chunk_documents(docs, "test-doc-id")
        assert len(chunks) > 1, "Long doc should produce multiple chunks"

    def test_metadata_preserved(self):
        from app.ingestion.chunker import chunk_documents
        from langchain_core.documents import Document

        doc = Document(page_content="Short content", metadata={"filename": "my.txt", "source_type": "txt"})
        chunks = chunk_documents([doc], "doc-123")
        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "doc-123"
            assert chunk.metadata["filename"] == "my.txt"

    def test_contextual_prefix(self):
        from app.ingestion.chunker import chunk_documents
        from langchain_core.documents import Document

        doc = Document(page_content="Some content", metadata={"filename": "guide.pdf", "source_type": "pdf"})
        chunks = chunk_documents([doc], "abc")
        assert "Document: guide.pdf" in chunks[0].page_content


# ─── bm25.py ─────────────────────────────────────────────────────────────────

class TestBM25:
    def test_search_returns_results(self):
        from app.retrieval.bm25 import BM25Index

        idx = BM25Index()
        idx.add_documents(["The quick brown fox", "A lazy dog", "Machine learning models"])
        results = idx.search("machine learning", top_k=2)
        assert results, "Expected BM25 results"
        assert results[0][0] == "Machine learning models"

    def test_empty_index(self):
        from app.retrieval.bm25 import BM25Index

        idx = BM25Index()
        results = idx.search("anything")
        assert results == []


class TestIngestionJobs:
    def test_register_job_is_immediately_available(self):
        from app.ingestion.pipeline import get_job, register_job

        register_job("queued-job", "queued.md")
        job = get_job("queued-job")

        assert job is not None
        assert job["status"] == "processing"
        assert job["filename"] == "queued.md"
        assert job["stage"] == "queued"
        assert job["progress"] == 0
        assert job["chunks_indexed"] == 0
