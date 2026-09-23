"""
app/ingestion/chunker.py — RecursiveCharacterTextSplitter wrapper.

Prepends document title + nearest heading to each chunk before embedding
(contextual retrieval — significantly improves hit rate on ambiguous queries).
"""
from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

_cfg = get_settings()

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_cfg.chunk_size,
    chunk_overlap=_cfg.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z\s]{5,}:?\s*)$", re.MULTILINE)


def _extract_nearest_heading(text: str) -> str:
    """Find the last Markdown heading or ALL-CAPS line before the chunk."""
    matches = list(_HEADING_RE.finditer(text))
    return matches[-1].group().strip() if matches else ""


def chunk_documents(docs: List[Document], doc_id: str) -> List[Document]:
    """
    Split documents into chunks and enrich metadata.
    Also prepends title context to the chunk text.
    """
    chunks: list[Document] = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")

        raw_chunks = _SPLITTER.split_documents([doc])
        for idx, chunk in enumerate(raw_chunks):
            heading = _extract_nearest_heading(chunk.page_content)
            # Contextual prefix: "Document: <name> | Section: <heading>"
            context_prefix = f"Document: {filename}"
            if heading:
                context_prefix += f" | Section: {heading}"

            enriched_text = f"{context_prefix}\n\n{chunk.page_content}"

            chunk.page_content = enriched_text
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "source_type": source_type,
                    "filename": filename,
                }
            )
            chunks.append(chunk)

    return chunks
