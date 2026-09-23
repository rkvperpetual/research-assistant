"""
app/ingestion/parsers.py — Per-format document loaders.

Each parser function takes a Path and returns list[Document].
All metadata keys are normalised here so downstream code has a stable schema.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.ingestion.detect import FileType

log = logging.getLogger(__name__)

# ─── Metadata helpers ─────────────────────────────────────────────────────────

def _base_meta(path: Path, source_type: str, extra: dict | None = None) -> dict:
    return {
        "filename": path.name,
        "source_type": source_type,
        **(extra or {}),
    }


# ─── Text normalisation ───────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Collapse whitespace, de-hyphenate line-breaks, strip repeated blank lines."""
    # De-hyphenate: "infor-\nmation" → "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse excessive blank lines to double-newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


# ─── PDF ──────────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> List[Document]:
    """
    Use PyMuPDF (fitz) to extract text.
    If a page yields < 100 chars → route the whole doc through OCR.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        log.error("PyMuPDF not installed")
        return []

    doc = fitz.open(str(path))
    pages_text: list[tuple[int, str]] = []
    needs_ocr = False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if len(text.strip()) < 100:
            needs_ocr = True
        pages_text.append((page_num, text))

    doc.close()

    if needs_ocr:
        log.info("%s appears to be a scanned PDF — routing to OCR", path.name)
        from app.ingestion.ocr import ocr_pdf_pages

        full_text = ocr_pdf_pages(path)
        return [
            Document(
                page_content=_normalise(full_text),
                metadata=_base_meta(path, "pdf_scanned"),
            )
        ]

    docs: list[Document] = []
    for page_num, text in pages_text:
        if text.strip():
            docs.append(
                Document(
                    page_content=_normalise(text),
                    metadata=_base_meta(path, "pdf", {"page": page_num}),
                )
            )
    return docs


# ─── DOCX ─────────────────────────────────────────────────────────────────────

def parse_docx(path: Path) -> List[Document]:
    try:
        from docx import Document as DocxDocument  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
    except ImportError:
        log.error("python-docx not installed")
        return []

    dox = DocxDocument(str(path))
    parts: list[str] = []

    for para in dox.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    # Tables → pipe-separated markdown rows
    for table in dox.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            parts.append("\n".join(rows))

    full_text = "\n\n".join(parts)
    return [
        Document(
            page_content=_normalise(full_text),
            metadata=_base_meta(path, "docx"),
        )
    ]


# ─── HTML ─────────────────────────────────────────────────────────────────────

def parse_html(path: Path) -> List[Document]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        log.error("beautifulsoup4 not installed")
        return []

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Remove boilerplate elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return [
        Document(
            page_content=_normalise(text),
            metadata=_base_meta(path, "html"),
        )
    ]


def parse_url(url: str) -> List[Document]:
    """Fetch and parse a URL."""
    try:
        from langchain_community.document_loaders import WebBaseLoader  # type: ignore

        loader = WebBaseLoader(url)
        docs = loader.load()
        for d in docs:
            d.page_content = _normalise(d.page_content)
            d.metadata["source_type"] = "html_url"
            d.metadata["filename"] = url
        return docs
    except Exception as e:
        log.error("URL parse failed for %s: %s", url, e)
        return []


# ─── Image ────────────────────────────────────────────────────────────────────

def parse_image(path: Path) -> List[Document]:
    from app.ingestion.ocr import ocr_image_path

    text = ocr_image_path(path)
    return [
        Document(
            page_content=_normalise(text),
            metadata=_base_meta(path, "image"),
        )
    ]


# ─── Plain text / Markdown ────────────────────────────────────────────────────

def parse_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_type = "md" if path.suffix.lower() == ".md" else "txt"
    return [
        Document(
            page_content=_normalise(text),
            metadata=_base_meta(path, source_type),
        )
    ]


# ─── CSV / XLSX ───────────────────────────────────────────────────────────────

def parse_tabular(path: Path) -> List[Document]:
    try:
        import pandas as pd  # type: ignore

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        md_table = df.to_markdown(index=False)
        return [
            Document(
                page_content=md_table,
                metadata=_base_meta(path, "tabular"),
            )
        ]
    except Exception as e:
        log.error("Tabular parse failed for %s: %s", path.name, e)
        return []


# ─── Dispatch ─────────────────────────────────────────────────────────────────

_DISPATCH: dict[FileType, any] = {
    "pdf":     parse_pdf,
    "png":     parse_image,
    "jpg":     parse_image,
    "tiff":    parse_image,
    "docx":    parse_docx,
    "html":    parse_html,
    "txt":     parse_text,
    "md":      parse_text,
    "csv":     parse_tabular,
    "xlsx":    parse_tabular,
}


def parse_file(path: Path, file_type: FileType) -> List[Document]:
    """
    Dispatch to the appropriate parser.  Returns [] on unknown type.
    """
    fn = _DISPATCH.get(file_type)
    if fn is None:
        log.warning("No parser for type %r (%s)", file_type, path.name)
        return []
    try:
        return fn(path)
    except Exception as e:
        log.exception("Parser %s failed for %s: %s", fn.__name__, path.name, e)
        return []
