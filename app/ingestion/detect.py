"""
app/ingestion/detect.py — Format detection by magic bytes, fallback to extension.

Never trusts the client-supplied content-type header.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

# Magic byte signatures (big-endian, compared against the first N bytes)
_SIGNATURES: list[tuple[bytes, int, str]] = [
    (b"%PDF",          4,  "pdf"),
    (b"\x89PNG\r\n\x1a\n", 8, "png"),
    (b"\xff\xd8\xff",  3,  "jpg"),
    (b"II*\x00",       4,  "tiff"),
    (b"MM\x00*",       4,  "tiff"),
    (b"GIF87a",        6,  "gif"),
    (b"GIF89a",        6,  "gif"),
    # DOCX / XLSX / PPTX are ZIP files with specific internal structure
    (b"PK\x03\x04",   4,  "zip_office"),
    # HTML (often starts with <!DOCTYPE or <html)
    (b"<!DOCTYPE",     9,  "html"),
    (b"<html",         5,  "html"),
    (b"<HTML",         5,  "html"),
]

FileType = Literal["pdf", "png", "jpg", "tiff", "docx", "xlsx", "html", "txt", "md", "csv", "unknown"]


def detect_file_type(path: Path) -> FileType:
    """
    Detect file type by magic bytes first, then extension.
    Returns a normalised FileType string.
    """
    raw = b""
    try:
        with open(path, "rb") as f:
            raw = f.read(16)
    except OSError:
        pass

    for sig, length, ftype in _SIGNATURES:
        if raw[:length] == sig[:length]:
            if ftype == "zip_office":
                # distinguish DOCX from XLSX by extension
                ext = path.suffix.lower()
                if ext == ".docx":
                    return "docx"
                if ext == ".xlsx":
                    return "xlsx"
                return "docx"  # default to docx for unknown zip-office
            return ftype  # type: ignore[return-value]

    # Fall back to extension
    ext_map: dict[str, FileType] = {
        ".pdf":  "pdf",
        ".png":  "png",
        ".jpg":  "jpg",
        ".jpeg": "jpg",
        ".tiff": "tiff",
        ".tif":  "tiff",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".html": "html",
        ".htm":  "html",
        ".txt":  "txt",
        ".md":   "md",
        ".csv":  "csv",
    }
    return ext_map.get(path.suffix.lower(), "unknown")
