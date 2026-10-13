"""Extract (page_number, text) pairs from PDF, DOCX, and plain-text files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


def enforce_size_limit(path: str, max_mb: float = 5.0) -> None:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_mb:
        raise FileTooLarge(
            f"{path} is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit."
        )


def extract_pages(path: str) -> List[Tuple[int, str]]:
    """Return a list of (page_number, text) tuples, 1-indexed.

    DOCX and plain-text files have no real pagination, so they are
    returned as a single "page" (page number 1).
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in (".txt", ".md"):
        return _extract_text(path)
    raise UnsupportedFileType(
        f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(path: str) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_no, text))
    return pages


def _extract_docx(path: str) -> List[Tuple[int, str]]:
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(1, text)] if text.strip() else []


def _extract_text(path: str) -> List[Tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []
