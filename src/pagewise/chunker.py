"""Split page text into overlapping, sentence-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int


def split_sentences(text: str) -> List[str]:
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def make_chunks(
    pages: List[Tuple[int, str]],
    window: int = 4,
    overlap: int = 1,
) -> List[Chunk]:
    """Build overlapping sentence-window chunks, tagged with their source page.

    `window` is the number of sentences per chunk, `overlap` is how many
    trailing sentences of one chunk are repeated at the start of the next.
    """
    if window <= overlap:
        raise ValueError("window must be greater than overlap")

    chunks: List[Chunk] = []
    step = window - overlap
    for page_no, text in pages:
        sentences = split_sentences(text)
        if not sentences:
            continue
        for start in range(0, len(sentences), step):
            piece = sentences[start : start + window]
            if not piece:
                continue
            chunks.append(Chunk(chunk_id=len(chunks), text=" ".join(piece), page=page_no))
            if start + window >= len(sentences):
                break
    return chunks
