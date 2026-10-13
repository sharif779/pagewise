"""Split page text into overlapping, sentence-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int
    sentences: Tuple[str, ...] = ()


def split_sentences(text: str) -> List[str]:
    """Split into sentences, treating line breaks as boundaries too.

    Many source documents (bullet-style FAQs, spec sheets, PDF line items)
    state one fact per line without a trailing period. Splitting only on
    `.!?` would merge unrelated facts like "Contact: a@b.com" and "Founded
    2006" into a single run-on "sentence" whenever they sit on adjacent
    lines, which can make the extracted answer mix the two together.
    """
    sentences: List[str] = []
    for line in text.splitlines():
        line = _INLINE_WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip())
    return sentences


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
            chunks.append(
                Chunk(chunk_id=len(chunks), text=" ".join(piece), page=page_no, sentences=tuple(piece))
            )
            if start + window >= len(sentences):
                break
    return chunks
