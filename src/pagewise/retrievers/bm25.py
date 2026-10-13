"""Keyword retrieval via BM25. Zero ML, catches exact terms and model numbers."""

from __future__ import annotations

import re
from typing import List

import numpy as np
from rank_bm25 import BM25Plus

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, standard English stopword list. Filtering these out keeps BM25's
# term matches meaningful even on very small corpora, where a word appearing
# in exactly half the chunks would otherwise get a near-zero idf by chance.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "on", "in", "at", "to", "for", "and", "or", "with", "as", "by",
    "it", "its", "this", "that", "these", "those", "do", "does", "did",
    "have", "has", "had", "i", "you", "he", "she", "we", "they", "what",
    "which", "who", "how", "when", "where", "why", "can", "could", "will",
    "would", "should", "about", "into", "than", "then", "so", "but", "if",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self) -> None:
        self._bm25: BM25Plus | None = None
        self._corpus: List[List[str]] = []

    def fit(self, texts: List[str]) -> None:
        self._corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Plus(self._corpus) if self._corpus else None

    def score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(0)
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)

    def state(self) -> dict:
        return {"corpus": self._corpus}

    def load_state(self, state: dict) -> None:
        self._corpus = state["corpus"]
        self._bm25 = BM25Plus(self._corpus) if self._corpus else None
