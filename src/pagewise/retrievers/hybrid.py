"""Combine BM25 (exact keywords) with embeddings (paraphrases) by score fusion."""

from __future__ import annotations

from typing import List

import numpy as np

from .base import min_max_normalize
from .bm25 import BM25Retriever
from .embedding import EmbeddingRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25_weight: float = 0.4,
        embedding_weight: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.bm25 = BM25Retriever()
        self.embedding = EmbeddingRetriever(embedding_model)

    def fit(self, texts: List[str]) -> None:
        self.bm25.fit(texts)
        self.embedding.fit(texts)

    def score(self, query: str) -> np.ndarray:
        bm25_scores = min_max_normalize(self.bm25.score(query))
        emb_scores = min_max_normalize(self.embedding.score(query))
        if bm25_scores.size == 0:
            return emb_scores
        if emb_scores.size == 0:
            return bm25_scores
        return self.bm25_weight * bm25_scores + self.embedding_weight * emb_scores

    def state(self) -> dict:
        return {
            "bm25_weight": self.bm25_weight,
            "embedding_weight": self.embedding_weight,
            "bm25": self.bm25.state(),
            "embedding": self.embedding.state(),
        }

    def load_state(self, state: dict) -> None:
        self.bm25_weight = state["bm25_weight"]
        self.embedding_weight = state["embedding_weight"]
        self.bm25.load_state(state["bm25"])
        self.embedding.load_state(state["embedding"])
