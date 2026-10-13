"""Semantic retrieval via a small sentence-embedding model (not an LLM).

Requires the optional `semantic` extra: pip install pagewise[semantic]
"""

from __future__ import annotations

import importlib.util
from typing import List, Optional

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def is_semantic_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class EmbeddingRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._vectors: Optional[np.ndarray] = None

    @property
    def model(self):
        """The underlying sentence encoder, loaded on first use."""
        return self._load_model()

    def _load_model(self):
        if self._model is None:
            if not is_semantic_available():
                raise ImportError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install it with: pip install pagewise[semantic]"
                )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: List[str]) -> None:
        model = self._load_model()
        if not texts:
            self._vectors = np.zeros((0, 0))
            return
        self._vectors = np.asarray(
            model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        )

    def score(self, query: str) -> np.ndarray:
        if self._vectors is None or self._vectors.size == 0:
            return np.zeros(0)
        model = self._load_model()
        qv = np.asarray(model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0])
        return self._vectors @ qv

    def state(self) -> dict:
        return {"model_name": self.model_name, "vectors": self._vectors}

    def load_state(self, state: dict) -> None:
        self.model_name = state["model_name"]
        self._vectors = state["vectors"]
        self._model = None
