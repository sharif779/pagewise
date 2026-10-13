"""Common interface every retriever implements."""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Retriever(Protocol):
    def fit(self, texts: List[str]) -> None:
        """Index the given chunk texts, in order. chunk index == list index."""

    def score(self, query: str) -> np.ndarray:
        """Return a similarity score for every indexed chunk, highest = best match."""

    def state(self) -> dict:
        """Return a picklable snapshot of this retriever's fitted state."""

    def load_state(self, state: dict) -> None:
        """Restore a snapshot produced by `state()`."""


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A tie at zero means "nothing matched" (all zeros);
    a tie above zero means "every candidate matched equally well" (all ones).
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores) if hi > 1e-12 else np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
