"""Manual FAQ overrides, checked before falling back to document retrieval.

Letting an admin pin exact Q&A pairs catches the most common questions
perfectly, without depending on how the retriever happens to score them.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from .answer import _lexical_cosine, _lexical_vector


class FAQStore:
    def __init__(self) -> None:
        self._entries: List[Tuple[str, str, Counter]] = []

    def add(self, question: str, answer: str) -> None:
        self._entries.append((question, answer, _lexical_vector(question)))

    def match(self, question: str, threshold: float = 0.6) -> Optional[dict]:
        if not self._entries:
            return None
        qvec = _lexical_vector(question)
        best = max(self._entries, key=lambda e: _lexical_cosine(qvec, e[2]), default=None)
        if best is None:
            return None
        score = _lexical_cosine(qvec, best[2])
        if score < threshold:
            return None
        return {"question": best[0], "answer": best[1], "score": score}

    def state(self) -> List[Tuple[str, str]]:
        return [(q, a) for q, a, _ in self._entries]

    def load_state(self, entries: List[Tuple[str, str]]) -> None:
        self._entries = []
        for q, a in entries:
            self.add(q, a)

    def __len__(self) -> int:
        return len(self._entries)
