"""Pick the best sentence(s) inside a set of candidate chunks.

Uses the semantic embedder when available, otherwise falls back to a
dependency-free lexical cosine similarity so pure-BM25 installs still work.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .chunker import Chunk, split_sentences
from .retrievers.bm25 import tokenize


def _lexical_vector(text: str) -> Counter:
    return Counter(tokenize(text))


def _lexical_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in shared)
    if numerator == 0:
        return 0.0
    denom_a = sum(v * v for v in a.values()) ** 0.5
    denom_b = sum(v * v for v in b.values()) ** 0.5
    return numerator / (denom_a * denom_b)


def best_sentence(
    candidate_chunks: List[Chunk],
    question: str,
    embedder: Optional[object] = None,
) -> dict:
    """Return {"sentence": str, "page": int} for the best-matching sentence.

    `embedder` is any object exposing `.encode(list[str], normalize_embeddings=True)`
    (e.g. a sentence-transformers model). If omitted, lexical overlap is used.
    """
    sentence_pages = []
    for chunk in candidate_chunks:
        for sentence in split_sentences(chunk.text):
            sentence_pages.append((sentence, chunk.page))

    if not sentence_pages:
        return {"sentence": "", "page": candidate_chunks[0].page if candidate_chunks else None}

    sentences = [s for s, _ in sentence_pages]

    if embedder is not None:
        import numpy as np

        qv = np.asarray(embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0])
        sv = np.asarray(embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False))
        scores = sv @ qv
        best_idx = int(scores.argmax())
    else:
        qvec = _lexical_vector(question)
        scores = [_lexical_cosine(qvec, _lexical_vector(s)) for s in sentences]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

    sentence, page = sentence_pages[best_idx]
    return {"sentence": sentence, "page": page}
