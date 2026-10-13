"""DocQA: extractive question answering over a single PDF/DOCX/TXT file.

No LLM. It finds the passage that already contains the answer and returns
it verbatim, with a page reference, instead of generating new text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import parsers
from .answer import best_sentence
from .chunker import Chunk, make_chunks
from .faq import FAQStore
from .retrievers import BM25Retriever, HybridRetriever, is_semantic_available
from .retrievers.base import min_max_normalize
from .store import load_index, save_index

NOT_FOUND_MESSAGE = "I couldn't find that in the document."

Mode = str  # "auto" | "bm25" | "semantic" | "hybrid"


@dataclass
class AnswerResult:
    answer: str
    found: bool
    page: Optional[int] = None
    score: float = 0.0
    context: str = ""
    source: str = "document"  # "document" or "faq"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "found": self.found,
            "page": self.page,
            "score": round(self.score, 4),
            "context": self.context,
            "source": self.source,
        }


def _resolve_mode(mode: Mode) -> Mode:
    if mode != "auto":
        return mode
    return "hybrid" if is_semantic_available() else "bm25"


def _build_retriever(mode: Mode, embedding_model: str, bm25_weight: float, embedding_weight: float):
    mode = _resolve_mode(mode)
    if mode == "bm25":
        return mode, BM25Retriever()
    if mode in ("semantic", "hybrid"):
        return mode, HybridRetriever(
            bm25_weight=0.0 if mode == "semantic" else bm25_weight,
            embedding_weight=1.0 if mode == "semantic" else embedding_weight,
            embedding_model=embedding_model,
        )
    raise ValueError(f"Unknown mode {mode!r}. Use 'auto', 'bm25', 'semantic', or 'hybrid'.")


class DocQA:
    def __init__(
        self,
        mode: Mode = "auto",
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 0.4,
        embedding_weight: float = 0.6,
    ) -> None:
        self.mode, self.retriever = _build_retriever(
            mode, embedding_model, bm25_weight, embedding_weight
        )
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.faq = FAQStore()
        self.source_file: Optional[str] = None

    # -- ingestion ---------------------------------------------------

    def ingest(self, path: str, window: int = 4, overlap: int = 1, max_mb: float = 5.0) -> int:
        """Parse, chunk, and index a document. Returns the number of chunks created."""
        parsers.enforce_size_limit(path, max_mb=max_mb)
        pages = parsers.extract_pages(path)
        self.chunks = make_chunks(pages, window=window, overlap=overlap)
        self.retriever.fit([c.text for c in self.chunks])
        self.source_file = path
        return len(self.chunks)

    def add_faq(self, question: str, answer: str) -> None:
        self.faq.add(question, answer)

    # -- querying ------------------------------------------------------

    def ask(self, question: str, k: int = 3, threshold: float = 0.2, faq_threshold: float = 0.6) -> AnswerResult:
        faq_hit = self.faq.match(question, threshold=faq_threshold)
        if faq_hit is not None:
            return AnswerResult(
                answer=faq_hit["answer"],
                found=True,
                score=faq_hit["score"],
                source="faq",
            )

        if not self.chunks:
            return AnswerResult(answer=NOT_FOUND_MESSAGE, found=False)

        raw_scores = self.retriever.score(question)
        if raw_scores.size == 0:
            return AnswerResult(answer=NOT_FOUND_MESSAGE, found=False)

        # Normalize to [0, 1] so `threshold` means the same thing regardless
        # of which retriever produced the (otherwise unbounded) raw scores.
        scores = min_max_normalize(raw_scores)
        top_idx = scores.argsort()[::-1][:k]
        top_score = float(scores[top_idx[0]])
        if top_score < threshold:
            return AnswerResult(answer=NOT_FOUND_MESSAGE, found=False, score=top_score)

        candidates = [self.chunks[i] for i in top_idx]
        embedder = getattr(self.retriever, "embedding", None)
        embedder_model = embedder.model if embedder is not None else None

        picked = best_sentence(candidates, question, embedder=embedder_model)
        return AnswerResult(
            answer=picked["sentence"] or candidates[0].text,
            found=True,
            page=picked["page"],
            score=top_score,
            context=candidates[0].text,
            source="document",
        )

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        save_index(
            path,
            {
                "mode": self.mode,
                "embedding_model": self.embedding_model,
                "source_file": self.source_file,
                "chunks": [(c.chunk_id, c.text, c.page) for c in self.chunks],
                "retriever_state": self.retriever.state(),
                "faq_state": self.faq.state(),
            },
        )

    @classmethod
    def load(cls, path: str) -> "DocQA":
        data = load_index(path)
        qa = cls(mode=data["mode"], embedding_model=data["embedding_model"])
        qa.source_file = data["source_file"]
        qa.chunks = [Chunk(chunk_id=cid, text=text, page=page) for cid, text, page in data["chunks"]]
        qa.retriever.load_state(data["retriever_state"])
        qa.faq.load_state(data["faq_state"])
        return qa
