"""pagewise: extractive question answering over a single PDF/DOCX/TXT file, no LLM."""

from .core import AnswerResult, DocQA, NOT_FOUND_MESSAGE

__version__ = "0.1.0"
__all__ = ["DocQA", "AnswerResult", "NOT_FOUND_MESSAGE", "__version__"]
