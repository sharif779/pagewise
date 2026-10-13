"""FastAPI wrapper exposing /upload and /ask. pip install pagewise[server].

Single-tenant by default: one index file on disk, one document at a time.
For multiple tenants, run one process per tenant with a different --index
path, or extend `_load_or_create` to key the index by an API key/tenant id.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core import DocQA
from .parsers import FileTooLarge, UnsupportedFileType

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class AskRequest(BaseModel):
    question: str
    k: int = 3
    threshold: float = 0.2


class IngestResponse(BaseModel):
    chunks: int
    mode: str


def create_app(index_path: str = "pagewise_index.pkl", max_mb: float = 5.0) -> FastAPI:
    app = FastAPI(title="pagewise", description="Extractive document Q&A, no LLM.")

    allowed_origins = os.environ.get("PAGEWISE_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict = {"qa": None}

    def _load_existing() -> None:
        if state["qa"] is None and Path(index_path).exists():
            state["qa"] = DocQA.load(index_path)

    _load_existing()

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": state["qa"] is not None}

    @app.post("/upload", response_model=IngestResponse)
    def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        try:
            qa = DocQA(mode="auto")
            chunks = qa.ingest(tmp_path, max_mb=max_mb)
            qa.save(index_path)
            state["qa"] = qa
        except FileTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except UnsupportedFileType as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            os.unlink(tmp_path)

        return IngestResponse(chunks=chunks, mode=qa.mode)

    @app.post("/ask")
    def ask(payload: AskRequest):
        if state["qa"] is None:
            raise HTTPException(400, "No document has been uploaded yet. POST a file to /upload first.")
        result = state["qa"].ask(payload.question, k=payload.k, threshold=payload.threshold)
        return result.to_dict()

    return app
