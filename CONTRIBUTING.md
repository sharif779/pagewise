# Contributing to pagewise

## Setup

```bash
git clone https://github.com/sharif779/pagewise.git
cd pagewise
pip install -e ".[dev,semantic,server]"
```

## Running tests

```bash
pytest
```

Tests that require the `semantic` or `server` extras skip automatically if those
packages aren't installed (via `pytest.importorskip`).

## Project layout

- `src/pagewise/parsers.py` — PDF/DOCX/TXT text extraction.
- `src/pagewise/chunker.py` — sentence-window chunking.
- `src/pagewise/retrievers/` — BM25, embedding, and hybrid retrieval.
- `src/pagewise/answer.py` — sentence-level answer picking.
- `src/pagewise/faq.py` — manual Q&A overrides.
- `src/pagewise/core.py` — the `DocQA` public API.
- `src/pagewise/server.py` — FastAPI `/upload` and `/ask`.
- `src/pagewise/cli.py` — `pagewise ingest|ask|serve`.
- `widget/` — the embeddable chat bubble (vanilla JS, no build step).

## Pull requests

- Keep the base install free of heavy ML dependencies — anything beyond
  `pypdf`/`python-docx`/`rank-bm25`/`numpy` belongs behind an extra in `pyproject.toml`.
  Avoid AGPL-licensed dependencies (e.g. `PyMuPDF`); MIT/BSD/Apache-2.0 only.
  Add a test for new behavior, and run `pytest` before opening the PR.
