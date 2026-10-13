# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `DELETE /document` endpoint to clear the uploaded document and its index file.

### Fixed

- Sentences on separate lines without trailing punctuation (bullet-style FAQs,
  spec sheets) no longer get merged into one run-on answer. See `Chunk.sentences`.

## [0.1.0]

### Added

- Initial release: `DocQA` core API (`ingest`, `ask`, `save`, `load`).
- PDF, DOCX, and TXT parsing.
- BM25, embedding, and hybrid retrievers.
- Extractive sentence-level answer picking with a "not found" threshold.
- Manual FAQ override layer.
- `pagewise` CLI (`ingest`, `ask`, `serve`).
- FastAPI server with `/upload` and `/ask`.
- Embeddable vanilla-JS chat widget.
