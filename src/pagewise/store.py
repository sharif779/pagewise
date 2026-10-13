"""Save/load a DocQA index to a single file on disk. No database server needed."""

from __future__ import annotations

import pickle
from pathlib import Path

FORMAT_VERSION = 2


def save_index(path: str, data: dict) -> None:
    payload = {"format_version": FORMAT_VERSION, **data}
    Path(path).write_bytes(pickle.dumps(payload))


def load_index(path: str) -> dict:
    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Index at {path} has format_version={payload.get('format_version')!r}, "
            f"expected {FORMAT_VERSION}. Re-ingest the source document."
        )
    return payload
