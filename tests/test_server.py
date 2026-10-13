import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pagewise.server import create_app


def test_upload_then_ask(tmp_path, sample_path):
    app = create_app(index_path=str(tmp_path / "index.pkl"))
    client = TestClient(app)

    with open(sample_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("sample.txt", f, "text/plain")})
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0

    resp = client.post("/ask", json={"question": "How much does the Pro plan cost?"})
    assert resp.status_code == 200
    assert "29" in resp.json()["answer"]


def test_ask_before_upload_returns_400(tmp_path):
    app = create_app(index_path=str(tmp_path / "missing.pkl"))
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 400


def test_upload_rejects_unsupported_extension(tmp_path):
    app = create_app(index_path=str(tmp_path / "index.pkl"))
    client = TestClient(app)
    resp = client.post("/upload", files={"file": ("sample.exe", b"junk", "application/octet-stream")})
    assert resp.status_code == 400


def test_delete_document_clears_state_and_index_file(tmp_path, sample_path):
    index_path = tmp_path / "index.pkl"
    app = create_app(index_path=str(index_path))
    client = TestClient(app)

    with open(sample_path, "rb") as f:
        client.post("/upload", files={"file": ("sample.txt", f, "text/plain")})
    assert index_path.exists()
    assert client.get("/health").json()["ready"] is True

    resp = client.delete("/document")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "ready": False}
    assert not index_path.exists()
    assert client.get("/health").json()["ready"] is False

    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 400


def test_delete_document_is_safe_when_nothing_uploaded(tmp_path):
    app = create_app(index_path=str(tmp_path / "missing.pkl"))
    client = TestClient(app)
    resp = client.delete("/document")
    assert resp.status_code == 200
