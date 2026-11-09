"""
tests/unit/test_api.py
Unit tests for the FastAPI ingestion + job endpoints.
Uses FastAPI TestClient — no real Redis, Weaviate, or Celery needed.
All async tasks are mocked to return immediately.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF   = REPO_ROOT / "backend" / "data" / "DAA.pdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Fresh SQLite DB per test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr("backend.core.config.settings.DB_URL", f"sqlite:///{db_file}")
    import importlib, backend.db.database as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    yield


@pytest.fixture(autouse=True)
def mock_celery_tasks():
    """Prevent real Celery dispatch — return a fake AsyncResult."""
    fake_task = MagicMock()
    fake_task.id = "fake-task-id-001"
    with patch("backend.workers.tasks.ingest_single_document.apply_async",
               return_value=fake_task) as mock_single, \
         patch("backend.workers.tasks.ingest_batch_documents.apply_async",
               return_value=fake_task) as mock_batch:
        yield mock_single, mock_batch


@pytest.fixture(autouse=True)
def mock_weaviate_health():
    """Don't try to connect to Weaviate on startup."""
    with patch("backend.main.health_check", return_value=False), \
         patch("backend.main.ensure_schema"):
        yield


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with a clean app instance."""
    monkeypatch.setattr(
        "backend.core.config.settings.UPLOAD_DIR",
        str(tmp_path / "uploads"),
    )
    from backend.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def pdf_bytes():
    """Return bytes of the real DAA.pdf if available, else a minimal fake."""
    if DAA_PDF.exists():
        return DAA_PDF.read_bytes()
    # Minimal valid PDF bytes for testing
    return b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\nxref\n0 1\n0000000000 65535 f\n%%EOF"


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

class TestRootAndHealth:

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "PDG_RAG"

    def test_health_endpoint_exists(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "dependencies" in body


# ---------------------------------------------------------------------------
# POST /ingest/upload
# ---------------------------------------------------------------------------

class TestUploadEndpoint:

    def test_upload_pdf_returns_202(self, client, pdf_bytes):
        r = client.post(
            "/ingest/upload",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "queued"
        assert body["doc_id"]
        assert body["job_id"]

    def test_upload_creates_document_record(self, client, pdf_bytes):
        r = client.post(
            "/ingest/upload",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 202
        doc_id = r.json()["doc_id"]

        # Verify record in registry
        from backend.db.repository import DocumentRepository
        doc = DocumentRepository().get_by_id(doc_id)
        assert doc is not None
        assert doc.filename == "doc.pdf"

    def test_duplicate_upload_returns_duplicate_status(self, client, pdf_bytes):
        # First upload
        r1 = client.post(
            "/ingest/upload",
            files={"file": ("dup.pdf", pdf_bytes, "application/pdf")},
        )
        assert r1.status_code == 202

        # Second upload (same content)
        r2 = client.post(
            "/ingest/upload",
            files={"file": ("dup.pdf", pdf_bytes, "application/pdf")},
        )
        assert r2.status_code == 202
        assert r2.json()["status"] == "duplicate"

    def test_unsupported_file_type_returns_415(self, client):
        r = client.post(
            "/ingest/upload",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert r.status_code == 415

    def test_oversized_file_returns_413(self, client, monkeypatch):
        monkeypatch.setattr("backend.core.config.settings.MAX_FILE_SIZE_MB", 0)
        r = client.post(
            "/ingest/upload",
            files={"file": ("big.pdf", b"X" * 1025, "application/pdf")},
        )
        assert r.status_code == 413

    def test_upload_dispatches_celery_task(self, client, pdf_bytes, mock_celery_tasks):
        mock_single, _ = mock_celery_tasks
        client.post(
            "/ingest/upload",
            files={"file": ("task.pdf", pdf_bytes, "application/pdf")},
        )
        mock_single.assert_called_once()

    def test_upload_txt_file(self, client):
        r = client.post(
            "/ingest/upload",
            files={"file": ("notes.txt", b"Hello world content.", "text/plain")},
        )
        assert r.status_code == 202
        assert r.json()["status"] == "queued"

    def test_upload_md_file(self, client):
        r = client.post(
            "/ingest/upload",
            files={"file": ("readme.md", b"# Title\n\nContent.", "text/markdown")},
        )
        assert r.status_code == 202


# ---------------------------------------------------------------------------
# POST /ingest/batch
# ---------------------------------------------------------------------------

class TestBatchEndpoint:

    def test_batch_upload_multiple_files(self, client, pdf_bytes):
        files = [
            ("files", ("a.pdf", pdf_bytes,        "application/pdf")),
            ("files", ("b.txt", b"Content B.",     "text/plain")),
            ("files", ("c.txt", b"Content C here.","text/plain")),
        ]
        r = client.post("/ingest/batch", files=files)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "queued"
        assert body["total"] == 3

    def test_batch_detects_duplicates(self, client, pdf_bytes):
        # Upload once
        client.post("/ingest/upload", files={"file": ("orig.pdf", pdf_bytes, "application/pdf")})

        # Batch with one duplicate + one new
        files = [
            ("files", ("orig.pdf", pdf_bytes, "application/pdf")),   # duplicate
            ("files", ("new.txt",  b"Brand new content here.", "text/plain")),
        ]
        r = client.post("/ingest/batch", files=files)
        assert r.status_code == 202
        body = r.json()
        assert body["duplicates"] == 1
        assert body["total"] == 1

    def test_batch_all_duplicates_returns_duplicate_status(self, client, pdf_bytes):
        client.post("/ingest/upload", files={"file": ("f.pdf", pdf_bytes, "application/pdf")})
        r = client.post("/ingest/batch", files=[("files", ("f.pdf", pdf_bytes, "application/pdf"))])
        assert r.status_code == 202
        assert r.json()["status"] == "duplicate"


# ---------------------------------------------------------------------------
# GET /ingest/documents
# ---------------------------------------------------------------------------

class TestDocumentListEndpoint:

    def test_empty_list(self, client):
        r = client.get("/ingest/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["documents"] == []

    def test_list_after_upload(self, client, pdf_bytes):
        client.post("/ingest/upload", files={"file": ("list.pdf", pdf_bytes, "application/pdf")})
        r = client.get("/ingest/documents")
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_pagination_params(self, client, pdf_bytes):
        # Upload 3 docs
        for i, content in enumerate([b"Doc one text.", b"Doc two text.", b"Doc three text."]):
            client.post("/ingest/upload", files={"file": (f"doc{i}.txt", content, "text/plain")})

        r = client.get("/ingest/documents?page=1&page_size=2")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert len(body["documents"]) == 2
        assert body["page"] == 1

    def test_status_filter(self, client, pdf_bytes):
        client.post("/ingest/upload", files={"file": ("f.pdf", pdf_bytes, "application/pdf")})
        r = client.get("/ingest/documents?status=pending")
        assert r.status_code == 200
        for doc in r.json()["documents"]:
            assert doc["status"] == "pending"

    def test_invalid_status_filter_returns_400(self, client):
        r = client.get("/ingest/documents?status=nonsense")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /ingest/documents/{doc_id}
# ---------------------------------------------------------------------------

class TestDeleteEndpoint:

    def test_delete_existing_document(self, client, pdf_bytes):
        r = client.post("/ingest/upload", files={"file": ("del.pdf", pdf_bytes, "application/pdf")})
        doc_id = r.json()["doc_id"]

        with patch("backend.indexing.indexer.WeaviateIndexer.delete_document",
                   return_value=True):
            dr = client.delete(f"/ingest/documents/{doc_id}")

        assert dr.status_code == 200
        assert dr.json()["deleted"] is True

        # Confirm removed from registry
        rr = client.get("/ingest/documents")
        assert rr.json()["total"] == 0

    def test_delete_nonexistent_returns_404(self, client):
        r = client.delete("/ingest/documents/nonexistent_id")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  &  GET /jobs
# ---------------------------------------------------------------------------

class TestJobEndpoints:

    def test_get_job_after_upload(self, client, pdf_bytes):
        r = client.post("/ingest/upload", files={"file": ("j.pdf", pdf_bytes, "application/pdf")})
        job_id = r.json()["job_id"]

        jr = client.get(f"/jobs/{job_id}")
        assert jr.status_code == 200
        body = jr.json()
        assert body["id"] == job_id
        assert body["total_documents"] == 1

    def test_get_job_not_found_returns_404(self, client):
        r = client.get("/jobs/nonexistent-job-id")
        assert r.status_code == 404

    def test_list_jobs(self, client, pdf_bytes):
        client.post("/ingest/upload", files={"file": ("jl.pdf", pdf_bytes, "application/pdf")})
        r = client.get("/jobs")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_list_jobs_pagination(self, client):
        r = client.get("/jobs?page=1&page_size=10")
        assert r.status_code == 200
        body = r.json()
        assert "jobs" in body
        assert "total" in body
