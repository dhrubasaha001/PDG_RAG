"""
tests/unit/test_repository.py
Unit tests for DocumentRepository and JobRepository.
Uses a temporary SQLite database per test (via monkeypatch on settings.DB_URL).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pytest

from backend.db.models import Document, DocumentStatus, IngestionJob, JobStatus


# ---------------------------------------------------------------------------
# Fixtures — isolated in-memory DB per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Point settings.DB_URL at a temp file for every test.
    Ensures each test starts with a clean schema.
    """
    db_file = tmp_path / "test_pdg_rag.db"
    monkeypatch.setattr(
        "backend.core.config.settings.DB_URL",
        f"sqlite:///{db_file}",
    )
    # Re-import after monkeypatch so database.py uses the new path
    import importlib
    import backend.db.database as db_mod
    importlib.reload(db_mod)

    db_mod.init_db()
    yield
    # Cleanup handled by tmp_path


@pytest.fixture
def doc_repo():
    from backend.db.repository import DocumentRepository
    return DocumentRepository()


@pytest.fixture
def job_repo():
    from backend.db.repository import JobRepository
    return JobRepository()


def _make_document(doc_id: str = None, filename: str = "test.pdf") -> Document:
    did = doc_id or ("a" * 64)
    return Document(
        id=did,
        filename=filename,
        file_hash=did,
        format="pdf",
        file_size_bytes=1024,
        file_path=f"data/uploads/{filename}",
        status=DocumentStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# DocumentRepository tests
# ---------------------------------------------------------------------------

class TestDocumentRepository:

    def test_create_and_get_by_id(self, doc_repo):
        doc = _make_document()
        created = doc_repo.create(doc)

        fetched = doc_repo.get_by_id(doc.id)
        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.filename == doc.filename
        assert fetched.status == DocumentStatus.PENDING

    def test_get_by_id_returns_none_for_missing(self, doc_repo):
        assert doc_repo.get_by_id("nonexistent") is None

    def test_get_by_hash_for_duplicate_detection(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        found = doc_repo.get_by_hash(doc.file_hash)
        assert found is not None
        assert found.id == doc.id

    def test_get_by_hash_returns_none_for_missing(self, doc_repo):
        assert doc_repo.get_by_hash("nonexistent_hash") is None

    def test_duplicate_hash_raises(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        duplicate = _make_document(doc_id="b" * 64)
        duplicate.file_hash = doc.file_hash   # same hash, different id
        with pytest.raises(Exception):         # sqlite3 UNIQUE constraint
            doc_repo.create(duplicate)

    def test_update_status_to_processing(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        updated = doc_repo.update_status(doc.id, DocumentStatus.PROCESSING)
        assert updated is not None
        assert updated.status == DocumentStatus.PROCESSING

    def test_update_status_to_indexed_sets_indexed_at(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        updated = doc_repo.update_status(
            doc.id,
            DocumentStatus.INDEXED,
            chunk_count=10,
            parent_chunk_count=3,
        )
        assert updated is not None
        assert updated.status == DocumentStatus.INDEXED
        assert updated.chunk_count == 10
        assert updated.parent_chunk_count == 3
        assert updated.indexed_at is not None

    def test_update_status_to_failed_stores_error(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        updated = doc_repo.update_status(
            doc.id,
            DocumentStatus.FAILED,
            error_message="Parser crashed",
        )
        assert updated.status == DocumentStatus.FAILED
        assert updated.error_message == "Parser crashed"

    def test_list_documents_pagination(self, doc_repo):
        # Create 5 documents
        for i in range(5):
            doc_repo.create(_make_document(doc_id=f"{'a' * 63}{i}", filename=f"doc{i}.pdf"))

        docs, total = doc_repo.list_documents(page=1, page_size=3)
        assert total == 5
        assert len(docs) == 3

        docs_p2, total_p2 = doc_repo.list_documents(page=2, page_size=3)
        assert total_p2 == 5
        assert len(docs_p2) == 2

    def test_list_documents_status_filter(self, doc_repo):
        doc_repo.create(_make_document(doc_id="a" * 64, filename="a.pdf"))
        doc_repo.create(_make_document(doc_id="b" * 64, filename="b.pdf"))
        doc_repo.update_status("a" * 64, DocumentStatus.INDEXED)

        indexed, total = doc_repo.list_documents(status_filter=DocumentStatus.INDEXED)
        assert total == 1
        assert indexed[0].status == DocumentStatus.INDEXED

    def test_delete_document(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)

        deleted = doc_repo.delete(doc.id)
        assert deleted is True
        assert doc_repo.get_by_id(doc.id) is None

    def test_delete_nonexistent_returns_false(self, doc_repo):
        assert doc_repo.delete("nonexistent") is False

    def test_timestamps_are_set_on_create(self, doc_repo):
        doc = _make_document()
        doc_repo.create(doc)
        fetched = doc_repo.get_by_id(doc.id)

        assert isinstance(fetched.created_at, datetime)
        assert isinstance(fetched.updated_at, datetime)
        assert fetched.indexed_at is None


# ---------------------------------------------------------------------------
# JobRepository tests
# ---------------------------------------------------------------------------

class TestJobRepository:

    def test_create_job(self, job_repo):
        job = job_repo.create_job(["doc1", "doc2"], queue="high_priority")

        assert job.id is not None
        assert job.total_documents == 2
        assert job.status == JobStatus.QUEUED
        assert job.queue == "high_priority"
        assert "doc1" in job.document_ids

    def test_get_job(self, job_repo):
        job = job_repo.create_job(["doc_a"])
        fetched = job_repo.get_job(job.id)

        assert fetched is not None
        assert fetched.id == job.id
        assert fetched.document_ids == ["doc_a"]

    def test_get_job_returns_none_for_missing(self, job_repo):
        assert job_repo.get_job("nonexistent-uuid") is None

    def test_update_job_progress(self, job_repo):
        job = job_repo.create_job(["d1", "d2", "d3"])

        updated = job_repo.update_job_progress(
            job.id,
            processed_delta=2,
            failed_delta=1,
            status=JobStatus.PROCESSING,
        )
        assert updated.processed_documents == 2
        assert updated.failed_documents == 1
        assert updated.status == JobStatus.PROCESSING

    def test_update_job_progress_accumulates(self, job_repo):
        job = job_repo.create_job(["d1", "d2", "d3"])

        job_repo.update_job_progress(job.id, processed_delta=1)
        job_repo.update_job_progress(job.id, processed_delta=1)
        final = job_repo.update_job_progress(job.id, processed_delta=1)

        assert final.processed_documents == 3

    def test_complete_job(self, job_repo):
        job = job_repo.create_job(["d1"])
        completed = job_repo.complete_job(job.id)

        assert completed.status == JobStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_job_partial(self, job_repo):
        job = job_repo.create_job(["d1", "d2"])
        job_repo.update_job_progress(job.id, processed_delta=1, failed_delta=1)
        completed = job_repo.complete_job(job.id, status=JobStatus.PARTIAL)

        assert completed.status == JobStatus.PARTIAL

    def test_set_celery_task_id(self, job_repo):
        job = job_repo.create_job(["d1"])
        task_id = str(uuid.uuid4())
        job_repo.set_celery_task_id(job.id, task_id)

        fetched = job_repo.get_job(job.id)
        assert fetched.celery_task_id == task_id

    def test_list_jobs_pagination(self, job_repo):
        for _ in range(5):
            job_repo.create_job(["doc"])

        jobs, total = job_repo.list_jobs(page=1, page_size=3)
        assert total == 5
        assert len(jobs) == 3

    def test_document_ids_roundtrip(self, job_repo):
        """document_ids JSON roundtrip must preserve order and values."""
        ids = [str(uuid.uuid4()) for _ in range(5)]
        job = job_repo.create_job(ids)
        fetched = job_repo.get_job(job.id)

        assert fetched.document_ids == ids
