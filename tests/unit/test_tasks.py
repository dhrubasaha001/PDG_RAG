"""
tests/unit/test_tasks.py
Unit tests for Celery tasks — runs without Redis using ALWAYS_EAGER mode.

ALWAYS_EAGER executes tasks synchronously in-process, no broker needed.
Integration tests (with real Redis + Weaviate) live in tests/integration/.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF = REPO_ROOT / "backend" / "data" / "DAA.pdf"


# ---------------------------------------------------------------------------
# Fixtures — isolated DB + ALWAYS_EAGER Celery
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(
        "backend.core.config.settings.DB_URL",
        f"sqlite:///{db_file}",
    )
    import importlib
    import backend.db.database as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    yield


@pytest.fixture(autouse=True)
def celery_eager():
    """Run Celery tasks synchronously in-process — no broker needed."""
    from backend.workers.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False  # don't bubble exceptions
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = True


@pytest.fixture
def doc_repo():
    from backend.db.repository import DocumentRepository
    return DocumentRepository()


@pytest.fixture
def job_repo():
    from backend.db.repository import JobRepository
    return JobRepository()


def _create_db_records(doc_repo, job_repo, doc_id: str, file_path: str):
    """Helper: create Document + IngestionJob records in the registry."""
    from backend.db.models import Document, DocumentStatus
    doc = Document(
        id=doc_id,
        filename=Path(file_path).name,
        file_hash=doc_id,
        format="pdf",
        file_size_bytes=1024,
        file_path=file_path,
        status=DocumentStatus.PENDING,
    )
    doc_repo.create(doc)
    job = job_repo.create_job([doc_id], queue="high_priority")
    return doc, job


def _make_mock_self(max_retries: int = 3, current_retries: int = 3):
    """
    Build a mock Celery task 'self' with retries already exhausted.
    When retries >= max_retries the task takes the permanent-failure branch.
    """
    mock_self = MagicMock()
    mock_self.max_retries = max_retries
    mock_self.request.retries = current_retries
    # retry() should raise so the task doesn't loop
    mock_self.retry.side_effect = Exception("max retries reached")
    return mock_self


# ---------------------------------------------------------------------------
# ingest_single_document — success path
# ---------------------------------------------------------------------------

class TestIngestSingleDocument:

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_successful_ingestion_updates_registry(self, doc_repo, job_repo):
        """Full pipeline run on DAA.pdf — checks registry is updated to INDEXED."""
        from backend.workers.tasks import ingest_single_document
        from backend.db.models import DocumentStatus
        from backend.indexing.indexer import IndexResult

        doc_id = "a" * 64
        _, job = _create_db_records(doc_repo, job_repo, doc_id, str(DAA_PDF))

        mock_result = IndexResult(
            doc_id=doc_id,
            success=True,
            chunks_indexed=3,
            parents_indexed=2,
            errors=[],
            duration_seconds=0.1,
        )
        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_result,
        ):
            result = ingest_single_document.delay(doc_id, str(DAA_PDF), job.id)

        assert result.successful()
        output = result.get()
        assert output["status"] == "indexed"
        assert output["chunk_count"] == 3

        doc = doc_repo.get_by_id(doc_id)
        assert doc.status == DocumentStatus.INDEXED
        assert doc.chunk_count == 3
        assert doc.indexed_at is not None

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_successful_ingestion_increments_job_progress(self, doc_repo, job_repo):
        from backend.workers.tasks import ingest_single_document
        from backend.indexing.indexer import IndexResult

        doc_id = "b" * 64
        _, job = _create_db_records(doc_repo, job_repo, doc_id, str(DAA_PDF))

        mock_result = IndexResult(
            doc_id=doc_id, success=True,
            chunks_indexed=3, parents_indexed=2, errors=[],
        )
        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_result,
        ):
            ingest_single_document.delay(doc_id, str(DAA_PDF), job.id)

        updated_job = job_repo.get_job(job.id)
        assert updated_job.processed_documents == 1
        assert updated_job.failed_documents == 0


# ---------------------------------------------------------------------------
# ingest_single_document — failure paths (call run() directly with mock self)
# ---------------------------------------------------------------------------

    def test_invalid_file_path_marks_failed(self, doc_repo, job_repo):
        """
        A missing file should mark the document as FAILED.
        We call the task's run() directly with a mocked self whose
        request.retries == max_retries so the permanent-failure branch executes.
        """
        from backend.workers.tasks import ingest_single_document
        from backend.db.models import DocumentStatus

        doc_id = "c" * 64
        bad_path = "/nonexistent/file.pdf"
        _, job = _create_db_records(doc_repo, job_repo, doc_id, bad_path)

        mock_self = _make_mock_self()
        # Call the underlying function directly, bypassing Celery machinery
        result = ingest_single_document.run.__func__(
            mock_self, doc_id, bad_path, job.id
        )

        assert result["status"] == "failed"
        doc = doc_repo.get_by_id(doc_id)
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message is not None

    def test_indexing_failure_marks_failed(self, doc_repo, job_repo):
        """Weaviate indexing failure should mark the document FAILED."""
        from backend.workers.tasks import ingest_single_document
        from backend.db.models import DocumentStatus
        from backend.indexing.indexer import IndexResult

        if not DAA_PDF.exists():
            pytest.skip("DAA.pdf not found")

        doc_id = "d" * 64
        _, job = _create_db_records(doc_repo, job_repo, doc_id, str(DAA_PDF))

        mock_result = IndexResult(
            doc_id=doc_id, success=False,
            chunks_indexed=0, parents_indexed=0,
            errors=["Weaviate connection refused"],
        )
        mock_self = _make_mock_self()

        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_result,
        ):
            result = ingest_single_document.run.__func__(
                mock_self, doc_id, str(DAA_PDF), job.id
            )

        assert result["status"] == "failed"
        doc = doc_repo.get_by_id(doc_id)
        assert doc.status == DocumentStatus.FAILED

    def test_failed_doc_increments_job_failed_counter(self, doc_repo, job_repo):
        """Failed document must increment job.failed_documents."""
        from backend.workers.tasks import ingest_single_document

        doc_id = "e" * 64
        _, job = _create_db_records(doc_repo, job_repo, doc_id, "/bad/path.pdf")

        mock_self = _make_mock_self()
        ingest_single_document.run.__func__(mock_self, doc_id, "/bad/path.pdf", job.id)

        updated = job_repo.get_job(job.id)
        assert updated.failed_documents == 1


# ---------------------------------------------------------------------------
# ingest_batch_documents
# ---------------------------------------------------------------------------

class TestIngestBatchDocuments:

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_batch_dispatches_subtasks(self, doc_repo, job_repo):
        """Batch task should fan-out to individual ingest tasks."""
        from backend.workers.tasks import ingest_batch_documents
        from backend.db.models import DocumentStatus
        from backend.indexing.indexer import IndexResult

        doc_ids = ["1" * 64, "2" * 64, "3" * 64]
        items = [{"doc_id": did, "file_path": str(DAA_PDF)} for did in doc_ids]

        for did in doc_ids:
            _create_db_records(doc_repo, job_repo, did, str(DAA_PDF))

        job = job_repo.create_job(doc_ids, queue="batch")

        mock_result = IndexResult(
            doc_id="any", success=True,
            chunks_indexed=3, parents_indexed=2, errors=[],
        )
        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_result,
        ):
            result = ingest_batch_documents.delay(items, job.id)

        output = result.get()
        assert output["total_dispatched"] == 3
        assert output["status"] == "dispatched"

    def test_batch_task_name(self):
        from backend.workers.tasks import ingest_batch_documents
        assert ingest_batch_documents.name == "backend.workers.tasks.ingest_batch_documents"

    def test_single_task_name(self):
        from backend.workers.tasks import ingest_single_document
        assert ingest_single_document.name == "backend.workers.tasks.ingest_single_document"


# ---------------------------------------------------------------------------
# Celery app configuration
# ---------------------------------------------------------------------------

class TestCeleryAppConfig:

    def test_both_queues_configured(self):
        from backend.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert "backend.workers.tasks.ingest_single_document" in routes
        assert "backend.workers.tasks.ingest_batch_documents" in routes

    def test_time_limits(self):
        from backend.workers.celery_app import celery_app
        assert celery_app.conf.task_time_limit == 600
        assert celery_app.conf.task_soft_time_limit == 540

    def test_prefetch_multiplier(self):
        from backend.workers.celery_app import celery_app
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_acks_late(self):
        from backend.workers.celery_app import celery_app
        assert celery_app.conf.task_acks_late is True
