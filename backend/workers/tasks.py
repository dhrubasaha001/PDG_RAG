"""
backend/workers/tasks.py
Celery task definitions for document ingestion.

Tasks
-----
ingest_single_document(doc_id, file_path, job_id)
    Full pipeline: parse → chunk → enrich → index → update registry.
    Queue: high_priority
    Retries: 3 × exponential backoff (1 min, 2 min, 4 min)

ingest_batch_documents(items, job_id)
    Fan-out: creates a Celery group of ingest_single_document tasks.
    Queue: batch
    items: list of {"doc_id": str, "file_path": str}
"""

from __future__ import annotations

from typing import Any

from celery import group
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger

from backend.workers.celery_app import celery_app
from backend.core.logging import get_logger

task_logger = get_task_logger(__name__)
logger = get_logger("workers.tasks")


# ---------------------------------------------------------------------------
# Single-document ingestion task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="backend.workers.tasks.ingest_single_document",
    max_retries=3,
    queue="high_priority",
    acks_late=True,
)
def ingest_single_document(
    self,
    doc_id: str,
    file_path: str,
    job_id: str,
) -> dict[str, Any]:
    """
    Run the full ingestion pipeline for a single document.

    Steps:
      1. Mark document PROCESSING in registry
      2. Parse  → ParsedDocument
      3. Chunk  → ChunkedDocument
      4. Enrich → EnrichedDocument
      5. Index  → IndexResult (Weaviate)
      6. Mark document INDEXED, update chunk counts
      7. Update job progress counter

    On failure:
      - Marks document FAILED with error message
      - Retries with exponential backoff (60s × 2^n)
      - After max retries, marks document FAILED permanently

    Returns:
        dict with doc_id, status, chunk_count, duration_seconds
    """
    import time
    start = time.perf_counter()

    logger.info(
        "task_started",
        doc_id=doc_id,
        job_id=job_id,
        attempt=self.request.retries + 1,
    )

    doc_repo, job_repo = _get_repos()

    # ---- Step 1: mark PROCESSING ----------------------------------------
    doc_repo.update_status(doc_id, _DocumentStatus().PROCESSING)

    try:
        # ---- Steps 2-5: full pipeline via orchestrator -------------------
        from backend.ingestion.pipeline import run_pipeline
        pipeline_result = run_pipeline(doc_id=doc_id, file_path=file_path)

        if not pipeline_result.success:
            raise RuntimeError(pipeline_result.error)

        # ---- Step 6: mark INDEXED ----------------------------------------
        doc_repo.update_status(
            doc_id,
            _DocumentStatus().INDEXED,
            chunk_count=pipeline_result.child_chunks,
            parent_chunk_count=pipeline_result.parent_chunks,
        )

        # ---- Step 7: update job ------------------------------------------
        job_repo.update_job_progress(job_id, processed_delta=1)

        elapsed = round(time.perf_counter() - start, 3)
        logger.info(
            "task_completed",
            doc_id=doc_id,
            job_id=job_id,
            chunks=pipeline_result.child_chunks,
            duration_seconds=elapsed,
        )

        return {
            "doc_id": doc_id,
            "status": "indexed",
            "chunk_count": pipeline_result.child_chunks,
            "parent_count": pipeline_result.parent_chunks,
            "duration_seconds": elapsed,
        }

    except SoftTimeLimitExceeded:
        _handle_timeout(doc_id, job_id, doc_repo, job_repo)
        raise   # re-raise so Celery records the failure

    except Exception as exc:
        error_msg = str(exc)
        logger.warning(
            "task_failed",
            doc_id=doc_id,
            job_id=job_id,
            attempt=self.request.retries + 1,
            max_retries=self.max_retries,
            error=error_msg,
        )

        if self.request.retries >= self.max_retries:
            # Final failure — mark permanently failed
            doc_repo.update_status(
                doc_id,
                _DocumentStatus().FAILED,
                error_message=f"Failed after {self.max_retries + 1} attempts: {error_msg}",
            )
            job_repo.update_job_progress(job_id, failed_delta=1)
            logger.error(
                "task_permanently_failed",
                doc_id=doc_id,
                job_id=job_id,
                error=error_msg,
            )
            return {"doc_id": doc_id, "status": "failed", "error": error_msg}

        # Exponential backoff: 60s, 120s, 240s
        countdown = 60 * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


# ---------------------------------------------------------------------------
# Batch ingestion task (fan-out)
# ---------------------------------------------------------------------------

@celery_app.task(
    name="backend.workers.tasks.ingest_batch_documents",
    queue="batch",
)
def ingest_batch_documents(
    items: list[dict[str, str]],
    job_id: str,
) -> dict[str, Any]:
    """
    Fan out a list of documents to individual ingest_single_document tasks.

    Args:
        items: List of {"doc_id": str, "file_path": str}
        job_id: The IngestionJob id to track progress against

    Returns:
        dict with job_id and total tasks dispatched
    """
    logger.info(
        "batch_task_dispatching",
        job_id=job_id,
        total=len(items),
    )

    # Update job status to PROCESSING
    _, job_repo = _get_repos()
    job_repo.update_job_progress(job_id, status=_JobStatus().PROCESSING)

    # Fan out — each doc gets its own high-priority task
    subtask_group = group(
        ingest_single_document.s(
            item["doc_id"],
            item["file_path"],
            job_id,
        )
        for item in items
    )
    result = subtask_group.apply_async(queue="high_priority")

    logger.info(
        "batch_tasks_dispatched",
        job_id=job_id,
        total=len(items),
        group_id=result.id if hasattr(result, "id") else None,
    )

    return {
        "job_id": job_id,
        "total_dispatched": len(items),
        "status": "dispatched",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_repos():
    """Return (DocumentRepository, JobRepository) — imported lazily."""
    from backend.db.repository import DocumentRepository, JobRepository
    return DocumentRepository(), JobRepository()


def _handle_timeout(doc_id: str, job_id: str, doc_repo, job_repo) -> None:
    doc_repo.update_status(
        doc_id,
        _DocumentStatus().FAILED,
        error_message="Task timed out after 9 minutes",
    )
    job_repo.update_job_progress(job_id, failed_delta=1)
    logger.error("task_timed_out", doc_id=doc_id, job_id=job_id)


# Lazy enum imports (avoids circular import at module load time)
class _DocumentStatus:
    def __getattr__(self, name):
        from backend.db.models import DocumentStatus
        return getattr(DocumentStatus, name)


class _JobStatus:
    def __getattr__(self, name):
        from backend.db.models import JobStatus
        return getattr(JobStatus, name)
