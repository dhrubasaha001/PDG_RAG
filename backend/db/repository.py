"""
backend/db/repository.py
CRUD operations for the document registry.

Two repository classes:
  DocumentRepository  — CRUD for the documents table
  JobRepository       — CRUD for the ingestion_jobs table

Both operate via the get_connection() context manager — no ORM, no SQLAlchemy.
All datetimes are stored as ISO-8601 strings in UTC and parsed back on read.
document_ids in IngestionJob are stored as a JSON array string.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from backend.core.logging import get_logger
from backend.db.database import get_connection
from backend.db.models import Document, DocumentStatus, IngestionJob, JobStatus

logger = get_logger("db.repository")


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# Row → dataclass converters
# ---------------------------------------------------------------------------

def _row_to_document(row) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        file_hash=row["file_hash"],
        format=row["format"],
        file_size_bytes=row["file_size_bytes"],
        file_path=row["file_path"],
        status=DocumentStatus(row["status"]),
        chunk_count=row["chunk_count"],
        parent_chunk_count=row["parent_chunk_count"],
        error_message=row["error_message"],
        created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
        updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
        indexed_at=_parse_dt(row["indexed_at"]),
    )


def _row_to_job(row) -> IngestionJob:
    return IngestionJob(
        id=row["id"],
        status=JobStatus(row["status"]),
        total_documents=row["total_documents"],
        processed_documents=row["processed_documents"],
        failed_documents=row["failed_documents"],
        queue=row["queue"],
        celery_task_id=row["celery_task_id"],
        document_ids=json.loads(row["document_ids"] or "[]"),
        created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
        completed_at=_parse_dt(row["completed_at"]),
    )


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------

class DocumentRepository:
    """CRUD for the documents table."""

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self, doc: Document) -> Document:
        """
        Insert a new document record.

        Raises:
            ValueError: If a document with the same file_hash already exists.
        """
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, filename, file_hash, format, file_size_bytes, file_path,
                     status, chunk_count, parent_chunk_count, error_message,
                     created_at, updated_at, indexed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    doc.id,
                    doc.filename,
                    doc.file_hash,
                    doc.format,
                    doc.file_size_bytes,
                    doc.file_path,
                    doc.status.value,
                    doc.chunk_count,
                    doc.parent_chunk_count,
                    doc.error_message,
                    now,
                    now,
                    doc.indexed_at.isoformat() if doc.indexed_at else None,
                ),
            )

        doc.created_at = _parse_dt(now)  # type: ignore[assignment]
        doc.updated_at = _parse_dt(now)  # type: ignore[assignment]

        logger.info("document_created", doc_id=doc.id, filename=doc.filename)
        return doc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Return document by id (SHA-256 hash), or None."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """
        Return document by file_hash.
        Used for duplicate detection: same hash → same content.
        """
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[DocumentStatus] = None,
    ) -> tuple[list[Document], int]:
        """
        Return a paginated list of documents and the total count.

        Args:
            page: 1-based page number.
            page_size: Records per page (max 100).
            status_filter: Optional status to filter on.

        Returns:
            (list_of_documents, total_count)
        """
        page_size = min(page_size, 100)
        offset = (page - 1) * page_size

        with get_connection() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status_filter.value, page_size, offset),
                ).fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE status = ?",
                    (status_filter.value,),
                ).fetchone()[0]
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC "
                    "LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) FROM documents"
                ).fetchone()[0]

        return [_row_to_document(r) for r in rows], total

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        *,
        chunk_count: Optional[int] = None,
        parent_chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Update document status and optional metadata.
        Sets indexed_at when status transitions to INDEXED.
        """
        now = _now_iso()
        indexed_at = now if status == DocumentStatus.INDEXED else None

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents SET
                    status              = ?,
                    chunk_count         = COALESCE(?, chunk_count),
                    parent_chunk_count  = COALESCE(?, parent_chunk_count),
                    error_message       = COALESCE(?, error_message),
                    updated_at          = ?,
                    indexed_at          = COALESCE(?, indexed_at)
                WHERE id = ?
                """,
                (
                    status.value,
                    chunk_count,
                    parent_chunk_count,
                    error_message,
                    now,
                    indexed_at,
                    doc_id,
                ),
            )

        logger.info("document_status_updated", doc_id=doc_id, status=status.value)
        return self.get_by_id(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document record. Returns True if a row was deleted."""
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ?", (doc_id,)
            )
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# JobRepository
# ---------------------------------------------------------------------------

class JobRepository:
    """CRUD for the ingestion_jobs table."""

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_job(
        self,
        document_ids: list[str],
        queue: str = "high_priority",
        celery_task_id: Optional[str] = None,
    ) -> IngestionJob:
        """Create and persist a new IngestionJob."""
        now = _now_iso()
        job = IngestionJob(
            id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            total_documents=len(document_ids),
            processed_documents=0,
            failed_documents=0,
            queue=queue,
            celery_task_id=celery_task_id,
            document_ids=document_ids,
            created_at=_parse_dt(now),  # type: ignore[assignment]
        )

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs
                    (id, status, total_documents, processed_documents,
                     failed_documents, queue, celery_task_id, document_ids,
                     created_at, completed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.id,
                    job.status.value,
                    job.total_documents,
                    job.processed_documents,
                    job.failed_documents,
                    job.queue,
                    job.celery_task_id,
                    json.dumps(job.document_ids),
                    now,
                    None,
                ),
            )

        logger.info("job_created", job_id=job.id, total=job.total_documents, queue=queue)
        return job

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Return job by id, or None."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[IngestionJob], int]:
        """Return paginated list of jobs and total count."""
        page_size = min(page_size, 100)
        offset = (page - 1) * page_size

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC "
                "LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM ingestion_jobs"
            ).fetchone()[0]

        return [_row_to_job(r) for r in rows], total

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_job_progress(
        self,
        job_id: str,
        *,
        processed_delta: int = 0,
        failed_delta: int = 0,
        status: Optional[JobStatus] = None,
        celery_task_id: Optional[str] = None,
    ) -> Optional[IngestionJob]:
        """
        Atomically increment processed/failed counters.
        Optionally sets status and celery_task_id.
        """
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs SET
                    processed_documents = processed_documents + ?,
                    failed_documents    = failed_documents + ?,
                    status              = COALESCE(?, status),
                    celery_task_id      = COALESCE(?, celery_task_id)
                WHERE id = ?
                """,
                (
                    processed_delta,
                    failed_delta,
                    status.value if status else None,
                    celery_task_id,
                    job_id,
                ),
            )

        return self.get_job(job_id)

    def complete_job(
        self,
        job_id: str,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> Optional[IngestionJob]:
        """Mark job as complete and record completed_at timestamp."""
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs SET
                    status       = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (status.value, now, job_id),
            )

        logger.info("job_completed", job_id=job_id, status=status.value)
        return self.get_job(job_id)

    def set_celery_task_id(self, job_id: str, celery_task_id: str) -> None:
        """Attach the Celery task ID to a job after dispatch."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE ingestion_jobs SET celery_task_id = ? WHERE id = ?",
                (celery_task_id, job_id),
            )
