"""
backend/api/routes/ingest.py
Document ingestion endpoints.

POST /ingest/upload       — single file upload (high_priority queue)
POST /ingest/batch        — multi-file batch upload (batch queue)
GET  /ingest/documents    — paginated document list with status
DELETE /ingest/documents/{doc_id} — remove document from index + registry
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from backend.api.dependencies import get_doc_repo, get_job_repo
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.db.models import Document, DocumentStatus
from backend.db.repository import DocumentRepository, JobRepository
from backend.ingestion.parsers.factory import ParserFactory
from backend.indexing.indexer import get_indexer
from backend.workers.tasks import ingest_batch_documents, ingest_single_document
from backend.schemas import (
    BatchUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    UploadResponse,
)

logger = get_logger("api.ingest")

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# ---------------------------------------------------------------------------
# Supported MIME types (validated before saving to disk)
# ---------------------------------------------------------------------------
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/plain",
    "text/html",
    "text/markdown",
    "text/csv",
    "application/octet-stream",  # some clients send this for PDFs
}

_MAX_BATCH_FILES = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile) -> None:
    """Raise HTTPException if file type or size is unacceptable."""
    content_type = (file.content_type or "").split(";")[0].strip()
    ext = Path(file.filename or "").suffix.lower()

    if not ParserFactory.is_supported(f"file{ext}"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(ParserFactory.supported_extensions())}"
            ),
        )


def _save_upload(file_bytes: bytes, filename: str, doc_id: str) -> Path:
    """Save uploaded bytes to UPLOAD_DIR/{doc_id}/{filename}. Returns the path."""
    dest_dir = settings.upload_path / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    dest.write_bytes(file_bytes)
    return dest


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _doc_to_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        format=doc.format,
        file_size_bytes=doc.file_size_bytes,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        parent_chunk_count=doc.parent_chunk_count,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        indexed_at=doc.indexed_at,
    )


# ---------------------------------------------------------------------------
# POST /ingest/upload — single file
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    doc_repo: DocumentRepository = Depends(get_doc_repo),
    job_repo: JobRepository = Depends(get_job_repo),
) -> UploadResponse:
    """
    Upload a single document for ingestion.

    Returns immediately with a job_id. Poll GET /jobs/{job_id} for status.
    """
    # --- Validate file type
    _validate_upload(file)

    # --- Read content (respects MAX_FILE_SIZE_MB)
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit "
                f"({len(file_bytes) // 1024 // 1024} MB uploaded)."
            ),
        )

    # --- Duplicate detection via SHA-256
    file_hash = _hash_bytes(file_bytes)
    existing = doc_repo.get_by_hash(file_hash)
    if existing:
        logger.info(
            "duplicate_upload_detected",
            filename=file.filename,
            existing_doc_id=existing.id,
            existing_status=existing.status.value,
        )
        return UploadResponse(
            job_id="",
            doc_id=existing.id,
            status="duplicate",
            message=(
                f"Document already exists (status: {existing.status.value}). "
                f"doc_id: {existing.id}"
            ),
        )

    # --- Determine format from extension
    ext = Path(file.filename or "file").suffix.lower().lstrip(".")
    if not ext:
        ext = "txt"

    # --- Save to disk
    doc_id = file_hash   # SHA-256 is the doc_id
    filename = file.filename or f"{doc_id}.bin"
    saved_path = _save_upload(file_bytes, filename, doc_id)

    # --- Create Document record (PENDING)
    doc = Document(
        id=doc_id,
        filename=filename,
        file_hash=file_hash,
        format=ext,
        file_size_bytes=len(file_bytes),
        file_path=str(saved_path),
        status=DocumentStatus.PENDING,
    )
    doc_repo.create(doc)

    # --- Create IngestionJob
    job = job_repo.create_job([doc_id], queue=settings.CELERY_HIGH_PRIORITY_QUEUE)

    # --- Dispatch Celery task
    task = ingest_single_document.apply_async(
        args=[doc_id, str(saved_path), job.id],
        queue=settings.CELERY_HIGH_PRIORITY_QUEUE,
    )
    job_repo.set_celery_task_id(job.id, task.id)

    logger.info(
        "upload_queued",
        doc_id=doc_id,
        filename=filename,
        job_id=job.id,
        task_id=task.id,
        file_size_bytes=len(file_bytes),
    )

    return UploadResponse(
        job_id=job.id,
        doc_id=doc_id,
        status="queued",
        message=f"Document '{filename}' queued for ingestion.",
    )


# ---------------------------------------------------------------------------
# POST /ingest/batch — multiple files
# ---------------------------------------------------------------------------

@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_batch(
    files: list[UploadFile] = File(...),
    doc_repo: DocumentRepository = Depends(get_doc_repo),
    job_repo: JobRepository = Depends(get_job_repo),
) -> BatchUploadResponse:
    """
    Upload multiple documents in one request (max 100 files).

    Returns a single job_id that tracks the whole batch.
    """
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch limit is {_MAX_BATCH_FILES} files. Got {len(files)}.",
        )

    new_doc_ids: list[str] = []
    duplicate_count = 0
    items_for_worker: list[dict] = []

    for file in files:
        _validate_upload(file)
        file_bytes = await file.read()

        if len(file_bytes) > settings.max_file_size_bytes:
            logger.warning(
                "batch_file_too_large",
                filename=file.filename,
                size_bytes=len(file_bytes),
            )
            continue  # skip oversized files silently, don't fail whole batch

        file_hash = _hash_bytes(file_bytes)
        existing = doc_repo.get_by_hash(file_hash)
        if existing:
            duplicate_count += 1
            continue

        ext = Path(file.filename or "file").suffix.lower().lstrip(".") or "txt"
        doc_id = file_hash
        filename = file.filename or f"{doc_id}.bin"
        saved_path = _save_upload(file_bytes, filename, doc_id)

        doc = Document(
            id=doc_id,
            filename=filename,
            file_hash=file_hash,
            format=ext,
            file_size_bytes=len(file_bytes),
            file_path=str(saved_path),
            status=DocumentStatus.PENDING,
        )
        doc_repo.create(doc)
        new_doc_ids.append(doc_id)
        items_for_worker.append({"doc_id": doc_id, "file_path": str(saved_path)})

    if not new_doc_ids:
        return BatchUploadResponse(
            job_id="",
            doc_ids=[],
            total=0,
            duplicates=duplicate_count,
            status="duplicate",
            message="All uploaded files were already indexed.",
        )

    job = job_repo.create_job(new_doc_ids, queue=settings.CELERY_BATCH_QUEUE)

    task = ingest_batch_documents.apply_async(
        args=[items_for_worker, job.id],
        queue=settings.CELERY_BATCH_QUEUE,
    )
    job_repo.set_celery_task_id(job.id, task.id)

    logger.info(
        "batch_queued",
        job_id=job.id,
        total_new=len(new_doc_ids),
        duplicates=duplicate_count,
    )

    return BatchUploadResponse(
        job_id=job.id,
        doc_ids=new_doc_ids,
        total=len(new_doc_ids),
        duplicates=duplicate_count,
        status="queued",
        message=(
            f"{len(new_doc_ids)} document(s) queued for ingestion"
            + (f", {duplicate_count} duplicate(s) skipped." if duplicate_count else ".")
        ),
    )


# ---------------------------------------------------------------------------
# GET /ingest/documents
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    doc_repo: DocumentRepository = Depends(get_doc_repo),
) -> DocumentListResponse:
    """List all documents with optional status filter and pagination."""
    status_enum = None
    if status_filter:
        try:
            status_enum = DocumentStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid status '{status_filter}'. "
                    f"Valid values: {[s.value for s in DocumentStatus]}"
                ),
            )

    docs, total = doc_repo.list_documents(page=page, page_size=page_size, status_filter=status_enum)
    return DocumentListResponse(
        documents=[_doc_to_response(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# DELETE /ingest/documents/{doc_id}
# ---------------------------------------------------------------------------

@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(
    doc_id: str,
    doc_repo: DocumentRepository = Depends(get_doc_repo),
) -> dict:
    """
    Delete a document from Weaviate, the file system, and the registry.
    """
    doc = doc_repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    # Delete from Weaviate (best-effort — may already not be indexed)
    try:
        get_indexer().delete_document(doc_id)
    except Exception as exc:
        logger.warning("weaviate_delete_failed", doc_id=doc_id, error=str(exc))

    # Delete file from disk
    try:
        upload_dir = settings.upload_path / doc_id
        if upload_dir.exists():
            shutil.rmtree(str(upload_dir))
    except Exception as exc:
        logger.warning("file_delete_failed", doc_id=doc_id, error=str(exc))

    # Delete registry record
    doc_repo.delete(doc_id)

    logger.info("document_deleted", doc_id=doc_id, filename=doc.filename)
    return {"doc_id": doc_id, "filename": doc.filename, "deleted": True}
