"""
backend/schemas.py
Pydantic request/response schemas for the API layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Query (kept for backward compatibility with existing /ask endpoint)
# ---------------------------------------------------------------------------

class Question(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Ingest responses
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str                 # "queued" | "duplicate"
    message: str


class BatchUploadResponse(BaseModel):
    job_id: str
    doc_ids: list[str]
    total: int
    duplicates: int
    status: str
    message: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    format: str
    file_size_bytes: int
    status: str
    chunk_count: Optional[int] = None
    parent_chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Job responses
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    queue: str
    celery_task_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    page: int
    page_size: int
