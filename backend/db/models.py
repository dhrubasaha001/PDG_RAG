"""
backend/db/models.py
Plain data classes for the document registry.
These are not ORM models — they map 1-to-1 to the sqlite3 tables defined
in database.py and used by repository.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    INDEXED    = "indexed"
    FAILED     = "failed"
    DUPLICATE  = "duplicate"


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    PARTIAL    = "partial"    # some docs failed, some succeeded


@dataclass
class Document:
    """Represents one ingested document tracked in the registry."""

    id: str                             # doc_id = SHA-256 of file content
    filename: str
    file_hash: str                      # same as id; kept explicit for queries
    format: str                         # pdf | docx | pptx | xlsx | md | html | txt
    file_size_bytes: int
    file_path: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: Optional[int] = None
    parent_chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None


@dataclass
class IngestionJob:
    """Tracks a single ingestion job (one upload or one batch)."""

    id: str                             # UUID4
    status: JobStatus = JobStatus.QUEUED
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    queue: str = "high_priority"        # high_priority | batch
    celery_task_id: Optional[str] = None
    document_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
