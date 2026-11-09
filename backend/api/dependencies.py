"""
backend/api/dependencies.py
Shared FastAPI dependencies injected via Depends().
"""

from __future__ import annotations

from backend.db.repository import DocumentRepository, JobRepository


def get_doc_repo() -> DocumentRepository:
    return DocumentRepository()


def get_job_repo() -> JobRepository:
    return JobRepository()
