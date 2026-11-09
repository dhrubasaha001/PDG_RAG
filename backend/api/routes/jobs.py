"""
backend/api/routes/jobs.py
Job status endpoints.

GET /jobs/{job_id}   — single job status + progress
GET /jobs            — paginated job list
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.dependencies import get_job_repo
from backend.db.repository import JobRepository
from backend.schemas import JobListResponse, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status.value,
        total_documents=job.total_documents,
        processed_documents=job.processed_documents,
        failed_documents=job.failed_documents,
        queue=job.queue,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    job_repo: JobRepository = Depends(get_job_repo),
) -> JobResponse:
    """Return status and progress for a specific ingestion job."""
    job = job_repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return _job_to_response(job)


@router.get("", response_model=JobListResponse)
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    job_repo: JobRepository = Depends(get_job_repo),
) -> JobListResponse:
    """Return a paginated list of all ingestion jobs."""
    jobs, total = job_repo.list_jobs(page=page, page_size=page_size)
    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )
