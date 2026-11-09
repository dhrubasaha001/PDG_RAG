"""
backend/workers/celery_app.py
Celery application configuration for PDG_RAG.

Two queues:
  high_priority — real-time single-file uploads (fast turnaround)
  batch         — bulk ingestion jobs (throughput-optimised)

Windows note:
  Celery on Windows requires --pool=solo or --pool=gevent.
  Start workers with:
    celery -A backend.workers.celery_app worker -Q high_priority,batch --pool=solo -l info

  For production Linux:
    celery -A backend.workers.celery_app worker -Q high_priority,batch --concurrency=4 -l info
"""

from __future__ import annotations

from celery import Celery

from backend.core.config import settings

# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

celery_app = Celery(
    "pdg_rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["backend.workers.tasks"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Routing — task → queue mapping
    task_routes={
        "backend.workers.tasks.ingest_single_document": {
            "queue": settings.CELERY_HIGH_PRIORITY_QUEUE,
        },
        "backend.workers.tasks.ingest_batch_documents": {
            "queue": settings.CELERY_BATCH_QUEUE,
        },
    },

    # Time limits per task
    task_time_limit=600,          # 10 min hard kill
    task_soft_time_limit=540,     # 9 min → SoftTimeLimitExceeded raised in task

    # Memory / concurrency safety
    worker_prefetch_multiplier=1, # one task at a time per worker slot
    task_acks_late=True,          # ack only after the task completes (safe retry)

    # Result expiry
    result_expires=86_400,        # keep results for 24 hours

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
