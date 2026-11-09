"""
backend/main.py
FastAPI application entry point for PDG_RAG.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Celery workers (separate terminal):
    # Windows:
    celery -A backend.workers.celery_app worker -Q high_priority,batch --pool=solo -l info

    # Linux / Docker:
    celery -A backend.workers.celery_app worker -Q high_priority,batch --concurrency=4 -l info

Flower monitoring (another terminal):
    celery -A backend.workers.celery_app flower --port=5555
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.indexing.weaviate_client import ensure_schema, health_check

logger = get_logger("api.main")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks, yield, then run shutdown tasks."""
    # ---- Startup --------------------------------------------------------
    logger.info("app_starting", version="1.0.0")

    # Ensure upload directory exists
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    # Initialise SQLite document registry
    from backend.db.database import init_db
    init_db()
    logger.info("db_ready")

    # Ensure Weaviate schema (non-blocking — logs warning if Weaviate is down)
    try:
        if health_check():
            ensure_schema()
            logger.info("weaviate_schema_ready")
        else:
            logger.warning(
                "weaviate_not_reachable",
                message="Weaviate is down — ingestion will fail until it is started.",
                hint="Run: docker compose -f docker/docker-compose.yml up -d",
            )
    except Exception as exc:
        logger.warning("weaviate_startup_check_failed", error=str(exc))

    logger.info("app_started")
    yield

    # ---- Shutdown -------------------------------------------------------
    logger.info("app_stopping")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDG_RAG — Production Document RAG API",
    description=(
        "A production-grade Retrieval-Augmented Generation API. "
        "Upload documents (PDF, DOCX, PPTX, XLSX, MD, HTML, TXT), "
        "index them into Weaviate via async Celery workers, "
        "then query the knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with method, path, and duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from backend.api.routes.ingest import router as ingest_router  # noqa: E402
from backend.api.routes.jobs import router as jobs_router       # noqa: E402
from backend.api.routes.query import router as query_router     # noqa: E402

app.include_router(ingest_router)
app.include_router(jobs_router)
app.include_router(query_router)


# ---------------------------------------------------------------------------
# Root + health endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    return {
        "service": "PDG_RAG",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    """Liveness + dependency health check."""
    weaviate_ok = health_check()

    # Redis health — try a quick ping
    redis_ok = False
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        redis_ok = r.ping()
        r.close()
    except Exception:
        pass

    overall = "healthy" if (weaviate_ok and redis_ok) else "degraded"

    return {
        "status": overall,
        "dependencies": {
            "weaviate": "up" if weaviate_ok else "down",
            "redis": "up" if redis_ok else "down",
        },
    }
