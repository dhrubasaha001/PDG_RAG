"""
backend/db/database.py
SQLite connection management and schema bootstrap (no ORM, pure sqlite3).

Usage
-----
from backend.db.database import get_connection, init_db

init_db()   # call once at startup — idempotent, creates tables if absent

with get_connection() as conn:
    conn.execute("SELECT ...")
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Generator

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger("db.database")

# ---------------------------------------------------------------------------
# DDL — CREATE TABLE IF NOT EXISTS (idempotent)
# ---------------------------------------------------------------------------

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT    PRIMARY KEY,
    filename            TEXT    NOT NULL,
    file_hash           TEXT    NOT NULL UNIQUE,
    format              TEXT    NOT NULL,
    file_size_bytes     INTEGER NOT NULL DEFAULT 0,
    file_path           TEXT    NOT NULL,
    status              TEXT    NOT NULL DEFAULT 'pending',
    chunk_count         INTEGER,
    parent_chunk_count  INTEGER,
    error_message       TEXT,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    indexed_at          TEXT
);
"""

_CREATE_DOCUMENTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);
"""

_CREATE_DOCUMENTS_HASH_IDX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash
    ON documents (file_hash);
"""

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id                   TEXT    PRIMARY KEY,
    status               TEXT    NOT NULL DEFAULT 'queued',
    total_documents      INTEGER NOT NULL DEFAULT 0,
    processed_documents  INTEGER NOT NULL DEFAULT 0,
    failed_documents     INTEGER NOT NULL DEFAULT 0,
    queue                TEXT    NOT NULL DEFAULT 'high_priority',
    celery_task_id       TEXT,
    document_ids         TEXT    NOT NULL DEFAULT '[]',
    created_at           TEXT    NOT NULL,
    completed_at         TEXT
);
"""

_CREATE_JOBS_IDX = """
CREATE INDEX IF NOT EXISTS idx_jobs_status
    ON ingestion_jobs (status);
"""


# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    """
    Resolve the SQLite database file path from settings.DB_URL.
    Supports both:
      sqlite:///./data/pdg_rag.db   (relative)
      sqlite:////abs/path/db.sqlite (absolute)
    """
    url = settings.DB_URL
    # Strip the sqlite:/// prefix
    if url.startswith("sqlite:///"):
        path_str = url[len("sqlite:///"):]
    else:
        path_str = url

    path = Path(path_str)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding a sqlite3 Connection with WAL mode and
    row_factory set to sqlite3.Row for dict-like access.

    Always commits on clean exit, rolls back on exception.
    """
    db_path = _db_path()
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables and indexes if they don't exist.
    Safe to call multiple times (idempotent).
    """
    with get_connection() as conn:
        conn.execute(_CREATE_DOCUMENTS)
        conn.execute(_CREATE_DOCUMENTS_IDX)
        conn.execute(_CREATE_DOCUMENTS_HASH_IDX)
        conn.execute(_CREATE_JOBS)
        conn.execute(_CREATE_JOBS_IDX)

    logger.info("db_initialised", db_path=str(_db_path()))
