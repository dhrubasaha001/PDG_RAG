"""
backend/core/config.py
Centralised, typed configuration via pydantic-settings.
All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

import torch
from functools import lru_cache
from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Weaviate
    # ------------------------------------------------------------------ #
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_BATCH_SIZE: int = 100

    # ------------------------------------------------------------------ #
    # Redis / Celery
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_HIGH_PRIORITY_QUEUE: str = "high_priority"
    CELERY_BATCH_QUEUE: str = "batch"

    # ------------------------------------------------------------------ #
    # Embedding model
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32
    # "cuda" auto-falls back to "cpu" if CUDA is unavailable — see validator
    EMBEDDING_DEVICE: str = "cuda"

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    PARENT_CHUNK_SIZE: int = 1024   # tokens
    CHILD_CHUNK_SIZE: int = 256     # tokens
    CHUNK_OVERLAP: int = 51         # tokens (~20 % of child chunk size)

    # ------------------------------------------------------------------ #
    # Enrichment
    # ------------------------------------------------------------------ #
    MAX_KEYWORDS: int = 10

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    UPLOAD_DIR: str = "data/uploads"
    DB_URL: str = "sqlite:///./data/pdg_rag.db"

    # ------------------------------------------------------------------ #
    # File validation
    # ------------------------------------------------------------------ #
    MAX_FILE_SIZE_MB: int = 100

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"   # "json" | "console"

    # ------------------------------------------------------------------ #
    # pydantic-settings config
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("EMBEDDING_DEVICE", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Auto-fall back to CPU when CUDA is not available."""
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return v

    @model_validator(mode="after")
    def ensure_dirs(self) -> "Settings":
        """Create upload and data directories if they don't exist."""
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------------ #
    # Derived helpers (not stored as fields)
    # ------------------------------------------------------------------ #
    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level singleton — import this everywhere
settings = get_settings()
