"""
tests/unit/test_config.py
Unit tests for backend/core/config.py
"""

import os
import pytest
from unittest.mock import patch


def test_settings_load_with_defaults():
    """Settings should load with sensible defaults without a .env file."""
    # Import fresh — lru_cache means we test the cached singleton
    from backend.core.config import settings

    assert settings.WEAVIATE_URL == "http://localhost:8080"
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
    assert settings.EMBEDDING_BATCH_SIZE == 32
    assert settings.PARENT_CHUNK_SIZE == 1024
    assert settings.CHILD_CHUNK_SIZE == 256
    assert settings.CHUNK_OVERLAP == 51
    assert settings.MAX_KEYWORDS == 10
    assert settings.WEAVIATE_BATCH_SIZE == 100
    assert settings.MAX_FILE_SIZE_MB == 100


def test_settings_embedding_device_is_valid():
    """EMBEDDING_DEVICE must be either 'cuda' or 'cpu' after validation."""
    from backend.core.config import settings

    assert settings.EMBEDDING_DEVICE in ("cuda", "cpu")


def test_settings_max_file_size_bytes():
    """max_file_size_bytes property must equal MAX_FILE_SIZE_MB * 1024 * 1024."""
    from backend.core.config import settings

    assert settings.max_file_size_bytes == settings.MAX_FILE_SIZE_MB * 1024 * 1024


def test_settings_upload_path_is_path_object():
    """upload_path property should return a Path object."""
    from pathlib import Path
    from backend.core.config import settings

    assert isinstance(settings.upload_path, Path)


def test_env_override(tmp_path, monkeypatch):
    """Environment variables should override defaults."""
    monkeypatch.setenv("WEAVIATE_URL", "http://custom-host:9090")
    monkeypatch.setenv("MAX_KEYWORDS", "5")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")

    # Create a fresh Settings instance (bypass lru_cache)
    from backend.core.config import Settings
    fresh_settings = Settings()

    assert fresh_settings.WEAVIATE_URL == "http://custom-host:9090"
    assert fresh_settings.MAX_KEYWORDS == 5
    assert fresh_settings.EMBEDDING_DEVICE == "cpu"


def test_cuda_fallback_to_cpu(monkeypatch):
    """If CUDA is not available, EMBEDDING_DEVICE should fall back to cpu."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

    from backend.core.config import Settings
    fresh_settings = Settings()

    assert fresh_settings.EMBEDDING_DEVICE == "cpu"
