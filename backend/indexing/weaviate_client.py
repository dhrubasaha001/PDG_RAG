"""
backend/indexing/weaviate_client.py
Weaviate v4 connection management and schema lifecycle.

Usage
-----
# Option 1 — explicit lifecycle (recommended in long-running processes)
with get_weaviate_client() as client:
    ensure_schema(client)
    ...

# Option 2 — get a connected client without context manager
client = get_weaviate_client().__enter__()   # remember to call __exit__

# Option 3 — module-level helpers (manage connection internally)
ensure_schema()
health_check()
"""

from __future__ import annotations

import contextlib
from typing import Generator

import weaviate
from weaviate.exceptions import WeaviateConnectionError

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.indexing.schema import (
    CHILD_COLLECTION,
    PARENT_COLLECTION,
    child_collection_config,
    parent_collection_config,
)

logger = get_logger("indexing.weaviate_client")


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _parse_weaviate_url() -> tuple[str, int]:
    """
    Extract host and port from settings.WEAVIATE_URL.
    e.g. "http://localhost:8080"  →  ("localhost", 8080)
    """
    from urllib.parse import urlparse
    parsed = urlparse(settings.WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return host, port


@contextlib.contextmanager
def get_weaviate_client() -> Generator[weaviate.WeaviateClient, None, None]:
    """
    Context manager that yields a connected Weaviate v4 client.

    Raises:
        WeaviateConnectionError: If Weaviate is not reachable.
        RuntimeError: If the URL cannot be parsed.
    """
    host, port = _parse_weaviate_url()

    logger.debug("connecting_to_weaviate", host=host, port=port)

    client = weaviate.connect_to_local(host=host, port=port)

    if not client.is_ready():
        client.close()
        raise WeaviateConnectionError(
            f"Weaviate at {settings.WEAVIATE_URL} is not ready. "
            "Is the Docker container running? "
            "Run: docker compose -f docker/docker-compose.yml up -d"
        )

    logger.info("weaviate_connected", host=host, port=port)

    try:
        yield client
    finally:
        client.close()
        logger.debug("weaviate_disconnected", host=host, port=port)


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def ensure_schema(client: weaviate.WeaviateClient | None = None) -> None:
    """
    Idempotently create both Weaviate collections if they do not exist.

    Safe to call on every application startup — skips creation if the
    collection already exists.

    Args:
        client: An already-connected WeaviateClient.  If None, a new
                connection is opened and closed within this call.
    """
    def _create_if_missing(c: weaviate.WeaviateClient) -> None:
        existing = {col.name for col in c.collections.list_all().values()}  # type: ignore[attr-defined]

        for name, config_fn in [
            (PARENT_COLLECTION, parent_collection_config),
            (CHILD_COLLECTION, child_collection_config),
        ]:
            if name in existing:
                logger.debug("schema_collection_exists", collection=name)
            else:
                c.collections.create(**config_fn())
                logger.info("schema_collection_created", collection=name)

    if client is not None:
        _create_if_missing(client)
    else:
        with get_weaviate_client() as c:
            _create_if_missing(c)


def drop_schema(client: weaviate.WeaviateClient) -> None:
    """
    Delete both collections. Destructive — used in tests / re-index scenarios.

    Args:
        client: An already-connected WeaviateClient.
    """
    for name in [CHILD_COLLECTION, PARENT_COLLECTION]:
        try:
            client.collections.delete(name)
            logger.info("schema_collection_dropped", collection=name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "schema_collection_drop_failed",
                collection=name,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(client: weaviate.WeaviateClient | None = None) -> bool:
    """
    Return True if Weaviate is reachable and ready.

    Args:
        client: Optional already-connected client.  If None, a transient
                connection is opened.
    """
    try:
        if client is not None:
            return client.is_ready()

        host, port = _parse_weaviate_url()
        c = weaviate.connect_to_local(host=host, port=port)
        ready = c.is_ready()
        c.close()
        return ready
    except Exception as exc:  # noqa: BLE001
        logger.warning("weaviate_health_check_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Collection accessors (convenience)
# ---------------------------------------------------------------------------

def get_parent_collection(client: weaviate.WeaviateClient):
    """Return the DocumentParent collection object."""
    return client.collections.get(PARENT_COLLECTION)


def get_child_collection(client: weaviate.WeaviateClient):
    """Return the DocumentChunk collection object."""
    return client.collections.get(CHILD_COLLECTION)
