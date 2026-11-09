"""
tests/integration/test_weaviate.py
Integration tests for Weaviate schema + client.

These tests require a running Weaviate instance.
Start it with: docker compose -f docker/docker-compose.yml up -d

Tests are automatically skipped if Weaviate is not reachable.
"""

from __future__ import annotations

import pytest

from backend.indexing.weaviate_client import health_check
from backend.indexing.schema import PARENT_COLLECTION, CHILD_COLLECTION


# ---------------------------------------------------------------------------
# Skip fixture — skip all tests if Weaviate is not up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def require_weaviate():
    if not health_check():
        pytest.skip(
            "Weaviate not reachable at configured URL. "
            "Start with: docker compose -f docker/docker-compose.yml up -d"
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check_returns_true():
    assert health_check() is True


def test_health_check_returns_false_for_bad_url(monkeypatch):
    """A wrong URL should return False, not raise."""
    import backend.indexing.weaviate_client as wc_module
    monkeypatch.setattr(
        wc_module,
        "_parse_weaviate_url",
        lambda: ("localhost", 9999),  # nothing listening on 9999
    )
    assert health_check() is False


# ---------------------------------------------------------------------------
# Schema lifecycle
# ---------------------------------------------------------------------------

class TestSchemaLifecycle:

    def test_ensure_schema_creates_collections(self):
        """ensure_schema() should create both collections."""
        from backend.indexing.weaviate_client import (
            get_weaviate_client,
            ensure_schema,
            drop_schema,
        )

        with get_weaviate_client() as client:
            # Start clean
            drop_schema(client)

            ensure_schema(client)

            existing = {col.name for col in client.collections.list_all().values()}
            assert PARENT_COLLECTION in existing
            assert CHILD_COLLECTION in existing

    def test_ensure_schema_is_idempotent(self):
        """Calling ensure_schema() twice should not raise."""
        from backend.indexing.weaviate_client import (
            get_weaviate_client,
            ensure_schema,
        )

        with get_weaviate_client() as client:
            ensure_schema(client)   # first call
            ensure_schema(client)   # second call — must not raise or duplicate

            existing = {col.name for col in client.collections.list_all().values()}
            assert PARENT_COLLECTION in existing
            assert CHILD_COLLECTION in existing

    def test_parent_collection_has_expected_properties(self):
        """DocumentParent must have doc_id, text, summary, keywords."""
        from backend.indexing.weaviate_client import (
            get_weaviate_client,
            ensure_schema,
            get_parent_collection,
        )

        with get_weaviate_client() as client:
            ensure_schema(client)
            col = get_parent_collection(client)
            prop_names = {p.name for p in col.config.get().properties}

            for expected in ["doc_id", "text", "summary", "keywords", "chunk_index"]:
                assert expected in prop_names, f"Missing property '{expected}' in DocumentParent"

    def test_child_collection_has_expected_properties(self):
        """DocumentChunk must have doc_id, parent_id, text, keywords, summary."""
        from backend.indexing.weaviate_client import (
            get_weaviate_client,
            ensure_schema,
            get_child_collection,
        )

        with get_weaviate_client() as client:
            ensure_schema(client)
            col = get_child_collection(client)
            prop_names = {p.name for p in col.config.get().properties}

            for expected in ["doc_id", "parent_id", "text", "keywords", "summary", "token_count"]:
                assert expected in prop_names, f"Missing property '{expected}' in DocumentChunk"

    def test_drop_schema_removes_collections(self):
        """drop_schema() should remove both collections."""
        from backend.indexing.weaviate_client import (
            get_weaviate_client,
            ensure_schema,
            drop_schema,
        )

        with get_weaviate_client() as client:
            ensure_schema(client)
            drop_schema(client)

            existing = {col.name for col in client.collections.list_all().values()}
            assert PARENT_COLLECTION not in existing
            assert CHILD_COLLECTION not in existing

            # Restore for other tests
            ensure_schema(client)
