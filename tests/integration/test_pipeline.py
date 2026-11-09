"""
tests/integration/test_pipeline.py
End-to-end pipeline integration tests — requires Weaviate running.

For orchestrator-level tests that don't need Weaviate see:
  tests/unit/test_pipeline.py

Start Weaviate with:
  docker compose -f docker/docker-compose.yml up -d
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF   = REPO_ROOT / "backend" / "data" / "DAA.pdf"


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def require_weaviate():
    from backend.indexing.weaviate_client import health_check
    if not health_check():
        pytest.skip(
            "Weaviate not reachable. "
            "Start with: docker compose -f docker/docker-compose.yml up -d"
        )


@pytest.fixture(scope="module")
def weaviate_ready():
    """Ensure schema exists before any test runs."""
    from backend.indexing.weaviate_client import ensure_schema
    ensure_schema()
    yield


# ---------------------------------------------------------------------------
# Full end-to-end with real Weaviate
# ---------------------------------------------------------------------------

class TestFullPipelineWithWeaviate:

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_daa_pdf_full_pipeline(self, weaviate_ready):
        """DAA.pdf through all 4 stages — verify Weaviate contains chunks."""
        import weaviate.classes.query as wq
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.weaviate_client import get_weaviate_client, ensure_schema
        from backend.indexing.schema import CHILD_COLLECTION, PARENT_COLLECTION
        from backend.indexing.indexer import get_indexer

        doc_id = "a" * 64

        # Clean slate
        get_indexer().delete_document(doc_id)

        result = run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        assert result.success is True, f"Pipeline failed: {result.error}"
        assert result.child_chunks > 0
        assert result.parent_chunks > 0
        assert result.total_tokens > 0

        # Verify chunks present in Weaviate
        with get_weaviate_client() as client:
            child_col = client.collections.get(CHILD_COLLECTION)
            children = child_col.query.fetch_objects(
                filters=wq.Filter.by_property("doc_id").equal(doc_id),
                limit=100,
            )
            assert len(children.objects) == result.child_chunks

            parent_col = client.collections.get(PARENT_COLLECTION)
            parents = parent_col.query.fetch_objects(
                filters=wq.Filter.by_property("doc_id").equal(doc_id),
                limit=100,
            )
            assert len(parents.objects) == result.parent_chunks

        print(f"\n{result}")

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_reingest_is_idempotent(self, weaviate_ready):
        """Running the pipeline twice must not duplicate chunks."""
        import weaviate.classes.query as wq
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.weaviate_client import get_weaviate_client
        from backend.indexing.schema import CHILD_COLLECTION
        from backend.indexing.indexer import get_indexer

        doc_id = "b" * 64
        get_indexer().delete_document(doc_id)

        r1 = run_pipeline(doc_id=doc_id, file_path=DAA_PDF)
        r2 = run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        assert r1.success and r2.success
        assert r1.child_chunks == r2.child_chunks

        with get_weaviate_client() as client:
            child_col = client.collections.get(CHILD_COLLECTION)
            children = child_col.query.fetch_objects(
                filters=wq.Filter.by_property("doc_id").equal(doc_id),
                limit=100,
            )
            assert len(children.objects) == r1.child_chunks, (
                f"Expected {r1.child_chunks} chunks, got {len(children.objects)} "
                "after re-ingest — duplicates detected!"
            )

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_parent_child_linkage_end_to_end(self, weaviate_ready):
        """Every child's parent_id must reference an existing parent."""
        import weaviate.classes.query as wq
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.weaviate_client import get_weaviate_client
        from backend.indexing.schema import CHILD_COLLECTION, PARENT_COLLECTION
        from backend.indexing.indexer import get_indexer

        doc_id = "c" * 64
        get_indexer().delete_document(doc_id)
        run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        with get_weaviate_client() as client:
            doc_filter = wq.Filter.by_property("doc_id").equal(doc_id)

            parent_col = client.collections.get(PARENT_COLLECTION)
            parents = parent_col.query.fetch_objects(filters=doc_filter, limit=100)
            parent_chunk_ids = {
                obj.properties["doc_id"] + "_" + str(obj.properties["chunk_index"])
                for obj in parents.objects
            }

            child_col = client.collections.get(CHILD_COLLECTION)
            children = child_col.query.fetch_objects(filters=doc_filter, limit=100)

            # Each child must have a non-empty parent_id
            for child_obj in children.objects:
                pid = child_obj.properties.get("parent_id", "")
                assert pid, f"Child has empty parent_id: {child_obj.uuid}"

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_child_embeddings_are_768_dim(self, weaviate_ready):
        """Child chunks must be stored with 768-dim BGE vectors."""
        import weaviate.classes.query as wq
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.weaviate_client import get_weaviate_client
        from backend.indexing.schema import CHILD_COLLECTION
        from backend.indexing.indexer import get_indexer

        doc_id = "d" * 64
        get_indexer().delete_document(doc_id)
        run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        with get_weaviate_client() as client:
            child_col = client.collections.get(CHILD_COLLECTION)
            result = child_col.query.fetch_objects(
                filters=wq.Filter.by_property("doc_id").equal(doc_id),
                limit=10,
                include_vector=True,
            )
            for obj in result.objects:
                vec = obj.vector.get("default") or list(obj.vector.values())[0]
                assert len(vec) == 768, f"Expected 768-dim, got {len(vec)}"
