"""
tests/integration/test_indexer.py
Integration tests for WeaviateIndexer.

Requires Weaviate running:
  docker compose -f docker/docker-compose.yml up -d

All tests are skipped if Weaviate is unreachable.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from backend.indexing.weaviate_client import health_check
from backend.indexing.schema import PARENT_COLLECTION, CHILD_COLLECTION

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF = REPO_ROOT / "backend" / "data" / "DAA.pdf"


# ---------------------------------------------------------------------------
# Skip fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def require_weaviate():
    if not health_check():
        pytest.skip(
            "Weaviate not reachable. "
            "Start with: docker compose -f docker/docker-compose.yml up -d"
        )


# ---------------------------------------------------------------------------
# Shared enriched document fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def enriched_daa():
    """Parse + chunk + enrich DAA.pdf once for the whole module."""
    if not DAA_PDF.exists():
        pytest.skip("DAA.pdf not found")

    from backend.ingestion.parsers.pdf_parser import PDFParser
    from backend.ingestion.chunker import HierarchicalChunker
    from backend.ingestion.enricher import get_enricher

    doc = PDFParser().parse(DAA_PDF)
    chunked = HierarchicalChunker().chunk(doc)
    enriched = get_enricher().enrich(chunked)
    return enriched


@pytest.fixture(scope="module")
def weaviate_client_module():
    """Single Weaviate connection shared across the module."""
    from backend.indexing.weaviate_client import get_weaviate_client, ensure_schema
    with get_weaviate_client() as client:
        ensure_schema(client)
        yield client


# ---------------------------------------------------------------------------
# document_exists / delete_document
# ---------------------------------------------------------------------------

class TestDocumentExistsAndDelete:

    def test_document_does_not_exist_initially(self, enriched_daa, weaviate_client_module):
        from backend.indexing.indexer import get_indexer
        indexer = get_indexer()

        # Ensure clean state
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)
        assert indexer.document_exists(enriched_daa.doc_id, weaviate_client_module) is False

    def test_document_exists_after_index(self, enriched_daa, weaviate_client_module):
        from backend.indexing.indexer import get_indexer
        indexer = get_indexer()

        indexer.index_document(enriched_daa, weaviate_client_module)
        assert indexer.document_exists(enriched_daa.doc_id, weaviate_client_module) is True

    def test_delete_document_removes_all_chunks(self, enriched_daa, weaviate_client_module):
        from backend.indexing.indexer import get_indexer
        indexer = get_indexer()

        indexer.index_document(enriched_daa, weaviate_client_module)
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)
        assert indexer.document_exists(enriched_daa.doc_id, weaviate_client_module) is False


# ---------------------------------------------------------------------------
# index_document
# ---------------------------------------------------------------------------

class TestIndexDocument:

    def setup_method(self, _method):
        """Clean state before each test."""
        pass  # cleanup handled inside each test

    def test_index_document_returns_success(self, enriched_daa, weaviate_client_module):
        from backend.indexing.indexer import get_indexer, IndexResult
        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)

        result = indexer.index_document(enriched_daa, weaviate_client_module)

        assert isinstance(result, IndexResult)
        assert result.success is True
        assert result.doc_id == enriched_daa.doc_id
        assert result.errors == []

    def test_index_document_correct_chunk_counts(self, enriched_daa, weaviate_client_module):
        from backend.indexing.indexer import get_indexer
        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)

        result = indexer.index_document(enriched_daa, weaviate_client_module)

        assert result.chunks_indexed == len(enriched_daa.children)
        assert result.parents_indexed == len(enriched_daa.parents)

    def test_reindex_no_duplicates(self, enriched_daa, weaviate_client_module):
        """Re-indexing the same document must not create duplicate chunks."""
        from backend.indexing.indexer import get_indexer
        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)

        # First index
        indexer.index_document(enriched_daa, weaviate_client_module)

        # Second index (upsert)
        indexer.index_document(enriched_daa, weaviate_client_module)

        # Count children in Weaviate
        import weaviate.classes.query as wq
        child_col = weaviate_client_module.collections.get(CHILD_COLLECTION)
        result = child_col.query.fetch_objects(
            filters=wq.Filter.by_property("doc_id").equal(enriched_daa.doc_id),
            limit=1000,
        )
        assert len(result.objects) == len(enriched_daa.children), (
            f"Expected {len(enriched_daa.children)} children, "
            f"got {len(result.objects)} after re-index"
        )

    def test_chunks_have_correct_doc_id(self, enriched_daa, weaviate_client_module):
        """All indexed chunks must carry the correct doc_id."""
        from backend.indexing.indexer import get_indexer
        import weaviate.classes.query as wq

        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)
        indexer.index_document(enriched_daa, weaviate_client_module)

        child_col = weaviate_client_module.collections.get(CHILD_COLLECTION)
        result = child_col.query.fetch_objects(
            filters=wq.Filter.by_property("doc_id").equal(enriched_daa.doc_id),
            limit=1000,
        )
        for obj in result.objects:
            assert obj.properties["doc_id"] == enriched_daa.doc_id

    def test_chunks_have_embeddings(self, enriched_daa, weaviate_client_module):
        """Child chunks must be stored with their 768-dim vectors."""
        from backend.indexing.indexer import get_indexer, _child_uuid
        import weaviate.classes.query as wq

        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)
        indexer.index_document(enriched_daa, weaviate_client_module)

        child_col = weaviate_client_module.collections.get(CHILD_COLLECTION)
        # Fetch with vectors included
        result = child_col.query.fetch_objects(
            filters=wq.Filter.by_property("doc_id").equal(enriched_daa.doc_id),
            limit=10,
            include_vector=True,
        )
        for obj in result.objects:
            assert obj.vector is not None
            vec = obj.vector.get("default") or list(obj.vector.values())[0]
            assert len(vec) == 768, f"Expected 768-dim vector, got {len(vec)}"

    def test_parent_child_linkage_in_weaviate(self, enriched_daa, weaviate_client_module):
        """Each child's parent_id must match a real parent UUID in Weaviate."""
        from backend.indexing.indexer import get_indexer, _parent_uuid
        import weaviate.classes.query as wq

        indexer = get_indexer()
        indexer.delete_document(enriched_daa.doc_id, weaviate_client_module)
        indexer.index_document(enriched_daa, weaviate_client_module)

        # Build set of valid parent chunk_ids
        valid_parent_chunk_ids = {p.chunk_id for p in enriched_daa.parents}

        child_col = weaviate_client_module.collections.get(CHILD_COLLECTION)
        result = child_col.query.fetch_objects(
            filters=wq.Filter.by_property("doc_id").equal(enriched_daa.doc_id),
            limit=1000,
        )
        for obj in result.objects:
            assert obj.properties["parent_id"] in valid_parent_chunk_ids


# ---------------------------------------------------------------------------
# index_batch
# ---------------------------------------------------------------------------

class TestIndexBatch:

    def test_index_batch_three_docs(self, enriched_daa, weaviate_client_module):
        """index_batch should return one IndexResult per document."""
        from backend.indexing.indexer import get_indexer
        from backend.ingestion.enricher import EnrichedDocument

        indexer = get_indexer()

        # Create 3 shallow copies with distinct doc_ids for the batch test
        docs = []
        for i in range(3):
            import copy
            d = copy.deepcopy(enriched_daa)
            new_id = f"batch_test_doc_{i}"
            d.doc_id = new_id
            for p in d.parents:
                p.doc_id = new_id
            for c in d.children:
                c.doc_id = new_id
            docs.append(d)

        # Clean up first
        for d in docs:
            indexer.delete_document(d.doc_id, weaviate_client_module)

        results = indexer.index_batch(docs, weaviate_client_module)

        assert len(results) == 3
        for r in results:
            assert r.success is True
            assert r.chunks_indexed == len(enriched_daa.children)

        # Cleanup
        for d in docs:
            indexer.delete_document(d.doc_id, weaviate_client_module)
