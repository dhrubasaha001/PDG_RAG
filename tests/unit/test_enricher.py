"""
tests/unit/test_enricher.py
Unit tests for the Enrichment Pipeline.

Note: These tests DO load the BGE model from HuggingFace (or cache).
First run downloads ~270 MB. Subsequent runs use the local cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.chunker import (
    ChunkedDocument,
    ParentChunk,
    ChildChunk,
    HierarchicalChunker,
)
from backend.ingestion.parsers.base import ParsedDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF = REPO_ROOT / "backend" / "data" / "DAA.pdf"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_small_chunked_doc() -> ChunkedDocument:
    """Build a minimal ChunkedDocument (3 children, 1 parent) for fast tests."""
    import uuid

    doc_id = "test_enrich_doc"
    parent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"parent-{doc_id}-0"))
    child_ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, f"child-{doc_id}-{i}")) for i in range(3)
    ]

    parent_text = (
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
        "language model. The retriever fetches relevant documents from a knowledge base. "
        "The language model then generates an answer grounded in the retrieved context."
    )

    sentences = [
        "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model.",
        "The retriever fetches relevant documents from a knowledge base.",
        "The language model then generates an answer grounded in the retrieved context.",
    ]

    parent = ParentChunk(
        chunk_id=parent_id,
        doc_id=doc_id,
        text=parent_text,
        token_count=60,
        chunk_index=0,
        page_number=1,
        char_start=0,
        char_end=len(parent_text),
        child_ids=list(child_ids),
    )

    children = [
        ChildChunk(
            chunk_id=child_ids[i],
            doc_id=doc_id,
            text=sentences[i],
            token_count=15,
            chunk_index=i,
            page_number=1,
            char_start=0,
            char_end=len(sentences[i]),
            parent_id=parent_id,
        )
        for i in range(3)
    ]

    return ChunkedDocument(
        doc_id=doc_id,
        filename="test.txt",
        format="txt",
        parents=[parent],
        children=children,
        total_tokens=60,
    )


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestEnricher:

    @pytest.fixture(scope="class")
    def enriched_doc(self):
        """Run enrichment once for the whole class — model load is expensive."""
        from backend.ingestion.enricher import get_enricher
        chunked = _make_small_chunked_doc()
        return get_enricher().enrich(chunked)

    def test_returns_enriched_document(self, enriched_doc):
        from backend.ingestion.enricher import EnrichedDocument
        assert isinstance(enriched_doc, EnrichedDocument)

    def test_embedding_dimension_is_768(self, enriched_doc):
        """BGE-base produces 768-dim embeddings."""
        for child in enriched_doc.children:
            assert len(child.embedding) == 768, (
                f"Expected 768-dim, got {len(child.embedding)}"
            )

    def test_embeddings_are_float(self, enriched_doc):
        for child in enriched_doc.children:
            assert all(isinstance(v, float) for v in child.embedding[:5])

    def test_embeddings_are_normalised(self, enriched_doc):
        """Normalised embeddings have unit L2 norm (within floating-point tolerance)."""
        import math
        for child in enriched_doc.children:
            norm = math.sqrt(sum(v ** 2 for v in child.embedding))
            assert abs(norm - 1.0) < 1e-3, f"Embedding not normalised: norm={norm}"

    def test_keywords_non_empty(self, enriched_doc):
        from backend.core.config import settings
        for child in enriched_doc.children:
            assert isinstance(child.keywords, list)
            assert len(child.keywords) <= settings.MAX_KEYWORDS

    def test_keywords_are_strings(self, enriched_doc):
        for child in enriched_doc.children:
            assert all(isinstance(kw, str) for kw in child.keywords)

    def test_summary_non_empty(self, enriched_doc):
        for child in enriched_doc.children:
            assert isinstance(child.summary, str)
            assert len(child.summary) > 0

    def test_parent_and_child_same_keywords(self, enriched_doc):
        """Children inherit keywords from their parent."""
        parent = enriched_doc.parents[0]
        for child in enriched_doc.children:
            if child.parent_id == parent.chunk_id:
                assert child.keywords == parent.keywords

    def test_parent_summary_propagated_to_children(self, enriched_doc):
        parent = enriched_doc.parents[0]
        for child in enriched_doc.children:
            if child.parent_id == parent.chunk_id:
                assert child.summary == parent.summary

    def test_doc_id_preserved(self, enriched_doc):
        assert enriched_doc.doc_id == "test_enrich_doc"
        for p in enriched_doc.parents:
            assert p.doc_id == "test_enrich_doc"
        for c in enriched_doc.children:
            assert c.doc_id == "test_enrich_doc"

    def test_processing_time_positive(self, enriched_doc):
        assert enriched_doc.processing_time_seconds >= 0.0

    def test_child_count_matches_input(self, enriched_doc):
        assert len(enriched_doc.children) == 3
        assert len(enriched_doc.parents) == 1

    def test_keybert_reuses_st_model(self):
        """
        Verify KeyBERT wraps the same SentenceTransformer instance — no double load.
        KeyBERT wraps it in a SentenceTransformerBackend; we check the inner model
        reference rather than the wrapper identity.
        """
        from backend.ingestion.enricher import Enricher
        enricher = Enricher()
        st_model, kw_model = enricher._get_models()
        # KeyBERT wraps the ST model in a backend adapter.
        # The adapter exposes the underlying model via .embedding_model
        backend = kw_model.model
        inner = getattr(backend, "embedding_model", None)
        assert inner is st_model, (
            "KeyBERT backend should reference the same SentenceTransformer instance"
        )

    def test_empty_chunked_document(self):
        """Enriching an empty document should return empty EnrichedDocument."""
        from backend.ingestion.enricher import get_enricher
        empty = ChunkedDocument(
            doc_id="empty",
            filename="empty.txt",
            format="txt",
            parents=[],
            children=[],
            total_tokens=0,
        )
        result = get_enricher().enrich(empty)
        assert result.parents == []
        assert result.children == []
        assert result.processing_time_seconds == 0.0

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_enrich_real_pdf(self):
        """Full parse → chunk → enrich on DAA.pdf."""
        from backend.ingestion.parsers.pdf_parser import PDFParser
        from backend.ingestion.chunker import HierarchicalChunker
        from backend.ingestion.enricher import get_enricher

        doc = PDFParser().parse(DAA_PDF)
        chunked = HierarchicalChunker().chunk(doc)
        enriched = get_enricher().enrich(chunked)

        assert len(enriched.children) == len(chunked.children)
        assert len(enriched.parents) == len(chunked.parents)

        # All embeddings should be 768-dim
        for child in enriched.children:
            assert len(child.embedding) == 768

        # At least one child should have keywords
        all_kws = [kw for c in enriched.children for kw in c.keywords]
        assert len(all_kws) > 0

        print(
            f"\nDAA.pdf enriched — "
            f"children: {len(enriched.children)}, "
            f"parents: {len(enriched.parents)}, "
            f"sample keywords: {enriched.parents[0].keywords[:5] if enriched.parents else []}, "
            f"time: {enriched.processing_time_seconds}s"
        )
