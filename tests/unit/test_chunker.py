"""
tests/unit/test_chunker.py
Unit tests for the HierarchicalChunker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingestion.chunker import (
    HierarchicalChunker,
    ChunkedDocument,
    ParentChunk,
    ChildChunk,
    count_tokens,
    split_into_sentences,
)
from backend.ingestion.parsers.base import ParsedDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF = REPO_ROOT / "backend" / "data" / "DAA.pdf"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "testdoc123") -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename="test.txt",
        format="txt",
        text=text,
        page_count=1,
        file_path="test.txt",
        file_size_bytes=len(text.encode()),
    )


def _chunker(parent=1024, child=256, overlap=51) -> HierarchicalChunker:
    return HierarchicalChunker(
        parent_chunk_size=parent,
        child_chunk_size=child,
        chunk_overlap=overlap,
    )


# -----------------------------------------------------------------------
# Sentence splitter
# -----------------------------------------------------------------------

class TestSentenceSplitter:
    def test_splits_on_period_uppercase(self):
        text = "First sentence. Second sentence. Third one."
        sents = split_into_sentences(text)
        assert len(sents) >= 2

    def test_splits_on_blank_line(self):
        text = "Paragraph one.\n\nParagraph two."
        sents = split_into_sentences(text)
        assert len(sents) >= 2

    def test_no_empty_strings(self):
        text = "Hello world.  \n\n  Another sentence!"
        sents = split_into_sentences(text)
        assert all(s.strip() for s in sents)


# -----------------------------------------------------------------------
# Token counter
# -----------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") == 1

    def test_known_count(self):
        # "hello world" = 2 tokens in cl100k_base
        assert count_tokens("hello world") == 2


# -----------------------------------------------------------------------
# Core chunker — basic guarantees
# -----------------------------------------------------------------------

class TestHierarchicalChunker:

    def test_empty_document_returns_empty(self):
        chunker = _chunker()
        doc = _make_doc("")
        result = chunker.chunk(doc)
        assert isinstance(result, ChunkedDocument)
        assert result.parents == []
        assert result.children == []
        assert result.total_tokens == 0

    def test_returns_chunked_document(self):
        text = "This is sentence one. This is sentence two. And a third one here."
        chunker = _chunker()
        result = chunker.chunk(_make_doc(text))
        assert isinstance(result, ChunkedDocument)
        assert len(result.parents) >= 1
        assert len(result.children) >= 1

    def test_no_child_exceeds_max_tokens(self):
        """Every child chunk must be within the configured limit."""
        # Use a large body of text with small child size to force many chunks
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 200)
        chunker = _chunker(parent=512, child=64, overlap=10)
        result = chunker.chunk(_make_doc(text))

        for child in result.children:
            assert child.token_count <= 64 + 10, (
                f"Child {child.chunk_id} has {child.token_count} tokens "
                f"(limit 64 + margin 10)"
            )

    def test_every_child_has_valid_parent(self):
        """Every child.parent_id must reference an existing parent."""
        text = " ".join(["Sentence number {}.".format(i) for i in range(100)])
        chunker = _chunker(parent=256, child=64, overlap=10)
        result = chunker.chunk(_make_doc(text))

        parent_ids = {p.chunk_id for p in result.parents}
        for child in result.children:
            assert child.parent_id in parent_ids, (
                f"Child {child.chunk_id} references unknown parent {child.parent_id}"
            )

    def test_parent_child_ids_are_wired(self):
        """Each parent.child_ids must match the children that reference it."""
        text = " ".join(["Sentence {i} is here.".format(i=i) for i in range(80)])
        chunker = _chunker(parent=256, child=64, overlap=10)
        result = chunker.chunk(_make_doc(text))

        for parent in result.parents:
            for cid in parent.child_ids:
                child_ids_in_doc = {c.chunk_id for c in result.children}
                assert cid in child_ids_in_doc

    def test_deterministic_chunk_ids(self):
        """Same document must always produce the same chunk IDs."""
        text = "Deterministic chunking test. " * 50
        chunker = _chunker()
        r1 = chunker.chunk(_make_doc(text, doc_id="det_test"))
        r2 = chunker.chunk(_make_doc(text, doc_id="det_test"))

        ids1 = [c.chunk_id for c in r1.children]
        ids2 = [c.chunk_id for c in r2.children]
        assert ids1 == ids2

    def test_different_doc_ids_produce_different_chunk_ids(self):
        text = "Same text content. " * 50
        chunker = _chunker()
        r1 = chunker.chunk(_make_doc(text, doc_id="doc_aaa"))
        r2 = chunker.chunk(_make_doc(text, doc_id="doc_bbb"))

        ids1 = set(c.chunk_id for c in r1.children)
        ids2 = set(c.chunk_id for c in r2.children)
        assert ids1.isdisjoint(ids2)

    def test_overlap_in_consecutive_children(self):
        """
        Consecutive child chunks from the same parent should share some
        tokens (due to overlap). We verify by checking shared sentences.
        """
        # Build a predictable multi-sentence text
        sentences = ["Sentence number {i} with some padding words to fill tokens.".format(i=i)
                     for i in range(60)]
        text = " ".join(sentences)
        chunker = _chunker(parent=600, child=80, overlap=20)
        result = chunker.chunk(_make_doc(text))

        # Find two consecutive children from the same parent
        pairs = [
            (result.children[i], result.children[i + 1])
            for i in range(len(result.children) - 1)
            if result.children[i].parent_id == result.children[i + 1].parent_id
        ]

        if pairs:
            c1, c2 = pairs[0]
            # At least some content from c1 should appear in c2 (overlap)
            words1 = set(c1.text.split())
            words2 = set(c2.text.split())
            shared = words1 & words2
            assert len(shared) > 0, "Expected overlap between consecutive children"

    def test_very_short_document(self):
        """A document shorter than child_size should produce exactly one child."""
        text = "Short doc."
        chunker = _chunker()
        result = chunker.chunk(_make_doc(text))
        assert len(result.children) == 1
        assert len(result.parents) == 1

    def test_single_sentence_document(self):
        """A single sentence should produce one parent and one child."""
        text = "This is the only sentence in the document."
        chunker = _chunker()
        result = chunker.chunk(_make_doc(text))
        assert len(result.parents) == 1
        assert len(result.children) == 1
        assert result.children[0].parent_id == result.parents[0].chunk_id

    def test_doc_id_propagated(self):
        text = "Some text. More text here."
        chunker = _chunker()
        result = chunker.chunk(_make_doc(text, doc_id="myid_abc"))
        assert result.doc_id == "myid_abc"
        for p in result.parents:
            assert p.doc_id == "myid_abc"
        for c in result.children:
            assert c.doc_id == "myid_abc"

    def test_page_number_extracted_from_markers(self):
        """PDF page markers in the text must be reflected in chunk metadata."""
        text = (
            "\n\n--- PAGE 1 ---\n\nFirst page content here. More content on page one.\n\n"
            "--- PAGE 2 ---\n\nSecond page content here. Even more content.\n\n"
            "--- PAGE 3 ---\n\nThird page content here."
        )
        chunker = _chunker(parent=200, child=60, overlap=10)
        result = chunker.chunk(_make_doc(text))

        # At least one chunk should have a page number set
        all_page_nums = [c.page_number for c in result.children if c.page_number]
        assert len(all_page_nums) > 0
        assert all(isinstance(n, int) for n in all_page_nums)

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_chunk_real_pdf(self):
        """Full end-to-end chunking on DAA.pdf."""
        from backend.ingestion.parsers.pdf_parser import PDFParser
        from backend.core.config import settings

        doc = PDFParser().parse(DAA_PDF)
        chunker = _chunker()
        result = chunker.chunk(doc)

        assert len(result.parents) > 0
        assert len(result.children) > 0
        assert result.total_tokens > 0

        # All children link to valid parents
        parent_ids = {p.chunk_id for p in result.parents}
        for child in result.children:
            assert child.parent_id in parent_ids

        # No child exceeds limit
        for child in result.children:
            assert child.token_count <= settings.CHILD_CHUNK_SIZE + settings.CHUNK_OVERLAP

        print(
            f"\nDAA.pdf — parents: {len(result.parents)}, "
            f"children: {len(result.children)}, "
            f"total_tokens: {result.total_tokens}"
        )
