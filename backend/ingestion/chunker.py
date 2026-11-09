"""
backend/ingestion/chunker.py
Hierarchical, token-aware, sentence-boundary-respecting chunker.

Design
------
1. Text is first split into sentences using a lightweight regex splitter
   (no NLTK/spaCy dependency).
2. Sentences are greedily accumulated into PARENT chunks (≤ PARENT_CHUNK_SIZE
   tokens).  A parent is closed when adding the next sentence would exceed the
   limit OR a structural boundary marker is encountered.
3. Each parent is then sub-chunked into CHILD chunks (≤ CHILD_CHUNK_SIZE tokens)
   with CHUNK_OVERLAP token overlap, still respecting sentence boundaries.
4. Every chunk carries deterministic UUIDs derived from doc_id + chunk_index,
   so re-ingesting the same document produces the same IDs.
5. Page numbers are recovered from the `--- PAGE {n} ---` markers injected by
   PDFParser.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.ingestion.parsers.base import ParsedDocument

logger = get_logger("ingestion.chunker")

# ---------------------------------------------------------------------------
# Shared tokenizer (cl100k_base — compatible with BGE, GPT-4, etc.)
# ---------------------------------------------------------------------------
_ENCODING_NAME = "cl100k_base"
_TOKENIZER: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding(_ENCODING_NAME)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(_get_tokenizer().encode(text))


# ---------------------------------------------------------------------------
# Structural markers (injected by parsers)
# ---------------------------------------------------------------------------
_STRUCTURAL_MARKER_RE = re.compile(
    r"---\s*(PAGE\s+\d+|SLIDE\s+\d+|SHEET:\s*\S+)\s*---",
    re.IGNORECASE,
)

_PAGE_MARKER_RE = re.compile(r"---\s*PAGE\s+(\d+)\s*---", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sentence splitter (regex-based, no heavy NLP dependency)
# ---------------------------------------------------------------------------

# Split after sentence-ending punctuation followed by whitespace + uppercase,
# or after a blank line (paragraph break).
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\'\(])"  # after . ! ? → uppercase
    r"|(?<=[.!?])\n"                  # after . ! ? → newline
    r"|\n{2,}",                       # blank-line paragraph break
)


def split_into_sentences(text: str) -> list[str]:
    """
    Split *text* into sentence-like segments.
    Returns a list of non-empty stripped strings.
    """
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str           # deterministic UUID5
    doc_id: str
    text: str
    token_count: int
    chunk_index: int        # 0-based, globally unique within the document
    page_number: Optional[int]
    char_start: int
    char_end: int


@dataclass
class ParentChunk(Chunk):
    child_ids: list[str] = field(default_factory=list)


@dataclass
class ChildChunk(Chunk):
    parent_id: str = ""


@dataclass
class ChunkedDocument:
    doc_id: str
    filename: str
    format: str
    parents: list[ParentChunk]
    children: list[ChildChunk]
    total_tokens: int


# ---------------------------------------------------------------------------
# UUID helpers
# ---------------------------------------------------------------------------

def _make_parent_id(doc_id: str, parent_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"parent-{doc_id}-{parent_index}"))


def _make_child_id(doc_id: str, child_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"child-{doc_id}-{child_index}"))


# ---------------------------------------------------------------------------
# Page number detection
# ---------------------------------------------------------------------------

def _extract_page_number(text: str, preceding_text: str = "") -> Optional[int]:
    """
    Try to determine the PDF page number for a chunk.
    Checks for --- PAGE n --- markers inside the chunk text first,
    then scans the preceding text for the most recent marker.
    """
    # Check inside the chunk itself
    m = _PAGE_MARKER_RE.search(text)
    if m:
        return int(m.group(1))
    # Fall back to most recent marker in text seen before this chunk
    matches = list(_PAGE_MARKER_RE.finditer(preceding_text))
    if matches:
        return int(matches[-1].group(1))
    return None


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

class HierarchicalChunker:
    """
    Produces parent + child chunks from a ParsedDocument.

    Parameters are read from ``settings`` but can be overridden per-instance
    for testing.
    """

    def __init__(
        self,
        parent_chunk_size: int | None = None,
        child_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.parent_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        self.child_size = child_chunk_size or settings.CHILD_CHUNK_SIZE
        self.overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chunk(self, doc: ParsedDocument) -> ChunkedDocument:
        """
        Chunk *doc* into a hierarchical structure.

        Returns:
            ChunkedDocument with parents and children linked by ID.
        """
        text = doc.text
        sentences = split_into_sentences(text)

        if not sentences:
            logger.warning("empty_document", doc_id=doc.doc_id, filename=doc.filename)
            return ChunkedDocument(
                doc_id=doc.doc_id,
                filename=doc.filename,
                format=doc.format,
                parents=[],
                children=[],
                total_tokens=0,
            )

        parents: list[ParentChunk] = self._build_parents(doc.doc_id, text, sentences)
        children: list[ChildChunk] = self._build_children(doc.doc_id, text, parents)

        # Wire child_ids into each parent
        parent_map = {p.chunk_id: p for p in parents}
        for child in children:
            if child.parent_id in parent_map:
                parent_map[child.parent_id].child_ids.append(child.chunk_id)

        total_tokens = sum(p.token_count for p in parents)

        logger.info(
            "document_chunked",
            doc_id=doc.doc_id,
            filename=doc.filename,
            parent_count=len(parents),
            child_count=len(children),
            total_tokens=total_tokens,
        )

        return ChunkedDocument(
            doc_id=doc.doc_id,
            filename=doc.filename,
            format=doc.format,
            parents=parents,
            children=children,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Parent chunk construction
    # ------------------------------------------------------------------

    def _build_parents(
        self, doc_id: str, full_text: str, sentences: list[str]
    ) -> list[ParentChunk]:
        parents: list[ParentChunk] = []
        parent_index = 0

        current_sentences: list[str] = []
        current_tokens = 0
        # Track character position within full_text
        search_offset = 0

        def flush_parent(sents: list[str]) -> None:
            nonlocal parent_index, search_offset

            if not sents:
                return

            parent_text = " ".join(sents)
            token_count = count_tokens(parent_text)

            # Locate char_start / char_end in full_text
            char_start = full_text.find(sents[0], search_offset)
            if char_start == -1:
                char_start = 0
            char_end = char_start + len(parent_text)

            page_number = _extract_page_number(
                parent_text, full_text[:char_start]
            )

            pid = _make_parent_id(doc_id, parent_index)
            parents.append(
                ParentChunk(
                    chunk_id=pid,
                    doc_id=doc_id,
                    text=parent_text,
                    token_count=token_count,
                    chunk_index=parent_index,
                    page_number=page_number,
                    char_start=char_start,
                    char_end=char_end,
                    child_ids=[],
                )
            )
            # Advance search offset past the first sentence of this parent
            search_offset = char_end
            parent_index += 1

        for sentence in sentences:
            # Skip pure structural markers — don't include them in chunk text,
            # but use them as natural flush boundaries.
            if _STRUCTURAL_MARKER_RE.match(sentence):
                flush_parent(current_sentences)
                current_sentences = []
                current_tokens = 0
                continue

            sent_tokens = count_tokens(sentence)

            # If a single sentence is larger than parent_size, include it alone
            if sent_tokens >= self.parent_size:
                flush_parent(current_sentences)
                current_sentences = [sentence]
                flush_parent(current_sentences)
                current_sentences = []
                current_tokens = 0
                continue

            if current_tokens + sent_tokens > self.parent_size and current_sentences:
                flush_parent(current_sentences)
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        flush_parent(current_sentences)  # flush tail

        return parents

    # ------------------------------------------------------------------
    # Child chunk construction
    # ------------------------------------------------------------------

    def _build_children(
        self, doc_id: str, full_text: str, parents: list[ParentChunk]
    ) -> list[ChildChunk]:
        children: list[ChildChunk] = []
        child_index = 0

        for parent in parents:
            parent_sentences = split_into_sentences(parent.text)

            if not parent_sentences:
                continue

            # Build children using a sliding window over sentences
            # tracking token counts for overlap
            window: list[str] = []
            window_tokens = 0
            child_char_offset = parent.char_start

            def flush_child(sents: list[str]) -> None:
                nonlocal child_index, child_char_offset

                if not sents:
                    return

                child_text = " ".join(sents)
                token_count = count_tokens(child_text)

                # Locate within parent text
                c_start = full_text.find(sents[0], child_char_offset)
                if c_start == -1:
                    c_start = parent.char_start
                c_end = c_start + len(child_text)

                page_number = _extract_page_number(
                    child_text, full_text[:c_start]
                )

                cid = _make_child_id(doc_id, child_index)
                children.append(
                    ChildChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        text=child_text,
                        token_count=token_count,
                        chunk_index=child_index,
                        page_number=page_number,
                        char_start=c_start,
                        char_end=c_end,
                        parent_id=parent.chunk_id,
                    )
                )
                child_index += 1

            i = 0
            while i < len(parent_sentences):
                sent = parent_sentences[i]
                sent_tokens = count_tokens(sent)

                # Single sentence exceeds child size — add as standalone child
                if sent_tokens >= self.child_size:
                    if window:
                        flush_child(window)
                        window = []
                        window_tokens = 0
                    flush_child([sent])
                    i += 1
                    continue

                if window_tokens + sent_tokens > self.child_size and window:
                    flush_child(window)

                    # Compute overlap: keep sentences from the END of window
                    # whose cumulative tokens ≤ self.overlap
                    overlap_sents: list[str] = []
                    overlap_tokens = 0
                    for s in reversed(window):
                        t = count_tokens(s)
                        if overlap_tokens + t > self.overlap:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tokens += t

                    window = overlap_sents
                    window_tokens = overlap_tokens

                window.append(sent)
                window_tokens += sent_tokens
                i += 1

            if window:
                flush_child(window)

        return children
