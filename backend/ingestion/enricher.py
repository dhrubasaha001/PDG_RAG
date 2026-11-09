"""
backend/ingestion/enricher.py
Enrichment pipeline: GPU-accelerated BGE embeddings + KeyBERT keywords + chunk summaries.

Design
------
- Loads BAAI/bge-base-en-v1.5 once via SentenceTransformer.
- Uses the SAME model instance for KeyBERT (no double load, no double VRAM).
- BGE asymmetric retrieval requires the passage prefix
  "Represent this sentence: " on child chunk texts at index time.
  Query encoding uses the query prefix in the RAG pipeline.
- Embeddings are batched (EMBEDDING_BATCH_SIZE=32) to stay within 6 GB VRAM.
- Keywords are extracted per-parent (more context → better keywords),
  then propagated to all its children.
- Summary = first 2 complete sentences of the parent text (cheap, no LLM call).
- Returns EnrichedDocument with all data ready to feed into the Weaviate indexer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.ingestion.chunker import ChunkedDocument, ParentChunk, ChildChunk

logger = get_logger("ingestion.enricher")

# BGE passage prefix — required for asymmetric retrieval quality
_BGE_PASSAGE_PREFIX = "Represent this sentence: "


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class EnrichedChildChunk:
    # All ChildChunk fields
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    page_number: Optional[int]
    char_start: int
    char_end: int
    parent_id: str
    # Enrichment fields
    embedding: list[float]          # 768-dim BGE embedding
    keywords: list[str]             # top-K from parent context
    summary: str                    # first 2 sentences of parent text


@dataclass
class EnrichedParentChunk:
    # All ParentChunk fields
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    page_number: Optional[int]
    char_start: int
    char_end: int
    child_ids: list[str]
    # Enrichment fields
    summary: str
    keywords: list[str]


@dataclass
class EnrichedDocument:
    doc_id: str
    filename: str
    format: str
    parents: list[EnrichedParentChunk]
    children: list[EnrichedChildChunk]
    total_tokens: int
    processing_time_seconds: float


# ---------------------------------------------------------------------------
# Sentence extractor for summaries (reuses regex from chunker)
# ---------------------------------------------------------------------------

def _first_n_sentences(text: str, n: int = 2) -> str:
    """Return the first *n* complete sentences from *text*."""
    from backend.ingestion.chunker import split_into_sentences
    sentences = split_into_sentences(text)
    return " ".join(sentences[:n]) if sentences else text[:200]


# ---------------------------------------------------------------------------
# Enricher class (singleton-friendly — model loaded once)
# ---------------------------------------------------------------------------

class Enricher:
    """
    Enriches a ChunkedDocument with embeddings, keywords, and summaries.

    The underlying SentenceTransformer and KeyBERT models are loaded lazily
    on first call and cached for the lifetime of the process.
    """

    _st_model = None       # SentenceTransformer singleton
    _kw_model = None       # KeyBERT singleton (wraps the same ST model)

    def __init__(self) -> None:
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_models(self):
        """Load models on first call; return (st_model, kw_model, device)."""
        if Enricher._st_model is None:
            from sentence_transformers import SentenceTransformer
            from keybert import KeyBERT
            import torch

            device = settings.EMBEDDING_DEVICE
            # Re-validate at runtime (settings validator already ran, but
            # CUDA availability can change if a driver loads late)
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.warning(
                    "cuda_not_available",
                    fallback="cpu",
                    message="CUDA unavailable at model load time, using CPU",
                )

            logger.info(
                "loading_embedding_model",
                model=settings.EMBEDDING_MODEL,
                device=device,
            )
            st_model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
            kw_model = KeyBERT(model=st_model)

            Enricher._st_model = st_model
            Enricher._kw_model = kw_model

            logger.info(
                "embedding_model_loaded",
                model=settings.EMBEDDING_MODEL,
                device=device,
                embedding_dim=st_model.get_sentence_embedding_dimension(),
            )

        return Enricher._st_model, Enricher._kw_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, chunked_doc: ChunkedDocument) -> EnrichedDocument:
        """
        Enrich all chunks in *chunked_doc*.

        Steps:
          1. Generate summaries and keywords for each parent.
          2. Batch-encode all child chunks with the BGE passage prefix.
          3. Assemble EnrichedDocument.

        Returns:
            EnrichedDocument ready for indexing.
        """
        start = time.perf_counter()
        st_model, kw_model = self._get_models()

        parents = chunked_doc.parents
        children = chunked_doc.children

        if not parents and not children:
            logger.warning(
                "empty_chunked_document",
                doc_id=chunked_doc.doc_id,
            )
            return EnrichedDocument(
                doc_id=chunked_doc.doc_id,
                filename=chunked_doc.filename,
                format=chunked_doc.format,
                parents=[],
                children=[],
                total_tokens=0,
                processing_time_seconds=0.0,
            )

        # ---- Step 1: per-parent summary + keywords ----------------------
        parent_id_to_summary: dict[str, str] = {}
        parent_id_to_keywords: dict[str, list[str]] = {}

        for parent in parents:
            summary = _first_n_sentences(parent.text, n=2)
            parent_id_to_summary[parent.chunk_id] = summary

            # KeyBERT extraction on parent text (richer context)
            try:
                kw_pairs = kw_model.extract_keywords(
                    parent.text,
                    top_n=settings.MAX_KEYWORDS,
                    stop_words="english",
                    use_mmr=True,          # Maximal Marginal Relevance for diversity
                    diversity=0.5,
                )
                keywords = [kw for kw, _score in kw_pairs]
            except Exception as exc:
                logger.warning(
                    "keyword_extraction_failed",
                    doc_id=chunked_doc.doc_id,
                    chunk_id=parent.chunk_id,
                    error=str(exc),
                )
                keywords = []

            parent_id_to_keywords[parent.chunk_id] = keywords

        # ---- Step 2: batch-encode child chunks --------------------------
        # Apply BGE passage prefix to each child text
        child_texts_prefixed = [
            _BGE_PASSAGE_PREFIX + child.text for child in children
        ]

        logger.info(
            "encoding_chunks",
            doc_id=chunked_doc.doc_id,
            child_count=len(children),
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

        embeddings_np = st_model.encode(
            child_texts_prefixed,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,    # cosine similarity via dot product
            convert_to_numpy=True,
        )

        # ---- Step 3: assemble enriched parents --------------------------
        enriched_parents: list[EnrichedParentChunk] = []
        for parent in parents:
            enriched_parents.append(
                EnrichedParentChunk(
                    chunk_id=parent.chunk_id,
                    doc_id=parent.doc_id,
                    text=parent.text,
                    token_count=parent.token_count,
                    chunk_index=parent.chunk_index,
                    page_number=parent.page_number,
                    char_start=parent.char_start,
                    char_end=parent.char_end,
                    child_ids=list(parent.child_ids),
                    summary=parent_id_to_summary[parent.chunk_id],
                    keywords=parent_id_to_keywords[parent.chunk_id],
                )
            )

        # ---- Step 4: assemble enriched children -------------------------
        enriched_children: list[EnrichedChildChunk] = []
        for i, child in enumerate(children):
            parent_keywords = parent_id_to_keywords.get(child.parent_id, [])
            parent_summary = parent_id_to_summary.get(child.parent_id, "")

            enriched_children.append(
                EnrichedChildChunk(
                    chunk_id=child.chunk_id,
                    doc_id=child.doc_id,
                    text=child.text,
                    token_count=child.token_count,
                    chunk_index=child.chunk_index,
                    page_number=child.page_number,
                    char_start=child.char_start,
                    char_end=child.char_end,
                    parent_id=child.parent_id,
                    embedding=embeddings_np[i].tolist(),
                    keywords=parent_keywords,
                    summary=parent_summary,
                )
            )

        elapsed = time.perf_counter() - start

        logger.info(
            "enrichment_complete",
            doc_id=chunked_doc.doc_id,
            filename=chunked_doc.filename,
            parent_count=len(enriched_parents),
            child_count=len(enriched_children),
            embedding_dim=len(enriched_children[0].embedding) if enriched_children else 0,
            duration_seconds=round(elapsed, 3),
        )

        return EnrichedDocument(
            doc_id=chunked_doc.doc_id,
            filename=chunked_doc.filename,
            format=chunked_doc.format,
            parents=enriched_parents,
            children=enriched_children,
            total_tokens=chunked_doc.total_tokens,
            processing_time_seconds=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# Module-level singleton (import and use directly)
# ---------------------------------------------------------------------------
_enricher_instance: Enricher | None = None


def get_enricher() -> Enricher:
    """Return the process-level Enricher singleton."""
    global _enricher_instance
    if _enricher_instance is None:
        _enricher_instance = Enricher()
    return _enricher_instance
