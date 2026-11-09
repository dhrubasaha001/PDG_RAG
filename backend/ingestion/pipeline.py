"""
backend/ingestion/pipeline.py
Central pipeline orchestrator — parse → chunk → enrich → index.

This module is the single place that wires all four stages together.
Both the Celery tasks and the integration tests import this, keeping
the task definitions thin.

Usage
-----
from backend.ingestion.pipeline import run_pipeline

result = run_pipeline(doc_id="<sha256>", file_path="/path/to/doc.pdf")
print(result)
# PipelineResult(doc_id=..., success=True, child_chunks=3, parents=2,
#                stage_ms={parse:12, chunk:4, enrich:1250, index:88},
#                total_ms=1354)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.core.logging import get_logger

logger = get_logger("ingestion.pipeline")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    doc_id: str
    filename: str
    success: bool
    child_chunks: int = 0
    parent_chunks: int = 0
    total_tokens: int = 0
    stage_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            stages = " | ".join(f"{k}:{v:.0f}ms" for k, v in self.stage_ms.items())
            return (
                f"PipelineResult(doc_id={self.doc_id[:12]}… "
                f"children={self.child_chunks} parents={self.parent_chunks} "
                f"tokens={self.total_tokens} total={self.total_ms:.0f}ms [{stages}])"
            )
        return f"PipelineResult(doc_id={self.doc_id[:12]}… FAILED: {self.error})"


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(doc_id: str, file_path: str | Path) -> PipelineResult:
    """
    Run the full ingestion pipeline for one document.

    Stages (timed individually):
        1. Parse   — format-aware text + metadata extraction
        2. Chunk   — hierarchical parent/child chunking
        3. Enrich  — BGE embeddings + KeyBERT keywords + summaries
        4. Index   — upsert into Weaviate

    Args:
        doc_id:    SHA-256 hash of the file (used as the document's identity).
        file_path: Absolute path to the file on disk.

    Returns:
        PipelineResult with success flag, chunk counts, per-stage timing.

    Raises:
        Does NOT raise — all exceptions are caught and returned in
        PipelineResult.error so callers can decide what to do.
    """
    wall_start = time.perf_counter()
    stage_ms: dict[str, float] = {}
    filename = Path(file_path).name

    # ------------------------------------------------------------------ #
    # Stage 1 — Parse
    # ------------------------------------------------------------------ #
    try:
        t0 = time.perf_counter()
        from backend.ingestion.parsers.factory import ParserFactory
        parsed = ParserFactory.parse(file_path)
        stage_ms["parse"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "pipeline_stage_complete",
            stage="parse",
            doc_id=doc_id,
            ms=stage_ms["parse"],
            page_count=parsed.page_count,
            chars=len(parsed.text),
        )
    except Exception as exc:
        return _fail(doc_id, filename, "parse", exc, stage_ms, wall_start)

    # ------------------------------------------------------------------ #
    # Stage 2 — Chunk
    # ------------------------------------------------------------------ #
    try:
        t0 = time.perf_counter()
        from backend.ingestion.chunker import HierarchicalChunker
        chunked = HierarchicalChunker().chunk(parsed)
        stage_ms["chunk"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "pipeline_stage_complete",
            stage="chunk",
            doc_id=doc_id,
            ms=stage_ms["chunk"],
            parents=len(chunked.parents),
            children=len(chunked.children),
            total_tokens=chunked.total_tokens,
        )
    except Exception as exc:
        return _fail(doc_id, filename, "chunk", exc, stage_ms, wall_start)

    # ------------------------------------------------------------------ #
    # Stage 3 — Enrich
    # ------------------------------------------------------------------ #
    try:
        t0 = time.perf_counter()
        from backend.ingestion.enricher import get_enricher
        enriched = get_enricher().enrich(chunked)
        stage_ms["enrich"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "pipeline_stage_complete",
            stage="enrich",
            doc_id=doc_id,
            ms=stage_ms["enrich"],
            embedding_dim=len(enriched.children[0].embedding) if enriched.children else 0,
        )
    except Exception as exc:
        return _fail(doc_id, filename, "enrich", exc, stage_ms, wall_start)

    # ------------------------------------------------------------------ #
    # Stage 4 — Index
    # ------------------------------------------------------------------ #
    try:
        t0 = time.perf_counter()
        from backend.indexing.indexer import get_indexer
        index_result = get_indexer().index_document(enriched)
        stage_ms["index"] = round((time.perf_counter() - t0) * 1000, 1)

        if not index_result.success:
            raise RuntimeError("; ".join(index_result.errors))

        logger.info(
            "pipeline_stage_complete",
            stage="index",
            doc_id=doc_id,
            ms=stage_ms["index"],
            children_indexed=index_result.chunks_indexed,
            parents_indexed=index_result.parents_indexed,
        )
    except Exception as exc:
        return _fail(doc_id, filename, "index", exc, stage_ms, wall_start)

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    total_ms = round((time.perf_counter() - wall_start) * 1000, 1)

    result = PipelineResult(
        doc_id=doc_id,
        filename=filename,
        success=True,
        child_chunks=index_result.chunks_indexed,
        parent_chunks=index_result.parents_indexed,
        total_tokens=chunked.total_tokens,
        stage_ms=stage_ms,
        total_ms=total_ms,
    )

    logger.info(
        "pipeline_complete",
        doc_id=doc_id,
        filename=filename,
        child_chunks=result.child_chunks,
        parent_chunks=result.parent_chunks,
        total_tokens=result.total_tokens,
        total_ms=total_ms,
        throughput_docs_per_min=round(60_000 / total_ms, 1) if total_ms > 0 else 0,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _fail(
    doc_id: str,
    filename: str,
    stage: str,
    exc: Exception,
    stage_ms: dict[str, float],
    wall_start: float,
) -> PipelineResult:
    total_ms = round((time.perf_counter() - wall_start) * 1000, 1)
    error_msg = f"[{stage}] {type(exc).__name__}: {exc}"
    logger.error(
        "pipeline_stage_failed",
        stage=stage,
        doc_id=doc_id,
        error=error_msg,
        total_ms=total_ms,
    )
    return PipelineResult(
        doc_id=doc_id,
        filename=filename,
        success=False,
        stage_ms=stage_ms,
        total_ms=total_ms,
        error=error_msg,
    )
