"""
tests/unit/test_pipeline.py
Unit tests for the pipeline orchestrator (no Weaviate needed — index is mocked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DAA_PDF   = REPO_ROOT / "backend" / "data" / "DAA.pdf"


class TestPipelineOrchestrator:

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_pipeline_succeeds_on_real_pdf(self):
        """Stages 1-3 run for real; index stage is mocked."""
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.indexer import IndexResult

        doc_id  = "a" * 64
        mock_ir = IndexResult(
            doc_id=doc_id, success=True,
            chunks_indexed=3, parents_indexed=2, errors=[],
        )
        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_ir,
        ):
            result = run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        assert result.success is True
        assert result.child_chunks == 3
        assert result.parent_chunks == 2
        assert result.total_tokens > 0
        assert result.total_ms > 0
        for stage in ("parse", "chunk", "enrich", "index"):
            assert stage in result.stage_ms
            assert result.stage_ms[stage] >= 0

    def test_pipeline_returns_failure_on_missing_file(self):
        """Bad path → PipelineResult.success=False, no exception raised."""
        from backend.ingestion.pipeline import run_pipeline

        result = run_pipeline(doc_id="b" * 64, file_path="/no/such/file.pdf")

        assert result.success is False
        assert result.error is not None
        assert result.child_chunks == 0
        assert result.parent_chunks == 0

    @pytest.mark.skipif(not DAA_PDF.exists(), reason="DAA.pdf not found")
    def test_pipeline_reports_index_failure(self):
        """If Weaviate returns failure, pipeline should surface it."""
        from backend.ingestion.pipeline import run_pipeline
        from backend.indexing.indexer import IndexResult

        doc_id  = "c" * 64
        mock_ir = IndexResult(
            doc_id=doc_id, success=False,
            chunks_indexed=0, parents_indexed=0,
            errors=["connection refused"],
        )
        with patch(
            "backend.indexing.indexer.WeaviateIndexer.index_document",
            return_value=mock_ir,
        ):
            result = run_pipeline(doc_id=doc_id, file_path=DAA_PDF)

        assert result.success is False
        assert "connection refused" in (result.error or "")

    def test_pipeline_result_str_success(self):
        from backend.ingestion.pipeline import PipelineResult
        r = PipelineResult(
            doc_id="x" * 64, filename="test.pdf", success=True,
            child_chunks=5, parent_chunks=2, total_tokens=400,
            stage_ms={"parse": 10, "chunk": 5, "enrich": 800, "index": 50},
            total_ms=865,
        )
        s = str(r)
        assert "children=5" in s
        assert "parse:10ms" in s
        assert "FAILED" not in s

    def test_pipeline_result_str_failure(self):
        from backend.ingestion.pipeline import PipelineResult
        r = PipelineResult(
            doc_id="y" * 64, filename="bad.pdf", success=False,
            error="[parse] FileNotFoundError: not found",
        )
        s = str(r)
        assert "FAILED" in s
        assert "FileNotFoundError" in s
