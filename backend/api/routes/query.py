"""
backend/api/routes/query.py
Query endpoint — ask questions against the indexed knowledge base.

POST /ask   — question answering via RAG pipeline
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.core.logging import get_logger
from backend.schemas import Question

logger = get_logger("api.query")

router = APIRouter(tags=["query"])


@router.post("/ask")
def ask(data: Question) -> dict:
    """
    Ask a question against the indexed document corpus.

    Uses the RAG pipeline: Weaviate hybrid search → LLM answer generation.
    """
    if not data.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    try:
        from backend.rag_pipeline import ask_question
        answer = ask_question(data.question)
        return {"answer": answer}
    except Exception as exc:
        logger.error("query_failed", question=data.question[:100], error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Query pipeline error: {exc}",
        )
