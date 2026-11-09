"""
backend/rag_pipeline.py
RAG query pipeline — Weaviate hybrid search + Ollama LLM generation.

Flow
----
1. Embed the question with BGE (same model used at index time, no prefix for queries)
2. Hybrid search in Weaviate (BM25 sparse + dense vector) on DocumentChunk
3. Fetch parent chunks by parent_id for broader context
4. Build a grounded prompt and call the LLM
5. Return the answer

Hybrid search rationale
-----------------------
- Dense (vector) search catches semantic meaning: "cardiac arrest" ≈ "heart attack"
- Sparse (BM25) search catches exact terminology: model names, IDs, acronyms
- Combined they consistently outperform either alone on domain-specific corpora
"""

from __future__ import annotations

from typing import Optional

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger("rag_pipeline")

# BGE query prefix — asymmetric retrieval: passages use "Represent this sentence:",
# queries use "Represent this question:" for best recall
_BGE_QUERY_PREFIX = "Represent this question: "

# Number of child chunks to retrieve
_TOP_K = 5


# ---------------------------------------------------------------------------
# Embedding helper (reuses the singleton from enricher)
# ---------------------------------------------------------------------------

def _embed_query(question: str) -> list[float]:
    """Embed the query with the BGE model (query-side prefix, no normalisation needed)."""
    from backend.ingestion.enricher import get_enricher
    enricher = get_enricher()
    st_model, _ = enricher._get_models()
    vec = st_model.encode(
        [_BGE_QUERY_PREFIX + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0].tolist()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(question: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve the most relevant chunks for *question* using Weaviate hybrid search.

    Returns a list of dicts, each with:
        text        — child chunk text
        parent_text — broader context from the parent chunk
        summary     — auto-generated summary
        keywords    — extracted keywords
        doc_id      — source document SHA-256
        page_number — source page (PDF only)
        score       — hybrid search score
    """
    from backend.indexing.weaviate_client import get_weaviate_client, health_check
    from backend.indexing.schema import CHILD_COLLECTION, PARENT_COLLECTION
    import weaviate.classes.query as wq

    if not health_check():
        logger.warning("weaviate_not_reachable_for_query")
        return []

    query_vector = _embed_query(question)
    results: list[dict] = []

    with get_weaviate_client() as client:
        child_col  = client.collections.get(CHILD_COLLECTION)
        parent_col = client.collections.get(PARENT_COLLECTION)

        # ---- Hybrid search on child chunks --------------------------------
        response = child_col.query.hybrid(
            query=question,           # BM25 uses the raw question text
            vector=query_vector,      # dense vector search
            limit=top_k,
            return_metadata=wq.MetadataQuery(score=True),
        )

        # ---- Fetch parent context for each result -------------------------
        for obj in response.objects:
            props      = obj.properties
            parent_id  = props.get("parent_id", "")
            parent_text = ""

            if parent_id:
                try:
                    # Fetch parent by doc_id + chunk_index stored in parent_id
                    parent_resp = parent_col.query.fetch_objects(
                        filters=wq.Filter.by_property("doc_id").equal(
                            props.get("doc_id", "")
                        ),
                        limit=100,
                    )
                    for p in parent_resp.objects:
                        # Parent IDs are deterministic UUIDs shared by the
                        # chunker and indexer.
                        if str(p.uuid) == parent_id:
                            parent_text = p.properties.get("text", "")
                            break
                        # fallback: first parent of this doc
                        if not parent_text:
                            parent_text = p.properties.get("text", "")
                except Exception as exc:
                    logger.warning("parent_fetch_failed", error=str(exc))

            results.append({
                "text":        props.get("text", ""),
                "parent_text": parent_text,
                "summary":     props.get("summary", ""),
                "keywords":    props.get("keywords", []),
                "doc_id":      props.get("doc_id", ""),
                "page_number": props.get("page_number"),
                "score":       obj.metadata.score if obj.metadata else 0.0,
            })

    logger.info(
        "retrieval_complete",
        question=question[:80],
        results_returned=len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Construct a grounded RAG prompt.
    Uses parent_text (broad context) when available, falls back to chunk text.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("parent_text") or chunk.get("text", "")
        summary = chunk.get("summary", "")
        kws = ", ".join(chunk.get("keywords", [])[:5])
        page = chunk.get("page_number")
        source_note = f" [page {page}]" if page else ""

        context_parts.append(
            f"[Context {i}{source_note}]\n"
            + (f"Summary: {summary}\n" if summary else "")
            + (f"Keywords: {kws}\n" if kws else "")
            + f"{text}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""You are a precise assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so clearly — do not fabricate.

Context:
{context_str}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Main entry point (called by /ask endpoint)
# ---------------------------------------------------------------------------

def ask_question(question: str, top_k: int = _TOP_K) -> str:
    """
    Run the full RAG pipeline: retrieve → build prompt → generate answer.

    Args:
        question: The user's question.
        top_k:    Number of chunks to retrieve (default 5).

    Returns:
        Generated answer string.

    Raises:
        RuntimeError: If the LLM call fails.
    """
    # Retrieve relevant context
    context_chunks = retrieve_context(question, top_k=top_k)

    if not context_chunks:
        return (
            "I could not find any relevant documents in the knowledge base. "
            "Please upload and index documents before asking questions."
        )

    # Build grounded prompt
    prompt = _build_prompt(question, context_chunks)

    logger.info(
        "llm_call_start",
        question=question[:80],
        context_chunks=len(context_chunks),
        prompt_chars=len(prompt),
    )

    # Call LLM
    try:
        import ollama
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"]
    except Exception as exc:
        logger.error("llm_call_failed", error=str(exc))
        raise RuntimeError(
            f"LLM call failed: {exc}. "
            "Make sure Ollama is running: ollama serve"
        ) from exc

    logger.info("llm_call_complete", answer_chars=len(answer))
    return answer
