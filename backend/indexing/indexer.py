"""
backend/indexing/indexer.py
WeaviateIndexer — batch upsert of EnrichedDocuments into Weaviate.

Design
------
- Uses weaviate-client v4 dynamic batching context manager for throughput.
- Deterministic UUID5 for every object → safe idempotent re-ingestion.
- Upsert pattern: if doc_id already exists, delete all old objects first,
  then re-index (clean slate per document).
- Parents are indexed before children (children reference parent UUIDs).
- Batch errors are collected into IndexResult.errors — never silently dropped.
- Logs throughput (chunks/second) per document.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import weaviate

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.indexing.schema import CHILD_COLLECTION, PARENT_COLLECTION
from backend.indexing.weaviate_client import (
    get_child_collection,
    get_parent_collection,
    get_weaviate_client,
)
from backend.ingestion.enricher import EnrichedDocument, EnrichedChildChunk, EnrichedParentChunk

logger = get_logger("indexing.indexer")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    doc_id: str
    success: bool
    chunks_indexed: int       # child chunks
    parents_indexed: int
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# UUID helpers — deterministic, collision-free across re-ingests
# ---------------------------------------------------------------------------

def _parent_uuid(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"parent-{doc_id}-{chunk_index}"))


def _child_uuid(chunk_id: str) -> str:
    """chunk_id is already a UUID5; wrap in another for namespace isolation."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"child-{chunk_id}"))


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class WeaviateIndexer:
    """
    Indexes EnrichedDocuments into Weaviate.

    Can be used with an externally managed client (pass *client* to each
    method) or will open/close its own connection per call.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(
        self,
        enriched_doc: EnrichedDocument,
        client: weaviate.WeaviateClient | None = None,
    ) -> IndexResult:
        """
        Index a single EnrichedDocument.

        If the document already exists (same doc_id), it is deleted first
        then re-indexed (clean upsert).

        Args:
            enriched_doc: Fully enriched document from the enrichment pipeline.
            client: Optional connected WeaviateClient.

        Returns:
            IndexResult with counts and any errors.
        """
        if client is not None:
            return self._index_document_with_client(enriched_doc, client)

        with get_weaviate_client() as c:
            return self._index_document_with_client(enriched_doc, c)

    def index_batch(
        self,
        enriched_docs: list[EnrichedDocument],
        client: weaviate.WeaviateClient | None = None,
    ) -> list[IndexResult]:
        """
        Index a list of EnrichedDocuments, sharing a single connection.

        Args:
            enriched_docs: List of enriched documents.
            client: Optional connected WeaviateClient.

        Returns:
            List of IndexResult, one per document.
        """
        if client is not None:
            return [self._index_document_with_client(doc, client) for doc in enriched_docs]

        with get_weaviate_client() as c:
            return [self._index_document_with_client(doc, c) for doc in enriched_docs]

    def document_exists(
        self,
        doc_id: str,
        client: weaviate.WeaviateClient | None = None,
    ) -> bool:
        """
        Return True if any child chunk with this doc_id exists in Weaviate.

        Args:
            doc_id: SHA-256 hash of the source document.
            client: Optional connected WeaviateClient.
        """
        def _check(c: weaviate.WeaviateClient) -> bool:
            col = get_child_collection(c)
            result = col.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id),
                limit=1,
            )
            return len(result.objects) > 0

        if client is not None:
            return _check(client)
        with get_weaviate_client() as c:
            return _check(c)

    def delete_document(
        self,
        doc_id: str,
        client: weaviate.WeaviateClient | None = None,
    ) -> bool:
        """
        Delete all parent and child objects for a given doc_id.

        Args:
            doc_id: SHA-256 hash of the source document.
            client: Optional connected WeaviateClient.

        Returns:
            True if deletion completed without errors.
        """
        def _delete(c: weaviate.WeaviateClient) -> bool:
            success = True
            doc_filter = weaviate.classes.query.Filter.by_property("doc_id").equal(doc_id)

            for col_name, col_getter in [
                (CHILD_COLLECTION, get_child_collection),
                (PARENT_COLLECTION, get_parent_collection),
            ]:
                try:
                    col = col_getter(c)
                    col.data.delete_many(where=doc_filter)
                    logger.debug(
                        "document_deleted_from_collection",
                        doc_id=doc_id,
                        collection=col_name,
                    )
                except Exception as exc:
                    logger.error(
                        "document_delete_failed",
                        doc_id=doc_id,
                        collection=col_name,
                        error=str(exc),
                    )
                    success = False
            return success

        if client is not None:
            return _delete(client)
        with get_weaviate_client() as c:
            return _delete(c)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _index_document_with_client(
        self,
        enriched_doc: EnrichedDocument,
        client: weaviate.WeaviateClient,
    ) -> IndexResult:
        start = time.perf_counter()
        errors: list[str] = []
        parents_indexed = 0
        children_indexed = 0

        doc_id = enriched_doc.doc_id

        # ---- Upsert: clean up existing data first ----------------------
        if self.document_exists(doc_id, client):
            logger.info("document_exists_reingest", doc_id=doc_id)
            self.delete_document(doc_id, client)

        # ---- Index parents first (children reference parent UUIDs) -----
        parent_col = get_parent_collection(client)

        try:
            with parent_col.batch.dynamic() as batch:
                for parent in enriched_doc.parents:
                    obj_uuid = _parent_uuid(doc_id, parent.chunk_index)
                    props = _parent_to_props(parent)
                    batch.add_object(properties=props, uuid=obj_uuid)

            # Collect batch errors after context manager exits
            if parent_col.batch.failed_objects:
                for fo in parent_col.batch.failed_objects:
                    msg = f"Parent batch error: {fo.message}"
                    errors.append(msg)
                    logger.warning("batch_parent_error", doc_id=doc_id, error=fo.message)
            else:
                parents_indexed = len(enriched_doc.parents)

        except Exception as exc:
            msg = f"Parent indexing failed: {exc}"
            errors.append(msg)
            logger.error("parent_indexing_failed", doc_id=doc_id, error=str(exc))

        # ---- Index children with embeddings ----------------------------
        child_col = get_child_collection(client)

        try:
            with child_col.batch.dynamic() as batch:
                for child in enriched_doc.children:
                    obj_uuid = _child_uuid(child.chunk_id)
                    props = _child_to_props(child)
                    vector = child.embedding  # 768-dim float list
                    batch.add_object(
                        properties=props,
                        uuid=obj_uuid,
                        vector=vector,
                    )

            if child_col.batch.failed_objects:
                for fo in child_col.batch.failed_objects:
                    msg = f"Child batch error: {fo.message}"
                    errors.append(msg)
                    logger.warning("batch_child_error", doc_id=doc_id, error=fo.message)
            else:
                children_indexed = len(enriched_doc.children)

        except Exception as exc:
            msg = f"Child indexing failed: {exc}"
            errors.append(msg)
            logger.error("child_indexing_failed", doc_id=doc_id, error=str(exc))

        elapsed = time.perf_counter() - start
        success = len(errors) == 0
        total_chunks = parents_indexed + children_indexed
        throughput = total_chunks / elapsed if elapsed > 0 else 0.0

        logger.info(
            "document_indexed",
            doc_id=doc_id,
            filename=enriched_doc.filename,
            parents_indexed=parents_indexed,
            children_indexed=children_indexed,
            errors=len(errors),
            duration_seconds=round(elapsed, 3),
            throughput_chunks_per_sec=round(throughput, 1),
        )

        return IndexResult(
            doc_id=doc_id,
            success=success,
            chunks_indexed=children_indexed,
            parents_indexed=parents_indexed,
            errors=errors,
            duration_seconds=round(elapsed, 3),
        )


# ---------------------------------------------------------------------------
# Property serialisers
# ---------------------------------------------------------------------------

def _parent_to_props(parent: EnrichedParentChunk) -> dict:
    return {
        "doc_id": parent.doc_id,
        "filename": getattr(parent, "filename", ""),
        "format": getattr(parent, "format", ""),
        "text": parent.text,
        "summary": parent.summary,
        "keywords": parent.keywords,
        "chunk_index": parent.chunk_index,
        "chunk_count": len(parent.child_ids),
        "token_count": parent.token_count,
        "page_number": parent.page_number if parent.page_number is not None else 0,
        "char_start": parent.char_start,
        "char_end": parent.char_end,
    }


def _child_to_props(child: EnrichedChildChunk) -> dict:
    return {
        "doc_id": child.doc_id,
        "parent_id": child.parent_id,
        "chunk_index": child.chunk_index,
        "page_number": child.page_number if child.page_number is not None else 0,
        "text": child.text,
        "keywords": child.keywords,
        "summary": child.summary,
        "char_start": child.char_start,
        "char_end": child.char_end,
        "token_count": child.token_count,
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_indexer_instance: WeaviateIndexer | None = None


def get_indexer() -> WeaviateIndexer:
    """Return the process-level WeaviateIndexer singleton."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = WeaviateIndexer()
    return _indexer_instance
