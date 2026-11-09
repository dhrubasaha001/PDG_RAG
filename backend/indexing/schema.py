"""
backend/indexing/schema.py
Weaviate v4 collection definitions for the PDG_RAG system.

Two collections:
  DocumentParent  — parent chunks (broad context, ~1024 tokens)
  DocumentChunk   — child chunks  (precise retrieval, ~256 tokens, carries embedding)

HNSW parameters tuned for the 1M-doc target while running comfortably on 16 GB RAM:
  efConstruction = 128   (index build quality — higher = better recall, slower build)
  maxConnections  = 64   (graph connectivity — 64 is the sweet spot for 1M vectors)
  ef              = 64   (query-time beam width — trade recall vs latency)

These can be relaxed (lower) for the 10K dev run and promoted when scaling.
"""

from __future__ import annotations

import weaviate.classes.config as wc

# ---------------------------------------------------------------------------
# Collection names — single source of truth
# ---------------------------------------------------------------------------
PARENT_COLLECTION = "DocumentParent"
CHILD_COLLECTION = "DocumentChunk"

# HNSW index config shared by both collections
_HNSW_CONFIG = wc.Configure.VectorIndex.hnsw(
    ef_construction=128,
    max_connections=64,
    ef=64,
)


def parent_collection_config() -> dict:
    """
    Return the keyword-argument dict for creating the DocumentParent collection.
    Passed to  client.collections.create(**parent_collection_config()).
    """
    return dict(
        name=PARENT_COLLECTION,
        description="Parent chunks — broad context windows (~1024 tokens)",
        vectorizer_config=wc.Configure.Vectorizer.none(),   # we supply vectors
        vector_index_config=_HNSW_CONFIG,
        properties=[
            wc.Property(
                name="doc_id",
                data_type=wc.DataType.TEXT,
                description="SHA-256 hash of the source document",
                index_filterable=True,
                index_searchable=True,
            ),
            wc.Property(
                name="filename",
                data_type=wc.DataType.TEXT,
                description="Original filename",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="format",
                data_type=wc.DataType.TEXT,
                description="Document format: pdf | docx | pptx | xlsx | md | html | txt",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="text",
                data_type=wc.DataType.TEXT,
                description="Full parent chunk text",
                index_filterable=False,
                index_searchable=True,
            ),
            wc.Property(
                name="summary",
                data_type=wc.DataType.TEXT,
                description="Auto-generated summary (first 2 sentences)",
                index_filterable=False,
                index_searchable=True,
            ),
            wc.Property(
                name="keywords",
                data_type=wc.DataType.TEXT_ARRAY,
                description="KeyBERT-extracted keywords",
                index_filterable=True,
                index_searchable=True,
            ),
            wc.Property(
                name="chunk_index",
                data_type=wc.DataType.INT,
                description="0-based index of this parent within the document",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="chunk_count",
                data_type=wc.DataType.INT,
                description="Number of child chunks that belong to this parent",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="token_count",
                data_type=wc.DataType.INT,
                description="Token count of this parent chunk",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="page_number",
                data_type=wc.DataType.INT,
                description="Source page number (PDF only, nullable)",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="char_start",
                data_type=wc.DataType.INT,
                description="Character offset start in the original document",
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="char_end",
                data_type=wc.DataType.INT,
                description="Character offset end in the original document",
                index_filterable=False,
                index_searchable=False,
            ),
        ],
    )


def child_collection_config() -> dict:
    """
    Return the keyword-argument dict for creating the DocumentChunk collection.
    """
    return dict(
        name=CHILD_COLLECTION,
        description="Child chunks — precise retrieval units (~256 tokens, carry BGE embeddings)",
        vectorizer_config=wc.Configure.Vectorizer.none(),
        vector_index_config=_HNSW_CONFIG,
        properties=[
            wc.Property(
                name="doc_id",
                data_type=wc.DataType.TEXT,
                description="SHA-256 hash of the source document",
                index_filterable=True,
                index_searchable=True,
            ),
            wc.Property(
                name="parent_id",
                data_type=wc.DataType.TEXT,
                description="UUID of the parent DocumentParent object",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="chunk_index",
                data_type=wc.DataType.INT,
                description="0-based global child chunk index within the document",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="page_number",
                data_type=wc.DataType.INT,
                description="Source page number (PDF only, nullable)",
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="text",
                data_type=wc.DataType.TEXT,
                description="Child chunk text (without BGE prefix)",
                index_filterable=False,
                index_searchable=True,
            ),
            wc.Property(
                name="keywords",
                data_type=wc.DataType.TEXT_ARRAY,
                description="Keywords inherited from parent chunk",
                index_filterable=True,
                index_searchable=True,
            ),
            wc.Property(
                name="summary",
                data_type=wc.DataType.TEXT,
                description="Summary inherited from parent chunk",
                index_filterable=False,
                index_searchable=True,
            ),
            wc.Property(
                name="char_start",
                data_type=wc.DataType.INT,
                description="Character offset start in the original document",
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="char_end",
                data_type=wc.DataType.INT,
                description="Character offset end in the original document",
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="token_count",
                data_type=wc.DataType.INT,
                description="Token count of this child chunk",
                index_filterable=True,
                index_searchable=False,
            ),
        ],
    )
