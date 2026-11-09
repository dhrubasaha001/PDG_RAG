# PDG_RAG — Production Document RAG System

A production-grade **Retrieval-Augmented Generation** system that ingests documents at scale, indexes them into a persistent vector store, and answers questions via hybrid search + LLM generation.

---

## Architecture

```
Upload (API)  →  Redis Queue  →  Celery Worker
                                      │
                          ┌───────────┼───────────┐
                       Parse       Chunk        Enrich
                    (pypdf/docx) (tiktoken) (BGE + KeyBERT)
                                      │
                                  Weaviate
                               (HNSW index)
                                      │
                          Query API → Hybrid Search → LLM → Answer
```

**Stack:**
| Layer | Technology |
|---|---|
| API | FastAPI |
| Task Queue | Celery + Redis |
| Vector Store | Weaviate (self-hosted) |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim) |
| Keywords | KeyBERT (reuses embedding model) |
| Tokenizer | tiktoken cl100k_base |
| Registry | SQLite (stdlib sqlite3) |
| Frontend | Streamlit |
| Monitoring | Flower |

---

## Quick Start

### 1. Start infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

Services started:
- Weaviate at http://localhost:8080
- Redis at localhost:6379
- Flower at http://localhost:5555

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (RTX 3050 / CUDA 12.x):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you need to change ports or model settings
```

### 4. Start the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 5. Start Celery workers

**Windows:**
```bash
celery -A backend.workers.celery_app worker -Q high_priority,batch --pool=solo -l info
```

**Linux / Docker:**
```bash
celery -A backend.workers.celery_app worker -Q high_priority,batch --concurrency=4 -l info
```

### 6. Start the frontend (optional)

```bash
streamlit run frontend/app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/upload` | Upload a single document (returns job_id immediately) |
| `POST` | `/ingest/batch` | Upload up to 100 documents at once |
| `GET`  | `/ingest/documents` | List all documents with status + pagination |
| `DELETE` | `/ingest/documents/{doc_id}` | Remove document from index + registry |
| `GET`  | `/jobs/{job_id}` | Poll ingestion job status |
| `GET`  | `/jobs` | List all jobs |
| `POST` | `/ask` | Ask a question against the indexed corpus |
| `GET`  | `/health` | Liveness + dependency health check |

### Upload a document

```bash
curl -F "file=@document.pdf" http://localhost:8000/ingest/upload
# → {"job_id": "...", "doc_id": "...", "status": "queued"}
```

### Poll job status

```bash
curl http://localhost:8000/jobs/{job_id}
# → {"status": "indexed", "processed_documents": 1, "total_documents": 1}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Master Theorem?"}'
```

---

## Supported File Formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| PowerPoint | `.pptx` |
| Excel | `.xlsx`, `.xls` |
| Markdown | `.md`, `.markdown` |
| HTML | `.html`, `.htm` |
| Plain text | `.txt`, `.rst`, `.csv` |

---

## Ingestion Pipeline

```
File Upload
    │
    ▼
ParserFactory           — auto-detect format, extract text + metadata
    │                     scanned PDF detection (flagged, not OCR'd)
    ▼
HierarchicalChunker     — token-aware, sentence-boundary-respecting
    │                     parent chunks ~1024 tokens
    │                     child chunks ~256 tokens, 20% overlap
    │                     deterministic UUID5 chunk IDs
    ▼
Enricher                — BAAI/bge-base-en-v1.5 embeddings (768-dim, GPU)
    │                     KeyBERT keywords per parent (MMR diversity)
    │                     first-2-sentence summaries
    ▼
WeaviateIndexer         — dynamic batch upsert
    │                     parents (DocumentParent) + children (DocumentChunk)
    │                     HNSW: efConstruction=128, maxConnections=64
    ▼
Document Registry       — SQLite status tracking
                          PENDING → PROCESSING → INDEXED / FAILED
```

---

## Scaling

The system is designed for **10K docs now, 1M+ docs later**:

- **Celery workers** are stateless — add more workers by running the command again
- **Weaviate HNSW** disk-offloads at large scale — tune `ef` / `efConstruction` in `backend/indexing/schema.py`
- **SQLite → Postgres**: change `DB_URL` in `.env` (swap driver, schema is portable)
- **Batch sizes** are config-driven: `EMBEDDING_BATCH_SIZE`, `WEAVIATE_BATCH_SIZE`

### Load test (10K docs)

```bash
# With infrastructure and workers running:
python tests/load/batch_ingest_10k.py --docs 10000 --batch 50

# With your own PDFs:
python tests/load/batch_ingest_10k.py --docs 10000 --real-dir /path/to/pdfs
```

---

## Running Tests

```bash
# Unit tests (no infrastructure needed)
pytest tests/unit/ -v

# Integration tests (requires Docker infra running)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

---

## Project Structure

```
PDG_RAG/
├── backend/
│   ├── api/routes/         ingest.py, jobs.py, query.py
│   ├── ingestion/          pipeline.py, chunker.py, enricher.py
│   │   └── parsers/        pdf, docx, pptx, xlsx, text, factory
│   ├── indexing/           weaviate_client.py, schema.py, indexer.py
│   ├── workers/            celery_app.py, tasks.py
│   ├── db/                 database.py, models.py, repository.py
│   ├── core/               config.py, logging.py
│   ├── main.py
│   ├── rag_pipeline.py
│   └── schemas.py
├── docker/
│   ├── docker-compose.yml  Weaviate + Redis + Flower
│   └── weaviate.env
├── tests/
│   ├── unit/               96 tests (no infra needed)
│   ├── integration/        17 tests (Weaviate + skip guard)
│   └── load/               batch_ingest_10k.py
├── frontend/app.py
├── .env.example
└── requirements.txt
```
