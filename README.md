# Medical AI Analytics Platform

A **privacy-first** medical analytics system combining **RAG (Retrieval-Augmented Generation)** and **NLP-to-SQL** capabilities. Features a multi-agent architecture that intelligently routes queries to the right data source—all running **100% locally** via Ollama.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Privacy First** | No data leaves your machine. LLM (qwen2.5:14b) + embeddings run locally via Ollama |
| **Multi-Agent System** | 4 specialized agents: SQL, RAG, Hybrid, Knowledge |
| **Query Orchestration** | Decomposes complex multi-part questions automatically |
| **4-Stage RAG Pipeline** | BM25 + Semantic search → RRF fusion → LLM reranking → MMR diversity |
| **Scalable to 1M+ rows** | 27 database indexes, 6 materialized views, LRU query cache |
| **Guardrails** | Grounding validation, confidence scoring, source attribution |
| **Modern UI** | Full-width responsive dashboard with markdown rendering |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                             │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Orchestrator                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Decomposer  │→ │   Router    │→ │  Query Result Cache     │  │
│  │(multi-part) │  │(SQL/RAG/etc)│  │  (LRU, 1hr TTL)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
        ┌─────────────────┼─────────────────┬─────────────────┐
        ▼                 ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   SQL Agent   │ │   RAG Agent   │ │ Hybrid Agent  │ │Knowledge Agent│
│               │ │               │ │               │ │               │
│ - Schema-aware│ │ - 4-stage     │ │ - Filter mode │ │ - General     │
│ - MV-optimized│ │   retrieval   │ │ - Assist mode │ │   medical     │
│ - Few-shot    │ │ - LLM rerank  │ │ - Combines    │ │   knowledge   │
│ - Retry logic │ │ - MMR diverse │ │   SQL + RAG   │ │               │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Guardrails                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Grounding  │  │ Confidence  │  │  Source Attribution     │  │
│  │ Validation  │  │  Scoring    │  │  [Note X] citations     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline (4 Stages)

```
Query → [BM25 + Semantic] → RRF Fusion → LLM Reranker → MMR Filter → Context
         (dual retrieval)    (k=60)      (relevance)    (diversity)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | qwen2.5:14b via Ollama |
| **Embeddings** | nomic-embed-text (768-dim) |
| **Vector Store** | ChromaDB (local) |
| **Database** | SQLite with indexes + materialized views |
| **Backend** | FastAPI + Python |
| **Frontend** | Tailwind CSS + vanilla JS |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Installation

```bash
# Clone the repository
git clone https://github.com/jarharsh1/Medical-Chatbot-using-RAG-and-NLP2SqL.git
cd Medical-Chatbot-using-RAG-and-NLP2SqL

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### Run the Server

```bash
python -m backend.app
```

Open http://localhost:8000 in your browser.

---

## Project Structure

```
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── config.py              # Centralized constants
│   ├── agents/
│   │   ├── router.py          # Query classification
│   │   ├── sql_agent.py       # Text-to-SQL generation
│   │   ├── rag_agent.py       # RAG-based answers
│   │   ├── hybrid_agent.py    # Combined SQL + RAG
│   │   ├── orchestrator.py    # Multi-part question handling
│   │   ├── decomposer.py      # Question decomposition
│   │   └── query_rewriter.py  # Vague → measurable queries
│   ├── rag/
│   │   ├── retriever.py       # 4-stage retrieval pipeline
│   │   ├── vectorstore.py     # ChromaDB integration
│   │   ├── bm25.py            # BM25 sparse retrieval
│   │   ├── reranker.py        # LLM-based reranking
│   │   └── embeddings.py      # Ollama embeddings
│   ├── guardrails/
│   │   ├── grounding.py       # Hallucination detection
│   │   ├── confidence.py      # Margin-based scoring
│   │   └── attribution.py     # Source citation
│   ├── memory/
│   │   ├── conversation.py    # Session memory
│   │   └── query_result_cache.py  # LRU cache
│   └── scripts/
│       └── optimize_database.py   # Indexes + materialized views
├── frontend/
│   ├── index.html             # Main UI
│   ├── app.js                 # Frontend logic
│   └── style.css              # Custom styles
└── data/                      # CSV data files
```

---

## Example Queries

| Type | Example |
|------|---------|
| **SQL** | "How many patients have diabetes?" |
| **RAG** | "What symptoms are mentioned in hypertension notes?" |
| **Hybrid** | "What medications are prescribed for patients with chest pain?" |
| **Multi-part** | "What are the top 5 clinics for hyperlipidemia patients and what treatments are recommended?" |
| **Knowledge** | "What causes gout?" |

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/query` | Main chat endpoint |
| `GET /api/health` | Health check |
| `GET /api/filters` | Get filter options for dashboard |
| `POST /api/dashboard` | Fetch dashboard data |
| `GET /api/cache/stats` | Cache statistics |
| `POST /api/cache/clear` | Clear query cache |
| `POST /api/db/refresh-views` | Refresh materialized views |

---

## Scalability Features

For databases with 1M+ rows:

1. **27 Database Indexes** - On frequently queried columns
2. **6 Materialized Views** - Pre-computed aggregations:
   - `mv_condition_stats` - Patient counts by condition
   - `mv_clinic_stats` - Clinic performance metrics
   - `mv_doctor_stats` - Doctor workload stats
   - `mv_medication_stats` - Prescription analytics
3. **Query Result Cache** - LRU with 1hr TTL, instant response on cache hit

Run optimization script:
```bash
python -m backend.scripts.optimize_database
```

---

## License

- **Code**: MIT License
- **Data**: Ensure compliance with HIPAA/GDPR regulations
