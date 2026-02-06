# Medical AI Analytics Platform

A production-grade **Medical Chatbot** combining **RAG (Retrieval-Augmented Generation)** and **NLP-to-SQL** capabilities. Built with a multi-agent architecture that intelligently routes queries to the optimal data source—running **100% locally** for complete data privacy.

> **Why this matters**: Healthcare data is sensitive. This system ensures no patient data ever leaves your infrastructure while providing powerful AI-driven insights.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Component Deep Dive](#component-deep-dive)
- [RAG Pipeline Explained](#rag-pipeline-explained)
- [Agent System](#agent-system)
- [Scalability & Performance](#scalability--performance)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Key Features

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **100% Local Execution** | LLM + embeddings run via Ollama | Zero data leakage, HIPAA-friendly |
| **Multi-Agent Architecture** | 4 specialized agents route queries intelligently | Right tool for each question type |
| **Query Decomposition** | Breaks complex questions into sub-parts | Handles "What are top clinics AND what treatments?" |
| **4-Stage RAG Pipeline** | BM25 → Semantic → Rerank → Diversify | High precision + recall + diversity |
| **Guardrails** | Grounding validation + confidence scoring | Reduces hallucinations, builds trust |
| **1M+ Row Ready** | Indexes + materialized views + caching | Sub-second responses at scale |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Question                                  │
│            "What clinics do diabetes patients visit most?"               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY ORCHESTRATOR                                │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐     │
│  │  Decomposer  │ →  │   Router     │ →  │  Query Result Cache    │     │
│  │              │    │              │    │                        │     │
│  │ Splits multi-│    │ Classifies:  │    │ LRU cache with 1hr TTL │     │
│  │ part questions│   │ SQL/RAG/     │    │ Instant cache hits     │     │
│  │ into sub-parts│   │ Hybrid/Know  │    │                        │     │
│  └──────────────┘    └──────────────┘    └────────────────────────┘     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    SQL Agent    │   │    RAG Agent    │   │  Hybrid Agent   │
│                 │   │                 │   │                 │
│ Structured data │   │ Clinical note   │   │ Combines both   │
│ queries         │   │ content search  │   │ approaches      │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          GUARDRAILS LAYER                                │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐     │
│  │  Grounding   │    │  Confidence  │    │  Source Attribution    │     │
│  │  Validation  │    │   Scoring    │    │                        │     │
│  │              │    │              │    │ Every claim cites      │     │
│  │ Checks if    │    │ 35% margin + │    │ [Note X] source        │     │
│  │ answer is    │    │ 35% coverage │    │                        │     │
│  │ grounded in  │    │ + 30% LLM    │    │                        │     │
│  │ retrieved    │    │ self-assess  │    │                        │     │
│  │ documents    │    │              │    │                        │     │
│  └──────────────┘    └──────────────┘    └────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Query Router

**What**: LLM-based classifier that determines query type.

**Why**: Different questions need different approaches:
- "How many patients have diabetes?" → **SQL** (needs COUNT aggregation)
- "What symptoms are in the notes?" → **RAG** (needs text understanding)
- "What meds for patients with chest pain?" → **Hybrid** (needs both)

**How**: Prompt engineering with clear category definitions. Falls back to "hybrid" on classification failure for safety.

```python
# Routes to: SQL | RAG | HYBRID | KNOWLEDGE
route = router.classify(question)
```

---

### 2. Query Decomposer

**What**: Breaks multi-part questions into independent sub-questions.

**Why**: Complex questions like *"What are top clinics for diabetes AND what treatments are common?"* contain multiple information needs that require different data sources.

**How**: LLM analyzes the question structure, identifies distinct parts, and assigns each a route.

```python
# Input: "Top clinics for diabetes and common treatments?"
# Output: [
#   {"sub_question": "Top clinics for diabetes?", "route": "sql"},
#   {"sub_question": "Common treatments for diabetes?", "route": "rag"}
# ]
```

---

### 3. Query Rewriter

**What**: Transforms vague/colloquial terms into precise, SQL-friendly language.

**Why**: Users ask *"famous clinic"* but databases have `patient_count`. The rewriter bridges this gap.

**How**: LLM rewrites while preserving intent:
- "famous clinic" → "clinic with most patients"
- "best doctor" → "doctor who has seen the most patients"

---

### 4. SQL Agent

**What**: Generates and executes SQLite queries from natural language.

**Why**: Structured data questions (counts, aggregations, filters) are best answered with precise SQL.

**Features**:
- **Schema-aware**: Injects actual table schemas into prompt
- **Few-shot examples**: Learns from similar past queries
- **Retry logic**: Auto-corrects on SQL errors (max 3 attempts)
- **MV-optimized**: Uses materialized views for fast aggregations
- **Safe execution**: Blocks DROP, DELETE, UPDATE queries

```python
# "How many patients have hypertension?"
# → SELECT patient_count FROM mv_condition_stats
#   WHERE condition_name LIKE '%Hypertension%'
```

---

### 5. RAG Agent

**What**: Retrieves relevant clinical notes and generates grounded answers.

**Why**: Unstructured text (symptoms, observations, treatment notes) can't be queried with SQL.

**Pipeline**:
```
Query → Dual Retrieval → Fusion → Rerank → Diversify → Generate
```

*(See RAG Pipeline section below for full details)*

---

### 6. Hybrid Agent

**What**: Combines SQL and RAG for questions needing both.

**Why**: *"What medications are prescribed for patients whose notes mention chest pain?"* needs:
1. RAG to find patients with "chest pain" in notes
2. SQL to get their prescription records

**Two Modes**:
| Mode | When Used | How It Works |
|------|-----------|--------------|
| **Filter** | High confidence RAG results, <100 patients | Uses RAG patient IDs as SQL WHERE clause |
| **Assist** | Lower confidence or too many results | RAG provides context to inform SQL generation |

---

### 7. Knowledge Agent

**What**: Answers general medical questions using LLM knowledge.

**Why**: Questions like *"What causes gout?"* aren't in our database—they need medical knowledge.

**Guardrail**: Clearly labels responses as general knowledge, not from patient records.

---

## RAG Pipeline Explained

Our 4-stage retrieval pipeline maximizes both **precision** and **diversity**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DUAL RETRIEVAL                          │
│                                                                      │
│   ┌─────────────────┐              ┌─────────────────┐              │
│   │      BM25       │              │    Semantic     │              │
│   │  (Sparse Search)│              │  (Dense Search) │              │
│   │                 │              │                 │              │
│   │ • Keyword match │              │ • Meaning match │              │
│   │ • Exact terms   │              │ • Synonyms work │              │
│   │ • Fast lookup   │              │ • Context-aware │              │
│   │                 │              │                 │              │
│   │ "diabetes" finds│              │ "diabetes" finds│              │
│   │ docs with exact │              │ "high blood     │              │
│   │ word "diabetes" │              │ sugar" too      │              │
│   └────────┬────────┘              └────────┬────────┘              │
│            │ Top 20                         │ Top 20                │
│            └──────────────┬─────────────────┘                       │
│                           ▼                                          │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 2: RECIPROCAL RANK FUSION (RRF)                │
│                                                                      │
│   WHY: Combines rankings from different retrieval methods fairly     │
│                                                                      │
│   HOW: RRF_score = Σ 1/(k + rank)  where k=60                       │
│                                                                      │
│   • Doc ranked #1 in BM25 and #5 in Semantic:                       │
│     Score = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318         │
│                                                                      │
│   • Prevents any single method from dominating                       │
│   • Documents good in BOTH methods rise to top                       │
│                                                                      │
│   Output: ~38 unique documents, merged and re-scored                 │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: LLM RERANKING                            │
│                                                                      │
│   WHY: BM25/embeddings miss nuanced relevance                        │
│                                                                      │
│   HOW: LLM scores each doc's relevance to query (0.0 - 1.0)         │
│                                                                      │
│   Example:                                                           │
│   Query: "What symptoms do hypertension patients report?"            │
│                                                                      │
│   Doc A: "Patient has hypertension. BP 140/90." → Score: 0.3        │
│          (mentions condition but not symptoms)                       │
│                                                                      │
│   Doc B: "Hypertension patient reports headaches, dizziness,        │
│           and fatigue." → Score: 0.95                                │
│          (directly answers the question)                             │
│                                                                      │
│   Output: Top 15 documents by relevance score                        │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 4: MMR DIVERSITY FILTER                           │
│                                                                      │
│   WHY: Avoid redundant documents saying the same thing               │
│                                                                      │
│   HOW: Maximal Marginal Relevance (λ=0.7)                           │
│                                                                      │
│   MMR = λ × Relevance - (1-λ) × Max_Similarity_to_Selected          │
│                                                                      │
│   • λ=0.7 means 70% relevance, 30% diversity                        │
│   • Each new doc must add NEW information                            │
│   • Prevents 5 docs all saying "patient has diabetes"               │
│                                                                      │
│   Output: 8 diverse, highly relevant documents                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Pipeline?

| Stage | Problem Solved |
|-------|----------------|
| **Dual Retrieval** | Keywords alone miss synonyms; embeddings alone miss exact matches |
| **RRF Fusion** | Fairly combines different ranking methods without bias |
| **LLM Reranking** | Understands *intent*, not just word overlap |
| **MMR Diversity** | Ensures varied information, not repetitive docs |

---

## Guardrails System

### 1. Grounding Validation

**Problem**: LLMs can hallucinate facts not in the source documents.

**Solution**: After generating an answer, we check if every claim is supported by the retrieved documents.

```python
# Returns: {"is_grounded": True, "grounding_score": 0.92, "unsupported_claims": []}
```

### 2. Confidence Scoring

**Problem**: Not all answers are equally reliable.

**Solution**: Multi-factor confidence score:

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| **Retrieval Margin** | 35% | Gap between top doc and others (higher = more decisive) |
| **Query Coverage** | 35% | How much of the query terms appear in retrieved docs |
| **LLM Self-Assessment** | 30% | Model's own confidence in its answer |

```python
confidence = 0.35 * retrieval_margin + 0.35 * coverage + 0.30 * llm_confidence
```

### 3. Source Attribution

**Problem**: Users need to verify AI claims.

**Solution**: Every factual claim cites its source:

> "Patient shows signs of hypertension **[Note 1042]** with BP readings of 150/95 **[Note 1042]**."

---

## Scalability & Performance

### The Problem

At 1M+ rows, naive queries like `SELECT COUNT(*) FROM clinical_notes WHERE condition_name LIKE '%Diabetes%'` take **5+ seconds**.

### Our Solution

#### 1. Database Indexes (27 total)

```sql
-- Example: Makes condition lookups 100x faster
CREATE INDEX idx_clinical_notes_condition ON clinical_notes(condition_name);
CREATE INDEX idx_prescriptions_medication ON prescriptions(medication_name);
CREATE INDEX idx_patients_clinic ON patients(clinic_id);
```

#### 2. Materialized Views (6 pre-computed tables)

| View | Use Case | Speed Improvement |
|------|----------|-------------------|
| `mv_condition_stats` | "How many patients have X?" | 5000ms → 2ms |
| `mv_clinic_stats` | "Which clinic has most patients?" | 3000ms → 1ms |
| `mv_doctor_stats` | "Who is the busiest doctor?" | 2000ms → 1ms |
| `mv_medication_stats` | "Most prescribed medication?" | 4000ms → 2ms |

```sql
-- Instead of joining 3 tables and aggregating...
SELECT patient_count FROM mv_condition_stats WHERE condition_name LIKE '%Diabetes%';
-- Returns in <5ms
```

#### 3. Query Result Cache

```python
# LRU Cache Configuration
max_entries: 1000
ttl: 3600 seconds (1 hour)
thread_safe: True

# Same query twice? Second one is instant (0ms)
```

#### Performance Summary

| Query Type | Without Optimization | With Optimization |
|------------|---------------------|-------------------|
| Count by condition | ~5000ms | ~2ms (MV) |
| Repeated query | ~5000ms | ~0ms (cache) |
| Filter by clinic | ~3000ms | ~50ms (index) |

---

## Tech Stack

| Component | Choice | Why This? |
|-----------|--------|-----------|
| **LLM** | qwen2.5:14b via Ollama | Best open-source balance of speed + quality for medical domain |
| **Embeddings** | nomic-embed-text (768-dim) | Optimized for retrieval, runs locally |
| **Vector Store** | ChromaDB | Simple, local, no server needed |
| **Sparse Search** | BM25 (rank_bm25) | Industry standard, fast keyword matching |
| **Database** | SQLite | Zero-config, portable, surprisingly fast with indexes |
| **Backend** | FastAPI | Async support, auto-generated docs, type hints |
| **Frontend** | Tailwind CSS + vanilla JS | No build step, fast iteration |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed

### Installation

```bash
# Clone repository
git clone https://github.com/jarharsh1/Medical-Chatbot-using-RAG-and-NLP2SqL.git
cd Medical-Chatbot-using-RAG-and-NLP2SqL

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# (Optional) Optimize database for large datasets
python -m backend.scripts.optimize_database
```

### Run

```bash
python -m backend.app
```

Open http://localhost:8000

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Main chat endpoint. Body: `{"question": "..."}` |
| `/api/health` | GET | Health check |
| `/api/filters` | GET | Dashboard filter options |
| `/api/dashboard` | POST | Paginated dashboard data |
| `/api/cache/stats` | GET | Cache hit/miss statistics |
| `/api/cache/clear` | POST | Clear query cache |
| `/api/db/refresh-views` | POST | Refresh materialized views |

---

## Project Structure

```
├── backend/
│   ├── app.py                    # FastAPI application entry point
│   ├── config.py                 # Centralized configuration constants
│   │
│   ├── agents/                   # Multi-agent system
│   │   ├── router.py             # Query classification (SQL/RAG/Hybrid/Knowledge)
│   │   ├── sql_agent.py          # Text-to-SQL with retry logic
│   │   ├── rag_agent.py          # RAG-based answer generation
│   │   ├── hybrid_agent.py       # Combined SQL + RAG (filter/assist modes)
│   │   ├── orchestrator.py       # Coordinates multi-part questions
│   │   ├── decomposer.py         # Splits complex questions
│   │   ├── query_rewriter.py     # Vague → precise term transformation
│   │   └── prompts.py            # All LLM prompts centralized
│   │
│   ├── rag/                      # Retrieval pipeline
│   │   ├── retriever.py          # 4-stage retrieval orchestration
│   │   ├── vectorstore.py        # ChromaDB integration
│   │   ├── bm25.py               # Sparse retrieval
│   │   ├── reranker.py           # LLM-based reranking
│   │   ├── embeddings.py         # Ollama embedding wrapper
│   │   ├── chunking.py           # Document chunking strategies
│   │   └── context_window.py     # Token budget management
│   │
│   ├── guardrails/               # Safety & reliability
│   │   ├── grounding.py          # Hallucination detection
│   │   ├── confidence.py         # Multi-factor confidence scoring
│   │   └── attribution.py        # Source citation extraction
│   │
│   ├── memory/                   # State management
│   │   ├── conversation.py       # Session-based chat history
│   │   └── query_result_cache.py # LRU cache with TTL
│   │
│   ├── scripts/                  # Utilities
│   │   ├── optimize_database.py  # Create indexes + materialized views
│   │   └── evaluate_queries.py   # Query performance testing
│   │
│   └── observability/            # Logging & monitoring
│       └── logger.py             # Structured logging to `runs` table
│
├── frontend/
│   ├── index.html                # Main UI with Tailwind
│   ├── app.js                    # Chat logic + dashboard
│   └── style.css                 # Custom component styles
│
├── data/                         # CSV source files
│   ├── patients.csv
│   ├── clinical_notes.csv
│   ├── prescriptions.csv
│   └── clinics.csv
│
└── evaluation/                   # Quality assurance
    └── golden_set.json           # 50 test cases for regression testing
```

---

## Example Queries

| Type | Example | What Happens |
|------|---------|--------------|
| **SQL** | "How many patients have diabetes?" | Routes to SQL → Uses `mv_condition_stats` → Returns count |
| **RAG** | "What symptoms do hypertension patients report?" | Routes to RAG → 4-stage retrieval → Generates cited answer |
| **Hybrid** | "What meds are prescribed for patients with chest pain in notes?" | RAG finds patients → SQL gets their prescriptions |
| **Multi-part** | "Top 5 clinics for diabetes AND common treatments?" | Decomposes → SQL for clinics, RAG for treatments → Combines |
| **Knowledge** | "What causes gout?" | Routes to Knowledge agent → LLM medical knowledge |

---

## License

- **Code**: MIT License
- **Data**: Ensure compliance with HIPAA/GDPR regulations

---

## Contributing

Issues and PRs welcome! Please ensure any changes pass the evaluation suite:

```bash
python -m backend.scripts.evaluate_queries
```
