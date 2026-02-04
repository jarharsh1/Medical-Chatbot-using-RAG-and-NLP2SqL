"""
Centralized configuration for the Medical AI Backend.
All paths, model names, thresholds, and feature toggles in one place.
"""

import os

# ---------------------------
# PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # project root
DB_PATH = os.path.join(BASE_DIR, "medical_records.db")
DB_URI = f"sqlite:///{DB_PATH}"
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

DATA_DIR = None
for path in [
    os.path.join(PROJECT_ROOT, "data"),       # <project_root>/data/
    os.path.join(BASE_DIR, "data"),            # backend/data/ (legacy)
    os.path.join(os.getcwd(), "data"),         # cwd/data/
]:
    if os.path.exists(path) and os.path.isdir(path):
        DATA_DIR = path
        break

# ---------------------------
# MODELS
# ---------------------------
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0
EMBED_MODEL = "nomic-embed-text"
EMBED_MODEL_VERSION = "1.0"
EMBED_DIMENSIONS = 768

# ---------------------------
# CHUNKING
# ---------------------------
CHUNK_MODE = "note"  # "note" (1 note = 1 chunk) or "encounter" (group by patient+visit_date)
CHUNK_VERSION = 1

# ---------------------------
# RETRIEVAL
# ---------------------------
BM25_TOP_K = 20
SEMANTIC_TOP_K = 20
RRF_K = 60              # RRF constant
RERANK_TOP_K = 15        # docs to keep after re-ranking
RERANK_BATCH_SIZE = 5    # docs per LLM rerank call
RERANK_ENABLED = True    # set False to skip re-ranking for low-latency mode
MMR_LAMBDA = 0.7         # 0.7 relevance, 0.3 diversity
MMR_TOP_K = 8            # final docs after MMR

# ---------------------------
# CONTEXT WINDOW
# ---------------------------
CONTEXT_BUDGET_TOTAL = 4096
CONTEXT_BUDGET = {
    "system_prompt": 800,
    "schema": 500,
    "conversation": 600,
    "few_shot": 400,
    "retrieved_docs": 1200,
    "question": 100,
    "buffer": 396,
}

# ---------------------------
# CONFIDENCE THRESHOLDS
# ---------------------------
CONFIDENCE_WEIGHTS = {
    "retrieval_margin": 0.35,
    "coverage": 0.35,
    "llm_self_assessment": 0.30,
}

CONFIDENCE_THRESHOLDS = {
    "rag":    {"high": 0.7, "low": 0.4},
    "hybrid": {"high": 0.6, "low": 0.3},
    "sql":    {"high": 0.8, "low": 0.5},
}

# ---------------------------
# MEMORY
# ---------------------------
MAX_CONVERSATION_TURNS = 10
MAX_FEW_SHOT_EXAMPLES = 3

# ---------------------------
# SQL SAFETY
# ---------------------------
MAX_SQL_RETRIES = 3
SQL_BANNED_OPS = ["insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "detach"]

# ---------------------------
# SERVER
# ---------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
