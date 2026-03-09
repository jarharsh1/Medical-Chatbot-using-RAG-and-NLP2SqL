"""
Centralized configuration for the Medical AI Backend.
All paths, model names, thresholds, and feature toggles in one place.
"""

import os
import threading

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
# Recommended models (in order of capability):
#   - qwen2.5:14b     (best overall, needs ~9GB VRAM)
#   - llama3.1:8b     (good balance, needs ~5GB VRAM)
#   - deepseek-coder-v2:16b  (best for SQL, needs ~10GB VRAM)
#   - llama3.2        (fast but weak, needs ~2GB VRAM)
#
# To switch: ollama pull <model_name>, then change LLM_MODEL below
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:14b")
LLM_TEMPERATURE = 0

# Per-request model override via thread-local (used by model switcher UI)
_thread_local = threading.local()


def get_active_model() -> str:
    """Return the model for the current thread (falls back to LLM_MODEL)."""
    return getattr(_thread_local, "model_override", None) or LLM_MODEL


def set_thread_model(model: str | None) -> None:
    """Set (or clear) a per-request model override for the current thread."""
    _thread_local.model_override = model or None

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
RERANK_ENABLED = False   # disabled: LLM reranker adds 150-200s latency on CPU; RRF scores sufficient
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

# ---------------------------
# OLLAMA / RELIABILITY
# ---------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = 120  # Max time for LLM calls
OLLAMA_HEALTH_CACHE_TTL = 30  # Seconds between health checks
OLLAMA_RETRY_ATTEMPTS = 2     # Retries on transient failures
OLLAMA_RETRY_BACKOFF = 1.0    # Seconds between retries
