"""
Embedding wrapper using Ollama's nomic-embed-text model.
768-dimensional vectors, runs 100% locally.

Requires: ollama pull nomic-embed-text
"""

import logging
from typing import List, Optional, Protocol

from backend.config import EMBED_MODEL

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Pluggable tokenizer interface for future upgrades."""
    def count_tokens(self, text: str) -> int: ...


class ApproxTokenizer:
    """Rough estimator: 1 token ~ 4 chars for English text."""
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


_embeddings_instance = None


def get_embeddings():
    """Lazy singleton for the embedding model."""
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            from langchain_ollama import OllamaEmbeddings
            _embeddings_instance = OllamaEmbeddings(model=EMBED_MODEL)
            logger.info(f"Initialized OllamaEmbeddings with model={EMBED_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    return _embeddings_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts and return their vectors."""
    emb = get_embeddings()
    return emb.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    emb = get_embeddings()
    return emb.embed_query(query)
