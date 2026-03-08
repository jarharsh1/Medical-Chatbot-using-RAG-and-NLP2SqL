"""
Query Router: classifies incoming questions as SQL, RAG, or HYBRID.

Uses LLM-based classification with structured prompt.
Falls back to HYBRID if classification fails (safest default).
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from backend.agents.prompts import ROUTER_PROMPT
from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)

VALID_TYPES = {"sql", "rag", "hybrid", "knowledge"}


def classify_query(question: str) -> str:
    """
    Classify a user question into: "sql", "rag", or "hybrid".

    Returns lowercase string. Falls back to "hybrid" on failure.
    """
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        prompt = ROUTER_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip().lower()

        # Extract the classification word
        for valid in VALID_TYPES:
            if valid in raw:
                logger.info(f"Query classified as: {valid.upper()} — '{question[:60]}...'")
                return valid

        logger.warning(f"Router returned unrecognized output: '{raw}'. Defaulting to hybrid.")
        return "hybrid"

    except Exception as e:
        logger.error(f"Router classification failed: {e}. Defaulting to hybrid.")
        return "hybrid"
