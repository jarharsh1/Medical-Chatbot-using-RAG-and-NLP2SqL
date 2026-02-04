"""
Short-Term Memory: per-session conversation buffer.

Stores the last N turns (question + answer) per session_id.
Formatted as conversation context string for LLM prompts.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from backend.config import MAX_CONVERSATION_TURNS

logger = logging.getLogger(__name__)


class ConversationMemory:
    """In-memory conversation buffer keyed by session_id."""

    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS):
        self.max_turns = max_turns
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns))

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        query_type: str = "",
        sql_query: Optional[str] = None,
        source_doc_ids: Optional[List[str]] = None,
    ):
        """Record a conversation turn."""
        self._sessions[session_id].append({
            "role": "user",
            "content": question,
            "timestamp": time.time(),
        })
        self._sessions[session_id].append({
            "role": "assistant",
            "content": answer,
            "query_type": query_type,
            "sql_query": sql_query,
            "source_doc_ids": source_doc_ids or [],
            "timestamp": time.time(),
        })

    def get_context(self, session_id: str) -> str:
        """
        Format conversation history as a context string for LLM prompts.

        Returns empty string if no history.
        """
        if session_id not in self._sessions:
            return ""

        turns = self._sessions[session_id]
        if not turns:
            return ""

        parts = []
        for turn in turns:
            role = turn["role"].capitalize()
            content = turn["content"]
            # Truncate long answers to save context budget
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get raw turn history for a session."""
        if session_id not in self._sessions:
            return []
        return list(self._sessions[session_id])

    def clear_session(self, session_id: str):
        """Clear history for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Module-level singleton
_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
