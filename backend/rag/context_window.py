"""
Context Window Manager: enforces token budgets to prevent "Lost in the Middle" degradation.

Budget: 4,096 tokens total (conservative — forces precision over quantity).
Pluggable tokenizer interface for future upgrades.
"""

import logging
from typing import Callable, Dict, List, Optional, Protocol

from langchain_core.documents import Document

from backend.config import CONTEXT_BUDGET, CONTEXT_BUDGET_TOTAL

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Interface for token counting. Swap in tiktoken or Ollama's tokenizer later."""
    def count_tokens(self, text: str) -> int: ...


class ApproxTokenizer:
    """Rough estimator: 1 token ~ 4 chars for English text."""
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class ContextWindowManager:
    """
    Manages the total context window budget across all components.
    Greedily fits content by priority/relevance order.
    """

    def __init__(
        self,
        max_tokens: int = CONTEXT_BUDGET_TOTAL,
        budgets: Optional[Dict[str, int]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.max_tokens = max_tokens
        self.budgets = budgets or CONTEXT_BUDGET
        self.tokenizer = tokenizer or ApproxTokenizer()

    def estimate_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def fit_documents(
        self,
        docs: List[Dict],
        budget: Optional[int] = None,
    ) -> List[Dict]:
        """
        Greedily fit documents into the token budget, prioritized by relevance.
        Docs should already be sorted by relevance (post-rerank + MMR).

        Each doc is expected to have a "content" key.
        Returns the subset of docs that fit.
        """
        budget = budget or self.budgets.get("retrieved_docs", 1200)
        fitted = []
        used_tokens = 0

        for doc in docs:
            content = doc.get("content", "")
            doc_tokens = self.estimate_tokens(content)

            if used_tokens + doc_tokens <= budget:
                fitted.append(doc)
                used_tokens += doc_tokens
            else:
                # Try truncation for partial fit
                remaining = budget - used_tokens
                if remaining > 50:
                    char_limit = remaining * 4  # reverse the 4-char estimate
                    truncated_content = content[:char_limit] + "..."
                    truncated_doc = dict(doc)
                    truncated_doc["content"] = truncated_content
                    truncated_doc["truncated"] = True
                    fitted.append(truncated_doc)
                break

        logger.debug(f"Fit {len(fitted)}/{len(docs)} docs into {budget} token budget ({used_tokens} used)")
        return fitted

    def fit_conversation(
        self,
        turns: List[str],
        budget: Optional[int] = None,
    ) -> List[str]:
        """
        Keep the most recent conversation turns that fit in the budget.
        Drops oldest turns first (sliding window).
        """
        budget = budget or self.budgets.get("conversation", 600)
        fitted = []
        used = 0

        for turn in reversed(turns):
            t = self.estimate_tokens(turn)
            if used + t <= budget:
                fitted.insert(0, turn)
                used += t
            else:
                break

        return fitted

    def fit_schema(
        self,
        schema_text: str,
        budget: Optional[int] = None,
    ) -> str:
        """
        Truncate schema to fit within budget.
        In practice, the schema retriever (Phase 2) selects only relevant tables,
        so this is a safety net.
        """
        budget = budget or self.budgets.get("schema", 500)
        tokens = self.estimate_tokens(schema_text)

        if tokens <= budget:
            return schema_text

        char_limit = budget * 4
        truncated = schema_text[:char_limit] + "\n... (schema truncated)"
        logger.warning(f"Schema truncated from {tokens} to ~{budget} tokens")
        return truncated

    def fit_few_shot(
        self,
        examples: List[str],
        budget: Optional[int] = None,
    ) -> List[str]:
        """Fit few-shot examples into budget, prioritizing most relevant first."""
        budget = budget or self.budgets.get("few_shot", 400)
        fitted = []
        used = 0

        for ex in examples:
            t = self.estimate_tokens(ex)
            if used + t <= budget:
                fitted.append(ex)
                used += t
            else:
                break

        return fitted

    def remaining_budget(self, used: Dict[str, int]) -> int:
        """Calculate remaining tokens given already-used budgets."""
        total_used = sum(used.values())
        return max(0, self.max_tokens - total_used)


# Module-level singleton
_context_manager = None


def get_context_manager() -> ContextWindowManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextWindowManager()
    return _context_manager
