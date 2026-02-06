"""
Question Decomposer: splits multi-part questions for independent routing.

Detects when a question has multiple distinct information needs
(e.g., SQL count + RAG content + general knowledge) and decomposes
them for separate handling.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from backend.agents.prompts import DECOMPOSE_PROMPT
from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)

VALID_ROUTES = {"sql", "rag", "hybrid", "knowledge"}


def _parse_json_array(text: str) -> Optional[List[Dict]]:
    """Extract and parse JSON array from LLM response."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def decompose_question(question: str) -> List[Dict[str, Any]]:
    """
    Analyze a question and decompose into sub-questions if needed.

    Returns list of dicts:
    [
        {
            "sub_question": "...",
            "route": "sql" | "rag" | "hybrid" | "knowledge",
            "depends_on": int | None
        },
        ...
    ]

    For simple questions, returns single-element list.
    """
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        prompt = DECOMPOSE_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip()

        parsed = _parse_json_array(raw)
        if not parsed:
            logger.warning(f"Failed to parse decomposition response: {raw[:200]}")
            return [{"sub_question": question, "route": "hybrid", "depends_on": None}]

        # Validate and clean up
        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            sub_q = item.get("sub_question", "").strip()
            route = item.get("route", "hybrid").lower()
            depends = item.get("depends_on")

            if not sub_q:
                continue
            if route not in VALID_ROUTES:
                route = "hybrid"
            if depends is not None and not isinstance(depends, int):
                depends = None

            result.append({
                "sub_question": sub_q,
                "route": route,
                "depends_on": depends,
            })

        if not result:
            logger.warning("Decomposition returned empty result, using original question")
            return [{"sub_question": question, "route": "hybrid", "depends_on": None}]

        logger.info(f"Decomposed question into {len(result)} parts: {[r['route'] for r in result]}")
        return result

    except Exception as e:
        logger.error(f"Decomposition failed: {e}. Using original question.")
        return [{"sub_question": question, "route": "hybrid", "depends_on": None}]


def is_simple_question(question: str) -> bool:
    """
    Quick heuristic check if question is likely simple (single-part).
    Used to skip LLM decomposition for obvious cases.
    """
    # Count question marks
    q_marks = question.count("?")
    if q_marks > 1:
        return False

    # Check for multi-part indicators
    multi_indicators = [
        " and also ",
        " as well as ",
        ". also ",
        ". what ",
        ". how ",
        ". who ",
        "? what ",
        "? how ",
        "? who ",
        "firstly",
        "secondly",
        "additionally",
    ]
    lower = question.lower()
    for indicator in multi_indicators:
        if indicator in lower:
            return False

    # Check word count - very long questions are often multi-part
    if len(question.split()) > 30:
        return False

    return True
