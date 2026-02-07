"""
Query Rewriter: Transforms vague/ambiguous questions into precise, SQL-friendly queries.

Handles:
- Vague terms ("famous", "popular", "best") → measurable metrics (COUNT, MAX)
- Implicit aggregations → explicit GROUP BY hints
- Ambiguous references → specific column names
- Follow-up context resolution
"""

import logging
import re
from typing import Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)

# Rewriting rules for common vague patterns
REWRITE_RULES = [
    # "famous/popular/best clinic" → "clinic with most patients"
    (r"\b(famous|popular|best|top)\s+(clinic|hospital|center)\b",
     "clinic with the highest number of patients"),

    # "famous/popular doctor" → "doctor with most patients"
    (r"\b(famous|popular|best|top)\s+(doctor|physician|dr\.?)\b",
     "doctor who has seen the most patients"),

    # "common/frequent medication" → "most prescribed medication"
    (r"\b(common|frequent|popular)\s+(medication|medicine|drug|prescription)\b",
     "most frequently prescribed medication (by count)"),

    # "common condition/disease" → "condition with most patients"
    (r"\b(common|frequent|prevalent)\s+(condition|disease|illness|diagnosis)\b",
     "condition affecting the most patients"),

    # "how many" without specific metric → count distinct
    (r"how many\s+(patient|people|person)s?\s+(have|with|diagnosed)",
     "count of distinct patients with"),
]

# Medical term normalization: maps specific terms to their root for broader SQL matching
# These suffixes often cause LIKE mismatches (e.g., "thyroidism" vs "Thyroid Cancer")
MEDICAL_ROOT_MAP = {
    "thyroidism": "thyroid",
    "hypothyroidism": "thyroid",
    "hyperthyroidism": "thyroid",
    "diabetic": "diabet",
    "diabetes": "diabet",
    "hypertensive": "hypertens",
    "hypertension": "hypertens",
    "arthritis": "arthr",
    "osteoarthritis": "arthr",
    "rheumatoid arthritis": "arthr",
    "asthmatic": "asthm",
    "pneumonia": "pneumon",
    "bronchitis": "bronch",
    "dermatitis": "dermat",
    "hepatitis": "hepat",
    "gastritis": "gastr",
    "sinusitis": "sinus",
    "tendinitis": "tendin",
    "anemia": "anem",
    "anaemia": "anem",
    "epilepsy": "epilep",
    "epileptic": "epilep",
}

REWRITER_PROMPT = """You are a query precision specialist for a medical database.

Your job is to rewrite vague or ambiguous questions into precise, measurable queries.

DATABASE SCHEMA:
- patients: patient_id, full_name, dob, gender, insurance_provider, clinic_id
- clinics: clinic_id, name, location
- clinical_notes: note_id, patient_id, visit_date, doctor_name, condition_name, note_text
- prescriptions: rx_id, patient_id, medication_name, dosage, status

REWRITING RULES:
1. "famous/popular/best clinic" → "clinic with the highest patient count"
2. "famous/popular doctor" → "doctor who has treated the most patients"
3. "common medication" → "most prescribed medication by count"
4. "consulting doctors there" → "doctors who have clinical_notes at that clinic"
5. Multi-part questions should be preserved but clarified

IMPORTANT:
- Preserve the user's intent
- Make implicit metrics explicit (counts, aggregations)
- If the question is already precise, return it unchanged
- Keep it as a natural language question (not SQL)

Original question: {question}

Rewritten question (or original if already precise):"""


def apply_rule_based_rewrites(question: str) -> str:
    """Apply regex-based rewriting rules for common patterns."""
    rewritten = question
    for pattern, replacement in REWRITE_RULES:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return rewritten


def rewrite_query(question: str, use_llm: bool = True) -> Tuple[str, bool]:
    """
    Rewrite a vague question into a precise, SQL-friendly query.

    Returns:
        Tuple of (rewritten_question, was_rewritten)
    """
    original = question.strip()

    # Step 1: Apply rule-based rewrites
    rule_rewritten = apply_rule_based_rewrites(original)

    # If rule-based rewrite changed something significant, use that
    if rule_rewritten.lower() != original.lower():
        logger.info(f"Rule-based rewrite: '{original[:50]}' → '{rule_rewritten[:50]}'")
        return rule_rewritten, True

    # Step 2: Check if LLM rewriting is needed
    vague_indicators = [
        "famous", "popular", "best", "top", "common", "frequent",
        "good", "bad", "main", "important", "key", "typical"
    ]

    needs_llm_rewrite = any(ind in original.lower() for ind in vague_indicators)

    if not needs_llm_rewrite or not use_llm:
        return original, False

    # Step 3: Use LLM for complex rewrites
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        prompt = REWRITER_PROMPT.format(question=original)

        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = (response.content or "").strip()

        # Clean up LLM response
        rewritten = rewritten.strip('"\'')

        # Validate the rewrite isn't garbage
        if len(rewritten) < 10 or len(rewritten) > len(original) * 3:
            logger.warning(f"LLM rewrite rejected (length issue): {rewritten[:100]}")
            return original, False

        if rewritten.lower() != original.lower():
            logger.info(f"LLM rewrite: '{original[:50]}' → '{rewritten[:50]}'")
            return rewritten, True

        return original, False

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return original, False


def rewrite_for_sql(question: str) -> str:
    """
    Specialized rewriter for SQL-bound queries.
    Adds explicit hints for aggregations and joins.
    """
    rewritten, _ = rewrite_query(question)

    # Add SQL hints for common patterns
    hints = []

    lower = rewritten.lower()

    # Hint: medical term root matching for LIKE filters
    for term, root in MEDICAL_ROOT_MAP.items():
        if term in lower:
            hints.append(f"(Use LIKE '%{root.title()}%' for condition matching — covers all variants)")
            break

    # Hint: clinic queries need JOIN with patients
    if "clinic" in lower and ("patient" in lower or "most" in lower or "count" in lower):
        hints.append("(Join clinics with patients via clinic_id)")

    # Hint: doctor queries need clinical_notes
    if "doctor" in lower and ("patient" in lower or "most" in lower):
        hints.append("(Doctors are in clinical_notes.doctor_name, not a separate table)")

    # Hint: condition queries need clinical_notes
    if any(word in lower for word in ["condition", "diagnosis", "disease"]):
        hints.append("(Conditions are in clinical_notes.condition_name)")

    # Hint: medication queries need prescriptions
    if any(word in lower for word in ["medication", "medicine", "drug", "prescription"]):
        hints.append("(Medications are in prescriptions.medication_name)")

    if hints:
        rewritten = f"{rewritten}\n\nHints: {' '.join(hints)}"

    return rewritten
