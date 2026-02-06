"""
Answer Faithfulness Metric

Measures if the generated answer is grounded in the retrieved context.
Extends guardrails/grounding.py with claim-level analysis.

Faithfulness = (claims supported by context) / (total claims)

High faithfulness = answer doesn't hallucinate
Low faithfulness = answer makes unsupported claims
"""

import json
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)


EXTRACT_CLAIMS_PROMPT = """Extract the factual claims from this answer. A claim is a statement that can be verified as true or false.

Answer: {answer}

Return a JSON array of claims:
["claim 1", "claim 2", "claim 3"]

Rules:
- Extract only factual statements, not opinions or hedged statements
- Each claim should be a single, atomic fact
- Ignore filler phrases like "Based on the records..."
- Maximum 10 claims

Return ONLY the JSON array, no other text."""


VERIFY_CLAIM_PROMPT = """Determine if this claim is supported by the context.

Claim: {claim}

Context:
{context}

Is this claim supported by the context?
Return ONLY a JSON object:
{{"supported": true/false, "evidence": "quote from context or 'not found'"}}"""


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    supported: bool
    evidence: str


def extract_claims(answer: str, llm: Optional[ChatOllama] = None) -> List[str]:
    """
    Extract factual claims from an answer.

    Args:
        answer: Generated answer text
        llm: Optional LLM instance

    Returns:
        List of claim strings
    """
    if not answer or len(answer.strip()) < 20:
        return []

    if llm is None:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = EXTRACT_CLAIMS_PROMPT.format(answer=answer[:2000])

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        claims = json.loads(content)

        if isinstance(claims, list):
            return [str(c).strip() for c in claims if c][:10]

        return []

    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        # Fallback: split by sentences
        sentences = re.split(r'[.!?]+', answer)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:10]


def verify_claim(
    claim: str,
    context: str,
    llm: Optional[ChatOllama] = None,
) -> ClaimVerification:
    """
    Verify if a claim is supported by the context.

    Args:
        claim: The claim to verify
        context: Retrieved context to check against
        llm: Optional LLM instance

    Returns:
        ClaimVerification with supported status and evidence
    """
    if llm is None:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Truncate context if too long
    context_truncated = context[:3000] if len(context) > 3000 else context

    prompt = VERIFY_CLAIM_PROMPT.format(
        claim=claim,
        context=context_truncated,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)

        return ClaimVerification(
            claim=claim,
            supported=bool(result.get("supported", False)),
            evidence=str(result.get("evidence", ""))[:200],
        )

    except Exception as e:
        logger.warning(f"Claim verification failed: {e}")
        # Conservative: mark as unsupported if verification fails
        return ClaimVerification(
            claim=claim,
            supported=False,
            evidence=f"verification_error: {str(e)[:50]}",
        )


def compute_faithfulness(
    answer: str,
    context: str,
    max_claims: int = 10,
) -> Dict:
    """
    Compute answer faithfulness score.

    Args:
        answer: Generated answer
        context: Retrieved context used to generate answer
        max_claims: Maximum claims to verify

    Returns:
        {
            "score": 0.85,
            "total_claims": 7,
            "supported_claims": 6,
            "unsupported_claims": 1,
            "claim_details": [
                {"claim": "...", "supported": true, "evidence": "..."},
                ...
            ],
            "interpretation": "high",
            "unsupported_list": ["claim that wasn't supported"]
        }
    """
    if not answer or not context:
        return {
            "score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "claim_details": [],
            "interpretation": "no_data",
            "unsupported_list": [],
        }

    # Create LLM instance once
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Step 1: Extract claims
    claims = extract_claims(answer, llm)[:max_claims]

    if not claims:
        return {
            "score": 1.0,  # No claims = nothing to verify
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "claim_details": [],
            "interpretation": "no_claims",
            "unsupported_list": [],
        }

    # Step 2: Verify each claim
    verifications = [verify_claim(claim, context, llm) for claim in claims]

    # Step 3: Compute metrics
    supported = sum(1 for v in verifications if v.supported)
    unsupported = len(verifications) - supported
    score = supported / len(verifications)

    # Interpretation
    if score >= 0.9:
        interpretation = "excellent"
    elif score >= 0.7:
        interpretation = "good"
    elif score >= 0.5:
        interpretation = "moderate"
    else:
        interpretation = "poor"

    return {
        "score": round(score, 3),
        "total_claims": len(claims),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "claim_details": [
            {
                "claim": v.claim,
                "supported": v.supported,
                "evidence": v.evidence,
            }
            for v in verifications
        ],
        "interpretation": interpretation,
        "unsupported_list": [v.claim for v in verifications if not v.supported],
    }


def compute_faithfulness_fast(
    answer: str,
    context: str,
) -> Dict:
    """
    Fast faithfulness check using keyword overlap (no LLM calls).

    Less accurate but much faster. Use for real-time monitoring.

    Args:
        answer: Generated answer
        context: Retrieved context

    Returns:
        Approximate faithfulness metrics
    """
    if not answer or not context:
        return {"score": 0.0, "method": "keyword_overlap", "interpretation": "no_data"}

    # Tokenize and find key entities
    import re
    from collections import Counter

    def extract_entities(text: str) -> set:
        """Extract potential entities (capitalized words, numbers, medical terms)."""
        # Find capitalized words, numbers with units, medical-looking terms
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|%|mmHg)\b',  # Measurements
            r'\b(?:patient|medication|diagnosis|treatment|symptom|condition)\b',
        ]

        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(m.lower() for m in matches)

        return entities

    context_entities = extract_entities(context)
    answer_entities = extract_entities(answer)

    if not answer_entities:
        return {"score": 1.0, "method": "keyword_overlap", "interpretation": "no_entities"}

    # What fraction of answer entities appear in context?
    overlap = answer_entities & context_entities
    score = len(overlap) / len(answer_entities)

    if score >= 0.8:
        interpretation = "high"
    elif score >= 0.5:
        interpretation = "moderate"
    else:
        interpretation = "low"

    return {
        "score": round(score, 3),
        "method": "keyword_overlap",
        "answer_entities": len(answer_entities),
        "context_entities": len(context_entities),
        "overlapping": len(overlap),
        "interpretation": interpretation,
    }
