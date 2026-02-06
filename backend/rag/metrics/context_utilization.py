"""
Context Utilization Metric

Measures how much of the retrieved context was actually used in the answer.
High utilization = retrieved docs were relevant and used.
Low utilization = wasted retrieval or answer ignores context.

No LLM calls required - uses token/n-gram overlap.
"""

import re
from typing import Dict, List, Set
from collections import Counter


# Common English stopwords to exclude from analysis
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "whom", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "same", "so", "than", "too", "very",
    "just", "also", "now", "here", "there", "then", "once", "if", "as",
}

# Medical terms to prioritize (these indicate meaningful overlap)
MEDICAL_TERMS = {
    "patient", "diagnosis", "treatment", "medication", "prescription",
    "symptom", "condition", "chronic", "acute", "clinical", "doctor",
    "physician", "nurse", "hospital", "clinic", "dosage", "mg", "ml",
    "daily", "twice", "blood", "pressure", "diabetes", "hypertension",
    "pain", "fever", "cough", "allergy", "surgery", "test", "lab",
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words.
    Removes punctuation and numbers-only tokens.
    """
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Filter very short tokens
    return [t for t in tokens if len(t) > 2]


def get_ngrams(tokens: List[str], n: int) -> Set[str]:
    """Generate n-grams from token list."""
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def compute_context_utilization(
    context: str,
    answer: str,
    include_bigrams: bool = True,
    include_trigrams: bool = False,
) -> Dict:
    """
    Compute how much of the context was utilized in the answer.

    Args:
        context: The retrieved documents/context provided to LLM
        answer: The generated answer
        include_bigrams: Include 2-gram overlap analysis
        include_trigrams: Include 3-gram overlap analysis

    Returns:
        {
            "score": 0.35,  # Overall utilization score (0-1)
            "unigram_overlap": 0.28,
            "bigram_overlap": 0.15,
            "context_tokens": 450,
            "answer_tokens": 120,
            "unique_context_terms": 180,
            "terms_used_in_answer": 52,
            "medical_terms_used": ["diabetes", "medication", "dosage"],
            "key_terms_from_context": ["hypertension", "metformin", "blood pressure"],
            "interpretation": "moderate"
        }
    """
    if not context or not answer:
        return {
            "score": 0.0,
            "unigram_overlap": 0.0,
            "bigram_overlap": 0.0,
            "context_tokens": 0,
            "answer_tokens": 0,
            "unique_context_terms": 0,
            "terms_used_in_answer": 0,
            "medical_terms_used": [],
            "key_terms_from_context": [],
            "interpretation": "no_data",
        }

    # Tokenize
    context_tokens = tokenize(context)
    answer_tokens = tokenize(answer)

    # Remove stopwords for meaningful comparison
    context_meaningful = [t for t in context_tokens if t not in STOPWORDS]
    answer_meaningful = [t for t in answer_tokens if t not in STOPWORDS]

    context_set = set(context_meaningful)
    answer_set = set(answer_meaningful)

    # Unigram overlap
    overlap = context_set & answer_set
    unigram_overlap = len(overlap) / len(context_set) if context_set else 0.0

    # Bigram overlap
    bigram_overlap = 0.0
    if include_bigrams:
        context_bigrams = get_ngrams(context_meaningful, 2)
        answer_bigrams = get_ngrams(answer_meaningful, 2)
        if context_bigrams:
            bigram_overlap = len(context_bigrams & answer_bigrams) / len(context_bigrams)

    # Trigram overlap (more strict)
    trigram_overlap = 0.0
    if include_trigrams:
        context_trigrams = get_ngrams(context_meaningful, 3)
        answer_trigrams = get_ngrams(answer_meaningful, 3)
        if context_trigrams:
            trigram_overlap = len(context_trigrams & answer_trigrams) / len(context_trigrams)

    # Find medical terms used
    medical_in_context = context_set & MEDICAL_TERMS
    medical_in_answer = answer_set & MEDICAL_TERMS
    medical_terms_used = list(medical_in_context & medical_in_answer)

    # Find key terms from context (most frequent non-stopwords)
    context_freq = Counter(context_meaningful)
    key_terms = [term for term, count in context_freq.most_common(10)]

    # Compute overall score (weighted combination)
    # Weight: 60% unigram, 30% bigram, 10% medical term bonus
    medical_bonus = min(0.1, len(medical_terms_used) * 0.02)
    score = (0.6 * unigram_overlap) + (0.3 * bigram_overlap) + medical_bonus

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # Interpretation
    if score >= 0.5:
        interpretation = "high"
    elif score >= 0.25:
        interpretation = "moderate"
    elif score >= 0.1:
        interpretation = "low"
    else:
        interpretation = "very_low"

    return {
        "score": round(score, 3),
        "unigram_overlap": round(unigram_overlap, 3),
        "bigram_overlap": round(bigram_overlap, 3),
        "trigram_overlap": round(trigram_overlap, 3) if include_trigrams else None,
        "context_tokens": len(context_tokens),
        "answer_tokens": len(answer_tokens),
        "unique_context_terms": len(context_set),
        "terms_used_in_answer": len(overlap),
        "medical_terms_used": medical_terms_used[:10],
        "key_terms_from_context": key_terms,
        "interpretation": interpretation,
    }


def compute_citation_coverage(
    answer: str,
    source_doc_ids: List[str],
) -> Dict:
    """
    Check how well the answer cites its sources.

    Args:
        answer: Generated answer text
        source_doc_ids: List of retrieved document IDs

    Returns:
        {
            "citations_found": ["note:1042", "note:1055"],
            "citations_missing": ["note:1089"],
            "coverage": 0.67,
            "all_cited": False
        }
    """
    # Find citations in answer (format: [Note note:XXXX] or [note:XXXX])
    citation_pattern = r'\[(?:Note\s+)?(note:\d+)\]'
    found_citations = set(re.findall(citation_pattern, answer, re.IGNORECASE))

    source_set = set(source_doc_ids)

    cited = found_citations & source_set
    missing = source_set - found_citations

    coverage = len(cited) / len(source_set) if source_set else 1.0

    return {
        "citations_found": list(cited),
        "citations_missing": list(missing),
        "coverage": round(coverage, 3),
        "all_cited": len(missing) == 0,
    }
