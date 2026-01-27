"""
Input Moderation Layer - LinkedIn Post Q2 Solution

Q: "User asks harmful query. How do you protect?"
A: Multi-layered defense BEFORE LLM sees the query

Defense Layers:
1. Pattern Matching (fast regex checks)
2. ML Classification (intent detection)
3. Dosage Manipulation Detection
4. Medical Abuse Detection

Critical for healthcare: Prevent "how to overdose", "increase dosage 10x", etc.
"""

import re
import logging
from typing import Tuple, List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModerationResult:
    """Result of moderation check"""
    is_safe: bool
    threat_level: ThreatLevel
    reason: str
    matched_pattern: Optional[str] = None
    intent: Optional[str] = None


class InputModerationLayer:
    """
    Multi-layered input moderation for healthcare queries.

    Implements defense-in-depth strategy:
    - Layer 1: Fast pattern matching (< 1ms)
    - Layer 2: Dosage manipulation detection
    - Layer 3: Medical abuse detection
    - Layer 4: ML classification (optional, slower)

    Example:
        moderator = InputModerationLayer()
        result = moderator.moderate("How do I take 10x my dosage?")
        if not result.is_safe:
            return error_response(result.reason)
    """

    def __init__(self, enable_ml_classification: bool = False):
        """
        Initialize moderation layer.

        Args:
            enable_ml_classification: Enable ML-based intent classification
                                     (slower but more accurate)
        """
        self.enable_ml_classification = enable_ml_classification
        self.harmful_patterns = self._load_harmful_patterns()
        self.dosage_patterns = self._load_dosage_patterns()
        self.medical_abuse_patterns = self._load_medical_abuse_patterns()

        # Statistics for monitoring
        self.stats = {
            "total_queries": 0,
            "blocked_queries": 0,
            "threat_levels": {level.value: 0 for level in ThreatLevel}
        }

    def moderate(self, query: str) -> ModerationResult:
        """
        Moderate user query through multiple layers.

        Args:
            query: User's natural language query

        Returns:
            ModerationResult with safety assessment

        Example:
            >>> moderator = InputModerationLayer()
            >>> result = moderator.moderate("What is diabetes?")
            >>> result.is_safe
            True
            >>> result = moderator.moderate("How to overdose on insulin?")
            >>> result.is_safe
            False
        """
        self.stats["total_queries"] += 1

        # Layer 1: Fast pattern matching
        result = self._check_harmful_patterns(query)
        if not result.is_safe:
            self._log_blocked_query(query, result)
            return result

        # Layer 2: Dosage manipulation detection
        result = self._check_dosage_manipulation(query)
        if not result.is_safe:
            self._log_blocked_query(query, result)
            return result

        # Layer 3: Medical abuse detection
        result = self._check_medical_abuse(query)
        if not result.is_safe:
            self._log_blocked_query(query, result)
            return result

        # Layer 4: ML classification (optional)
        if self.enable_ml_classification:
            result = self._ml_classify_intent(query)
            if not result.is_safe:
                self._log_blocked_query(query, result)
                return result

        # Query passed all checks
        self.stats["threat_levels"][ThreatLevel.SAFE.value] += 1
        return ModerationResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            reason="Query passed all safety checks"
        )

    def _check_harmful_patterns(self, query: str) -> ModerationResult:
        """Layer 1: Fast regex pattern matching"""
        query_lower = query.lower()

        for pattern_info in self.harmful_patterns:
            pattern = pattern_info["pattern"]
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.stats["blocked_queries"] += 1
                self.stats["threat_levels"][pattern_info["threat_level"]] += 1

                return ModerationResult(
                    is_safe=False,
                    threat_level=ThreatLevel(pattern_info["threat_level"]),
                    reason=pattern_info["reason"],
                    matched_pattern=pattern_info["name"]
                )

        return ModerationResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            reason="No harmful patterns detected"
        )

    def _check_dosage_manipulation(self, query: str) -> ModerationResult:
        """
        Layer 2: Detect attempts to manipulate dosages.

        Critical for healthcare: Prevent users from asking how to:
        - Increase dosages beyond prescribed
        - Take multiple doses
        - Override medical advice
        """
        query_lower = query.lower()

        for pattern_info in self.dosage_patterns:
            if re.search(pattern_info["pattern"], query_lower, re.IGNORECASE):
                self.stats["blocked_queries"] += 1
                self.stats["threat_levels"][ThreatLevel.HIGH.value] += 1

                return ModerationResult(
                    is_safe=False,
                    threat_level=ThreatLevel.HIGH,
                    reason=pattern_info["reason"],
                    matched_pattern=pattern_info["name"]
                )

        return ModerationResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            reason="No dosage manipulation detected"
        )

    def _check_medical_abuse(self, query: str) -> ModerationResult:
        """
        Layer 3: Detect medical abuse patterns.

        Detects:
        - Prescription fraud attempts
        - Drug seeking behavior
        - Illegal prescription requests
        """
        query_lower = query.lower()

        for pattern_info in self.medical_abuse_patterns:
            if re.search(pattern_info["pattern"], query_lower, re.IGNORECASE):
                self.stats["blocked_queries"] += 1
                self.stats["threat_levels"][ThreatLevel.CRITICAL.value] += 1

                return ModerationResult(
                    is_safe=False,
                    threat_level=ThreatLevel.CRITICAL,
                    reason=pattern_info["reason"],
                    matched_pattern=pattern_info["name"]
                )

        return ModerationResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            reason="No medical abuse detected"
        )

    def _ml_classify_intent(self, query: str) -> ModerationResult:
        """
        Layer 4: ML-based intent classification (optional, slower).

        Uses transformer model to detect:
        - Self-harm intent
        - Violence
        - Harmful medical advice requests

        Note: This is a placeholder for ML classification.
        In production, would use a fine-tuned BERT model.
        """
        # Placeholder: In production, load a fine-tuned classifier
        # Example: harmful_intent_classifier = pipeline("text-classification", model="...")
        # prediction = harmful_intent_classifier(query)

        # For now, return safe (implement when ML model is available)
        return ModerationResult(
            is_safe=True,
            threat_level=ThreatLevel.SAFE,
            reason="ML classification passed"
        )

    def _log_blocked_query(self, query: str, result: ModerationResult):
        """Log blocked queries for security monitoring"""
        logger.warning(
            f"BLOCKED QUERY | "
            f"Threat: {result.threat_level.value} | "
            f"Reason: {result.reason} | "
            f"Pattern: {result.matched_pattern} | "
            f"Query: {query[:100]}..."
        )

    def _load_harmful_patterns(self) -> List[Dict]:
        """
        Load harmful query patterns.

        Returns:
            List of pattern definitions with threat levels
        """
        return [
            {
                "name": "self_harm",
                "pattern": r"\b(kill|suicide|self.harm|end.my.life|hurt.myself)\b",
                "reason": "Self-harm intent detected",
                "threat_level": "critical"
            },
            {
                "name": "overdose_inquiry",
                "pattern": r"\b(overdose|lethal.dose|toxic.dose|how.many.to.die)\b",
                "reason": "Overdose inquiry detected",
                "threat_level": "critical"
            },
            {
                "name": "illegal_prescription",
                "pattern": r"\b(without.prescription|fake.prescription|forge.prescription|buy.without.doctor)\b",
                "reason": "Illegal prescription request",
                "threat_level": "critical"
            },
            {
                "name": "recreational_drug_abuse",
                "pattern": r"\b(get.high|recreational.use|abuse|snort|inject.to.get)\b",
                "reason": "Drug abuse inquiry",
                "threat_level": "high"
            },
            {
                "name": "pregnancy_termination",
                "pattern": r"\b(abort|terminate.pregnancy|miscarriage.methods|end.pregnancy.at.home)\b",
                "reason": "Unsafe pregnancy termination inquiry",
                "threat_level": "critical"
            }
        ]

    def _load_dosage_patterns(self) -> List[Dict]:
        """
        Load dosage manipulation patterns.

        Critical for healthcare safety.
        """
        return [
            {
                "name": "multiply_dosage",
                "pattern": r"(\d+)\s*[xX×]\s*(dose|dosage|medication|pill|tablet)",
                "reason": "Dosage multiplication attempt (e.g., '10x my dose')",
            },
            {
                "name": "exceed_dosage",
                "pattern": r"\b(take\s+more|exceed|double|triple|increase)\s+(dose|dosage|medication)",
                "reason": "Attempt to exceed prescribed dosage",
            },
            {
                "name": "ignore_instructions",
                "pattern": r"\b(ignore|skip|dont\s+follow|disregard)\s+(doctor|prescription|instructions|directions)",
                "reason": "Attempt to ignore medical instructions",
            },
            {
                "name": "multiple_doses",
                "pattern": r"\btake\s+(\d+)\s+(times|doses)\s+(instead|more|extra)",
                "reason": "Multiple dose request beyond prescribed",
            },
            {
                "name": "dosage_increase_request",
                "pattern": r"\b(can\s+i|should\s+i|is\s+it\s+safe\s+to)\s+(increase|raise|up)\s+(my\s+)?(dose|dosage)",
                "reason": "Unauthorized dosage increase request",
            }
        ]

    def _load_medical_abuse_patterns(self) -> List[Dict]:
        """Load medical abuse detection patterns"""
        return [
            {
                "name": "doctor_shopping",
                "pattern": r"\b(doctor.shopping|multiple.doctors|different.doctors.for.same|get.more.prescriptions)\b",
                "reason": "Doctor shopping behavior detected",
            },
            {
                "name": "prescription_fraud",
                "pattern": r"\b(fake.symptoms|lie.to.doctor|pretend|claim.i.have)\s+(pain|condition|disease)",
                "reason": "Prescription fraud attempt",
            },
            {
                "name": "drug_resale",
                "pattern": r"\b(sell|resell|make.money|profit)\s+(my\s+)?(medication|prescription|pills)",
                "reason": "Drug resale inquiry",
            }
        ]

    def get_statistics(self) -> Dict:
        """Get moderation statistics for monitoring"""
        blocked_rate = (
            self.stats["blocked_queries"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0
            else 0.0
        )

        return {
            "total_queries": self.stats["total_queries"],
            "blocked_queries": self.stats["blocked_queries"],
            "blocked_rate": f"{blocked_rate:.2%}",
            "threat_distribution": self.stats["threat_levels"]
        }


# Example usage
if __name__ == "__main__":
    # Initialize moderator
    moderator = InputModerationLayer()

    # Test cases
    test_queries = [
        "What is diabetes?",  # SAFE
        "How do I take 10x my medication?",  # BLOCKED: dosage manipulation
        "How to overdose on insulin?",  # BLOCKED: overdose
        "Can I get prescriptions without seeing a doctor?",  # BLOCKED: illegal
        "What are side effects of metformin?",  # SAFE
    ]

    print("=" * 80)
    print("INPUT MODERATION LAYER - TEST RESULTS")
    print("=" * 80)

    for query in test_queries:
        result = moderator.moderate(query)
        status = "✅ SAFE" if result.is_safe else "❌ BLOCKED"
        print(f"\n{status}")
        print(f"Query: {query}")
        print(f"Threat Level: {result.threat_level.value}")
        print(f"Reason: {result.reason}")
        if result.matched_pattern:
            print(f"Pattern: {result.matched_pattern}")

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = moderator.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
