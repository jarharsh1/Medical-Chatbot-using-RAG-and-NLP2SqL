"""
Runtime Hallucination Detection - LinkedIn Post Q4 Solution

Q: "How do you detect hallucinations at runtime?"
A: Multi-layered verification with source attribution enforcement

Detection Strategies:
1. Claim Extraction (parse LLM response into verifiable claims)
2. Source Attribution (verify each claim has supporting evidence)
3. Medical Entity Validation (check drugs, dosages against known databases)
4. Confidence Scoring (quantify reliability of each claim)
5. Contradiction Detection (flag conflicting information)

Critical for healthcare: Never let LLM make up medical facts
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for claim verification"""
    HIGH = "high"           # >0.8 - Strong source support
    MEDIUM = "medium"       # 0.5-0.8 - Partial support
    LOW = "low"            # 0.3-0.5 - Weak support
    UNSUPPORTED = "unsupported"  # <0.3 - No source support (hallucination)


class ClaimType(Enum):
    """Types of medical claims"""
    MEDICATION = "medication"          # Drug names, uses
    DOSAGE = "dosage"                 # Dosage information
    SIDE_EFFECT = "side_effect"       # Side effects
    CONDITION = "condition"           # Medical conditions
    TREATMENT = "treatment"           # Treatment recommendations
    STATISTIC = "statistic"           # Numerical facts
    GENERAL = "general"               # General medical information


@dataclass
class Claim:
    """Represents a verifiable claim from LLM response"""
    text: str                          # The claim text
    claim_type: ClaimType              # Type of claim
    source_ids: List[str]              # IDs of supporting sources
    confidence: float                  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    is_supported: bool                 # Has adequate source support
    explanation: str                   # Why claim is/isn't supported
    medical_entities: List[str]        # Extracted entities (drugs, conditions)


@dataclass
class HallucinationDetectionResult:
    """Result of hallucination detection"""
    claims: List[Claim]
    hallucination_detected: bool       # Any unsupported claims?
    overall_confidence: float          # Average confidence across all claims
    unsupported_claims: List[Claim]    # Claims flagged as hallucinations
    risk_level: str                    # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str                # Action to take


class RuntimeHallucinationDetector:
    """
    Detects hallucinations in LLM responses at runtime.

    Implements multi-layered verification:
    1. Extract claims from response
    2. Match claims to source documents
    3. Validate medical entities
    4. Assign confidence scores
    5. Flag unsupported claims

    Example:
        detector = RuntimeHallucinationDetector()
        result = detector.detect_hallucinations(
            llm_response="Metformin treats diabetes and costs $5",
            source_documents=[{"text": "Metformin is used for diabetes...", "id": "doc1"}]
        )
        if result.hallucination_detected:
            log_hallucination(result)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        enable_strict_mode: bool = True,
        known_medications: Optional[Set[str]] = None
    ):
        """
        Initialize hallucination detector.

        Args:
            confidence_threshold: Minimum confidence to consider claim supported
            enable_strict_mode: If True, flag any unsupported claim as hallucination
            known_medications: Set of known medication names for validation
        """
        self.confidence_threshold = confidence_threshold
        self.enable_strict_mode = enable_strict_mode
        self.known_medications = known_medications or self._load_common_medications()

        # Statistics
        self.stats = {
            "total_responses": 0,
            "hallucinations_detected": 0,
            "total_claims": 0,
            "unsupported_claims": 0,
            "claim_types": {ct.value: 0 for ct in ClaimType}
        }

    def detect_hallucinations(
        self,
        llm_response: str,
        source_documents: List[Dict],
        query: Optional[str] = None
    ) -> HallucinationDetectionResult:
        """
        Detect hallucinations in LLM response by verifying against sources.

        Args:
            llm_response: The LLM's generated response
            source_documents: Retrieved source documents (with "text" and "id" keys)
            query: Optional original user query for context

        Returns:
            HallucinationDetectionResult with detection info and flagged claims

        Example:
            >>> detector = RuntimeHallucinationDetector()
            >>> result = detector.detect_hallucinations(
            ...     llm_response="Metformin treats diabetes",
            ...     source_documents=[{"text": "Metformin is used for type 2 diabetes", "id": "1"}]
            ... )
            >>> result.hallucination_detected
            False
        """
        self.stats["total_responses"] += 1

        # Step 1: Extract claims from response
        claims = self._extract_claims(llm_response)
        self.stats["total_claims"] += len(claims)

        # Step 2: Verify each claim against sources
        verified_claims = []
        for claim in claims:
            verified_claim = self._verify_claim(claim, source_documents)
            verified_claims.append(verified_claim)

            # Update statistics
            self.stats["claim_types"][verified_claim.claim_type.value] += 1
            if not verified_claim.is_supported:
                self.stats["unsupported_claims"] += 1

        # Step 3: Identify unsupported claims (hallucinations)
        unsupported_claims = [c for c in verified_claims if not c.is_supported]

        # Step 4: Calculate overall confidence
        overall_confidence = (
            sum(c.confidence for c in verified_claims) / len(verified_claims)
            if verified_claims else 1.0
        )

        # Step 5: Determine risk level
        hallucination_detected = len(unsupported_claims) > 0
        risk_level = self._assess_risk_level(unsupported_claims, overall_confidence)
        recommendation = self._generate_recommendation(risk_level, unsupported_claims)

        if hallucination_detected:
            self.stats["hallucinations_detected"] += 1
            self._log_hallucination(llm_response, unsupported_claims)

        return HallucinationDetectionResult(
            claims=verified_claims,
            hallucination_detected=hallucination_detected,
            overall_confidence=overall_confidence,
            unsupported_claims=unsupported_claims,
            risk_level=risk_level,
            recommendation=recommendation
        )

    def _extract_claims(self, response: str) -> List[Claim]:
        """
        Extract verifiable claims from LLM response.

        Splits response into sentences and identifies claim type.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        claims = []
        for sentence in sentences:
            # Skip very short sentences (likely not claims)
            if len(sentence) < 10:
                continue

            # Classify claim type
            claim_type = self._classify_claim_type(sentence)

            # Extract medical entities
            entities = self._extract_medical_entities(sentence)

            # Create unverified claim
            claim = Claim(
                text=sentence,
                claim_type=claim_type,
                source_ids=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNSUPPORTED,
                is_supported=False,
                explanation="Not yet verified",
                medical_entities=entities
            )
            claims.append(claim)

        return claims

    def _classify_claim_type(self, text: str) -> ClaimType:
        """Classify the type of claim based on content patterns"""
        text_lower = text.lower()

        # Medication patterns
        if any(med in text_lower for med in self.known_medications):
            if any(word in text_lower for word in ['mg', 'dose', 'take', 'tablet', 'pill']):
                return ClaimType.DOSAGE
            return ClaimType.MEDICATION

        # Side effect patterns
        if any(word in text_lower for word in ['side effect', 'adverse', 'reaction', 'cause']):
            return ClaimType.SIDE_EFFECT

        # Treatment patterns
        if any(word in text_lower for word in ['treatment', 'therapy', 'manage', 'cure']):
            return ClaimType.TREATMENT

        # Condition patterns
        if any(word in text_lower for word in ['disease', 'condition', 'disorder', 'syndrome']):
            return ClaimType.CONDITION

        # Statistic patterns (numbers, percentages)
        if re.search(r'\d+(\.\d+)?%|\d+\s*(patients|people|cases)', text_lower):
            return ClaimType.STATISTIC

        return ClaimType.GENERAL

    def _extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities (drugs, conditions) from text"""
        entities = []
        text_lower = text.lower()

        # Extract known medications
        for med in self.known_medications:
            if med in text_lower:
                entities.append(med)

        # Extract dosage patterns
        dosage_patterns = re.findall(r'\d+\s*mg|\d+\s*g|\d+\s*ml', text_lower)
        entities.extend(dosage_patterns)

        return entities

    def _verify_claim(self, claim: Claim, source_documents: List[Dict]) -> Claim:
        """
        Verify claim against source documents.

        Strategy:
        1. Find best matching source for claim
        2. Calculate similarity score
        3. Check if medical entities are mentioned in sources
        4. Assign confidence score
        """
        if not source_documents:
            # No sources - cannot verify
            claim.confidence = 0.0
            claim.confidence_level = ConfidenceLevel.UNSUPPORTED
            claim.is_supported = False
            claim.explanation = "No source documents provided"
            return claim

        # Find best matching source
        best_match_score = 0.0
        best_source_id = None

        for doc in source_documents:
            doc_text = doc.get("text", "")
            similarity = self._calculate_similarity(claim.text, doc_text)

            # Boost score if medical entities match
            entity_match_bonus = self._check_entity_match(claim.medical_entities, doc_text)
            total_score = similarity + entity_match_bonus

            if total_score > best_match_score:
                best_match_score = total_score
                best_source_id = doc.get("id", "unknown")

        # Assign confidence based on match score
        claim.confidence = min(best_match_score, 1.0)
        claim.source_ids = [best_source_id] if best_source_id else []

        # Determine confidence level
        if claim.confidence >= 0.8:
            claim.confidence_level = ConfidenceLevel.HIGH
            claim.is_supported = True
            claim.explanation = f"Strong source support (score: {claim.confidence:.2f})"
        elif claim.confidence >= 0.5:
            claim.confidence_level = ConfidenceLevel.MEDIUM
            claim.is_supported = True
            claim.explanation = f"Moderate source support (score: {claim.confidence:.2f})"
        elif claim.confidence >= self.confidence_threshold:
            claim.confidence_level = ConfidenceLevel.LOW
            claim.is_supported = True
            claim.explanation = f"Weak source support (score: {claim.confidence:.2f})"
        else:
            claim.confidence_level = ConfidenceLevel.UNSUPPORTED
            claim.is_supported = False
            claim.explanation = f"No adequate source support (score: {claim.confidence:.2f})"

        return claim

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between claim and source text.

        Uses sequence matching for semantic similarity.
        """
        # Normalize texts
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check for exact substring match (high confidence)
        if text1_lower in text2_lower:
            return 0.95

        # Calculate sequence similarity
        similarity = SequenceMatcher(None, text1_lower, text2_lower).ratio()

        # Boost if key terms overlap
        text1_words = set(text1_lower.split())
        text2_words = set(text2_lower.split())
        word_overlap = len(text1_words & text2_words) / len(text1_words) if text1_words else 0

        # Weighted combination
        final_score = 0.6 * similarity + 0.4 * word_overlap

        return final_score

    def _check_entity_match(self, entities: List[str], source_text: str) -> float:
        """
        Check if medical entities appear in source text.
        Returns bonus score (0.0 to 0.2)
        """
        if not entities:
            return 0.0

        source_lower = source_text.lower()
        matched_entities = sum(1 for entity in entities if entity in source_lower)
        match_rate = matched_entities / len(entities)

        return match_rate * 0.2  # Up to 20% bonus

    def _assess_risk_level(
        self,
        unsupported_claims: List[Claim],
        overall_confidence: float
    ) -> str:
        """
        Assess risk level based on unsupported claims.

        Returns: LOW, MEDIUM, HIGH, or CRITICAL
        """
        num_unsupported = len(unsupported_claims)

        # Check for critical claim types
        has_critical_type = any(
            c.claim_type in [ClaimType.MEDICATION, ClaimType.DOSAGE, ClaimType.TREATMENT]
            for c in unsupported_claims
        )

        if num_unsupported == 0:
            return "LOW"
        elif num_unsupported == 1 and not has_critical_type:
            return "MEDIUM"
        elif num_unsupported <= 2 and not has_critical_type:
            return "HIGH"
        else:
            return "CRITICAL"

    def _generate_recommendation(
        self,
        risk_level: str,
        unsupported_claims: List[Claim]
    ) -> str:
        """Generate action recommendation based on risk level"""
        recommendations = {
            "LOW": "Safe to present response to user",
            "MEDIUM": "Present response with disclaimer about limited source verification",
            "HIGH": "Flag response for review before presenting to user",
            "CRITICAL": "Block response - contains unsupported medical claims that could be harmful"
        }

        base_recommendation = recommendations[risk_level]

        if unsupported_claims:
            claim_texts = [f"'{c.text}'" for c in unsupported_claims[:2]]
            base_recommendation += f". Unsupported claims: {', '.join(claim_texts)}"

        return base_recommendation

    def _load_common_medications(self) -> Set[str]:
        """Load common medication names for entity recognition"""
        return {
            "metformin", "insulin", "aspirin", "ibuprofen", "acetaminophen",
            "lisinopril", "amlodipine", "atorvastatin", "simvastatin",
            "omeprazole", "levothyroxine", "albuterol", "gabapentin",
            "losartan", "hydrochlorothiazide", "prednisone", "amoxicillin",
            "azithromycin", "ciprofloxacin", "warfarin", "clopidogrel",
            "sertraline", "escitalopram", "fluoxetine", "alprazolam",
            "lorazepam", "zolpidem", "tramadol", "oxycodone", "morphine",
            "furosemide", "pantoprazole", "ranitidine", "montelukast",
            "pravastatin", "rosuvastatin", "metoprolol", "carvedilol",
            "glipizide", "glyburide", "pioglitazone", "sitagliptin"
        }

    def _log_hallucination(self, response: str, unsupported_claims: List[Claim]):
        """Log detected hallucination for monitoring"""
        claim_texts = [c.text for c in unsupported_claims]
        logger.warning(
            f"HALLUCINATION DETECTED | "
            f"Response: {response[:100]}... | "
            f"Unsupported Claims: {claim_texts}"
        )

    def get_statistics(self) -> Dict:
        """Get hallucination detection statistics"""
        detection_rate = (
            self.stats["hallucinations_detected"] / self.stats["total_responses"]
            if self.stats["total_responses"] > 0
            else 0.0
        )

        unsupported_rate = (
            self.stats["unsupported_claims"] / self.stats["total_claims"]
            if self.stats["total_claims"] > 0
            else 0.0
        )

        return {
            "total_responses": self.stats["total_responses"],
            "hallucinations_detected": self.stats["hallucinations_detected"],
            "detection_rate": f"{detection_rate:.2%}",
            "total_claims": self.stats["total_claims"],
            "unsupported_claims": self.stats["unsupported_claims"],
            "unsupported_rate": f"{unsupported_rate:.2%}",
            "claim_type_distribution": self.stats["claim_types"]
        }


# Example usage
if __name__ == "__main__":
    detector = RuntimeHallucinationDetector(confidence_threshold=0.3)

    # Test case 1: Supported claim
    print("=" * 80)
    print("TEST CASE 1: Supported Claim")
    print("=" * 80)

    result1 = detector.detect_hallucinations(
        llm_response="Metformin is used to treat type 2 diabetes by improving insulin sensitivity.",
        source_documents=[
            {
                "text": "Metformin is a medication used to treat type 2 diabetes. It works by improving insulin sensitivity and reducing glucose production in the liver.",
                "id": "doc1"
            }
        ]
    )

    print(f"Hallucination Detected: {result1.hallucination_detected}")
    print(f"Overall Confidence: {result1.overall_confidence:.2f}")
    print(f"Risk Level: {result1.risk_level}")
    print(f"Recommendation: {result1.recommendation}")
    print(f"\nClaims:")
    for claim in result1.claims:
        print(f"  - {claim.text}")
        print(f"    Confidence: {claim.confidence:.2f} ({claim.confidence_level.value})")
        print(f"    Supported: {claim.is_supported}")

    # Test case 2: Hallucination (made-up fact)
    print("\n" + "=" * 80)
    print("TEST CASE 2: Hallucination (Made-up Fact)")
    print("=" * 80)

    result2 = detector.detect_hallucinations(
        llm_response="Metformin costs only $5 per month and cures diabetes completely.",
        source_documents=[
            {
                "text": "Metformin is a medication used to treat type 2 diabetes.",
                "id": "doc1"
            }
        ]
    )

    print(f"Hallucination Detected: {result2.hallucination_detected}")
    print(f"Overall Confidence: {result2.overall_confidence:.2f}")
    print(f"Risk Level: {result2.risk_level}")
    print(f"Recommendation: {result2.recommendation}")
    print(f"\nUnsupported Claims:")
    for claim in result2.unsupported_claims:
        print(f"  - {claim.text}")
        print(f"    Confidence: {claim.confidence:.2f}")
        print(f"    Explanation: {claim.explanation}")

    # Test case 3: No sources (all hallucinations)
    print("\n" + "=" * 80)
    print("TEST CASE 3: No Sources")
    print("=" * 80)

    result3 = detector.detect_hallucinations(
        llm_response="Aspirin treats headaches and reduces fever.",
        source_documents=[]
    )

    print(f"Hallucination Detected: {result3.hallucination_detected}")
    print(f"Overall Confidence: {result3.overall_confidence:.2f}")
    print(f"Risk Level: {result3.risk_level}")
    print(f"Recommendation: {result3.recommendation}")

    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
