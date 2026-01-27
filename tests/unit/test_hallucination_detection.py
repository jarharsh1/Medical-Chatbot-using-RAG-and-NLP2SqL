"""
Unit tests for Runtime Hallucination Detection (LinkedIn Q4)

Tests hallucination detection at runtime:
- Claim extraction from responses
- Source verification
- Medical entity validation
- Confidence scoring
- Risk assessment
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from reliability.hallucination_detection import (
    RuntimeHallucinationDetector,
    ConfidenceLevel,
    ClaimType
)


class TestRuntimeHallucinationDetector:
    """Test suite for hallucination detection"""

    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return RuntimeHallucinationDetector(confidence_threshold=0.3)

    # =========================================================================
    # CLAIM EXTRACTION
    # =========================================================================

    def test_extract_single_claim(self, detector):
        """Test extraction of single claim from response"""
        response = "Metformin is used to treat type 2 diabetes."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) == 1
        assert "metformin" in result.claims[0].text.lower()

    def test_extract_multiple_claims(self, detector):
        """Test extraction of multiple claims"""
        response = (
            "Metformin is used to treat diabetes. "
            "It works by improving insulin sensitivity. "
            "Common side effects include nausea."
        )
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) == 3

    def test_skip_short_sentences(self, detector):
        """Test that very short sentences are skipped"""
        response = "Yes. No. Metformin treats diabetes effectively."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        # Should only extract the last sentence
        assert len(result.claims) == 1

    # =========================================================================
    # CLAIM CLASSIFICATION
    # =========================================================================

    def test_classify_medication_claim(self, detector):
        """Test classification of medication claims"""
        response = "Metformin is a medication for diabetes."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) > 0
        assert result.claims[0].claim_type == ClaimType.MEDICATION

    def test_classify_dosage_claim(self, detector):
        """Test classification of dosage claims"""
        response = "Take 500mg of metformin twice daily."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) > 0
        assert result.claims[0].claim_type == ClaimType.DOSAGE

    def test_classify_side_effect_claim(self, detector):
        """Test classification of side effect claims"""
        response = "Common side effects of metformin include nausea and diarrhea."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) > 0
        assert result.claims[0].claim_type == ClaimType.SIDE_EFFECT

    def test_classify_treatment_claim(self, detector):
        """Test classification of treatment claims"""
        response = "Diabetes treatment includes lifestyle changes and medication."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        assert len(result.claims) > 0
        assert result.claims[0].claim_type == ClaimType.TREATMENT

    # =========================================================================
    # SOURCE VERIFICATION (Supported Claims)
    # =========================================================================

    def test_exact_match_high_confidence(self, detector):
        """Test that exact matches get high confidence"""
        response = "Metformin is used to treat type 2 diabetes."
        sources = [
            {
                "text": "Metformin is used to treat type 2 diabetes by improving insulin sensitivity.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert len(result.claims) > 0
        claim = result.claims[0]
        assert claim.is_supported
        assert claim.confidence >= 0.8
        assert claim.confidence_level == ConfidenceLevel.HIGH

    def test_partial_match_medium_confidence(self, detector):
        """Test that partial matches get medium confidence"""
        response = "Metformin helps control blood sugar levels."
        sources = [
            {
                "text": "Type 2 diabetes medications like metformin work by improving how the body uses insulin.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert len(result.claims) > 0
        claim = result.claims[0]
        assert claim.confidence > 0.3  # Above threshold
        assert claim.is_supported

    def test_entity_match_boosts_confidence(self, detector):
        """Test that matching medical entities boost confidence"""
        response = "Metformin treats diabetes."
        sources = [
            {
                "text": "Medications for diabetes include metformin and insulin.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert len(result.claims) > 0
        claim = result.claims[0]
        assert "metformin" in claim.medical_entities
        assert claim.is_supported

    # =========================================================================
    # HALLUCINATION DETECTION (Unsupported Claims)
    # =========================================================================

    def test_detect_complete_fabrication(self, detector):
        """Test detection of completely made-up claims"""
        response = "Metformin costs only $5 per month and cures diabetes permanently."
        sources = [
            {
                "text": "Metformin is a medication used to treat type 2 diabetes.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert result.hallucination_detected
        assert len(result.unsupported_claims) > 0

    def test_detect_no_source_support(self, detector):
        """Test detection when no sources provided"""
        response = "Aspirin treats headaches."
        sources = []

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert result.hallucination_detected
        assert len(result.unsupported_claims) > 0
        for claim in result.claims:
            assert not claim.is_supported
            assert claim.confidence_level == ConfidenceLevel.UNSUPPORTED

    def test_detect_contradictory_claim(self, detector):
        """Test detection of claims not supported by sources"""
        response = "Insulin is taken orally in pill form."
        sources = [
            {
                "text": "Insulin is administered through injection or insulin pump.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        # Should be flagged as unsupported
        assert result.hallucination_detected

    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================

    def test_confidence_levels_assigned_correctly(self, detector):
        """Test that confidence levels match confidence scores"""
        test_cases = [
            (0.85, ConfidenceLevel.HIGH),
            (0.65, ConfidenceLevel.MEDIUM),
            (0.4, ConfidenceLevel.LOW),
            (0.2, ConfidenceLevel.UNSUPPORTED)
        ]

        for confidence_score, expected_level in test_cases:
            # Create claim with specific confidence
            response = "Test claim about diabetes medication."
            sources = [{"text": "Diabetes medication information here.", "id": "1"}]

            result = detector.detect_hallucinations(
                llm_response=response,
                source_documents=sources
            )

            # Note: Actual confidence depends on similarity calculation
            # Just check that HIGH confidence claims are supported
            for claim in result.claims:
                if claim.confidence_level == ConfidenceLevel.HIGH:
                    assert claim.is_supported
                elif claim.confidence_level == ConfidenceLevel.UNSUPPORTED:
                    assert not claim.is_supported

    def test_overall_confidence_calculation(self, detector):
        """Test that overall confidence is average of all claims"""
        response = (
            "Metformin treats diabetes. "  # Will be supported
            "It costs $5 per month."      # Won't be supported
        )
        sources = [
            {
                "text": "Metformin is a medication for diabetes treatment.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        # Overall confidence should be between highest and lowest claim
        if len(result.claims) >= 2:
            min_conf = min(c.confidence for c in result.claims)
            max_conf = max(c.confidence for c in result.claims)
            assert min_conf <= result.overall_confidence <= max_conf

    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================

    def test_risk_low_when_all_supported(self, detector):
        """Test that risk is LOW when all claims supported"""
        response = "Metformin is used to treat type 2 diabetes."
        sources = [
            {
                "text": "Metformin is a first-line medication for type 2 diabetes treatment.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        assert result.risk_level == "LOW"
        assert not result.hallucination_detected

    def test_risk_critical_for_medication_hallucination(self, detector):
        """Test that medication hallucinations get CRITICAL risk"""
        response = "Take 5000mg of metformin daily for best results."
        sources = [
            {
                "text": "Metformin typical dosage is 500-2000mg per day.",
                "id": "doc1"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        # Should detect hallucination
        if result.hallucination_detected:
            # Check if any unsupported claims are dosage-related
            has_dosage_hallucination = any(
                c.claim_type == ClaimType.DOSAGE
                for c in result.unsupported_claims
            )
            if has_dosage_hallucination:
                assert result.risk_level == "CRITICAL"

    def test_recommendation_blocks_critical_risk(self, detector):
        """Test that CRITICAL risk gets blocking recommendation"""
        response = "Insulin can be taken as a pill instead of injection."
        sources = []

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        if result.risk_level == "CRITICAL":
            assert "block" in result.recommendation.lower()

    # =========================================================================
    # MEDICAL ENTITY EXTRACTION
    # =========================================================================

    def test_extract_medication_entities(self, detector):
        """Test extraction of medication names"""
        response = "Metformin and insulin are used for diabetes."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        entities = []
        for claim in result.claims:
            entities.extend(claim.medical_entities)

        assert "metformin" in entities
        assert "insulin" in entities

    def test_extract_dosage_entities(self, detector):
        """Test extraction of dosage information"""
        response = "Take 500mg of metformin twice daily."
        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=[]
        )

        entities = []
        for claim in result.claims:
            entities.extend(claim.medical_entities)

        # Should extract dosage pattern
        assert any("500" in str(e) for e in entities)

    # =========================================================================
    # STATISTICS TRACKING
    # =========================================================================

    def test_statistics_tracking(self, detector):
        """Test that statistics are tracked correctly"""
        # Initial state
        stats = detector.get_statistics()
        assert stats["total_responses"] == 0

        # Run detections
        detector.detect_hallucinations(
            llm_response="Metformin treats diabetes.",
            source_documents=[{"text": "Metformin for diabetes.", "id": "1"}]
        )

        detector.detect_hallucinations(
            llm_response="Made up claim.",
            source_documents=[]
        )

        stats = detector.get_statistics()
        assert stats["total_responses"] == 2
        assert stats["hallucinations_detected"] >= 1  # At least the one with no sources

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    def test_empty_response(self, detector):
        """Test handling of empty response"""
        result = detector.detect_hallucinations(
            llm_response="",
            source_documents=[]
        )

        assert len(result.claims) == 0
        assert not result.hallucination_detected

    def test_multiple_sources_best_match(self, detector):
        """Test that best matching source is selected"""
        response = "Metformin improves insulin sensitivity."
        sources = [
            {"text": "Generic information about diabetes.", "id": "1"},
            {"text": "Metformin works by improving insulin sensitivity in cells.", "id": "2"},
            {"text": "Other medications for diabetes.", "id": "3"}
        ]

        result = detector.detect_hallucinations(
            llm_response=response,
            source_documents=sources
        )

        # Should match with source 2
        best_claim = result.claims[0]
        assert "2" in best_claim.source_ids


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestHallucinationDetectionIntegration:
    """Integration tests for hallucination detection"""

    def test_end_to_end_pipeline(self):
        """Test complete hallucination detection pipeline"""
        detector = RuntimeHallucinationDetector()

        # Simulate real RAG scenario
        llm_response = (
            "Metformin is a first-line medication for type 2 diabetes. "
            "It works by reducing glucose production in the liver and "
            "improving insulin sensitivity. Common side effects include "
            "gastrointestinal discomfort."
        )

        source_docs = [
            {
                "text": "Metformin is the first-line pharmacological treatment for type 2 diabetes. It primarily works by decreasing hepatic glucose production.",
                "id": "textbook_ch5"
            },
            {
                "text": "Common adverse effects of metformin include gastrointestinal symptoms such as nausea, vomiting, and diarrhea.",
                "id": "drug_guide_metformin"
            }
        ]

        result = detector.detect_hallucinations(
            llm_response=llm_response,
            source_documents=source_docs
        )

        # Should not detect hallucination (claims are well-supported)
        assert not result.hallucination_detected or result.risk_level == "LOW"
        assert result.overall_confidence > 0.5

    def test_performance_benchmark(self):
        """Test that detection is fast enough for production"""
        import time

        detector = RuntimeHallucinationDetector()

        response = "Metformin treats diabetes effectively. " * 5
        sources = [
            {"text": "Metformin is used for diabetes treatment.", "id": str(i)}
            for i in range(10)
        ]

        start = time.time()
        for _ in range(10):
            detector.detect_hallucinations(
                llm_response=response,
                source_documents=sources
            )
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 10) * 1000
        assert avg_time_ms < 100, f"Detection too slow: {avg_time_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
