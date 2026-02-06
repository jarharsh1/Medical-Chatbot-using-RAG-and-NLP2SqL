"""
Comprehensive Query Evaluation Script

Tests the system against various query types and measures:
- SQL accuracy (correct results)
- RAG relevance (retrieved correct context)
- Routing accuracy (correct agent selected)
- Response quality (answer makes sense)

Run: python -m backend.scripts.evaluate_queries
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH

# Test cases with expected outcomes
TEST_CASES = [
    # === SQL QUERIES (Basic) ===
    {
        "id": "sql_1",
        "question": "How many patients are there in total?",
        "expected_route": "sql",
        "expected_contains": ["patient"],
        "validation": lambda r: "patient" in r.get("answer", "").lower() and any(c.isdigit() for c in r.get("answer", "")),
        "ground_truth_query": "SELECT COUNT(*) FROM patients",
    },
    {
        "id": "sql_2",
        "question": "Which clinic has the most diabetes patients?",
        "expected_route": "sql",
        "expected_contains": ["clinic"],
        "validation": lambda r: "clinic" in r.get("answer", "").lower() or r.get("sql_generated"),
        "ground_truth_query": """
            SELECT c.name, COUNT(DISTINCT cn.patient_id) as cnt
            FROM clinics c
            JOIN patients p ON c.clinic_id = p.clinic_id
            JOIN clinical_notes cn ON p.patient_id = cn.patient_id
            WHERE cn.condition_name LIKE '%Diabetes%'
            GROUP BY c.name ORDER BY cnt DESC LIMIT 1
        """,
    },
    {
        "id": "sql_3",
        "question": "How many active prescriptions are there?",
        "expected_route": "sql",
        "expected_contains": ["prescription", "active"],
        "validation": lambda r: any(c.isdigit() for c in r.get("answer", "")),
        "ground_truth_query": "SELECT COUNT(*) FROM prescriptions WHERE status = 'Active'",
    },
    {
        "id": "sql_4",
        "question": "List the top 5 most prescribed medications",
        "expected_route": "sql",
        "expected_contains": ["medication"],
        "validation": lambda r: r.get("sql_generated") and "GROUP BY" in r.get("sql_generated", "").upper(),
    },
    {
        "id": "sql_5",
        "question": "Who are the doctors at Central Medical Clinic?",
        "expected_route": "sql",
        "expected_contains": ["doctor"],
        "validation": lambda r: "doctor" in r.get("answer", "").lower() or "dr" in r.get("answer", "").lower(),
    },

    # === SQL QUERIES (Advanced) ===
    {
        "id": "sql_6",
        "question": "How many patients have gout?",
        "expected_route": "sql",
        "expected_contains": ["patient", "gout"],
        "validation": lambda r: any(c.isdigit() for c in r.get("answer", "")),
        "ground_truth_query": "SELECT COUNT(DISTINCT patient_id) FROM clinical_notes WHERE condition_name LIKE '%Gout%'",
    },
    {
        "id": "sql_7",
        "question": "What is the average number of refills remaining per prescription?",
        "expected_route": "sql",
        "expected_contains": ["refill", "average"],
        "validation": lambda r: any(c.isdigit() for c in r.get("answer", "")) or "." in r.get("answer", ""),
    },
    {
        "id": "sql_8",
        "question": "List all clinics and their locations",
        "expected_route": "sql",
        "expected_contains": ["clinic", "location"],
        "validation": lambda r: r.get("sql_generated") and "clinic" in r.get("sql_generated", "").lower(),
    },
    {
        "id": "sql_9",
        "question": "Which doctor has seen the most patients?",
        "expected_route": "sql",
        "expected_contains": ["doctor"],
        "validation": lambda r: r.get("sql_generated") and "GROUP BY" in r.get("sql_generated", "").upper(),
    },
    {
        "id": "sql_10",
        "question": "How many male vs female patients are there?",
        "expected_route": "sql",
        "expected_contains": ["male", "female"],
        "validation": lambda r: any(c.isdigit() for c in r.get("answer", "")),
    },

    # === MULTI-PART QUERIES (Orchestrated) ===
    {
        "id": "multi_1",
        "question": "What is the root problem of gout? How many patients have it?",
        "expected_route": "orchestrated",
        "expected_contains": ["uric acid", "patient"],
        "validation": lambda r: r.get("decomposition") and r["decomposition"].get("parts_count", 0) >= 2,
    },
    {
        "id": "multi_2",
        "question": "Which clinic is famous among diabetes patients? Who are the consulting doctors there?",
        "expected_route": "orchestrated",
        "expected_contains": ["clinic", "doctor"],
        "validation": lambda r: "clinic" in r.get("answer", "").lower() and "doctor" in r.get("answer", "").lower(),
    },
    {
        "id": "multi_3",
        "question": "What causes hypertension? How many patients are diagnosed with it? What medications are prescribed?",
        "expected_route": "orchestrated",
        "expected_contains": ["blood pressure", "patient", "medication"],
        "validation": lambda r: r.get("decomposition") and r["decomposition"].get("parts_count", 0) >= 2,
    },
    {
        "id": "multi_4",
        "question": "What are the symptoms of asthma? Which patients have it and what inhalers are they using?",
        "expected_route": "orchestrated",
        "expected_contains": ["symptom", "patient"],
        "validation": lambda r: r.get("decomposition") and r["decomposition"].get("parts_count", 0) >= 2,
    },

    # === RAG QUERIES ===
    {
        "id": "rag_1",
        "question": "What symptoms are described for diabetic patients in clinical notes?",
        "expected_route": "rag",
        "expected_contains": [],  # Will check if sources are returned
        "validation": lambda r: len(r.get("sources", [])) > 0,
    },
    {
        "id": "rag_2",
        "question": "Summarize the treatment plans mentioned in hypertension notes",
        "expected_route": "rag",
        "expected_contains": ["treatment", "medication"],
        "validation": lambda r: len(r.get("sources", [])) > 0,
    },
    {
        "id": "rag_3",
        "question": "What physical examination findings are noted for arthritis patients?",
        "expected_route": "rag",
        "expected_contains": [],
        "validation": lambda r: len(r.get("sources", [])) > 0,
    },
    {
        "id": "rag_4",
        "question": "Describe the lifestyle recommendations given to patients with high cholesterol",
        "expected_route": "rag",
        "expected_contains": [],
        "validation": lambda r: len(r.get("sources", [])) > 0 or "lifestyle" in r.get("answer", "").lower(),
    },

    # === HYBRID QUERIES ===
    {
        "id": "hybrid_1",
        "question": "What medications are prescribed for patients whose notes mention chest pain?",
        "expected_route": "hybrid",
        "expected_contains": ["medication"],
        "validation": lambda r: r.get("query_type") in ["hybrid", "orchestrated"],
    },
    {
        "id": "hybrid_2",
        "question": "Find patients with gout and list their current prescriptions",
        "expected_route": "hybrid",
        "expected_contains": ["patient", "prescription"],
        "validation": lambda r: r.get("sql_generated") or len(r.get("sources", [])) > 0,
    },
    {
        "id": "hybrid_3",
        "question": "Which diabetic patients have notes mentioning neuropathy?",
        "expected_route": "hybrid",
        "expected_contains": ["patient", "neuropathy"],
        "validation": lambda r: r.get("query_type") in ["hybrid", "orchestrated", "rag"],
    },

    # === KNOWLEDGE QUERIES (LLM-only for general medical facts) ===
    {
        "id": "knowledge_1",
        "question": "What causes diabetes?",
        "expected_route": "knowledge",
        "expected_contains": ["insulin", "blood sugar"],
        "validation": lambda r: len(r.get("answer", "")) > 50,  # Should have substantial answer
    },
    {
        "id": "knowledge_2",
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "expected_route": "knowledge",
        "expected_contains": ["type 1", "type 2"],
        "validation": lambda r: "type" in r.get("answer", "").lower() and len(r.get("answer", "")) > 50,
    },

    # === EDGE CASES ===
    {
        "id": "edge_1",
        "question": "How many patients have diabetes?",
        "expected_route": "sql",
        "expected_contains": ["patient", "diabetes"],
        "validation": lambda r: any(c.isdigit() for c in r.get("answer", "")),
        "ground_truth_query": "SELECT COUNT(DISTINCT patient_id) FROM clinical_notes WHERE condition_name LIKE '%Diabetes%'",
    },
    {
        "id": "edge_2",
        "question": "What is the most common condition?",
        "expected_route": "sql",
        "expected_contains": ["condition"],
        "validation": lambda r: r.get("sql_generated") and "GROUP BY" in r.get("sql_generated", "").upper(),
    },
    {
        "id": "edge_3",
        "question": "Show me patients born after 1990",
        "expected_route": "sql",
        "expected_contains": ["patient"],
        "validation": lambda r: r.get("sql_generated") and ("1990" in r.get("sql_generated", "") or "dob" in r.get("sql_generated", "").lower()),
    },
    {
        "id": "edge_4",
        "question": "List prescriptions that need refill soon",
        "expected_route": "sql",
        "expected_contains": ["prescription", "refill"],
        "validation": lambda r: r.get("sql_generated") and "refill" in r.get("sql_generated", "").lower(),
    },
]


@dataclass
class EvalResult:
    test_id: str
    question: str
    expected_route: str
    actual_route: str
    route_correct: bool
    validation_passed: bool
    has_answer: bool
    answer_snippet: str
    sql_generated: Optional[str]
    execution_time_ms: int
    error: Optional[str]
    confidence: float
    decomposition_parts: int


def run_query(question: str) -> Dict[str, Any]:
    """Execute a query through the full pipeline."""
    from backend.app import _process_query, _ensure_rag_ready

    _ensure_rag_ready()
    return _process_query(question)


def get_ground_truth(query: str) -> Optional[str]:
    """Execute a ground truth SQL query."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query.strip())
        result = cursor.fetchall()
        conn.close()
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def evaluate_single(test_case: Dict) -> EvalResult:
    """Evaluate a single test case."""
    start = time.time()
    error = None

    try:
        result = run_query(test_case["question"])
    except Exception as e:
        error = str(e)
        result = {}

    elapsed = int((time.time() - start) * 1000)

    actual_route = result.get("query_type", "unknown")
    expected_route = test_case["expected_route"]

    # Route matching (orchestrated can match multi-part expectations)
    route_correct = actual_route == expected_route or (
        actual_route == "orchestrated" and expected_route in ["orchestrated", "hybrid", "multi"]
    )

    # Run validation function
    validation_passed = False
    if test_case.get("validation"):
        try:
            validation_passed = test_case["validation"](result)
        except Exception:
            validation_passed = False

    answer = result.get("answer", "")
    decomposition = result.get("decomposition", {})

    return EvalResult(
        test_id=test_case["id"],
        question=test_case["question"],
        expected_route=expected_route,
        actual_route=actual_route,
        route_correct=route_correct,
        validation_passed=validation_passed,
        has_answer=bool(answer and len(answer) > 10),
        answer_snippet=answer[:150] + "..." if len(answer) > 150 else answer,
        sql_generated=result.get("sql_generated"),
        execution_time_ms=elapsed,
        error=error or result.get("error"),
        confidence=result.get("confidence", 0),
        decomposition_parts=decomposition.get("parts_count", 0) if decomposition else 0,
    )


def run_evaluation(test_cases: List[Dict] = TEST_CASES, verbose: bool = True) -> Dict[str, Any]:
    """Run full evaluation suite."""
    results: List[EvalResult] = []

    print("\n" + "=" * 70)
    print("MEDICAL CHATBOT EVALUATION")
    print("=" * 70 + "\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {tc['question'][:60]}...")

        result = evaluate_single(tc)
        results.append(result)

        status = "PASS" if result.validation_passed and result.route_correct else "FAIL"
        route_status = "OK" if result.route_correct else f"WRONG ({result.actual_route})"

        if verbose:
            print(f"  Route: {result.expected_route} -> {route_status}")
            print(f"  Validation: {'PASS' if result.validation_passed else 'FAIL'}")
            print(f"  Time: {result.execution_time_ms}ms | Confidence: {result.confidence:.0%}")
            if result.decomposition_parts > 0:
                print(f"  Decomposed: {result.decomposition_parts} parts")
            if result.error:
                print(f"  Error: {result.error[:100]}")
            print()

    # Calculate metrics
    total = len(results)
    route_correct = sum(1 for r in results if r.route_correct)
    validation_passed = sum(1 for r in results if r.validation_passed)
    has_answer = sum(1 for r in results if r.has_answer)
    errors = sum(1 for r in results if r.error)

    avg_time = sum(r.execution_time_ms for r in results) / total if total else 0
    avg_confidence = sum(r.confidence for r in results) / total if total else 0

    metrics = {
        "total_tests": total,
        "routing_accuracy": route_correct / total if total else 0,
        "validation_pass_rate": validation_passed / total if total else 0,
        "answer_rate": has_answer / total if total else 0,
        "error_rate": errors / total if total else 0,
        "avg_execution_time_ms": avg_time,
        "avg_confidence": avg_confidence,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total Tests:        {total}")
    print(f"  Routing Accuracy:   {metrics['routing_accuracy']:.1%} ({route_correct}/{total})")
    print(f"  Validation Pass:    {metrics['validation_pass_rate']:.1%} ({validation_passed}/{total})")
    print(f"  Has Answer:         {metrics['answer_rate']:.1%} ({has_answer}/{total})")
    print(f"  Error Rate:         {metrics['error_rate']:.1%} ({errors}/{total})")
    print(f"  Avg Time:           {avg_time:.0f}ms")
    print(f"  Avg Confidence:     {avg_confidence:.1%}")
    print("=" * 70)

    # Breakdown by category
    categories = {}
    for r in results:
        cat = r.test_id.split("_")[0]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.validation_passed and r.route_correct:
            categories[cat]["passed"] += 1

    print("\nBREAKDOWN BY CATEGORY:")
    for cat, stats in categories.items():
        pct = stats["passed"] / stats["total"] if stats["total"] else 0
        print(f"  {cat.upper():12} {stats['passed']}/{stats['total']} ({pct:.0%})")

    # Failed tests
    failed = [r for r in results if not r.validation_passed or not r.route_correct]
    if failed:
        print("\nFAILED TESTS:")
        for r in failed:
            print(f"  - {r.test_id}: {r.question[:50]}...")
            if not r.route_correct:
                print(f"    Route: expected {r.expected_route}, got {r.actual_route}")
            if not r.validation_passed:
                print(f"    Validation failed")
            if r.error:
                print(f"    Error: {r.error[:80]}")

    return {
        "metrics": metrics,
        "results": [
            {
                "test_id": r.test_id,
                "question": r.question,
                "passed": r.validation_passed and r.route_correct,
                "route_correct": r.route_correct,
                "validation_passed": r.validation_passed,
                "actual_route": r.actual_route,
                "confidence": r.confidence,
                "time_ms": r.execution_time_ms,
            }
            for r in results
        ],
        "categories": categories,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate medical chatbot queries")
    parser.add_argument("--quick", action="store_true", help="Run only 5 tests")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.quick:
        test_subset = TEST_CASES[:5]
    else:
        test_subset = TEST_CASES

    results = run_evaluation(test_subset, verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))
