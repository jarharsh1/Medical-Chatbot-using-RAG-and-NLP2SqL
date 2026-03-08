"""
Evaluation Report Generator.

Outputs a terminal-friendly table and JSON report.
"""

import json
import os
from typing import Any, Dict, List


def print_report(metrics: Dict[str, Any], results: List[Dict]):
    """Print a formatted evaluation report to the terminal."""
    print("\n" + "=" * 70)
    print("  MEDICAL AI EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Total Test Cases: {metrics['total_tests']}")
    print(f"  Avg Latency:      {metrics['avg_latency_ms']}ms")

    # Overall Metrics
    print("\n" + "-" * 70)
    print("  OVERALL METRICS")
    print("-" * 70)
    _print_metric("Overall Accuracy", metrics["overall_accuracy"])
    _print_metric("Router Accuracy", metrics["router_accuracy"])
    _print_metric("SQL Valid Rate", metrics["sql_valid_rate"])
    _print_metric("SQL Contains Accuracy", metrics["sql_contains_accuracy"])

    # Safety Metrics
    print("\n" + "-" * 70)
    print("  SAFETY METRICS")
    print("-" * 70)
    _print_metric("Hallucination Rate", metrics["hallucination_rate"], lower_is_better=True)
    _print_metric("Refusal Accuracy", metrics["refusal_accuracy"])
    _print_metric("Injection Safety", metrics["injection_safety"])

    # Calibration
    print("\n" + "-" * 70)
    print("  CALIBRATION")
    print("-" * 70)
    _print_metric("Avg Calibration Error", metrics["avg_calibration_error"], lower_is_better=True)

    if metrics.get("calibration_bins"):
        print("\n  Bin Range    | Count | Avg Conf | Accuracy | Cal Error")
        print("  " + "-" * 58)
        for b in metrics["calibration_bins"]:
            print(
                f"  {b['bin_range']:12s} | {b['count']:5d} | {b['avg_confidence']:8.3f} | "
                f"{b['accuracy']:8.3f} | {b['calibration_error']:8.3f}"
            )

    # Router Confusion Matrix
    if metrics.get("router_confusion_matrix"):
        print("\n" + "-" * 70)
        print("  ROUTER CONFUSION MATRIX (rows=expected, cols=predicted)")
        print("-" * 70)
        matrix = metrics["router_confusion_matrix"]
        labels = sorted(matrix.keys())
        header = "            " + "  ".join(f"{l:>8s}" for l in labels)
        print(f"  {header}")
        for true_label in labels:
            row_vals = "  ".join(f"{matrix[true_label].get(pred, 0):8d}" for pred in labels)
            print(f"  {true_label:>10s}  {row_vals}")

    # Per-Test Results Summary
    print("\n" + "-" * 70)
    print("  PER-TEST RESULTS")
    print("-" * 70)
    print(f"  {'ID':>3s} | {'Type':>7s} | {'Pred':>7s} | {'Correct':>7s} | {'Conf':>5s} | {'ms':>5s} | Question")
    print("  " + "-" * 80)
    for r in results:
        correct_mark = "PASS" if r.get("is_correct") else "FAIL"
        print(
            f"  {r['test_id']:3d} | {r.get('expected_type', '?'):>7s} | "
            f"{r.get('actual_type', '?'):>7s} | {correct_mark:>7s} | "
            f"{r.get('confidence', 0):5.2f} | {r.get('elapsed_ms', 0):5d} | "
            f"{r['question'][:40]}"
        )

    # Detailed test results (all tests: expected vs actual)
    print("\n" + "-" * 70)
    print("  DETAILED TEST RESULTS (Expected vs Actual)")
    print("-" * 70)
    for r in results:
        correct_mark = "PASS" if r.get("is_correct") else "FAIL"
        print(f"\n  [{correct_mark}] Test {r['test_id']}: {r['question']}")
        print(f"    Route:    expected={r.get('expected_type', '?')}, actual={r.get('actual_type', '?')}")

        # Expected output
        expected_parts = []
        if r.get("expected_sql_contains"):
            expected_parts.append(f"SQL keywords: {r['expected_sql_contains']}")
        if r.get("expected_answer_contains"):
            expected_parts.append(f"Answer phrases: {r['expected_answer_contains']}")
        if r.get("expected_refusal"):
            expected_parts.append("Should refuse (out-of-scope)")
        if r.get("expected_no_injection"):
            expected_parts.append("Should block injection")
        if not expected_parts:
            expected_parts.append("Any valid answer (no specific constraints)")
        print(f"    Expected: {'; '.join(expected_parts)}")

        # Actual output
        if r.get("error"):
            print(f"    Actual:   ERROR — {r['error'][:120]}")
        elif r.get("answer"):
            print(f"    Actual:   {r['answer'][:150]}")
        else:
            print(f"    Actual:   (empty)")

        if r.get("sql_generated"):
            print(f"    SQL:      {r['sql_generated'][:120]}")

    # Failed tests summary
    failed = [r for r in results if not r.get("is_correct")]
    if failed:
        print("\n" + "-" * 70)
        print(f"  FAILED TESTS SUMMARY ({len(failed)})")
        print("-" * 70)
        for r in failed:
            print(f"  - Test {r['test_id']}: {r['question'][:60]}")

    print("\n" + "=" * 70)
    print()


def _print_metric(name: str, value: float, lower_is_better: bool = False):
    """Print a metric with color-coded status indicator."""
    pct = value * 100
    if lower_is_better:
        indicator = "GOOD" if pct <= 5 else ("WARN" if pct <= 15 else "FAIL")
    else:
        indicator = "GOOD" if pct >= 85 else ("WARN" if pct >= 60 else "FAIL")

    print(f"  {name:.<40s} {pct:6.1f}%  [{indicator}]")


def save_json_report(metrics: Dict, results: List[Dict], path: str):
    """Save full evaluation report as JSON."""
    report = {
        "metrics": metrics,
        "results": [
            {
                "test_id": r["test_id"],
                "question": r["question"],
                "expected_type": r.get("expected_type"),
                "actual_type": r.get("actual_type"),
                "is_correct": r.get("is_correct"),
                "confidence": r.get("confidence"),
                "elapsed_ms": r.get("elapsed_ms"),
                "error": r.get("error"),
                "tags": r.get("tags", []),
                # Expected output
                "expected_sql_contains": r.get("expected_sql_contains", []),
                "expected_answer_contains": r.get("expected_answer_contains", []),
                "expected_refusal": r.get("expected_refusal", False),
                "expected_no_injection": r.get("expected_no_injection", False),
                # Actual output
                "answer": (r.get("answer") or "")[:300],
                "sql_generated": r.get("sql_generated"),
            }
            for r in results
        ],
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved to: {path}")
