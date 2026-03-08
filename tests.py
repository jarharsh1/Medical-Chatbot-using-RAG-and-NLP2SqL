"""
Run all evaluation tests locally.

Usage:
    python tests.py

Results are saved to:
    evaluation/eval_report.json
    evaluation/baseline_metrics.json
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    print("Running evaluation tests...")
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate"],
        cwd=project_root,
        env=env,
    )
    sys.exit(result.returncode)
