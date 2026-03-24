"""
CLI script to run the golden evaluation suite.

Usage (from project root):
    python tests/eval/run_eval.py

Options (set via env vars):
    EVAL_DATASET  — path to eval dataset (default: tests/eval/eval_dataset.json)
    SAVE_BASELINE — if "true", overwrites tests/eval/baseline.json with this run

Workflow:
    # First run — establish a baseline
    SAVE_BASELINE=true python tests/eval/run_eval.py

    # After code changes — compare against baseline
    python tests/eval/run_eval.py

    # Override the dataset path
    EVAL_DATASET=path/to/custom_dataset.json python tests/eval/run_eval.py

Note: SAVE_BASELINE and EVAL_DATASET are environment variables, not CLI
arguments. The prefix syntax (VAR=value python ...) works on Linux/macOS.
On Windows PowerShell use:
    $env:SAVE_BASELINE="true"; python tests/eval/run_eval.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.eval_runner import run_full_eval, save_eval_report
from src.evaluation.regression_detector import detect_regressions, load_baseline, save_baseline


async def main() -> None:
    dataset = os.environ.get("EVAL_DATASET", "tests/eval/eval_dataset.json")
    should_save_baseline = os.environ.get("SAVE_BASELINE", "").lower() == "true"

    print(f"\nRunning evaluation on: {dataset}")
    print("This may take a few minutes...\n")

    report = await run_full_eval(dataset_path=dataset)

    # Persist report
    report_path = save_eval_report(report)

    # Print summary
    agg = report["aggregates"]
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total cases:         {report['total_cases']}")
    print(f"Avg overall score:   {agg['avg_overall_score']:.2f} / 5")
    print(f"Avg accuracy:        {agg['avg_accuracy_score']:.2f} / 5")
    print(f"Keyword coverage:    {agg['avg_keyword_coverage']:.1%}")
    print(f"Source coverage:     {agg['avg_source_coverage']:.1%}")
    print(f"Failures (score<3):  {agg['failure_count']}")
    print(f"\nReport saved to:     {report_path}")

    if report["failures"]:
        print("\n" + "=" * 60)
        print("FAILING CASES")
        print("=" * 60)
        for f in report["failures"]:
            print(f"\n  [{f['case_id']}] {f['question']}")
            print(f"  Score: {f['evaluation']['overall']}/5")
            issues = f['evaluation'].get("issues", [])
            if issues:
                print(f"  Issues: {', '.join(issues)}")

    # Regression check (if baseline exists)
    baseline_path = "tests/eval/baseline.json"
    if Path(baseline_path).exists() and not should_save_baseline:
        print("\n" + "=" * 60)
        print("REGRESSION CHECK")
        print("=" * 60)
        baseline = load_baseline(baseline_path)
        regressions = detect_regressions(report, baseline)
        if regressions:
            print(f"⚠  {len(regressions)} regression(s) detected:")
            for r in regressions:
                if r["type"] == "overall_score":
                    print(f"  CRITICAL — overall score dropped {r['drop_pct']}%"
                          f" ({r['baseline']} → {r['current']})")
                else:
                    print(f"  WARNING  — [{r['case_id']}] score dropped {r['drop_pct']}%"
                          f" ({r['baseline_score']} → {r['current_score']})")
        else:
            print("✓ No regressions detected.")

    if should_save_baseline:
        save_baseline(report, baseline_path)
        print(f"\n✓ Baseline saved to {baseline_path}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
