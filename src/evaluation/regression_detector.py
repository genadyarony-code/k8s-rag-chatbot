"""
Regression detection by comparing evaluation runs against a stored baseline.

Usage pattern:
1. Run `python tests/eval/run_eval.py` to produce an eval report.
2. Copy the best report to `tests/eval/baseline.json`.
3. After future code changes, run eval again and call `detect_regressions()`
   to identify cases where quality has dropped.

Threshold semantics (default 0.3 = 30 %):
    A regression is flagged when a score drops by more than 30 % relative
    to the baseline. E.g. baseline overall = 4.0 → regression if < 2.8.

Two levels of regression:
    critical  — overall aggregate score drops below threshold
    warning   — individual test case score drops below threshold
"""

import json
from pathlib import Path

from src.observability.logging_config import get_logger

log = get_logger(__name__)

_DEFAULT_BASELINE = "tests/eval/baseline.json"


def load_baseline(path: str = _DEFAULT_BASELINE) -> dict:
    """
    Load a baseline evaluation report from disk.

    Raises:
        FileNotFoundError: if the baseline file does not exist yet. Callers
            that expect the file may not exist yet should check
            ``Path(path).exists()`` before calling, or catch this exception.
    """
    baseline_path = Path(path)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found at '{path}'. "
            "Run the evaluator once with SAVE_BASELINE=true to create it:\n"
            "  SAVE_BASELINE=true python tests/eval/run_eval.py"
        )
    with open(baseline_path) as f:
        return json.load(f)


def detect_regressions(
    current_report: dict,
    baseline_report: dict,
    threshold: float = 0.3,
) -> list[dict]:
    """
    Compare `current_report` against `baseline_report` and return regressions.

    Args:
        current_report:  Freshly generated eval report (from run_full_eval).
        baseline_report: Saved baseline report (the "good" run to compare to).
        threshold:       Relative drop threshold; 0.3 = flag if score drops > 30 %.

    Returns:
        List of regression dicts. Empty list means no regressions detected.
        Each dict has: type, severity, and score fields relevant to that type.
    """
    regressions: list[dict] = []

    # ── Aggregate score regression ────────────────────────────────────────────
    cur_agg = current_report["aggregates"]["avg_overall_score"]
    base_agg = baseline_report["aggregates"]["avg_overall_score"]

    if base_agg > 0 and cur_agg < base_agg * (1 - threshold):
        drop = round(base_agg - cur_agg, 3)
        regressions.append({
            "type": "overall_score",
            "severity": "critical",
            "current": cur_agg,
            "baseline": base_agg,
            "drop": drop,
            "drop_pct": round(drop / base_agg * 100, 1),
        })
        log.error(
            "regression_detected_critical",
            current=cur_agg,
            baseline=base_agg,
            drop=drop,
        )

    # ── Per-case regressions ──────────────────────────────────────────────────
    baseline_cases = {r["case_id"]: r for r in baseline_report.get("results", [])}

    for current_case in current_report.get("results", []):
        case_id = current_case["case_id"]
        if case_id not in baseline_cases:
            continue  # New test case — no baseline to compare against

        base_case = baseline_cases[case_id]
        cur_score = current_case["evaluation"].get("overall", 0)
        base_score = base_case["evaluation"].get("overall", 0)

        if base_score > 0 and cur_score < base_score * (1 - threshold):
            drop = round(base_score - cur_score, 3)
            regressions.append({
                "type": "case_regression",
                "severity": "warning",
                "case_id": case_id,
                "question": current_case["question"],
                "current_score": cur_score,
                "baseline_score": base_score,
                "drop": drop,
                "drop_pct": round(drop / base_score * 100, 1),
            })
            log.warning(
                "regression_detected_case",
                case_id=case_id,
                current_score=cur_score,
                baseline_score=base_score,
                drop=drop,
            )

    log.info(
        "regression_detection_completed",
        regressions_found=len(regressions),
        critical=[r for r in regressions if r["severity"] == "critical"],
    )

    return regressions


def save_baseline(report: dict, path: str = _DEFAULT_BASELINE) -> None:
    """Overwrite the baseline file with a new report."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("baseline_saved", path=path)
