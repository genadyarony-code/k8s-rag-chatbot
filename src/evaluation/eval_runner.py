"""
Evaluation runner for the golden test set.

Loads all cases from tests/eval/eval_dataset.json, runs each through the
full RAG pipeline (retrieve → generate), scores the answer with the LLM
judge, and writes a timestamped JSON report to tests/eval/reports/.

Designed to run as an offline job (CI, nightly cron) rather than during
normal request handling. Import and call `run_full_eval()` directly, or
use the CLI wrapper at tests/eval/run_eval.py.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agent.nodes import generate_node, retrieve_node
from src.evaluation.llm_judge import get_judge
from src.observability.logging_config import get_logger

log = get_logger(__name__)

_DEFAULT_DATASET = "tests/eval/eval_dataset.json"
_DEFAULT_REPORTS = "tests/eval/reports"


def load_eval_dataset(path: str = _DEFAULT_DATASET) -> dict:
    """Load and return the evaluation dataset."""
    with open(path) as f:
        return json.load(f)


async def run_eval_case(test_case: dict) -> dict:
    """
    Run a single test case through the full pipeline and score the result.

    Returns a dict with: case_id, question, answer, sources, evaluation,
    keyword_coverage, source_coverage, difficulty.
    """
    case_id = test_case["id"]
    question = test_case["question"]

    log.info("eval_case_started", case_id=case_id)

    # Run retrieval + generation (both are sync; wrap in threads)
    state: dict = {"question": question, "session_id": f"eval_{case_id}"}
    state = await asyncio.to_thread(retrieve_node, state)
    state = await asyncio.to_thread(generate_node, state)

    answer: str = state["answer"]
    sources: list[str] = state["sources"]

    # LLM judge evaluation
    judge = get_judge()
    evaluation = await asyncio.to_thread(judge.evaluate, question, answer, sources)

    # Keyword coverage — fraction of expected keywords present in the answer
    expected_kw = test_case.get("expected_keywords", [])
    matched_kw = [kw for kw in expected_kw if kw.lower() in answer.lower()]
    keyword_coverage = len(matched_kw) / max(len(expected_kw), 1)

    # Source coverage — fraction of expected source files referenced
    expected_src = test_case.get("expected_sources", [])
    matched_src = [s for s in expected_src if any(s in src for src in sources)]
    source_coverage = len(matched_src) / max(len(expected_src), 1)

    result = {
        "case_id": case_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "evaluation": evaluation,
        "keyword_coverage": round(keyword_coverage, 3),
        "source_coverage": round(source_coverage, 3),
        "difficulty": test_case.get("difficulty", "unknown"),
    }

    log.info(
        "eval_case_completed",
        case_id=case_id,
        overall_score=evaluation.get("overall"),
        keyword_coverage=round(keyword_coverage, 3),
        source_coverage=round(source_coverage, 3),
    )

    return result


async def run_full_eval(dataset_path: str = _DEFAULT_DATASET) -> dict:
    """
    Run the complete evaluation suite and return a report dict.

    The report structure is compatible with regression_detector.detect_regressions().
    """
    dataset = load_eval_dataset(dataset_path)
    test_cases = dataset["test_cases"]

    log.info("eval_run_started", total_cases=len(test_cases), dataset_version=dataset.get("version"))

    results = []
    for test_case in test_cases:
        result = await run_eval_case(test_case)
        results.append(result)

    # Aggregate scores (guard against evaluation_error zeros)
    valid = [r for r in results if r["evaluation"].get("overall", 0) > 0]

    def _avg(key: str) -> float:
        vals = [r["evaluation"][key] for r in valid if key in r["evaluation"]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    avg_overall = _avg("overall")
    avg_accuracy = _avg("accuracy")
    avg_keyword = round(sum(r["keyword_coverage"] for r in results) / len(results), 3)
    avg_source = round(sum(r["source_coverage"] for r in results) / len(results), 3)
    failures = [r for r in results if r["evaluation"].get("overall", 0) < 3]

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_version": dataset.get("version"),
        "total_cases": len(test_cases),
        "results": results,
        "aggregates": {
            "avg_overall_score": avg_overall,
            "avg_accuracy_score": avg_accuracy,
            "avg_keyword_coverage": avg_keyword,
            "avg_source_coverage": avg_source,
            "failure_count": len(failures),
            "failure_rate": round(len(failures) / len(test_cases), 3),
        },
        "failures": failures,
    }

    log.info(
        "eval_run_completed",
        total_cases=len(test_cases),
        avg_overall_score=avg_overall,
        failures=len(failures),
    )

    return report


def save_eval_report(report: dict, output_path: str = _DEFAULT_REPORTS) -> str:
    """Persist an eval report to a timestamped JSON file and return its path."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}/eval_report_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    log.info("eval_report_saved", path=filename)
    return filename
