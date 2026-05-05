"""
Unit tests for the evaluation and A/B testing modules.

All tests here are fast (no API calls, no model downloads):
- LLM judge JSON parsing (including markdown fences)
- A/B variant assignment determinism and traffic distribution
- Regression detector logic
- Live sampler probabilistic sampling
- Eval dataset loads and validates
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.ab_testing import ABTest, PROMPT_STYLE_TEST
from src.evaluation.regression_detector import detect_regressions


# ── A/B Testing ───────────────────────────────────────────────────────────────


def test_ab_variant_assignment_is_deterministic():
    """Same session_id must always receive the same variant."""
    session = "stable_session_abc123"
    v1 = PROMPT_STYLE_TEST.assign_variant(session)
    v2 = PROMPT_STYLE_TEST.assign_variant(session)
    v3 = PROMPT_STYLE_TEST.assign_variant(session)
    assert v1 == v2 == v3


def test_ab_variants_cover_all_sessions():
    """A large sample of sessions should hit every variant."""
    variants_seen = set()
    for i in range(500):
        v = PROMPT_STYLE_TEST.assign_variant(f"session_{i}")
        variants_seen.add(v)
    assert variants_seen == {"control", "concise", "detailed"}


def test_ab_traffic_split_respected():
    """control=50 %, concise=25 %, detailed=25 % (±5 % tolerance)."""
    counts: dict[str, int] = {"control": 0, "concise": 0, "detailed": 0}
    n = 2000
    for i in range(n):
        v = PROMPT_STYLE_TEST.assign_variant(f"traffic_test_{i}")
        counts[v] += 1

    assert 0.45 <= counts["control"] / n <= 0.55, f"control={counts['control']/n:.2%}"
    assert 0.20 <= counts["concise"] / n <= 0.30, f"concise={counts['concise']/n:.2%}"
    assert 0.20 <= counts["detailed"] / n <= 0.30, f"detailed={counts['detailed']/n:.2%}"


def test_ab_traffic_split_must_sum_to_one():
    with pytest.raises(AssertionError):
        ABTest(
            test_id="bad_test",
            variants={"a": {}, "b": {}},
            traffic_split={"a": 0.6, "b": 0.6},  # sums to 1.2
        )


def test_ab_get_variant_config_returns_dict():
    cfg = PROMPT_STYLE_TEST.get_variant_config("any_session")
    assert isinstance(cfg, dict)
    assert "instruction" in cfg


def test_ab_different_sessions_can_differ():
    """Not all sessions should end up in the same bucket."""
    variants = {PROMPT_STYLE_TEST.assign_variant(f"s_{i}") for i in range(20)}
    assert len(variants) > 1


# ── LLM Judge JSON parsing ────────────────────────────────────────────────────


def _make_eval_dict(overall: int = 4) -> dict:
    return {
        "accuracy": overall,
        "completeness": overall,
        "relevance": overall,
        "grounding": overall,
        "overall": overall,
        "explanation": "Test explanation.",
        "issues": [],
    }


def test_judge_parses_clean_json():
    """Judge must handle clean JSON output without fences."""
    from src.evaluation.llm_judge import LLMJudge

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(_make_eval_dict(4))

    with patch("src.evaluation.llm_judge.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        judge = LLMJudge()
        result = judge.evaluate("Q", "A", ["source.md"])

    assert result["overall"] == 4
    assert result["accuracy"] == 4


def test_judge_strips_markdown_fences():
    """Judge must strip ```json ... ``` wrapper that GPT sometimes adds."""
    from src.evaluation.llm_judge import LLMJudge

    fenced = "```json\n" + json.dumps(_make_eval_dict(3)) + "\n```"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = fenced

    with patch("src.evaluation.llm_judge.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        judge = LLMJudge()
        result = judge.evaluate("Q", "A", [])

    assert result["overall"] == 3


def test_judge_returns_zero_on_api_error():
    """Judge must return a safe zero-dict rather than raising on failure."""
    from src.evaluation.llm_judge import LLMJudge

    with patch("src.evaluation.llm_judge.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("API down")
        judge = LLMJudge()
        result = judge.evaluate("Q", "A", [])

    assert result["overall"] == 0
    assert "evaluation_error" in result["issues"]


# ── Regression Detector ───────────────────────────────────────────────────────


def _make_report(overall: float, cases: list[tuple[str, int]] = None) -> dict:
    """Build a minimal eval report for regression testing."""
    cases = cases or []
    return {
        "aggregates": {"avg_overall_score": overall},
        "results": [
            {
                "case_id": cid,
                "question": f"Q for {cid}",
                "evaluation": {"overall": score},
            }
            for cid, score in cases
        ],
    }


def test_regression_no_regressions():
    baseline = _make_report(4.0, [("c1", 4)])
    current = _make_report(4.0, [("c1", 4)])
    assert detect_regressions(current, baseline) == []


def test_regression_detects_critical_drop():
    baseline = _make_report(4.0)
    current = _make_report(2.0)  # 50 % drop > 30 % threshold
    regressions = detect_regressions(current, baseline, threshold=0.3)
    assert len(regressions) == 1
    assert regressions[0]["severity"] == "critical"
    assert regressions[0]["type"] == "overall_score"


def test_regression_detects_case_drop():
    baseline = _make_report(4.0, [("case1", 5)])
    current = _make_report(4.0, [("case1", 2)])  # 60 % drop
    regressions = detect_regressions(current, baseline, threshold=0.3)
    case_regressions = [r for r in regressions if r["type"] == "case_regression"]
    assert len(case_regressions) == 1
    assert case_regressions[0]["case_id"] == "case1"


def test_regression_threshold_respected():
    """A small drop (< threshold) must NOT be flagged."""
    baseline = _make_report(4.0, [("c1", 4)])
    current = _make_report(3.8, [("c1", 4)])  # 5 % drop < 30 %
    assert detect_regressions(current, baseline, threshold=0.3) == []


def test_regression_ignores_new_cases():
    """New test cases with no baseline counterpart must not raise an error."""
    baseline = _make_report(4.0, [("c1", 4)])
    current = _make_report(4.0, [("c1", 4), ("c_new", 3)])
    assert detect_regressions(current, baseline) == []


# ── Live Sampler ──────────────────────────────────────────────────────────────


def test_sampler_always_samples_at_rate_1():
    from src.evaluation.live_sampler import LiveSampler
    sampler = LiveSampler(sample_rate=1.0)
    assert all(sampler.should_sample() for _ in range(20))


def test_sampler_never_samples_at_rate_0():
    from src.evaluation.live_sampler import LiveSampler
    sampler = LiveSampler(sample_rate=0.0)
    assert not any(sampler.should_sample() for _ in range(20))


def test_sampler_approximate_rate():
    from src.evaluation.live_sampler import LiveSampler
    sampler = LiveSampler(sample_rate=0.5)
    hits = sum(sampler.should_sample() for _ in range(1000))
    # 50 % ± 10 %
    assert 400 <= hits <= 600, f"Expected ~500 hits, got {hits}"


# ── Eval Dataset ──────────────────────────────────────────────────────────────


def test_eval_dataset_loads():
    with open("tests/eval/eval_dataset.json") as f:
        dataset = json.load(f)

    assert "test_cases" in dataset
    assert len(dataset["test_cases"]) >= 5

    for case in dataset["test_cases"]:
        assert "id" in case
        assert "question" in case
        assert "expected_keywords" in case
        assert "difficulty" in case


def test_eval_dataset_has_varied_difficulty():
    with open("tests/eval/eval_dataset.json") as f:
        dataset = json.load(f)

    difficulties = {c["difficulty"] for c in dataset["test_cases"]}
    assert len(difficulties) >= 2  # at least easy + medium or medium + hard
