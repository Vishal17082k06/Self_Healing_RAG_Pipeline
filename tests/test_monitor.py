import math
import pytest

from monitor import check_degradation

BASELINE = {
    "thresholds": {
        "faithfulness_min": 0.80,
        "context_recall_min": 0.80,
        "context_precision_min": 0.80,
        "answer_relevancy_answerable_min": 0.85,
    }
}

HEALTHY_SCORES = {
    "faithfulness": 0.93,
    "context_recall": 0.97,
    "context_precision": 0.92,
    "answer_relevancy_answerable": 0.99,
}


def test_no_issues_when_all_scores_above_threshold():
    assert check_degradation(HEALTHY_SCORES, BASELINE) == []


def test_issue_reported_when_score_below_threshold():
    scores = {**HEALTHY_SCORES, "faithfulness": 0.5}
    issues = check_degradation(scores, BASELINE)
    assert len(issues) == 1
    assert "faithfulness" in issues[0]


def test_nan_metric_is_treated_as_a_failure_not_a_silent_pass():
    """Regression test: NaN < threshold is False in Python, so an unguarded comparison
    would let a failed Ragas evaluation (rate limits, timeouts) look like a healthy run."""
    scores = {**HEALTHY_SCORES, "context_recall": math.nan}
    issues = check_degradation(scores, BASELINE)
    assert len(issues) == 1
    assert "context_recall" in issues[0]
    assert "missing/NaN" in issues[0]


def test_missing_metric_key_is_treated_as_a_failure():
    scores = {k: v for k, v in HEALTHY_SCORES.items() if k != "context_precision"}
    issues = check_degradation(scores, BASELINE)
    assert len(issues) == 1
    assert "context_precision" in issues[0]


def test_multiple_nan_metrics_all_reported():
    scores = {**HEALTHY_SCORES, "faithfulness": math.nan, "context_recall": math.nan}
    issues = check_degradation(scores, BASELINE)
    assert len(issues) == 2
