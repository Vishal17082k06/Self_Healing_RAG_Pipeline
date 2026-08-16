from monitor import check_questions_staleness


def test_no_warning_when_hashes_match():
    baseline = {"questions_hash": "abc123"}
    assert check_questions_staleness(baseline, current_questions_hash="abc123") is None


def test_warning_when_hashes_differ():
    baseline = {"questions_hash": "abc123"}
    warning = check_questions_staleness(baseline, current_questions_hash="different")
    assert warning is not None
    assert "test_questions.json" in warning


def test_warning_when_baseline_has_no_questions_hash():
    baseline = {}
    warning = check_questions_staleness(baseline, current_questions_hash="abc123")
    assert warning is not None
