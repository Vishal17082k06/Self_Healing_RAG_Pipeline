import hashlib
import pytest
from generate_eval_questions import hash_text, should_regenerate, parse_llm_questions


def test_hash_text_is_deterministic_sha256_hexdigest():
    assert hash_text("hello") == hashlib.sha256("hello".encode("utf-8")).hexdigest()
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")


def test_should_regenerate_true_when_no_prior_meta():
    assert should_regenerate(current_doc_hash="abc123", meta=None) is True


def test_should_regenerate_true_when_hash_differs():
    assert should_regenerate(current_doc_hash="abc123", meta={"source_doc_hash": "different"}) is True


def test_should_regenerate_false_when_hash_matches():
    assert should_regenerate(current_doc_hash="abc123", meta={"source_doc_hash": "abc123"}) is False


def test_parse_llm_questions_accepts_valid_list():
    raw = '[{"question": "Where?", "ground_truth": "Here", "expects_refusal": false}]'
    parsed = parse_llm_questions(raw)
    assert parsed == [{"question": "Where?", "ground_truth": "Here", "expects_refusal": False}]


def test_parse_llm_questions_strips_markdown_code_fence():
    raw = '```json\n[{"question": "Where?", "ground_truth": "Here", "expects_refusal": false}]\n```'
    parsed = parse_llm_questions(raw)
    assert len(parsed) == 1


@pytest.mark.parametrize("raw", [
    '{"not": "a list"}',
    '[{"question": "Where?"}]',
    '[{"question": "Where?", "ground_truth": "Here", "expects_refusal": "not_a_bool"}]',
    'not json at all',
])
def test_parse_llm_questions_rejects_malformed_output(raw):
    with pytest.raises(ValueError):
        parse_llm_questions(raw)
