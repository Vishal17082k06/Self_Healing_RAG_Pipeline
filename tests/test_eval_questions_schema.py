import json


def test_test_questions_json_has_valid_schema():
    with open("test_questions.json") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) > 0

    for item in data:
        assert isinstance(item.get("question"), str) and item["question"].strip()
        assert isinstance(item.get("ground_truth"), str) and item["ground_truth"].strip()
        assert isinstance(item.get("expects_refusal"), bool)


def test_test_questions_json_has_at_least_one_refusal_question():
    with open("test_questions.json") as f:
        data = json.load(f)

    assert any(item["expects_refusal"] for item in data), (
        "Golden eval set should include at least one refusal question so "
        "answer_relevancy_refusals stays meaningful"
    )
