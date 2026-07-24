import os
import json
import requests
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("Event_Chatbot_Evaluations")

def run_evaluation():
    print("Loading Golden Dataset...")
    with open("test_questions.json", "r") as f:
        test_data = json.load(f)

    questions, answers, contexts_list, ground_truths, expects_refusal_flags = [], [], [], [], []

    print("Querying chatbot for each test question...")
    for index, item in enumerate(test_data):
        question = item["question"]
        expected = item["ground_truth"]
        expects_refusal = item.get("expects_refusal", False)

        response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"question": question}
        )
        result = response.json()

        if response.status_code != 200:
            raise RuntimeError(f"Chat request failed ({response.status_code}): {result.get('detail', result)}")

        bot_answer = result["answer"]
        contexts = result["contexts"]

        print(f"\n--- Test {index + 1} ---")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Bot: {bot_answer}")

        questions.append(question)
        answers.append(bot_answer)
        contexts_list.append(contexts)
        ground_truths.append(expected)
        expects_refusal_flags.append(expects_refusal)

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("\nRunning Ragas evaluation...")
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = results.to_pandas()
    scores["expects_refusal"] = expects_refusal_flags

    print("\n--- Ragas Scores (per question) ---")
    print(scores[["question", "expects_refusal", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    # Split answer_relevancy: answerable questions vs. expected-refusal questions
    answerable = scores[scores["expects_refusal"] == False]
    refusals = scores[scores["expects_refusal"] == True]

    answer_relevancy_answerable = answerable["answer_relevancy"].mean()
    answer_relevancy_refusals = refusals["answer_relevancy"].mean() if len(refusals) > 0 else None

    print(f"\nanswer_relevancy (answerable only): {answer_relevancy_answerable:.3f}")
    print(f"answer_relevancy (refusal questions): {answer_relevancy_refusals}")
    print("Note: refusal questions score ~0 on answer_relevancy by design of the metric — not a quality issue.")

    with mlflow.start_run(run_name="ragas_eval"):
        mlflow.log_param("num_test_questions", len(test_data))
        mlflow.log_param("num_refusal_questions", len(refusals))

        mlflow.log_metric("faithfulness", scores["faithfulness"].mean())
        mlflow.log_metric("context_precision", scores["context_precision"].mean())
        mlflow.log_metric("context_recall", scores["context_recall"].mean())

        # Split relevancy metrics, logged separately and honestly
        mlflow.log_metric("answer_relevancy_overall", scores["answer_relevancy"].mean())
        mlflow.log_metric("answer_relevancy_answerable", answer_relevancy_answerable)
        if answer_relevancy_refusals is not None:
            mlflow.log_metric("answer_relevancy_refusals", answer_relevancy_refusals)

        mlflow.log_table(scores, artifact_file="ragas_detailed_results.json")

    print("\nEvaluation complete. Scores logged to MLflow.")

if __name__ == "__main__":
    run_evaluation()