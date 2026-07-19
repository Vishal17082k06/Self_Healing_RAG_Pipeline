import json
import requests
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Event_Chatbot_Evaluations")

def run_evaluation():
    print("Loading Golden Dataset...")
    with open("test_questions.json", "r") as f:
        test_data = json.load(f)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    print("Querying chatbot for each test question...")
    for index, item in enumerate(test_data):
        question = item["question"]
        expected = item["ground_truth"]

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": question}
        )
        result = response.json()
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

    # Build Ragas-compatible dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("\nRunning Ragas evaluation (this calls an LLM as judge — costs a few cents)...")
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = results.to_pandas()
    print("\n--- Ragas Scores (per question) ---")
    print(scores.columns.tolist())
    print(scores[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    print(scores[["question", "answer_relevancy"]].to_string())
    with mlflow.start_run(run_name="ragas_eval"):
        mlflow.log_param("num_test_questions", len(test_data))
        mlflow.log_metric("faithfulness", scores["faithfulness"].mean())
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"].mean())
        mlflow.log_metric("context_precision", scores["context_precision"].mean())
        mlflow.log_metric("context_recall", scores["context_recall"].mean())
        mlflow.log_table(scores, artifact_file="ragas_detailed_results.json")

    print("\nEvaluation complete. Scores logged to MLflow.")

if __name__ == "__main__":
    run_evaluation()