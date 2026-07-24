import sys
import os
import json
import subprocess
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("Self_Healing_Monitor")

def load_baseline():
    with open("baseline_metrics.json", "r") as f:
        return json.load(f)

def run_eval_and_get_scores():
    """Runs eval.py as a subprocess (using the current venv's Python) and pulls the latest MLflow run's metrics."""
    print("Running evaluation...")
    subprocess.run([sys.executable, "eval.py"], check=True)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Event_Chatbot_Evaluations")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    latest_run = runs[0]
    return latest_run.data.metrics

def check_degradation(current_scores, baseline):
    thresholds = baseline["thresholds"]
    issues = []

    if current_scores.get("faithfulness", 1.0) < thresholds["faithfulness_min"]:
        issues.append(f"faithfulness dropped to {current_scores['faithfulness']:.2f} (min: {thresholds['faithfulness_min']})")

    if current_scores.get("context_recall", 1.0) < thresholds["context_recall_min"]:
        issues.append(f"context_recall dropped to {current_scores['context_recall']:.2f} (min: {thresholds['context_recall_min']})")

    if current_scores.get("context_precision", 1.0) < thresholds["context_precision_min"]:
        issues.append(f"context_precision dropped to {current_scores['context_precision']:.2f} (min: {thresholds['context_precision_min']})")

    # Use the refusal-excluded relevancy metric, not the blended one — see baseline_metrics.json note
    if current_scores.get("answer_relevancy_answerable", 1.0) < thresholds["answer_relevancy_answerable_min"]:
        issues.append(
            f"answer_relevancy_answerable dropped to {current_scores['answer_relevancy_answerable']:.2f} "
            f"(min: {thresholds['answer_relevancy_answerable_min']})"
        )

    return issues

def trigger_healing(issues):
    """The actual self-healing action: re-chunk with adjusted parameters."""
    print(f"\n⚠️  DEGRADATION DETECTED: {len(issues)} issue(s)")
    for issue in issues:
        print(f"  - {issue}")

    print("\nTriggering self-healing: rebuilding vector store with adjusted chunking...")

    with mlflow.start_run(run_name="healing_trigger", nested=True):
        mlflow.log_param("trigger_reason", "; ".join(issues))
        mlflow.log_param("action_taken", "rebuild_with_smaller_chunks")

        from main import build_vector_db
        build_vector_db(chunk_size=300, chunk_overlap=75, force_rebuild=True)

        mlflow.log_metric("healing_triggered", 1)

    print("Rebuild complete. Recommend re-running eval.py to confirm improvement.")

def run_monitor():
    baseline = load_baseline()
    current_scores = run_eval_and_get_scores()

    print(f"\nCurrent scores: {current_scores}")

    issues = check_degradation(current_scores, baseline)

    with mlflow.start_run(run_name="monitor_check"):
        loggable_metrics = {k: v for k, v in current_scores.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(loggable_metrics)
        mlflow.log_metric("issues_found", len(issues))

        if issues:
            trigger_healing(issues)
        else:
            print("\n✅ All metrics within acceptable range. No healing needed.")

if __name__ == "__main__":
    run_monitor()