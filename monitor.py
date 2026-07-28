import sys
import os
import json
import subprocess
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("Self_Healing_Monitor")

MAX_HEALING_ATTEMPTS = 2

# Strategy bank to attempt sequentially if baseline drops
HEALING_STRATEGIES = [
    {"chunk_size": 300, "chunk_overlap": 75, "name": "smaller_chunks_higher_overlap"},
    {"chunk_size": 200, "chunk_overlap": 50, "name": "granular_chunks"},
]

def load_baseline():
    with open("baseline_metrics.json", "r") as f:
        return json.load(f)

def run_eval_and_get_scores():
    """Runs eval.py as a subprocess and pulls the latest MLflow run's metrics."""
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
        issues.append(f"faithfulness dropped to {current_scores.get('faithfulness', 0):.2f} (min: {thresholds['faithfulness_min']})")

    if current_scores.get("context_recall", 1.0) < thresholds["context_recall_min"]:
        issues.append(f"context_recall dropped to {current_scores.get('context_recall', 0):.2f} (min: {thresholds['context_recall_min']})")

    if current_scores.get("context_precision", 1.0) < thresholds["context_precision_min"]:
        issues.append(f"context_precision dropped to {current_scores.get('context_precision', 0):.2f} (min: {thresholds['context_precision_min']})")

    if current_scores.get("answer_relevancy_answerable", 1.0) < thresholds["answer_relevancy_answerable_min"]:
        issues.append(
            f"answer_relevancy_answerable dropped to {current_scores.get('answer_relevancy_answerable', 0):.2f} "
            f"(min: {thresholds['answer_relevancy_answerable_min']})"
        )

    return issues

def apply_healing_strategy(strategy, issues):
    """Executes vector store rebuild using a specific strategy."""
    print(f"\nTriggering self-healing: rebuilding vector store with {strategy['name']}...")
    print(f"Parameters: chunk_size={strategy['chunk_size']}, chunk_overlap={strategy['chunk_overlap']}")

    with mlflow.start_run(run_name=f"healing_attempt_{strategy['name']}", nested=True):
        mlflow.log_param("trigger_reason", "; ".join(issues))
        mlflow.log_param("strategy_applied", strategy['name'])
        mlflow.log_param("chunk_size", strategy['chunk_size'])
        mlflow.log_param("chunk_overlap", strategy['chunk_overlap'])

        from main import build_vector_db
        build_vector_db(
            chunk_size=strategy["chunk_size"], 
            chunk_overlap=strategy["chunk_overlap"], 
            force_rebuild=True
        )

        mlflow.log_metric("healing_triggered", 1)

def run_monitor():
    baseline = load_baseline()
    
    with mlflow.start_run(run_name="monitor_check"):
        # 1. Initial Evaluation Pass
        current_scores = run_eval_and_get_scores()
        print(f"\nCurrent scores: {current_scores}")

        issues = check_degradation(current_scores, baseline)
        loggable_metrics = {k: v for k, v in current_scores.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(loggable_metrics)
        mlflow.log_metric("issues_found", len(issues))

        if not issues:
            print("\n All metrics within acceptable range. No healing needed.")
            sys.exit(0)

        print(f"\n DEGRADATION DETECTED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  - {issue}")

        # 2. Closed-Loop Healing Attempt Sequence
        for attempt_idx, strategy in enumerate(HEALING_STRATEGIES, start=1):
            print(f"\n [Healing Attempt {attempt_idx}/{MAX_HEALING_ATTEMPTS}]")
            apply_healing_strategy(strategy, issues)

            print("\n Verifying post-healing scores...")
            healed_scores = run_eval_and_get_scores()
            healed_issues = check_degradation(healed_scores, baseline)

            if not healed_issues:
                print(f"\n Self-healing SUCCEEDED on attempt {attempt_idx} using {strategy['name']}!")
                mlflow.log_metric("healing_success", 1)
                sys.exit(0)
            else:
                print(f" Attempt {attempt_idx} failed to resolve degradation. Remaining issues: {len(healed_issues)}")

        print("\n CRITICAL: Self-healing failed to restore metrics above baseline thresholds.")
        print("Failing build to block deployment of unrecoverable data/code degradation.")
        mlflow.log_metric("healing_success", 0)
        sys.exit(1)

if __name__ == "__main__":
    run_monitor()