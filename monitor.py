import sys
import os

# On Windows, importing mlflow before torch corrupts torch's DLL loading (see main.py for
# details) and crashes the first time this process later imports torch — either directly
# or via `from main import build_vector_db` in apply_healing_strategy().
import torch  # noqa: F401

import hashlib
import json
import math
import subprocess
import mlflow
import requests
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("Self_Healing_Monitor")

MAX_HEALING_ATTEMPTS = 2
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
RAG_APP_URL = os.getenv("RAG_APP_URL", "http://127.0.0.1:8000")

# Strategy bank to attempt sequentially if baseline drops
HEALING_STRATEGIES = [
    {"chunk_size": 300, "chunk_overlap": 75, "name": "smaller_chunks_higher_overlap"},
    {"chunk_size": 200, "chunk_overlap": 50, "name": "granular_chunks"},
]

def load_baseline():
    with open("baseline_metrics.json", "r") as f:
        return json.load(f)

def check_questions_staleness(baseline, current_questions_hash):
    """Warns (does not fail the build) when test_questions.json has changed since
    baseline_metrics.json's thresholds were last calibrated against it — see
    generate_eval_questions.py, which stamps questions_hash on re-baseline."""
    baseline_hash = baseline.get("questions_hash")
    if baseline_hash != current_questions_hash:
        return (
            "test_questions.json does not match the golden set baseline_metrics.json was "
            "calibrated against (questions_hash mismatch) — thresholds may be stale. "
            "Re-run eval.py and update baseline_metrics.json's questions_hash."
        )
    return None

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

def _is_missing(value):
    """True for None or NaN — Ragas returns NaN for any row a metric job failed on
    (rate limits, timeouts, parse errors), and `NaN < threshold` is always False in
    Python, so an unguarded comparison would silently treat a failed eval as healthy."""
    return value is None or (isinstance(value, float) and math.isnan(value))

TRACKED_METRICS = [
    ("faithfulness", "faithfulness_min"),
    ("context_recall", "context_recall_min"),
    ("context_precision", "context_precision_min"),
    ("answer_relevancy_answerable", "answer_relevancy_answerable_min"),
]

def check_degradation(current_scores, baseline):
    thresholds = baseline["thresholds"]
    issues = []

    for metric_name, threshold_key in TRACKED_METRICS:
        value = current_scores.get(metric_name)
        if _is_missing(value):
            issues.append(f"{metric_name} is missing/NaN — evaluation failed to produce a score, treating as a failure")
        elif value < thresholds[threshold_key]:
            issues.append(f"{metric_name} dropped to {value:.2f} (min: {thresholds[threshold_key]})")

    return issues

def trigger_index_reload():
    """Tells the already-running rag_app to atomically swap in the newly built
    index version, so healing takes effect without restarting the app."""
    try:
        resp = requests.post(
            f"{RAG_APP_URL}/admin/reload-index",
            headers={"X-Admin-Token": ADMIN_TOKEN},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"Index reload triggered successfully: {resp.json()}")
        else:
            print(f"Index reload failed ({resp.status_code}): {resp.text}")
    except requests.RequestException as e:
        print(f"Could not reach rag_app to trigger index reload: {e}")

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
        trigger_index_reload()

        mlflow.log_metric("healing_triggered", 1)

def run_monitor():
    baseline = load_baseline()

    with mlflow.start_run(run_name="monitor_check"):
        with open("test_questions.json", "rb") as f:
            current_questions_hash = hashlib.sha256(f.read()).hexdigest()
        staleness_warning = check_questions_staleness(baseline, current_questions_hash)
        if staleness_warning:
            print(f"\n WARNING: {staleness_warning}")
            mlflow.set_tag("questions_hash_stale", True)

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