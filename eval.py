import os

# On Windows, importing mlflow before torch corrupts torch's DLL loading (see main.py for
# details) and crashes the first time HuggingFaceEmbeddings pulls torch in later.
import torch  # noqa: F401

import json
import requests
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Ragas defaults to OpenAI (ChatOpenAI + OpenAIEmbeddings) for grading if no llm/embeddings
# are passed to evaluate() — independent of whatever LLM/embeddings the app under test uses.
# Point it at Gemini + the same local embedding model main.py uses so grading has no OpenAI
# dependency. Groq was dropped entirely (see main.py) after its free-tier rate limits (8000
# TPM / 200000 TPD, account-wide) were being hit routinely under normal eval/CI load — see
# DEBUGGING_LOG.md Case 07.
JUDGE_LLM_MODEL = "gemini-3.6-flash"  # gemini-2.0-flash was retired; API's own 404 pointed here
JUDGE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

judge_llm = ChatGoogleGenerativeAI(model=JUDGE_LLM_MODEL, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
judge_embeddings = HuggingFaceEmbeddings(model_name=JUDGE_EMBEDDING_MODEL)
judge_run_config = RunConfig(max_workers=2, max_wait=90, max_retries=5)

# answer_relevancy defaults to strictness=3 (3 completions per call, averaged). Ragas'
# is_multiple_completion_supported() only recognizes native OpenAI/VertexAI classes, so for
# any other provider (Gemini included) it falls back to firing the same prompt 3 separate
# times and merging the results via the provider's own _combine_llm_outputs(). We hit a real
# bug in langchain_groq's version of that merge (dict += dict, TypeError) — see
# DEBUGGING_LOG.md Case 07/12 — and there's no reason to assume another provider's merge
# code is bug-free either. strictness=1 means only one completion, so that merge path is
# never exercised at all; it also cuts this metric's call volume by 3x.
answer_relevancy.strictness = 1

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
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=judge_run_config,
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
