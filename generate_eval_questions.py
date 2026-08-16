import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

SOURCE_DOC_PATH = "data/event_schedule.md"
QUESTIONS_PATH = "test_questions.json"
META_PATH = "test_questions.meta.json"
GENERATION_MODEL = "llama-3.3-70b-versatile"


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_regenerate(current_doc_hash, meta):
    if meta is None:
        return True
    return meta.get("source_doc_hash") != current_doc_hash


def parse_llm_questions(raw):
    text = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output was not valid JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError("LLM output must be a JSON array of question objects")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Each question entry must be an object, got: {item!r}")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Entry missing non-empty 'question' string: {item!r}")
        if not isinstance(item.get("ground_truth"), str) or not item["ground_truth"].strip():
            raise ValueError(f"Entry missing non-empty 'ground_truth' string: {item!r}")
        if not isinstance(item.get("expects_refusal"), bool):
            raise ValueError(f"Entry 'expects_refusal' must be a bool: {item!r}")

    return data


def load_meta():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


def build_prompt(doc_text, num_questions, num_refusals):
    return f"""You are generating a golden evaluation set for a RAG chatbot that answers
questions about the event schedule below.

Generate a JSON array (and nothing else - no markdown, no commentary) of exactly
{num_questions + num_refusals} objects, each with keys:
- "question": a natural-language question
- "ground_truth": the correct answer, grounded strictly in the document below
- "expects_refusal": boolean

Exactly {num_questions} objects must be answerable strictly from the document below,
with "expects_refusal": false and "ground_truth" containing the real answer.

Exactly {num_refusals} objects must ask about specific facts that are plausible for this
kind of event but are NOT mentioned anywhere in the document (e.g. a fee, a workshop, a
date that isn't listed), with "expects_refusal": true and "ground_truth" stating that the
schedule does not mention it.

Document:
---
{doc_text}
---

Respond with ONLY the JSON array."""


def generate_questions(doc_text, num_questions, num_refusals):
    llm = ChatGroq(model=GENERATION_MODEL, temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_prompt(doc_text, num_questions, num_refusals)
    response = llm.invoke(prompt)
    return parse_llm_questions(response.content)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the golden eval set from data/event_schedule.md")
    parser.add_argument("--num-questions", type=int, default=12)
    parser.add_argument("--num-refusals", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Regenerate even if the source doc hasn't changed")
    args = parser.parse_args()

    with open(SOURCE_DOC_PATH) as f:
        doc_text = f.read()
    doc_hash = hash_text(doc_text)

    meta = load_meta()
    if not args.force and not should_regenerate(doc_hash, meta):
        print(f"{SOURCE_DOC_PATH} unchanged since last generation — nothing to do. Use --force to regenerate anyway.")
        return

    print(f"Generating {args.num_questions} answerable + {args.num_refusals} refusal questions from {SOURCE_DOC_PATH}...")
    questions = generate_questions(doc_text, args.num_questions, args.num_refusals)

    with open(QUESTIONS_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    with open(META_PATH, "w") as f:
        json.dump({
            "source_doc_hash": doc_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"Wrote {len(questions)} questions to {QUESTIONS_PATH}.")
    print(json.dumps(questions, indent=2))
    print(f"\nReview the questions above, then re-run eval.py and re-baseline baseline_metrics.json.")


if __name__ == "__main__":
    main()
