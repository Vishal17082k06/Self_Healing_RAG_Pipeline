# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG (Retrieval-Augmented Generation) chatbot for a college event website, with a
**self-healing pipeline**: a monitor evaluates answer quality after each build, and if
quality degrades below baseline, it automatically retries ingestion with different
chunking strategies and hot-swaps the new index into the running app — no redeploy needed.

## Running locally

```bash
# Start supporting services (MLflow tracking + Phoenix tracing)
docker compose up -d mlflow phoenix

# Run the app directly (needs GROQ_API_KEY and/or OPENAI_API_KEY in .env)
uvicorn main:app --reload

# Or run the full stack in containers
docker compose up -d
```

Copy `.env.example` to `.env` and fill in keys before running. `ADMIN_TOKEN` protects
the `/admin/reload-index` endpoint that the monitor uses to trigger hot-swaps.

## Tests

```bash
pytest -v
```

`pytest.ini` sets `pythonpath = .`. Tests live in `tests/test_rag.py` and mock
`main.retriever` / `main.llm` — they don't hit real APIs or vector stores.

## Evaluation and self-healing

```bash
python eval.py      # Runs test_questions.json against a live /chat endpoint (localhost:8000), scores with Ragas, logs to MLflow
python monitor.py   # Runs eval.py, checks scores against baseline_metrics.json thresholds, self-heals on degradation
```

Both require the app (`main.py`) to already be running and reachable, since `eval.py`
calls the live `/chat` endpoint rather than importing the chain directly.

**Self-healing loop** (`monitor.py`):
1. Runs eval, pulls latest Ragas metrics from MLflow (`Event_Chatbot_Evaluations` experiment).
2. Compares against `baseline_metrics.json` thresholds (faithfulness, context_recall,
   context_precision, answer_relevancy_answerable).
3. On degradation, tries `HEALING_STRATEGIES` in order (currently smaller-chunk /
   higher-overlap variants), rebuilding the vector index and calling
   `POST /admin/reload-index` on the running app after each attempt.
4. Re-evaluates after each attempt; stops on first success. If `MAX_HEALING_ATTEMPTS`
   (2) are exhausted, exits non-zero to fail the CI build.

Note: `answer_relevancy_overall` is intentionally excluded from healing triggers because
it conflates correct refusals (questions the bot should decline to answer) with genuinely
poor answers. Use `answer_relevancy_answerable` instead — see the note in
`baseline_metrics.json`.

**Golden eval set** (`test_questions.json`) is generated from `data/event_schedule.md` by
`python generate_eval_questions.py` — it only calls the LLM when the source doc's hash has
changed since the last run (tracked in `test_questions.meta.json`); `--force` overrides.
Regenerating does **not** auto-update `baseline_metrics.json` — `monitor.py` warns (does
not fail the build) if `test_questions.json`'s hash no longer matches
`baseline_metrics.json`'s `questions_hash`, signaling a deliberate re-run of `eval.py` and
re-stamp is needed before the thresholds can be trusted again.

## CI/CD (Jenkins)

`JenkinsFile` runs on every build: builds the `rag_app` image, brings up the full docker
compose stack, runs `dvc pull` for data sync, runs pytest inside the container, then runs
`monitor.py` (the self-healing check), then re-queries the latest MLflow run as a final
quality gate that fails the build if metrics are still below threshold after healing.

Requires three Jenkins credentials (Secret text), injected via the pipeline's
`environment {}` block and passed through to the `rag_app` container by
`docker-compose.yml`'s `environment:` list: `openai-api-key`, `groq-api-key`,
`admin-token`. `admin-token` must match whatever `main.py`'s `/admin/reload-index`
checks against — without it, `monitor.py`'s reload call 403s silently and healing
rebuilds never actually reach the running app (see `DEBUGGING_LOG.md` Case 09).

## Architecture

**Versioned vector index (`main.py`)** — Chroma indexes are never rebuilt in place.
Each build writes to `chroma_db/v_<timestamp>/`, and only once the build succeeds does
`current_version.txt` get updated to point at it. Old versions beyond
`MAX_INDEX_VERSIONS` (2) are pruned, but the version currently live is never deleted
out from under in-flight requests. `chroma_db/` itself is DVC-tracked
(`chroma_db.dvc`), not committed directly.

**Hot-swap without restart** — `swap_retriever()` replaces the module-level
`vectorstore`/`retriever` globals under a lock. `POST /admin/reload-index` (admin-token
gated) is the only way to trigger this outside of process start; `monitor.py` calls it
remotely over HTTP after a successful healing rebuild, treating the app as a separate
running service rather than importing it.

**LLM fallback chain** — Groq (`llama-3.3-70b-versatile`) is primary for speed/cost;
OpenAI (`gpt-3.5-turbo`) is the fallback, both at request-init time (`get_llm()`) and
per-request if the primary call throws (`/chat` endpoint's inner try/except). Which path
was used gets logged to MLflow (`fallback_triggered`, `actual_llm_used`).

**Observability is two-layered and intentionally separate**:
- **Phoenix** (`phoenix.otel` + `openinference` LangChain instrumentor) — traces every
  LangChain call automatically via OpenTelemetry, for debugging chain behavior.
- **MLflow** — manually logged params/metrics per run, organized into distinct
  experiments: `self-healing-rag` (ingestion + query runs from `main.py`),
  `Event_Chatbot_Evaluations` (Ragas scores from `eval.py`), `Self_Healing_Monitor`
  (monitor.py's own run tree, with nested `healing_attempt_*` runs per strategy).

**Feedback storage** (`feedback_store.py`) — plain SQLite (`feedback.db`), no ORM.
`POST /feedback` records thumbs-up/down + tags/comment against a `session_id`/`message_id`
pair from a prior `/chat` call. Not currently wired into the healing loop.

**Data flow**: source content lives in `data/event_schedule.md` (single file, loaded via
`TextLoader` + `CharacterTextSplitter`). `test_questions.json` is the golden eval set;
each entry has an `expects_refusal` flag distinguishing "should answer" vs. "should
decline" questions, which is why `answer_relevancy` is split into `_answerable` and
`_refusals` variants rather than averaged together.
