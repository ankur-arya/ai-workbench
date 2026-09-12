# AI Workbench

A lightweight UI that walks a user from **idea → experiment → dataset → models → evaluation → production** using the [MLflow AI Platform](https://mlflow.org/) (tracking, model registry aliases, and optional tracing). This repository is the workbench; MLflow remains the system of record for runs, artifacts, and registered models.

The concrete demo is **3-class sentiment classification** (positive / negative / neutral) on a tiny synthetic dataset. The primary metric is **macro F1**.

## Design decisions

### 1. When a model is finalized, is production another experiment, another MLflow instance, or something else?

**Something else: the same MLflow instance’s Model Registry, addressed by an alias.**

| Approach | Use it? | Why |
| --- | --- | --- |
| Another *training* experiment named “prod” | No | That would re-run research, not serve a frozen artifact. |
| A second MLflow server | No (for this demo / most teams) | Split-brain registry, extra ops. Use a second instance only if you truly isolate prod metadata. |
| **Model Registry alias (`champion` / `challenger`)** | **Yes** | Promotion is `mlflow.register_model` + `set_registered_model_alias`. Production loads `models:/sentiment-classifier@champion`. |

What *does* get a separate experiment is optional **production tracing** (`workbench-production-traces`). Those spans are observability, not a second training loop.

```text
Dev / eval runs          Promotion                 Production runtime
─────────────────        ──────────                ──────────────────
Experiment               Model Registry            pyfunc load
  sentiment-workbench      sentiment-classifier      models:/name@champion
  run metrics (F1…)        alias champion ────────►  optional traces in
  logged pyfunc model      alias challenger          workbench-production-traces
```

### 2. Do we need our own UI, or can we lean on MLflow’s UI?

**Both, with a clear split.**

- **This workbench UI** orchestrates the lifecycle: create an experiment, load/import a dataset, define a prompt+LLM on the Model step, run/eval, compare macro F1, promote a run, run champion inference.
- **MLflow’s native UI** is the deep-link destination for run details, artifacts, traces, and the registry. We do not reinvent the tracking UI.

### 3. Why is Model a separate step from Train?

**Define the artifact, then run it.** The Model step is where you attach a system prompt (`{labels}` placeholder) to an LLM or `local-heuristic`. That saved config is the candidate classifier. Train only **evaluates** that definition on the selected dataset and logs F1 / a pyfunc model to MLflow. Mixing prompt authoring into Train hid the fact that the prompt+LLM *is* the model.

## Stack

The repo started as an empty README (no existing OpenAI helper). The workbench is:

- **FastAPI** + a small static frontend (`app/static`)
- **MLflow 3.3** tracking + registry aliases + pyfunc logging + tracing
- **OpenAI** chat completions for `gpt-5-nano` (falls back to `OPENAI_FALLBACK_MODEL`, then `gpt-4.1-nano` / `gpt-4o-mini`)
- **`local-heuristic`** keyword classifier so the full lifecycle can be demoed offline

Secrets stay in a local `.env` (gitignored). Copy `.env.example`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — LLM path requires all three: OPENAI_API_KEY, OPENAI_ORG, OPENAI_PROJECT
```

Start the MLflow tracking server (SQLite + local artifacts) and the workbench:

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

Or in two terminals:

```bash
mlflow server \
  --backend-store-uri sqlite:///./mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 --port 5000

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Workbench: http://127.0.0.1:8000
- MLflow UI: http://127.0.0.1:5000

## Demo walkthrough

1. Open the workbench. Confirm the MLflow pill is connected.
2. **Experiment** — create `sentiment-workbench` (or pick an existing one).
3. **Dataset** — load the bundled CSV, generate extra rows, or **Import CSV** (`split`, `text`, `label` columns; labels must be positive / negative / neutral).
4. **Model** — edit the prompt template, pick `local-heuristic` or `gpt-5-nano`, save the config. This is the classifier definition.
5. **Train** — run evaluation using the saved config + selected dataset (overrides optional). Logs macro F1, the prompt text, and a pyfunc model.
6. **Evaluate** — compare runs (sorted by F1). Follow the MLflow links for artifacts.
7. **Promote** — register the best run as `sentiment-classifier` and set alias `champion` (optionally `challenger` first).
8. **Production** — score new lines of text. The app loads `models:/sentiment-classifier@champion` and writes an inference span to `workbench-production-traces`.

Headless equivalent (uses the local heuristic, no API key):

```bash
# MLflow server must be running, or set MLFLOW_TRACKING_URI=file:./mlruns
python scripts/run_demo.py
```

### Models

| Id | When to use |
| --- | --- |
| `gpt-5-nano` | Default cheap OpenAI path (`.env` `OPENAI_MODEL`) |
| `gpt-4o-mini` | Configured fallback (`OPENAI_FALLBACK_MODEL`) |
| `local-heuristic` | Offline demo / CI |
| `local-heuristic-weak` | Weaker lexicon so comparison is visible |

LLM calls construct `OpenAI(api_key=..., organization=..., project=...)`. If any of `OPENAI_API_KEY`, `OPENAI_ORG`, or `OPENAI_PROJECT` is missing, the workbench raises a clear error instead of sending an unscoped request. The local heuristic path does not need these variables.

GPT-5 family calls use `max_completion_tokens` (not `max_tokens`). If the preferred id is missing or rejected, the client walks the fallback chain.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Tests use a temporary file-backed MLflow store and the local heuristic (no OpenAI key, no tracking server).

## Repository layout

```text
app/                 FastAPI + static workbench UI
workbench/           config, dataset, classifier, metrics, MLflow ops, pipeline
data/sentiment.csv   synthetic train/test split
scripts/start.sh     MLflow server + workbench
scripts/run_demo.py  headless lifecycle
tests/               unit + API + MLflow lifecycle
.env.example         placeholder keys only
```

`.env`, `.env.local`, and other `.env.*` files are gitignored and must not be committed.
