#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${ROOT}/mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${ROOT}/mlartifacts}"
WORKBENCH_HOST="${WORKBENCH_HOST:-127.0.0.1}"
WORKBENCH_PORT="${WORKBENCH_PORT:-8000}"

mkdir -p "$MLFLOW_ARTIFACT_ROOT"

if ! curl -sf "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; then
  echo "Starting MLflow tracking server at ${MLFLOW_TRACKING_URI}"
  mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --host 127.0.0.1 \
    --port 5000 \
    >/tmp/mlflow-server.log 2>&1 &
  for _ in $(seq 1 40); do
    if curl -sf "${MLFLOW_TRACKING_URI}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

echo "MLflow UI: ${MLFLOW_TRACKING_URI}"
echo "Workbench: http://${WORKBENCH_HOST}:${WORKBENCH_PORT}"
exec uvicorn app.main:app --host "$WORKBENCH_HOST" --port "$WORKBENCH_PORT"
