#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$ROOT_DIR/apps/api"
WEB_DIR="$ROOT_DIR/apps/web"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Neither python nor python3 found. Activate your conda environment first or set PYTHON_BIN." >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

require_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "Missing directory: $dir" >&2
    exit 1
  fi
}

require_dir "$API_DIR"
require_dir "$WEB_DIR"

resolve_project_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

if [[ -z "${MOONDREAM_CHECKPOINT:-}" ]]; then
  default_checkpoint="$(
    find "$ROOT_DIR/moondream-mini-v6-checkpoint" "$ROOT_DIR/checkpoints" \
      -name '*_best.pt' -print 2>/dev/null | sort | head -1 || true
  )"
  if [[ -n "$default_checkpoint" ]]; then
    export MOONDREAM_CHECKPOINT="$default_checkpoint"
  fi
fi

export MOONDREAM_TOKENIZER="$(resolve_project_path "${MOONDREAM_TOKENIZER:-artifacts/moondream_starmie_v1}")"
export MOONDREAM_DATA_ROOT="$(resolve_project_path "${MOONDREAM_DATA_ROOT:-moondream_ppe_vqa_data_v6}")"
if [[ -n "${MOONDREAM_CHECKPOINT:-}" ]]; then
  export MOONDREAM_CHECKPOINT="$(resolve_project_path "$MOONDREAM_CHECKPOINT")"
fi
export MOONDREAM_CORS_ORIGINS="${MOONDREAM_CORS_ORIGINS:-http://localhost:$FRONTEND_PORT,http://127.0.0.1:$FRONTEND_PORT}"

if [[ ! -d "$MOONDREAM_DATA_ROOT" ]]; then
  echo "Gallery data directory not found: $MOONDREAM_DATA_ROOT" >&2
  echo "Run scripts/convert_ppe_yolo.py first, or set MOONDREAM_DATA_ROOT in .env." >&2
  exit 1
fi

if [[ ! -d "$MOONDREAM_TOKENIZER" ]]; then
  echo "Tokenizer directory not found: $MOONDREAM_TOKENIZER" >&2
  echo "Place moondream_starmie_v1/tokenizer.json under artifacts, or set MOONDREAM_TOKENIZER in .env." >&2
  exit 1
fi

if [[ -z "${MOONDREAM_CHECKPOINT:-}" || ! -f "$MOONDREAM_CHECKPOINT" ]]; then
  echo "Checkpoint not found." >&2
  echo "Set MOONDREAM_CHECKPOINT in .env or place a *_best.pt file under checkpoints/." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fastapi, uvicorn, torch, PIL, tokenizers, moondream_mini
PY
then
  echo "Python dependencies are missing. Run: pip install -e ." >&2
  exit 1
fi

if [[ ! -d "$WEB_DIR/node_modules" ]]; then
  echo "Frontend dependencies not installed. Running npm install..."
  (cd "$WEB_DIR" && npm install)
fi

echo "Starting backend on http://$BACKEND_HOST:$BACKEND_PORT ..."
(
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m uvicorn apps.api.app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) > "$ROOT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 2
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start. See $ROOT_DIR/backend.log" >&2
  exit 1
fi

echo "Starting frontend on http://127.0.0.1:$FRONTEND_PORT ..."
(
  cd "$WEB_DIR"
  npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) > "$ROOT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

cat <<EOF

Demo is running.

Backend:  http://$BACKEND_HOST:$BACKEND_PORT
Frontend: http://127.0.0.1:$FRONTEND_PORT

Logs:
- $ROOT_DIR/backend.log
- $ROOT_DIR/frontend.log

Press Ctrl+C to stop both services.
EOF

wait "$FRONTEND_PID"
