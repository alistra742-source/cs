#!/bin/sh
set -eu

DEFAULT_MODEL="${DEFAULT_OLLAMA_MODEL:-ahmadwaqar/smolvlm2-256m-video:q8_0}"
# OLLAMA_MODEL is preferred. MODEL is supported as a Railway-friendly alias.
MODEL_NAME="${OLLAMA_MODEL:-${MODEL:-$DEFAULT_MODEL}}"
export OLLAMA_MODEL="$MODEL_NAME"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/ollama}"

if [ -z "${VISION_API_KEY:-}" ]; then
    echo '[vision-service] VISION_API_KEY is required; refusing to expose an unauthenticated model API.' >&2
    exit 1
fi

mkdir -p "$OLLAMA_MODELS"

echo "[vision-service] Starting Ollama locally..."
ollama serve &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT INT TERM

attempt=0
until curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 90 ]; then
        echo '[vision-service] Ollama did not become ready in 90 seconds.' >&2
        exit 1
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo '[vision-service] Ollama exited during startup.' >&2
        exit 1
    fi
    sleep 1
done

if ollama show "$MODEL_NAME" >/dev/null 2>&1; then
    echo "[vision-service] Configured model is already cached: $MODEL_NAME"
else
    echo "[vision-service] Installing configured model: $MODEL_NAME"
    # Railway currently returns EOF to Ollama's registry client while fetching
    # some public manifests. Install the same signed OCI blobs through plain
    # registry HTTP and verify every blob by size and SHA-256.
    python3 -u /service/pull_model.py "$MODEL_NAME"
fi

if ! ollama show "$MODEL_NAME" >/dev/null 2>&1; then
    echo "[vision-service] Model installation could not be verified: $MODEL_NAME" >&2
    exit 1
fi

echo "[vision-service] Model ready. Starting gateway on 0.0.0.0:${PORT:-8080}"
python3 -u /service/server.py
