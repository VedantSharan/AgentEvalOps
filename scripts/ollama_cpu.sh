#!/usr/bin/env bash
# Start the pinned Ollama server on CPU only (no --gpus) and pull the models.
#
# Every A2 experiment calls this server, and the Docker image copies the same
# Ollama binary from the same image digest, so reported runs and
# `docker run ... validate` use identical inference code. See MODELS.md.
#
# Usage: bash scripts/ollama_cpu.sh            # start (idempotent) and pull models
#        export OLLAMA_HOST=http://127.0.0.1:11435   # before running experiments
#        bash scripts/ollama_cpu.sh stop
set -euo pipefail

IMAGE="ollama/ollama:0.40.8@sha256:2b8eecdb85aa6ea67c381c8ecffec04b2658496cf2821a8f88b0485bdff198e8"
NAME="agentevalops-ollama"
# Host port 11435, so the server never collides with a desktop Ollama on 11434.
PORT="${AGENTEVALOPS_OLLAMA_PORT:-11435}"
VOLUME="agentevalops-ollama-models"
MODELS=("qwen2.5:3b" "qwen2.5:1.5b")

if [[ "${1:-}" == "stop" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # OLLAMA_NUM_PARALLEL=1: batching requests together changes the numerics.
  docker run -d --name "$NAME" \
    -p 127.0.0.1:"$PORT":11434 \
    -v "$VOLUME":/root/.ollama \
    -e OLLAMA_NUM_PARALLEL=1 \
    -e OLLAMA_MAX_LOADED_MODELS=2 \
    -e OLLAMA_KEEP_ALIVE=-1 \
    "$IMAGE" >/dev/null
  for _ in $(seq 1 60); do
    docker exec "$NAME" ollama list >/dev/null 2>&1 && break
    sleep 1
  done
fi

for m in "${MODELS[@]}"; do
  docker exec "$NAME" ollama pull "$m"
done
docker exec "$NAME" ollama list
