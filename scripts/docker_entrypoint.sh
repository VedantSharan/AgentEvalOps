#!/usr/bin/env bash
# Entry point of the AgentEvalOps image.
#   serve     start the model server and the API + UI on port 7860 (default)
#   validate  evaluate the deployed version on the full gold test set, print the
#             metrics, write /app/output/validation.json and compare with the
#             reported values (exit 1 if outside the stated tolerance)
#   smoke     start everything, send one /health and one /predict request, exit
#   <other>   run that command
set -euo pipefail
cd /app

start_ollama() {
  ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 120); do
    if ollama list > /dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "Ollama did not start:" >&2
  cat /tmp/ollama.log >&2
  exit 1
}

case "${1:-serve}" in
  serve)
    start_ollama
    # Load the production model into memory before taking requests, so the first
    # request is not slowed by the model load (which would look like latency drift).
    python -c "
import requests
from code.registry.registry import Registry
model = Registry(mlflow_sync=False).resolve('${AGENTEVALOPS_ALIAS:-production}')[1].model
requests.post('http://127.0.0.1:11434/api/generate', json={'model': model, 'keep_alive': -1}, timeout=600)
print('preloaded', model)
" || echo "model preload failed; continuing"
    exec uvicorn code.service.app:app --host 0.0.0.0 --port 7860
    ;;
  validate)
    start_ollama
    shift || true
    exec python -m code.service.validate "$@"
    ;;
  smoke)
    start_ollama
    AGENTEVALOPS_UI=0 uvicorn code.service.app:app --host 127.0.0.1 --port 7860 > /tmp/uvicorn.log 2>&1 &
    exec python -m code.service.smoke --url http://127.0.0.1:7860
    ;;
  *)
    exec "$@"
    ;;
esac
