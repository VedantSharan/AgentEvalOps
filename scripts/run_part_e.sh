#!/usr/bin/env bash
# Part E: confidence model, replayed drift study with thresholds chosen on
# validation, then the one-command retraining pipeline triggered by the alert.
set -euo pipefail
export AGENTEVALOPS_OLLAMA_URL="${AGENTEVALOPS_OLLAMA_URL:-http://127.0.0.1:11435}"
PY="${PY:-python}"
$PY -m code.service.confidence --version production
$PY -m code.monitoring.replay --version production
$PY -m code.retrain.pipeline --alert supporting_files/a2/monitoring/alerts.json --budget "${BUDGET:-16}"
