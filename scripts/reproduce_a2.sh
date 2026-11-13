#!/usr/bin/env bash
# Reproduce every Assignment 2 result (Parts A, B, C, E) in order.
#
# Agent runs are cached in data/a2/runs/ (DVC). With the cache restored, this
# recomputes every metric, plot and report from the stored trajectories and only
# calls the model for runs that are missing. Set A2_FRESH=1 to move the cache
# aside and rerun every agent call on the CPU server (several hours).
set -euo pipefail
export AGENTEVALOPS_OLLAMA_URL="${AGENTEVALOPS_OLLAMA_URL:-http://127.0.0.1:11435}"
export MLFLOW_DISABLE_AGENT_HINT=1
if [[ "${A2_FRESH:-0}" == "1" && -d data/a2/runs ]]; then
  mv data/a2/runs "data/a2/runs.bak.$(date +%s)"
fi
bash scripts/run_part_a.sh
bash scripts/run_part_b.sh
bash scripts/run_part_c.sh
bash scripts/run_part_e.sh
