#!/usr/bin/env bash
# Part B: configuration search on the validation split (random vs NSGA-II),
# search-analysis plots, and registration of the winner as "staging".
set -euo pipefail
export AGENTEVALOPS_OLLAMA_URL="${AGENTEVALOPS_OLLAMA_URL:-http://127.0.0.1:11435}"
PY="${PY:-python}"
$PY -m code.search.run_search
$PY -m code.search.plots
# The search winner (staging) becomes production on validation evidence only;
# test and gold are evaluated afterwards (Part C) and never used to choose.
$PY -m code.registry.registry promote staging production   --reason "Part B: best feasible validation success; chosen on validation only"
