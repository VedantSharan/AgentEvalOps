#!/usr/bin/env bash
# Part C: every registered baseline (aliases b0, b1, b2) and the final version
# (alias production) on validation, test, gold, shifted and the A1 augmented
# test tasks, then the comparison, slices, error sheet and robustness report.
set -euo pipefail
export AGENTEVALOPS_OLLAMA_URL="${AGENTEVALOPS_OLLAMA_URL:-http://127.0.0.1:11435}"
PY="${PY:-python}"
for ref in b0 b1 b2 production; do
  for split in validation test test_augmented shifted_test; do
    $PY -m code.evaluation.evaluate --version "$ref" --split "$split"
  done
  $PY -m code.evaluation.evaluate --version "$ref" --split gold --reason "Part C final evaluation of $ref"
done
# The full shifted split (96 tasks) for the final system and the cheap baselines;
# the slow original loop (b2) is compared on the held-out shifted_test families.
for ref in b0 b1 production; do
  $PY -m code.evaluation.evaluate --version "$ref" --split shifted
done
$PY -m code.reporting.part_c --final production
