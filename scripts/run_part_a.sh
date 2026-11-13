#!/usr/bin/env bash
# Part A: baselines and the proposed agent on the validation split, plus the
# data comparison (proposed agent with no exemplars, train_base, train_aug_al).
# Needs the CPU Ollama server: bash scripts/ollama_cpu.sh
set -euo pipefail
export AGENTEVALOPS_OLLAMA_URL="${AGENTEVALOPS_OLLAMA_URL:-http://127.0.0.1:11435}"
PY="${PY:-python}"

$PY -m code.agent.exemplars
for cfg in configs/b0_abstain.yaml configs/b1_direct.yaml configs/b2_original.yaml configs/proposed.yaml \
           configs/variants/proposed_train_base.yaml configs/variants/proposed_train_aug_al.yaml; do
  $PY -m code.evaluation.evaluate --version "$cfg" --split validation --repeats 1
done
$PY -m code.reporting.part_a
