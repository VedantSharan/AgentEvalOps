#!/usr/bin/env bash
# One-time migration of data/ from git to DVC (Assignment 1, Part E).
# Run ONCE from the repository root, inside the activated venv, on a clean git tree:
#     bash scripts/setup_dvc.sh
# What it does:
#   1. dvc init (keeps the existing .dvc/ folder if present)
#   2. stops git from tracking data files (they stay on disk) and puts raw/frozen inputs under DVC
#   3. runs the pipeline in dvc.yaml (dvc repro), which re-creates every derived file and records hashes in dvc.lock
#   4. configures a local DVC remote OUTSIDE the repo (../agentevalops-dvc-store) and pushes the data to it
# It does NOT commit; review `git status` and commit yourself (commands are printed at the end).
set -euo pipefail

if [ -n "$(git status --porcelain)" ]; then
  echo "Git tree is not clean. Commit or stash your changes first." >&2
  exit 1
fi

RAW=(data/raw data/experiments data/adversarial data/corrupted data/metadata.csv
     data/processed/tasks_revised.jsonl data/processed/trajectories_processed.jsonl
     data/labels/stratified_sample_45.json data/labels/stratified_sample_45.jsonl
     data/labels/round2_sample.json data/labels/round2_manifest.csv)
OUTS=(data/processed/regrade_v2.csv data/augmented data/labels/labels_round1.csv
      data/labels/labels_round2.csv data/active_learning data/splits)

[ -f .dvc/config ] || dvc init -f
git rm -r -q --cached --ignore-unmatch "${RAW[@]}" "${OUTS[@]}"
dvc add "${RAW[@]}"
dvc repro
dvc remote add -d -f localstore ../agentevalops-dvc-store
dvc push
dvc status

echo
echo "DVC set up. Now review and commit:"
echo "  git add -A && git status --short | head -50"
echo "  git commit -m 'Part E: version data with DVC (dvc add raw inputs, dvc.yaml pipeline, local remote)'"
