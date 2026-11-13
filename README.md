# AgentEvalOps

A systematic evaluation, optimisation and monitoring framework for tool-using LLM agents. This is the CS 203 course project at IIT Gandhinagar (Project 1).

**Repository:** https://github.com/VedantSharan/AgentEvalOps
**Assignment 1 state:** git tag `a1-final` · **Assignment 2 state:** git tag `a2-final`

A plain-Python function-calling agent uses typed tools (`retrieve`, `calculate`, `query_records`, `read_file`, `submit_answer`) on a 30-task benchmark of 6 template families. Assignment 1 collected, labelled, validated and split the data. Assignment 2 runs the agent on a **local quantised model on CPU** (`qwen2.5:3b` through Ollama). It compares baselines with the proposed agent, searches configurations, evaluates on held-out families, deploys an offline Docker service with a UI, and monitors drift with a retraining gate.

## Assignment 2: reproduce the results (5 commands)

You need Python 3.10, Docker and git. From the unzipped submission, run inside `code/`:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # 1  pinned environment
cp -r ../data/. data/ && dvc commit -f -q                                           # 2  frozen A1 data + A2 agent runs (data/a2)
bash scripts/ollama_cpu.sh && export AGENTEVALOPS_OLLAMA_URL=http://127.0.0.1:11435  # 3  pinned Ollama server on CPU + both models
bash scripts/reproduce_a2.sh                                                        # 4  Parts A, B, C, E: evaluations, search, reports
python -m pytest && dvc status                                                      # 5  tests pass; data and pipeline up to date
```

**Expected output.**
- Step 4 prints one JSON summary per evaluation and ends with the retraining gate decision.
- Every number in `supporting_files/a2/` is recomputed from the cached runs in `data/a2/runs/`, so with the cache restored (step 2) it matches the report exactly.
- `A2_FRESH=1 bash scripts/reproduce_a2.sh` reruns every agent call on CPU instead (several hours). The answers are identical on the same CPU and Ollama build (see `MODELS.md`).
- Re-registering the same configurations is idempotent, but promotions are appended again to `registry/history.jsonl`.
- Token counts can differ by a few percent in fresh runs. Ollama counts only prompt tokens not served from its prefix cache, which depends on the previous request; answers do not change.

Each part can also be run on its own: `scripts/run_part_a.sh` (baselines and data comparison), `run_part_b.sh` (search), `run_part_c.sh` (final evaluation), `run_part_e.sh` (drift and retraining). The single evaluation entry point is

```bash
python -m code.evaluation.evaluate --version <alias|vN|configs/x.yaml> --split <validation|test|gold|shifted|shifted_test|test_augmented> [--repeats k] [--reason "..."]
```

A gold evaluation needs `--reason`. It verifies the frozen checksums first and appends a row to `TEST_SET_LOG.md`.

## Assignment 2: Docker service

Building, loading, starting, validating and the expected output are documented in **`DOCKER.md`**. In short:

```bash
docker build --platform linux/amd64 -t agentevalops:a2 .                 # from code/ (the repository root)
docker run --rm -p 7860:7860 agentevalops:a2                             # UI: http://localhost:7860/ui  API: /predict /health /version
docker run --rm --network none agentevalops:a2 validate                  # full gold set offline; prints metrics, writes /app/output/validation.json
```

## Assignment 2: where things are

| What | Where |
|---|---|
| Agent loop (baselines and proposed system) | `code/agent/loop.py`, configurations `configs/*.yaml`, prompts `prompts/agent/`, `prompts/tools/`, `prompts/exemplars/` |
| Local inference with offline and CPU-only guards | `code/experiments/local_llm.py`, `scripts/ollama_cpu.sh`, `MODELS.md` |
| Evaluation, metrics, CIs, paired tests, failure categories | `code/evaluation/` |
| Registry with staging/production aliases (bundled + MLflow) | `code/registry/registry.py`, `registry/` |
| Configuration search | `code/search/`, `configs/search_space.yaml` |
| Monitoring, drift detection, replay | `code/monitoring/`, `configs/monitoring_thresholds.json` |
| Retraining pipeline and gate | `code/retrain/pipeline.py`, `configs/gate.yaml` |
| Service, UI, confidence, validation, load test | `code/service/`, `Dockerfile`, `scripts/docker_entrypoint.sh` |
| CI | `.github/workflows/ci.yml` |
| Evidence behind every result | `supporting_files/a2/` (each result links to an MLflow run ID) |
| Model card, data changes, gold-set log, Docker | `MODEL_CARD.md`, `DATA_CHANGES.md`, `TEST_SET_LOG.md`, `DOCKER.md`, `MODELS.md` |
| Evidence index (result -> file -> MLflow run) | `supporting_files/a2/evidence_index.csv` (`python -m code.reporting.evidence_index`) |
| Part summaries | `supporting_files/a2/partA/table.md`, `search/summary.json`, `partC/comparison.md`, `partC/error_analysis_summary.md`, `partC_v7/`, `partE_summary.md` |

**MLflow:** every evaluation is a run in experiment `AgentEvalOps-A2-eval`, every agent call is a trace (`AgentEvalOps-A2-traces`: agent, LLM-turn, tool and grader spans), and the registered model is `agentevalops-agent`. To view them, run `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`; the database is exported to `artefacts/` in the submission.

## Assignment 1: rebuild the data (at most 5 commands)

You need Python 3.10 (tested: 3.10.12 on Ubuntu 22.04) and git.

**A. From the unzipped submission** (folders `code/`, `data/`, `supporting_files/`, `report/`). Run these from inside `code/`:

```bash
python3 -m venv .venv && . .venv/bin/activate     # 1  isolated environment
pip install -r requirements.txt                    # 2  pinned packages, includes DVC
cp -r ../data/. data/ && dvc commit -f -q          # 3  frozen data into the DVC workspace; re-hashes it against the .dvc pointers
dvc repro --force                                  # 4  rebuild every derived file from the raw inputs (14 stages)
dvc status && git status --short                   # 5  "up to date" and no changed files = rebuild matches a1-final
```

**B. From a git clone with access to the team's DVC remote:**

```bash
git clone https://github.com/VedantSharan/AgentEvalOps.git && cd AgentEvalOps && git checkout a1-final
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
dvc pull                                           # needs the remote ../agentevalops-dvc-store (see HANDOFF_A2.md)
dvc repro --force
dvc status && git status --short
```

**Expected output:**
- Step 4 prints `Running stage …` 14 times and ends without errors. The `splits` stage prints 10 `[PASS]` lines, and `validate` reports that only the corrupted fixtures fail.
- Step 5 prints `Data and pipelines are up to date.` and no modified files.
- One exception: PNG plots can differ byte-wise under a different matplotlib build.
- If any frozen input differed from `a1-final`, step 3 rewrites its `.dvc` pointer, and step 5 shows it as modified (for example `M data/raw.dvc`).

**Run the tests:** `python -m pytest`. Every test should pass (the A2 count is in `supporting_files/a2/test_log.txt`). pytest is configured with `-p no:debugging`, because the package name `code` shadows a standard-library module.

## What the pipeline produces (`dvc.yaml`)

| Stage | Command | Main outputs |
|---|---|---|
| validate | `python -m code.validation.run_validation_evidence` | `supporting_files/validation/` (12-check suite on raw, processed, corrupted and augmented data) |
| regrade | `python -m code.tasks.grader_v2` | `data/processed/regrade_v2.csv` (format-tolerant grader v2 vs logged v1) |
| augment | `python -m code.augmentation.build_augmented` | `data/augmented/tasks_augmented.jsonl` (149 tasks) and label checks |
| agreement@round1/2 | `python -m code.annotation.agreement …` | `supporting_files/agreement/round*/` (kappa, confusion matrices, disagreements) |
| labels@round1/2 | `python -m code.annotation.build_labels …` | `data/labels/labels_round*.csv` (final labels; needs signed adjudications) |
| human_vs_grader@round1/2 | `python -m code.annotation.compare_human_grader …` | `supporting_files/agreement/round*/human_vs_grader*` |
| compare_rounds | `python -m code.annotation.compare_rounds` | `supporting_files/agreement/pre_post_comparison.*` |
| detectors | `python -m code.annotation.detectors` | `supporting_files/detectors/` (loop and malformed-call detectors vs humans) |
| active_learning | `python -m code.active_learning.al_vs_random …` | `data/active_learning/`, `supporting_files/active_learning/` (plot and curves) |
| splits | `python -m code.splits.make_splits` | `data/splits/` (train/validation/test/shifted manifests, gold test set, exclusions) |
| dataset_stats | `python -m code.reporting.dataset_stats` | `supporting_files/dataset_stats.json` (every number in the dataset card) |

**Raw and frozen inputs** are tracked with `dvc add` (`data/*.dvc`): tasks, the 154 collected trajectories, pilot and corrupted data, metadata, and the Label Studio import files. **Human inputs** stay in git, unchanged: the raw Label Studio exports (`supporting_files/labels/`), the signed adjudication tables, and the paraphrase spot check.

Collecting new trajectories needs API keys in `.env` and is **not** part of the rebuild. The collector is `python -m code.experiments.trajectory_worker`.

## MLflow tracing (Assignment 1 data)

The A1 traces were not recorded live. `python -m code.tracing.export_trajectories_to_mlflow` ingests the 154 collected trajectories into MLflow afterwards, with root and tool spans. The rule-based pilot runner (`code/agent/runner.py`) traces inline. Since A2, the agent loop traces every run live, including LLM-turn and grader spans (see above).

## Repository layout

```
code/            agent, tools, tasks, grader v1/v2, tracing, validation, annotation, augmentation,
                 active_learning, splits, reporting
data/            DVC-tracked data (see above); data/splits/ holds the Assignment-2 splits
docs/            guidelines v1/v2, annotation findings, dataset card, data collection, validation report
prompts/         every model prompt used by the code, and the AI-assistant prompts
supporting_files/ evidence behind every number in the report
tests/           pytest suite
dvc.yaml, dvc.lock, requirements.txt, requirements_freeze.txt
```

## Documentation

- `docs/DATASET_CARD.md`: sources, size, class balance, splits, known biases.
- `ENVIRONMENT.md`: software versions, models, seeds.
- `LABELLING_PROTOCOL.md`: who labelled what, the order of levels and items, what annotators saw, adjudication, timeline.
- `HANDOFF_A2.md`: frozen files, split definitions, label schema, gold-set checksum, limitations.
- `docs/annotation_findings.md`: agreement before and after the guideline revision, disagreements, detectors.
- `docs/annotation_guidelines_v1.md`, `docs/annotation_guidelines_v2.md`: annotation guidelines.
- `docs/DATA_COLLECTION.md`, `docs/VALIDATION_REPORT.md`: Parts A and B.
- `report/report.pdf`: the Assignment 1 report.
