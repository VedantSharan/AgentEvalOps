# AgentEvalOps

A systematic evaluation, optimisation and monitoring framework for tool-using LLM agents. This is the CS 203 course project at IIT Gandhinagar (Project 1).

**Repository:** https://github.com/VedantSharan/AgentEvalOps
**Assignment 1 state:** git tag `a1-final`

A plain-Python function-calling agent uses typed tools (`retrieve`, `calculate`, `query_records`, `read_file`, `submit_answer`) on a 30-task benchmark of 6 template families. Every run is a trajectory. Trajectories are labelled by humans, augmented, versioned with DVC and split by template family for Assignment 2.

## Rebuild the data (at most 5 commands)

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

**Run the tests:** `python -m pytest`. The expected result is 374+ passed. pytest is configured with `-p no:debugging`, because the package name `code` shadows a standard-library module.

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
