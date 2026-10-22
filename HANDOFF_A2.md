# Handoff to Assignment 2

Assignment 2 starts from git tag **`a1-final`**. Everything below is frozen at that tag.

## 1. Frozen files

| What | Path | Versioned by |
|---|---|---|
| Task suite (30 tasks, 6 families, gold answers computed in code) | `data/processed/tasks_revised.jsonl` | DVC (`.dvc` pointer) |
| Collected trajectories (154 runs, 10 models) | `data/experiments/multi_provider_run/trajectories.jsonl` | DVC (`data/experiments.dvc`) |
| Grader-v2 verdict per trajectory | `data/processed/regrade_v2.csv` | DVC pipeline |
| Augmented tasks (23 entity variants, 30 paraphrases, 96 fault-injected) | `data/augmented/tasks_augmented.jsonl` | DVC pipeline |
| Final human labels (85 trajectories) | `data/labels/labels_round1.csv`, `data/labels/labels_round2.csv` | DVC pipeline |
| Split manifests | `data/splits/{train,validation,test,shifted}.csv` | DVC pipeline |
| Gold test set | `data/splits/gold_test/` | DVC pipeline |
| Raw Label Studio exports, signed adjudications | `supporting_files/labels/`, `supporting_files/agreement/round*/adjudication_round*.csv` | git |
| MLflow traces & spans database (174 traces, 633 spans) | `mlruns/mlflow.db` (reproducible via `code/tracing/export_trajectories_to_mlflow.py`) | local SQLite store (`mlruns/` gitignored) |

**DVC output hashes at the end of Part E** (`dvc.lock`, commit `dee10ed`). The tagged commit's `dvc.lock` is authoritative.

| Output | md5 |
|---|---|
| `data/augmented/tasks_augmented.jsonl` | `131e0a00a2dd40092841b4f2664051f7` |
| `data/processed/regrade_v2.csv` | `973e22cd0c8f03217e33796bb47f3a7c` |
| `data/labels/labels_round1.csv` | `3892c16fc50413841b174e47963a19a2` |
| `data/labels/labels_round2.csv` | `dcde9a0a216ea74f98cc8bde4ad0ea8d` |
| `data/splits/` (9 files) | `aa909c8348827958130772ba497e93cb.dir` |

**DVC remote:** `localstore` = `../agentevalops-dvc-store`, a folder next to the repository on the machine that ran `scripts/setup_dvc.sh`. It is local, so other machines get the data from the submission's `data/` folder or from a copy of that store folder (README, option A).

## 2. Split definitions

Unit: **template family**. No family appears in more than one of train, validation and test. Assignment is deterministic, with no randomness. Definition and rationale: `data/splits/split_definition.json`.

| Split | Families | Contents |
|---|---|---|
| train | multi_hop_retrieval, compound_calculation, structured_aggregation | 15 tasks + 30 variants; 78 trajectories (46 labelled) |
| validation | cross_tool_chaining | 5 tasks + 8 variants; 32 trajectories (15 labelled) |
| **test** | file_sla_auditing, ambiguous_unanswerable | 10 tasks + 15 variants; 44 trajectories (24 labelled) |
| **shifted** | 96 fault-injected tasks from all answerable tool-using tasks | Evaluation only; the `parent_split` column gives each task's original split |

The fault types in the shifted split are:
- transient tool failure (first call fails);
- information outage (`retrieve`, `read_file` and `query_records` all fail, so the expected behaviour is to abstain);
- calculator outage (gold answer kept);
- slow response (+2000 ms);
- changed output format.

For a clean drift comparison, use the shifted rows with `parent_split = test` against the test split.

**Leakage checks:** `python -m code.splits.make_splits --check` runs 10 checks (`supporting_files/splits/leakage_report.json`).

## 3. Gold test set (do not tune on it)

| File | SHA-256 |
|---|---|
| `data/splits/gold_test/gold_tasks.jsonl` | `f4b15d31aafd6ce4eb76ceaaca7c630cc7c4c08c72e93ca1bcec94b53c20d994` |
| `data/splits/gold_test/gold_labels.csv` | `72e8f84fa8363e0f9a0da2dfc408fd3cee454f7befed875e9dba11cbd5ef78bc` |

The gold set contains the test-split tasks (10 base tasks and 15 verified variants) and the human labels of the 24 labelled test-split trajectories.

**Rule:** never use the gold set for prompt design, configuration search, threshold tuning or model selection. Verify it before every use:

```bash
cd data/splits/gold_test && sha256sum -c CHECKSUMS.sha256
```

## 4. Label schema

| Field | Values | Meaning |
|---|---|---|
| `outcome` | `success`, `failure`, `error` | Is the submitted answer right (format-insensitive)? `error` = infrastructure crash only (guidelines v2) |
| `primary_failure_category` | `no_failure`, `wrong_tool`, `malformed_arguments`, `hallucinated_result`, `loop_or_repetition`, `premature_stop`, `correct_process_with_wrong_answer` | The earliest root cause; `no_failure` if and only if success |
| `first_error_step` | integer ≥ 0, or empty | First step that departs from a valid solution path; empty for success |
| `label_source` | `agreed`, `adjudicated` | Annotators agreed, or the label was decided by ADJ1 |
| `round`, `guideline_version` | `round1`/`v1`, `round2`/`v2` | Labelling round and the guideline it followed |

Labels were filled in this order: outcome, then category, then step (`LABELLING_PROTOCOL.md` §4).

**Grading:** use `code/tasks/grader_v2.py`. Grader v1 rejects correctly formatted answers like `$25,920`. The trajectories' stored `outcome` field is grader v1's verdict.

## 5. Known limitations and open items (read before Assignment 2)

1. **Provenance metadata (resolved; owner: Vedant).** `data/metadata.csv` provides complete provenance across all 154 collected runs (including the 52 runs logged as `error`) with authentic microsecond-level timestamps recovered from the collection queue database (`work_queue.sqlite3`), all 30 base benchmark tasks, and all 149 augmented tasks (381 records total).
2. **Task execution coverage & reserve.** The 154 collected trajectories cover 20 tasks with 6–10 runs each across 10 models. The remaining 10 tasks in the 30-task suite have no trajectories in the frozen A1 dataset, providing a pristine, un-overfitted reserve of benchmark tasks ready for Assignment 2 agent runs.
3. **No agent runs on augmented or fault-injected tasks.** `code/augmentation/run_augmented.py` is ready but was not run, because no API keys were used in Assignment 1. Assignment 2's drift experiments need to run it.
4. **Failure-category labels are noisy.** Category κ on items both annotators called failure is 0.21 in round 2, and the adjudicator is annotator A2.
5. **Guideline gaps** (R1/R3 precedence, step-limit wandering, empty abstentions) are listed as v3 recommendations in `docs/annotation_findings.md` §5.
6. **The `calculate` tool returns raw binary floats** (for example `13774.999999999998`).
7. **Telemetry decoupling (runner vs trajectory worker).** Synchronous inline MLflow tracing is integrated in `code/experiments/runner.py`. The multi-provider batch collector (`code/experiments/trajectory_worker.py`) deliberately wrote to raw JSONL and an SQLite work queue to prevent SQLite concurrency lock contention across concurrent workers and isolate external API backoffs. The bridge script `code/tracing/export_trajectories_to_mlflow.py` ingests all processed trajectories into `mlruns/mlflow.db` for inspection with `mlflow ui`.

## 6. Freeze procedure (run once, at the very end)

```bash
python -m pytest                                   # all pass
dvc repro && dvc status                            # "Data and pipelines are up to date."
python -m code.splits.make_splits --check          # 10 x [PASS]
(cd data/splits/gold_test && sha256sum -c CHECKSUMS.sha256)
git status --short                                 # empty
dvc push
git tag -a a1-final -m "Assignment 1 final: data, labels, splits frozen"
git push origin a1-final
git rev-parse a1-final^{commit}                    # record this hash in the report appendix
```

**Never move or re-create `a1-final` after it has been pushed.**
