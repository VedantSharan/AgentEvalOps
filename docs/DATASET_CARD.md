# Dataset Card: AgentEvalOps Assignment-1 Dataset (tag `a1-final`)

All counts are computed by `code/reporting/dataset_stats.py` and stored in `supporting_files/dataset_stats.json`. Splits are described in `data/splits/split_definition.json`.

## Purpose and intended use

The dataset contains benchmark tasks and execution traces of tool-using LLM agents, with human failure labels. It is the fixed input for Assignment 2 of the course project: configuration search, regression testing and drift monitoring of a single function-calling agent.

**Intended uses:**
- Evaluating agent configurations on task success.
- Studying failure modes.
- Testing label-free monitors and failure detectors.
- Drift experiments on the fault-injected (shifted) split.

**Not intended for:**
- Training general-purpose models.
- Ranking the hosted models.
- Any claim about real users. All entities are fictitious.

## Sources and collection

| Component | How it was made | Size |
|---|---|---|
| Task suite | Hand-written templates expanded in code; gold answers computed by solver functions (`code/tasks/revised_generator.py`, `revised_fixtures.py`) | 30 tasks: 6 families × 5 |
| Tool environment | Typed tools (`retrieve`, `calculate`, `query_records`, `read_file`, `submit_answer`) over a synthetic knowledge base, virtual files and an employee table | 23 benchmark facts (27 in the runtime store, including 4 legacy facts), 16 benchmark virtual files (42 file names visible at runtime, including legacy and adversarial distractor files), 5 employees |
| Trajectories | Plain-Python function-calling agent (`code/experiments/agent_executor.py`) run on 21 Oct 2026 through 5 hosted free-tier APIs, 10 models, temperature 0, at most 5 LLM turns | 154 runs |
| Human labels | Two annotators in Label Studio, two rounds, with adjudication (`LABELLING_PROTOCOL.md`) | 85 trajectories |
| Augmented tasks | Entity/parameter variants, rule-based paraphrases, fault injection (`code/augmentation/`) | 149 tasks |
| MLflow traces | Hierarchical execution spans (root, LLM turns, tool calls, grader) captured via inline tracer (`code/experiments/runner.py`) and offline ingestion (`code/tracing/export_trajectories_to_mlflow.py`) | 174 traces, 633 spans in `mlruns/mlflow.db` |

**Licence and consent:**
- All tasks, entities, files and records are synthetic and written by the team; there is no personal data.
- Model outputs were obtained under each provider's free-tier terms (`docs/DATA_COLLECTION.md`).
- Annotators are team members, identified only as A1, A2 and ADJ1 in derived files.

## Size and structure

**Tasks:**
- 30 tasks: 25 answerable and 5 unanswerable or ambiguous.
- Evaluation rules: 12 `exact_match`, 13 `numeric_tolerance`, 5 `unanswerable`.

**Trajectories:**
- 154 runs over 20 of the 30 tasks, with 6–10 runs per task. 10 tasks were never run (see "Known limitations").
- 1–13 steps per run (median 4).
- 42 runs ended at the step limit without a submitted answer.

**Logged outcome (grader v1):** 48 success, 54 failure, 52 error.

**Answer correctness (grader v2, format-tolerant):** 87 correct, 67 wrong. Grader v1 rejected 39 correct answers because of formatting such as `$`, thousands separators, units or full sentences. Grader v2 is the recommended grader.

**Human labels:** 85 trajectories (45 in round 1, 40 in round 2), with 58 agreed labels and 27 adjudicated.

| Outcome | Count |
|---|---|
| success | 52 |
| failure | 33 |

| Primary failure category | Count |
|---|---|
| correct_process_with_wrong_answer | 11 |
| malformed_arguments | 10 |
| premature_stop | 4 |
| loop_or_repetition | 4 |
| hallucinated_result | 3 |
| wrong_tool | 1 |

Each label also has a `first_error_step`.

**Trajectories per model:**

| Model | Runs |
|---|---|
| cohere:command-r7b-12-2024 | 20 |
| groq:gpt-oss-20b | 20 |
| nvidia:nemotron-3.5-lightning-30b-a3b | 20 |
| openrouter:minimax-m3.1-flash | 20 |
| cohere:command-a-03-2025 | 19 |
| groq:qwen3.8-27b | 18 |
| gemini-3.5-flash-lite | 17 |
| groq:gpt-oss-120b | 13 |
| openrouter:nemotron-3.5-lightning | 5 |
| gemini-3.5-flash | 2 |

**Trace Telemetry (MLflow):**
- 174 traces and 633 spans stored in `mlruns/mlflow.db`, viewable via `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Experiments:**
  - `AgentEvalOps-Benchmark`: 72 traces captured inline by `code/experiments/runner.py`.
  - `AgentEvalOps-MultiProvider`: 102 traces ingested from recorded batch runs via `code/tracing/export_trajectories_to_mlflow.py`.
- **Span hierarchy:** Root agent span (configuration, outcome, total latency), LLM turn spans (tokens, prompt, completion, latency), Tool invocation spans (typed arguments, outputs, errors), and Grader verification spans.

## Splits (unit: template family; no family is in two splits)

| Split | Families | Tasks (+ variants) | Trajectories | Human-labelled (failures) |
|---|---|---|---|---|
| train | multi_hop_retrieval, compound_calculation, structured_aggregation | 15 (+30) | 78 | 46 (15) |
| validation | cross_tool_chaining | 5 (+8) | 32 | 15 (10) |
| test (**gold**, frozen) | file_sla_auditing, ambiguous_unanswerable | 10 (+15) | 44 | 24 (8) |
| shifted | 96 fault-injected tasks: transient failure, information or calculator outage, +2 s latency, changed output format | 96 | — | — |

Ten automatic leakage and completeness checks pass (`supporting_files/splits/leakage_report.json`). Gold-set SHA-256 values are in `data/splits/gold_test/CHECKSUMS.sha256`.

## Labelling and agreement (summary)

| | Round 1 (guideline v1) | Round 2 (guideline v2) |
|---|---|---|
| Outcome κ | 0.85 | 0.95 |
| First-error step κ | 0.65 | 0.87 |
| Failure category κ on items both called failure | 0.05 | 0.21 |

Category labels are the least reliable part of the dataset. See `docs/annotation_findings.md`.

## Known biases and limitations

- **Coverage.** 10 of 30 tasks were never executed, including 3 of the 5 `file_sla_auditing` (test) tasks. Test trajectories therefore cover only T5_audit_02, T5_audit_03 and the unanswerable family.
- **Size.** 154 trajectories, below the "few hundred" the brief expects. The augmented and fault-injected tasks have **no** agent runs (the runner exists but was not run, because no API keys were used).
- **Model mix.** The data comes from free-tier hosted models, with uneven run counts per model (2 to 20). Provider-side model versions can change without notice.
- **Sampling bias in the labels.** Round 1 excluded runs logged as `error`. Round 2 deliberately over-sampled failures. Neither labelled set is a random sample of all runs.
- **Label reliability.** Category agreement is low. The adjudicator (ADJ1) is one of the annotators (A2), and the final labels may lean toward A2's reading. Round-1 annotation was very fast (median about 5 s per item).
- **Guideline gap.** Three runs with an empty answer on unanswerable tasks are labelled `success` by agreement; grader v2 says failure.
- **Provenance metadata.** `data/metadata.csv` provides complete provenance across all 154 collected runs (including the 52 error runs) with authentic microsecond-level execution timestamps recovered from the collection queue database (`work_queue.sqlite3`), all 30 base benchmark tasks, and all 149 augmented tasks (381 records total).
- **Synthetic environment.** The knowledge base is small, and `retrieve` matches only when every query word appears in a fact. Long natural-language queries return nothing, which shapes the observed failure modes.
- **Tool artefact.** `calculate` returns raw binary floats (for example `13774.999999999998`).
- **Telemetry decoupling (runner vs trajectory worker).** Synchronous inline tracing is provided in `runner.py` for benchmark runs. Batch collection via `trajectory_worker.py` logged to append-only JSONL files and an SQLite queue to prevent database lock contention across concurrent workers and isolate external API backoffs. Offline ingestion via `code/tracing/export_trajectories_to_mlflow.py` maps all recorded telemetry to MLflow traces without data loss.
