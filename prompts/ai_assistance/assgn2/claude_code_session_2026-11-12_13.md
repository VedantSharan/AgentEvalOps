# Assignment 2: task prompts given to the AI assistant

| | |
|---|---|
| Assistant | Claude Code (CLI) |
| Model used | `claude-opus-5-5` |
| Used by | Shardul Junagade (23110297) |
| When | 12–13 November 2026, one working session |


## 1. Opening prompts (verbatim, in order)

### Opening prompt 1 (2026-11-12 15:25 UTC)

```text
Hey, so i am working on the project part of the course software tools and  techniques for AI. we were told to choose a project from @CS203_Projects.pdf and we chose the AgentEvalOps that should be the project 1. then we also submitted a proposal for the same, which u have to read from @proposal\ . the proposal was just a initial plan and we aare free to deviate from it for the project, but in case of a big deviation we should mention it in the report.

then read the code present in the agent eval ops folder and go through the entire code end-to-end and also read the submitted report that is present inside the assignment folder. there is also a zip in @CS203_A1_Project1_23110177_23110297_23110354.zip which contains the dvc also atleast that is what my group mate said who worked on the part1 of the project, extract the relevant files from the zip and finish the setup of the project so i can work on the next part of the assignment.

this is mainly so that u understand the entire project before starting working on the final part of the assignment.

please note that any change u make please keep it in unstaged, dont stage or commit anything, unless specifically i ask you to.
```

### Opening prompt 2 (2026-11-12 23:16 UTC)

```text
can this project be done on windows or should i use wsl instead?
```

### Opening prompt 3 (2026-11-12 23:22 UTC)

```text
can u save the entire context in a md file, its fine to keep it a long file to cover everything.
```

### Opening prompt 4 (2026-11-12 23:40 UTC)

```text
hey, please go through the @C:\Users\shardul\Desktop\sem-7\stt-ai\project\AgentEvalOps\assgns\txts\G2_feedback Assignment 1.md 
it contains the feedback shared by the TA for our assignment 1 which was based on @AgentEvalOps\assgns\txts\CS203_Assignment_2.md. read it all end to end and analyse the feedback. something might just contradicting the proposal and for that nothing needs to be done. ignore the unequal contribution n all, see what is really an issue and should be improved in the code, not the report before working on the next part of the assignment. give this in 1 liner bullet points so i can go through it once.
```

---

## 2. Standing instructions (applied to every task prompt)

```text
Context: CS203 Assignment 2, Project 1 (AgentEvalOps). Start from the A1 repository (tag a1-final). Read
assgns/txts/CS203_Assignment_2.md (and the PDF if the formatting is unclear), HANDOFF_A2.md and the A1
feedback before changing anything.

Rules for every task:
- Local inference only: a quantised open-weight model of at most 8B, run on CPU. Never use the GPU, not
  even for exploration, so every reported number can be reproduced by the CPU-only Docker image.
- Fixed splits: choose prompts, configurations and thresholds on validation only. Test and gold are for
  final evaluation. Never tune on gold; log every gold evaluation in TEST_SET_LOG.md.
- Version every data change with DVC and explain it in DATA_CHANGES.md. Never edit the gold test set.
- Write text files with LF line endings. The repository checks files out byte-for-byte, and DVC hashes
  depend on it.
- Add unit tests for every module and run the full suite before each commit.
- Commit each finished milestone as one complete, meaningful commit on branch a2, authored by me, with no
  AI co-author line. Do not push.
```

---

## 3. Task prompts

### Prompt 1: Line endings and DVC hashes

```text
The A1 feedback says the packaged repository shows ~145 files as modified on Linux (line endings), and
dvc.lock was hashed from CRLF copies on Windows. Make every platform check out exactly the committed bytes:
add a .gitattributes that turns off end-of-line conversion but keeps text diffs and merges (mark real
binaries such as png/pdf/docx/zip/sqlite as binary). Re-checkout the tree, re-record the DVC hashes
without re-running any stage, and confirm that `dvc status` is clean, all tests pass, the 10 split leakage
checks pass and both gold checksums match. No data file may change.
```

Produced: `.gitattributes`, `dvc.lock`.

### Prompt 2: Personal data in the Label Studio export

```text
The feedback found a personal Gmail address in the raw Label Studio export (supporting_files/labels/
annotator1_raw.json). Write code/annotation/anonymise_exports.py: replace every e-mail address in the
exports with a pseudonym for the annotator (A1@anonymised.invalid, ...) using a byte-level replacement so
nothing else in the file changes, plus a --check mode that exits 1 if any address is left. Show that the
labels, kappas and confusion matrices rebuilt from the anonymised export are identical, update the hash
recorded in the round-1 agreement summary, and record the change (original and new SHA-256) in a new
DATA_CHANGES.md. Also document there, without editing them, the three gold labels that mark an empty
answer as success.
```

Produced: `code/annotation/anonymise_exports.py`, `tests/test_anonymise_exports.py`, `DATA_CHANGES.md`, the anonymised export.

### Prompt 3: Validate every augmented task and add a CI sample mode

```text
In A1 the 12-check validation suite ran on only 53 of the 149 augmented tasks; the 96 fault-injected tasks
were skipped because they reuse their parent's prompt. Make the duplicate check aware of the tool
environment (the fault specification) so that two tasks are duplicates only if prompt and fault match, and
add an `augmented_all` dataset (all 149 tasks) to run_validation_evidence. Add --sample N, --seed and --out
so CI can validate a seeded sample without touching the DVC outputs, and write outputs with LF on every OS.
Re-run the validate stage and record the result in DATA_CHANGES.md.
```

Produced: changes to `code/validation/benchmark_validator.py` and `run_validation_evidence.py`, `tests/test_validation_augmented.py`, `supporting_files/validation/augmented_all.json`.

### Prompt 4: Local model client with offline and CPU-only guards

```text
Add local inference for the agent. Use Ollama with qwen2.5:3b (main model) and qwen2.5:1.5b (weaker model
for the search and the model-swap drift case), both Q4_K_M. Write code/experiments/local_llm.py with an
OllamaClient that has the same interface as the A1 ProviderClient but calls Ollama's native /api/chat, so
seed, num_ctx and num_thread can be pinned (seed 203, 4096, 8). Guards: with AGENTEVALOPS_OFFLINE=1 refuse
any non-loopback URL; after the first call per model, ask the server which models are in GPU memory and
raise GPUInUse if any are. Add scripts/ollama_cpu.sh that starts the pinned ollama/ollama:0.40.8 image by
digest on CPU only (no --gpus, one request at a time, no cloud calls) and pulls both models. Record the
image digest, model digests and fixed inference options in MODELS.md. Test the client with a fake server.
```

Produced: `code/experiments/local_llm.py`, `scripts/ollama_cpu.sh`, `MODELS.md`, `tests/test_local_llm.py`.

### Prompt 5: Versioned configurations and a prompt registry

```text
Make every compared system a versioned YAML configuration. Write code/agent/config.py with an AgentConfig
(strategy abstain/direct/tool_loop, model, prompt_version, tool_desc_version, temperature, max_steps,
max_tokens, retry_budget, error_feedback, loop_guard, step_guard, single_tool_call, exemplars) whose hash covers the fields
and the SHA-256 of every prompt file it uses. Export the A1 system instruction and tool schemas verbatim as
prompts/agent/system_v1.txt and prompts/tools/v1.json, and write v2 versions with tool-selection guidance,
retrieval keyword advice and explicit abstention rules. Add configs for the trivial always-abstain baseline
(b0), a direct no-tools baseline (b1), the original A1 loop (b2) and the proposed agent. Fix the promotion
gate in configs/gate.yaml before any gold evaluation: shifted gain >= 0.01 on held-out families and a gold
drop of at most 0.04.
```

Produced: `code/agent/config.py`, `configs/{b0_abstain,b1_direct,b2_original,proposed,gate}.yaml`, `prompts/agent/`, `prompts/tools/`, `tests/test_agent_config.py`.

### Prompt 6: The configuration-driven agent loop with live tracing

```text
Write code/agent/loop.py: run_agent(task, cfg) runs any configuration and returns an AgentRun with the
A1-format trajectory plus per-turn LLM latency, tokens, tool calls, the grader v2 verdict, the A1 detector
flags and the MLflow trace id. With every feature off, the tool loop must behave like the A1 loop, except
that it never reads the task's expected_tools (a deployed agent cannot see them). Proposed-system features:
harness retries for transient tool failures (retry_budget), error-feedback hints, a loop guard that blocks
identical repeated calls, a step guard that warns before the last turn and forces a submission, one tool
call per turn, and exemplars in the system prompt. Step latency must include the LLM call so total latency
is end to end. Trace every run live to MLflow with agent, llm_turn_N, tool and grader spans
(code/tracing/agent_tracer.py). Also fix the A1 executor: grade with v2 and count LLM latency. Test every
feature with a scripted fake model, including a live trace written to a temporary MLflow store.
```

Produced: `code/agent/loop.py`, `code/tracing/agent_tracer.py`, changes to `code/experiments/agent_executor.py`, `tests/test_agent_loop.py`.

### Prompt 7: One evaluation script, metrics and the configuration registry

```text
Write one evaluation script that accepts a registered version and a split:
python -m code.evaluation.evaluate --version <alias|vN|configs/x.yaml> --split <validation|test|gold|
shifted|shifted_<parent>|test_augmented|test_base|train> [--repeats k] [--reason].
- Load items from the frozen A1 manifests; shifted items run inside the A1 FaultInjector.
- Run in a fixed order with seed 203 + repeat, and cache every run per configuration hash in
  data/a2/runs/ (DVC).
- Metrics: success with a task-clustered 95% bootstrap CI (5000 resamples, seed 2026), pass^k, steps,
  tokens, p50/p95 latency, detector rates, A1 failure categories, and slices by family, answerability,
  augmentation type and fault type. Paired McNemar and paired bootstrap tests between systems.
- Log every evaluation as an MLflow run with parameters, metrics, the DVC data hashes and the git commit.
- Gold: verify both frozen checksums first, require --reason, and append a row to TEST_SET_LOG.md.
Use no LLM judge. Instead, add a rule-based categoriser for the A1 failure taxonomy built on the A1 failure
detectors; fit it on train labels only and report its Cohen's kappa against the human labels on validation. Add code/registry/registry.py: versioned configurations with staging/production
aliases, promote and rollback, a bundled registry/ folder the container reads offline, mirrored to the
MLflow model registry as a pyfunc model.
```

Produced: `code/evaluation/{data,gold,metrics,categorise,evaluate}.py`, `code/registry/registry.py`, `code/agent/adhoc.py`, `TEST_SET_LOG.md`, `supporting_files/a2/categoriser_vs_human.json`, `tests/test_evaluation.py`.

### Prompt 8: Exemplar banks from training data, and system prompt v3

```text
The agent has no weights to train, so "training" means choosing prompt material from train-split data.
Write code/agent/exemplars.py to build, from train data only:
- train_base: one worked example per train family (the shortest A1 trajectory that grader v2 marks
  correct, shown with the gold answer);
- train_aug_al: train_base plus paraphrase-augmented examples and pitfall notes from the failures that
  the A1 active-learning score ranks highest.
The A1 trajectories reused here came from hosted models, and their exact prompts are already versioned in
prompts/model_prompts/collection_agent. Add configs/variants/ with the proposed agent using each bank, for
the Part A "with and without augmented and actively selected data" comparison. Inspect the proposed agent's validation failures and add system
prompt v3: never assume a rate or threshold, and pass plain numbers to calculate.
```

Produced: `code/agent/exemplars.py`, `prompts/exemplars/{train_base,train_aug_al}.json`, `prompts/agent/system_v3.txt`, `configs/variants/`, `tests/test_exemplars.py`.

### Prompt 9: Drift monitoring

```text
Write code/monitoring/:
- signals.py: per-request label-free signals (tool failures including harness retries, malformed calls,
  loops, steps, latency, tokens) and the labelled success metric.
- detectors.py: one-sided CUSUM and Page-Hinkley on values standardised against a reference window.
- replay.py: seeded traffic streams from cached runs (300 requests, a 50-request reference window, drift
  from request 100 with a 100-request linear ramp to shifted traffic). Choose each signal's detector
  parameters on validation streams only (at most 0.5 false alarms per 100 requests, at least 90%
  detection, then the shortest delay) and keep the two best signals. Evaluate on test traffic shifting to
  fault-injected test tasks, plus a model-swap scenario and a drift-free stream. Report detection delay,
  false alarms and the Spearman correlation of each rolling signal with rolling success, with plots, and
  write alerts.json for the retraining pipeline.
- monitor.py: a LiveMonitor the service uses with the chosen thresholds.
```

Produced: `code/monitoring/{signals,detectors,replay,monitor}.py`, `tests/test_monitoring.py`.

### Prompt 10: One-command retraining with conditions for promoting a model

```text
Write code/retrain/pipeline.py, run as `python -m code.retrain.pipeline --alert <alerts.json> --budget N`:
1. Take the pool of fault-injected tasks whose parent family is in the train split.
2. Rank the production version's runs on it with the A1 active-learning score and select the top N.
3. Reveal their labels from a pre-labelled pool (simulated annotator: grader v2 outcome, A1 category,
   fault type), stored in data/a2/retrain/.
4. Validate the selection with the A1 12-check suite plus a leakage check against every evaluation split.
5. Retune: build a new exemplar bank from the selected successes and failures, and score candidate
   configurations on shifted_validation.
6. Compare candidate and production on shifted_test (held-out families) and gold.
7. Apply configs/gate.yaml: promote only if both conditions hold, otherwise register the candidate and
   reject it. Append every decision to a gate log.
Test selection, validation, leakage rejection and bank building without a model.
```

Produced: `code/retrain/pipeline.py`, `tests/test_retrain.py`.

### Prompt 11: Part A runs

```text
Run Part A on the validation split, CPU only: b0, b1, b2 (the A1 function-calling loop, which used hosted
APIs in A1, rerun on the local model), the proposed agent, and the proposed agent with the train_base and
train_aug_al banks. Write code/reporting/part_a.py to produce supporting_files/a2/partA/
with success and 95% CIs, cost, latency, paired tests, configuration hashes, seeds, DVC data version,
hardware and compute time, each with its MLflow run ID. Register the baselines as v1-v3 with aliases
b0/b1/b2 and track the runs in data/a2 with DVC. Report the result as it is, even where a variant does not
help. Make the split leakage check ignore data/a2 (A2 outputs, not A1 samples).
```

Produced: `code/reporting/{part_a,hardware}.py`, `scripts/run_part_a.sh`, `supporting_files/a2/partA/`, `registry/` (v1-v3), `data/a2.dvc`.

### Prompt 12: REST service, UI and answer confidence

```text
Write code/service/:
- app.py: FastAPI app serving the registry's production version, with
  - POST /predict (answer, trace id, confidence, latency, every step and tool call with arguments and
    outputs, grading for bundled tasks, version and commit);
  - GET /health and GET /version (git commit, registry version, config hash, model digest);
  - /monitoring and /validation.
  Log each request with version, latency and confidence, but store the task only as a SHA-256 hash and
  length, and mask e-mail addresses and phone numbers.
- ui.py: a Gradio UI on the same port (/ui) with bundled examples, fault injection and model swap for
  drift demos, a live monitoring tab, a validation tab that runs a fixed gold subset next to the reported
  values, and an About tab. Use system fonts only, so it works without internet.
- confidence.py: a logistic model on label-free run features, trained on train-split runs, with
  calibration (Brier, ECE) checked on validation.
- validate.py and smoke.py: offline gold validation and the CI smoke request.
Test the API with the model-free baseline.
```

Produced: `code/service/{app,ui,confidence,validate,smoke}.py`, `tests/test_service.py`.

### Prompt 13: Correct the A1 tracing documentation

```text
The A1 docs cite code/experiments/runner.py, which does not exist, and claim LLM-turn and grader spans for
the A1 traces. Correct them: the inline tracer lives in the rule-based pilot runner code/agent/runner.py,
the A1 traces have only root and tool spans, and only the A2 loop traces LLM turns and grading live.
```

Produced: corrections in `docs/DATASET_CARD.md`, `docs/DATA_COLLECTION.md`, `ENVIRONMENT.md`, `HANDOFF_A2.md`.

### Prompt 14: Part B configuration search

```text
Run a systematic search on validation only with Optuna (code/search/run_search.py,
configs/search_space.yaml): random search and NSGA-II, 10 trials each, seed 203, over prompt version,
temperature, step limit, retry budget, model and exemplar bank, under a token and p95-latency budget.
NSGA-II optimises success, tokens and latency. Every trial is a full evaluation and an MLflow run.
Include the Part A hand-designed configurations as candidates, and add a second-stage grid (3B model,
temperature 0) because stage 1 never tried that region. Select by validation success, then fewest tokens,
then latency. Make the search-analysis plots (Pareto front of success against tokens, best-so-far,
parameter effects including model size), so the quality/latency/size trade-off can be justified. Register
the winner as staging, and promote it to production on validation evidence only (the registry must accept
an alias such as "staging" as the version to promote).
```

Produced: `code/search/{run_search,plots}.py`, `configs/search_space.yaml`, `configs/search/`, `configs/selected.yaml`, `scripts/run_part_b.sh`, `supporting_files/a2/search/`, registry v4-v5, `tests/test_search.py`.

### Prompt 15: Part C final evaluation

```text
Evaluate the registered baselines and the production version on validation, test, gold, the A1 augmented
test tasks, shifted_test and the full shifted split (the slow b2 loop on shifted_test only). Write
code/reporting/part_c.py to produce:
- comparison tables with paired tests of the final system against each baseline;
- slices with the weakest groups;
- an error sheet with every test error categorised, with evidence and a suggested fix (at least 20
  errors);
- a robustness comparison of base and augmented test tasks;
- reported_values.json for the container validation.
Read the traces and write a short error-analysis summary. Do not change anything based on test results.
```

Produced: `code/reporting/part_c.py`, `scripts/run_part_c.sh`, `supporting_files/a2/partC/`, `supporting_files/a2/eval/`, `supporting_files/a2/reported_values.json`.

### Prompt 16: Part E drift, retraining, rejection and rollback

```text
Run Part E:
1. Train the confidence model.
2. Run the replay study for production and use its alert to trigger the retraining pipeline twice, with
   label budgets 4 and 16.
3. Send an externally proposed change (production on the 1.5B model, to cut cost) through the same gate
   with --candidate, to show a rejection.
4. Test the rollback process with the registry and restore the promoted version.
Log every gold gate check, report correlations for all six signals, and summarise delays, false alarms
and gate decisions in supporting_files/a2/partE_summary.md.
```

Produced: `configs/{monitoring_thresholds,confidence_model}.json`, `configs/{retrain,drift,proposals}/`, `prompts/exemplars/retrain_*_bank.json`, the `--candidate` mode of `code/retrain/pipeline.py`, `scripts/run_part_e.sh`, `data/a2/retrain/`, `supporting_files/a2/{monitoring,retrain}/`, `supporting_files/a2/partE_summary.md`, registry v6-v8 and `history.jsonl`.

### Prompt 17: Evaluate the deployed version

```text
After retraining, production changed. Evaluate the deployed version on every split like the other
systems, write its Part C report to a separate folder (part_c.py --out) so the earlier analysis stays,
regenerate reported_values.json for it, and record any trade-off honestly, including splits where it is
worse.
```

Produced: `supporting_files/a2/partC_v7/`, `supporting_files/a2/eval/v7__*.json`, updated `supporting_files/a2/reported_values.json`.

### Prompt 18: Offline CPU Docker image and CI

```text
Write code/Dockerfile (the repository-root Dockerfile) for a self-contained linux/amd64 CPU image:
- Base images: pinned python:3.10-slim-trixie and ollama/ollama:0.40.8 digests.
- Copy only the CPU inference backend, install a fully pinned lock file
  (deploy/requirements-docker.lock), pull both models at build time and verify their digests.
- Include the git commit hash.
- Bundle the code, registry, prompts, configs and the gold data, via a committed copy in deploy/data that
  a test checks against the DVC workspace.
- Entrypoints: serve (API + UI on 7860, model preloaded), validate (full gold set from scratch, compared
  with reported_values.json within +/-0.04, exit 1 otherwise) and smoke.
- No outbound calls at run time: offline mode, OLLAMA_NO_CLOUD, telemetry off.
- The live monitor must ignore the first warm-up requests, whose latency includes loading the model.
Verify on this machine: validate with --network none (log it in TEST_SET_LOG.md), an offline smoke
request, and a CPU load test (p50/p95 latency, throughput, memory). Add .github/workflows/ci.yml: on every push, run the unit tests
that need no DVC data, a seeded A1 validation sample from deploy/ci_sample, an image build and an offline
smoke request.
```

Produced: `Dockerfile`, `.dockerignore`, `requirements-docker.txt`, `scripts/docker_entrypoint.sh`, `deploy/`, `.github/workflows/ci.yml`, `code/service/loadtest.py`, `supporting_files/a2/{docker_validation,loadtest}.json`, `tests/test_deploy.py`.

### Prompt 19: Documentation, evidence and reproducibility

```text
Document A2:
- README: reproduce every result in at most five commands (scripts/reproduce_a2.sh), plus Docker build,
  start and validate commands with expected output.
- DOCKER.md (image size, RAM, hardware, load/run/validate commands with expected output and tolerance) and
  MODEL_CARD.md (intended use, data, metrics with intervals, slices, limitations, risks).
- An A2 section in ENVIRONMENT.md (hardware, software, seeds), the A2 additions in DATA_CHANGES.md,
  pinned requirements and a fresh freeze.
- Export the loop's fixed messages into prompts/model_prompts/a2_agent.
- An evidence index linking every result to its file and MLflow run ID, an export of the MLflow runs and
  registry, and a test log without local paths.
```

Produced: `README.md`, `DOCKER.md`, `MODEL_CARD.md`, `ENVIRONMENT.md`, `DATA_CHANGES.md`, `requirements*.txt`, `scripts/reproduce_a2.sh`, `code/reporting/{evidence_index,export_mlflow}.py`, `supporting_files/a2/{evidence_index.csv,mlflow/,test_log.txt}`.

### Prompt 20: Tag the release and package the artefacts

```text
Fast-forward main to a2 and tag a2-final locally. Rebuild the image from code/Dockerfile at a2-final so the
baked commit matches the tag, check that /app/GIT_COMMIT equals the tagged commit, then package the
artefacts outside the repository: docker save + gzip to docker_image.tar.gz, its SHA-256, a DOCKER.md with
the checksum, image ID, commit and load/run/validate commands, MODELS.md, the registry, and the MLflow
export (runs, registry, traces). Load the archive back to confirm the image ID matches.
```

Produced: tag `a2-final`; outside the repository, the submission's `artefacts/` folder (`docker_image.tar.gz`, `docker_image.tar.gz.sha256`, `DOCKER.md`, `MODELS.md`, `registry/`, `mlflow/`).