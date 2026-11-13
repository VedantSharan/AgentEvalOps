# Model card: AgentEvalOps agent, registry version v7 (production)

**What it is.** The deployed system is a configuration of a tool-using agent, not a trained network. A local quantised LLM (`qwen2.5:3b`) runs the function-calling loop in `code/agent/loop.py`, with versioned prompts, tool descriptions, an exemplar bank and guard settings. It was selected on validation in Part B (v5), then retuned twice by the drift-triggered retraining loop in Part E (v6, v7), passing the pre-registered gate each time.

| Item | Value |
|---|---|
| Registry version | v7 (alias `production`), configuration hash `da8a47d4bef7…`, MLflow model `agentevalops-agent` version 7 |
| Configuration | `registry/versions/v7.yaml` (source `configs/retrain/retrain_main_c0.yaml`) |
| Model | `qwen2.5:3b`: Qwen2.5-3B-Instruct, Q4_K_M GGUF from the Ollama library, digest `357c53fb…` (`MODELS.md`) |
| Prompt and tools | system prompt v3 (`prompts/agent/system_v3.txt`), tool descriptions v2 (`prompts/tools/v2.json`), exemplar bank `retrain_main_bank` (`prompts/exemplars/`) |
| Loop settings | temperature 0, at most 6 LLM turns, 512 tokens per turn, retry budget 3 for transient tool failures, error-feedback hints, loop guard, step guard, one tool call per turn |
| Inference | Ollama 0.40.8 on CPU only, `num_thread` 8, `num_ctx` 4096, seed 203 |
| Lineage | v5: Part B winner (`configs/proposed.yaml`) → v6: retrain with 4 labels → v7: retrain with 16 labels. A rollback drill moved production to v6 and back (`registry/history.jsonl`) |

## Intended use

- **What it does:** answers short business questions about a synthetic company (travel and bonus policies, employee records, consulting rates, service logs) by calling five local tools (`retrieve`, `calculate`, `query_records`, `read_file`, `submit_answer`). It answers "unanswerable" when the tools show the information does not exist or is ambiguous.
- **Who it is for:** the CS 203 course project. It demonstrates how a tool-using agent is evaluated, deployed and monitored as an ML system.
- **Out of scope:** real company data, open-domain questions, and any decision with consequences for a person. The tools hold only the synthetic Assignment 1 fixtures, so on any other question the agent can only abstain or guess.

## Data

| Split | Content | Use |
|---|---|---|
| train | 3 template families (multi-hop retrieval, compound calculation, structured aggregation): 15 base tasks + 30 variants, plus their A1 trajectories | exemplar banks, confidence model, retraining pool (their fault-injected versions) |
| validation | cross-tool chaining: 5 base tasks + 8 variants | every design choice: prompts, the search, drift thresholds, retraining candidates (their fault-injected versions) |
| test / gold | file and SLA auditing, ambiguous/unanswerable: 10 base tasks + 15 variants; the frozen gold file holds the same 25 tasks | final evaluation and gate checks only (`TEST_SET_LOG.md`) |
| shifted | 96 fault-injected tasks: transient failure, information outage, calculator outage, slow response, changed format | drift experiments; the test-family part (20 tasks) is the held-out gate set |

All tasks are synthetic, written by the team, with gold answers computed in code. Every answer is graded by grader v2 (`code/tasks/grader_v2.py`). See `DATA_CHANGES.md` for the A2 changes to the A1 data; the gold set is unchanged.

## Metrics and intervals (`supporting_files/a2/partC_v7/`)

Success rates have 95% bootstrap CIs over tasks. Decoding is greedy, so repeated runs give the same answer (pass^1 = success).

| Split (tasks) | v7 | B0 always abstain | B1 direct, no tools | B2 original A1 loop | MLflow run (v7) |
|---|---|---|---|---|---|
| validation (13) | 0.231 [0.000, 0.462] | 0.000 | 0.000 | 0.000 | `c269642e8f50…` |
| test (25) | 0.480 [0.280, 0.680] | 0.440 | 0.440 | 0.400 | `595b97a9f646…` |
| gold (25) | 0.480 [0.280, 0.680] | 0.440 | 0.440 | 0.400 | `13726561a087…` |
| augmented test (15) | 0.400 [0.133, 0.667] | 0.400 | 0.400 | 0.333 | `e2c9a2e740ba…` |
| shifted_test (20) | 0.350 [0.150, 0.550] | 0.250 | 0.250 | 0.150 | `0bc4f3ffa0b4…` |
| shifted (96) | 0.396 [0.302, 0.500] | 0.198 | 0.198 | not run | `3a705fd6b3e6…` |

**Paired tests (v7 vs baseline, same tasks):**
- **Full shifted split:** v7 beats B0 and B1 by +0.198 (McNemar p < 0.001).
- **Test and gold:** the differences from the baselines (+0.04 to +0.08) are not significant (p ≥ 0.45). With 25 tasks the CI half-width is about 0.2.

**Cost and latency.** These are per task on the i7-13700H CPU.

| Split | Steps | Tokens | p50 / p95 latency |
|---|---|---|---|
| test | 2.6 | 5.5k | 11.4 s / 56.2 s |
| shifted | 3.9 | 8.7k | 43.5 s p95 |

**Failure categories (test errors).** The categories are automatic A1 categories, which agree with human labels at κ = 0.53 on validation: correct process with wrong answer 10, premature stop 3.

**Confidence.** The confidence model has a validation Brier score of 0.155, against 0.237 for the base rate.

## Slices and weak groups

| Slice | v7 success |
|---|---|
| Test, ambiguous/unanswerable family | 0.909 (10 of 11) |
| Test, file and SLA auditing family | **0.143 (2 of 14)**, the weakest group |
| Shifted, information outage | 1.000 (19 of 19): the agent abstains when every data tool fails |
| Shifted, format change | **0.208** (5 of 24) |
| Shifted, slow response | 0.292 |
| Shifted, transient failure | 0.292 |
| Shifted, calculator outage | **0.000** (0 of 5) |

**File audits.** The agent often looks for the SLA threshold with `retrieve` instead of reading the second named file (`supporting_files/a2/partC/error_analysis_summary.md`).

**Robustness to A1 Part D augmentation.** Augmented test tasks (entity variants and paraphrases) score 0.400 against 0.600 on the 10 base test tasks.

## Limitations

- **Small evaluation sets.** Validation has 13 tasks and test/gold 25, so the 95% intervals are about ±0.2. Most test-set differences between systems are not significant.
- **The validation gain does not transfer cleanly.** The Part B winner (v5) scored 0.385 on validation but 0.400 on test, no better than abstaining. Retraining raised test/gold to 0.480 and shifted_test to 0.350, but lowered validation to 0.231. The pre-registered gate checks only shifted held-out tasks and gold, so it could not see the validation regression. A future gate should add a validation no-regression condition.
- **Narrow families.** Only one family is in validation, and the test families differ from every training family.
- **Small model.** A 3B model at Q4 chains tools inconsistently, stops early, and misses the second file in audits.
- **Drift coverage.** The one-sided detectors catch tool-failure drift (delay 38 requests) but not a swap to a weaker, faster model, which makes the agent do less work.
- **Proxy confidence.** Confidence is a logistic model on process features from 45 training runs: a rough signal, not a calibrated probability on new task types.
- **CPU reproducibility.** Answers are reproducible only with the same Ollama build, thread count and CPU kernel. On another CPU, a greedy decode can change; the stated validation tolerance is one gold task.

## Risks

- **Confident wrong answers.** The agent can state a wrong number with no visible doubt. Every answer is shown with its steps, tool outputs and confidence, so a user can check it.
- **Silent degradation.** Quality drops if tools fail or change format. The monitor watches tool failures and latency, and the retraining gate limits the gold-set drop to 0.04.
- **Personal data.** Requests are logged as a hash and length only, with e-mail addresses and phone numbers masked in logged answers. Request text is never stored.
- **Misuse as a general assistant.** The agent has no knowledge beyond its fixtures, so it should be used only for questions about the synthetic company.
