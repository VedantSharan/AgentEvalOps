# Part E: monitoring, drift and retraining

All numbers come from the files named in each section. Every agent call ran on CPU (same Ollama build as the image).

## Drift monitoring (`monitoring/summary.json`, `configs/monitoring_thresholds.json`)

**Thresholds were chosen on validation only.**
- **Data:** 20 seeded streams of validation traffic (drift-free) and 20 shifting to `shifted_validation`.
- **Grid:** each label-free signal × {CUSUM, Page-Hinkley} × slack × threshold.
- **Selection rule:**
  - at most 0.5 false alarms per 100 drift-free requests;
  - at least 90% of drifts detected;
  - then the shortest mean delay.
- **Result:** two signals qualified, tool failures (slack 1.0, threshold 4.0; validation delay 21.8 requests) and latency (slack 1.0, threshold 3.0; delay 33.7). The other four never met the criteria. Both detectors chose the same parameters.

**Evaluation replay on the test families** (`stream_*.csv`). Seed 203, 300 requests, a 50-request reference window, and drift starting at request 100 with a 100-request linear ramp.

| Scenario | Success before → after | First alarm (delay) | False alarms before drift | Spearman ρ, rolling signal vs rolling success |
|---|---|---|---|---|
| Fault shift (test → fault-injected test tasks) | 0.40 → 0.24 | tool failures, CUSUM and PH: request 138 (**38**) | 0 | latency −0.83, steps −0.64, tokens −0.64, malformed −0.28, tool failures −0.06 |
| Model swap to `qwen2.5:1.5b` at request 100 | 0.40 → 0.34 | **not detected** | 0 | malformed +0.56, tool failures +0.49, steps +0.34 |
| No drift (test traffic only) | 0.41 | – | 1 per signal in 250 monitored requests (0.4 per 100) | – |

**Findings**
- **Fault drift is caught.** Tool failures (including retries the model never sees) rise from 0.45 to 1.79 per request and fire both detectors 38 requests into the ramp, while rolling success falls. Latency tracks quality most closely (ρ = −0.83, slow responses), but its threshold, set on validation traffic, did not fire on test traffic.
- **The model swap is missed.** The 1.5B model makes the agent faster and lighter: latency drops from 47 to 19 s, steps from 5.1 to 3.6, tokens from 11.9k to 7.9k. The detectors are one-sided and watch only for increases, so a drop in effort does not register, even though quality falls.
- **Fix for the model swap:**
  - run two-sided detectors on steps and latency;
  - add a per-request canary task with a known answer, i.e. the labelled metric.

  Both are left as future work, because thresholds must again be chosen on validation.

## Answer confidence (`configs/confidence_model.json`)

- **Model:** logistic regression on nine label-free run features.
- **Training data:** v5's 45 train-split runs (success rate 0.33).
- **Calibration on the 13 validation runs:** Brier score 0.155, against 0.237 for always predicting the base rate; expected calibration error 0.17.

## Retraining and the promotion gate (`retrain/*/report.json`, `retrain/gate_log.jsonl`)

**Trigger:** the fault-shift alert (`monitoring/alerts.json`: tool failures, CUSUM, request 138).

**Each run:**
1. The pool is the fault-injected tasks with a train-family parent (56 tasks).
2. The A1 active-learning score ranks production's runs on that pool.
3. Labels are revealed for the top runs from the pre-labelled pool.
4. The 12-check validation and the leakage check run (both passed).
5. A retuned exemplar bank and candidates are built, and one is chosen on `shifted_validation`.
6. Candidate and production are compared on `shifted_test` (held-out families) and gold.

**Gate** (`configs/gate.yaml`, fixed before any gold run): shifted gain ≥ 0.01 and gold drop ≤ 0.04.

| Run | Labels | Production → candidate | shifted_test | gold | Decision |
|---|---|---|---|---|---|
| `retrain_small_budget` | 4 | v5 → v6 (new bank, retry budget 3) | 0.25 → 0.30 | 0.40 → 0.44 | **promoted** v6 |
| `retrain_main` | 16 | v6 → v7 (new bank) | 0.30 → 0.35 | 0.44 → 0.48 | **promoted** v7 |
| `gate_cheaper_model` (proposal to cut cost) | – | v7 → v8 (v7 on `qwen2.5:1.5b`) | 0.35 → 0.25 | 0.48 → 0.44 | **rejected** (shifted gain −0.10) |

**Rollback drill** (`registry/history.jsonl`): `python -m code.registry.registry rollback production` moved production from v7 back to v6 in one command. v7 was then re-promoted, so the deployed version is v7.

**Gold evaluations:** every candidate and production evaluation on gold during these gate checks is logged in `TEST_SET_LOG.md`. Gold results only ever decided pass/fail against the fixed tolerance; they were never used to choose a candidate.

**Caveat:** the shifted test set has 20 tasks, so each gain of 0.05 is one task. With greedy decoding the result is deterministic but statistically small. The improvement is consistent across the two retraining rounds and in the same direction on gold.

## The deployed version v7 on every split (`partC_v7/`)

| Split | v5 (Part B selection) | v7 (deployed after retraining) |
|---|---|---|
| validation | 0.385 | **0.231** |
| test / gold | 0.400 | **0.480** |
| shifted_test | 0.250 | 0.350 |
| shifted (96) | 0.406 | 0.396 |

**Trade-off.** Retraining helped the held-out test families and the fault-injected test tasks the gate checks, but v7 is worse than v5 on the validation family (cross-tool chaining). The gate fixed in advance (`configs/gate.yaml`) looks only at shifted held-out tasks and gold, so it could not catch this. Gold also contains the unanswerable family, where more abstention helps.

**Lesson.** A gate needs a third condition: no regression on the in-distribution validation split beyond a tolerance. The gate was not changed after seeing these results; this is recorded as a limitation in `MODEL_CARD.md`.
