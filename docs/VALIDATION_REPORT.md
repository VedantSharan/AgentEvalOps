# Benchmark Data Validation Report (Part B)

**Project:** AgentEvalOps (Project 1)  
**Standard:** CS203 Assignment 1 — Part B: Data Validation  
**Date:** 2026-10-21  
**Validation Suite:** [`code/validation/benchmark_validator.py`](../code/validation/benchmark_validator.py)  
**Test Suite:** [`tests/test_benchmark_validator.py`](../tests/test_benchmark_validator.py)

---

## 1. Overview and Automated Check Suite

To ensure high data quality, integrity, determinism, and absence of answer leakage, AgentEvalOps implements a comprehensive **12-check automated validation suite**.

| # | Check Identifier | Scope | Description & Rationale |
|---|---|---|---|
| 1 | `1_jsonl_parsing` | Syntax | Validates that every line in task and trajectory files parses as a valid, non-empty JSON object. Catches truncated records, unescaped characters, and encoding errors. |
| 2 | `2_pydantic_schemas` | Schema | Validates records against strict Pydantic models (`TaskRecord` and `TrajectoryRecord`) with `extra='forbid'`, strict enum enforcement, and field type checking. |
| 3 | `3_unique_identifiers` | Uniqueness | Enforces uniqueness of `task_id` across tasks and `run_id` across trajectories, preventing duplicate logging or indexing collisions. |
| 4 | `4_reference_integrity` | Relational | Enforces foreign-key integrity ensuring every trajectory references an extant `task_id` in the benchmark. |
| 5 | `5_content_hashes` | Provenance | Recalculates canonical SHA-256 digests over `(task_id, prompt, gold_answer)` and verifies against recorded `content_hash` to detect data tampering or drift. |
| 6 | `6_duplicate_tasks` | Deduplication | Detects semantic and whitespace-normalized prompt duplicates across tasks to prevent over-representing identical tasks. |
| 7 | `7_prompt_answer_leakage` | Grounding | Verifies that gold answers do not appear verbatim within the task prompt text, ensuring the agent must invoke tools rather than copy-pasting from instructions. |
| 8 | `8_tool_determinism` | Reproducibility | Executes pure tools (`calculate`, `read_file`) multiple times under identical inputs to guarantee deterministic output. |
| 9 | `9_tool_call_arguments` | Tool Protocol | Verifies that recorded tool calls adhere to expected argument structures (e.g. `calculate` receives `{"expression": ...}`, `retrieve` receives `{"query": ...}`). |
| 10 | `10_metric_value_ranges` | Metrics | Verifies metric boundaries (`latency >= 0`, `prompt_tokens >= 0`, `completion_tokens >= 0`) and validates that step indices form a contiguous, zero-based sequence matching `total_steps`. |
| 11 | `11_missing_values` | Completeness | Verifies required core fields are non-empty and ensures runs marked with `outcome='success'` possess non-null `final_answer`. |
| 12 | `12_gold_answer_grading_consistency` | Alignment | Evaluates all trajectories claiming `outcome='success'` against the ground-truth grader (`code/tasks/grader.py`), catching false-positive success claims. |

---

## 2. Before-and-After Remediation Table

During initial benchmark validation, our automated suite flagged a critical **prompt answer leakage** violation in task family 5 (`log_file_audit`). The issue was remediated, and the pipeline revalidated.

| Metric / Check | Initial Raw Runs (Before) | Official Processed Benchmark (After) | Corrupted Test Fixtures (`data/corrupted/`) |
|---|---|---|---|
| **Total Tasks Evaluated** | 30 | 30 | 4 validly parsed (6 lines) |
| **Total Trajectories Evaluated** | 154 (incl. 52 error runs) | **102 (100% valid: 48 success, 54 failure)** | 6 validly parsed (7 lines) |
| **Excluded Infrastructure Errors** | 0 (all raw retained) | **52 (retained in raw only, excluded from processed/sampling)** | N/A |
| **Overall Status** | **FAILED** (due to leakage) | **PASSED** | **FAILED** |
| **Checks Passed** | 11 / 12 | **12 / 12 (100%)** | 1 / 12 (Tool determinism only) |
| **Checks Failed** | 1 / 12 | **0 / 12 (0%)** | **11 / 12 (91.7%)** |
| `1_jsonl_parsing` | PASS | PASS | **FAIL** (Unterminated JSON at line 1) |
| `2_pydantic_schemas` | PASS | PASS | **FAIL** (Invalid enum `non_existent_rule_enum`) |
| `3_unique_identifiers` | PASS | PASS | **FAIL** (Duplicate `task_id` & duplicate `run_id`) |
| `4_reference_integrity` | PASS | PASS | **FAIL** (Trajectory references `NON_EXISTENT_TASK_999`) |
| `5_content_hashes` | PASS | PASS | **FAIL** (Hash mismatch on `VALID_TASK_01`) |
| `6_duplicate_tasks` | PASS | PASS | **FAIL** (Prompt duplicate in `CORRUPT_DUP_TASK_02`) |
| `7_prompt_answer_leakage` | **FAIL** (`T5_audit_04`) | **PASS** (Remediated) | **FAIL** (Leaked token `AlphaBravoCharlie99`) |
| `8_tool_determinism` | PASS | PASS | PASS (Environment tool test) |
| `9_tool_call_arguments` | PASS | PASS | **FAIL** (Missing `expression` in `calculate`) |
| `10_metric_value_ranges` | PASS | PASS | **FAIL** (Negative latency & step count mismatch) |
| `11_missing_values` | PASS | PASS | **FAIL** (Null `final_answer` on success run) |
| `12_gold_answer_grading_consistency` | PASS | PASS | **FAIL** (Grader failure on success claim) |

---

## 3. Dataset Pipeline and Error Exclusion Policy

Per benchmark design principles, raw executions that terminated due to infrastructure/execution halts (such as step exhaustion or network/tool exceptions, `outcome='error'`, total 52 runs) do not represent completed agent evaluations on task objectives:
1. **Raw Log Preservation:** All 154 raw execution traces are preserved immutably in [`data/experiments/multi_provider_run/trajectories.jsonl`](../data/experiments/multi_provider_run/trajectories.jsonl) as audit evidence.
2. **Processed Benchmark Dataset:** Filtered into [`data/processed/trajectories_processed.jsonl`](../data/processed/trajectories_processed.jsonl) via [`code/experiments/filter_processed_dataset.py`](../code/experiments/filter_processed_dataset.py). Contains strictly genuine agent completions: **102 trajectories** (48 success, 54 failure, 0 error).
3. **Sampling Exclusivity:** Human annotation pools ([`data/labels/stratified_sample_45.json`](../data/labels/stratified_sample_45.json)) and downstream evaluation pipelines draw exclusively from the processed dataset, completely excluding error runs.

---

## 4. Remediation Case Study: Prompt Answer Leakage in `T5_audit_04`

### The Issue
- **Task ID:** `T5_audit_04`
- **Family:** `log_file_audit`
- **Gold Answer:** `IngressRouter`
- **Initial Prompt Text:**
  > *"Analyze the cluster event logs (`cluster_events.json`). Compare the error frequency between `IngressRouter` and `EdgeProxy-A`. Which service had more high-latency violations?"*
- **Check Failure:** `7_prompt_answer_leakage` flagged that the gold answer string `"IngressRouter"` appeared verbatim inside the prompt text. An LLM could guess or parrot the service name without retrieving or parsing the log file.

### The Fix
1. In [`code/tasks/revised_generator.py`](../code/tasks/revised_generator.py) and [`code/tasks/revised_fixtures.py`](../code/tasks/revised_fixtures.py), rewritten prompt:
   > *"Analyze the cluster event logs in `cluster_events.json`. Identify all services reporting latency violations above 500ms, and report the name of the service with the highest count of such violations."*
2. Re-generated [`data/processed/tasks_revised.jsonl`](../data/processed/tasks_revised.jsonl) with updated canonical SHA-256 content hashes.
3. Updated metadata registry [`data/metadata.csv`](../data/metadata.csv).
4. Re-ran validation suite: Check 7 passed cleanly.

---

## 5. Verification with Corrupted Fixtures

To verify that the validation checks are sensitive and fail deterministically when presented with malformed or invalid data, we generated test fixtures in `data/corrupted/`:
- [`data/corrupted/tasks_corrupted.jsonl`](../data/corrupted/tasks_corrupted.jsonl)
- [`data/corrupted/trajectories_corrupted.jsonl`](../data/corrupted/trajectories_corrupted.jsonl)

Running the benchmark validator:
```bash
python -m code.validation.benchmark_validator \
  --tasks data/corrupted/tasks_corrupted.jsonl \
  --trajectories data/corrupted/trajectories_corrupted.jsonl
```
Result: **11 failed, 1 passed (Exit code 1)**. Every targeted failure category was caught with descriptive line-number references and diagnostic messages.

Continuous integration is enforced via [`tests/test_benchmark_validator.py`](../tests/test_benchmark_validator.py).
