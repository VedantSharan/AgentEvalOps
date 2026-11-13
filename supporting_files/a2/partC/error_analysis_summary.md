# Part C error analysis (test split)

`error_analysis.csv` lists all **58 test-split errors** of the final system (v5) and the three baselines. Each error has an automatic A1 failure category (`code/evaluation/categorise.py`), the evidence (the first failed or last tool call), a suggested fix and the MLflow trace ID. This file summarises the patterns, which were read from the traces. The test split was not used to change anything; the fixes below are proposals for future work, to be validated on validation data first.

| System | Errors (of 25) | Main category |
|---|---|---|
| final (v5) | 15 | correct_process_with_wrong_answer (13), malformed_arguments (2) |
| b2_original | 15 | correct_process_with_wrong_answer (6), malformed_arguments (4), premature_stop (2), hallucinated_result (2), loop (1) |
| b1_direct | 14 | premature_stop (14): no tools, so every file audit fails |
| b0_abstain | 14 | premature_stop (14): abstains on every answerable task |

## Pattern 1: the final agent abstains on every file audit (14 of 14 file-audit tasks)

- **What happens.** Each audit task names two files, a log and `sla_reference.json`. The agent reads the log, then looks for the SLA threshold with `retrieve` (13 of 14 errors end on a `retrieve` call). The knowledge base does not hold thresholds, so it finds nothing and answers "unanswerable".
- **Cause.** System prompt v3 added "never assume a rate, percentage, threshold or policy value: look it up with retrieve or read_file". It was written from validation failures, where the agent invented a bonus rate. On the unseen file-audit family it points the model to `retrieve`, and the 3B model does not go back to `read_file` for the second named file. The original A1 loop, without this rule, solves 2 of 14 file audits; v5 solves 0 of 14. On unanswerable tasks, v5 gets 10 of 11 against B2's 8 of 11, so on test the two systems tie at 0.400 overall.
- **Suggested fix:**
  - In the prompt and tool description: "when the question names a file, read every named file before using retrieve".
  - A plan step that lists the named files.
  - A validation-family exemplar that reads two files.

  These are untested on held-out data. Following the brief, any change would be chosen on validation, then re-evaluated once on test/gold.

## Pattern 2: abstention is the final agent's strength

- **Unanswerable test tasks:** 10 of 11 correct.
- **Shifted information outage:** 19 of 19 correct. When every data tool fails, it abstains instead of inventing a value.
- **B2 comparison:** B2 often answers with an invented value, giving 2 hallucinated_result errors.

## Pattern 3: malformed calls (final 2, B2 4)

- **What happens:** `query_records` is called with columns that do not exist (for example `satisfaction_score` on the employee table), and file arguments are sometimes wrong.
- **Fix:** the column enum in tool descriptions v2 reduced this relative to B2. Validating arguments against the schema before execution would remove the rest.

## Weakest groups (final system, from `slices.json`)

| Split | Slice | Success | Tasks |
|---|---|---|---|
| test | file_sla_auditing | 0.000 | 14 |
| shifted | format_change | 0.208 | 24 |
| shifted | calculator_outage | 0.200 | 5 |
| shifted | file_sla_auditing family | 0.250 | 20 |
| shifted | transient_failure / slow_response | 0.292 | 24 each |

Format changes (CSV instead of JSON records, renamed keys) and slow tools hurt most among the faults. Retries hide transient failures from the model, but several of those runs fail for the pattern-1 reason anyway.
