# AgentEvalOps — Annotation Guidelines v2

**Applies to:** annotation round 2 onward. Round 1 used [v1](annotation_guidelines_v1.md), which is kept unchanged.
**Label Studio config:** `code/annotation/label_studio_config_v2.xml`
**Why v2 exists:** see "Changes from v1" at the end. Every change is traced to a round-1 disagreement in `supporting_files/agreement/round1/adjudication_round1.csv`.

## 1. What you see and what you label

Each item shows the task prompt, the gold answer, the agent's submitted answer, and the step-by-step transcript (steps are numbered from 0, and every step counts, including `llm_response` and `submit_answer`). In v2 you do **not** see the automatic grader's verdict: it was shown in round 1 and is known to be wrong for formatted answers.

Fill the three levels **in this order**, and do not go back to change a level after reading later parts of the form:

1. **Outcome**: is the submitted answer right?
2. **Primary failure category**: if not, what went wrong first?
3. **First error step**: where did it go wrong?

Then write a one-line note for every `failure` (what happened and at which step).

## 2. Level 1: outcome

| Label | Use when |
|---|---|
| `success` | The submitted answer states the gold value, **regardless of format** (R7): `$25,920`, `25,920 USD` and "Alice's salary is $95,000" are all correct for gold `25920.00` / `95000`. For an unanswerable or ambiguous task: the agent abstains or explains the ambiguity **without committing to a value** (R6). |
| `failure` | Wrong value; no answer; a value given for an unanswerable task (including `0`); or the run hit the step limit without submitting. |
| `error` | **Only** an infrastructure failure where the agent never acted (provider/API crash, empty transcript). Step-limit exhaustion is **not** `error` (R7). |

## 3. Level 2: primary failure category (pick the first error, not the most visible one)

| Category | v2 definition (decisive test) |
|---|---|
| `no_failure` | Use if and only if outcome = `success`. |
| `wrong_tool` | **R1.** The agent called a tool that **cannot** provide the needed information, or a tool that does not exist. Using the expected tool badly is *not* `wrong_tool`. *Test: could a better call to the same tool have worked? If yes, it is not wrong_tool.* |
| `malformed_arguments` | **R3.** The tool **rejected** the call because of its arguments: a missing or wrong key, a wrong type, an unparseable expression, or a non-existent file or field name (the tool result shows an error). A well-formed call with the wrong *values* (e.g. `45000 * 0.82` when `1.18` was needed) is **not** malformed. |
| `hallucinated_result` | The answer (or an intermediate claim) states facts or values that the tool outputs do not support, including any value given for an unanswerable task. |
| `loop_or_repetition` | **R5.** The same tool with identical arguments is called 3 or more times without progress, **or** the run hits the step limit while cycling between calls without progress. |
| `premature_stop` | **R5.** The agent gave up or answered while the needed evidence **was obtainable** with the available tools (e.g. it stopped after a few unproductive queries without rephrasing). |
| `correct_process_with_wrong_answer` | **R2.** The right tools were used and the needed evidence was obtained, but the agent miscounted, mis-added, or misread the question (e.g. subtracted tax for "post-tax"). |

**R8, the root-cause rule (unchanged from v1):** when several things go wrong, label the **earliest** one.
**R5, unproductive but valid calls:** the call itself is not an error category. Label how the agent responded to it: repeating it is `loop_or_repetition`, and giving up is `premature_stop`.

## 4. Level 3: first error step (R4)

* The **index of the first step that departs from a valid solution path**. Count every step type.
* An unproductive call that could never have worked (e.g. a 25-word sentence sent to keyword `retrieve`) **is** the first error step, even if the visible symptom (a repeat, giving up) comes later.
* For `success`, **leave it empty**. In v1, `0` meant both "no error" and "error at step 0".
* If the error is a wrong number in an `llm_response` and is then submitted, the `llm_response` step is the first error.

## 5. Edge cases settled in round 1

| Situation | Label |
|---|---|
| Recovered error (e.g. file-not-found at step 0, fixed at step 1, correct answer) | `success` / `no_failure` (v1 Rule 1, unchanged) |
| Empty log: "average latency cannot be computed (or is 0)", then submits `0` | `failure` / `hallucinated_result` / step where `0` is first asserted |
| Ambiguous name: answer lists both matching employees from the retrieved fact and does not pick one | `success` / `no_failure` |
| Expected tool used with a non-matching query three times, identical arguments | `failure` / `loop_or_repetition` / step 0 |
| Expected tool used with three different non-matching queries, then gives up | `failure` / `premature_stop` / step 0 |
| Correct file read, count wrong in the submitted answer | `failure` / `correct_process_with_wrong_answer` / the step that states the wrong count |

## 6. Worked examples (from round-1 data)

**A. `26ff8938` (T4_chain_02).** Step 0 `retrieve` returns a tax rate of 18%. Step 1 `read_file` gives revenue 45,000. Step 2 `calculate("45000 * 0.82")` = 36,900; submitted `$36,900.00`; gold 53,100. → `failure` / `correct_process_with_wrong_answer` / **2**. Not `malformed_arguments`: the call is well-formed (R3).

**B. `0c9eefc9` (T4_chain_04).** Steps 0–2 send three different whole-sentence queries to `retrieve` and get 0 matches each; then "I was unable to find the information". The facts exist in the knowledge base. → `failure` / `premature_stop` / **0** (R1, R5).

**C. `b77bee95` (T6_unans_03).** `retrieve` says there are two employees named Rahul Sharma (Engineering, Sales); the answer reports both. → `success` / `no_failure` / *(empty)* (R6).

## 7. Independence rules

Work alone. Do not discuss items, look at the other annotator's project, or open grader outputs until you have exported your labels. Work through the items in the order Label Studio presents them (the order is fixed by the sampling seed; see `LABELLING_PROTOCOL.md`). Read the whole transcript before choosing a label.

## Changes from v1

| # | v1 | v2 | Round-1 evidence |
|---|---|---|---|
| R1 | `wrong_tool` = "inappropriate tool" (undefined) | Only if the tool cannot provide the information, or does not exist | A1 used `wrong_tool` 5 times where the expected tool was used (08eeee21, 0c9eefc9, 2ce94e88, 31eabcb1, 4393465c) |
| R2 | CPWA = "arithmetic/logic error" | Includes misreading the question when evidence was obtained | 26ff8938, 2ce94e88, 4393465c, 08eeee21 |
| R3 | `malformed_arguments` = "invalid schema" | The tool rejected the arguments; wrong values in a valid call are not malformed | 26ff8938 (A1: malformed) |
| R4 | `first_error_step` = 0 for success | Empty for success; first deviation; every step type counts | Step κ 0.57 as labelled vs 0.65 recoded; 2ce94e88, 31eabcb1, 097b92f6 |
| R5 | No rule for valid but unproductive calls | Label the agent's response to them (repeat → loop, give up → premature stop) | 0c9eefc9, 31eabcb1 |
| R6 | Unanswerable: "reports unanswerability" | Abstention **or** ambiguity disclosure, with no committed value; `0` counts as a value | 097b92f6, b77bee95 |
| R7 | No guidance on answer format; `error` included step exhaustion | Format-insensitive correctness; `error` only for infrastructure crashes | 13 items marked success by both annotators were logged as failures by grader v1 (see `code/tasks/grader_v2.py`) |
| R8 | Root-cause rule | Unchanged; now stated as "earliest error wins" | — |
| UI | Logged grader outcome shown; step field always visible, default 0 | Grader outcome hidden; step field shown only for failure/error; note required for failures | Default 0 makes "not filled in" indistinguishable from "step 0" |
| QA | "20% subset double-annotated" | All round-2 items are double-annotated (as round 1 actually was) | Round-1 exports: 45/45 overlap |
