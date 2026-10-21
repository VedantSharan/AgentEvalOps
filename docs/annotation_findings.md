# Annotation Findings: Round 1 (v1) vs Round 2 (v2), Detectors, and v3 Recommendations

Every number below is produced by a script and stored in the file named next to it. Numbers in §3 and §4 depend on the round-2 adjudication. They are final only after `supporting_files/agreement/round2/adjudication_round2.csv` has been confirmed (`confirmed_by` filled in) and the scripts have been rerun.

## 1. Rounds

| | Round 1 | Round 2 |
|---|---|---|
| Guidelines / Label Studio config | v1 / `label_studio_config.xml` | v2 / `label_studio_config_v2.xml` |
| Items (both annotators, full overlap) | 45, stratified from completed runs | 40, seeded stratified sample of the 109 unlabelled runs, including step-limit runs (`code/annotation/sample_round2.py`, seed 2026) |
| Failure prevalence (A1 / A2) | 0.16 / 0.20 | 0.63 / 0.60 (deliberately enriched) |
| Items with any disagreement | 8 | 19 |
| Median time per item (A1 / A2) | 4.2 s / 6.8 s | 34.5 s / 28.0 s |

Source: `supporting_files/agreement/pre_post_comparison.csv`, from `code/annotation/compare_rounds.py`.

## 2. Agreement before vs after the guideline revision

| Measure | Round 1 (v1) | Round 2 (v2) |
|---|---|---|
| Outcome κ (raw agreement) | 0.848 (0.956) | 0.947 (0.975) |
| Category κ, all items (raw, PABAK) | 0.448 (0.822, 0.793) | 0.484 (0.600, 0.533) |
| Category κ, items both called failure (n) | 0.045 (7) | 0.212 (24) |
| First-error step κ, nominal / quadratic (no_failure coded as −1) | 0.646 / 0.689 | 0.865 / 0.789 |
| Exploratory 4-class taxonomy κ, both-failure items | 0.045 | 0.285 |

Interpretation:
- Agreement on outcome and first-error step improved after v2.
- Category agreement on genuine failures improved, but it is still poor.
- The raw category agreement fell because round 2 contains three times as many failures, and failures are where the categories differ. Kappa is prevalence-dependent, so the rounds are not perfectly comparable.
- Merging categories into 4 coarse groups (post hoc, exploratory) helps only slightly. The disagreement is systematic between annotators, not a matter of fine boundaries.
  - A1 used `wrong_tool` 6 times and `correct_process_with_wrong_answer` 0 times.
  - A2 used them 0 and 7 times respectively.

## 3. Disagreements reviewed (27 in total)

Each disagreement has trace evidence, a proposed resolution, a justification and a confirmation field:
- `supporting_files/agreement/round1/adjudication_round1.csv` (8 rows, confirmed by ADJ1)
- `supporting_files/agreement/round2/adjudication_round2.csv` (19 rows)

Recurring patterns:
1. **R1 / R3 overlap.** `query_records` (employee table) is called with ticket or project fields and returns "Unknown field". The tool rejects the arguments (R3), and the tool can never answer the question (R1). Examples: 77461287, 961e13f8.
2. **Step-limit wandering.** The agent has the answer early, keeps searching, and submits nothing at the step limit. The written R5 says `loop_or_repetition`; annotators chose `premature_stop`, `wrong_tool` or `correct_process_with_wrong_answer`. Examples: 0ca1febf, 6b41cf74, a97af026, db85cc0b.
3. **Well-formed call with wrong values labelled `malformed_arguments` by A1.** Example: `45000 * 0.82`. R3 already excludes this case. Examples: d4490fcf, d5755a30.
4. **Number of steps vs. first deviation.** One annotator recorded the step where the run *ended* rather than where it first went wrong. Examples: 0ca1febf (11 vs 1), a97af026 (12 vs 2).
5. **UI residue.** A step value entered before switching the outcome to `success` stays in the export, although the field is hidden (1 item: d1cbda85).

Human vs grader:
- Grader v2 was designed on round-1 items. On round 2, which it had not seen, agreement with humans is κ = 0.83 (A1) and 0.78 (A2); grader v1 scored 0.52 and 0.48. Source: `supporting_files/agreement/round2/human_vs_grader.summary.json`.
- Three items that **both** annotators labelled `success` are runs that submitted an *empty* answer on an unanswerable task at the step limit: 3fcee431, 58d66f53, ee77ee60. Grader v2 says failure. These are agreed labels and were not overridden; this is a guideline gap (see §5).

## 4. Automatic detectors vs human primary labels

Source: `supporting_files/detectors/detector_metrics.csv`, from `code/annotation/detectors.py`.

| Detector | vs final labels: P / R / κ | vs A1 κ | vs A2 κ |
|---|---|---|---|
| `malformed_strict` (tool rejected the arguments; file-not-found excluded) | 0.67 / 0.86 / 0.72 | 0.40 | 0.88 |
| `malformed_any` (file-not-found included) | 0.37 / 1.00 / 0.48 | 0.26 | 0.58 |
| `loop_strict` (≥3 identical calls) | 0.33 / 0.25 / 0.26 | −0.02 | 0.49 |
| `loop_extended` (+ step limit with no answer) | 0.29 / 1.00 / 0.40 | 0.11 | 0.11 |

Most "false positives" are real events that humans did not choose as the *primary* cause. Examples are a recovered file-not-found, or a malformed call that occurred after an earlier root cause. Each case is listed in `supporting_files/detectors/detector_errors.csv`.

## 5. Recommended changes for a v3 guideline (not applied to any round in Assignment 1)

1. **Precedence rule R1 over R3.** If the chosen tool can never hold the needed data, label `wrong_tool`, even when the tool rejected the arguments.
2. **R5 made explicit.** "Had the evidence, never submitted before the step limit" is `loop_or_repetition`. `premature_stop` requires an explicit give-up or an answer.
3. **Abstention must be explicit.** An empty submission is a failure, even on unanswerable tasks.
4. **Calibration.** Before a round, both annotators label 5 shared practice items and discuss them. Rounds 1 and 2 show that systematic differences between annotators, not the taxonomy alone, drive category disagreement.
5. **UI.** Clear hidden fields when the outcome changes. Show the step index next to every transcript step.
6. **Tool layer.** `calculate` returns raw binary floats (`13774.999999999998`). Round results to 10 significant digits, so agents do not truncate them to a wrong cent (a0a088d0).
