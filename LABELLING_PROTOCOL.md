# Labelling Protocol

This protocol lets someone repeat the AgentEvalOps labelling the way it was actually done.

- Facts read from the export files are marked *(export)*.
- Facts that a team member still has to confirm are marked **To confirm**.
- All times are UTC, as stored by Label Studio.

## 1. People (anonymised)

| ID | Role |
|---|---|
| **A1** | Annotator in rounds 1 and 2 (team member) |
| **A2** | Annotator in rounds 1 and 2 (team member) |
| **ADJ1** | Adjudicator for both rounds. **ADJ1 is A2**, acting as adjudicator after both rounds' exports were saved. The adjudicator was therefore not independent of one of the annotators (see §7). |

No external annotators took part. No personal data about annotators is stored in the derived files; all outputs use only A1, A2 and ADJ1. The raw round-1 export of A1 is submitted unchanged, as the brief requires, and contains that annotator's Label Studio account e-mail in `completed_by`.

## 2. What was labelled, by whom, in which round

| Round | Items | A1 labelled | A2 labelled | Item list |
|---|---|---|---|---|
| 1 | 45 trajectories | all 45 | all 45 | `data/labels/stratified_sample_45.json` (seed 42, `code/annotation/sample_for_labelling.py`) |
| 2 | 40 trajectories, disjoint from round 1 | all 40 | all 40 | `data/labels/round2_sample.json` and `round2_manifest.csv` (seed 2026, `code/annotation/sample_round2.py`) |

Both rounds are **fully double-annotated**: overlap 45/45 and 40/40 *(export)*.

Round-2 sample design: the 40 items were drawn from the 109 unlabelled runs, stratified as 12 correct answers, 10 wrong answers and 18 runs that hit the step limit or a tool error. This deliberately raised the failure prevalence above round 1's.

## 3. Tools, configuration and guidelines per round

| Round | Label Studio version | Project config | Guidelines | Label Studio projects |
|---|---|---|---|---|
| 1 | **Not recorded** (the exports do not store it) | `code/annotation/label_studio_config.xml` | `docs/annotation_guidelines_v1.md` (v1) | A1: project 11, A2: project 4 *(export)*; separate projects |
| 2 | A2: 1.23.2 (observed). A1: **To confirm** | `code/annotation/label_studio_config_v2.xml` | `docs/annotation_guidelines_v2.md` (v2) | A1: project 12, A2: project 5 *(export)*; separate machines |

Changes from v1 to v2 are listed at the end of `docs/annotation_guidelines_v2.md`. Each change is traced to a round-1 disagreement.

The round-1 config: the file in the repository is the version at the time of the export. **To confirm** that no edit was made to it between labelling and the commit "Modified Final XML".

## 4. Order of label levels

All three levels were filled top-down in this order:

1. **outcome** (success / failure / error)
2. **primary_failure_category** (`no_failure` when outcome = success)
3. **first_error_step**

In round 2, a one-line note for every failure was added as a fourth item.

**Why this order:**
- The outcome is the most objective judgement: is the submitted answer right?
- The category depends on the outcome, because it only exists for failures.
- The step locates the first occurrence of that category.

Deciding the category first and then the outcome would bias the outcome toward whatever category had been chosen.

**Same order for every annotator:** both annotators used the same form, where the levels appear top-down, and the written guidelines tell them not to go back. Going back cannot be technically prevented in Label Studio; the order was **instructed, not enforced**.

**v2 form change:** the step field (and the note) is shown only after `failure` or `error` is selected. A value typed before switching the outcome back to success remains in the export. This happened once (A1, run d1cbda85), and it is treated as "no step" (guidelines v2, R4).

## 5. Order of items

| Round | Order shown | Order actually followed *(export timestamps)* |
|---|---|---|
| 1 | Import order of `stratified_sample_45.json` (Label Studio inner_id 1–45). No shuffle at labelling time; the sample itself was drawn with seed 42. | A2: 1→45 in sequence. A1: 1→45 in sequence except item 4, which was labelled last. |
| 2 | `presentation_order` 1–40: a seeded shuffle (seed 2026) inside `sample_round2.py`, identical for both annotators; Label Studio task sampling set to Sequential | Both: 1→40 in sequence |

## 6. What annotators could see

| | Round 1 | Round 2 |
|---|---|---|
| Prompt, gold answer, agent's final answer, full step transcript | yes | yes |
| Run ID, task ID, family, model name, evaluation rule | yes | yes |
| **Logged grader outcome** (grader v1) | **yes**: shown as "Logged System Outcome" | **no**: removed, because grader v1 was shown to be wrong on formatted answers |
| Model predictions / pre-annotations | no (0 predictions in every export) *(export)* | no *(export)* |
| The other annotator's labels | no (separate Label Studio projects) | no (separate projects on separate machines; instructions forbade opening the other export before saving one's own) |

Annotation time per item, from Label Studio `lead_time`:
- Round 1: median 4.2 s (A1), 6.8 s (A2).
- Round 2: median 34.5 s (A1), 28.0 s (A2).

Source: `supporting_files/agreement/pre_post_comparison.csv`. The very short round-1 times are reported as a limitation.

## 7. Disagreements: how they were resolved and who decided

1. `code/annotation/agreement.py` matches the two exports by `run_id` and lists every item where any of the three levels differs. Round 1 has 8 such items; round 2 has 19.
2. For each disagreement, a proposed resolution was **drafted with AI assistance** (Claude). Each draft rests on the trace evidence, with gold answers recomputed from the tool outputs, and on the written guideline rules. Every row records its evidence, justification, guideline rule and confidence, and whether team confirmation is needed:
   - `supporting_files/agreement/round1/adjudication_round1.csv`
   - `supporting_files/agreement/round2/adjudication_round2.csv`
3. **ADJ1 (= A2) made the final decision on every row** and signed it (`confirmed_by`, `confirmed_on`).
   - Round 1: all 8 proposals accepted.
   - Round 2: 17 of 19 accepted. **2 were changed**: 77461287 and 961e13f8 went from the proposed `wrong_tool` to `malformed_arguments`, which was A2's own round-2 label.
4. `code/annotation/build_labels.py` builds the final labels:
   - agreed labels are used as given (`label_source = agreed`);
   - adjudicated rows are used for disagreements (`label_source = adjudicated`);
   - it refuses to run while any adjudication is unconfirmed.
   - Outputs: `data/labels/labels_round1.csv` and `labels_round2.csv`.
5. **Agreed labels were never overridden.** Three round-2 items were labelled `success` by both annotators but contain an empty submission on an unanswerable task (3fcee431, 58d66f53, ee77ee60). Grader v2 and a strict reading of guideline v2 would say failure. They are kept as agreed and reported as a guideline gap (`docs/annotation_findings.md`, v3 recommendation 3).

**Bias note:** the adjudicator is one of the two annotators, and the drafted proposals sided with A2 more often than with A1 (round 2: 11 vs 2). Final labels may therefore lean toward A2's reading of the taxonomy. Agreement statistics are computed on the raw, independent labels, so they are not affected. The detector evaluation is reported separately against A1's and A2's own raw labels, as well as against the final labels (`supporting_files/detectors/detector_metrics.csv`).

## 8. Timeline and export files

| Round | Annotator | Labelling window (UTC) *(export)* | Raw export (unchanged) | SHA-256 (first 16 hex) |
|---|---|---|---|---|
| 1 | A1 | 2026-10-21 17:45 – 18:01 | `supporting_files/labels/annotator1_raw.json` | `da65349a0b51aca1` |
| 1 | A2 | 2026-10-21 17:55 – 18:06 | `supporting_files/labels/annotator2_raw.json` | `212012fdc9471604` |
| 2 | A1 | 2026-10-21 19:38 – 20:09 | `supporting_files/labels/round2/annotator1_raw.json` | `ea92d8888b208ebe` |
| 2 | A2 | 2026-10-21 19:38 – 19:55 | `supporting_files/labels/round2/annotator2_raw.json` | `428b05b5ce42fdd7` |
| adjudication r1 | ADJ1 | confirmed 2026-10-22 (`confirmed_on` "22/10/26") | `supporting_files/agreement/round1/adjudication_round1.csv` | — |
| adjudication r2 | ADJ1 | confirmed 2026-10-22 | `supporting_files/agreement/round2/adjudication_round2.csv` | — |

Full checksums: `sha256sum supporting_files/labels/*.json supporting_files/labels/round2/*.json`.

Repairing an adjudication file: while ADJ1 edited `adjudication_round2.csv` in a spreadsheet, two `run_id` cells were altered by the program. They were restored from `disagreements.csv` by matching the first 8 characters; no label value was changed. The annotators' export files themselves were never edited.

## 9. Repeating the labelling

1. Install the pinned environment (`README.md`).
2. Start Label Studio and create one project per annotator with the round's config.
3. Import the round's sample file and set task sampling to Sequential.
4. Give each annotator the round's guidelines, plus `docs/ROUND2_ANNOTATOR_INSTRUCTIONS.md`.
5. Export JSON (not JSON-MIN) unchanged.
6. Run `dvc repro`. It recomputes agreement, and it rebuilds the final labels from the exports and the signed adjudication files.
