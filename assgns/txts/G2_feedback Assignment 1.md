# CS203 Assignment 1 Evaluation: Project 1 (AgentEvalOps)

Submission: CS203_A1_Project1_23110177_23110297_23110354.zip (3
members). Evaluated against the Assignment 1 brief (Parts A to F, 100
marks). Scores are an estimate, not official marks.

## Scope and method

I read the report, git history, data, labels, prompts and supporting
files, and checked report numbers against their source files. I also
recomputed the inter-annotator kappas independently from the raw Label
Studio exports. I did not re-run dvc repro or pytest, and I did not
watch the screen recording (its length, 131.8 s, was checked).

Checked and matching: the report's own traceability file (93 of 93
numbers match), the gold-set SHA-256 values (both files), the 154
trajectories in the multi-provider run, the validation JSON outputs (raw
12/12, corrupted 1/12, augmented 12/12), and the split counts (78 / 32 /
44 trajectories, 96 shifted tasks).

Independently recomputed from the raw exports (matches the report
exactly): round 1 outcome kappa 0.848, category kappa 0.448, step kappa
0.646; round 2 outcome kappa 0.947, category kappa 0.483, step kappa
0.865; median seconds per item 4.2 versus 34.5 for annotator A1.

## Overall: about 79/100 (plausible range 75 to 83)

A technically strong and unusually well-evidenced submission: an
automated report-number checker, candid bias notes, and honest
limitations. The main losses are the trajectory count, an MLflow trace
store that is not in the zip, a shifted split with no agent runs, and a
heavily compressed timeline with unequal contributions.

| Part | Marks | Est. | Verdict |
|------|------:|------|---------|
| A: Collection | 15 | ~11 | Good build, thin collection, MLflow store missing |
| B: Validation | 20 | ~15 | Solid suite, weak before/after evidence |
| C: Labelling | 25 | ~21 | Very strong |
| D: Augmentation and active learning | 20 | ~16 | Good, honestly caveated |
| E: Versioning and splits | 10 | ~8 | Strong, shifted split has no runs |
| F: Report and reproducibility | 10 | ~8 | Strong evidence index, some inconsistencies |

## Part A: Collection (~11/15)

### Strengths

- A plain-Python function-calling agent with five typed tools (retrieve,
  calculate, query_records, read_file, submit_answer).

- A task suite of 6 template families x 5 instances = 30 tasks, with
  gold answers computed by solver functions and 5 ambiguous or
  unanswerable tasks. This is within the brief's 20 to 30 range.

- A queue-based multi-provider collection worker that ran 154
  trajectories across 10 models and 5 free-tier providers.

- metadata.csv has source, timestamp, collector, tool version and
  content hash on every row (no missing values), and the dataset card
  carries a licence and consent note.

### Gaps

- 154 trajectories is below the brief's "a few hundred", and only 20 of
  the 30 tasks were ever run. 52 runs ended in provider errors and 42
  ended without a submitted answer.

- Model coverage is uneven: 2 to 20 runs per model.

- The mlruns/mlflow.db store is gitignored and not in the zip, so the
  headline "174 traces, 633 spans" claim cannot be verified from the
  submission. The real collection was also not traced live; trajectories
  were ingested into MLflow afterwards.

- The report says metadata has 381 records (30 base tasks + 154
  trajectories + 149 augmented), but those parts add to 333. The file
  actually holds 178 trajectory rows and 54 task rows because it
  includes the 24-run pilot.

## Part B: Validation (~15/20)

### Strengths

- A 12-check suite covering JSON parsing, Pydantic schemas, unique IDs,
  reference integrity, content hashes, duplicates, answer leakage into
  prompts, tool determinism, tool-call arguments, metric ranges, missing
  values and gold-answer grading consistency. This matches every Project
  1 check in the brief.

- On the deliberately corrupted fixtures, 11 of 12 checks fail as
  intended.

- A genuinely valuable finding came from labelling: grader v1 rejected
  correctly formatted answers (for example "\$25,920"). Grader v2 raised
  correct answers from 48 to 87, with 39 flips from wrong to correct and
  0 the other way.

### Gaps

- The suite itself found only one real problem (answer leakage in
  T5_audit_04). The failing "before" output was not saved, so the
  before-and-after claim has no file behind it. The report's table
  compares raw, processed, corrupted and augmented sets, which are not
  really before and after.

- The 12-check suite was run on only 53 of the 149 augmented tasks; the
  96 fault-injected tasks rely on separate augmentation checks.

- Internal inconsistency: the contribution table and AI register say the
  suite has 10 checks, while the code, validation report and results say
  12.

## Part C: Labelling and agreement (~21/25)

### Strengths

- Two independent rounds in separate Label Studio projects: 45
  stratified trajectories in round 1 and 40 new ones in round 2 (85
  total, within the 40 to 50 per-round range). Both annotators labelled
  every item.

- All three required label levels are used in a fixed order: outcome,
  primary failure category (7 classes), first-error step.

- Guidelines v1 (about 1,100 words) and v2 (about 1,300 words), with the
  changes tied to disagreements.

- Agreement is reported honestly with Cohen's kappa, PABAK and a
  both-failure subset: outcome kappa 0.85 to 0.95, step kappa 0.65 to
  0.87, category kappa 0.45 to 0.48 (0.05 to 0.21 on genuine failures).

- All 27 disagreements are discussed with trace evidence, which is more
  than the 10 required.

- Automatic detectors were compared with human labels: malformed-call
  detector precision 0.89, recall 0.80; loop detector recall rises from
  0.25 to 1.0 when the step-limit rule is added, at precision 0.29.

- A gold test set is frozen with SHA-256 values for both files.

- The labelling protocol is candid about its own weaknesses.

### Gaps

- Category agreement on real failures stays low, so this is the main
  data-quality risk for Assignment 2. Round 2 was deliberately enriched
  with failures (prevalence 0.625 versus 0.16), so the two rounds are
  not directly comparable.

- Round 1 median time per item was 4.2 s for annotator A1, which
  suggests shallow reading. Round 1 annotators also saw grader v1's
  verdict, which risks anchoring. The Label Studio version for round 1
  was not recorded.

- The adjudicator is annotator A2, and AI-drafted proposals sided with
  A2 more often than A1 (round 2: 11 versus 2). The protocol discloses
  this bias. ADJ1 confirmed 27 proposals and changed 2.

- Three empty submissions on unanswerable tasks were labelled success by
  both annotators.

- The frozen gold set contains only 24 human-labelled trajectories (16
  success, 8 failure).

- The raw Label Studio exports contain a personal Gmail address (90
  occurrences), which sits uneasily with the anonymisation requirement.

## Part D: Augmentation and active learning (~16/20)

### Strengths

- 149 augmented tasks across the three types the brief names: entity and
  parameter variants (23), paraphrases (30) and fault injection (96:
  transient failure, outage, +2 s latency, changed output format).

- Every augmentation has an executed check. The variant check rejected 3
  candidates whose evidence the tools cannot reach, and paraphrase
  intent was confirmed in a human spot check (10 of 10).

- Active learning versus random at 3 budgets (10, 20, 30) with 1,000
  seeded trials. Failure coverage was 0.29, 0.43, 0.55 for the
  label-free score against 0.12, 0.24, 0.35 for random.

- The report states the limits itself: the labelled pool is not a random
  sample, 13 of the top 30 picks never submitted an answer, and random
  selection reached more distinct categories at budget 30 (0.79 versus
  0.67).

### Gaps

- No agent runs were made on any augmented task (no API keys), so the
  fault-injection behaviour changes, such as 19 tasks relabelled
  unanswerable, are checked by rules only.

- Paraphrases are rule-based rather than LLM-generated, and only 23
  entity variants were built.

- The active-learning comparison runs on the 85 already-labelled
  trajectories rather than a fresh unlabelled pool.

## Part E: Versioning and splits (~8/10)

### Strengths

- Splits by template family, as the brief requires: three families in
  train, one in validation, two in test (including the whole
  ambiguous/unanswerable family). Variants, repeated runs and labels
  follow their parent's split.

- The shifted split is the 96 fault-injected tasks, with a parent_split
  column so Assignment 2 can compare against the test families.

- Ten automatic leakage checks all pass, with negative tests on a
  tampered manifest. A 14-stage DVC pipeline reproduces the derived
  files, and the a1-final tag points at the final commit.

### Gaps

- The shifted split has no agent trajectories, so Assignment 2's drift
  experiments will need new runs.

- Only three families train the models while two form the test set,
  which is a thin basis.

- The DVC remote is a local folder that is not in the zip. The README's
  offline route copies the frozen data back in instead.

- The git working tree shows 145 files modified against the a1-final tag
  (identical insertions and deletions of 27,696 lines, consistent with
  line-ending changes), so the repository is not clean.

## Part F: Report and reproducibility (~8/10)

### Strengths

- A report body of about 4 pages plus appendices, with an evidence index
  of 93 IDs and a script that re-reads every number (93 of 93 match).
  This is a standout feature.

- Dataset card, ENVIRONMENT.md, LABELLING_PROTOCOL.md, HANDOFF_A2.md,
  and a README that rebuilds in 5 commands. A saved test log shows 314
  tests passing.

- The full Claude session prompts are included verbatim, and the report
  discloses AI use.

### Gaps

- The report's appendix still says the screen recording is "(to add)",
  although a 131.8 s recording is in the zip.

- The Gemini (Antigravity) prompt log keeps only "actionable engineering
  prompts", not every prompt as the brief requires.

- README says 374+ tests pass, but the saved log shows 314.

- A local path (/home/kshitij/...) appears in the saved test log, and
  the package is named code, which shadows a standard-library module
  (documented workaround).

## Process issues to raise with the professor

1. **Possible late submission.** The brief's deadline is 18 October,
    11:59 PM. The repository's initial commit is 18 October, but
    substantive work starts on 20 October. The a1-final tag is dated 22
    October 14:38 IST and the zip files are dated 22 October. Unless the
    deadline was extended, this is late, so please confirm.

2. **Unequal contribution.** The git history shows 29 commits from
    Vedant Sharan (including merges) and 18 from Kshitij Giri, and none
    from Shardul Junagade (23110297). The report's contribution table
    credits him with proposal drafting, domain research, guideline
    review and report review, but no code or labelling. The brief says
    individual marks are affected when contributions differ a lot.

3. **Heavy AI use.** Gemini 3.8 Flash (Antigravity IDE) built Parts A
    and B, and Claude built Parts C to F. This is disclosed in full,
    which counts in the team's favour, but the brief requires each
    member to be able to explain the code.

4. **Compressed timeline.** Most of the work, including both labelling
    rounds and all of Parts D to F, happened in roughly 48 hours, which
    may explain the shallow round 1 annotation.
