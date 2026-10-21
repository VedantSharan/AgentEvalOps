# AI assistance: Claude session, 22 Oct 2026 (Kshitij Giri)

- **Assistant:** Claude in the claude.ai app (agentic workspace), configured model `claude-opus-5-5`.
- **Attachments given with the first prompt:** `AgentEvalOps-main.zip` (the repository), `CS203_Assignment1.pdf`, `proposal.pdf`.

The prompts are reproduced exactly as typed, typing errors included. Pasted terminal output is not repeated; it is marked `[terminal output pasted: …]`. "Output went into" lists the files created or changed as a result. The assistant worked on a copy of the repository and delivered stage zips, which were then unzipped, run and committed by the user (see the git log).

---

### Prompt 1: master prompt

````text
# Master Prompt: Complete CS203 Assignment 1 — AgentEvalOps (Remaining Parts C Agreement, D, E and F)

## 1. Your role

Act as my **senior ML engineer, data engineering mentor, code reviewer, and step-by-step technical guide** for CS203 Software Tools and Techniques in AI at IIT Gandhinagar.

I am a student working on Project 1: **AgentEvalOps — A Systematic Evaluation, Optimization, and Monitoring Framework for Tool-Using LLM Agents**.

I want you to help me complete the remaining work in Assignment 1 using my existing project repository. I need both technically correct implementation and a submission that satisfies the official assignment requirements.

I am a beginner in some of these tools and workflows. Therefore, explain everything in simple English, starting with the basic concept, then explaining what we need to do, why we need it, which files are involved, and exactly how to execute it.

Do not just give me a final code dump. Guide me through the actual implementation, execution, debugging, validation, and preparation of submission evidence.

## 2. Files I have uploaded

You have three source files:

1. `CS203_Assignment1.pdf` — the official assignment brief.
2. `proposal.pdf` — our AgentEvalOps project proposal.
3. `AgentEvalOps-main.zip` — the complete ZIP of my current GitHub repository.

**Read all three files thoroughly before proposing any changes.**

Use the official assignment PDF as the primary authority for grading requirements. Use the proposal to understand our intended project design. Use the repository as the source of truth for what has actually been implemented.

If the proposal and assignment differ, identify the difference explicitly and follow the official assignment requirements unless the instructor has specified otherwise. Do not silently substitute one requirement for another.

## 3. Current progress and scope

Our understanding of the current progress is:

- **Part A — Collection and Instrumentation:** Completed.
- **Part B — Data Validation:** Completed.
- **Part C — Labelling and Annotator Agreement:** Labeling is done. The remaining work is annotator agreement and its supporting analysis and evidence.
- **Part D — Augmentation and Active Learning:** Not yet completed.
- **Part E — Versioning and Splits:** Not yet completed.
- **Part F — Report and Reproducibility:** Not yet completed.

Treat this as my initial progress statement, not as proof that every component is correct. Inspect the repository and verify the actual state before deciding what needs to be done.

Do not unnecessarily rewrite Parts A and B. If you discover a genuine issue in them that blocks the remaining work or threatens assignment compliance, explain the evidence and propose the smallest necessary correction.

## 4. First task: Thoroughly audit the repository

Before writing code or changing files, inspect the entire repository ZIP and construct a clear picture of the existing project.

Identify:

- The repository's folder structure and the purpose of important files.
- The task suite, task templates, generated instances, agent implementation, typed tools, and trace collection pipeline.
- The raw trajectories and their schemas.
- The current labeling guidelines, Label Studio configuration, annotation exports, and any scripts related to labeling.
- Existing validation code, tests, outputs, and reports.
- Any existing augmentation, active-learning, splitting, or DVC implementation.
- The current README, dependency files, environment documentation, and report files.
- Any temporary files, generated artifacts, missing files, broken paths, or duplicate implementations that matter to the remaining assignment.

Read the actual source code and data schemas where possible. Do not rely only on filenames or the README.

Create a status table with these columns:

| Requirement | Current status | Evidence in repository | Remaining work | Priority |
|---|---|---|---|---|

Classify every relevant requirement as:
- Complete and verified.
- Partially complete.
- Missing.
- Present but incorrect or unverified.

Distinguish implementation from evidence. For example, the existence of a Python script does not prove that it runs successfully or produces the required output.

At the end of the audit, present the exact implementation order you recommend. Identify dependencies between tasks so that we do not waste time doing work that must later be redone.

**Do not start modifying the project until you have explained your findings and proposed the implementation plan.**

## 5. Part C: Complete annotator agreement

Our trajectories have already been labeled in Label Studio. The remaining work is to analyze the independent annotations and produce defensible agreement results.

First, inspect the actual Label Studio exports and determine:

1. How many trajectories were labeled.
2. How many independent annotators participated.
3. Which items were labeled by each annotator.
4. How much overlap exists between their annotations.
5. Whether both annotators labeled the same trajectories independently.
6. Whether the exported JSON has the correct schema and contains valid, non-empty annotation results.
7. Whether each annotation has the required fields:
   - `outcome`
   - `primary_failure_category`
   - `first_error_step`
8. Whether values are valid and whether missing, duplicate, or conflicting annotations exist.

Do not confuse multiple annotations or repeated task identifiers with independent annotators. Establish annotator identity and trajectory identity using the actual export structure.

### C1. Calculate agreement correctly

Implement a reproducible Python analysis for the overlapping annotations.

At minimum:

- Calculate Cohen's kappa for `outcome`.
- Calculate Cohen's kappa for `primary_failure_category`.
- Analyze `first_error_step` appropriately. Explain that this is a numerical field and that exact agreement, absolute step difference, and/or a justified weighted agreement measure may be useful alongside categorical kappa.
- Generate confusion matrices for the categorical fields.
- Report the number of overlapping items, raw agreement, observed disagreements, and kappa for each applicable field.
- Handle missing labels and invalid values explicitly.
- Explain the limitations of kappa when categories are imbalanced or annotators rarely use certain categories.

Use a reliable library such as scikit-learn where appropriate. Do not calculate agreement by comparing rows in arbitrary export order. Match annotations by a stable trajectory identifier and annotator identity.

If only one annotator's export exists, do not pretend that inter-annotator agreement can be calculated. Tell me precisely what additional independent annotations are needed.

### C2. Resolve the annotation and agreement questions

Check the current labels against the underlying trajectories wherever required. Investigate inconsistencies between an existing task's stored outcome and its human annotation, but do not automatically assume either is correct.

For example, a trajectory whose stored outcome says `failure` but whose human annotation says `success` must be reviewed against the trace and the written guidelines. Do not silently overwrite human labels with the stored outcome.

Identify disagreements and prepare a table containing:

- Trajectory ID.
- Annotator A's label.
- Annotator B's label.
- The fields that disagree.
- The relevant trace evidence.
- The proposed resolution, with justification.
- Whether the decision requires human confirmation.

The assignment requires detailed discussion of at least 10 disagreements. If fewer than 10 genuine disagreements exist, report the actual number and do not fabricate examples.

### C3. Revise the guidelines and evaluate automatic detectors

Inspect the existing annotation taxonomy and guidelines.

- Produce a revised guideline version, v2, based on actual disagreements.
- Preserve the original v1 and the original annotation exports.
- Document changes between v1 and v2.
- Record the adjudication process and who made the final decision.
- Compare simple automatic detectors for loops and malformed tool calls against human labels.
- Calculate suitable classification metrics, confusion matrices, and error examples.

Be precise about which results are pre-revision and post-revision. Do not claim that agreement improved after guideline revision unless a comparable second annotation round has actually been completed. If a second independent round is necessary, help me conduct it properly.

### C4. Required deliverables

Determine which of these are missing and create or update them as needed:

- Reproducible agreement-analysis script or notebook.
- Agreement results in CSV or JSON.
- Confusion matrices and plots.
- A disagreement review table covering at least 10 genuine disagreements when available.
- Revised guidelines v2 and a record of changes.
- Detector evaluation results.
- Supporting evidence and a clear description of the annotation rounds.

Use the official assignment requirements to decide the exact deliverables and their locations.

## 6. Part D: Augmentation and active learning

Implement Part D using the real AgentEvalOps task suite and trajectory structure. Do not build an unrelated toy project just to produce a plot.

### D1. Augmentation

The official project-specific requirements call for:

1. Task variants with changed entities and parameters.
2. Fault injection, including tool failure, slow response, and changed output format.
3. Prompt paraphrases.

Inspect the current tool layer, task templates, trace schema, and fault-injection support before deciding what to implement.

For each augmentation type:

- Explain the concept in simple terms.
- Identify the source data and output data.
- Implement the transformation in code.
- Preserve the original sample and its provenance.
- Record the augmentation type, parameters, seed where applicable, and parent sample ID.
- Validate that the intended label or expected behavior remains valid.
- For fault-injected cases, distinguish deliberate injected faults from naturally occurring agent failures.
- Run the code and show real outputs and representative examples.

Important: not every transformation is label-preserving in the same way. Changing entities or parameters requires updating the gold answer and validating it. Fault injection may intentionally change the expected execution behavior. Prompt paraphrases should preserve task intent, but that should be checked rather than assumed.

Implement at least three meaningful augmentation strategies, covering the project-specific requirements. Demonstrate label preservation or the expected behavior change with rule-based checks or human review, as appropriate.

Do not count a transformation as completed merely because its code exists. Execute it and save the supporting evidence.

### D2. Active learning versus random selection

Implement an actual, reproducible comparison between active-learning selection and random selection.

The assignment specifically suggests informativeness signals such as:

- Trajectory step count.
- Repeated tool calls.
- Tool or execution errors.

Use the available signals supported by our real trace data. Explain any scoring formula and its limitations.

Requirements:

1. Identify a pool of candidate trajectories.
2. Define the labeling budget and at least three budget sizes.
3. Select trajectories using an informativeness-based strategy.
4. Select trajectories randomly using a fixed seed as the baseline.
5. Keep the comparison fair: use the same candidate pool, budget, label availability assumptions, and evaluation criteria.
6. Calculate failure coverage for each selection strategy at each budget.
7. Plot a learning-efficiency or failure-coverage curve.
8. Explain the findings, limitations, and what the plot actually demonstrates.

Use human labels as ground truth only where those labels are available. Be explicit about the limited labeled sample and how you measure coverage on any larger pool. Do not pretend that unlabeled trajectories have known failure labels.

If you need repeated random trials to make the comparison more robust, implement them with fixed seeds and report variability. If the available labels are insufficient for a meaningful experiment, explain the limitation and implement the strongest honest evaluation the data supports.

### D3. Part D deliverables

Provide:

- Augmentation implementation code.
- Generated augmented samples or traces.
- Label-preservation and fault-injection checks.
- Active-learning and random-selection scripts.
- Selection manifests for each budget.
- Metrics in CSV or JSON.
- Plot files and their underlying data.
- A concise findings summary suitable for the final report.

## 7. Part E: DVC versioning and leakage-free splits

Implement Part E based on the actual task-template structure and assignment requirements.

### E1. Split design

The official brief requires holding out whole template families for the test set, rather than merely splitting individual instances. The shifted split must consist of fault-injected tasks.

Inspect the real task IDs, template identifiers, trajectory relationships, and augmentation metadata. Determine the correct grouping key.

Create train, validation, test, and shifted-split manifests.

Ensure that:

- No template family appears in both train and test.
- Repeated runs of the same task do not leak across splits.
- Parent tasks and augmented variants cannot leak across the relevant splits.
- The shifted split is built from fault-injected tasks.
- The splits are reproducible using fixed seeds.
- All samples in the manifests actually exist.
- Every intended sample is accounted for, or exclusions are explicitly documented.

Do not simply use a random row-level split. Explain why the chosen grouping prevents leakage.

If the dataset structure makes one of these requirements impossible to satisfy as written, explain why and propose the smallest defensible solution.

### E2. DVC versioning

Set up or complete DVC versioning for the relevant raw tasks, traces, processed data, and labels.

- Inspect the existing Git and DVC configuration first.
- Reuse existing configuration where valid.
- Track the appropriate data artifacts.
- Configure a local or other supported DVC remote if needed.
- Record the commands required to reproduce the workflow.
- Explain which files are tracked by Git and which are tracked by DVC.
- Ensure that manifests and data correspond to the same final version.

Do not invent successful remote pushes, commits, tags, or version identifiers. Run commands where possible and distinguish verified results from commands I still need to execute.

### E3. Final freeze

The official assignment requires a repository tag named `a1-final`.

Prepare a clear freeze procedure covering:

- Final data and label validation.
- Final split validation.
- DVC status and tracking.
- Git status and commit.
- Creation of the `a1-final` tag.
- Verification that the tag points to the intended final commit.
- Preservation of the gold test set and its checksum.

Do not create or overwrite the final tag prematurely. Give me the commands and explain when it is safe to run them.

## 8. Part F: Report and reproducibility

Part F must document what we actually implemented and how another person can reproduce it.

Create or update the following deliverables according to the official assignment:

### F1. Dataset card

Document:

- Project purpose and intended use.
- Data sources and collection method.
- Dataset size and structure.
- Task and template coverage.
- Outcome and failure-category distributions.
- Known biases and limitations.
- Labeling process and agreement findings.
- Split definitions.
- Known limitations of the available sample.

Compute all counts and statistics from the actual data. Never invent numbers to make the project look more complete.

### F2. Environment and dependencies

Prepare:

- `requirements.txt` with pinned package versions.
- `requirements_freeze.txt` containing the actual environment's package freeze or equivalent export.
- `ENVIRONMENT.md` documenting the actual Python version, operating system, relevant tool versions, Label Studio version, DVC version, model names and exact versions where recoverable, and random seeds.

Distinguish observed versions from unavailable historical information. Do not claim to know the exact version of a model or package if the repository does not record it.

Do not blindly replace working dependency files with incompatible versions. Inspect the existing environment and test changes.

### F3. Label reproducibility

Create `LABELLING_PROTOCOL.md` documenting:

- Anonymized annotator IDs.
- Which items each annotator labeled in each round.
- Label Studio version and project configuration.
- Guidelines v1 and v2.
- Label-level order: outcome, primary failure category, and first-error step.
- Item presentation order and shuffle seed, if applicable.
- Whether annotators could see predictions, pre-annotations, or each other's labels.
- How disagreements were resolved and who made the final decision.
- Dates of each round and the corresponding export filenames.

Use evidence from the exports and repository. Clearly mark any details that require me or another team member to confirm.

### F4. README and handoff

Update the repository README so a new person can reproduce the data pipeline in at most five commands, as required by the assignment.

Include the correct setup instructions, working-directory assumptions, relative paths, expected outputs, and links or references to relevant documentation.

Create or update `HANDOFF_A2.md`, documenting:

- The frozen dataset and labels.
- The split definitions.
- The gold test set and checksum.
- The label schema.
- Relevant DVC and Git identifiers.
- Any known limitations and instructions for Assignment 2.

Verify that the README commands correspond to actual scripts and that their output paths match the repository structure.

### F5. Final report

Prepare the evidence and content needed for `report/report.pdf`, with a maximum of six pages plus appendices.

The report must cover Parts A–F, including:

- Methods and validation findings.
- Annotator agreement and disagreements.
- Augmentation and active-learning results.
- Split design and leakage checks.
- Dataset card summary.
- Software versions and labeling protocol.
- Contribution table.
- Appendix indexing each reported number and finding to the file that supports it.
- Appendix describing AI assistance.

Do not write unsupported claims. Every reported number, plot, and conclusion must be traceable to an actual file or executed experiment.

If the report cannot be finalized until I execute something or confirm a fact, prepare a clearly marked draft and tell me exactly what remains unresolved.

## 9. Strict coding and repository rules

Follow these rules throughout the work:

1. **Preserve existing working code.** Do not rewrite Parts A and B without a demonstrated need.
2. Do not replace the repository with a simplified example.
3. Do not delete raw data, original annotation exports, existing evidence, or working scripts merely to make the structure cleaner.
4. Preserve original files when correcting exports. Save corrected versions separately and include the correction script where applicable.
5. Use relative paths and portable code. Avoid hard-coded paths specific to my machine.
6. Keep random seeds explicit and record them.
7. Validate input schemas before processing data.
8. Handle errors clearly.
9. Avoid adding unnecessary dependencies.
10. Do not fabricate annotation results, experiment results, model versions, plots, or execution logs.
11. Never claim that code works without actually running it or clearly identifying it as untested.
12. Follow the repository's existing coding style and naming conventions where reasonable.
13. Maintain traceability from raw data to processed data, labels, augmented data, split manifests, and report figures.
14. Do not expose personal information in the final submission.
15. Preserve the project-specific meaning of `outcome`, `primary_failure_category`, and `first_error_step`.

For each file you change or create, explain:
- Its purpose.
- Why it is needed.
- What changed.
- How to run it.
- What output it should produce.
- How to verify the output.

## 10. How I want you to guide me

Work interactively, in manageable stages.

For each stage:

1. Explain the objective in simple English.
2. Identify the exact files involved.
3. Give the complete code for any new or modified file, not just isolated fragments.
4. Tell me the exact terminal commands to run, including the correct working directory.
5. Explain what successful output looks like.
6. Tell me how to verify correctness.
7. If an error occurs, help debug the actual error rather than jumping to an unrelated rewrite.
8. Save or generate the required evidence and explain where it belongs.
9. Update the checklist of requirements completed and still pending.

When possible, inspect the files yourself and produce complete ready-to-use scripts or documents. If you cannot modify my uploaded ZIP directly, provide the complete file contents and exact paths, or generate replacement files that I can download and put into the repository.

If an action requires my local environment, credentials, Label Studio access, a second annotator, or a manual decision, explain exactly what I need to do. Never pretend you have performed an action that you cannot perform.

Do not dump the entire implementation in one enormous response. Complete one logical stage at a time, verify it, and then proceed to the next stage. You may prepare related files together when that is more efficient, but keep execution and verification steps clear.

## 11. Submission requirements and final quality audit

Before declaring the assignment ready, cross-check the implementation against the official PDF, especially the project-specific requirements for AgentEvalOps.

Prepare a final checklist covering:

- Part C: independent annotations, agreement statistics, disagreement analysis, guideline revision, detector evaluation, and supporting files.
- Part D: at least three meaningful augmentations, validation of label preservation or expected behavior, active learning versus random selection, at least three budgets, failure-coverage results, plots, and underlying data.
- Part E: DVC versioning, leakage-free template-family splits, shifted fault-injected split, gold test-set freeze, and `a1-final` tag.
- Part F: dataset card, clean repository, README reproduction in at most five commands, pinned dependencies, environment record, labeling protocol, handoff document, and report with supporting evidence.
- Overall: no broken paths, no unsupported claims, no missing evidence, and consistency between code, data, DVC, Git, and report.

Also inspect the submission packaging requirements in the official assignment. Help me prepare the required `code/`, `data/`, `supporting_files/`, and `report/` structure without destroying my existing repository or its Git history.

Perform a final traceability audit: every reported metric must match its supporting artifact, every submitted dataset file must be accounted for by the pipeline and versioning, and all required AI-assistance prompts must be documented as the assignment requests.

Do not mark a requirement complete merely because a file exists.

## 12. Your first response

Start by reading all three uploaded files and auditing the repository.

Your first response must contain:

1. A concise summary of the project and the official requirements for the remaining parts.
2. A repository-grounded progress table showing what is actually complete, incomplete, or uncertain.
3. Any critical issues that could block completion.
4. A prioritized step-by-step implementation plan.
5. The first concrete action we should take.

Do not immediately start rewriting files. First establish what is actually present, what is missing, and what must be done to produce a correct, reproducible submission.

Our goal is to **complete the remaining assignment rigorously, preserve the work already done, understand every important step, and produce a submission backed by real code and evidence.** Guide me step by step tog et everything dne
````

**Output went into:** the audit and the plan, given as a chat reply; no files.

### Prompt 2
```text
answer: 1: yes, 2: yes, 3: yes, 4: No
```
**Output went into:** `code/annotation/agreement.py`, `tests/test_agreement.py`, `supporting_files/agreement/round1/*` (stage C1).

### Prompt 3
```text
[terminal output pasted: unzip of stage C1, `python` not found, pytest plugin error]
curent terminal results what to do next?
```
**Output went into:** chat instructions (virtual environment, `-p no:debugging`); `pyproject.toml` `addopts` (edited by the user).

### Prompt 4
```text
[terminal output pasted: 6 passed; full suite 314 passed]
what to do next? cmmit? or what?
```
**Output went into:** `code/tasks/grader_v2.py`, `tests/test_grader_v2.py`, `data/processed/regrade_v2.csv`, `supporting_files/grader_v2/regrade_summary.json`, `supporting_files/agreement/round1/adjudication_round1.csv` (draft), `code/annotation/build_labels.py`, `code/annotation/compare_human_grader.py`, `supporting_files/agreement/round1/human_vs_grader*`, `docs/annotation_guidelines_v2.md`, `code/annotation/label_studio_config_v2.xml` (stage C2). After the user confirmed the next step: `code/annotation/sample_round2.py`, `tests/test_sample_round2.py`, `data/labels/round2_sample.json`, `data/labels/round2_manifest.csv` (stage C3a).

### Prompt 5
```text
[attachment: excerpt "Step 2: add the sample and commit it before anyone labels"]
i have completed till step2 and now for the enxt step 3 , the team is not togetrh and he is using a spreate pc somehwere else so i need to first commit all on github and then call him and gide him on  waht to dom so hn do htis, help me commit all reent chnages on github for him to pull and then guide me and him step by step of what to doexactly without any confusion
```
**Output went into:** `docs/ROUND2_ANNOTATOR_INSTRUCTIONS.md`.

### Prompt 6
```text
how to label and push for me he will do ti on his part, for me guide me i am A2
```
**Output went into:** chat instructions only. The round-2 labels themselves were made by the human annotators without AI help (`supporting_files/labels/round2/*`).

### Prompts 7–8
```text
[terminal output pasted: export copy, `python` not found]
[terminal output pasted: 40 40, commit and push of annotator2_raw.json]
```
**Output went into:** chat instructions; then stage D1: `code/augmentation/*`, `tests/test_augmentation.py`, `data/augmented/*`, `supporting_files/augmentation/*`.

### Prompt 9
```text
[terminal output pasted: run_augmented --dry-run "No configured free-tier models found"]
```
**Output went into:** chat instructions only.

### Prompt 10
```text
do this part without involving api keys
```
**Output went into:** `code/active_learning/al_vs_random.py`, `tests/test_active_learning.py`, `data/active_learning/*`, `supporting_files/active_learning/*` (stage D2).

### Prompt 11
```text
C1, C2, C3a done, ran all the D1 and D2 commits without error. my temaater has pushed his annotator roudn 2 file accordngly, i ave als pulled it. i have also completed the paraphrase spot check. next what? i can attached both the annotoart= files stored in supporting_files/lables/round2 and also what to do about paraphrase_spot)check..csv, its chnagesa re yet to be saved or pushed. guide accoridngly
```
**Output went into:** chat instructions.

### Prompt 12
```text
[attachment: for_claude_C3b.zip; terminal output pasted: zip and git log]
```
**Output went into:** `supporting_files/agreement/round2/*` (agreement), `supporting_files/agreement/round2/adjudication_round2.csv` (draft), `code/annotation/detectors.py`, `code/annotation/compare_rounds.py`, `tests/test_detectors.py`, `supporting_files/agreement/pre_post_comparison.*`, `docs/annotation_findings.md` (stage C3b).

### Prompt 13
```text
[terminal output pasted]
done, start part E, ADJ1 is A2 acting as adjusdicator
```
**Output went into:** `code/splits/make_splits.py`, `tests/test_splits.py`, `code/validation/run_validation_evidence.py`, `dvc.yaml`, `scripts/setup_dvc.sh`, `data/splits/*`, `supporting_files/splits/*`, `supporting_files/validation/*` (stage E).

### Prompts 14–16
```text
[terminal output pasted: make_splits FileNotFoundError data/labels/labels_round2.csv]
[terminal output pasted: build_labels "2 disagreements have no adjudication row"]
[terminal output pasted: labels built, detectors, AL, splits 10 PASS, 374 passed]
done, waht next?
```
**Output went into:** chat instructions (repair of two spreadsheet-altered `run_id` cells, described in `LABELLING_PROTOCOL.md` §8).

### Prompt 17
```text
[attachment: for_claude_F.zip; terminal output pasted: DVC commit and push, gold checksums]
team roll no:
Kshitij Giri 23110177
Shardul Junagade 23110297
Vedant Sharan 23110354

url: https://github.com/VedantSharan/AgentEvalOps.git

contribution skip, do not add

for the answer of 3rd, keep it a space of doubt as of now vedant will adres it later

so let us complete F now.
```
**Output went into:** `requirements.txt`, `requirements_freeze.txt` (the user's own `pip freeze`), `ENVIRONMENT.md`, `LABELLING_PROTOCOL.md`, `docs/DATASET_CARD.md`, `code/reporting/*`, `tests/test_prompts.py`, `prompts/*`, `README.md`, `HANDOFF_A2.md`, `report/*`, `scripts/package_submission.sh`, and the updated `docs/annotation_findings.md` (stage F).

---

**Human decisions not delegated to the assistant:**
- All round-2 labels (A1, A2).
- Confirmation or change of every adjudication (ADJ1; 2 of 19 round-2 proposals were changed).
- The paraphrase spot check (10 items).
- The decision to run no API calls.
- The choice of what to commit.
