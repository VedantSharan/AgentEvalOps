# AgentEvalOps — Project Context

## 1. Project Overview

**Project:** AgentEvalOps: A Systematic Evaluation, Optimization, and Monitoring Framework for Tool-Using LLM Agents

**Course:** CS 203 — Software Tools and Techniques in AI, IIT Gandhinagar

The project aims to build a single tool-using LLM agent and treat its execution as a reproducible ML experiment. The system should collect execution traces, evaluate task success and operational efficiency, identify failure modes, support configuration optimization, and eventually monitor degradation.

The project is divided into two assignments:

- **Assignment 1 (A1):** Build and validate the task dataset, agent execution traces, human labels, augmentation experiments, active-learning experiments, data splits, versioning, and reproducibility infrastructure.
- **Assignment 2 (A2):** Use the frozen A1 dataset to compare and optimize agent configurations, deploy the agent, and investigate monitoring and detection of performance degradation.

A1 is the current priority. Do not implement the entire A2 system prematurely.

## 2. Authoritative Project Documents

The repository contains a gitignored `temp/` directory holding the original project proposal and Assignment 1 PDF.

- `temp/proposal.pdf`: Overall project objectives, proposed architecture, research questions, and intended A2 experiments.
- `temp/CS203_Assignment1.pdf`: Detailed A1 requirements, marks, project-specific requirements, submission format, and evaluation criteria.

Read both documents before making substantial implementation decisions. The A1 handout's project-specific requirements take precedence when determining what must be delivered for A1. Do not assume a proposal-level target automatically overrides an explicit A1 requirement.

The `temp/` directory is intentionally gitignored. Never remove the ignore rule, commit these PDFs, or make the code depend on their local paths. All essential implementation instructions and reproducibility information must live in tracked files.

## 3. A1 Scope

A1 consists of six parts:

| Part | Requirement | Marks |
|---|---|---:|
| A | Collection and instrumentation | 15 |
| B | Data validation | 20 |
| C | Labelling and agreement | 25 |
| D | Augmentation and active learning | 20 |
| E | Versioning and splits | 10 |
| F | Report and reproducibility | 10 |

### Part A: Collection and Instrumentation

- Implement a function-calling agent loop directly in Python.
- Implement typed tools for retrieval, calculation, structured data/file operations, and `submit_answer`.
- Build 5–6 task templates expanded to 20–30 instances.
- Include ambiguous or unanswerable tasks.
- Compute gold answers programmatically wherever possible.
- Instrument execution using MLflow Tracing.
- Record task/configuration identifiers, model and prompt versions, LLM calls, tool calls, arguments, outputs, errors, latency, token usage, and final answers.
- Generate sample metadata with provenance, timestamps, collector, tool version, and content hashes.
- Document the collection pipeline and data sources.

### Part B: Data Validation

- Implement at least 10 meaningful automated checks using pytest, Pandera, or Great Expectations.
- Cover trace schemas, types, tool calls and arguments, gold-answer correctness, tool determinism, duplicates, label validity, missing values, prompt leakage, and split leakage.
- Run checks on raw data, fix detected problems, and report before-and-after results.
- Deliberately corrupt data and demonstrate that the checks fail.

### Part C: Labelling and Agreement

- Generate a few hundred execution trajectories where feasible.
- Use Label Studio.
- Have two annotators independently label an overlapping sample of 40–50 trajectories.
- Label each trajectory with:
  1. Outcome.
  2. Primary failure category.
  3. First-error step.
- Failure categories include wrong tool, malformed arguments, hallucinated result, loop/repetition, premature stop, correct process with wrong final answer, and no failure.
- Create guideline versions v1 and v2.
- Calculate Cohen's kappa, create a confusion matrix, investigate at least 10 disagreements, and measure agreement after guideline revision.
- Compare simple automatic loop and malformed-call detectors against human labels.
- Freeze a gold test set that is not used for tuning.

### Part D: Augmentation and Active Learning

Implement three augmentation categories:

1. Task variants with changed entities and parameters.
2. Fault injection: tool failure, slow response, or changed output format.
3. Prompt paraphrases.

Verify label preservation or recompute expected behavior where fault injection changes the outcome.

Compare informativeness-based trajectory selection against random selection using equal labelling budgets. Use signals such as step count, repeated calls, and errors. Report failure coverage at approximately three budget sizes and produce a reproducible plot.

### Part E: Versioning and Splits

- Version raw data, processed data, and labels using DVC.
- Create leakage-free train, validation, and test splits.
- Hold out entire task-template families for the test set.
- Create a separate shifted split consisting of fault-injected tasks.
- Tag the final repository state `a1-final`.

### Part F: Report and Reproducibility

Required artifacts include:

- Dataset card.
- `README.md`, with data-rebuild instructions requiring no more than five commands.
- Pinned `requirements.txt`.
- `requirements_freeze.txt` containing the full environment freeze.
- `ENVIRONMENT.md` documenting software, model versions, hardware where relevant, and seeds.
- `LABELLING_PROTOCOL.md`.
- `HANDOFF_A2.md`.
- A report of at most six pages plus appendices.
- Contribution table, evidence index, and AI-assistance disclosure.
- A two-minute recording of the pipeline running end to end.

The final ZIP must contain `code/`, `data/`, `supporting_files/`, and `report/`, including the Git repository at `a1-final`.

## 4. Implementation Strategy

Available effort is approximately 30 combined person-hours, not 30 hours per member.

Prioritize a small, complete, reproducible implementation over a sophisticated but incomplete system.

Principles:

- Build the end-to-end pipeline early.
- Keep the agent, tools, tasks, tracing, validation, annotation, and experiments modular.
- Automate repetitive work wherever possible.
- Use programmatic gold answers and graders instead of LLM judges when feasible.
- Use deterministic local tools and small datasets to minimize execution cost.
- Prefer simple, testable implementations over elaborate abstractions.
- Parallelize work between the two members after interfaces are agreed upon.
- Preserve evidence for every reported result.
- Never fabricate metrics, logs, labels, test results, or experimental outcomes.
- Document limitations and unavailable metrics honestly.
- Do not spend A1 time implementing the full A2 optimization and monitoring framework.

## 5. Initial Task Families

Use six small, deterministic task families:

1. Information retrieval from a local collection of facts.
2. Multi-step question answering.
3. Numerical calculation.
4. Structured-data operations.
5. Controlled file operations.
6. Ambiguous or unanswerable tasks.

A practical initial allocation is four instances per family, for 24 tasks total.

Generate tasks programmatically using fixed random seeds. Each task must have a stable `task_id`, a `template_family`, a prompt, expected tools where applicable, a gold answer or explicit unanswerable criterion, and an evaluation rule.

Do not manually write dozens of similar records.

## 6. Tool Layer

Implement a small typed tool layer with the following capabilities:

- Retrieval.
- Calculator.
- Structured-data operations.
- Controlled file operations.
- `submit_answer`.

Every tool must have:

- A unique name.
- A documented argument schema.
- Predictable output structure.
- Explicit error handling.
- Unit tests for valid and invalid inputs.

Tools should be deterministic where possible. Avoid arbitrary code execution or unrestricted filesystem access.

## 7. Agent and Trace Design

The agent should implement a straightforward function-calling loop:

1. Receive a task.
2. Send the task and tool definitions to the model.
3. Parse and validate the model's response.
4. Execute the requested tool.
5. Record the call and return the tool result to the model.
6. Repeat until answer submission or a maximum-step limit.
7. Grade the final answer and record the outcome.

Keep model configuration, prompt variants, step limits, and retry budgets configurable.

Every trajectory should be identifiable and traceable to its originating task and configuration. Use MLflow Tracing and preserve machine-readable trace exports for downstream processing.

Do not fabricate token counts or latencies if the selected model interface does not expose them.

## 8. Shared Data Contracts

Agree on the task, tool, trajectory, and configuration schemas before parallel implementation.

A task record should include, at minimum:

- `task_id`
- `template_family`
- `task_type`
- `prompt`
- `gold_answer` or explicit evaluation criteria
- `expected_tools`, where applicable
- `evaluation_rule`

A trajectory should include, at minimum:

- `run_id`
- `task_id`
- `configuration_hash`
- `model_version`
- `prompt_version`
- Ordered execution steps
- Tool-call arguments and results
- Errors and execution metrics
- Final answer
- Success/failure outcome

A configuration should identify the model, prompt variant, tool-selection strategy, temperature, maximum steps, and retry budget as applicable.

Use stable IDs, JSON-serializable records, explicit types, and deterministic hashing.

## 9. Repository Conventions

Use a modular structure similar to:

    code/
      agent/
      tasks/
      tracing/
      validation/
      experiments/
      annotation/
      prompts/
    data/
      raw/
      processed/
      labels/
      splits/
    supporting_files/
    report/
    tests/
    README.md
    requirements.txt
    requirements_freeze.txt
    ENVIRONMENT.md
    LABELLING_PROTOCOL.md
    HANDOFF_A2.md
    CONTEXT.md

The exact structure may evolve, but required assignment filenames and submission artifacts must be preserved.

Additional conventions:

- Use relative paths.
- Keep generated data separate from source code.
- Preserve raw artifacts; use scripts to transform them.
- Pin dependencies and record seeds.
- Keep temporary files, caches, local environments, secrets, and source PDFs out of Git.
- Never hard-code local absolute paths.
- Use tests for deterministic functionality.
- Record exact AI prompts used for implementation and the files they informed, as required by the assignment.
- Ensure each member's Git commits reflect their actual contributions.

## 10. Initial Ticket Plan

### P1-T01: Repository and shared contracts
Create the repository skeleton, environment, schemas, tool interface, and initial tests.

### P1-T02: Task suite and gold answers
Generate 20–30 tasks across 5–6 families, implement deterministic grading, and create metadata.

### P1-T03: Integration smoke tests
Validate task schemas, gold answers, tool interfaces, serialization, and deterministic configuration hashes.

After these tickets, parallelize agent implementation and trace instrumentation against the agreed interfaces.

## 11. Definition of Done for A1

A1 is complete only when:

- All six assignment parts have corresponding implementations and evidence.
- The agent can execute benchmark tasks and produce traceable trajectories.
- At least 10 meaningful validation checks are implemented and tested.
- Human annotation, agreement calculations, disagreement analysis, and automatic-detector comparisons are included.
- Augmentation and active-learning comparisons are reproducible.
- Splits are leakage-free and versioned using DVC.
- The gold test set and shifted split are frozen and documented.
- The repository is tagged `a1-final`.
- Documentation, report, supporting files, and recording are complete.
- The submission ZIP can be extracted into a clean directory and the documented data-rebuild commands work.

## 12. Instructions to AI Coding Agents

When working on this repository:

1. Read this file first.
2. Read `temp/proposal.pdf` and `temp/CS203_Assignment1.pdf` when available before making decisions about assignment requirements.
3. Work only on the requested ticket and its necessary dependencies.
4. Inspect existing code before creating or modifying files.
5. Reuse shared schemas rather than defining incompatible duplicates.
6. Keep changes small and modular.
7. Add or update tests with each implementation.
8. Run relevant tests and report actual results.
9. Never claim an implementation or test succeeded without verifying it.
10. Do not silently omit assignment requirements; record any deferred work explicitly.
11. Do not commit the gitignored PDFs or any secrets.
12. When a requirement is ambiguous, distinguish what the assignment explicitly says from the proposed implementation choice.
13. At the end of each ticket, report files changed, tests executed, results, unresolved issues, and the next dependency.

The priority is to deliver a complete, honest, reproducible A1 pipeline within the available team capacity.