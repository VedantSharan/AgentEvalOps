# Other AI Assistants

The assignment brief requires every prompt given to AI assistants (such as ChatGPT, Claude, Copilot, or IDE-integrated agents) used to automate any part of the project, with the model version, date, and the specific files the outputs went into.

- **Claude session (Parts C–F):** Documented in [`claude_session_2026-10-22.md`](claude_session_2026-10-22.md) (used by Kshitij Giri).
- **Antigravity IDE / Gemini session (Parts A, B, and Infrastructure):** Documented in detail in [`gemini_antigravity_sessions.md`](gemini_antigravity_sessions.md) (used by Vedant Sharan).

Below is the complete summary register of AI assistance used across the project:

| Date | Assistant / Model | Team Member | Prompt Reference / Focus | Output Went Into |
|---|---|---|---|---|
| 2026-10-19 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 1: T01 Shared data models, schemas, config hashing | `code/shared/models.py`, `tests/test_models.py`, `supporting_files/SCHEMAS.md`, `pyproject.toml` |
| 2026-10-19 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 2: T02 Deterministic task generator (24 tasks, 6 families) & grader | `code/tasks/fixtures.py`, `code/tasks/generator.py`, `code/tasks/grader.py`, `data/raw/tasks.jsonl`, `tests/test_tasks.py` |
| 2026-10-19 to 2026-10-20 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 3: T03 Tool-using agent loop, typed tool registry, trajectory recorder | `code/agent/tools.py`, `code/agent/runner.py`, `code/agent/trajectory.py`, `data/raw/trajectories.jsonl`, `tests/test_agent.py` |
| 2026-10-20 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 4: T04 MLflow tracing integration | `code/tracing/mlflow_tracer.py`, `tests/test_tracing.py` |
| 2026-10-20 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 5: T05 Dataset validation suite (10 automated checks) | `code/validation/dataset_validator.py`, `tests/test_validation.py`, `docs/VALIDATION_REPORT.md`, `data/corrupted/*` |
| 2026-10-21 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 6: Benchmark task suite substantive revision (30 tasks across 6 families) | `code/tasks/revised_fixtures.py`, `code/tasks/revised_generator.py`, `data/processed/tasks_revised.jsonl`, `tests/test_revised_benchmark.py` |
| 2026-10-21 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 7: Persistent multi-provider trajectory worker & SQLite work queue | `code/experiments/work_queue.py`, `code/experiments/providers.py`, `code/experiments/agent_executor.py`, `code/experiments/trajectory_worker.py`, `tests/test_trajectory_pipeline.py` |
| 2026-10-21 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 8: Multi-provider collection run (154 runs), stratified sample, Label Studio XML | `data/experiments/multi_provider_run/trajectories.jsonl`, `code/annotation/sample_for_labelling.py`, `data/labels/stratified_sample_45.jsonl`, `code/annotation/label_studio_config.xml` |
| 2026-10-21 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 9: Forensic audit of 154-run collection | `temp/AUDIT_REPORT.md` |
| 2026-10-21 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 10: Collection pipeline remediation (inner-loop cap, provider error decoupling) | `temp/REMEDIATION_REPORT.md`, `temp/test_audit_remediation.py`, `code/experiments/trajectory_worker.py` |
| 2026-10-21 to 2026-10-22 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 11: MLflow telemetry trace ingestion (`export_to_mlflow.py`) | `code/tracing/export_to_mlflow.py`, `tests/test_mlflow_export.py`, `README.md`, `HANDOFF_A2.md`, `report/report.md` |
| 2026-10-22 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 12: Part A provenance metadata gap resolution (100% sample coverage) | `code/experiments/generate_metadata.py`, `data/metadata.csv`, `tests/test_metadata.py`, `HANDOFF_A2.md`, `docs/DATASET_CARD.md`, `report/report.md` |
| 2026-10-22 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 13: Task execution coverage reframing & reserve justification | `docs/DATASET_CARD.md`, `report/report.md`, `HANDOFF_A2.md` |
| 2026-10-22 | Gemini 3.8 Flash (High) / Antigravity IDE | Vedant Sharan | Prompt 14: Environment & labelling protocol confirmations | `ENVIRONMENT.md`, `LABELLING_PROTOCOL.md` |
| 2026-10-22 | Claude Opus 5.5 / claude.ai | Kshitij Giri | Prompts 1–17: Agreement analysis, grader v2, adjudication, guidelines v2, AL vs random, splits, DVC, report | See `claude_session_2026-10-22.md` |

### Statement on Additional Tools
No other AI assistants (e.g., ChatGPT, GitHub Copilot, Cursor) were used by any team member during the completion of this assignment. All AI-assisted workflows are fully captured in [`claude_session_2026-10-22.md`](claude_session_2026-10-22.md) and [`gemini_antigravity_sessions.md`](gemini_antigravity_sessions.md).
