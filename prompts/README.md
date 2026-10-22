# Prompts

The brief requires every prompt used to automate any part of the work with an AI model, exactly as used. There are two kinds.

## 1. Prompts the code sends to models: `model_prompts/`

`python -m code.reporting.export_prompts` exports these from the code itself. `tests/test_prompts.py` fails if the files differ from the code.

| File | Sent by | Used for | Models |
|---|---|---|---|
| `collection_agent/system_instruction.txt` | `code/experiments/agent_executor.py` (constant from `code/experiments/providers.py`) | All 154 trajectories in `data/experiments/multi_provider_run/` (prompt_version `v1`) | The 10 models listed in `ENVIRONMENT.md` §2 |
| `collection_agent/tool_schemas.json` | same, as the `tools` field of every request | same | same |
| `collection_agent/submit_nudge_user_message.txt` | same; sent when the model replies with text instead of a tool call | same | same |
| *(task prompt)* | same; the user message is the task's `prompt` field verbatim | same | same |
| `collection_agent/smoke_test_user_message.txt` | `providers.ProviderClient.smoke_test` (`trajectory_worker --smoke-test`) | Provider checks before collection; produces no dataset records | configured providers |
| `pilot_llm_runner/*` | `code/agent/llm_runner.py` | Adversarial pilot only (`data/adversarial/`); excluded from the splits (`data/splits/exclusions.csv`) | `gemini-3.8-flash` |

The task prompts are the `prompt` fields of `data/processed/tasks_revised.jsonl`; they are not duplicated here.

No model generates labels, grades or augmentations. Paraphrases are rule-based.

## 2. Prompts given to AI assistants: `ai_assistance/`

| File | Assistant | Covers |
|---|---|---|
| `ai_assistance/claude_session_2026-10-22.md` | Claude (claude.ai, model `claude-opus-5-5` as configured; the serving model may differ) | Parts C (agreement, grader v2, adjudication drafts, guidelines v2, round-2 sample, detectors), D, E, F |
| `ai_assistance/gemini_antigravity_sessions.md` | Gemini 3.8 Flash (High) (Google DeepMind Antigravity IDE) | Parts A & B (schemas, task suite, agent loop, tools, MLflow tracing, validation), multi-provider collection pipeline, forensic audit & remediation, telemetry ingestion, Part A metadata provenance fix |
| `ai_assistance/OTHER_TOOLS.md` | Comprehensive AI Assistance Register & Team Statement | Cross-referencing both sessions, verifying no other tools were used |

