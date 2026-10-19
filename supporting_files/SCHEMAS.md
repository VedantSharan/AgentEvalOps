# Shared Data Contracts

This document describes the shared schemas implemented in
`code/shared/models.py`.  All models use Pydantic v2 and enforce strict
validation (`extra='forbid'`).  Every record is JSON-serializable via
`.model_dump_json()` and round-trippable via `.model_validate_json()`.

---

## Enumerations

### `EvaluationRule`
Permitted values for grading a task's final answer.

| Value | Meaning |
|---|---|
| `exact_match` | Answer must match gold exactly (string equality). |
| `numeric_tolerance` | Answer must be within a numeric tolerance of gold. |
| `structured_match` | Answer must satisfy a structured schema comparison. |
| `unanswerable` | Task has no correct answer; agent must refuse to answer. |

### `ActionType`
| Value | Meaning |
|---|---|
| `tool_call` | Step represents a call to a named tool. |
| `llm_response` | Step represents a raw LLM text generation. |
| `submit_answer` | Step represents the agent submitting its final answer. |

### `Outcome`
| Value | Meaning |
|---|---|
| `success` | Agent answered correctly. |
| `failure` | Agent answered incorrectly. |
| `error` | Agent failed with an unrecoverable error. |

---

## `TaskMetadata`

Provenance information optionally embedded in a `TaskRecord`.

| Field | Type | Required | Description |
|---|---|---|---|
| `provenance` | `str` | ✓ | Source or generator of the task. |
| `created_at` | `str` | ✓ | ISO-8601 creation timestamp. |
| `generator_version` | `str` | ✓ | Version of the generator script. |
| `seed` | `int \| None` | — | Random seed used during generation. |
| `content_hash` | `str \| None` | — | SHA-256 hex digest of the canonical task content. |

---

## `TaskRecord`

A single evaluation task.

| Field | Type | Required | Description |
|---|---|---|---|
| `task_id` | `str` | ✓ | Stable unique identifier. |
| `template_family` | `str` | ✓ | Family identifier used for leakage-free splitting. |
| `task_type` | `str` | ✓ | Operation or question category. |
| `prompt` | `str` | ✓ | Instruction given to the agent. |
| `answerable` | `bool` | ✓ | Whether sufficient information exists to answer. |
| `gold_answer` | `str \| None` | — | Expected answer; `null` for unanswerable tasks. |
| `expected_tools` | `list[str]` | — | Tool names expected to be useful (default `[]`). |
| `evaluation_rule` | `EvaluationRule` | ✓ | Rule used to grade the agent's final answer. |
| `metadata` | `TaskMetadata \| None` | — | Provenance, timestamps, and content hash. |

**Cross-field validation rules:**
- When `answerable=False`: `evaluation_rule` must be `unanswerable` and
  `gold_answer` must be `null`.
- When `answerable=True`: `evaluation_rule` must not be `unanswerable` and
  `gold_answer` must not be `null`.

---

## `ToolResult`

Structured result returned by a single tool invocation.

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_name` | `str` | ✓ | Name of the tool that was called. |
| `success` | `bool` | ✓ | Whether the invocation succeeded. |
| `output` | `Any \| None` | — | JSON-serializable output; `null` on failure. |
| `error` | `str \| None` | — | Error message; `null` on success. |
| `latency_ms` | `float \| None` (≥ 0) | — | Wall-clock duration in milliseconds. |

**Validation rules:**
- `success=True` → `error` must be `null`.
- `success=False` → `error` must be a non-null string.
- `latency_ms`, if provided, must be ≥ 0.

---

## `TrajectoryStep`

One step in an agent execution trajectory.

| Field | Type | Required | Description |
|---|---|---|---|
| `step_index` | `int` (≥ 0) | ✓ | Zero-based position in the trajectory. |
| `action_type` | `ActionType` | ✓ | Kind of action taken. |
| `tool_name` | `str \| None` | — | Tool name; required when `action_type='tool_call'`. |
| `arguments` | `dict \| None` | — | Arguments passed to the tool. |
| `tool_result` | `ToolResult \| None` | — | Result of the tool call, if applicable. |
| `llm_output` | `str \| None` | — | Raw text produced by the LLM, if applicable. |
| `error` | `str \| None` | — | Step-level error, if any. |
| `latency_ms` | `float \| None` (≥ 0) | — | Wall-clock duration of this step in milliseconds. |
| `prompt_tokens` | `int \| None` (≥ 0) | — | Prompt token count for this step's LLM call. |
| `completion_tokens` | `int \| None` (≥ 0) | — | Completion token count for this step's LLM call. |

**Validation rules:**
- `step_index` must be ≥ 0.
- `action_type='tool_call'` requires a non-null `tool_name`.
- `latency_ms`, `prompt_tokens`, `completion_tokens`, if provided, must be ≥ 0.
- Unavailable token counts and timings must not be fabricated; use `null`.

---

## `TrajectoryMetrics`

Aggregate metrics over a complete trajectory.

| Field | Type | Required | Description |
|---|---|---|---|
| `total_steps` | `int` (≥ 0) | ✓ | Number of steps executed. |
| `total_latency_ms` | `float \| None` (≥ 0) | — | Sum of step latencies in milliseconds. |
| `total_prompt_tokens` | `int \| None` (≥ 0) | — | Sum of prompt tokens across all LLM calls. |
| `total_completion_tokens` | `int \| None` (≥ 0) | — | Sum of completion tokens across all LLM calls. |

---

## `TrajectoryRecord`

Full execution trajectory for one agent run on one task.

| Field | Type | Required | Description |
|---|---|---|---|
| `run_id` | `str` | ✓ | Unique identifier for this run. |
| `task_id` | `str` | ✓ | Identifier of the task that was executed. |
| `configuration_hash` | `str` | ✓ | SHA-256 hash of the `ExecutionConfig` used. |
| `model_version` | `str` | ✓ | Model identifier string. |
| `prompt_version` | `str` | ✓ | Prompt template version tag. |
| `steps` | `list[TrajectoryStep]` | — | Ordered execution steps (default `[]`). |
| `final_answer` | `str \| None` | — | Answer submitted by the agent. |
| `outcome` | `Outcome` | ✓ | Terminal outcome of this run. |
| `metrics` | `TrajectoryMetrics` | ✓ | Aggregate run metrics. |
| `errors` | `list[str]` | — | Top-level errors for this run (default `[]`). |

**Validation rules:**
- `steps` must have contiguous, zero-based `step_index` values (0, 1, 2, …).

---

## `ExecutionConfig`

Configuration that governs one agent execution.

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `str` | ✓ | Model identifier (e.g. `gpt-4o-mini`). |
| `prompt_version` | `str` | ✓ | Prompt template version tag. |
| `tool_selection_strategy` | `str` | ✓ | Strategy used to select available tools. |
| `temperature` | `float \| None` (0–2) | — | Sampling temperature; `null` when not supported. |
| `max_steps` | `int` (≥ 1) | ✓ | Maximum number of agent steps. |
| `retry_budget` | `int` (≥ 0) | ✓ | Retries allowed for recoverable tool errors. |

**Hashing:**
`ExecutionConfig.compute_hash()` returns a deterministic SHA-256 hex digest
computed from the canonical JSON representation of the configuration (sorted
keys, no extra whitespace).  Two configs with identical field values always
produce the same hash.

---

## General rules

- All records must be JSON-serializable.
- Task generation uses explicit seeds.
- Raw artifacts are never overwritten by processing scripts.
- Gold answers are generated or verified independently of agent responses.
- Schema changes require corresponding code and test changes.
- Do not fabricate token counts or latencies; use `null` if unavailable.
