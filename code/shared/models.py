"""
Shared typed data models for AgentEvalOps.

Covers the five core schemas agreed upon in CONTEXT.md §8 and SCHEMAS.md:
  - TaskRecord
  - ToolResult
  - TrajectoryStep
  - TrajectoryRecord
  - ExecutionConfig  (with deterministic SHA-256 hash)

All models are JSON-serializable via .model_dump() / .model_dump_json() and
round-trippable via model_validate() / model_validate_json().
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EvaluationRule(str, Enum):
    """Permitted evaluation strategies for a task."""

    exact_match = "exact_match"
    numeric_tolerance = "numeric_tolerance"
    structured_match = "structured_match"
    unanswerable = "unanswerable"


class ActionType(str, Enum):
    """Action kinds that can appear in a trajectory step."""

    tool_call = "tool_call"
    llm_response = "llm_response"
    submit_answer = "submit_answer"


class Outcome(str, Enum):
    """Terminal outcome of a trajectory."""

    success = "success"
    failure = "failure"
    error = "error"


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------


class TaskMetadata(BaseModel):
    """Provenance information attached to a task."""

    provenance: str = Field(..., description="Source or generator of the task.")
    created_at: str = Field(..., description="ISO-8601 creation timestamp.")
    generator_version: str = Field(..., description="Version of the generator script.")
    seed: int | None = Field(None, description="Random seed used during generation.")
    content_hash: str | None = Field(
        None, description="SHA-256 hex digest of the canonical task content."
    )

    model_config = {"extra": "forbid"}


class TaskRecord(BaseModel):
    """A single evaluation task."""

    task_id: str = Field(..., description="Stable unique identifier.")
    template_family: str = Field(
        ..., description="Family identifier used for leakage-free splitting."
    )
    task_type: str = Field(..., description="Operation or question category.")
    prompt: str = Field(..., description="Instruction given to the agent.")
    answerable: bool = Field(
        ..., description="Whether sufficient information exists to answer."
    )
    gold_answer: str | None = Field(
        None,
        description="Expected answer; null for unanswerable tasks.",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tool names expected to be useful for this task.",
    )
    evaluation_rule: EvaluationRule = Field(
        ..., description="Rule used to grade the agent's final answer."
    )
    metadata: TaskMetadata | None = Field(
        None, description="Provenance, timestamps, and content hash."
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_answerability_consistency(self) -> "TaskRecord":
        """
        Enforce the constraint between `answerable`, `gold_answer`, and
        `evaluation_rule`:
          - Unanswerable tasks must have evaluation_rule == 'unanswerable'
            and gold_answer == None.
          - Answerable tasks must NOT use evaluation_rule == 'unanswerable'
            and must provide a non-None gold_answer.
        """
        if not self.answerable:
            if self.gold_answer is not None:
                raise ValueError(
                    "Unanswerable tasks must have gold_answer=None, "
                    f"got {self.gold_answer!r}."
                )
            if self.evaluation_rule != EvaluationRule.unanswerable:
                raise ValueError(
                    "Unanswerable tasks must use evaluation_rule='unanswerable', "
                    f"got {self.evaluation_rule!r}."
                )
        else:
            if self.evaluation_rule == EvaluationRule.unanswerable:
                raise ValueError(
                    "Answerable tasks must not use evaluation_rule='unanswerable'."
                )
            if self.gold_answer is None:
                raise ValueError(
                    "Answerable tasks must provide a non-None gold_answer."
                )
        return self


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Structured result returned by a tool invocation."""

    tool_name: str = Field(..., description="Name of the tool that was called.")
    success: bool = Field(..., description="Whether the invocation succeeded.")
    output: Any = Field(
        None, description="JSON-serializable tool output; None on failure."
    )
    error: str | None = Field(
        None, description="Error message; None on success."
    )
    latency_ms: float | None = Field(
        None,
        ge=0.0,
        description="Wall-clock duration of the tool call in milliseconds.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_error_consistency(self) -> "ToolResult":
        """
        Successful results must not carry an error message.
        Failed results must carry an error message.
        """
        if self.success and self.error is not None:
            raise ValueError(
                "Successful ToolResult must not have an error message, "
                f"got {self.error!r}."
            )
        if not self.success and self.error is None:
            raise ValueError(
                "Failed ToolResult must include an error message."
            )
        return self


# ---------------------------------------------------------------------------
# TrajectoryStep
# ---------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """One step in an agent execution trajectory."""

    step_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this step in the trajectory.",
    )
    action_type: ActionType = Field(..., description="Kind of action taken.")
    tool_name: str | None = Field(
        None,
        description="Name of the tool called; required when action_type='tool_call'.",
    )
    arguments: dict[str, Any] | None = Field(
        None, description="Arguments passed to the tool."
    )
    tool_result: ToolResult | None = Field(
        None, description="Result of the tool call, if applicable."
    )
    llm_output: str | None = Field(
        None, description="Raw text produced by the LLM, if applicable."
    )
    error: str | None = Field(None, description="Step-level error, if any.")
    latency_ms: float | None = Field(
        None,
        ge=0.0,
        description="Wall-clock duration of this step in milliseconds.",
    )
    prompt_tokens: int | None = Field(
        None,
        ge=0,
        description="Prompt token count for this step's LLM call.",
    )
    completion_tokens: int | None = Field(
        None,
        ge=0,
        description="Completion token count for this step's LLM call.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_tool_call_fields(self) -> "TrajectoryStep":
        """tool_name must be set for tool_call steps."""
        if self.action_type == ActionType.tool_call and self.tool_name is None:
            raise ValueError(
                "TrajectoryStep with action_type='tool_call' must include tool_name."
            )
        return self


# ---------------------------------------------------------------------------
# TrajectoryRecord
# ---------------------------------------------------------------------------


class TrajectoryMetrics(BaseModel):
    """Aggregate metrics over an entire trajectory."""

    total_steps: int = Field(..., ge=0, description="Number of steps executed.")
    total_latency_ms: float | None = Field(
        None, ge=0.0, description="Sum of step latencies in milliseconds."
    )
    total_prompt_tokens: int | None = Field(
        None, ge=0, description="Sum of prompt tokens across all LLM calls."
    )
    total_completion_tokens: int | None = Field(
        None, ge=0, description="Sum of completion tokens across all LLM calls."
    )

    model_config = {"extra": "forbid"}


class TrajectoryRecord(BaseModel):
    """Full execution trajectory for one agent run on one task."""

    run_id: str = Field(..., description="Unique identifier for this run.")
    task_id: str = Field(
        ..., description="Identifier of the task that was executed."
    )
    configuration_hash: str = Field(
        ..., description="SHA-256 hash of the ExecutionConfig used."
    )
    model_version: str = Field(..., description="Model identifier string.")
    prompt_version: str = Field(..., description="Prompt template version tag.")
    steps: list[TrajectoryStep] = Field(
        default_factory=list,
        description="Ordered sequence of execution steps.",
    )
    final_answer: str | None = Field(
        None, description="Answer submitted by the agent."
    )
    outcome: Outcome = Field(..., description="Terminal outcome of this run.")
    metrics: TrajectoryMetrics = Field(..., description="Aggregate run metrics.")
    errors: list[str] = Field(
        default_factory=list, description="Top-level errors for this run."
    )

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    @model_validator(mode="after")
    def _check_step_indices(self) -> "TrajectoryRecord":
        """Step indices must form a contiguous, zero-based sequence."""
        for expected, step in enumerate(self.steps):
            if step.step_index != expected:
                raise ValueError(
                    f"Step at position {expected} has step_index={step.step_index}; "
                    "indices must be zero-based and contiguous."
                )
        return self


# ---------------------------------------------------------------------------
# ExecutionConfig
# ---------------------------------------------------------------------------


class ExecutionConfig(BaseModel):
    """Configuration that governs one agent execution."""

    model_id: str = Field(..., description="Model identifier (e.g. gpt-4o-mini).")
    prompt_version: str = Field(..., description="Prompt template version tag.")
    tool_selection_strategy: str = Field(
        ..., description="Strategy used to select or filter available tools."
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; None when not supported by the model.",
    )
    max_steps: int = Field(..., ge=1, description="Maximum number of agent steps.")
    retry_budget: int = Field(
        ...,
        ge=0,
        description="Number of retries allowed for recoverable tool errors.",
    )

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    def compute_hash(self) -> str:
        """
        Return a deterministic SHA-256 hex digest of this configuration.

        Serialisation is canonical: keys are sorted, no extra whitespace,
        floats and None values are JSON-encoded consistently.  This ensures
        that two ExecutionConfig objects with identical field values always
        produce the same hash regardless of Python dict ordering.
        """
        canonical = json.dumps(
            self.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
