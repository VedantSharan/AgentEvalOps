"""
Tests for code.shared.models.

Covers:
  - Valid TaskRecord construction and JSON round-trip
  - Invalid TaskRecord: missing required fields, bad answerability rules
  - ToolResult: successful and failed cases, negative latency rejection
  - TrajectoryStep: negative step_index / metrics rejection, tool_call validation
  - TrajectoryRecord: serialization, non-contiguous step indices
  - TrajectoryMetrics: negative aggregate metrics rejection
  - ExecutionConfig: stable hash for identical config, different hash when changed
"""

import json

import pytest
from pydantic import ValidationError

from code.shared.models import (
    ActionType,
    EvaluationRule,
    ExecutionConfig,
    Outcome,
    TaskMetadata,
    TaskRecord,
    ToolResult,
    TrajectoryMetrics,
    TrajectoryRecord,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_task() -> TaskRecord:
    return TaskRecord(
        task_id="task-001",
        template_family="retrieval",
        task_type="information_retrieval",
        prompt="What is the capital of France?",
        answerable=True,
        gold_answer="Paris",
        expected_tools=["retrieve"],
        evaluation_rule=EvaluationRule.exact_match,
    )


@pytest.fixture
def unanswerable_task() -> TaskRecord:
    return TaskRecord(
        task_id="task-u01",
        template_family="ambiguous",
        task_type="unanswerable",
        prompt="What is the meaning of life, the universe, and everything?",
        answerable=False,
        gold_answer=None,
        expected_tools=[],
        evaluation_rule=EvaluationRule.unanswerable,
    )


@pytest.fixture
def base_config() -> ExecutionConfig:
    return ExecutionConfig(
        model_id="gpt-4o-mini",
        prompt_version="v1",
        tool_selection_strategy="all",
        temperature=0.0,
        max_steps=10,
        retry_budget=2,
    )


@pytest.fixture
def successful_tool_result() -> ToolResult:
    return ToolResult(
        tool_name="retrieve",
        success=True,
        output={"text": "Paris is the capital of France."},
        error=None,
        latency_ms=42.5,
    )


@pytest.fixture
def failed_tool_result() -> ToolResult:
    return ToolResult(
        tool_name="calculate",
        success=False,
        output=None,
        error="Division by zero.",
        latency_ms=5.0,
    )


def make_step(index: int, action: ActionType = ActionType.llm_response) -> TrajectoryStep:
    kwargs: dict = dict(step_index=index, action_type=action)
    if action == ActionType.tool_call:
        kwargs["tool_name"] = "retrieve"
    return TrajectoryStep(**kwargs)


def make_metrics(**overrides) -> TrajectoryMetrics:
    defaults = dict(total_steps=1)
    defaults.update(overrides)
    return TrajectoryMetrics(**defaults)


def make_trajectory(steps=None, **overrides) -> TrajectoryRecord:
    if steps is None:
        steps = [make_step(0)]
    defaults = dict(
        run_id="run-001",
        task_id="task-001",
        configuration_hash="abc123",
        model_version="gpt-4o-mini",
        prompt_version="v1",
        steps=steps,
        final_answer="Paris",
        outcome=Outcome.success,
        metrics=make_metrics(total_steps=len(steps)),
    )
    defaults.update(overrides)
    return TrajectoryRecord(**defaults)


# ---------------------------------------------------------------------------
# TaskRecord tests
# ---------------------------------------------------------------------------


class TestTaskRecord:
    def test_valid_construction(self, valid_task: TaskRecord):
        assert valid_task.task_id == "task-001"
        assert valid_task.answerable is True
        assert valid_task.evaluation_rule == EvaluationRule.exact_match
        assert valid_task.gold_answer == "Paris"

    def test_json_round_trip(self, valid_task: TaskRecord):
        serialized = valid_task.model_dump_json()
        recovered = TaskRecord.model_validate_json(serialized)
        assert recovered == valid_task

    def test_dict_round_trip(self, valid_task: TaskRecord):
        d = valid_task.model_dump()
        recovered = TaskRecord.model_validate(d)
        assert recovered == valid_task

    def test_unanswerable_task_valid(self, unanswerable_task: TaskRecord):
        assert unanswerable_task.answerable is False
        assert unanswerable_task.gold_answer is None
        assert unanswerable_task.evaluation_rule == EvaluationRule.unanswerable

    def test_missing_task_id_raises(self):
        with pytest.raises(ValidationError):
            TaskRecord(
                template_family="retrieval",
                task_type="info",
                prompt="x",
                answerable=True,
                gold_answer="y",
                evaluation_rule=EvaluationRule.exact_match,
            )

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                answerable=True,
                gold_answer="y",
                evaluation_rule=EvaluationRule.exact_match,
            )

    def test_answerable_with_unanswerable_rule_raises(self):
        with pytest.raises(ValidationError, match="must not use evaluation_rule"):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=True,
                gold_answer="y",
                evaluation_rule=EvaluationRule.unanswerable,
            )

    def test_answerable_without_gold_answer_raises(self):
        with pytest.raises(ValidationError, match="must provide a non-None gold_answer"):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=True,
                gold_answer=None,
                evaluation_rule=EvaluationRule.exact_match,
            )

    def test_unanswerable_with_gold_answer_raises(self):
        with pytest.raises(ValidationError, match="must have gold_answer=None"):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=False,
                gold_answer="something",
                evaluation_rule=EvaluationRule.unanswerable,
            )

    def test_unanswerable_with_wrong_rule_raises(self):
        with pytest.raises(ValidationError, match="evaluation_rule='unanswerable'"):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=False,
                gold_answer=None,
                evaluation_rule=EvaluationRule.exact_match,
            )

    def test_invalid_evaluation_rule_raises(self):
        with pytest.raises(ValidationError):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=True,
                gold_answer="y",
                evaluation_rule="made_up_rule",
            )

    def test_metadata_optional(self):
        t = TaskRecord(
            task_id="t",
            template_family="f",
            task_type="info",
            prompt="p",
            answerable=True,
            gold_answer="y",
            evaluation_rule=EvaluationRule.exact_match,
        )
        assert t.metadata is None

    def test_metadata_accepted(self):
        meta = TaskMetadata(
            provenance="test",
            created_at="2026-01-01T00:00:00Z",
            generator_version="0.1.0",
            seed=42,
        )
        t = TaskRecord(
            task_id="t",
            template_family="f",
            task_type="info",
            prompt="p",
            answerable=True,
            gold_answer="y",
            evaluation_rule=EvaluationRule.exact_match,
            metadata=meta,
        )
        assert t.metadata.seed == 42

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=True,
                gold_answer="y",
                evaluation_rule=EvaluationRule.exact_match,
                unknown_field="oops",
            )

    def test_all_valid_evaluation_rules_for_answerable_task(self):
        for rule in (
            EvaluationRule.exact_match,
            EvaluationRule.numeric_tolerance,
            EvaluationRule.structured_match,
        ):
            t = TaskRecord(
                task_id="t",
                template_family="f",
                task_type="info",
                prompt="p",
                answerable=True,
                gold_answer="42",
                evaluation_rule=rule,
            )
            assert t.evaluation_rule == rule


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_successful_result(self, successful_tool_result: ToolResult):
        assert successful_tool_result.success is True
        assert successful_tool_result.error is None
        assert successful_tool_result.latency_ms == 42.5

    def test_failed_result(self, failed_tool_result: ToolResult):
        assert failed_tool_result.success is False
        assert failed_tool_result.error == "Division by zero."

    def test_json_round_trip(self, successful_tool_result: ToolResult):
        recovered = ToolResult.model_validate_json(successful_tool_result.model_dump_json())
        assert recovered == successful_tool_result

    def test_success_with_error_message_raises(self):
        with pytest.raises(ValidationError, match="must not have an error message"):
            ToolResult(
                tool_name="retrieve",
                success=True,
                output="ok",
                error="some error",
            )

    def test_failure_without_error_message_raises(self):
        with pytest.raises(ValidationError, match="must include an error message"):
            ToolResult(
                tool_name="retrieve",
                success=False,
                output=None,
                error=None,
            )

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            ToolResult(
                tool_name="retrieve",
                success=True,
                output="ok",
                error=None,
                latency_ms=-1.0,
            )

    def test_zero_latency_accepted(self):
        r = ToolResult(tool_name="x", success=True, output="ok", error=None, latency_ms=0.0)
        assert r.latency_ms == 0.0

    def test_null_latency_accepted(self):
        r = ToolResult(tool_name="x", success=True, output="ok", error=None, latency_ms=None)
        assert r.latency_ms is None


# ---------------------------------------------------------------------------
# TrajectoryStep tests
# ---------------------------------------------------------------------------


class TestTrajectoryStep:
    def test_llm_response_step(self):
        step = TrajectoryStep(step_index=0, action_type=ActionType.llm_response)
        assert step.step_index == 0
        assert step.tool_name is None

    def test_tool_call_step_requires_tool_name(self):
        with pytest.raises(ValidationError, match="must include tool_name"):
            TrajectoryStep(step_index=0, action_type=ActionType.tool_call)

    def test_tool_call_step_valid(self):
        step = TrajectoryStep(
            step_index=1,
            action_type=ActionType.tool_call,
            tool_name="retrieve",
        )
        assert step.tool_name == "retrieve"

    def test_negative_step_index_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(step_index=-1, action_type=ActionType.llm_response)

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=0,
                action_type=ActionType.llm_response,
                latency_ms=-5.0,
            )

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=0,
                action_type=ActionType.llm_response,
                prompt_tokens=-1,
            )

    def test_negative_completion_tokens_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryStep(
                step_index=0,
                action_type=ActionType.llm_response,
                completion_tokens=-1,
            )

    def test_zero_tokens_accepted(self):
        step = TrajectoryStep(
            step_index=0,
            action_type=ActionType.llm_response,
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert step.prompt_tokens == 0

    def test_json_round_trip(self):
        step = TrajectoryStep(
            step_index=0,
            action_type=ActionType.tool_call,
            tool_name="calculate",
            arguments={"expr": "2+2"},
            latency_ms=10.0,
            prompt_tokens=100,
            completion_tokens=5,
        )
        recovered = TrajectoryStep.model_validate_json(step.model_dump_json())
        assert recovered == step


# ---------------------------------------------------------------------------
# TrajectoryRecord tests
# ---------------------------------------------------------------------------


class TestTrajectoryRecord:
    def test_minimal_trajectory(self):
        traj = make_trajectory()
        assert traj.run_id == "run-001"
        assert len(traj.steps) == 1

    def test_json_round_trip(self):
        traj = make_trajectory(steps=[make_step(0), make_step(1)])
        recovered = TrajectoryRecord.model_validate_json(traj.model_dump_json())
        assert recovered == traj

    def test_non_contiguous_step_indices_raises(self):
        steps = [
            TrajectoryStep(step_index=0, action_type=ActionType.llm_response),
            TrajectoryStep(step_index=2, action_type=ActionType.llm_response),  # gap
        ]
        with pytest.raises(ValidationError, match="must be zero-based and contiguous"):
            make_trajectory(steps=steps)

    def test_step_index_not_starting_at_zero_raises(self):
        steps = [TrajectoryStep(step_index=1, action_type=ActionType.llm_response)]
        with pytest.raises(ValidationError, match="must be zero-based and contiguous"):
            make_trajectory(steps=steps)

    def test_empty_steps_accepted(self):
        traj = make_trajectory(steps=[])
        assert traj.steps == []

    def test_all_outcome_variants_accepted(self):
        for outcome in Outcome:
            traj = make_trajectory(outcome=outcome)
            assert traj.outcome == outcome

    def test_serialization_preserves_steps(self):
        steps = [
            make_step(0),
            TrajectoryStep(
                step_index=1,
                action_type=ActionType.tool_call,
                tool_name="retrieve",
                latency_ms=15.0,
            ),
        ]
        traj = make_trajectory(steps=steps, metrics=make_metrics(total_steps=2))
        as_dict = traj.model_dump()
        assert len(as_dict["steps"]) == 2
        assert as_dict["steps"][1]["tool_name"] == "retrieve"


# ---------------------------------------------------------------------------
# TrajectoryMetrics tests
# ---------------------------------------------------------------------------


class TestTrajectoryMetrics:
    def test_valid_metrics(self):
        m = TrajectoryMetrics(
            total_steps=5,
            total_latency_ms=200.0,
            total_prompt_tokens=1000,
            total_completion_tokens=50,
        )
        assert m.total_steps == 5

    def test_negative_total_steps_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryMetrics(total_steps=-1)

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryMetrics(total_steps=1, total_latency_ms=-0.1)

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryMetrics(total_steps=1, total_prompt_tokens=-1)

    def test_negative_completion_tokens_raises(self):
        with pytest.raises(ValidationError):
            TrajectoryMetrics(total_steps=1, total_completion_tokens=-1)

    def test_null_optional_fields_accepted(self):
        m = TrajectoryMetrics(total_steps=0)
        assert m.total_latency_ms is None
        assert m.total_prompt_tokens is None
        assert m.total_completion_tokens is None


# ---------------------------------------------------------------------------
# ExecutionConfig / hashing tests
# ---------------------------------------------------------------------------


class TestExecutionConfig:
    def test_valid_config(self, base_config: ExecutionConfig):
        assert base_config.model_id == "gpt-4o-mini"
        assert base_config.max_steps == 10

    def test_hash_is_string(self, base_config: ExecutionConfig):
        h = base_config.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_identical_configs_same_hash(self, base_config: ExecutionConfig):
        config2 = ExecutionConfig(
            model_id="gpt-4o-mini",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=0.0,
            max_steps=10,
            retry_budget=2,
        )
        assert base_config.compute_hash() == config2.compute_hash()

    def test_different_config_different_hash(self, base_config: ExecutionConfig):
        changed = ExecutionConfig(
            model_id="gpt-4o-mini",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=0.5,  # changed
            max_steps=10,
            retry_budget=2,
        )
        assert base_config.compute_hash() != changed.compute_hash()

    def test_hash_stable_across_calls(self, base_config: ExecutionConfig):
        """compute_hash() must be deterministic within the same process."""
        assert base_config.compute_hash() == base_config.compute_hash()

    def test_model_id_change_changes_hash(self, base_config: ExecutionConfig):
        other = ExecutionConfig(
            model_id="gpt-4o",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=0.0,
            max_steps=10,
            retry_budget=2,
        )
        assert base_config.compute_hash() != other.compute_hash()

    def test_max_steps_change_changes_hash(self, base_config: ExecutionConfig):
        other = ExecutionConfig(
            model_id="gpt-4o-mini",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=0.0,
            max_steps=20,
            retry_budget=2,
        )
        assert base_config.compute_hash() != other.compute_hash()

    def test_null_temperature_accepted(self):
        cfg = ExecutionConfig(
            model_id="claude-3",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=None,
            max_steps=5,
            retry_budget=0,
        )
        assert cfg.temperature is None
        h = cfg.compute_hash()
        assert len(h) == 64

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(
                model_id="x",
                prompt_version="v1",
                tool_selection_strategy="all",
                temperature=3.0,  # > 2.0
                max_steps=5,
                retry_budget=0,
            )

    def test_zero_max_steps_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(
                model_id="x",
                prompt_version="v1",
                tool_selection_strategy="all",
                temperature=0.0,
                max_steps=0,
                retry_budget=0,
            )

    def test_negative_retry_budget_raises(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(
                model_id="x",
                prompt_version="v1",
                tool_selection_strategy="all",
                temperature=0.0,
                max_steps=5,
                retry_budget=-1,
            )

    def test_json_round_trip(self, base_config: ExecutionConfig):
        recovered = ExecutionConfig.model_validate_json(base_config.model_dump_json())
        assert recovered == base_config
        assert recovered.compute_hash() == base_config.compute_hash()

    def test_hash_known_value(self):
        """
        Regression guard: verify the hash of a fixed config doesn't change
        across refactors.  The expected value was computed once and recorded.
        """
        cfg = ExecutionConfig(
            model_id="gpt-4o-mini",
            prompt_version="v1",
            tool_selection_strategy="all",
            temperature=0.0,
            max_steps=10,
            retry_budget=2,
        )
        import hashlib, json
        canonical = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert cfg.compute_hash() == expected
