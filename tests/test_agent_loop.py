"""Tests for code/agent/loop.py with a scripted fake model (no Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from code.agent.config import CONFIGS_DIR, AgentConfig, load_config
from code.agent.loop import LAST_TURN_WARNING, run_agent
from code.augmentation.fault_injection import FaultInjector, FaultSpec
from code.experiments.providers import LLMResponse
from code.shared.models import EvaluationRule, TaskRecord
from code.tracing.agent_tracer import AgentTracer

POLICY_TASK = TaskRecord(task_id="T1_x", template_family="multi_hop_retrieval", task_type="t",
                         prompt="What is the bonus rate for Engineering?", answerable=True,
                         gold_answer="14.5%", expected_tools=["retrieve"], evaluation_rule=EvaluationRule.exact_match)
SALARY_TASK = TaskRecord(task_id="T3_x", template_family="structured_aggregation", task_type="t",
                         prompt="Salary of E001?", answerable=True, gold_answer="95000",
                         expected_tools=["query_records"], evaluation_rule=EvaluationRule.numeric_tolerance)
UNANS_TASK = TaskRecord(task_id="T6_x", template_family="ambiguous_unanswerable", task_type="t",
                        prompt="Salary of E999?", answerable=False, gold_answer=None,
                        expected_tools=["query_records"], evaluation_rule=EvaluationRule.unanswerable)


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"id": f"c_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeClient:
    """Returns scripted replies; records what the agent sent."""

    provider_name = "fake"

    def __init__(self, replies: list[dict[str, Any]], latency_ms: float = 100.0) -> None:
        self.replies = list(replies)
        self.latency_ms = latency_ms
        self.requests: list[dict[str, Any]] = []

    def call_chat_completion(self, model_id: str, messages: list[dict[str, Any]], tools: Any = None,
                             temperature: float = 0.0, timeout: float = 0.0, seed: int | None = None,
                             max_tokens: int | None = None) -> LLMResponse:
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools, "seed": seed})
        reply = self.replies.pop(0)
        return LLMResponse(content=reply.get("content"), tool_calls=reply.get("tool_calls"), prompt_tokens=50,
                           completion_tokens=10, resolved_model=model_id, latency_ms=self.latency_ms,
                           raw_response={})


def cfg(name: str, **overrides: Any) -> AgentConfig:
    base = load_config(CONFIGS_DIR / f"{name}.yaml")
    return base.model_copy(update=overrides)


def test_abstain_baseline_needs_no_model() -> None:
    run = run_agent(UNANS_TASK, cfg("b0_abstain"))
    assert run.final_answer == "unanswerable" and run.correct and run.llm_calls == 0
    assert not run_agent(SALARY_TASK, cfg("b0_abstain")).correct


def test_direct_baseline_takes_first_line_and_sends_no_tools() -> None:
    client = FakeClient([{"content": "95,000\nbecause records say so"}])
    run = run_agent(SALARY_TASK, cfg("b1_direct"), client=client)
    assert run.final_answer == "95,000" and run.correct
    assert client.requests[0]["tools"] is None


def test_tool_loop_records_llm_latency_and_counts_tokens_once_per_turn() -> None:
    client = FakeClient([
        {"tool_calls": [call("query_records", operation="filter", field="emp_id", value="E001")]},
        {"tool_calls": [call("submit_answer", answer="$95,000")]},
    ], latency_ms=250.0)
    run = run_agent(SALARY_TASK, cfg("b2_original"), client=client)
    assert run.correct, run.grading_explanation          # grader v2 accepts "$95,000"
    assert run.llm_latency_ms == pytest.approx(500.0)
    assert run.trajectory.metrics.total_latency_ms >= 500.0
    assert run.trajectory.metrics.total_prompt_tokens == 100
    assert run.summary()["steps"] == 2 and run.summary()["tool_calls"] == 1


def test_original_loop_runs_parallel_calls_but_proposed_runs_only_the_first() -> None:
    replies = [
        {"tool_calls": [call("query_records", operation="filter", field="emp_id", value="E001"),
                        call("calculate", expression="salary * 2")]},
        {"tool_calls": [call("submit_answer", answer="95000")]},
    ]
    original = run_agent(SALARY_TASK, cfg("b2_original"), client=FakeClient(replies))
    proposed = run_agent(SALARY_TASK, cfg("proposed"), client=FakeClient(replies))
    assert [c.tool for c in original.tool_calls] == ["query_records", "calculate"]
    assert [c.tool for c in proposed.tool_calls] == ["query_records"]


def test_loop_guard_blocks_identical_repeat() -> None:
    same = call("query_records", operation="filter", field="emp_id", value="E001")
    client = FakeClient([{"tool_calls": [same]}, {"tool_calls": [same]},
                         {"tool_calls": [call("submit_answer", answer="95000")]}])
    run = run_agent(SALARY_TASK, cfg("proposed"), client=client)
    assert run.blocked_repeats == 1 and run.tool_calls[1].blocked_repeat
    assert "already called" in client.requests[2]["messages"][-1]["content"]


def test_retry_budget_hides_transient_failure_from_the_model() -> None:
    replies = [{"tool_calls": [call("query_records", operation="filter", field="emp_id", value="E001")]},
               {"tool_calls": [call("submit_answer", answer="95000")]}]
    with FaultInjector([FaultSpec("tool_failure", "query_records", trigger="first_call")]):
        run = run_agent(SALARY_TASK, cfg("proposed"), client=FakeClient(replies))
    assert run.correct and run.retries_used == 1 and run.tool_calls[0].attempts == 2
    with FaultInjector([FaultSpec("tool_failure", "query_records", trigger="first_call")]):
        original = run_agent(SALARY_TASK, cfg("b2_original"), client=FakeClient(replies))
    assert original.retries_used == 0 and original.tool_errors == 1


def test_error_feedback_adds_a_hint() -> None:
    client = FakeClient([{"tool_calls": [call("query_records", operation="filter", field="salaryy", value="1")]},
                         {"tool_calls": [call("submit_answer", answer="unanswerable")]}])
    run_agent(SALARY_TASK, cfg("proposed"), client=client)
    tool_msg = json.loads(client.requests[1]["messages"][-1]["content"])
    assert "error" in tool_msg and "hint" in tool_msg


def test_step_guard_warns_then_forces_a_submission() -> None:
    replies = [{"content": "I think it is 95000."}] * 3
    run = run_agent(SALARY_TASK, cfg("proposed", max_steps=3), client=FakeClient(list(replies)))
    assert run.forced_submit and run.final_answer == "I think it is 95000." and run.correct
    no_guard = run_agent(SALARY_TASK, cfg("b2_original", max_steps=3), client=FakeClient(list(replies)))
    assert no_guard.final_answer is None and not no_guard.correct and no_guard.detectors["step_limit_no_answer"]


def test_last_turn_warning_is_sent() -> None:
    client = FakeClient([{"tool_calls": [call("retrieve", query="bonus")]},
                         {"tool_calls": [call("submit_answer", answer="14.5%")]}])
    run_agent(POLICY_TASK, cfg("proposed", max_steps=2), client=client)
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": LAST_TURN_WARNING}


def test_malformed_arguments_are_counted() -> None:
    bad = {"id": "c", "type": "function", "function": {"name": "calculate", "arguments": "{not json"}}
    run = run_agent(SALARY_TASK, cfg("b2_original", max_steps=2),
                    client=FakeClient([{"tool_calls": [bad]}, {"tool_calls": [call("submit_answer", answer="1")]}]))
    assert run.malformed_calls == 1 and run.tool_errors == 1


def test_seed_is_passed_to_the_model() -> None:
    client = FakeClient([{"tool_calls": [call("submit_answer", answer="95000")]}])
    run_agent(SALARY_TASK, cfg("b2_original"), client=client, seed=211)
    assert client.requests[0]["seed"] == 211


def test_live_trace_has_llm_tool_and_grader_spans(tmp_path: Path) -> None:
    tracer = AgentTracer(tracking_uri=str(tmp_path / "mlflow.db"), experiment="test-traces")
    client = FakeClient([{"tool_calls": [call("query_records", operation="filter", field="emp_id", value="E001")]},
                         {"tool_calls": [call("submit_answer", answer="95000")]}])
    run = run_agent(SALARY_TASK, cfg("b2_original"), client=client, tracer=tracer)
    tracer.flush()
    import mlflow

    trace = mlflow.get_trace(run.trace_id)
    names = [s.name for s in trace.data.spans]
    assert {"agent", "llm_turn_0", "llm_turn_1", "query_records", "grader"} <= set(names)
