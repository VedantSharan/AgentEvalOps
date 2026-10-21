"""Tests for code/augmentation (Part D). Includes negative cases: every check must be able to fail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.agent.tools import TOOL_REGISTRY
from code.augmentation.build_augmented import build
from code.augmentation.entity_variants import REJECTED_CANDIDATES, VARIANTS, check_evidence
from code.augmentation.fault_injection import FaultInjector, FaultSpec, canonical
from code.augmentation.paraphrase import paraphrase, spans_preserved

PROMPT = "In 'server_metrics.csv', how many servers in the 'US-EAST' region have status 'ONLINE' and memory_usage_pct strictly greater than 70.0?"


# --- paraphrase ------------------------------------------------------------------

def test_paraphrase_is_deterministic_and_changes_text() -> None:
    a, _ = paraphrase(PROMPT, 7)
    b, _ = paraphrase(PROMPT, 7)
    assert a == b and a != PROMPT
    assert spans_preserved(PROMPT, a)[0]


@pytest.mark.parametrize("broken", [
    PROMPT.replace("strictly ", ""),                 # logic word dropped
    PROMPT.replace("70.0", "75.0"),                  # threshold changed
    PROMPT.replace("'US-EAST'", "'EU-WEST'"),        # entity changed
    PROMPT.replace("?", ", excluding EU servers?"),  # logic word added
])
def test_span_check_catches_meaning_changes(broken: str) -> None:
    assert not spans_preserved(PROMPT, broken)[0]


# --- entity variants ---------------------------------------------------------------

def test_all_variants_have_evidence() -> None:
    assert all(check_evidence(v)[0] for v in VARIANTS)


def test_rejected_candidates_fail_evidence_check() -> None:
    for v in REJECTED_CANDIDATES:
        ok, _ = check_evidence(v)
        assert not ok, v.task_id


# --- fault injection -----------------------------------------------------------------

ARGS = {"filename": "server_metrics.csv"}


def test_transient_failure_then_recovery() -> None:
    clean = TOOL_REGISTRY["read_file"](**ARGS)
    with FaultInjector([FaultSpec("tool_failure", "read_file", "first_call")]) as inj:
        first = TOOL_REGISTRY["read_file"](**ARGS)
        second = TOOL_REGISTRY["read_file"](**ARGS)
    assert not first.success and first.error and "503" in first.error
    assert second.success and second.output == clean.output
    assert len(inj.log) == 1


def test_slow_response_keeps_output_and_adds_latency() -> None:
    clean = TOOL_REGISTRY["calculate"](expression="2 + 3")
    slept: list[float] = []
    with FaultInjector([FaultSpec("slow_response", "calculate", "always", delay_ms=1500)], sleep=slept.append):
        slow = TOOL_REGISTRY["calculate"](expression="2 + 3")
    assert slow.output == clean.output and slept == [1.5] and slow.latency_ms >= 1500


@pytest.mark.parametrize("tool,args", [
    ("read_file", {"filename": "server_metrics.csv"}),
    ("read_file", {"filename": "logs/payment_api.json"}),
    ("read_file", {"filename": "sla_reference.json"}),
    ("retrieve", {"query": "tax"}),
    ("calculate", {"expression": "45000 * 1.18"}),
    ("query_records", {"operation": "filter", "field": "emp_id", "value": "E003"}),
])
def test_format_change_keeps_information(tool: str, args: dict) -> None:
    clean = TOOL_REGISTRY[tool](**args)
    with FaultInjector([FaultSpec("format_change", tool)]):
        changed = TOOL_REGISTRY[tool](**args)
    assert changed.output != clean.output
    assert canonical(tool, changed.output) == canonical(tool, clean.output)


def test_registry_restored_even_after_exception() -> None:
    original = TOOL_REGISTRY["retrieve"]
    with pytest.raises(RuntimeError):
        with FaultInjector([FaultSpec("tool_failure", "retrieve")]):
            raise RuntimeError("boom")
    assert TOOL_REGISTRY["retrieve"] is original


def test_invalid_fault_specs_rejected() -> None:
    with pytest.raises(ValueError):
        FaultSpec("explode", "read_file")
    with pytest.raises(ValueError):
        FaultSpec("tool_failure", "submit_answer")
    with pytest.raises(ValueError):
        FaultSpec("slow_response", "read_file")  # no delay


# --- full build ------------------------------------------------------------------------

def test_build_passes_all_checks_and_is_reproducible(tmp_path: Path) -> None:
    s1 = build(Path("data/processed/tasks_revised.jsonl"), tmp_path / "a", tmp_path / "ea")
    build(Path("data/processed/tasks_revised.jsonl"), tmp_path / "b", tmp_path / "eb")
    assert s1["all_checks_passed"] and s1["rejected_candidates_correctly_failed"]
    assert s1["validation_suite_on_flat_tasks"]["passed"]
    assert (tmp_path / "a" / "tasks_augmented.jsonl").read_bytes() == (tmp_path / "b" / "tasks_augmented.jsonl").read_bytes()
    recs = [json.loads(line) for line in (tmp_path / "a" / "tasks_augmented.jsonl").read_text().splitlines()]
    assert all(r["augmentation"]["parent_task_id"] for r in recs)
    outage = [r for r in recs if r["augmentation"]["params"].get("fault_label") == "information_outage"]
    assert outage and all(r["task"]["gold_answer"] is None and not r["task"]["answerable"] for r in outage)
    calc = [r for r in recs if r["augmentation"]["params"].get("fault_label") == "calculator_outage"]
    assert calc and all(r["task"]["gold_answer"] is not None for r in calc)


# --- integration: faults reach the real agent loop ---------------------------------------

class _ScriptedClient:
    """Fake provider: read_file twice (to see a transient failure recover), then submit."""

    provider_name = "scripted"

    def __init__(self) -> None:
        self.turn = 0

    def call_chat_completion(self, **_: object):
        from code.experiments.providers import LLMResponse
        self.turn += 1
        if self.turn <= 2:
            call = {"id": f"c{self.turn}", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"filename": "server_metrics.csv"})}}
        else:
            call = {"id": "c3", "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps({"answer": "5"})}}
        return LLMResponse(content=None, tool_calls=[call], prompt_tokens=None, completion_tokens=None,
                           resolved_model="scripted", latency_ms=0.0, raw_response={})


def test_fault_is_visible_in_real_agent_trajectory() -> None:
    from code.experiments.agent_executor import execute_agent_task
    from code.shared.models import ExecutionConfig, TaskRecord

    task = TaskRecord.model_validate_json(Path("data/processed/tasks_revised.jsonl").read_text().splitlines()[12])
    assert task.task_id == "T3_struct_03"
    cfg = ExecutionConfig(model_id="scripted", prompt_version="v1", tool_selection_strategy="openai_function_calling",
                          temperature=0.0, max_steps=5, retry_budget=1)
    with FaultInjector([FaultSpec("tool_failure", "read_file", "first_call")]) as inj:
        res = execute_agent_task(task=task, provider_client=_ScriptedClient(), model_id="scripted", config=cfg, rate_limit_delay=0)
    steps = res.trajectory.steps
    assert steps[0].tool_result.success is False and "503" in steps[0].tool_result.error   # injected failure recorded
    assert steps[1].tool_result.success is True                                              # recovery recorded
    assert res.final_answer == "5" and len(inj.log) == 1
