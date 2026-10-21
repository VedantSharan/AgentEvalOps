"""
Comprehensive validation and diagnostic tests for the revised 30-task benchmark suite.

Validates:
  1. Task count (exactly 30) and family distribution (5 per family across 6 families).
  2. Task ID uniqueness, schema conformance, and stable content hashes.
  3. Independent ground-truth verification using an independent test solver.
  4. Plausible distractor and failure mode rejection by the grader.
  5. Matched pair discrimination (stale vs current, strict vs inclusive boundaries).
  6. Recoverable tool error workflows (field name recovery in query_records).
  7. Virtual file and knowledge store accessibility.
  8. Leakage-free train/val/test allocation.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import pytest

from code.agent.tools import (
    TOOL_REGISTRY,
    _VIRTUAL_FILES,
    calculate,
    query_records,
    read_file,
    retrieve,
    submit_answer,
)
from code.shared.models import EvaluationRule, TaskRecord
from code.tasks.fixtures import EMPLOYEE_RECORDS
from code.tasks.grader import grade
from code.tasks.revised_fixtures import REVISED_KB_DOCS, REVISED_VIRTUAL_FILES
from code.tasks.revised_generator import (
    BENCHMARK_SEED,
    DEFAULT_OUTPUT_PATH,
    compute_content_hash,
    generate_revised_tasks,
    load_tasks_jsonl,
)


@pytest.fixture(scope="module")
def benchmark_tasks() -> list[TaskRecord]:
    """Load or generate the revised benchmark tasks."""
    if DEFAULT_OUTPUT_PATH.exists():
        return load_tasks_jsonl(DEFAULT_OUTPUT_PATH)
    return generate_revised_tasks()


# ---------------------------------------------------------------------------
# Independent Oracle Solvers (recomputed independently of generator solvers)
# ---------------------------------------------------------------------------

def _independent_recompute_gold(task: TaskRecord) -> str | None:
    """Independent oracle recalculating expected answers directly from raw fixture structures."""
    tid = task.task_id

    # T1
    if tid == "T1_multihop_01":
        return str(REVISED_KB_DOCS["travel_policy"]["2025_v2"]["Cloud Infrastructure"])
    if tid == "T1_multihop_02":
        return str(REVISED_KB_DOCS["travel_policy"]["2024_v1"]["Cloud Infrastructure"])
    if tid == "T1_multihop_03":
        return str(REVISED_KB_DOCS["projects"]["PRJ-DAT-714"]["on-call escalation SLA (in minutes)"])
    if tid == "T1_multihop_04":
        return str(REVISED_KB_DOCS["projects"]["PRJ-INF-932"]["security compliance tier"])
    if tid == "T1_multihop_05":
        return "3"

    # T2
    if tid == "T2_calc_01":
        reg = 40 * 150.0
        ot = 15 * (150.0 * 1.5)
        return f"{reg + ot:.2f}"
    if tid == "T2_calc_02":
        total = (40 + 15) * 150.0
        return f"{total:.2f}"
    if tid == "T2_calc_03":
        b_charge = 5000 * 0.022
        o_charge = (14800 - 5000) * 0.016
        r_charge = (4200000 / 10000) * 0.45
        return f"{b_charge + o_charge + r_charge:.2f}"
    if tid == "T2_calc_04":
        bits = 4.5 * 1000 * 1000 * 8
        rate = 750 * 0.85
        return str(math.ceil((bits / rate) / 60))
    if tid == "T2_calc_05":
        y1 = 72000 * 0.60
        y2 = y1 * 0.60
        return f"{y2:.2f}"

    # T3
    if tid == "T3_struct_01":
        emp = [e for e in EMPLOYEE_RECORDS if e["emp_id"] == "E001"][0]
        return str(emp["salary"])
    if tid == "T3_struct_02":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["server_metrics.csv"]))]
        matches = [r for r in lines if r["region"] == "US-EAST" and r["status"] == "ONLINE" and float(r["memory_usage_pct"]) > 70.0]
        return str(len(matches))
    if tid == "T3_struct_03":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["server_metrics.csv"]))]
        matches = [r for r in lines if r["region"] == "US-EAST" and r["status"] == "ONLINE" and float(r["memory_usage_pct"]) >= 70.0]
        return str(len(matches))
    if tid == "T3_struct_04":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["support_tickets.csv"])) if r["department"] == "Billing"]
        top_res = max(float(r["resolution_hrs"]) for r in lines)
        ties = [r for r in lines if float(r["resolution_hrs"]) == top_res]
        chosen = min(ties, key=lambda x: int(x["ticket_id"].split("-")[1]))
        return chosen["ticket_id"]
    if tid == "T3_struct_05":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["support_tickets.csv"])) if r["department"] == "Billing"]
        qualifying = [r for r in lines if r["satisfaction_score"].strip() and r["satisfaction_score"].strip() != "5"]
        return str(len(qualifying))

    # T4
    if tid == "T4_chain_01":
        sal = [e for e in EMPLOYEE_RECORDS if e["emp_id"] == "E001"][0]["salary"]
        rate = REVISED_KB_DOCS["bonus_policy"]["Engineering"]
        return f"{sal * (rate / 100.0):.2f}"
    if tid == "T4_chain_02":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["q1_sales.csv"])) if r["category"] == "Enterprise SaaS"]
        rev = sum(float(r["revenue"]) for r in lines)
        tax = REVISED_KB_DOCS["tax_policy"]["Enterprise SaaS"]
        return f"{rev * (1.0 + tax / 100.0):.2f}"
    if tid == "T4_chain_03":
        sal = [e for e in EMPLOYEE_RECORDS if e["emp_id"] == "E002"][0]["salary"]
        hourly = sal / 2000.0
        mult = REVISED_KB_DOCS["overtime_policy"]["Marketing"]
        return f"{hourly * mult * 40.0:.2f}"
    if tid == "T4_chain_04":
        base_budget = REVISED_KB_DOCS["projects"]["PRJ-INF-932"]["hardware_base_budget"]
        sub = REVISED_KB_DOCS["infrastructure_policy"]["Cloud Infrastructure"]
        return f"{base_budget * (1.0 + sub / 100.0):.2f}"
    if tid == "T4_chain_05":
        lines = [r for r in csv.DictReader(io.StringIO(REVISED_VIRTUAL_FILES["consulting_ledger.csv"]))]
        rev = sum(float(r["revenue"]) for r in lines if r["category"] == "Consulting Services")
        tax = REVISED_KB_DOCS["tax_policy"]["Consulting Services"]
        return f"{rev * (1.0 + tax / 100.0):.2f}"

    # T5
    if tid == "T5_audit_01":
        entries = json.loads(REVISED_VIRTUAL_FILES["logs/gateway_01.json"])
        logs = [e for e in entries if e["service"] == "AuthService"]
        violations = sum(1 for e in logs if e["latency_ms"] > 200)
        return f"{(violations / len(logs)) * 100.0:.1f}"
    if tid == "T5_audit_02":
        entries = json.loads(REVISED_VIRTUAL_FILES["logs/payment_api.json"])
        logs = [e for e in entries if e["service"] == "PaymentGateway"]
        return str(sum(1 for e in logs if e["latency_ms"] > 350))
    if tid == "T5_audit_03":
        entries = json.loads(REVISED_VIRTUAL_FILES["logs/payment_api.json"])
        logs = [e for e in entries if e["service"] == "PaymentGateway"]
        return str(sum(1 for e in logs if e["latency_ms"] >= 350))
    if tid == "T5_audit_04":
        entries = json.loads(REVISED_VIRTUAL_FILES["logs/cluster_events.json"])
        v_a = sum(1 for e in entries if e["service"] == "IngressRouter" and e["latency_ms"] > 100)
        v_b = sum(1 for e in entries if e["service"] == "WorkerNode" and e["latency_ms"] > 250)
        return "IngressRouter" if v_a > v_b else "WorkerNode"
    if tid == "T5_audit_05":
        entries = json.loads(REVISED_VIRTUAL_FILES["logs/order_dispatch.json"])
        logs = [e for e in entries if e["service"] == "OrderDispatcher"]
        failures = sum(1 for e in logs if e["status"] >= 400 or e["latency_ms"] > 400)
        return f"{(failures / len(logs)) * 100.0:.1f}"

    # T6
    if tid.startswith("T6_unans"):
        return None

    raise ValueError(f"Unknown task {tid}")


class TestSuiteStructureAndIntegrity:
    """Validate suite dimensions, schema validity, and identifiers."""

    def test_total_task_count(self, benchmark_tasks: list[TaskRecord]):
        assert len(benchmark_tasks) == 30, f"Expected 30 tasks, got {len(benchmark_tasks)}"

    def test_unique_task_ids(self, benchmark_tasks: list[TaskRecord]):
        task_ids = [t.task_id for t in benchmark_tasks]
        assert len(task_ids) == len(set(task_ids)), "Task IDs must be globally unique"

    def test_family_distribution(self, benchmark_tasks: list[TaskRecord]):
        expected_families = {
            "multi_hop_retrieval": 5,
            "compound_calculation": 5,
            "structured_aggregation": 5,
            "cross_tool_chaining": 5,
            "file_sla_auditing": 5,
            "ambiguous_unanswerable": 5,
        }
        counts: dict[str, int] = {}
        for t in benchmark_tasks:
            counts[t.template_family] = counts.get(t.template_family, 0) + 1
        assert counts == expected_families

    def test_deterministic_hashes_without_timestamps(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            assert task.metadata is not None
            assert task.metadata.seed == BENCHMARK_SEED
            expected_hash = compute_content_hash(task.task_id, task.prompt, task.gold_answer)
            assert task.metadata.content_hash == expected_hash

    def test_deterministic_regeneration(self):
        run1 = generate_revised_tasks(seed=42)
        run2 = generate_revised_tasks(seed=42)
        assert len(run1) == len(run2)
        for t1, t2 in zip(run1, run2):
            assert t1.task_id == t2.task_id
            assert t1.prompt == t2.prompt
            assert t1.gold_answer == t2.gold_answer
            assert t1.metadata.content_hash == t2.metadata.content_hash


class TestIndependentGroundTruthAndGrader:
    """Validate every task's gold answer against an independent calculation and grader acceptance."""

    def test_independent_gold_recomputation_matches(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            expected_gold = _independent_recompute_gold(task)
            assert task.gold_answer == expected_gold, (
                f"Mismatch for {task.task_id}: task.gold_answer={task.gold_answer!r} "
                f"!= independent_oracle={expected_gold!r}"
            )

    def test_all_gold_answers_pass_grader(self, benchmark_tasks: list[TaskRecord]):
        for task in benchmark_tasks:
            submission = task.gold_answer if task.answerable else "unanswerable"
            res = grade(task, submission)
            assert res.correct, f"Grading failed for {task.task_id}: {res.detail}"


class TestPlausibleDistractorRejection:
    """Ensure the grader rejects plausible incorrect distractors and failure modes."""

    def test_reject_stale_per_diem_distractor(self, benchmark_tasks: list[TaskRecord]):
        t1_01 = next(t for t in benchmark_tasks if t.task_id == "T1_multihop_01")
        # Submitting superseded 2024 value (180 USD) to 2025 task must fail
        assert not grade(t1_01, "180 USD").correct

    def test_reject_current_per_diem_on_stale_task(self, benchmark_tasks: list[TaskRecord]):
        t1_02 = next(t for t in benchmark_tasks if t.task_id == "T1_multihop_02")
        # Submitting current 2025 value (240 USD) to historical 2024 task must fail
        assert not grade(t1_02, "240 USD").correct

    def test_reject_hallucinated_overtime_multiplier(self, benchmark_tasks: list[TaskRecord]):
        t2_02 = next(t for t in benchmark_tasks if t.task_id == "T2_calc_02")
        # Submitting 9375.00 (from assuming 1.5x weekend markup) must fail on 0% surcharge task
        assert not grade(t2_02, "9375.00").correct

    def test_reject_boundary_errors_in_server_metrics(self, benchmark_tasks: list[TaskRecord]):
        t3_02 = next(t for t in benchmark_tasks if t.task_id == "T3_struct_02")
        t3_03 = next(t for t in benchmark_tasks if t.task_id == "T3_struct_03")
        # T3_02 is strict > 70.0 (count = 4). Submitting inclusive count (5) must fail.
        assert not grade(t3_02, "5").correct
        # T3_03 is inclusive >= 70.0 (count = 5). Submitting strict count (4) must fail.
        assert not grade(t3_03, "4").correct

    def test_reject_unmigrated_sales_ledger_distractor(self, benchmark_tasks: list[TaskRecord]):
        t4_05 = next(t for t in benchmark_tasks if t.task_id == "T4_chain_05")
        # Submitting un-migrated revenue (13200.00 from q1_sales.csv) must fail
        assert not grade(t4_05, "13200.00").correct

    def test_reject_boundary_errors_in_sla_audit(self, benchmark_tasks: list[TaskRecord]):
        t5_02 = next(t for t in benchmark_tasks if t.task_id == "T5_audit_02")
        t5_03 = next(t for t in benchmark_tasks if t.task_id == "T5_audit_03")
        # T5_02 is strict > 350 (count = 4). Submitting 5 must fail.
        assert not grade(t5_02, "5").correct
        # T5_03 is inclusive >= 350 (count = 5). Submitting 4 must fail.
        assert not grade(t5_03, "4").correct

    def test_reject_hallucinated_answers_for_unanswerable_tasks(self, benchmark_tasks: list[TaskRecord]):
        t6_tasks = [t for t in benchmark_tasks if t.template_family == "ambiguous_unanswerable"]
        hallucinations = ["105000", "AES-256", "Engineering", "450.0", "125.0"]
        for task, fake_answer in zip(t6_tasks, hallucinations):
            assert not grade(task, fake_answer).correct


class TestToolInterfaceAndRecoverableErrors:
    """Verify tool behavior, error recovery paths, and resource accessibility."""

    def test_query_records_field_recovery(self):
        """Simulate realistic tool recovery when natural language uses 'employee_id' instead of 'emp_id'."""
        # Initial call with bad field fails deterministically
        bad_res = query_records(operation="filter", field="employee_id", value="E001")
        assert bad_res.success is False
        assert "Unknown field 'employee_id'" in bad_res.error

        # Recovering with schema-valid field 'emp_id' succeeds
        recovered_res = query_records(operation="filter", field="emp_id", value="E001")
        assert recovered_res.success is True
        assert recovered_res.output["count"] == 1
        assert recovered_res.output["rows"][0]["salary"] == 95000

    def test_all_referenced_virtual_files_exist(self):
        required_files = [
            "server_metrics.csv",
            "support_tickets.csv",
            "q1_sales.csv",
            "consulting_ledger.csv",
            "logs/gateway_01.json",
            "logs/payment_api.json",
            "logs/cluster_events.json",
            "logs/order_dispatch.json",
            "logs/empty_service.json",
            "sla_reference.json",
            "project_registry.json",
        ]
        for f in required_files:
            res = read_file(f)
            assert res.success is True, f"Failed reading {f}: {res.error}"
            assert res.output is not None

    def test_empty_dataset_handling(self):
        res = read_file("logs/empty_service.json")
        assert res.success is True
        assert json.loads(res.output["content"]) == []

    def test_unnecessary_tool_control_task(self, benchmark_tasks: list[TaskRecord]):
        ctrl = next(t for t in benchmark_tasks if t.task_id == "T1_multihop_05")
        assert ctrl.expected_tools == []
        assert "replication factor of 3" in ctrl.prompt
        assert ctrl.gold_answer == "3"

    def test_leakage_free_split_allocation(self, benchmark_tasks: list[TaskRecord]):
        train = [t for t in benchmark_tasks if t.template_family in {"multi_hop_retrieval", "compound_calculation", "structured_aggregation"}]
        val = [t for t in benchmark_tasks if t.template_family == "cross_tool_chaining"]
        test = [t for t in benchmark_tasks if t.template_family == "file_sla_auditing"]
        strat = [t for t in benchmark_tasks if t.template_family == "ambiguous_unanswerable"]

        assert len(train) == 15
        assert len(val) == 5
        assert len(test) == 5
        assert len(strat) == 5
