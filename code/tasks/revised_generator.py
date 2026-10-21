"""
Revised benchmark task suite generator for AgentEvalOps.

Generates exactly 30 TaskRecord instances across the 6 core template families:
  T1: Multi-Hop Document Retrieval & QA (5 instances, Train)
  T2: Multi-Step Numerical & Compound Calculation (5 instances, Train)
  T3: Structured Tabular/JSON Filtering & Aggregation (5 instances, Train)
  T4: Cross-Tool Chaining (5 instances, Validation)
  T5: Log/File Auditing & Threshold Verification (5 instances, Test / Held-Out)
  T6: Ambiguous & Unanswerable Tasks (5 instances, Stratified)

Outputs: data/processed/tasks_revised.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

from code.shared.models import EvaluationRule, TaskMetadata, TaskRecord
from code.tasks.revised_fixtures import (
    solve_t1_multihop,
    solve_t1_replication_factor,
    solve_t1_travel,
    solve_t2_bandwidth,
    solve_t2_depreciation,
    solve_t2_standard_overtime,
    solve_t2_storage,
    solve_t2_zero_overtime,
    solve_t3_emp_salary,
    solve_t3_server_memory_threshold,
    solve_t3_ticket_dept_max,
    solve_t3_ticket_negative_count,
    solve_t4_bonus,
    solve_t4_consulting_migrated,
    solve_t4_overtime,
    solve_t4_sales,
    solve_t4_subsidy,
    solve_t5_failure_rate,
    solve_t5_inclusive_breach_count,
    solve_t5_strict_breach_count,
    solve_t5_violation_rate,
    solve_t5_worst_service,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = _REPO_ROOT / "data" / "processed" / "tasks_revised.jsonl"
GENERATOR_VERSION = "2.0.0-substantive"
BENCHMARK_SEED = 42


def compute_content_hash(task_id: str, prompt: str, gold_answer: str | None) -> str:
    """Compute canonical SHA-256 hash without timestamps for deterministic verification."""
    payload = json.dumps(
        {"task_id": task_id, "prompt": prompt, "gold_answer": gold_answer},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_metadata(task_id: str, prompt: str, gold_answer: str | None, seed: int) -> TaskMetadata:
    return TaskMetadata(
        provenance="code/tasks/revised_generator.py",
        created_at=datetime.now(timezone.utc).isoformat(),
        generator_version=GENERATOR_VERSION,
        seed=seed,
        content_hash=compute_content_hash(task_id, prompt, gold_answer),
    )


def generate_revised_tasks(seed: int = BENCHMARK_SEED) -> list[TaskRecord]:
    """Generate all 30 substantively revised benchmark tasks with programmatically solved gold answers."""
    tasks: list[TaskRecord] = []

    # -----------------------------------------------------------------------
    # Family 1 (T1): Multi-Hop Document Retrieval & QA (5 tasks, Train)
    # -----------------------------------------------------------------------
    t1_specs = [
        (
            "T1_multihop_01",
            "current_policy_per_diem",
            "According to the current 2025 company travel policy (v2.1), what is the maximum domestic daily per diem allowance for Cloud Infrastructure staff?",
            solve_t1_travel("2025_v2"),
            ["retrieve"],
            EvaluationRule.exact_match,
        ),
        (
            "T1_multihop_02",
            "superseded_policy_per_diem",
            "What was the domestic daily per diem allowance for Cloud Infrastructure staff under the superseded 2024 travel policy (v1.0) before the 2025 revision took effect?",
            solve_t1_travel("2024_v1"),
            ["retrieve"],
            EvaluationRule.exact_match,
        ),
        (
            "T1_multihop_03",
            "entity_disambiguation_sla",
            "What is the on-call escalation SLA (in minutes) for the streaming data platform project led by Rohan Deshmukh?",
            solve_t1_multihop("Rohan Deshmukh", "Data Platform", "on-call escalation SLA (in minutes)", "active"),
            ["retrieve"],
            EvaluationRule.exact_match,
        ),
        (
            "T1_multihop_04",
            "active_vs_archived_spec",
            "Aarav Patel currently serves as technical lead for an active infrastructure project. What is the approved security compliance tier for Aarav's active project?",
            solve_t1_multihop("Aarav Patel", "Cloud Infrastructure", "security compliance tier", "active"),
            ["retrieve"],
            EvaluationRule.exact_match,
        ),
        (
            "T1_multihop_05",
            "governance_directive_control",
            "The 2025 Governance Directive explicitly mandates that all Tier-1 Enforced projects maintain a minimum data redundancy replication factor of 3 across distinct availability zones. Under this directive, what is the minimum data redundancy replication factor mandated for Tier-1 Enforced projects?",
            solve_t1_replication_factor(),
            [],  # No-tool control task: prompt contains complete evidence
            EvaluationRule.exact_match,
        ),
    ]

    for task_id, ttype, prompt, gold, tools, rule in t1_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="multi_hop_retrieval",
                task_type=ttype,
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=tools,
                evaluation_rule=rule,
                metadata=_make_metadata(task_id, prompt, gold, seed),
            )
        )

    # -----------------------------------------------------------------------
    # Family 2 (T2): Multi-Step Numerical & Compound Calculation (5 tasks, Train)
    # -----------------------------------------------------------------------
    t2_specs = [
        (
            "T2_calc_01",
            "standard_contract_overtime",
            "Under standard contract terms, engineering consulting is billed at 150.00 USD per hour, with overtime hours billed at a 1.5x multiplier. If a consultant logs 40 regular hours and 15 overtime hours, what is the total billed amount in USD?",
            solve_t2_standard_overtime(40, 15, 150.0, 1.5),
            ["calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T2_calc_02",
            "promotional_zero_surcharge_contract",
            "Under a promotional agreement, consulting is billed at 150.00 USD per hour. The agreement specifies that the weekend surcharge is explicitly 0.0% (weekend hours are billed at the standard rate of 150.00 USD per hour with no overtime markup). If a consultant logs 40 regular weekday hours and 15 weekend hours under these terms, what is the total billed amount in USD?",
            solve_t2_zero_overtime(40, 15, 150.0),
            ["calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T2_calc_03",
            "tiered_storage_and_requests",
            "An object storage pool contains 14,800 GB of stored data for a 30-day billing cycle. The first 5,000 GB are billed at a base rate of 0.022 USD per GB-month. All storage beyond the first 5,000 GB is billed at an overage tier of 0.016 USD per GB-month. Additionally, 4,200,000 read operations occurred at 0.45 USD per 10,000 requests. What is the total storage and operation charge in USD rounded to two decimal places?",
            solve_t2_storage(14800, 5000, 0.022, 0.016, 4200000, 0.45),
            ["calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T2_calc_04",
            "network_transfer_bandwidth_time",
            "A database backup of 4.5 TB must be transferred over a network link with a dedicated bandwidth of 750 Mbps. Assuming 1 TB = 1,000 GB, 1 GB = 1,000 MB, 1 MB = 8 Mb, and transmission operates at 85% effective protocol efficiency, how many minutes will the transfer take? Round up to the nearest whole minute.",
            solve_t2_bandwidth(4.5, 750, 85),
            ["calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T2_calc_05",
            "accelerated_depreciation_salvage",
            "A server appliance was purchased for 72,000 USD with a salvage floor of 10,000 USD. Under accelerated depreciation, Year 1 depreciation is 40.0% of the initial cost, and Year 2 depreciation is 40.0% of the remaining book value. What is the book value of the appliance in USD at the end of Year 2?",
            solve_t2_depreciation(72000, 40.0, 40.0),
            ["calculate"],
            EvaluationRule.numeric_tolerance,
        ),
    ]

    for task_id, ttype, prompt, gold, tools, rule in t2_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="compound_calculation",
                task_type=ttype,
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=tools,
                evaluation_rule=rule,
                metadata=_make_metadata(task_id, prompt, gold, seed),
            )
        )

    # -----------------------------------------------------------------------
    # Family 3 (T3): Structured CSV/JSON Filtering & Aggregation (5 tasks, Train)
    # -----------------------------------------------------------------------
    t3_specs = [
        (
            "T3_struct_01",
            "employee_salary_lookup_recovery",
            "Query the employee records table to find the annual salary of employee ID 'E001' (Alice).",
            solve_t3_emp_salary("E001"),
            ["query_records"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T3_struct_02",
            "server_filter_strict_exceedance",
            "In 'server_metrics.csv', how many servers in the 'US-EAST' region have status 'ONLINE' and memory_usage_pct strictly greater than 70.0?",
            solve_t3_server_memory_threshold("US-EAST", "ONLINE", 70.0, strict=True),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T3_struct_03",
            "server_filter_boundary_inclusion",
            "In 'server_metrics.csv', how many servers in the 'US-EAST' region have status 'ONLINE' and memory_usage_pct greater than or equal to 70.0?",
            solve_t3_server_memory_threshold("US-EAST", "ONLINE", 70.0, strict=False),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T3_struct_04",
            "ticket_filter_tie_breaker",
            "In 'support_tickets.csv', find the ticket_id of the ticket in the 'Billing' department that has the highest resolution_hrs. If multiple tickets tie for the highest resolution time, return the ticket_id with the smaller numeric ID.",
            solve_t3_ticket_dept_max("Billing"),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T3_struct_05",
            "ticket_negative_condition_null_exclusion",
            "In 'support_tickets.csv', count the number of resolved tickets for the 'Billing' department that have a recorded satisfaction_score that is NOT equal to 5 (tickets with missing or blank scores must be excluded from the count).",
            solve_t3_ticket_negative_count("Billing", "5"),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
    ]

    for task_id, ttype, prompt, gold, tools, rule in t3_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="structured_aggregation",
                task_type=ttype,
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=tools,
                evaluation_rule=rule,
                metadata=_make_metadata(task_id, prompt, gold, seed),
            )
        )

    # -----------------------------------------------------------------------
    # Family 4 (T4): Cross-Tool Chaining (5 tasks, Validation)
    # -----------------------------------------------------------------------
    t4_specs = [
        (
            "T4_chain_01",
            "policy_to_salary_bonus",
            "What is the annual performance bonus payout in USD for employee 'E001' (Alice) based on her department's official bonus rate policy and her salary in the employee records?",
            solve_t4_bonus("Engineering", "E001"),
            ["query_records", "retrieve", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T4_chain_02",
            "policy_to_sales_tax",
            "Calculate the total post-tax revenue in USD generated by all 'Enterprise SaaS' transactions recorded in 'q1_sales.csv', applying the official sales tax policy rate for that category.",
            solve_t4_sales("Enterprise SaaS"),
            ["read_file", "retrieve", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T4_chain_03",
            "policy_to_overtime_compensation",
            "Employee 'E002' (Bob) logged 40 hours of overtime during the last quarter. Based on Bob's salary and his department's HR overtime multiplier, what is Bob's total overtime compensation in USD? (Use the standard formula: base hourly rate = salary / 2000 hours; overtime pay = base hourly rate * department multiplier * overtime hours).",
            solve_t4_overtime("Marketing", "E002", 40.0),
            ["query_records", "retrieve", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T4_chain_04",
            "policy_to_project_subsidy",
            "What is the final subsidized hardware budget in USD for project 'PRJ-INF-932' after applying the equipment subsidy percentage designated for its owning department in the corporate infrastructure policy?",
            solve_t4_subsidy("Cloud Infrastructure", "PRJ-INF-932"),
            ["retrieve", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T4_chain_05",
            "ledger_migration_redirection_tax",
            "Following the corporate revenue tracking handbook, what is the total post-tax revenue in USD for all Q1 'Consulting Services' engagements after applying the standard consulting tax rate?",
            solve_t4_consulting_migrated("Consulting Services"),
            ["retrieve", "read_file", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
    ]

    for task_id, ttype, prompt, gold, tools, rule in t4_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="cross_tool_chaining",
                task_type=ttype,
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=tools,
                evaluation_rule=rule,
                metadata=_make_metadata(task_id, prompt, gold, seed),
            )
        )

    # -----------------------------------------------------------------------
    # Family 5 (T5): File SLA Auditing & Threshold Verification (Held-Out Test Family)
    # -----------------------------------------------------------------------
    t5_specs = [
        (
            "T5_audit_01",
            "sla_violation_rate",
            "Auditing 'logs/gateway_01.json' against the latency thresholds in 'sla_reference.json': what percentage of requests for 'AuthService' exceeded the SLA threshold? Round to 1 decimal place.",
            solve_t5_violation_rate("logs/gateway_01.json", "AuthService"),
            ["read_file", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
        (
            "T5_audit_02",
            "sla_strict_breach_count",
            "In 'logs/payment_api.json', how many requests for 'PaymentGateway' strictly breached the SLA maximum latency defined in 'sla_reference.json'? Note: requests exactly equal to the threshold are compliant.",
            solve_t5_strict_breach_count("logs/payment_api.json", "PaymentGateway"),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T5_audit_03",
            "sla_inclusive_breach_count",
            "In 'logs/payment_api.json', how many requests for 'PaymentGateway' reached or exceeded the maximum latency threshold defined in 'sla_reference.json' (latency_ms >= threshold)?",
            solve_t5_inclusive_breach_count("logs/payment_api.json", "PaymentGateway"),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T5_audit_04",
            "sla_worst_performing_service",
            "Auditing 'logs/cluster_events.json' against the latency thresholds in 'sla_reference.json': which recorded service had the higher number of SLA latency violations?",
            solve_t5_worst_service("logs/cluster_events.json", "IngressRouter", "WorkerNode"),
            ["read_file"],
            EvaluationRule.exact_match,
        ),
        (
            "T5_audit_05",
            "sla_degraded_service_failure_rate",
            "Examine 'logs/order_dispatch.json' for 'OrderDispatcher'. Calculate the overall defect percentage, defined as the percentage of requests that either returned an HTTP error status (status >= 400) OR exceeded the SLA latency limit in 'sla_reference.json'. Round to 1 decimal place.",
            solve_t5_failure_rate("logs/order_dispatch.json", "OrderDispatcher", 400),
            ["read_file", "calculate"],
            EvaluationRule.numeric_tolerance,
        ),
    ]

    for task_id, ttype, prompt, gold, tools, rule in t5_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="file_sla_auditing",
                task_type=ttype,
                prompt=prompt,
                answerable=True,
                gold_answer=gold,
                expected_tools=tools,
                evaluation_rule=rule,
                metadata=_make_metadata(task_id, prompt, gold, seed),
            )
        )

    # -----------------------------------------------------------------------
    # Family 6 (T6): Ambiguous & Unanswerable Tasks (5 tasks, Stratified)
    # -----------------------------------------------------------------------
    t6_specs = [
        (
            "T6_unans_01",
            "missing_database_entity",
            "What is the annual salary of employee ID 'E999' according to the corporate employee records?",
            ["query_records"],
        ),
        (
            "T6_unans_02",
            "absent_policy_specification",
            "What is the certified data encryption standard for project 'PRJ-QUANTUM-X' according to the corporate security policy handbook?",
            ["retrieve"],
        ),
        (
            "T6_unans_03",
            "ambiguous_duplicate_identity",
            "What department does Rahul Sharma work in according to the staff directory?",
            ["retrieve"],
        ),
        (
            "T6_unans_04",
            "missing_schema_column",
            "In 'server_metrics.csv', what is the total power consumption in kilowatt-hours (kWh) across all online servers?",
            ["read_file"],
        ),
        (
            "T6_unans_05",
            "empty_dataset_undefined_metric",
            "Audit the service log in 'logs/empty_service.json' and calculate the average latency of the logged requests.",
            ["read_file"],
        ),
    ]

    for task_id, ttype, prompt, tools in t6_specs:
        tasks.append(
            TaskRecord(
                task_id=task_id,
                template_family="ambiguous_unanswerable",
                task_type=ttype,
                prompt=prompt,
                answerable=False,
                gold_answer=None,
                expected_tools=tools,
                evaluation_rule=EvaluationRule.unanswerable,
                metadata=_make_metadata(task_id, prompt, None, seed),
            )
        )

    assert len(tasks) == 30, f"Expected exactly 30 tasks, got {len(tasks)}"
    return tasks


def export_tasks_jsonl(
    tasks: list[TaskRecord],
    output_path: pathlib.Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = True,
) -> pathlib.Path:
    """Export tasks to JSONL file."""
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(t.model_dump_json() + "\n")
    return output_path


def load_tasks_jsonl(path: pathlib.Path) -> list[TaskRecord]:
    """Load TaskRecord instances from JSONL."""
    path = pathlib.Path(path).resolve()
    tasks: list[TaskRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(TaskRecord.model_validate_json(line))
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate revised 30-task benchmark suite.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED)
    args = parser.parse_args()

    revised_tasks = generate_revised_tasks(seed=args.seed)
    out_file = export_tasks_jsonl(revised_tasks, args.output, overwrite=True)
    print(f"Generated {len(revised_tasks)} revised tasks -> {out_file}")
    sys.exit(0)
