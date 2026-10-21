"""
Entity / parameter variants of the 30 base tasks (Assignment 1, Part D).

Each variant keeps the parent's template (wording, family, evaluation rule,
expected tools) and changes entities (employee, region, department, service,
person) or numeric parameters.  The gold answer is recomputed with the SAME
solver function the base suite uses (code/tasks/revised_fixtures.py), never
copied or typed by hand.

Label preservation is checked by ``evidence``: tool calls that a correct agent
needs.  ``check_evidence`` runs them against the real tool layer and requires
each to succeed and to contain the stated value (or, for unanswerable
variants, to return no rows).  Variants whose evidence is not available in the
knowledge base were deliberately NOT created (e.g. Cloud Storage tax rate,
Sanya Verma's on-call SLA: no fact exists, so the task would silently become
unanswerable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from code.agent.tools import TOOL_REGISTRY
from code.tasks import revised_fixtures as fx


@dataclass(frozen=True)
class Probe:
    tool: str
    args: dict[str, Any]
    must_contain: str | None = None  # None + expect_empty=True -> absence check
    expect_empty: bool = False


@dataclass(frozen=True)
class Variant:
    parent_task_id: str
    suffix: str
    prompt: str
    solve: Callable[[], str] | None  # None for unanswerable variants
    params: dict[str, Any]
    evidence: tuple[Probe, ...] = field(default_factory=tuple)

    @property
    def task_id(self) -> str:
        return f"{self.parent_task_id}__{self.suffix}"


def _t2_overtime(reg: int, ot: int, rate: float, mult: float) -> str:
    return (f"Under standard contract terms, engineering consulting is billed at {rate:.2f} USD per hour, with overtime hours "
            f"billed at a {mult}x multiplier. If a consultant logs {reg} regular hours and {ot} overtime hours, what is the "
            "total billed amount in USD?")


def _t2_zero(reg: int, wk: int, rate: float) -> str:
    return (f"Under a promotional agreement, consulting is billed at {rate:.2f} USD per hour. The agreement specifies that the "
            f"weekend surcharge is explicitly 0.0% (weekend hours are billed at the standard rate of {rate:.2f} USD per hour "
            f"with no overtime markup). If a consultant logs {reg} regular weekday hours and {wk} weekend hours under these "
            "terms, what is the total billed amount in USD?")


def _t2_storage(gb: int, lim: int, r1: float, r2: float, reqs: int, rr: float) -> str:
    return (f"An object storage pool contains {gb:,} GB of stored data for a 30-day billing cycle. The first {lim:,} GB are "
            f"billed at a base rate of {r1} USD per GB-month. All storage beyond the first {lim:,} GB is billed at an overage "
            f"tier of {r2} USD per GB-month. Additionally, {reqs:,} read operations occurred at {rr:.2f} USD per 10,000 "
            "requests. What is the total storage and operation charge in USD rounded to two decimal places?")


def _t2_bandwidth(tb: float, mbps: int, eff: int) -> str:
    return (f"A database backup of {tb} TB must be transferred over a network link with a dedicated bandwidth of {mbps} Mbps. "
            "Assuming 1 TB = 1,000 GB, 1 GB = 1,000 MB, 1 MB = 8 Mb, and transmission operates at "
            f"{eff}% effective protocol efficiency, how many minutes will the transfer take? Round up to the nearest whole minute.")


def _t2_depr(cost: int, floor: int, r: float) -> str:
    return (f"A server appliance was purchased for {cost:,} USD with a salvage floor of {floor:,} USD. Under accelerated "
            f"depreciation, Year 1 depreciation is {r}% of the initial cost, and Year 2 depreciation is {r}% of the "
            "remaining book value. What is the book value of the appliance in USD at the end of Year 2?")


def _emp(emp_id: str, salary: int) -> Probe:
    return Probe("query_records", {"operation": "filter", "field": "emp_id", "value": emp_id}, str(salary))


VARIANTS: tuple[Variant, ...] = (
    # --- T1 multi-hop retrieval: different person -> project -> attribute chains
    Variant("T1_multihop_04", "ev01",
            "Meera Nair currently leads an active security project. What is the primary data retention policy (in days) for Meera's active project?",
            lambda: fx.solve_t1_multihop("Meera Nair", "Security Ops", "data retention policy (in days)"),
            {"person": "Meera Nair", "department": "Security Ops", "attribute": "data retention policy (in days)"},
            (Probe("retrieve", {"query": "Meera Nair"}, "PRJ-SEC-581"), Probe("retrieve", {"query": "PRJ-SEC-581"}, "730"))),
    Variant("T1_multihop_04", "ev02",
            "Sanya Verma currently leads an active AI project. What is the cluster node quota for Sanya's active project?",
            lambda: fx.solve_t1_multihop("Sanya Verma", "Core AI", "cluster node quota"),
            {"person": "Sanya Verma", "department": "Core AI", "attribute": "cluster node quota"},
            (Probe("retrieve", {"query": "Sanya Verma"}, "PRJ-AI-840"), Probe("retrieve", {"query": "PRJ-AI-840"}, "64 nodes"))),
    Variant("T1_multihop_03", "ev01",
            "What is the on-call escalation SLA (in minutes) for the batch ETL project led by Rohan Mehta?",
            lambda: fx.solve_t1_multihop("Rohan Mehta", "Analytics", "on-call escalation SLA (in minutes)"),
            {"person": "Rohan Mehta", "department": "Analytics", "attribute": "on-call escalation SLA (in minutes)"},
            (Probe("retrieve", {"query": "Rohan Mehta"}, "PRJ-ANA-204"), Probe("retrieve", {"query": "PRJ-ANA-204"}, "60"))),
    # --- T2 compound calculation: new numeric parameters
    Variant("T2_calc_01", "pv01", _t2_overtime(32, 10, 180.0, 1.5), lambda: fx.solve_t2_standard_overtime(32, 10, 180.0, 1.5),
            {"regular_hours": 32, "overtime_hours": 10, "base_rate": 180.0, "multiplier": 1.5}),
    Variant("T2_calc_01", "pv02", _t2_overtime(45, 8, 120.0, 2.0), lambda: fx.solve_t2_standard_overtime(45, 8, 120.0, 2.0),
            {"regular_hours": 45, "overtime_hours": 8, "base_rate": 120.0, "multiplier": 2.0}),
    Variant("T2_calc_02", "pv01", _t2_zero(38, 12, 160.0), lambda: fx.solve_t2_zero_overtime(38, 12, 160.0),
            {"regular_hours": 38, "weekend_hours": 12, "base_rate": 160.0}),
    Variant("T2_calc_03", "pv01", _t2_storage(9200, 5000, 0.022, 0.016, 2600000, 0.45),
            lambda: fx.solve_t2_storage(9200, 5000, 0.022, 0.016, 2600000, 0.45),
            {"total_gb": 9200, "tier1_limit": 5000, "tier1_rate": 0.022, "overage_rate": 0.016, "read_requests": 2600000, "rate_per_10k": 0.45}),
    Variant("T2_calc_04", "pv01", _t2_bandwidth(2.0, 500, 80), lambda: fx.solve_t2_bandwidth(2.0, 500, 80),
            {"size_tb": 2.0, "bandwidth_mbps": 500, "efficiency_pct": 80}),
    Variant("T2_calc_05", "pv01", _t2_depr(60000, 8000, 35.0), lambda: fx.solve_t2_depreciation(60000, 35.0, 35.0),
            {"initial_cost": 60000, "salvage_floor": 8000, "rate": 35.0}),
    # --- T3 structured aggregation: other employees, regions, thresholds, departments
    Variant("T3_struct_01", "ev01", "Query the employee records table to find the annual salary of employee ID 'E003' (Carol).",
            lambda: fx.solve_t3_emp_salary("E003"), {"emp_id": "E003"}, (_emp("E003", 105000),)),
    Variant("T3_struct_01", "ev02", "Query the employee records table to find the annual salary of employee ID 'E005' (Eve).",
            lambda: fx.solve_t3_emp_salary("E005"), {"emp_id": "E005"}, (_emp("E005", 88000),)),
    Variant("T3_struct_02", "ev01",
            "In 'server_metrics.csv', how many servers in the 'EU-WEST' region have status 'ONLINE' and memory_usage_pct strictly greater than 70.0?",
            lambda: fx.solve_t3_server_memory_threshold("EU-WEST", "ONLINE", 70.0, strict=True),
            {"region": "EU-WEST", "threshold": 70.0, "strict": True},
            (Probe("read_file", {"filename": "server_metrics.csv"}, "EU-WEST"),)),
    Variant("T3_struct_03", "ev01",
            "In 'server_metrics.csv', how many servers in the 'EU-WEST' region have status 'ONLINE' and memory_usage_pct greater than or equal to 72.0?",
            lambda: fx.solve_t3_server_memory_threshold("EU-WEST", "ONLINE", 72.0, strict=False),
            {"region": "EU-WEST", "threshold": 72.0, "strict": False},
            (Probe("read_file", {"filename": "server_metrics.csv"}, "SRV-112,EU-WEST,ONLINE,75.0,72.0"),)),
    Variant("T3_struct_04", "ev01",
            "In 'support_tickets.csv', find the ticket_id of the ticket in the 'Technical' department that has the highest resolution_hrs. If multiple tickets tie for the highest resolution time, return the ticket_id with the smaller numeric ID.",
            lambda: fx.solve_t3_ticket_dept_max("Technical"), {"department": "Technical"},
            (Probe("read_file", {"filename": "support_tickets.csv"}, "Technical"),)),
    Variant("T3_struct_05", "ev01",
            "In 'support_tickets.csv', count the number of resolved tickets for the 'Technical' department that have a recorded satisfaction_score that is NOT equal to 4 (tickets with missing or blank scores must be excluded from the count).",
            lambda: fx.solve_t3_ticket_negative_count("Technical", "4"), {"department": "Technical", "excluded_score": "4"},
            (Probe("read_file", {"filename": "support_tickets.csv"}, "Technical"),)),
    # --- T4 cross-tool chaining: other employees (only where the policy fact exists)
    Variant("T4_chain_01", "ev01",
            "What is the annual performance bonus payout in USD for employee 'E003' (Carol) based on her department's official bonus rate policy and her salary in the employee records?",
            lambda: fx.solve_t4_bonus("Engineering", "E003"), {"emp_id": "E003", "department": "Engineering"},
            (_emp("E003", 105000), Probe("retrieve", {"query": "bonus rate Engineering"}, "14.5%"))),
    Variant("T4_chain_01", "ev02",
            "What is the annual performance bonus payout in USD for employee 'E005' (Eve) based on her department's official bonus rate policy and her salary in the employee records?",
            lambda: fx.solve_t4_bonus("Engineering", "E005"), {"emp_id": "E005", "department": "Engineering"},
            (_emp("E005", 88000), Probe("retrieve", {"query": "bonus rate Engineering"}, "14.5%"))),
    Variant("T4_chain_03", "ev01",
            "Employee 'E004' (Dave) logged 25 hours of overtime during the last quarter. Based on Dave's salary and his department's HR overtime multiplier, what is Dave's total overtime compensation in USD? (Use the standard formula: base hourly rate = salary / 2000 hours; overtime pay = base hourly rate * department multiplier * overtime hours).",
            lambda: fx.solve_t4_overtime("Marketing", "E004", 25.0), {"emp_id": "E004", "department": "Marketing", "ot_hours": 25.0},
            (_emp("E004", 68000), Probe("retrieve", {"query": "overtime Marketing"}, "1.5x"))),
    # --- T5 file SLA auditing: other services / log files
    Variant("T5_audit_01", "ev01",
            "Auditing 'logs/cluster_events.json' against the latency thresholds in 'sla_reference.json': what percentage of requests for 'WorkerNode' exceeded the SLA threshold? Round to 1 decimal place.",
            lambda: fx.solve_t5_violation_rate("logs/cluster_events.json", "WorkerNode"), {"log": "logs/cluster_events.json", "service": "WorkerNode"},
            (Probe("read_file", {"filename": "logs/cluster_events.json"}, "WorkerNode"), Probe("read_file", {"filename": "sla_reference.json"}, "WorkerNode"))),
    Variant("T5_audit_01", "ev02",
            "Auditing 'logs/cluster_events.json' against the latency thresholds in 'sla_reference.json': what percentage of requests for 'IngressRouter' exceeded the SLA threshold? Round to 1 decimal place.",
            lambda: fx.solve_t5_violation_rate("logs/cluster_events.json", "IngressRouter"), {"log": "logs/cluster_events.json", "service": "IngressRouter"},
            (Probe("read_file", {"filename": "logs/cluster_events.json"}, "IngressRouter"), Probe("read_file", {"filename": "sla_reference.json"}, "IngressRouter"))),
    Variant("T5_audit_02", "ev01",
            "In 'logs/cluster_events.json', how many requests for 'IngressRouter' strictly breached the SLA maximum latency defined in 'sla_reference.json'? Note: requests exactly equal to the threshold are compliant.",
            lambda: fx.solve_t5_strict_breach_count("logs/cluster_events.json", "IngressRouter"), {"log": "logs/cluster_events.json", "service": "IngressRouter"},
            (Probe("read_file", {"filename": "logs/cluster_events.json"}, "IngressRouter"),)),
    Variant("T5_audit_03", "ev01",
            "In 'logs/cluster_events.json', how many requests for 'WorkerNode' reached or exceeded the maximum latency threshold defined in 'sla_reference.json' (latency_ms >= threshold)?",
            lambda: fx.solve_t5_inclusive_breach_count("logs/cluster_events.json", "WorkerNode"), {"log": "logs/cluster_events.json", "service": "WorkerNode"},
            (Probe("read_file", {"filename": "logs/cluster_events.json"}, "WorkerNode"),)),
    # --- T6 unanswerable: a different missing entity (absence is the evidence)
    Variant("T6_unans_01", "ev01", "What is the annual salary of employee ID 'E777' according to the corporate employee records?",
            None, {"emp_id": "E777"},
            (Probe("query_records", {"operation": "filter", "field": "emp_id", "value": "E777"}, expect_empty=True),)),
)


def check_evidence(variant: Variant) -> tuple[bool, str]:
    """Run every probe against the real tool layer. Returns (ok, detail)."""
    details = []
    for p in variant.evidence:
        res = TOOL_REGISTRY[p.tool](**p.args)
        text = json.dumps(res.output, sort_keys=True) if res.output is not None else ""
        if p.expect_empty:
            ok = res.success and res.output is not None and res.output.get("count", len(res.output.get("rows", []))) == 0
            details.append(f"{p.tool}({p.args}) empty={ok}")
        else:
            ok = res.success and p.must_contain is not None and p.must_contain in text
            details.append(f"{p.tool}({p.args}) contains {p.must_contain!r}={ok}")
        if not ok:
            return False, "; ".join(details)
    return True, "; ".join(details) if details else "no tool evidence needed (all inputs in prompt)"


# Candidates that were considered and REJECTED because the evidence check fails.
# They are kept (and re-checked on every build) to show the check has teeth:
# their solver can compute a gold answer from the fixture tables, but the agent's
# tools cannot reach the needed fact, so the label would silently be wrong.
REJECTED_CANDIDATES: tuple[Variant, ...] = (
    Variant("T4_chain_02", "rej01",
            "Calculate the total post-tax revenue in USD generated by all 'Cloud Storage' transactions recorded in 'q1_sales.csv', applying the official sales tax policy rate for that category.",
            lambda: fx.solve_t4_sales("Cloud Storage"), {"category": "Cloud Storage"},
            (Probe("read_file", {"filename": "q1_sales.csv"}, "Cloud Storage"), Probe("retrieve", {"query": "tax rate Cloud Storage"}, "15.0%"))),
    Variant("T1_multihop_03", "rej01",
            "What is the on-call escalation SLA (in minutes) for the inference runtime project led by Sanya Verma?",
            lambda: fx.solve_t1_multihop("Sanya Verma", "Core AI", "on-call escalation SLA (in minutes)"), {"person": "Sanya Verma"},
            (Probe("retrieve", {"query": "Sanya Verma"}, "PRJ-AI-840"), Probe("retrieve", {"query": "PRJ-AI-840 on-call"}, "20"))),
    Variant("T4_chain_01", "rej01",
            "What is the annual performance bonus payout in USD for employee 'E002' (Bob) based on his department's official bonus rate policy and his salary in the employee records?",
            lambda: fx.solve_t4_bonus("Marketing", "E002"), {"emp_id": "E002", "department": "Marketing"},
            (_emp("E002", 72000), Probe("retrieve", {"query": "bonus rate Marketing"}, "11.0%"))),
)
