"""
Revised benchmark task fixtures, synthetic corpora, virtual files, and programmatic solvers.

Provides six core families aligned with the task suite proposal:
  T1: Multi-Hop Document Retrieval & QA (5 instances, Train)
  T2: Multi-Step Numerical & Compound Calculation (5 instances, Train)
  T3: Structured Tabular/JSON Filtering & Aggregation (5 instances, Train)
  T4: Cross-Tool Chaining: Retrieval -> Table Lookup -> Calculator (5 instances, Validation)
  T5: Log/File Auditing & Threshold Verification (5 instances, Test / Held-Out)
  T6: Ambiguous & Unanswerable Tasks (5 instances, Stratified)
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

from code.tasks.fixtures import EMPLOYEE_RECORDS

# ---------------------------------------------------------------------------
# Structured Knowledge Base (for T1, T4, T6)
# ---------------------------------------------------------------------------

REVISED_KB_DOCS: dict[str, Any] = {
    "directory": {
        "Cloud Infrastructure": {
            "Aarav Patel": "PRJ-INF-932",
        },
        "Security Ops": {
            "Meera Nair": "PRJ-SEC-581",
        },
        "Data Platform": {
            "Rohan Deshmukh": "PRJ-DAT-714",
        },
        "Analytics": {
            "Rohan Mehta": "PRJ-ANA-204",
        },
        "Core AI": {
            "Sanya Verma": "PRJ-AI-840",
        },
    },
    "projects": {
        "PRJ-INF-932": {
            "name": "Cloud Mesh Engine",
            "lead": "Aarav Patel",
            "department": "Cloud Infrastructure",
            "status": "active",
            "security compliance tier": "Tier-1 Enforced",
            "hardware_base_budget": 50000,
        },
        "PRJ-INF-102": {
            "name": "Legacy Proxy",
            "lead": "Aarav Patel",
            "department": "Cloud Infrastructure",
            "status": "archived",
            "security compliance tier": "Tier-3 Standard",
            "hardware_base_budget": 20000,
        },
        "PRJ-SEC-581": {
            "name": "Sentinel Perimeter",
            "lead": "Meera Nair",
            "department": "Security Ops",
            "status": "active",
            "data retention policy (in days)": "730",
            "primary compliance tier": "Tier-1 High",
        },
        "PRJ-DAT-714": {
            "name": "HyperStream Lakehouse",
            "lead": "Rohan Deshmukh",
            "department": "Data Platform",
            "status": "active",
            "on-call escalation SLA (in minutes)": "15",
            "data retention window (in days)": "90",
            "hardware_base_budget": 35000,
        },
        "PRJ-ANA-204": {
            "name": "Legacy Batch ETL",
            "lead": "Rohan Mehta",
            "department": "Analytics",
            "status": "active",
            "on-call escalation SLA (in minutes)": "60",
        },
        "PRJ-AI-840": {
            "name": "Neural Inference Runtime",
            "lead": "Sanya Verma",
            "department": "Core AI",
            "status": "active",
            "cluster node quota": "64 nodes",
            "on-call escalation SLA (in minutes)": "20",
        },
    },
    "travel_policy": {
        "2024_v1": {
            "Cloud Infrastructure": "180 USD",
            "status": "superseded",
            "effective_date": "2024-03-01",
        },
        "2025_v2": {
            "Cloud Infrastructure": "240 USD",
            "status": "current",
            "effective_date": "2025-01-01",
        },
    },
    "tax_policy": {
        "Enterprise SaaS": 18.0,
        "Hardware Peripherals": 12.0,
        "Cloud Storage": 15.0,
        "Consulting Services": 10.0,
    },
    "bonus_policy": {
        "Engineering": 14.5,
        "Marketing": 11.0,
        "Security": 12.5,
        "Operations": 10.0,
    },
    "overtime_policy": {
        "Marketing": 1.5,
        "Operations": 1.5,
        "Support": 1.25,
    },
    "benefits_policy": {
        "Senior": 6.0,
        "Lead": 7.5,
        "Staff": 8.0,
    },
    "infrastructure_policy": {
        "Cloud Infrastructure": 12.0,  # 12.0% equipment subsidy
        "Data Platform": 10.0,
        "Core AI": 15.0,
    },
    "sla_limits": {
        "AuthService": 200,
        "PaymentGateway": 350,
        "SearchIndexer": 500,
        "UserProfileAPI": 150,
        "OrderDispatcher": 400,
        "IngressRouter": 100,
        "WorkerNode": 250,
    },
}

# Facts exposed to the retrieve tool
REVISED_FACTS: list[dict[str, str]] = [
    {
        "fact_id": "rf01",
        "topic": "directory",
        "question": "Who leads active project PRJ-INF-932 in Cloud Infrastructure?",
        "answer": "Aarav Patel leads active project PRJ-INF-932 (Cloud Mesh Engine) in Cloud Infrastructure.",
    },
    {
        "fact_id": "rf02",
        "topic": "directory",
        "question": "What is the legacy project for Aarav Patel in Cloud Infrastructure?",
        "answer": "Aarav Patel previously led archived project PRJ-INF-102 (Legacy Proxy) in Cloud Infrastructure.",
    },
    {
        "fact_id": "rf03",
        "topic": "directory",
        "question": "Which project is led by Meera Nair in Security Ops?",
        "answer": "Meera Nair leads project PRJ-SEC-581 (Sentinel Perimeter) in Security Ops.",
    },
    {
        "fact_id": "rf04",
        "topic": "directory",
        "question": "Which streaming project is led by Rohan Deshmukh in Data Platform?",
        "answer": "Rohan Deshmukh leads project PRJ-DAT-714 (HyperStream Lakehouse) in Data Platform.",
    },
    {
        "fact_id": "rf05",
        "topic": "directory",
        "question": "Which batch project is led by Rohan Mehta in Analytics?",
        "answer": "Rohan Mehta leads project PRJ-ANA-204 (Legacy Batch ETL) in Analytics.",
    },
    {
        "fact_id": "rf06",
        "topic": "directory",
        "question": "Which project is led by Sanya Verma in Core AI?",
        "answer": "Sanya Verma leads project PRJ-AI-840 (Neural Inference Runtime) in Core AI.",
    },
    {
        "fact_id": "rf07",
        "topic": "project_specs",
        "question": "What is the approved security compliance tier for active project PRJ-INF-932?",
        "answer": "Tier-1 Enforced",
    },
    {
        "fact_id": "rf08",
        "topic": "project_specs",
        "question": "What is the security compliance tier for archived project PRJ-INF-102?",
        "answer": "Tier-3 Standard",
    },
    {
        "fact_id": "rf09",
        "topic": "project_specs",
        "question": "What is the primary data retention policy (in days) for project PRJ-SEC-581?",
        "answer": "730",
    },
    {
        "fact_id": "rf10",
        "topic": "project_specs",
        "question": "What is the designated on-call escalation SLA (in minutes) for project PRJ-DAT-714?",
        "answer": "15",
    },
    {
        "fact_id": "rf11",
        "topic": "project_specs",
        "question": "What is the on-call escalation SLA (in minutes) for project PRJ-ANA-204?",
        "answer": "60",
    },
    {
        "fact_id": "rf12",
        "topic": "project_specs",
        "question": "What is the cluster node quota for project PRJ-AI-840?",
        "answer": "64 nodes",
    },
    {
        "fact_id": "rf13",
        "topic": "project_specs",
        "question": "What is the hardware base budget for project PRJ-INF-932?",
        "answer": "50000 USD",
    },
    {
        "fact_id": "rf14",
        "topic": "policy",
        "question": "What is the annual performance bonus rate for the Engineering department in company policy?",
        "answer": "The annual performance bonus rate for the Engineering department is 14.5%.",
    },
    {
        "fact_id": "rf15",
        "topic": "policy",
        "question": "What is the standard tax rate for Enterprise SaaS in sales tax policy?",
        "answer": "The standard tax rate for Enterprise SaaS is 18.0%.",
    },
    {
        "fact_id": "rf16",
        "topic": "policy",
        "question": "What is the standard tax rate for Consulting Services in sales tax policy?",
        "answer": "The standard tax rate for Consulting Services is 10.0%.",
    },
    {
        "fact_id": "rf17",
        "topic": "policy",
        "question": "What is the overtime hourly rate multiplier for the Marketing department?",
        "answer": "The overtime hourly rate multiplier for the Marketing department is 1.5x.",
    },
    {
        "fact_id": "rf18",
        "topic": "policy",
        "question": "What is the 401k employer match percentage for employee seniority tier Senior?",
        "answer": "The 401k employer match percentage for Senior tier is 6.0%.",
    },
    {
        "fact_id": "rf19",
        "topic": "policy",
        "question": "What is the equipment subsidy allowance percentage for Cloud Infrastructure projects?",
        "answer": "The equipment subsidy allowance percentage for Cloud Infrastructure projects is 12.0%.",
    },
    {
        "fact_id": "rf20",
        "topic": "directory",
        "question": "Who is Rahul Sharma in the staff directory?",
        "answer": "There are multiple employees named Rahul Sharma: Rahul Sharma in Engineering (Systems Architect) and Rahul Sharma in Sales (Regional Account Executive).",
    },
    {
        "fact_id": "rf21",
        "topic": "travel_policy",
        "question": "What is the maximum domestic daily per diem allowance for Cloud Infrastructure in the current 2025 travel policy (v2.1)?",
        "answer": "Under current 2025 travel policy (v2.1), the maximum domestic daily per diem allowance for Cloud Infrastructure is 240 USD.",
    },
    {
        "fact_id": "rf22",
        "topic": "travel_policy",
        "question": "What was the domestic daily per diem allowance for Cloud Infrastructure under the superseded 2024 travel policy (v1.0)?",
        "answer": "Under superseded 2024 travel policy (v1.0), the domestic daily per diem allowance for Cloud Infrastructure was 180 USD.",
    },
    {
        "fact_id": "rf23",
        "topic": "revenue_tracking",
        "question": "Where are Q1 Consulting Services transactions recorded according to the revenue tracking handbook?",
        "answer": "All Q1 Consulting Services transactions were migrated from q1_sales.csv to consulting_ledger.csv.",
    },
]

# ---------------------------------------------------------------------------
# Virtual Files (for T3, T4, T5, T6)
# ---------------------------------------------------------------------------

_SERVER_METRICS_CSV = (
    "server_id,region,status,cpu_load,memory_usage_pct,active_connections\n"
    "SRV-101,US-EAST,ONLINE,78.5,82.0,340\n"
    "SRV-102,US-EAST,ONLINE,62.0,55.0,180\n"
    "SRV-103,US-WEST,OFFLINE,10.0,15.0,0\n"
    "SRV-104,EU-WEST,ONLINE,85.0,76.0,410\n"
    "SRV-105,US-EAST,ONLINE,72.0,74.0,290\n"
    "SRV-106,APAC,ONLINE,91.0,88.0,520\n"
    "SRV-107,US-EAST,ONLINE,88.5,71.0,460\n"
    "SRV-108,EU-WEST,MAINTENANCE,45.0,50.0,50\n"
    "SRV-109,EU-WEST,ONLINE,65.0,68.0,210\n"
    "SRV-110,US-EAST,MAINTENANCE,0.0,20.0,0\n"
    "SRV-111,US-EAST,ONLINE,94.0,85.0,580\n"
    "SRV-112,EU-WEST,ONLINE,75.0,72.0,330\n"
    "SRV-113,US-EAST,ONLINE,70.0,70.0,200\n"
)

_SUPPORT_TICKETS_CSV = (
    "ticket_id,department,priority,resolution_hrs,satisfaction_score\n"
    "TCK-01,Billing,CRITICAL,5.5,3\n"
    "TCK-02,Technical,HIGH,3.0,4\n"
    "TCK-03,Billing,CRITICAL,6.0,2\n"
    "TCK-04,Security,CRITICAL,2.5,5\n"
    "TCK-05,Billing,LOW,1.5,5\n"
    "TCK-06,Billing,CRITICAL,4.5,4\n"
    "TCK-07,Technical,CRITICAL,7.0,1\n"
    "TCK-08,Billing,CRITICAL,6.0,4\n"
    "TCK-09,Billing,HIGH,3.5,\n"
    "TCK-10,Billing,MEDIUM,2.0,5\n"
    "TCK-11,Technical,LOW,1.0,4\n"
    "TCK-12,Billing,HIGH,4.0,2\n"
)

_Q1_SALES_CSV = (
    "sale_id,category,client,revenue,month\n"
    "S-101,Enterprise SaaS,Acme Corp,20000.0,January\n"
    "S-102,Hardware Peripherals,Beta LLC,8000.0,January\n"
    "S-103,Enterprise SaaS,Global Tech,25000.0,February\n"
    "S-104,Cloud Storage,Nova Inc,15000.0,March\n"
    "S-105,Consulting Services,Delta Co,12000.0,March\n"
)

_CONSULTING_LEDGER_CSV = (
    "sale_id,category,client,revenue,month\n"
    "C-201,Consulting Services,Beta Corp,18000.0,January\n"
    "C-202,Consulting Services,Apex Ltd,22000.0,February\n"
)

# 20 log entries for AuthService (5 exceed 200ms -> 25.0%)
_GATEWAY_01_LOGS = json.dumps([
    {"request_id": f"REQ-{i:03d}", "service": "AuthService", "latency_ms": lat, "status": 200}
    for i, lat in enumerate([
        140, 165, 210, 180, 245, 120, 190, 205, 150, 175,
        280, 130, 160, 195, 215, 145, 155, 170, 185, 190
    ], 1)
], indent=2)

# 25 log entries for PaymentGateway (threshold 350; 1 at 350 compliant, 4 strictly > 350)
_PAYMENT_API_LOGS = json.dumps([
    {"request_id": f"PAY-{i:03d}", "service": "PaymentGateway", "latency_ms": lat, "status": 200}
    for i, lat in enumerate([
        290, 310, 360, 280, 320, 410, 305, 295, 330, 315,
        340, 275, 385, 300, 310, 325, 305, 390, 290, 315,
        335, 340, 350, 320, 300
    ], 1)
], indent=2)

# Combined multi-service log: IngressRouter (threshold 100, 6 violations) vs WorkerNode (threshold 250, 2 violations)
_CLUSTER_EVENTS_LOGS = json.dumps([
    {"request_id": "EVT-001", "service": "IngressRouter", "latency_ms": 115, "status": 200},
    {"request_id": "EVT-002", "service": "IngressRouter", "latency_ms": 90, "status": 200},
    {"request_id": "EVT-003", "service": "IngressRouter", "latency_ms": 140, "status": 200},
    {"request_id": "EVT-004", "service": "IngressRouter", "latency_ms": 105, "status": 200},
    {"request_id": "EVT-005", "service": "IngressRouter", "latency_ms": 85, "status": 200},
    {"request_id": "EVT-006", "service": "IngressRouter", "latency_ms": 130, "status": 200},
    {"request_id": "EVT-007", "service": "IngressRouter", "latency_ms": 125, "status": 200},
    {"request_id": "EVT-008", "service": "IngressRouter", "latency_ms": 75, "status": 200},
    {"request_id": "EVT-009", "service": "IngressRouter", "latency_ms": 110, "status": 200},
    {"request_id": "EVT-010", "service": "IngressRouter", "latency_ms": 95, "status": 200},
    {"request_id": "EVT-011", "service": "WorkerNode", "latency_ms": 220, "status": 200},
    {"request_id": "EVT-012", "service": "WorkerNode", "latency_ms": 240, "status": 200},
    {"request_id": "EVT-013", "service": "WorkerNode", "latency_ms": 280, "status": 200},
    {"request_id": "EVT-014", "service": "WorkerNode", "latency_ms": 210, "status": 200},
    {"request_id": "EVT-015", "service": "WorkerNode", "latency_ms": 230, "status": 200},
    {"request_id": "EVT-016", "service": "WorkerNode", "latency_ms": 295, "status": 200},
    {"request_id": "EVT-017", "service": "WorkerNode", "latency_ms": 205, "status": 200},
    {"request_id": "EVT-018", "service": "WorkerNode", "latency_ms": 245, "status": 200},
    {"request_id": "EVT-019", "service": "WorkerNode", "latency_ms": 190, "status": 200},
    {"request_id": "EVT-020", "service": "WorkerNode", "latency_ms": 235, "status": 200},
], indent=2)

# 25 log entries for OrderDispatcher (threshold 400; status >= 400 OR latency > 400 -> exactly 7 requests: 7/25 = 28.0%)
_ORDER_DISPATCH_LOGS = json.dumps([
    {"request_id": f"ORD-{i:03d}", "service": "OrderDispatcher", "latency_ms": lat, "status": st}
    for i, (lat, st) in enumerate([
        (310, 200), (360, 200), (420, 200), (350, 400), (480, 500),
        (330, 200), (380, 200), (410, 200), (340, 200), (370, 200),
        (450, 200), (320, 200), (365, 200), (490, 200), (315, 200),
        (335, 200), (355, 200), (370, 200), (385, 200), (390, 200),
        (345, 503), (375, 200), (360, 200), (350, 200), (310, 200),
    ], 1)
], indent=2)

_EMPTY_SERVICE_LOGS = json.dumps([])

_SLA_REFERENCE_JSON = json.dumps(REVISED_KB_DOCS["sla_limits"], indent=2)

_PROJECT_REGISTRY_JSON = json.dumps(REVISED_KB_DOCS["projects"], indent=2)

REVISED_VIRTUAL_FILES: dict[str, str] = {
    "server_metrics.csv": _SERVER_METRICS_CSV,
    "support_tickets.csv": _SUPPORT_TICKETS_CSV,
    "q1_sales.csv": _Q1_SALES_CSV,
    "consulting_ledger.csv": _CONSULTING_LEDGER_CSV,
    "logs/gateway_01.json": _GATEWAY_01_LOGS,
    "gateway_01.json": _GATEWAY_01_LOGS,
    "logs/payment_api.json": _PAYMENT_API_LOGS,
    "payment_api.json": _PAYMENT_API_LOGS,
    "logs/cluster_events.json": _CLUSTER_EVENTS_LOGS,
    "cluster_events.json": _CLUSTER_EVENTS_LOGS,
    "logs/order_dispatch.json": _ORDER_DISPATCH_LOGS,
    "order_dispatch.json": _ORDER_DISPATCH_LOGS,
    "logs/empty_service.json": _EMPTY_SERVICE_LOGS,
    "empty_service.json": _EMPTY_SERVICE_LOGS,
    "sla_reference.json": _SLA_REFERENCE_JSON,
    "project_registry.json": _PROJECT_REGISTRY_JSON,
}


# ---------------------------------------------------------------------------
# Programmatic Solvers (Ground-Truth Computations)
# ---------------------------------------------------------------------------

def solve_t1_travel(version: str = "2025_v2") -> str:
    """Solve T1 travel per diem lookup for Cloud Infrastructure."""
    return str(REVISED_KB_DOCS["travel_policy"][version]["Cloud Infrastructure"])


def solve_t1_multihop(person_name: str, department: str, target_attribute: str, status: str = "active") -> str:
    """Solve T1 multi-hop retrieval directly from directory & project docs."""
    for proj_id, pdata in REVISED_KB_DOCS["projects"].items():
        if (
            pdata.get("lead") == person_name
            and pdata.get("department") == department
            and pdata.get("status") == status
        ):
            return str(pdata[target_attribute])
    raise KeyError(f"Project not found for {person_name} ({department}, {status})")


def solve_t1_replication_factor() -> str:
    """Solve T1 direct governance directive replication factor."""
    return "3"


def solve_t2_standard_overtime(regular_hours: int, overtime_hours: int, base_rate: float, multiplier: float) -> str:
    """Solve T2 standard overtime billing."""
    regular_pay = regular_hours * base_rate
    overtime_pay = overtime_hours * (base_rate * multiplier)
    total = round(regular_pay + overtime_pay, 2)
    return f"{total:.2f}"


def solve_t2_zero_overtime(regular_hours: int, weekend_hours: int, base_rate: float) -> str:
    """Solve T2 labor billing under promotional 0.0% surcharge (regular rate only)."""
    total = (regular_hours + weekend_hours) * base_rate
    return f"{total:.2f}"


def solve_t2_storage(
    total_gb: float,
    tier1_limit: float,
    tier1_rate: float,
    overage_rate: float,
    read_requests: int,
    request_rate_per_10k: float,
) -> str:
    """Solve T2 multi-tier storage and request charge."""
    base_charge = min(total_gb, tier1_limit) * tier1_rate
    overage_gb = max(0.0, total_gb - tier1_limit)
    overage_charge = overage_gb * overage_rate
    request_charge = (read_requests / 10000.0) * request_rate_per_10k
    total = round(base_charge + overage_charge + request_charge, 2)
    return f"{total:.2f}"


def solve_t2_bandwidth(
    size_tb: float,
    bandwidth_mbps: float,
    efficiency_pct: float,
) -> str:
    """Solve T2 network replication transfer time in minutes, rounded up."""
    size_mbits = size_tb * 1000.0 * 1000.0 * 8.0
    effective_bandwidth = bandwidth_mbps * (efficiency_pct / 100.0)
    seconds = size_mbits / effective_bandwidth
    minutes = math.ceil(seconds / 60.0)
    return str(minutes)


def solve_t2_depreciation(initial_cost: float, rate_yr1: float, rate_yr2: float) -> str:
    """Solve T2 accelerated depreciation book value at Year 2."""
    val_yr1 = initial_cost * (1.0 - rate_yr1 / 100.0)
    val_yr2 = round(val_yr1 * (1.0 - rate_yr2 / 100.0), 2)
    return f"{val_yr2:.2f}"


def solve_t3_emp_salary(emp_id: str) -> str:
    """Retrieve annual salary for employee ID from EMPLOYEE_RECORDS."""
    emp = next(e for e in EMPLOYEE_RECORDS if e["emp_id"] == emp_id)
    return str(emp["salary"])


def solve_t3_server_memory_threshold(region: str, status: str, threshold: float, strict: bool = True) -> str:
    """Count servers in server_metrics.csv matching region, status, and memory threshold."""
    reader = csv.DictReader(io.StringIO(_SERVER_METRICS_CSV))
    count = 0
    for r in reader:
        if r["region"] == region and r["status"] == status:
            val = float(r["memory_usage_pct"])
            if strict and val > threshold:
                count += 1
            elif not strict and val >= threshold:
                count += 1
    return str(count)


def solve_t3_ticket_dept_max(department: str) -> str:
    """Find ticket_id of ticket in department with max resolution_hrs, tie-break on smaller numeric id."""
    reader = list(csv.DictReader(io.StringIO(_SUPPORT_TICKETS_CSV)))
    dept_tickets = [r for r in reader if r["department"] == department]
    best = max(dept_tickets, key=lambda x: (float(x["resolution_hrs"]), -int(x["ticket_id"].split("-")[1])))
    return best["ticket_id"]


def solve_t3_ticket_negative_count(department: str, excluded_score: str = "5") -> str:
    """Count tickets for department where satisfaction_score is non-empty and != excluded_score."""
    reader = csv.DictReader(io.StringIO(_SUPPORT_TICKETS_CSV))
    count = 0
    for r in reader:
        if r["department"] == department:
            score = r["satisfaction_score"].strip()
            if score and score != excluded_score:
                count += 1
    return str(count)


def solve_t4_bonus(department: str, emp_id: str) -> str:
    """Compute bonus = salary * bonus_rate."""
    rate_pct = REVISED_KB_DOCS["bonus_policy"][department]
    emp = next(e for e in EMPLOYEE_RECORDS if e["emp_id"] == emp_id)
    bonus = round(emp["salary"] * (rate_pct / 100.0), 2)
    return f"{bonus:.2f}"


def solve_t4_sales(category: str) -> str:
    """Compute post-tax sales revenue = revenue * (1 + tax_rate)."""
    tax_rate_pct = REVISED_KB_DOCS["tax_policy"][category]
    reader = csv.DictReader(io.StringIO(_Q1_SALES_CSV))
    raw_rev = sum(float(r["revenue"]) for r in reader if r["category"] == category)
    post_tax = round(raw_rev * (1.0 + tax_rate_pct / 100.0), 2)
    return f"{post_tax:.2f}"


def solve_t4_overtime(department: str, emp_id: str, ot_hours: float) -> str:
    """Compute overtime pay = (salary / 2000) * multiplier * ot_hours."""
    multiplier = REVISED_KB_DOCS["overtime_policy"][department]
    emp = next(e for e in EMPLOYEE_RECORDS if e["emp_id"] == emp_id)
    base_hourly = emp["salary"] / 2000.0
    ot_pay = round(base_hourly * multiplier * ot_hours, 2)
    return f"{ot_pay:.2f}"


def solve_t4_subsidy(dept_key: str, project_id: str) -> str:
    """Compute subsidized budget = hardware_base_budget * (1 + subsidy_pct/100)."""
    subsidy_pct = REVISED_KB_DOCS["infrastructure_policy"][dept_key]
    base_budget = REVISED_KB_DOCS["projects"][project_id]["hardware_base_budget"]
    total = round(base_budget * (1.0 + subsidy_pct / 100.0), 2)
    return f"{total:.2f}"


def solve_t4_consulting_migrated(category: str = "Consulting Services") -> str:
    """Compute post-tax consulting revenue from migrated consulting_ledger.csv."""
    tax_rate_pct = REVISED_KB_DOCS["tax_policy"][category]
    reader = csv.DictReader(io.StringIO(_CONSULTING_LEDGER_CSV))
    raw_rev = sum(float(r["revenue"]) for r in reader if r["category"] == category)
    post_tax = round(raw_rev * (1.0 + tax_rate_pct / 100.0), 2)
    return f"{post_tax:.2f}"


def solve_t5_violation_rate(log_filename: str, service_name: str) -> str:
    """Compute violation percentage for service in log file against sla_limits."""
    sla_limit = REVISED_KB_DOCS["sla_limits"][service_name]
    log_content = REVISED_VIRTUAL_FILES[log_filename]
    entries = json.loads(log_content)
    service_logs = [e for e in entries if e["service"] == service_name]
    if not service_logs:
        return "0.0"
    violations = sum(1 for e in service_logs if e["latency_ms"] > sla_limit)
    pct = round((violations / len(service_logs)) * 100.0, 1)
    return f"{pct:.1f}"


def solve_t5_strict_breach_count(log_filename: str, service_name: str) -> str:
    """Count requests strictly exceeding sla_limit (equality is compliant)."""
    sla_limit = REVISED_KB_DOCS["sla_limits"][service_name]
    log_content = REVISED_VIRTUAL_FILES[log_filename]
    entries = json.loads(log_content)
    service_logs = [e for e in entries if e["service"] == service_name]
    breaches = sum(1 for e in service_logs if e["latency_ms"] > sla_limit)
    return str(breaches)


def solve_t5_inclusive_breach_count(log_filename: str, service_name: str) -> str:
    """Count requests reaching or exceeding sla_limit (latency_ms >= sla_limit)."""
    sla_limit = REVISED_KB_DOCS["sla_limits"][service_name]
    log_content = REVISED_VIRTUAL_FILES[log_filename]
    entries = json.loads(log_content)
    service_logs = [e for e in entries if e["service"] == service_name]
    breaches = sum(1 for e in service_logs if e["latency_ms"] >= sla_limit)
    return str(breaches)


def solve_t5_worst_service(log_filename: str, service_a: str, service_b: str) -> str:
    """Identify which service had more SLA breaches."""
    log_content = REVISED_VIRTUAL_FILES[log_filename]
    entries = json.loads(log_content)
    lim_a = REVISED_KB_DOCS["sla_limits"][service_a]
    lim_b = REVISED_KB_DOCS["sla_limits"][service_b]
    breaches_a = sum(1 for e in entries if e["service"] == service_a and e["latency_ms"] > lim_a)
    breaches_b = sum(1 for e in entries if e["service"] == service_b and e["latency_ms"] > lim_b)
    return service_a if breaches_a > breaches_b else service_b


def solve_t5_failure_rate(log_filename: str, service_name: str, min_error_status: int = 400) -> str:
    """Compute percentage of requests with status >= min_error_status OR latency > sla_limit."""
    sla_limit = REVISED_KB_DOCS["sla_limits"][service_name]
    log_content = REVISED_VIRTUAL_FILES[log_filename]
    entries = json.loads(log_content)
    service_logs = [e for e in entries if e["service"] == service_name]
    if not service_logs:
        return "0.0"
    failures = sum(
        1 for e in service_logs
        if e["status"] >= min_error_status or e["latency_ms"] > sla_limit
    )
    pct = round((failures / len(service_logs)) * 100.0, 1)
    return f"{pct:.1f}"
