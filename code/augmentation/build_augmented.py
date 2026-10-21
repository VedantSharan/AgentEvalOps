"""
Build and verify all augmented tasks (Assignment 1, Part D).

Augmentations
-------------
1. entity_variant     23 tasks. New entities/parameters; gold recomputed by the
                      base-suite solvers; evidence probes run on the real tools.
2. paraphrase         30 tasks (one per base task). Gold unchanged; protected
                      spans must be identical; seeded sample for human spot check.
3. fault_injection    96 tasks = 24 answerable tool-using base tasks x 4 faults on
                      the task's first expected tool:
                        tool_failure/first_call  -> transient: recover, gold kept
                        persistent failure       -> information_outage: retrieve,
                                                    read_file and query_records all
                                                    fail, so expected behaviour
                                                    CHANGES to abstaining (task
                                                    re-labelled unanswerable); or
                                                    calculator_outage for tasks whose
                                                    inputs are all in the prompt
                                                    (gold kept: computable by hand)
                        slow_response/always     -> gold kept (2000 ms delay)
                        format_change/always     -> gold kept (same information)
                      Each fault is executed through FaultInjector and checked.

Every augmented record keeps its provenance: parent_task_id, augmentation type,
parameters, seed, and expected label effect.  Base tasks are never modified.

Usage (from the repository root):
    python -m code.augmentation.build_augmented
Outputs:
    data/augmented/tasks_augmented.jsonl         {"task": TaskRecord, "augmentation": {...}} per line
    data/augmented/tasks_augmented_flat.jsonl    TaskRecords only (entity variants + paraphrases)
    supporting_files/augmentation/label_checks.csv
    supporting_files/augmentation/summary.json
    supporting_files/augmentation/examples.md
    supporting_files/augmentation/paraphrase_spot_check_template.csv   (blank sheet; the human-filled
                                                                       paraphrase_spot_check.csv is never overwritten)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any

from code.agent.tools import TOOL_REGISTRY
from code.augmentation.entity_variants import REJECTED_CANDIDATES, VARIANTS, check_evidence
from code.augmentation.fault_injection import FaultInjector, FaultSpec, canonical
from code.augmentation.paraphrase import paraphrase, spans_preserved
from code.shared.models import EvaluationRule, TaskMetadata, TaskRecord
from code.tasks.revised_generator import compute_content_hash
from code.validation.benchmark_validator import BenchmarkValidator

AUG_SEED = 203
AUG_VERSION = "aug-1.0.0"
SLOW_DELAY_MS = 2000
SPOT_CHECK_N = 10
INFO_TOOLS = ("retrieve", "read_file", "query_records")
# (fault_type, trigger, label); the persistent case is resolved per task in _fault_specs().
FAULT_PLAN = (
    ("tool_failure", "first_call", "transient_failure"),
    ("tool_failure", "always", "persistent_failure"),
    ("slow_response", "always", "slow_response"),
    ("format_change", "always", "format_change"),
)


def _load_tasks(path: Path) -> list[TaskRecord]:
    return [TaskRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_task(parent: TaskRecord, task_id: str, prompt: str, gold: str | None, seed: int, *, unanswerable: bool = False) -> TaskRecord:
    answerable = parent.answerable and not unanswerable
    return TaskRecord(
        task_id=task_id,
        template_family=parent.template_family,
        task_type=parent.task_type,
        prompt=prompt,
        answerable=answerable,
        gold_answer=gold if answerable else None,
        expected_tools=list(parent.expected_tools),
        evaluation_rule=parent.evaluation_rule if answerable else EvaluationRule.unanswerable,
        metadata=TaskMetadata(
            provenance="code/augmentation/build_augmented.py",
            # Inherit the parent's creation time: an augmented task is a deterministic
            # function of its parent, and this keeps the output byte-for-byte reproducible.
            created_at=parent.metadata.created_at if parent.metadata else "",
            generator_version=AUG_VERSION,
            seed=seed,
            content_hash=compute_content_hash(task_id, prompt, gold if answerable else None),
        ),
    )


def _probe_args(tool: str, task: TaskRecord) -> dict[str, Any]:
    """Task-specific arguments for exercising the faulted tool in the checks."""
    import re
    if tool == "read_file":
        files = re.findall(r"'([\w/]+\.(?:csv|json))'", task.prompt)
        return {"filename": files[0] if files else "sla_reference.json"}
    if tool == "retrieve":
        return {"query": {"multi_hop_retrieval": "travel", "cross_tool_chaining": "policy"}.get(task.template_family, "policy")}
    if tool == "calculate":
        nums = re.findall(r"\d+(?:\.\d+)?", task.prompt.replace(",", ""))
        return {"expression": f"{nums[0]} + {nums[1]}" if len(nums) > 1 else "1 + 1"}
    if tool == "query_records":
        ids = re.findall(r"'(E\d{3})'", task.prompt)
        return {"operation": "filter", "field": "emp_id", "value": ids[0] if ids else "E001"}
    raise ValueError(tool)


def _fault_specs(parent: TaskRecord, fault_type: str, trigger: str, label: str) -> tuple[list[FaultSpec], str, str, str]:
    """Return (specs, label, expected_behaviour, label_effect) for one fault on one task."""
    tool = parent.expected_tools[0]
    if label != "persistent_failure":
        delay = SLOW_DELAY_MS if fault_type == "slow_response" else 0
        expected = "recover_and_answer_gold" if label == "transient_failure" else "answer_gold"
        return [FaultSpec(fault_type, tool, trigger, delay)], label, expected, "gold_preserved"
    if not set(parent.expected_tools) & set(INFO_TOOLS):
        # Every input is in the prompt (e.g. T2 arithmetic): the answer is still derivable
        # without the calculator, so the gold label is kept.
        return ([FaultSpec("tool_failure", t, "always") for t in parent.expected_tools],
                "calculator_outage", "answer_gold_without_failed_tool", "gold_preserved")
    # Evidence lives behind the information tools; take ALL of them down so that no
    # alternate path (e.g. read_file('project_registry.json') instead of retrieve) exists.
    return ([FaultSpec("tool_failure", t, "always") for t in INFO_TOOLS],
            "information_outage", "abstain_report_tool_unavailable", "expected_behaviour_changed")


def check_fault(specs: list[FaultSpec], task: TaskRecord) -> tuple[bool, str]:
    details, all_ok = [], True
    for spec in specs:
        args = _probe_args(spec.tool, task)
        clean = TOOL_REGISTRY[spec.tool](**args)
        if not clean.success:
            return False, f"clean {spec.tool} call failed: {clean.error}"
        slept: list[float] = []
        with FaultInjector(specs, sleep=slept.append) as inj:
            calls = [TOOL_REGISTRY[spec.tool](**args) for _ in range(3)]
        restored = TOOL_REGISTRY[spec.tool].__name__ == spec.tool
        if spec.fault_type == "tool_failure" and spec.trigger == "first_call":
            ok = (not calls[0].success) and calls[1].success and calls[1].output == clean.output and len(inj.log) == 1
            d = f"{spec.tool}: call1 failed={not calls[0].success}, call2 identical to clean={calls[1].output == clean.output}"
        elif spec.fault_type == "tool_failure":
            ok = not any(c.success for c in calls)
            d = f"{spec.tool}: 3/3 calls failed={ok}"
        elif spec.fault_type == "slow_response":
            ok = all(c.output == clean.output for c in calls) and slept == [spec.delay_ms / 1000] * 3 \
                and all((c.latency_ms or 0) >= spec.delay_ms for c in calls)
            d = f"{spec.tool}: output identical and +{spec.delay_ms} ms on 3/3 calls={ok}"
        else:
            changed = all(c.output != clean.output for c in calls)
            same = all(canonical(spec.tool, c.output) == canonical(spec.tool, clean.output) for c in calls)
            ok = changed and same
            d = f"{spec.tool}: representation changed={changed}, canonical content identical={same}"
        all_ok &= ok and restored
        details.append(d + f", registry restored={restored}")
    return all_ok, "; ".join(details)


def build(tasks_path: Path, out_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    base = {t.task_id: t for t in _load_tasks(tasks_path)}
    records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def add_check(aug_id: str, aug_type: str, name: str, ok: bool, detail: str) -> None:
        checks.append({"aug_task_id": aug_id, "augmentation_type": aug_type, "check": name, "passed": ok, "detail": detail})

    # 1. Entity / parameter variants -------------------------------------------------
    for i, v in enumerate(VARIANTS):
        parent = base[v.parent_task_id]
        gold = v.solve() if v.solve else None
        task = _make_task(parent, v.task_id, v.prompt, gold, AUG_SEED)
        add_check(v.task_id, "entity_variant", "gold_recomputed_by_solver",
                  (gold is None) == (not parent.answerable) and (v.solve is None or v.solve() == gold),
                  f"gold={gold!r} (parent gold={parent.gold_answer!r}); solver deterministic")
        ok, detail = check_evidence(v)
        add_check(v.task_id, "entity_variant", "evidence_available_in_tools" if parent.answerable else "entity_absent_in_tools", ok, detail)
        add_check(v.task_id, "entity_variant", "prompt_differs_from_parent", v.prompt != parent.prompt, "")
        records.append({"task": task.model_dump(mode="json"), "augmentation": {
            "aug_task_id": v.task_id, "type": "entity_variant", "parent_task_id": parent.task_id,
            "template_family": parent.template_family, "params": v.params, "seed": None,
            "label_effect": "gold_recomputed", "expected_behavior": "answer_gold" if parent.answerable else "abstain",
            "fault": None}})

    rejected = []
    for v in REJECTED_CANDIDATES:
        ok, detail = check_evidence(v)
        rejected.append({"candidate_task_id": v.task_id, "solver_gold": v.solve() if v.solve else None,
                         "evidence_check_passed": ok, "detail": detail,
                         "decision": "REJECTED - fact not reachable with the agent's tools" if not ok else "UNEXPECTED PASS"})

    # 2. Paraphrases -------------------------------------------------------------------
    para_rows = []
    for i, parent in enumerate(base.values()):
        seed = AUG_SEED * 1000 + i
        text, info = paraphrase(parent.prompt, seed)
        aug_id = f"{parent.task_id}__pp01"
        task = _make_task(parent, aug_id, text, parent.gold_answer, seed)
        ok, detail = spans_preserved(parent.prompt, text)
        add_check(aug_id, "paraphrase", "protected_spans_identical", ok, detail)
        add_check(aug_id, "paraphrase", "prompt_differs_from_parent", text != parent.prompt, "; ".join(info["operations"]))
        add_check(aug_id, "paraphrase", "gold_unchanged", task.gold_answer == parent.gold_answer, f"gold={parent.gold_answer!r}")
        records.append({"task": task.model_dump(mode="json"), "augmentation": {
            "aug_task_id": aug_id, "type": "paraphrase", "parent_task_id": parent.task_id,
            "template_family": parent.template_family, "params": {"operations": info["operations"]}, "seed": seed,
            "label_effect": "gold_preserved", "expected_behavior": "answer_gold" if parent.answerable else "abstain",
            "fault": None}})
        para_rows.append({"aug_task_id": aug_id, "original_prompt": parent.prompt, "paraphrase": text,
                          "gold_answer": parent.gold_answer, "intent_preserved (yes/no)": "", "reviewer_id": "", "comment": ""})

    # 3. Fault injection --------------------------------------------------------------
    for parent in base.values():
        if not parent.answerable or not parent.expected_tools:
            continue
        for fault_type, trigger, plan_label in FAULT_PLAN:
            specs, label, expected, effect = _fault_specs(parent, fault_type, trigger, plan_label)
            aug_id = f"{parent.task_id}__fi_{label}"
            changed = effect == "expected_behaviour_changed"
            task = _make_task(parent, aug_id, parent.prompt, parent.gold_answer, AUG_SEED, unanswerable=changed)
            ok, detail = check_fault(specs, parent)
            add_check(aug_id, "fault_injection", f"fault_behaves_as_specified:{label}", ok, detail)
            if changed:
                needs_tools = bool(set(parent.expected_tools) & set(INFO_TOOLS))
                # Same convention as Part-B check 7: answers of <= 2 characters (e.g. a count of
                # "5") are too short to test for leakage meaningfully; longer ones must not
                # appear as a whole word in the prompt.
                g = (parent.gold_answer or "").strip().lower()
                gold_not_in_prompt = len(g) <= 2 or not re.search(r"(?<![\w.])" + re.escape(g) + r"(?![\w.])", parent.prompt.lower())
                add_check(aug_id, "fault_injection", "evidence_unreachable_so_abstain_is_correct",
                          needs_tools and gold_not_in_prompt and task.gold_answer is None,
                          f"task needs {sorted(set(parent.expected_tools) & set(INFO_TOOLS))}; all info tools down; "
                          f"gold {parent.gold_answer!r} not derivable from prompt={gold_not_in_prompt}"
                          + (" (<=2 chars: leakage test n/a, Part-B check-7 convention)" if len(g) <= 2 else "")
                          + "; re-labelled unanswerable")
            else:
                add_check(aug_id, "fault_injection", "gold_preserved", task.gold_answer == parent.gold_answer,
                          f"gold kept = {parent.gold_answer!r}")
            records.append({"task": task.model_dump(mode="json"), "augmentation": {
                "aug_task_id": aug_id, "type": "fault_injection", "parent_task_id": parent.task_id,
                "template_family": parent.template_family, "params": {"fault_label": label}, "seed": AUG_SEED,
                "label_effect": effect, "expected_behavior": expected, "fault": [sp.to_dict() for sp in specs]}})

    # Write outputs ---------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tasks_augmented.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    flat = [r["task"] for r in records if r["augmentation"]["type"] != "fault_injection"]
    with (out_dir / "tasks_augmented_flat.jsonl").open("w", encoding="utf-8") as fh:
        for t in flat:
            fh.write(json.dumps(t) + "\n")

    # Part-B validation suite on the prompt-changing augmentations
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "no_trajectories.jsonl"
        empty.write_text("", encoding="utf-8")
        suite = BenchmarkValidator(out_dir / "tasks_augmented_flat.jsonl", empty).run_all_checks()
    validator = {"passed": suite.passed, "checks_passed": suite.checks_passed, "checks_failed": suite.checks_failed,
                 "tasks_count": suite.tasks_count, "summary_by_check": suite.summary_by_check,
                 "issues": [i.model_dump() for i in suite.issues]}

    with (evidence_dir / "label_checks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(checks[0].keys()))
        w.writeheader()
        w.writerows(checks)

    with (evidence_dir / "rejected_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rejected[0].keys()))
        w.writeheader()
        w.writerows(rejected)

    rng = random.Random(AUG_SEED)
    sample = rng.sample(para_rows, SPOT_CHECK_N)
    # The human-filled sheet (paraphrase_spot_check.csv) is never overwritten; the generated
    # blank sheet goes to *_template.csv and is copied once if no human sheet exists yet.
    with (evidence_dir / "paraphrase_spot_check_template.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    human_sheet = evidence_dir / "paraphrase_spot_check.csv"
    if not human_sheet.exists():
        human_sheet.write_bytes((evidence_dir / "paraphrase_spot_check_template.csv").read_bytes())

    _write_examples(evidence_dir / "examples.md", records, base)

    by_type: dict[str, dict[str, int]] = {}
    for r in records:
        t = r["augmentation"]["type"]
        by_type.setdefault(t, {"tasks": 0})["tasks"] += 1
    for c in checks:
        d = by_type[c["augmentation_type"]]
        d["checks_passed"] = d.get("checks_passed", 0) + int(c["passed"])
        d["checks_failed"] = d.get("checks_failed", 0) + int(not c["passed"])
    summary = {
        "augmentation_version": AUG_VERSION,
        "seed": AUG_SEED,
        "base_tasks": len(base),
        "augmented_tasks": len(records),
        "by_type": by_type,
        "fault_label_effects": {e: sum(r["augmentation"]["label_effect"] == e for r in records if r["augmentation"]["type"] == "fault_injection")
                                for e in ("gold_preserved", "expected_behaviour_changed")},
        "all_checks_passed": all(c["passed"] for c in checks),
        "rejected_candidates": len(rejected),
        "rejected_candidates_correctly_failed": all(not r["evidence_check_passed"] for r in rejected),
        "validation_suite_on_flat_tasks": validator,
    }
    (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _write_examples(path: Path, records: list[dict], base: dict[str, TaskRecord]) -> None:
    pick = {"entity_variant": ["T4_chain_01__ev01", "T3_struct_03__ev01"],
            "paraphrase": ["T3_struct_02__pp01", "T4_chain_02__pp01"],
            "fault_injection": ["T3_struct_03__fi_format_change", "T2_calc_01__fi_calculator_outage", "T4_chain_01__fi_information_outage"]}
    by_id = {r["augmentation"]["aug_task_id"]: r for r in records}
    lines = ["# Augmentation examples (generated by code/augmentation/build_augmented.py)", ""]
    for kind, ids in pick.items():
        lines.append(f"## {kind}")
        for aid in ids:
            r = by_id[aid]
            p = base[r["augmentation"]["parent_task_id"]]
            lines += [f"### {aid}", "",
                      f"- **Parent** `{p.task_id}` gold `{p.gold_answer}`: {p.prompt}",
                      f"- **Augmented** gold `{r['task']['gold_answer']}` (rule `{r['task']['evaluation_rule']}`): {r['task']['prompt']}",
                      f"- **Label effect:** {r['augmentation']['label_effect']}; expected behaviour: {r['augmentation']['expected_behavior']}",
                      f"- **Params:** `{json.dumps(r['augmentation']['params'])}`" + (f"; fault `{json.dumps(r['augmentation']['fault'])}`" if r["augmentation"]["fault"] else ""),
                      ""]
    # Show what a format-changed tool output actually looks like
    from code.augmentation.fault_injection import change_format
    clean = TOOL_REGISTRY["read_file"](filename="server_metrics.csv").output
    lines += ["### What the agent sees under `format_change` on `read_file('server_metrics.csv')`", "",
              "Clean output (`content` field, first 2 lines):", "```", "\n".join(clean["content"].splitlines()[:2]), "```",
              "Fault-injected output (same rows as JSON records, different keys):", "```",
              json.dumps(change_format("read_file", clean))[:300] + " ...", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate and verify augmented tasks.")
    p.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/augmented"))
    p.add_argument("--evidence-dir", type=Path, default=Path("supporting_files/augmentation"))
    a = p.parse_args()
    s = build(a.tasks, a.out_dir, a.evidence_dir)
    print(json.dumps({k: s[k] for k in ("base_tasks", "augmented_tasks", "by_type", "fault_label_effects", "all_checks_passed")}, indent=2))
    print("Validation suite on augmented (non-fault) tasks:", s["validation_suite_on_flat_tasks"])
    print("Rejected candidates correctly failed evidence check:", s["rejected_candidates_correctly_failed"])
    if not s["all_checks_passed"] or not s["rejected_candidates_correctly_failed"]:
        raise SystemExit("Some augmentation checks FAILED - see supporting_files/augmentation/label_checks.csv")


if __name__ == "__main__":
    main()
