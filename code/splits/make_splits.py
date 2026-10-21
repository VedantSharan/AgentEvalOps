"""
Leakage-free splits by template family, shifted split, and gold test set (Assignment 1, Part E).

Unit of splitting: ``template_family`` (as required for Project 1).  Every item
inherits the split of its family:
  * base tasks                       -> by their family
  * entity variants and paraphrases  -> same split as their parent task
  * trajectories (incl. repeated runs of one task) and their human labels -> by their task
  * fault-injected tasks             -> the SHIFTED split (evaluation-only, for drift
                                        experiments); each row records its parent's split

Family assignment follows the plan declared in code/tasks/revised_generator.py
(T1-T3 train, T4 validation, T5 held-out test).  That plan called T6 "stratified",
which would put one family in several splits and break the family hold-out rule,
so T6 (ambiguous_unanswerable) is assigned to TEST as a whole.  The assignment is
deterministic (no randomness is involved).

Every file under data/ is either split here or listed with a reason in
data/splits/exclusions.csv.

Usage (from the repository root):
    python -m code.splits.make_splits            # build manifests + gold test set, run checks
    python -m code.splits.make_splits --check    # only re-run the leakage checks on existing manifests
Outputs:
    data/splits/split_definition.json
    data/splits/{train,validation,test,shifted}.csv
    data/splits/exclusions.csv
    data/splits/gold_test/{gold_tasks.jsonl,gold_labels.csv,CHECKSUMS.sha256}
    supporting_files/splits/leakage_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

FAMILY_SPLIT = {
    "multi_hop_retrieval": "train",
    "compound_calculation": "train",
    "structured_aggregation": "train",
    "cross_tool_chaining": "validation",
    "file_sla_auditing": "test",
    "ambiguous_unanswerable": "test",
}
SPLITS = ("train", "validation", "test")

TASKS = Path("data/processed/tasks_revised.jsonl")
TRAJECTORIES = Path("data/experiments/multi_provider_run/trajectories.jsonl")
AUGMENTED = Path("data/augmented/tasks_augmented.jsonl")
LABELS = (Path("data/labels/labels_round1.csv"), Path("data/labels/labels_round2.csv"))
OUT = Path("data/splits")
REPORT = Path("supporting_files/splits/leakage_report.json")

EXCLUSIONS = [
    ("data/raw/tasks.jsonl", "Pilot v1 task suite (24 tasks, different families) superseded by data/processed/tasks_revised.jsonl; kept as Part A/B provenance."),
    ("data/raw/trajectories.jsonl", "Rule-based pilot agent runs on the v1 suite; not LLM agent behaviour; Part A/B provenance only."),
    ("data/adversarial/tasks.jsonl", "Separate adversarial pilot suite; not part of the 6-family benchmark."),
    ("data/adversarial/trajectories_gemini_3.8_flash.jsonl", "Adversarial pilot runs: 29/30 ended in provider errors with 0 steps; unusable as behaviour data."),
    ("data/adversarial/results.jsonl", "Grading output of the adversarial pilot."),
    ("data/corrupted/tasks_corrupted.jsonl", "Deliberately corrupted fixture for Part B (validator must fail on it)."),
    ("data/corrupted/trajectories_corrupted.jsonl", "Deliberately corrupted fixture for Part B."),
    ("data/processed/trajectories_processed.jsonl", "Subset (102 runs) of the 154 collected runs without outcome='error'. Splits use all 154: the 52 'error' runs are agent failures (step limit, malformed calls), not infrastructure errors."),
    ("data/experiments/multi_provider_run/summary.json", "Collection run summary (metadata, not samples)."),
    ("data/experiments/default/work_queue.sqlite3", "Operational job-queue state of the collector (not samples)."),
    ("data/labels/stratified_sample_45.json", "Label Studio import file for round 1 (its items are covered through their trajectories and labels)."),
    ("data/labels/stratified_sample_45.jsonl", "Same content as stratified_sample_45.json, line format."),
    ("data/labels/round2_sample.json", "Label Studio import file for round 2 (covered through trajectories and labels)."),
    ("data/labels/round2_manifest.csv", "Sampling manifest of round 2 (covered through trajectories and labels)."),
    ("data/metadata.csv", "Provenance table for all samples (Part A); not itself a sample."),
    ("data/processed/regrade_v2.csv", "Grader-v2 verdict per trajectory; derived annotation, follows its trajectory."),
    ("data/augmented/tasks_augmented_flat.jsonl", "Same tasks as tasks_augmented.jsonl without provenance wrapper (validator input)."),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[tuple[dict, str]]:
    return [(json.loads(line), line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def build() -> dict[str, list[dict]]:
    tasks = {t["task_id"]: (t, line) for t, line in _jsonl(TASKS)}
    labels: dict[str, dict] = {}
    for lp in LABELS:
        for r in csv.DictReader(lp.open(encoding="utf-8")):
            labels[r["run_id"]] = r
    rows: dict[str, list[dict]] = {s: [] for s in (*SPLITS, "shifted")}

    for tid, (t, line) in sorted(tasks.items()):
        split = FAMILY_SPLIT[t["template_family"]]
        rows[split].append({"item_type": "task", "item_id": tid, "task_id": tid, "parent_task_id": "", "template_family": t["template_family"],
                            "source_file": TASKS.as_posix(), "record_sha256": _sha(line), "gold_answer": t["gold_answer"],
                            "human_outcome": "", "human_category": "", "parent_split": ""})

    for rec, line in _jsonl(AUGMENTED):
        aug, t = rec["augmentation"], rec["task"]
        parent_split = FAMILY_SPLIT[tasks[aug["parent_task_id"]][0]["template_family"]]
        split = "shifted" if aug["type"] == "fault_injection" else parent_split
        rows[split].append({"item_type": f"augmented:{aug['type']}", "item_id": t["task_id"], "task_id": t["task_id"],
                            "parent_task_id": aug["parent_task_id"], "template_family": aug["template_family"],
                            "source_file": AUGMENTED.as_posix(), "record_sha256": _sha(line), "gold_answer": t["gold_answer"],
                            "human_outcome": "", "human_category": "", "parent_split": parent_split})

    for tr, line in _jsonl(TRAJECTORIES):
        fam = tasks[tr["task_id"]][0]["template_family"]
        lab = labels.get(tr["run_id"], {})
        rows[FAMILY_SPLIT[fam]].append({"item_type": "trajectory", "item_id": tr["run_id"], "task_id": tr["task_id"], "parent_task_id": "",
                                        "template_family": fam, "source_file": TRAJECTORIES.as_posix(), "record_sha256": _sha(line),
                                        "gold_answer": tasks[tr["task_id"]][0]["gold_answer"],
                                        "human_outcome": lab.get("outcome", ""), "human_category": lab.get("primary_failure_category", ""),
                                        "parent_split": ""})
    return rows


def write(rows: dict[str, list[dict]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with (OUT / f"{split}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(items[0].keys()))
            w.writeheader()
            w.writerows(items)
    with (OUT / "exclusions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "reason"])
        w.writerows(EXCLUSIONS)

    # Gold test set: the test-split tasks (base + verified variants) and the human labels of test-split trajectories.
    gold = OUT / "gold_test"
    gold.mkdir(parents=True, exist_ok=True)
    tasks = {t["task_id"]: line for t, line in _jsonl(TASKS)}
    aug = {r["task"]["task_id"]: json.dumps(r["task"], sort_keys=True) for r, _ in _jsonl(AUGMENTED)}
    test = rows["test"]
    with (gold / "gold_tasks.jsonl").open("w", encoding="utf-8") as fh:
        for r in test:
            if r["item_type"] == "task":
                fh.write(tasks[r["item_id"]] + "\n")
            elif r["item_type"].startswith("augmented:"):
                fh.write(aug[r["item_id"]] + "\n")
    labs = []
    for lp in LABELS:
        labs += [r for r in csv.DictReader(lp.open(encoding="utf-8"))]
    test_traj = {r["item_id"] for r in test if r["item_type"] == "trajectory"}
    gold_labels = sorted((r for r in labs if r["run_id"] in test_traj), key=lambda r: r["run_id"])
    with (gold / "gold_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gold_labels[0].keys()))
        w.writeheader()
        w.writerows(gold_labels)
    with (gold / "CHECKSUMS.sha256").open("w", encoding="utf-8") as fh:
        for name in ("gold_tasks.jsonl", "gold_labels.csv"):
            fh.write(f"{hashlib.sha256((gold / name).read_bytes()).hexdigest()}  {name}\n")


def check() -> dict:
    """Leakage and completeness checks on the written manifests. Returns the report."""
    rows = {s: list(csv.DictReader((OUT / f"{s}.csv").open(encoding="utf-8"))) for s in (*SPLITS, "shifted")}
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    fam = {s: {r["template_family"] for r in rows[s]} for s in SPLITS}
    pairs = [(a, b) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]]
    overlap = {f"{a}&{b}": sorted(fam[a] & fam[b]) for a, b in pairs}
    add("1_template_families_disjoint", not any(overlap.values()), json.dumps(overlap))

    ids = {s: {r["item_id"] for r in rows[s]} for s in (*SPLITS, "shifted")}
    dup = {f"{a}&{b}": len(ids[a] & ids[b]) for a, b in [*pairs, *[(s, "shifted") for s in SPLITS]]}
    add("2_no_item_in_two_splits", not any(dup.values()), json.dumps(dup))

    task_split = {r["task_id"]: s for s in SPLITS for r in rows[s] if r["item_type"] == "task"}
    traj_bad = [r["item_id"] for s in SPLITS for r in rows[s] if r["item_type"] == "trajectory" and task_split.get(r["task_id"]) != s]
    add("3_trajectories_and_repeated_runs_follow_their_task", not traj_bad, f"{len(traj_bad)} misplaced")

    var_bad = [r["item_id"] for s in SPLITS for r in rows[s] if r["item_type"].startswith("augmented") and task_split.get(r["parent_task_id"]) != s]
    add("4_augmented_variants_follow_their_parent", not var_bad, f"{len(var_bad)} misplaced")

    shifted_types = {r["item_type"] for r in rows["shifted"]}
    in_main = [r["item_id"] for s in SPLITS for r in rows[s] if r["item_type"] == "augmented:fault_injection"]
    add("5_shifted_split_is_exactly_the_fault_injected_tasks", shifted_types == {"augmented:fault_injection"} and not in_main,
        f"shifted types={sorted(shifted_types)}; fault tasks outside shifted={len(in_main)}")

    prompts: dict[str, dict[str, str]] = {}
    tasks_all = {t["task_id"]: t for t, _ in _jsonl(TASKS)} | {r["task"]["task_id"]: r["task"] for r, _ in _jsonl(AUGMENTED)}
    for s in SPLITS:
        for r in rows[s]:
            if r["item_type"] != "trajectory":
                prompts.setdefault(_norm(tasks_all[r["item_id"]]["prompt"]), {})[s] = r["item_id"]
    cross = [v for v in prompts.values() if len(v) > 1]
    add("6_no_identical_prompt_across_train_val_test", not cross, f"{len(cross)} prompts shared across splits")

    # every manifest row points to a record that exists with the same content hash
    sources: dict[str, dict[str, str]] = {}
    for path in (TASKS, AUGMENTED, TRAJECTORIES):
        sources[path.as_posix()] = {}
        for rec, line in _jsonl(path):
            key = rec.get("run_id") or rec.get("task_id") or rec["task"]["task_id"]
            sources[path.as_posix()][key] = _sha(line)
    missing = [r["item_id"] for s in rows for r in rows[s] if sources[r["source_file"]].get(r["item_id"]) != r["record_sha256"]]
    add("7_every_manifest_item_exists_with_matching_hash", not missing, f"{len(missing)} missing or changed")

    counted = {k: set() for k in sources}
    for s in rows:
        for r in rows[s]:
            counted[r["source_file"]].add(r["item_id"])
    unaccounted = {k: len(set(v) - counted[k]) for k, v in sources.items()}
    excluded = {p for p, _ in csv.reader((OUT / "exclusions.csv").open(encoding="utf-8")) if p != "path"}
    data_files = {p.as_posix() for p in Path("data").rglob("*")
                  if p.is_file() and "splits" not in p.parts and "active_learning" not in p.parts
                  and p.suffix != ".dvc" and p.name != ".gitignore"}  # DVC pointer / ignore files are not data
    covered = set(sources) | {lp.as_posix() for lp in LABELS} | excluded
    loose = sorted(data_files - covered)
    add("8_every_sample_and_data_file_accounted_for", not any(unaccounted.values()) and not loose,
        f"unaccounted samples={unaccounted}; data files neither split nor excluded={loose}")

    labelled = set()
    for lp in LABELS:
        labelled |= {r["run_id"] for r in csv.DictReader(lp.open(encoding="utf-8"))}
    traj_ids = {r["item_id"] for s in SPLITS for r in rows[s] if r["item_type"] == "trajectory"}
    add("9_every_human_label_has_a_split", labelled <= traj_ids, f"{len(labelled - traj_ids)} labels without a split")

    gold = OUT / "gold_test"
    expected = dict(line.split()[::-1] for line in (gold / "CHECKSUMS.sha256").read_text().splitlines())
    actual = {n: hashlib.sha256((gold / n).read_bytes()).hexdigest() for n in expected}
    add("10_gold_test_checksums_match", expected == actual, json.dumps(actual))

    counts = {s: {} for s in rows}
    for s in rows:
        for r in rows[s]:
            counts[s][r["item_type"]] = counts[s].get(r["item_type"], 0) + 1
        counts[s]["human_labelled_trajectories"] = sum(1 for r in rows[s] if r["human_outcome"])
        counts[s]["human_labelled_failures"] = sum(1 for r in rows[s] if r["human_outcome"] and r["human_outcome"] != "success")
        counts[s]["families"] = sorted({r["template_family"] for r in rows[s]})
    report = {"all_passed": all(c["passed"] for c in checks), "checks": checks, "counts": counts, "family_split": FAMILY_SPLIT,
              "gold_test_checksums": actual}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "split_definition.json").write_text(json.dumps({
        "unit": "template_family",
        "family_split": FAMILY_SPLIT,
        "randomness": "none - deterministic family assignment",
        "rationale": {
            "plan": "T1-T3 train, T4 validation, T5 test as declared in code/tasks/revised_generator.py",
            "T6": "Declared 'stratified' in the generator; a family-level hold-out forbids one family in several splits, so the whole family is in test (unseen-family + abstention evaluation).",
            "shifted": "All 96 fault-injected tasks (tool failure, information outage, slow response, format change); evaluation-only; parent_split column allows A2 to use the test-family subset for a clean drift comparison.",
            "trajectories": "All 154 collected runs, including the 52 logged as 'error' (agent failures)."},
        "gold_test_set": {"files": ["gold_test/gold_tasks.jsonl", "gold_test/gold_labels.csv"], "sha256": actual,
                          "rule": "Never used for tuning, prompt design or model/config selection in Assignment 2."},
        "counts": counts}, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build and check family-level splits.")
    p.add_argument("--check", action="store_true", help="only re-run checks on existing manifests")
    a = p.parse_args()
    if not a.check:
        write(build())
    r = check()
    for c in r["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail'][:150]}")
    print(json.dumps({s: {k: v for k, v in c.items()} for s, c in r["counts"].items()}, indent=1))
    if not r["all_passed"]:
        raise SystemExit("Split checks FAILED")


if __name__ == "__main__":
    main()
