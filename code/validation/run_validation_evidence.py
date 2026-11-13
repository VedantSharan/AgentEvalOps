"""
Save the Part-B validation suite's results as evidence files.

Runs code/validation/benchmark_validator.BenchmarkValidator (unchanged checks) on:
  raw            revised tasks + all 154 collected trajectories (incl. outcome='error')
  processed      revised tasks + the 102 processed trajectories ("fixed" data)
  corrupted      the deliberately corrupted fixtures (the suite is expected to FAIL)
  augmented      the prompt-changing augmented tasks (entity variants + paraphrases)
  augmented_all  all 149 augmented tasks, including the 96 fault-injected ones (added in A2)

Fault-injected tasks keep their parent's prompt on purpose (the fault is in the
tool layer), so for ``augmented_all`` the duplicate check compares prompt plus
fault specification: two tasks are duplicates only if both match.

The "before" state of the leakage fix (original T5_audit_04 prompt) no longer
exists as a file; it is documented in docs/VALIDATION_REPORT.md.

Exit code is 0 when every dataset gives its EXPECTED result (corrupted must fail,
the others must pass), so this can run inside the DVC pipeline and in CI.

Usage (from the repository root):
    python -m code.validation.run_validation_evidence
    python -m code.validation.run_validation_evidence --sample 20 --out /tmp/validation   # CI
Outputs: <out>/{raw,processed,corrupted,augmented,augmented_all}.json and summary.json
(default <out> = supporting_files/validation). Files are written with LF line endings
on every platform, so DVC hashes do not depend on the operating system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile

from code.validation.benchmark_validator import BenchmarkValidator

OUT = Path("supporting_files/validation")
AUGMENTED_RECORDS = "data/augmented/tasks_augmented.jsonl"
RUNS = {
    "raw": ("data/processed/tasks_revised.jsonl", "data/experiments/multi_provider_run/trajectories.jsonl", True),
    "processed": ("data/processed/tasks_revised.jsonl", "data/processed/trajectories_processed.jsonl", True),
    "corrupted": ("data/corrupted/tasks_corrupted.jsonl", "data/corrupted/trajectories_corrupted.jsonl", False),
    "augmented": ("data/augmented/tasks_augmented_flat.jsonl", None, True),
    "augmented_all": (AUGMENTED_RECORDS, None, True),
}
SEED = 203


def augmented_tasks_and_environments(records_path: Path, tasks_out: Path) -> dict[str, str]:
    """Write the TaskRecord of every augmented record to ``tasks_out``.

    Returns a map task_id -> environment key (the canonical fault specification)
    for the fault-injected tasks.
    """
    environments: dict[str, str] = {}
    with records_path.open("r", encoding="utf-8") as src, tasks_out.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            task = record["task"]
            dst.write(json.dumps(task, sort_keys=True) + "\n")
            aug = record.get("augmentation", {})
            if aug.get("type") == "fault_injection":
                environments[task["task_id"]] = json.dumps(aug.get("fault", []), sort_keys=True)
    return environments


def sample_lines(src: Path, dst: Path, n: int, seed: int) -> None:
    """Copy a seeded random sample of ``n`` non-empty JSONL lines from ``src`` to ``dst``."""
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    picked = sorted(random.Random(seed).sample(range(len(lines)), min(n, len(lines))))
    dst.write_text("".join(lines[i] + "\n" for i in picked), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the 12-check suite on every A1 dataset.")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="validate only a seeded sample of N trajectories per dataset (CI smoke mode)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        empty = tmp_dir / "none.jsonl"
        empty.write_text("", encoding="utf-8")
        for name, (tasks, trajs, expect_pass) in RUNS.items():
            environments: dict[str, str] = {}
            tasks_path = Path(tasks)
            if tasks == AUGMENTED_RECORDS:
                tasks_path = tmp_dir / "augmented_all_tasks.jsonl"
                environments = augmented_tasks_and_environments(Path(AUGMENTED_RECORDS), tasks_path)
            trajs_path = Path(trajs) if trajs else empty
            if args.sample is not None and trajs:
                trajs_path = tmp_dir / f"{name}_sample.jsonl"
                sample_lines(Path(trajs), trajs_path, args.sample, args.seed)
            res = BenchmarkValidator(tasks_path, trajs_path, environments=environments).run_all_checks()
            (args.out / f"{name}.json").write_text(res.model_dump_json(indent=2), encoding="utf-8", newline="\n")
            summary[name] = {"tasks": tasks, "trajectories": trajs, "passed": res.passed, "expected_to_pass": expect_pass,
                             "as_expected": res.passed == expect_pass, "checks_passed": res.checks_passed,
                             "checks_failed": res.checks_failed,
                             "failed_checks": sorted(k for k, v in res.summary_by_check.items() if not v)}
            if args.sample is not None:
                summary[name]["trajectory_sample"] = {"n": args.sample, "seed": args.seed} if trajs else None
            print(f"{name:14s} passed={res.passed} ({res.checks_passed}/12)  expected={'pass' if expect_pass else 'fail'}  "
                  f"failed checks={summary[name]['failed_checks']}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    if not all(v["as_expected"] for v in summary.values()):
        raise SystemExit("A dataset did not give its expected validation result.")


if __name__ == "__main__":
    main()
