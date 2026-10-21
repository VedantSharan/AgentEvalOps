"""
Save the Part-B validation suite's results as evidence files.

Runs code/validation/benchmark_validator.BenchmarkValidator (unchanged) on:
  raw        revised tasks + all 154 collected trajectories (incl. outcome='error')
  processed  revised tasks + the 102 processed trajectories ("fixed" data)
  corrupted  the deliberately corrupted fixtures (the suite is expected to FAIL)
  augmented  the prompt-changing augmented tasks (entity variants + paraphrases)

The "before" state of the leakage fix (original T5_audit_04 prompt) no longer
exists as a file; it is documented in docs/VALIDATION_REPORT.md.

Exit code is 0 when every dataset gives its EXPECTED result (corrupted must fail,
the others must pass), so this can run inside the DVC pipeline.

Usage (from the repository root):
    python -m code.validation.run_validation_evidence
Outputs: supporting_files/validation/{raw,processed,corrupted,augmented}.json and summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from code.validation.benchmark_validator import BenchmarkValidator

OUT = Path("supporting_files/validation")
RUNS = {
    "raw": ("data/processed/tasks_revised.jsonl", "data/experiments/multi_provider_run/trajectories.jsonl", True),
    "processed": ("data/processed/tasks_revised.jsonl", "data/processed/trajectories_processed.jsonl", True),
    "corrupted": ("data/corrupted/tasks_corrupted.jsonl", "data/corrupted/trajectories_corrupted.jsonl", False),
    "augmented": ("data/augmented/tasks_augmented_flat.jsonl", None, True),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "none.jsonl"
        empty.write_text("", encoding="utf-8")
        for name, (tasks, trajs, expect_pass) in RUNS.items():
            res = BenchmarkValidator(Path(tasks), Path(trajs) if trajs else empty).run_all_checks()
            (OUT / f"{name}.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
            summary[name] = {"tasks": tasks, "trajectories": trajs, "passed": res.passed, "expected_to_pass": expect_pass,
                             "as_expected": res.passed == expect_pass, "checks_passed": res.checks_passed,
                             "checks_failed": res.checks_failed,
                             "failed_checks": sorted(k for k, v in res.summary_by_check.items() if not v)}
            print(f"{name:10s} passed={res.passed} ({res.checks_passed}/12)  expected={'pass' if expect_pass else 'fail'}  "
                  f"failed checks={summary[name]['failed_checks']}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not all(v["as_expected"] for v in summary.values()):
        raise SystemExit("A dataset did not give its expected validation result.")


if __name__ == "__main__":
    main()
