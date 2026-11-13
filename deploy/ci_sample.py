"""Write the small A1 data sample that CI validates (A2 Part D: "sample Assignment 1 validation").

CI cannot pull the DVC data (the remote is a local folder), so a seeded sample of
the A1 data is committed under deploy/ci_sample/, mirroring the data/ paths:
20 collected trajectories, 20 processed trajectories and the corrupted fixtures
(which the suite must reject). CI copies deploy/data/ and deploy/ci_sample/ into
data/ and runs ``python -m code.validation.run_validation_evidence --sample 20``.

    python deploy/ci_sample.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "deploy" / "ci_sample"
SEED = 203
N = 20
SAMPLED = ["data/experiments/multi_provider_run/trajectories.jsonl", "data/processed/trajectories_processed.jsonl"]
COPIED = ["data/corrupted/tasks_corrupted.jsonl", "data/corrupted/trajectories_corrupted.jsonl"]


def main() -> None:
    for rel in SAMPLED:
        lines = [ln for ln in (REPO / rel).read_text(encoding="utf-8").splitlines() if ln.strip()]
        picked = sorted(random.Random(SEED).sample(range(len(lines)), N))
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(lines[i] + "\n" for i in picked), encoding="utf-8", newline="\n")
        print(rel, N)
    for rel in COPIED:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
        print(rel)


if __name__ == "__main__":
    main()
