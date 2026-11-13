"""Copy the data the service needs into deploy/data/ (committed to git).

The DVC-tracked data is not in git, but the image must build from a clean
checkout of a2-final with no manual steps (A2 Part D). The service needs only
the task suite, the augmented tasks and the split manifests including the gold
test set (~300 KB). This script copies them byte-for-byte; tests/test_deploy.py
fails if the copy and the DVC workspace ever differ, and the image verifies the
gold checksums before every validation.

    python deploy/bundle.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = ["data/processed/tasks_revised.jsonl", "data/augmented/tasks_augmented.jsonl",
         "data/augmented/tasks_augmented_flat.jsonl"]
DIRS = ["data/splits"]
DEST = REPO / "deploy"


def bundle_paths() -> list[str]:
    paths = list(FILES)
    for d in DIRS:
        paths += sorted(p.relative_to(REPO).as_posix() for p in (REPO / d).rglob("*") if p.is_file())
    return paths


def main() -> None:
    for rel in bundle_paths():
        dst = DEST / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
        print(rel)


if __name__ == "__main__":
    main()
