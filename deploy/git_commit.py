"""Resolve the commit of a git checkout without git (used while building the image)."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve(git_dir: Path) -> str:
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1].strip()
    loose = git_dir / ref
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    return "unknown"


if __name__ == "__main__":
    print(resolve(Path(sys.argv[1] if len(sys.argv) > 1 else ".git")))
