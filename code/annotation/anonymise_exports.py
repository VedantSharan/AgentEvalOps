"""Remove personal e-mail addresses from the raw Label Studio exports.

Label Studio stores the annotator's account e-mail in every annotation's
``completed_by`` block. The A1 exports in ``supporting_files/labels/`` kept one
personal address (45 annotations), which the A1 feedback flagged and the A2
brief forbids ("Anonymise personal information").

Each address is replaced by a fixed pseudonym for the annotator who made the
export (``A1@anonymised.invalid`` for an ``annotator1_*`` file, and so on). The
edit is a plain byte-level string replacement, so every other byte of the
export, including the annotator's numeric ``id`` that the agreement code uses,
stays the same. The pipeline outputs built from the exports do not change.

Usage:
    python -m code.annotation.anonymise_exports            # rewrite in place
    python -m code.annotation.anonymise_exports --check    # exit 1 if any address is left
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LABELS_DIR = Path("supporting_files/labels")
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ANON_DOMAIN = b"anonymised.invalid"


def pseudonym_for(path: Path) -> bytes:
    """Return the pseudonymous address for the annotator who made ``path``."""
    match = re.match(r"annotator(\d+)", path.name)
    annotator = f"A{match.group(1)}" if match else "ANNOTATOR"
    return annotator.encode() + b"@" + ANON_DOMAIN


def find_addresses(data: bytes) -> list[bytes]:
    """Return every e-mail address in ``data`` that is not already anonymised."""
    return [m for m in EMAIL_RE.findall(data) if not m.endswith(ANON_DOMAIN)]


def anonymise_bytes(data: bytes, pseudonym: bytes) -> tuple[bytes, int]:
    """Replace every non-anonymised address in ``data`` with ``pseudonym``."""
    count = 0

    def _sub(match: re.Match[bytes]) -> bytes:
        nonlocal count
        if match.group(0).endswith(ANON_DOMAIN):
            return match.group(0)
        count += 1
        return pseudonym

    return EMAIL_RE.sub(_sub, data), count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--check", action="store_true", help="only report; exit 1 if an address is found")
    args = parser.parse_args(argv)

    leftover = 0
    for path in sorted(args.dir.rglob("*.json")):
        data = path.read_bytes()
        if args.check:
            found = find_addresses(data)
            if found:
                leftover += len(found)
                print(f"{path}: {len(found)} address(es)")
            continue
        new, count = anonymise_bytes(data, pseudonym_for(path))
        if count:
            path.write_bytes(new)
            print(f"{path}: replaced {count} address(es) with {pseudonym_for(path).decode()}")

    if args.check:
        print("no personal e-mail addresses found" if not leftover else f"{leftover} address(es) left")
        return 1 if leftover else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
