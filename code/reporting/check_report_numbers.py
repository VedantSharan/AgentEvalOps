"""
Traceability audit: re-read every number quoted in the report from the file it cites.

report/evidence_index.csv has one row per reported number:
    id, section, claim, value, file, locator
Locator forms:
    json:<dotted.path>                    value at that path in a JSON file (list index allowed: a.0.b;
                                          write a|b|c instead when a key contains a dot)
    csv:<col>=<val>[;<col>=<val>...]:<column>   value of <column> in the single row matching all filters
    csvcount:<col>=<val>[;...]            number of rows matching the filters (empty filter = all rows)
    jsonlcount:                           number of non-empty lines in a JSONL file
    sha256:                               SHA-256 of the file
Numbers are compared after rounding the file value to the number of decimals written
in the report (so 0.8485 in the file matches "0.85" in the report).

Usage (from the repository root):
    python -m code.reporting.check_report_numbers
Writes supporting_files/report_traceability.csv and exits 1 if any number does not match.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

INDEX = Path("report/evidence_index.csv")
OUT = Path("supporting_files/report_traceability.csv")


def _filters(spec: str) -> list[tuple[str, str]]:
    return [tuple(f.split("=", 1)) for f in spec.split(";") if f]


def resolve(file: str, locator: str) -> str:
    p = Path(file)
    kind, _, rest = locator.partition(":")
    if kind == "json":
        v = json.loads(p.read_text(encoding="utf-8"))
        for key in rest.split("|" if "|" in rest else "."):  # use | when a key itself contains dots
            v = v[int(key)] if isinstance(v, list) else v[key]
        return str(v)
    if kind in ("csv", "csvcount"):
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        if kind == "csvcount":
            return str(sum(all(r.get(c) == v for c, v in _filters(rest)) for r in rows))
        flt, _, col = rest.rpartition(":")
        match = [r for r in rows if all(r.get(c) == v for c, v in _filters(flt))]
        if len(match) != 1:
            raise LookupError(f"{len(match)} rows match {flt} in {file}")
        return match[0][col]
    if kind == "jsonlcount":
        return str(sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()))
    if kind == "sha256":
        return hashlib.sha256(p.read_bytes()).hexdigest()
    raise ValueError(f"unknown locator {locator!r}")


def matches(reported: str, actual: str) -> bool:
    if reported == actual:
        return True
    try:
        r, a = float(reported), float(actual)
    except ValueError:
        return reported.lower() == actual.lower()
    decimals = len(reported.split(".")[1]) if "." in reported else 0
    return round(a, decimals) == r


def main() -> None:
    rows = list(csv.DictReader(INDEX.open(encoding="utf-8")))
    out, bad = [], 0
    for r in rows:
        try:
            actual = resolve(r["file"], r["locator"])
            ok = matches(r["value"], actual)
        except Exception as exc:  # noqa: BLE001 - report every failure, do not stop
            actual, ok = f"ERROR: {exc}", False
        bad += not ok
        out.append({**r, "value_in_file": actual, "match": ok})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{len(rows) - bad}/{len(rows)} reported numbers match their files")
    for r in out:
        if not r["match"]:
            print(f"  MISMATCH {r['id']}: report={r['value']} file={r['value_in_file']} ({r['file']} {r['locator']})")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
