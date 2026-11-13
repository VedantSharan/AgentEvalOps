"""Offline smoke request against a running service (used by CI with --network none)."""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

TASK = "What is the annual salary of employee ID 'E999' according to the corporate employee records?"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args(argv)
    deadline = time.time() + 120
    while True:
        try:
            health = requests.get(f"{args.url}/health", timeout=5).json()
            break
        except requests.RequestException:
            if time.time() > deadline:
                print("service did not come up", file=sys.stderr)
                return 1
            time.sleep(1)
    version = requests.get(f"{args.url}/version", timeout=30).json()
    res = requests.post(f"{args.url}/predict", json={"task": TASK}, timeout=args.timeout).json()
    print(json.dumps({"health": health, "version": version,
                      "predict": {k: res.get(k) for k in ("answer", "confidence", "latency_ms", "trace_id", "correct")}},
                     indent=2))
    ok = health.get("status") == "ok" and res.get("answer") is not None and res.get("trace_id")
    print("SMOKE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
