"""CPU load test of the running service (Part D): p50/p95 latency, throughput, memory.

    docker run -d --name agentevalops -p 7860:7860 agentevalops:a2
    python -m code.service.loadtest --url http://127.0.0.1:7860 --container agentevalops --requests 20

Sends ``--requests`` /predict calls with bundled validation and test tasks
(fixed order, seed 203) at concurrency 1 (the model server processes one
request at a time, OLLAMA_NUM_PARALLEL=1), samples the container's memory with
``docker stats`` during the test, and writes supporting_files/a2/loadtest.json.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from code.agent.config import REPO_ROOT
from code.evaluation.data import load_split
from code.reporting.hardware import hardware_record

OUT = REPO_ROOT / "supporting_files" / "a2" / "loadtest.json"


def _mem_mib(container: str) -> float | None:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    used = out.split("/")[0].strip()
    num = float("".join(c for c in used if c.isdigit() or c == "."))
    return num * 1024 if used.endswith("GiB") else num if used.endswith("MiB") else num / 1024


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--container", default=None, help="docker container name for memory sampling")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--seed", type=int, default=203)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    tasks = [i.task.prompt for s in ("validation", "test") for i in sorted(load_split(s), key=lambda i: i.task_id)
             if i.item_type == "base"]
    random.Random(args.seed).shuffle(tasks)
    tasks = (tasks * (args.requests // len(tasks) + 1))[: args.requests]

    mem: list[float] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            m = _mem_mib(args.container) if args.container else None
            if m is not None:
                mem.append(m)
            stop.wait(2.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    version = requests.get(f"{args.url}/version", timeout=30).json()
    latencies, server_latencies, errors = [], [], 0
    t0 = time.perf_counter()
    for task in tasks:
        s = time.perf_counter()
        try:
            r = requests.post(f"{args.url}/predict", json={"task": task}, timeout=900)
            r.raise_for_status()
            server_latencies.append(r.json()["latency_ms"] / 1000)
        except requests.RequestException:
            errors += 1
        latencies.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=5)

    lat = np.array(latencies)
    result: dict[str, Any] = {
        "requests": len(tasks), "errors": errors, "concurrency": 1, "total_s": round(total, 1),
        "p50_latency_s": round(float(np.percentile(lat, 50)), 2), "p95_latency_s": round(float(np.percentile(lat, 95)), 2),
        "mean_latency_s": round(float(lat.mean()), 2), "throughput_req_per_min": round(60 * len(tasks) / total, 2),
        "memory_mib_peak": round(max(mem), 1) if mem else None,
        "memory_mib_mean": round(float(np.mean(mem)), 1) if mem else None,
        "version": version, "client_hardware": hardware_record(),
        "note": "Latency is end-to-end per /predict call (several LLM turns per task) on CPU only.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("version", "client_hardware")}, indent=2))


if __name__ == "__main__":
    main()
