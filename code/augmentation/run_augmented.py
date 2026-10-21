"""
Run the agent on augmented tasks, with faults injected (Assignment 1, Part D).

Uses the SAME agent loop and configuration as the original collection
(code/experiments/agent_executor.execute_agent_task, max_steps=5, temperature=0,
prompt_version v1), so augmented trajectories are comparable with the 154
original ones.  For fault-injection tasks the tool registry is wrapped by
FaultInjector for the duration of that single run.

Each (task, model) pair is assigned round-robin over the configured models
after a seeded shuffle, so every task is run once by default.  The runner is
resumable: pairs already present in the output file are skipped.

Outputs (appended):
    data/augmented/trajectories_augmented.jsonl   TrajectoryRecord per line
    data/augmented/injection_log.jsonl            run_id -> faults actually injected
    data/augmented/run_log.jsonl                  one line per attempt (incl. provider errors)

Usage (from the repository root, API keys in .env):
    python -m code.augmentation.run_augmented --dry-run            # show the plan, no API calls
    python -m code.augmentation.run_augmented --max-requests 400
    python -m code.augmentation.run_augmented --types fault_injection --limit 20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

from code.augmentation.fault_injection import FaultInjector, FaultSpec
from code.experiments.agent_executor import RequestTracker, execute_agent_task
from code.experiments.providers import PROVIDER_SPECS, ErrorType, ProviderClient
from code.shared.models import ExecutionConfig, TaskRecord

SEED = 203
OUT_DIR = Path("data/augmented")


ORIGINAL_RUN = Path("data/experiments/multi_provider_run/trajectories.jsonl")


def original_models() -> list[str]:
    """Models that produced at least one completed trajectory in the original collection."""
    tags = set()
    for line in ORIGINAL_RUN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            if t["outcome"] in ("success", "failure"):
                tags.add(t["model_version"])
    return sorted(tags)


def configured_targets(model_filters: list[str] | None) -> list[tuple[str, str]]:
    exact = set(original_models()) if not model_filters else None
    targets = []
    for pname in PROVIDER_SPECS:
        client = ProviderClient(pname)
        if not client.is_configured():
            continue
        for mod in client.config.default_models:
            if not mod.is_free_tier:
                continue
            tag = f"{pname}:{mod.model_id}"
            if exact is not None and tag not in exact:
                continue
            if model_filters and not any(f in tag for f in model_filters):
                continue
            targets.append((pname, mod.model_id))
    return targets


def _done_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            done.add((t["task_id"], t["model_version"]))
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Run the agent on augmented (incl. fault-injected) tasks.")
    p.add_argument("--augmented", type=Path, default=OUT_DIR / "tasks_augmented.jsonl")
    p.add_argument("--types", nargs="*", default=["entity_variant", "paraphrase", "fault_injection"])
    p.add_argument("--models", nargs="*", default=None, help="substring filters, e.g. groq gemini-3.5-flash-lite (default: the models that completed runs in the original collection)")
    p.add_argument("--limit", type=int, default=None, help="max number of tasks (after seeded shuffle)")
    p.add_argument("--max-requests", type=int, default=400, help="hard cap on LLM API requests")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    records = [json.loads(line) for line in a.augmented.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["augmentation"]["type"] in a.types]
    rng = random.Random(a.seed)
    rng.shuffle(records)
    if a.limit:
        records = records[: a.limit]

    targets = configured_targets(a.models)
    if not targets:
        raise SystemExit("No configured free-tier models found. Put API keys in .env (see code/experiments/providers.py).")
    plan = [(r, targets[i % len(targets)]) for i, r in enumerate(records)]

    traj_path, inj_path, log_path = OUT_DIR / "trajectories_augmented.jsonl", OUT_DIR / "injection_log.jsonl", OUT_DIR / "run_log.jsonl"
    done = _done_pairs(traj_path)
    plan = [(r, t) for r, t in plan if (r["task"]["task_id"], f"{t[0]}:{t[1]}") not in done]

    print(f"{len(plan)} runs planned over {len(targets)} model(s); {len(done)} already done.")
    for r, (prov, mod) in plan[:10]:
        print(f"  {r['task']['task_id']:45s} {prov}:{mod}")
    if a.dry_run:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = RequestTracker(max_requests=a.max_requests)
    clients = {prov: ProviderClient(prov) for prov, _ in targets}
    config = ExecutionConfig(model_id="dynamic", prompt_version="v1", tool_selection_strategy="openai_function_calling",
                             temperature=0.0, max_steps=5, retry_budget=1)
    completed = 0
    for r, (prov, mod) in plan:
        task = TaskRecord.model_validate(r["task"])
        specs = [FaultSpec(**s) for s in (r["augmentation"]["fault"] or [])]
        config.model_id = mod
        started = datetime.now(timezone.utc).isoformat()
        for attempt in range(3):
            with FaultInjector(specs) as inj:
                res = execute_agent_task(task=task, provider_client=clients[prov], model_id=mod, config=config, request_tracker=tracker)
            if res.error_type == ErrorType.RATE_LIMIT and attempt < 2:
                time.sleep(min(res.retry_after or 30.0, 90.0))
                continue
            break
        entry = {"aug_task_id": task.task_id, "model": f"{prov}:{mod}", "started_at": started,
                 "error_type": res.error_type.value if res.error_type else None, "provider_error": res.provider_error,
                 "run_id": res.trajectory.run_id if res.trajectory else None}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if res.error_type == ErrorType.BUDGET_EXHAUSTED:
            print("Request cap reached; stopping (rerun to resume).")
            break
        if res.trajectory is None:
            print(f"  [skip] {task.task_id} on {prov}:{mod}: {res.error_type} {res.provider_error}")
            continue
        with traj_path.open("a", encoding="utf-8") as fh:
            fh.write(res.trajectory.model_dump_json() + "\n")
        with inj_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": res.trajectory.run_id, "aug_task_id": task.task_id,
                                 "augmentation_type": r["augmentation"]["type"],
                                 "fault_specs": r["augmentation"]["fault"], "injected": inj.log}, default=str) + "\n")
        completed += 1
        print(f"  [{completed}] {task.task_id} on {prov}:{mod} -> {res.outcome.value if res.outcome else '?'} "
              f"(injections: {len(inj.log)})")
    print(f"Done: {completed} trajectories appended to {traj_path.as_posix()}")


if __name__ == "__main__":
    main()
