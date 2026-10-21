"""
Stratified Sampling for Human Annotation (Assignment 1, Part C).

Samples 45 trajectories from the multi-provider execution pool, balanced across:
- Providers (Gemini, Groq, NVIDIA, Cohere, OpenRouter)
- 6 Task Families (multi_hop_retrieval, compound_calculation, structured_aggregation,
                    cross_tool_chaining, file_sla_auditing, ambiguous_unanswerable)
- Execution Outcomes (success, failure, error)

Formats each sample into Label Studio import format with full trajectory context.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any


def format_steps_for_display(steps: list[dict[str, Any]]) -> str:
    """Format trajectory steps into a clean markdown transcript for the annotator."""
    lines = []
    for step in steps:
        idx = step.get("step_index", 0)
        action = step.get("action_type", "unknown")
        tool = step.get("tool_name")
        args = step.get("arguments")
        res = step.get("tool_result")
        llm = step.get("llm_output")
        err = step.get("error")

        lines.append(f"### Step {idx}: {action.upper()}" + (f" (`{tool}`)" if tool else ""))
        if args:
            lines.append(f"- **Arguments**: `{json.dumps(args)}`")
        if res:
            res_status = "SUCCESS" if res.get("success") else "FAILURE"
            lines.append(f"- **Tool Result** [{res_status}]:")
            if res.get("output") is not None:
                lines.append(f"  ```\n  {res.get('output')}\n  ```")
            if res.get("error"):
                lines.append(f"  **Error**: {res.get('error')}")
        if llm:
            lines.append(f"- **LLM Output**:\n  > {llm}")
        if err:
            lines.append(f"- **Step Error**: {err}")
        lines.append("")
    return "\n".join(lines)


def sample_stratified_dataset(
    trajectories_path: Path,
    tasks_path: Path,
    target_count: int = 45,
    seed: int = 42,
) -> list[dict[str, Any]]:
    random.seed(seed)

    tasks_raw = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]
    task_map = {t["task_id"]: t for t in tasks_raw}

    trajs_raw = [
        json.loads(line)
        for line in trajectories_path.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]

    # Filter to strictly valid outcomes; exclude execution errors per dataset policy
    valid_trajs = [tr for tr in trajs_raw if tr.get("outcome") in ("success", "failure")]

    # Group trajectories by (outcome, template_family, provider)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for tr in valid_trajs:
        task = task_map.get(tr["task_id"])
        if not task:
            continue
        outcome = tr["outcome"]
        family = task.get("template_family", "unknown")
        provider = tr["model_version"].split(":")[0]
        buckets[(outcome, family, provider)].append(tr)

    # We want target_count = 45, balanced between success and failure (e.g. 23 success, 22 failure)
    target_outcomes = ["success", "failure"]
    n_success = (target_count + 1) // 2  # 23
    n_failure = target_count // 2        # 22
    target_counts = {"success": n_success, "failure": n_failure}

    selected: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for target_outcome in target_outcomes:
        outcome_target = target_counts[target_outcome]
        outcome_pool = [tr for tr in valid_trajs if tr["outcome"] == target_outcome]
        # Shuffle pool deterministically
        random.shuffle(outcome_pool)

        # Prioritize diverse representation across (family, provider)
        by_family_provider: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for tr in outcome_pool:
            t = task_map.get(tr["task_id"], {})
            fam = t.get("template_family", "unknown")
            prov = tr["model_version"].split(":")[0]
            by_family_provider[(fam, prov)].append(tr)

        outcome_selected: list[dict[str, Any]] = []
        # Round robin across (family, provider) pairs
        keys = list(by_family_provider.keys())
        random.shuffle(keys)
        
        while len(outcome_selected) < outcome_target and keys:
            progress = False
            for k in list(keys):
                if by_family_provider[k]:
                    candidate = by_family_provider[k].pop(0)
                    if candidate["run_id"] not in seen_run_ids:
                        outcome_selected.append(candidate)
                        seen_run_ids.add(candidate["run_id"])
                        progress = True
                        if len(outcome_selected) == outcome_target:
                            break
                else:
                    keys.remove(k)
            if not progress:
                break

        # If still short, backfill from remaining outcome pool
        if len(outcome_selected) < outcome_target:
            for tr in outcome_pool:
                if tr["run_id"] not in seen_run_ids:
                    outcome_selected.append(tr)
                    seen_run_ids.add(tr["run_id"])
                    if len(outcome_selected) == outcome_target:
                        break

        selected.extend(outcome_selected)

    # Format for Label Studio / human review
    formatted_samples = []
    for idx, tr in enumerate(selected, 1):
        task = task_map[tr["task_id"]]
        sample = {
            "id": tr["run_id"],
            "sample_index": idx,
            "data": {
                "run_id": tr["run_id"],
                "task_id": tr["task_id"],
                "template_family": task.get("template_family"),
                "task_type": task.get("task_type"),
                "model_version": tr["model_version"],
                "prompt": task.get("prompt"),
                "gold_answer": task.get("gold_answer"),
                "evaluation_rule": task.get("evaluation_rule"),
                "final_answer": tr.get("final_answer"),
                "outcome": tr.get("outcome"),
                "total_steps": tr["metrics"]["total_steps"],
                "total_latency_ms": round(tr["metrics"]["total_latency_ms"] or 0, 2),
                "steps_transcript": format_steps_for_display(tr.get("steps", [])),
                "errors": tr.get("errors", []),
            }
        }
        formatted_samples.append(sample)

    return formatted_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified sampling for Label Studio annotation.")
    parser.add_argument("--trajectories", type=Path, default=Path("data/processed/trajectories_processed.jsonl"))
    parser.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    parser.add_argument("--output-json", type=Path, default=Path("data/labels/stratified_sample_45.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/labels/stratified_sample_45.jsonl"))
    parser.add_argument("--count", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    samples = sample_stratified_dataset(
        trajectories_path=args.trajectories,
        tasks_path=args.tasks,
        target_count=args.count,
        seed=args.seed,
    )

    # Write formatted JSON
    args.output_json.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    # Write JSONL for quick line-based parsing
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    print(f"Successfully sampled {len(samples)} stratified trajectories:")
    print(f"  JSON:  {args.output_json}")
    print(f"  JSONL: {args.output_jsonl}")

    # Summary breakdown
    from collections import Counter
    outcomes = Counter(s["data"]["outcome"] for s in samples)
    providers = Counter(s["data"]["model_version"].split(":")[0] for s in samples)
    families = Counter(s["data"]["template_family"] for s in samples)

    print("\nSample Distribution Breakdown:")
    print("  Outcomes:", dict(outcomes))
    print("  Providers:", dict(providers))
    print("  Task Families:", dict(families))


if __name__ == "__main__":
    main()
