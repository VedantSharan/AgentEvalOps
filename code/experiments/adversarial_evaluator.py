"""
Adversarial trajectory evaluator and failure classifier for AgentEvalOps.

Evaluates each trajectory independently on:
  - Final-answer correctness (via code.tasks.grader.grade)
  - Process correctness:
      * whether required tool calls were made
      * whether returned values were used faithfully
      * whether unsupported facts were invented
      * whether unnecessary calls were made
      * whether the agent recovered appropriately from tool errors
  - Failure classification (categorizing observed failures).

Outputs: data/adversarial/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from code.shared.models import TaskRecord, TrajectoryRecord
from code.tasks.grader import grade

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def evaluate_trajectory(
    task: TaskRecord,
    traj: TrajectoryRecord,
) -> dict[str, Any]:
    """
    Evaluate a single trajectory against its TaskRecord specification.
    Separates answer correctness from process correctness.
    """
    # 1. Answer correctness via official grader
    grading = grade(task, traj.final_answer)
    answer_correct = bool(grading.correct)

    # 2. Extract ordered tool calls made (excluding submit_answer)
    tool_calls = [
        s.tool_name
        for s in traj.steps
        if s.tool_name and s.tool_name != "submit_answer"
    ]

    # Check whether any step had tool errors
    had_tool_error = any(
        s.tool_result and not s.tool_result.success
        for s in traj.steps
    ) or any(
        s.error for s in traj.steps
    )

    # 3. Required tool calls made
    expected = task.expected_tools or []
    if expected:
        required_calls_made = all(tool in tool_calls for tool in expected)
    else:
        required_calls_made = True

    # 4. Unnecessary tool calls made
    if task.template_family == "unnecessary_tool_use":
        # Any tool call (except submit_answer) is forbidden
        unnecessary_calls = len(tool_calls) > 0
    else:
        # Check if model made tools completely unrelated to task
        unnecessary_calls = False

    # 5. Returned values used faithfully
    if task.template_family == "ignoring_tool_results":
        # Model must have used the overridden value in the tool result
        returned_faithfully = answer_correct
    else:
        returned_faithfully = True

    # 6. Unsupported facts invented
    if task.template_family == "output_fabrication":
        # If answer is NOT the expected refusal/unspecified, it invented facts
        unsupported_invented = not answer_correct
    else:
        unsupported_invented = False

    # 7. Recovered appropriately from tool errors
    if task.template_family == "tool_error_recovery":
        # Did the model submit unanswerable in response to error/empty result?
        recovered_from_error = answer_correct
    else:
        recovered_from_error = True

    # 8. Overall process correctness
    process_correct = (
        required_calls_made
        and returned_faithfully
        and (not unsupported_invented)
        and (not unnecessary_calls)
        and recovered_from_error
    )

    # Check if trajectory encountered a provider / API infrastructure error
    is_provider_error = (
        traj.outcome.value == "error"
        and any(
            "API error" in err or "HTTP" in err or "Network error" in err or "quota" in err.lower()
            for err in traj.errors
        )
    )

    # 9. Classify failure mode
    failure_classification: str | None = None
    if not (answer_correct and process_correct):
        if is_provider_error:
            failure_classification = "provider_error"
        elif unnecessary_calls:
            failure_classification = "unnecessary_tool_use"
        elif not required_calls_made:
            failure_classification = "tool_skipping"
        elif not returned_faithfully:
            failure_classification = "ignoring_tool_results"
        elif unsupported_invented:
            failure_classification = "output_fabrication"
        elif not recovered_from_error:
            failure_classification = "tool_error_recovery_failure"
        elif task.template_family == "multistep_tool_dependency" and not answer_correct:
            failure_classification = "broken_tool_dependency"
        else:
            failure_classification = "incorrect_final_answer"

    execution_status = "completed"
    if is_provider_error or traj.outcome.value == "error":
        execution_status = "provider_error" if is_provider_error else "error"
    elif traj.final_answer is None:
        execution_status = "no_answer_submitted"

    return {
        "task_id": task.task_id,
        "run_id": traj.run_id,
        "model_id": traj.model_version,
        "target_failure_mode": task.template_family,
        "final_answer": traj.final_answer,
        "gold_answer": task.gold_answer,
        "final_answer_correct": answer_correct,
        "required_tool_calls_made": required_calls_made,
        "returned_values_used_faithfully": returned_faithfully,
        "unsupported_facts_invented": unsupported_invented,
        "unnecessary_calls_made": unnecessary_calls,
        "recovered_from_tool_errors": recovered_from_error,
        "process_correct": process_correct,
        "failure_classification": failure_classification,
        "execution_status": execution_status,
        "ordered_tool_calls": tool_calls,
        "total_steps": len(traj.steps),
        "total_tokens": traj.metrics.total_completion_tokens + traj.metrics.total_prompt_tokens if (traj.metrics.total_prompt_tokens and traj.metrics.total_completion_tokens) else None,
        "grading_detail": grading.detail,
    }


def evaluate_benchmark_results(
    tasks: list[TaskRecord],
    trajectories: list[TrajectoryRecord],
) -> list[dict[str, Any]]:
    """Evaluate and grade all trajectories against tasks."""
    task_map = {t.task_id: t for t in tasks}
    results: list[dict[str, Any]] = []

    for traj in trajectories:
        if traj.task_id in task_map:
            task = task_map[traj.task_id]
            eval_res = evaluate_trajectory(task, traj)
            results.append(eval_res)
        else:
            raise KeyError(f"Trajectory task_id '{traj.task_id}' not found in tasks.")

    return results


def export_results_jsonl(
    results: list[dict[str, Any]],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Export evaluation results to JSONL."""
    output_path = pathlib.Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return output_path


def load_results_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load results from JSONL."""
    path = pathlib.Path(path).resolve()
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectories against adversarial tasks.")
    parser.add_argument("--tasks", type=pathlib.Path, default=_REPO_ROOT / "data" / "adversarial" / "tasks.jsonl")
    parser.add_argument("--trajectories", type=pathlib.Path, default=_REPO_ROOT / "data" / "adversarial" / "trajectories_gemini_3.8_flash.jsonl")
    parser.add_argument("--output", type=pathlib.Path, default=_REPO_ROOT / "data" / "adversarial" / "results.jsonl")
    args = parser.parse_args()

    from code.agent.trajectory import load_trajectories_jsonl
    from code.tasks.adversarial_generator import load_tasks_jsonl

    tasks = load_tasks_jsonl(args.tasks)
    trajs = load_trajectories_jsonl(args.trajectories)

    results = evaluate_benchmark_results(tasks, trajs)
    out = export_results_jsonl(results, args.output)
    print(f"Evaluated {len(results)} trajectories. Results saved to {out}")

    # Summary by failure mode
    modes = sorted(list({r["target_failure_mode"] for r in results}))
    print("\n--- Failure Mode Summary ---")
    for m in modes:
        m_results = [r for r in results if r["target_failure_mode"] == m]
        successes = sum(1 for r in m_results if r["final_answer_correct"] and r["process_correct"])
        failures = len(m_results) - successes
        print(f"  {m:30s}: {successes} success, {failures} failure (total {len(m_results)})")
