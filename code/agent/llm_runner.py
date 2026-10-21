"""
Live LLM Agent Runner for AgentEvalOps using OpenAI-compatible APIs (Gemini / OpenRouter).

Executes benchmark tasks using genuine model invocations, records structured tool calls,
observes tool failure modes, and saves validated TrajectoryRecords.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import requests
from dotenv import load_dotenv

from code.agent.tools import TOOL_REGISTRY, submit_answer
from code.agent.trajectory import TrajectoryBuilder, export_trajectories_jsonl
from code.shared.models import ExecutionConfig, Outcome, TaskRecord, ToolResult, TrajectoryRecord
from code.tasks.grader import grade

load_dotenv()

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_MODEL = "gemini-3.8-flash"

# Tool schemas exposed to the model
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Look up facts from the local knowledge base matching a query keyword or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or question fragment."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a safe arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate (e.g. '17 * 6' or 'round(100 / 7, 2)')."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": "Query or aggregate the employee records table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["filter", "aggregate"],
                        "description": "Operation type: 'filter' to filter rows, 'aggregate' to compute summary metric."
                    },
                    "field": {
                        "type": "string",
                        "description": "Column to filter on (emp_id, name, department, salary)."
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to match for the filter operation."
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["count", "sum", "avg", "max", "min", "list"],
                        "description": "Aggregation function to apply."
                    },
                    "agg_field": {
                        "type": "string",
                        "description": "Column to aggregate over (e.g. 'salary' or 'name')."
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a controlled virtual benchmark file. Only base filenames in the virtual store are permitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Base filename to read (e.g. 'auth_token_94.txt')."
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the agent's final answer to conclude the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer string to submit."
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


class RequestBudget:
    """Tracks and enforces total API request limits across free tier models."""

    def __init__(self, max_requests: int = 100):
        self.max_requests = max_requests
        self.total_requests = 0

    def consume(self) -> None:
        if self.total_requests >= self.max_requests:
            raise RuntimeError(
                f"Request budget cap reached ({self.total_requests}/{self.max_requests}). "
                "Stopping execution to stay within free quota."
            )
        self.total_requests += 1


SYSTEM_INSTRUCTION = (
    "You are an AI assistant evaluating tools for the AgentEvalOps benchmark. "
    "Carefully follow all instructions in the prompt. "
    "If the prompt explicitly specifies a POLICY prohibiting tool use or stating not to use tools, "
    "you must NOT call any tools and answer directly. "
    "Otherwise, use the appropriate tool to find the required information, "
    "faithfully use the returned tool results, and do not invent facts not present in the results. "
    "Submit your final answer using the submit_answer tool."
)


def run_llm_task(
    task: TaskRecord,
    config: ExecutionConfig,
    api_key: str | None = None,
    budget: RequestBudget | None = None,
    timeout: float = 30.0,
    rate_limit_delay: float = 2.0,
) -> TrajectoryRecord:
    """Execute a single task using the live LLM agent loop."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not resolved_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    config_hash = config.compute_hash()
    builder = TrajectoryBuilder(
        task=task,
        configuration_hash=config_hash,
        model_version=config.model_id,
        prompt_version=config.prompt_version,
    )

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": task.prompt},
    ]

    final_answer: str | None = None
    provider_error: str | None = None

    for step_num in range(config.max_steps):
        if budget:
            try:
                budget.consume()
            except RuntimeError as exc:
                builder.add_error(str(exc))
                provider_error = str(exc)
                break

        # Rate limit pause between calls to respect Gemini free RPM limits
        if step_num > 0 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        payload = {
            "model": config.model_id,
            "messages": messages,
            "tools": AGENT_TOOLS,
            "temperature": config.temperature or 0.0,
        }

        t0_call = time.perf_counter()
        try:
            resp = requests.post(
                GEMINI_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            call_lat = round((time.perf_counter() - t0_call) * 1000, 3)
            builder.add_error(f"Network error calling model: {exc}")
            provider_error = f"Network error: {exc}"
            break

        call_lat = round((time.perf_counter() - t0_call) * 1000, 3)

        if resp.status_code != 200:
            err_msg = f"API error HTTP {resp.status_code}: {resp.text[:300]}"
            builder.add_error(err_msg)
            provider_error = err_msg
            break

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            builder.add_error("Provider returned empty choices array.")
            provider_error = "Empty choices array"
            break

        msg = choices[0].get("message", {})
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        tool_calls = msg.get("tool_calls")
        content = msg.get("content")

        if tool_calls:
            # Model invoked one or more tools
            for tc in tool_calls:
                call_id = tc.get("id", f"call_{step_num}")
                fn_obj = tc.get("function", {})
                fn_name = fn_obj.get("name", "")
                raw_args = fn_obj.get("arguments", "{}")

                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    fn_args = {}

                if fn_name == "submit_answer":
                    sub_ans = fn_args.get("answer", "")
                    final_answer = str(sub_ans)
                    t0_sub = time.perf_counter()
                    sub_res = submit_answer(final_answer)
                    sub_lat = round((time.perf_counter() - t0_sub) * 1000, 3)
                    builder.add_submit(
                        answer=final_answer,
                        result=sub_res,
                        latency_ms=sub_lat,
                        llm_output=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    break
                elif fn_name in TOOL_REGISTRY:
                    tool_fn = TOOL_REGISTRY[fn_name]
                    t0_exec = time.perf_counter()
                    try:
                        tool_res = tool_fn(**fn_args)
                    except Exception as exc:
                        tool_res = ToolResult(
                            tool_name=fn_name,
                            success=False,
                            output=None,
                            error=f"Error invoking {fn_name}: {exc}",
                            latency_ms=0.0,
                        )
                    exec_lat = round((time.perf_counter() - t0_exec) * 1000, 3)

                    builder.add_tool_call(
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=tool_res,
                        latency_ms=exec_lat,
                        llm_output=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )

                    # Prepare conversation messages for next turn
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_res.output if tool_res.success else {"error": tool_res.error}),
                    })
                else:
                    err_res = ToolResult(
                        tool_name=fn_name,
                        success=False,
                        output=None,
                        error=f"Tool '{fn_name}' not found in registry.",
                        latency_ms=0.0,
                    )
                    builder.add_tool_call(
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=err_res,
                        latency_ms=0.0,
                        llm_output=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"error": f"Tool '{fn_name}' not found."}),
                    })

            if final_answer is not None:
                break

        elif content:
            # Model responded with plain text (no tool calls)
            builder.add_llm_response(
                content=content,
                latency_ms=call_lat,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # Check if task policy forbade tools or if model answered directly
            if not task.expected_tools or not task.answerable:
                # Direct submission
                candidate = content.strip()
                if "unanswerable" in candidate.lower():
                    final_answer = "unanswerable"
                else:
                    final_answer = candidate
                sub_res = submit_answer(final_answer)
                builder.add_submit(
                    answer=final_answer,
                    result=sub_res,
                    latency_ms=0.0,
                    llm_output=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                break
            else:
                # Prompt model to submit via submit_answer
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Please submit your final answer using the submit_answer tool.",
                })
        else:
            builder.add_error("Received empty response from assistant.")
            break

    # If loop ended without submission
    if final_answer is None and not provider_error:
        # Check if any step had text or attempt
        builder.add_error("Max steps reached without submitting an answer.")
        sub_res = submit_answer("")
        builder.add_submit("", sub_res, latency_ms=0.0)
        final_answer = None

    # Grade the trajectory
    grading = grade(task, final_answer)

    if grading.correct:
        outcome = Outcome.success
    elif provider_error or builder._errors:
        outcome = Outcome.error
    else:
        outcome = Outcome.failure

    return builder.finish(final_answer=final_answer, outcome=outcome)


def run_llm_benchmark(
    tasks: list[TaskRecord],
    config: ExecutionConfig,
    api_key: str | None = None,
    budget: RequestBudget | None = None,
    rate_limit_delay: float = 2.0,
) -> list[TrajectoryRecord]:
    """Execute a list of tasks and return trajectories."""
    trajectories: list[TrajectoryRecord] = []
    for idx, task in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] Running task {task.task_id} ({task.template_family}) ...")
        traj = run_llm_task(
            task=task,
            config=config,
            api_key=api_key,
            budget=budget,
            rate_limit_delay=rate_limit_delay,
        )
        trajectories.append(traj)
        print(f"       Outcome: {traj.outcome.value}, Final Answer: {traj.final_answer!r}, Steps: {len(traj.steps)}")
    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM benchmark agent.")
    parser.add_argument("--tasks", type=pathlib.Path, default=_REPO_ROOT / "data" / "adversarial" / "tasks.jsonl")
    parser.add_argument("--output", type=pathlib.Path, default=_REPO_ROOT / "data" / "adversarial" / "trajectories_gemini_3.8_flash.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run (e.g. 2 for smoke test)")
    args = parser.parse_args()

    from code.tasks.adversarial_generator import load_tasks_jsonl

    all_tasks = load_tasks_jsonl(args.tasks)
    if args.limit:
        run_tasks = all_tasks[:args.limit]
    else:
        run_tasks = all_tasks

    config = ExecutionConfig(
        model_id=args.model,
        prompt_version="v1",
        tool_selection_strategy="openai_function_calling",
        temperature=0.0,
        max_steps=5,
        retry_budget=0,
    )

    budget = RequestBudget(max_requests=args.max_requests)
    print(f"Running LLM agent on {len(run_tasks)} tasks with model {args.model} (budget={args.max_requests}) ...")
    trajs = run_llm_benchmark(run_tasks, config, budget=budget)

    export_trajectories_jsonl(trajs, args.output, overwrite=True)
    print(f"Wrote {len(trajs)} trajectories to {args.output}")
    print(f"Total API requests made: {budget.total_requests}")
