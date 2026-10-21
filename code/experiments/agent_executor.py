"""
Agent execution loop for live LLM providers in AgentEvalOps.

Handles:
  - Multi-turn interaction up to max_steps
  - Immediate budget checking before every inference call
  - Tool execution via TOOL_REGISTRY
  - Step recording and token tracking via TrajectoryBuilder
  - Final answer grading via code.tasks.grader.grade
  - Fine-grained 8-way outcome classification
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from code.agent.tools import TOOL_REGISTRY, submit_answer
from code.agent.trajectory import TrajectoryBuilder
from code.experiments.providers import (
    AGENT_TOOLS,
    ErrorType,
    ProviderClient,
    ProviderException,
    SYSTEM_INSTRUCTION,
)
from code.shared.models import (
    ExecutionConfig,
    Outcome,
    TaskRecord,
    ToolResult,
    TrajectoryRecord,
)
from code.tasks.grader import grade


@dataclass
class ExecutionResult:
    trajectory: TrajectoryRecord | None
    final_answer: str | None
    outcome: Outcome
    process_category: str
    requests_made: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    provider_error: str | None = None
    error_type: ErrorType | None = None
    retry_after: float | None = None


class RequestTracker:
    """Monitors and enforces the hard request cap across the experiment."""

    def __init__(self, max_requests: int = 100):
        self.max_requests = max_requests
        self.total_requests = 0

    def check_and_consume(self) -> None:
        """Verify budget immediately before every API request."""
        if self.total_requests >= self.max_requests:
            raise ProviderException(
                f"Global request cap reached ({self.total_requests}/{self.max_requests}).",
                ErrorType.BUDGET_EXHAUSTED,
            )
        self.total_requests += 1


def classify_process(
    task: TaskRecord,
    grading_correct: bool,
    steps: list[Any],
    had_tool_error: bool,
    had_recovery: bool,
    had_arg_error: bool,
    provider_error: str | None,
) -> str:
    """Classify execution into one of 8 diagnostic outcome categories."""
    if provider_error:
        return "provider_api_error"

    # Check for process violations (e.g. used tools when expected_tools was empty)
    tool_steps = [s for s in steps if getattr(s, "action_type", None) and s.action_type.value == "tool_call"]
    if task.expected_tools == [] and len(tool_steps) > 0 and grading_correct:
        return "correct_process_violation"

    if had_recovery and grading_correct:
        return "tool_failure_and_recovery"

    if had_arg_error or (len(tool_steps) > 0 and any(not s.tool_result.success for s in tool_steps if s.tool_result)):
        if not grading_correct:
            return "incorrect_tool_choice_or_arguments"

    if grading_correct:
        return "correct_acceptable_process"

    # If answer was submitted but incorrect
    submit_steps = [s for s in steps if getattr(s, "action_type", None) and s.action_type.value == "submit_answer"]
    if submit_steps:
        return "incorrect_final_answer"

    return "agent_execution_error"


def execute_agent_task(
    task: TaskRecord,
    provider_client: ProviderClient,
    model_id: str,
    config: ExecutionConfig,
    request_tracker: RequestTracker | None = None,
    timeout: float = 35.0,
    rate_limit_delay: float = 1.0,
    tracer: Any = None,
) -> ExecutionResult:
    """
    Execute a single task through the multi-turn agent loop.
    Checks budget immediately before every model invocation.
    """
    t0_start = time.perf_counter()
    config_hash = config.compute_hash()

    builder = TrajectoryBuilder(
        task=task,
        configuration_hash=config_hash,
        model_version=f"{provider_client.provider_name}:{model_id}",
        prompt_version=config.prompt_version,
        tracer=tracer,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": task.prompt},
    ]

    final_answer: str | None = None
    provider_error: str | None = None
    error_type: ErrorType | None = None
    retry_after: float | None = None
    requests_made = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    had_tool_error = False
    had_recovery = False
    had_arg_error = False

    for step_num in range(config.max_steps):
        # 1. Budget check immediately before every model request
        if request_tracker:
            try:
                request_tracker.check_and_consume()
                requests_made += 1
            except ProviderException as exc:
                provider_error = str(exc)
                error_type = exc.error_type
                retry_after = exc.retry_after
                break

        # Rate limit pause between multi-turn calls
        if step_num > 0 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # 2. Invoke model
        try:
            llm_resp = provider_client.call_chat_completion(
                model_id=model_id,
                messages=messages,
                tools=AGENT_TOOLS,
                temperature=config.temperature or 0.0,
                timeout=timeout,
            )
        except ProviderException as exc:
            provider_error = str(exc)
            error_type = exc.error_type
            retry_after = exc.retry_after
            break
        except Exception as exc:
            provider_error = f"Unexpected model error: {exc}"
            error_type = ErrorType.SERVER_ERROR
            break

        if llm_resp.prompt_tokens:
            total_prompt_tokens += llm_resp.prompt_tokens
        if llm_resp.completion_tokens:
            total_completion_tokens += llm_resp.completion_tokens

        # 3. Process model output
        if llm_resp.tool_calls:
            # Preserve complete assistant message including tool_calls and Gemini metadata (thought_signature)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": llm_resp.content if llm_resp.content is not None else None,
                "tool_calls": llm_resp.tool_calls,
            }
            if hasattr(llm_resp, "raw_message") and isinstance(llm_resp.raw_message, dict):
                for k in ("thought_signature", "thoughtSignature", "reasoning_content"):
                    if k in llm_resp.raw_message:
                        assistant_msg[k] = llm_resp.raw_message[k]
            messages.append(assistant_msg)

            for tc in llm_resp.tool_calls:
                call_id = tc.get("id", f"call_{step_num}")
                fn_obj = tc.get("function", {})
                fn_name = fn_obj.get("name", "")
                raw_args = fn_obj.get("arguments", "{}")

                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(fn_args, dict):
                        fn_args = {"_raw_arguments": fn_args}
                        had_arg_error = True
                except Exception:
                    fn_args = {"_raw_arguments": raw_args}
                    had_arg_error = True

                if fn_name == "submit_answer" and not had_arg_error and "answer" in fn_args:
                    sub_ans = fn_args.get("answer", "")
                    final_answer = str(sub_ans)
                    t0_sub = time.perf_counter()
                    sub_res = submit_answer(final_answer)
                    sub_lat = round((time.perf_counter() - t0_sub) * 1000, 3)
                    builder.add_submit(
                        answer=final_answer,
                        result=sub_res,
                        latency_ms=sub_lat,
                        llm_output=llm_resp.content,
                        prompt_tokens=llm_resp.prompt_tokens,
                        completion_tokens=llm_resp.completion_tokens,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"status": "submitted", "answer": final_answer}),
                    })
                    break
                elif fn_name == "submit_answer":
                    # Malformed submit_answer arguments - record as failed tool call without silent repair
                    had_arg_error = True
                    err_res = ToolResult(
                        tool_name="submit_answer",
                        success=False,
                        output=None,
                        error=f"Malformed arguments for submit_answer: {raw_args}",
                        latency_ms=0.0,
                    )
                    builder.add_tool_call(
                        tool_name="submit_answer",
                        arguments=fn_args,
                        result=err_res,
                        latency_ms=0.0,
                        llm_output=llm_resp.content,
                        prompt_tokens=llm_resp.prompt_tokens,
                        completion_tokens=llm_resp.completion_tokens,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"error": f"Malformed arguments for submit_answer: {raw_args}"}),
                    })
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
                            error=f"Invocation error for {fn_name}: {exc}",
                            latency_ms=0.0,
                        )
                        had_arg_error = True
                    exec_lat = round((time.perf_counter() - t0_exec) * 1000, 3)

                    if not tool_res.success:
                        had_tool_error = True
                    elif had_tool_error:
                        had_recovery = True

                    builder.add_tool_call(
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=tool_res,
                        latency_ms=exec_lat,
                        llm_output=llm_resp.content,
                        prompt_tokens=llm_resp.prompt_tokens,
                        completion_tokens=llm_resp.completion_tokens,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_res.output if tool_res.success else {"error": tool_res.error}),
                    })
                else:
                    had_arg_error = True
                    err_res = ToolResult(
                        tool_name=fn_name,
                        success=False,
                        output=None,
                        error=f"Tool '{fn_name}' not found.",
                        latency_ms=0.0,
                    )
                    builder.add_tool_call(
                        tool_name=fn_name,
                        arguments=fn_args,
                        result=err_res,
                        latency_ms=0.0,
                        llm_output=llm_resp.content,
                        prompt_tokens=llm_resp.prompt_tokens,
                        completion_tokens=llm_resp.completion_tokens,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"error": f"Tool '{fn_name}' not found in registry."}),
                    })

            if final_answer is not None:
                break

        elif llm_resp.content:
            builder.add_llm_response(
                content=llm_resp.content,
                latency_ms=llm_resp.latency_ms,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
            )

            # Direct submission if no tools required or explicit refusal
            candidate = llm_resp.content.strip()
            if "unanswerable" in candidate.lower() or not task.expected_tools:
                final_answer = "unanswerable" if "unanswerable" in candidate.lower() else candidate
                sub_res = submit_answer(final_answer)
                builder.add_submit(
                    answer=final_answer,
                    result=sub_res,
                    latency_ms=0.0,
                    llm_output=llm_resp.content,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                )
                break
            else:
                messages.append({"role": "assistant", "content": llm_resp.content})
                messages.append({
                    "role": "user",
                    "content": "Please submit your final answer using the submit_answer tool.",
                })
        else:
            builder.add_error("Received empty response from assistant.")
            break

    duration_ms = round((time.perf_counter() - t0_start) * 1000, 3)

    # 4. If loop ended without submission due to provider error
    if provider_error and final_answer is None:
        return ExecutionResult(
            trajectory=None,
            final_answer=None,
            outcome=Outcome.error,
            process_category="provider_api_error",
            requests_made=requests_made,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            duration_ms=duration_ms,
            provider_error=provider_error,
            error_type=error_type,
            retry_after=retry_after,
        )

    # If steps exhausted without submission, represent max-step exhaustion explicitly at trajectory level
    if final_answer is None:
        builder.add_error("Max steps reached without submitting an answer.")

    # 5. Independent Grading
    grading = grade(task, final_answer)

    if grading.correct:
        outcome = Outcome.success
    elif builder._errors:
        outcome = Outcome.error
    else:
        outcome = Outcome.failure

    category = classify_process(
        task=task,
        grading_correct=grading.correct,
        steps=builder._steps,
        had_tool_error=had_tool_error,
        had_recovery=had_recovery,
        had_arg_error=had_arg_error,
        provider_error=provider_error,
    )

    trajectory = builder.finish(final_answer=final_answer, outcome=outcome)

    return ExecutionResult(
        trajectory=trajectory,
        final_answer=final_answer,
        outcome=outcome,
        process_category=category,
        requests_made=requests_made,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        duration_ms=duration_ms,
        provider_error=provider_error,
        error_type=error_type,
        retry_after=retry_after,
    )
