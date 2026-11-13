"""The A2 agent: one configuration-driven loop for every compared system.

``run_agent(task, cfg)`` runs one task with one configuration (``configs/*.yaml``)
and returns an ``AgentRun``: the A1-format ``TrajectoryRecord`` plus the A2
measurements (per-turn LLM latency, tokens, grader v2 verdict, A1 failure
detectors, guard interventions and the MLflow trace id).

Strategies
----------
abstain    B0, trivial: submits "unanswerable" without calling a model.
direct     B1, unadapted: one model call without tools; the reply is the answer.
tool_loop  B2 and the proposed system: the A1 function-calling loop. With every
           feature switched off it behaves like the A1 loop
           (code/experiments/agent_executor.py), with one deliberate change: it
           never reads the task's ``expected_tools``, which a deployed agent
           cannot see. The A1 loop used them to decide whether a plain-text reply
           counts as the answer.

Proposed-system features (all off for B2)
-----------------------------------------
retry_budget    transient tool failures (HTTP 5xx, timeouts) are retried by the
                harness up to this many times per call before the model sees them.
error_feedback  failed tool results carry a short hint on how to fix the call.
loop_guard      an identical repeated tool call is not executed; the model is told
                to change the arguments or submit.
step_guard      before the last turn the model is told to submit; if it still has
                not, its last text reply becomes the answer instead of no answer.
single_tool_call
                only the first tool call of a turn runs, and the history shows only
                that call. Small models otherwise plan a whole chain at once with
                placeholders (e.g. calculate("bonus_rate * salary")) before seeing
                any result.
exemplars       worked examples from the train split appended to the system prompt.

Latency: each step's ``latency_ms`` is the wall time of the LLM call that
produced it (counted once per turn) plus the tool's execution time, so
``metrics.total_latency_ms`` is the agent's real end-to-end time. A1 recorded
only tool execution time.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from code.agent import tools as tool_layer
from code.agent.config import AgentConfig, exemplars, system_prompt, tool_schemas
from code.agent.trajectory import TrajectoryBuilder
from code.annotation.detectors import STEP_LIMIT, detect
from code.experiments.agent_executor import classify_process
from code.experiments.providers import ProviderClient, ProviderException
from code.shared.models import Outcome, TaskRecord, ToolResult, TrajectoryRecord
from code.tasks.grader_v2 import grade_v2
from code.tracing.agent_tracer import AgentTracer

ABSTAIN_ANSWER = "unanswerable"
SUBMIT_NUDGE = "Please submit your final answer using the submit_answer tool."
LAST_TURN_WARNING = ("This is your last turn. Call submit_answer now with your best answer, "
                     "or 'unanswerable: <reason>' if the tools did not give the information.")
TRANSIENT_ERROR = re.compile(r"\b5\d\d\b|unavailable|timed? ?out|temporar", re.IGNORECASE)


@dataclass
class TurnRecord:
    """One LLM call."""

    index: int
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    content: str | None
    tool_calls: list[dict[str, Any]]


@dataclass
class ToolCallRecord:
    """One tool call as the UI and the monitor see it."""

    turn: int
    tool: str
    arguments: dict[str, Any]
    success: bool
    output: Any
    error: str | None
    latency_ms: float
    attempts: int = 1
    blocked_repeat: bool = False


@dataclass
class AgentRun:
    """Everything one run produces."""

    run_id: str
    task_id: str
    config_name: str
    config_hash: str
    model: str | None
    seed: int
    trajectory: TrajectoryRecord
    final_answer: str | None
    correct: bool
    grading_explanation: str
    outcome: str
    process_category: str
    detectors: dict[str, Any]
    turns: list[TurnRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    wall_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_errors: int = 0
    malformed_calls: int = 0
    retries_used: int = 0
    blocked_repeats: int = 0
    forced_submit: bool = False
    provider_error: str | None = None
    trace_id: str | None = None

    @property
    def llm_calls(self) -> int:
        return len(self.turns)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict[str, Any]:
        """Flat record (one JSONL row per run) used by evaluation and monitoring."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "config_name": self.config_name,
            "config_hash": self.config_hash,
            "model": self.model,
            "seed": self.seed,
            "final_answer": self.final_answer,
            "correct": self.correct,
            "outcome": self.outcome,
            "grading_explanation": self.grading_explanation,
            "process_category": self.process_category,
            "steps": self.trajectory.metrics.total_steps,
            "llm_calls": self.llm_calls,
            "tool_calls": len(self.tool_calls),
            "tool_errors": self.tool_errors,
            "malformed_calls": self.malformed_calls,
            "retries_used": self.retries_used,
            "blocked_repeats": self.blocked_repeats,
            "forced_submit": self.forced_submit,
            "submitted": self.final_answer is not None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_latency_ms": round(self.llm_latency_ms, 3),
            "tool_latency_ms": round(self.tool_latency_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "loop_strict": bool(self.detectors.get("loop_strict")),
            "loop_extended": bool(self.detectors.get("loop_extended")),
            "malformed_any": bool(self.detectors.get("malformed_any")),
            "malformed_strict": bool(self.detectors.get("malformed_strict")),
            "max_identical_calls": int(self.detectors.get("max_identical_calls", 0)),
            "provider_error": self.provider_error,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> dict[str, Any]:
        """Full record: summary, turns, tool calls and the A1-format trajectory."""
        return {
            **self.summary(),
            "turns": [asdict(t) for t in self.turns],
            "tool_call_log": [asdict(c) for c in self.tool_calls],
            "trajectory": self.trajectory.model_dump(mode="json"),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exemplar_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    lines = ["", "Worked examples from earlier tasks (the data in them may differ from today's task):"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}. Task: {ex['task']}")
        for call in ex.get("calls", []):
            lines.append(f"  - {call['tool']}({json.dumps(call['arguments'])}) -> {call['result']}")
        lines.append(f"  - submit_answer({json.dumps({'answer': ex['answer']})})")
        if ex.get("note"):
            lines.append(f"  Note: {ex['note']}")
    return "\n".join(lines)


def _hint(tool: str, error: str) -> str:
    if TRANSIENT_ERROR.search(error):
        return ("The tool is failing on the server side. If it keeps failing, the information cannot be "
                "obtained: submit 'unanswerable: <reason>' rather than guessing.")
    if tool == "retrieve":
        return "Use 1 to 3 short keywords."
    if tool == "query_records":
        return ("Valid fields are emp_id, name, department, salary. Use operation='filter' with field and value, "
                "or operation='aggregate' with aggregate and agg_field.")
    if tool == "calculate":
        return "Use only numbers, + - * / ** and round(x, n) in the expression."
    if tool == "read_file":
        return "Use the exact filename from the question; the error lists the available files."
    return "Check the tool name and arguments against the tool descriptions."


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Return (arguments, malformed)."""
    if isinstance(raw, dict):
        return raw, False
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (json.JSONDecodeError, TypeError):
        return {"_raw_arguments": raw}, True
    if not isinstance(parsed, dict):
        return {"_raw_arguments": parsed}, True
    return parsed, False


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_agent(
    task: TaskRecord,
    cfg: AgentConfig,
    client: ProviderClient | None = None,
    tracer: AgentTracer | None = None,
    seed: int = 203,
    timeout: float = 600.0,
) -> AgentRun:
    """Run ``task`` with configuration ``cfg`` and grade the answer with grader v2."""
    tracer = tracer or AgentTracer(enabled=False)
    if cfg.strategy != "abstain" and client is None:
        from code.experiments.local_llm import OllamaClient

        client = OllamaClient(seed=seed)

    state = _RunState(task, cfg, seed)
    with tracer.span("agent", "AGENT",
                     inputs={"task_id": task.task_id, "prompt": task.prompt},
                     attributes={"config_name": cfg.name, "config_hash": state.config_hash, "model": cfg.model,
                                 "prompt_version": cfg.prompt_version, "tool_desc_version": cfg.tool_desc_version,
                                 "seed": seed, "template_family": task.template_family}) as root:
        state.trace_id = tracer.trace_id_of(root)
        if cfg.strategy == "abstain":
            state.submit(ABSTAIN_ANSWER, llm_output=None)
        elif cfg.strategy == "direct":
            _run_direct(state, client, tracer, timeout)
        else:
            _run_tool_loop(state, client, tracer, timeout)
        run = state.finish(tracer)
        root.set_outputs({"final_answer": run.final_answer, "correct": run.correct, "outcome": run.outcome})
        root.set_attributes({"steps": run.trajectory.metrics.total_steps, "llm_calls": run.llm_calls,
                             "total_tokens": run.total_tokens, "wall_ms": run.wall_ms,
                             "tool_errors": run.tool_errors, "process_category": run.process_category})
    return run


class _RunState:
    """Mutable state of one run, shared by the strategies."""

    def __init__(self, task: TaskRecord, cfg: AgentConfig, seed: int) -> None:
        self.task = task
        self.cfg = cfg
        self.seed = seed
        self.config_hash = cfg.config_hash()
        self.builder = TrajectoryBuilder(task=task, configuration_hash=self.config_hash,
                                         model_version=f"ollama:{cfg.model}" if cfg.model else "none",
                                         prompt_version=f"{cfg.prompt_version}/{cfg.tool_desc_version}")
        self.t0 = time.perf_counter()
        self.final_answer: str | None = None
        self.turns: list[TurnRecord] = []
        self.calls: list[ToolCallRecord] = []
        self.pending_llm_ms = 0.0
        self.pending_prompt_tokens: int | None = None
        self.pending_completion_tokens: int | None = None
        self.tool_ms = 0.0
        self.had_tool_error = False
        self.had_recovery = False
        self.had_arg_error = False
        self.malformed = 0
        self.retries_used = 0
        self.blocked = 0
        self.forced = False
        self.provider_error: str | None = None
        self.trace_id: str | None = None

    # The LLM time and tokens of a turn are charged to the first step it produces.
    def _take_turn_cost(self) -> tuple[float, int | None, int | None]:
        cost = (self.pending_llm_ms, self.pending_prompt_tokens, self.pending_completion_tokens)
        self.pending_llm_ms, self.pending_prompt_tokens, self.pending_completion_tokens = 0.0, None, None
        return cost

    def record_turn(self, index: int, resp: Any) -> None:
        self.turns.append(TurnRecord(index=index, latency_ms=resp.latency_ms, prompt_tokens=resp.prompt_tokens,
                                     completion_tokens=resp.completion_tokens, content=resp.content,
                                     tool_calls=resp.tool_calls or []))
        self.pending_llm_ms = resp.latency_ms
        self.pending_prompt_tokens = resp.prompt_tokens
        self.pending_completion_tokens = resp.completion_tokens

    def submit(self, answer: str, llm_output: str | None) -> None:
        llm_ms, pt, ct = self._take_turn_cost()
        result = tool_layer.submit_answer(answer)
        self.builder.add_submit(answer=answer, result=result, latency_ms=llm_ms, llm_output=llm_output,
                                prompt_tokens=pt, completion_tokens=ct)
        self.final_answer = answer

    def add_tool_step(self, rec: ToolCallRecord, result: ToolResult, llm_output: str | None) -> None:
        llm_ms, pt, ct = self._take_turn_cost()
        self.builder.add_tool_call(tool_name=rec.tool, arguments=rec.arguments, result=result,
                                   latency_ms=llm_ms + rec.latency_ms, llm_output=llm_output,
                                   prompt_tokens=pt, completion_tokens=ct)
        self.calls.append(rec)

    def add_text_step(self, content: str) -> None:
        llm_ms, pt, ct = self._take_turn_cost()
        self.builder.add_llm_response(content=content, latency_ms=llm_ms, prompt_tokens=pt, completion_tokens=ct)

    def finish(self, tracer: AgentTracer) -> AgentRun:
        if self.pending_llm_ms:
            # A turn that produced no step (empty reply) still cost time and tokens.
            self.add_text_step("")
        if self.final_answer is None and self.provider_error is None:
            self.builder.add_error(STEP_LIMIT)
        if self.provider_error:
            self.builder.add_error(f"Provider error: {self.provider_error}")

        with tracer.span("grader", "EVALUATOR", inputs={"answer": self.final_answer, "gold": self.task.gold_answer,
                                                        "rule": self.task.evaluation_rule.value}) as gspan:
            grading = grade_v2(self.task, self.final_answer)
            gspan.set_outputs({"correct": grading.correct, "explanation": grading.detail})

        if grading.correct:
            outcome = Outcome.success
        elif self.provider_error:
            outcome = Outcome.error
        else:
            outcome = Outcome.failure
        trajectory = self.builder.finish(final_answer=self.final_answer, outcome=outcome)
        category = classify_process(task=self.task, grading_correct=grading.correct, steps=trajectory.steps,
                                    had_tool_error=self.had_tool_error, had_recovery=self.had_recovery,
                                    had_arg_error=self.had_arg_error, provider_error=self.provider_error)
        detectors = detect(trajectory.model_dump(mode="json"))
        llm_ms = sum(t.latency_ms for t in self.turns)
        return AgentRun(
            run_id=trajectory.run_id, task_id=self.task.task_id, config_name=self.cfg.name,
            config_hash=self.config_hash, model=self.cfg.model, seed=self.seed, trajectory=trajectory,
            final_answer=self.final_answer, correct=grading.correct, grading_explanation=grading.detail,
            outcome=outcome.value, process_category=category, detectors=detectors, turns=self.turns,
            tool_calls=self.calls, llm_latency_ms=llm_ms, tool_latency_ms=self.tool_ms,
            wall_ms=(time.perf_counter() - self.t0) * 1000,
            prompt_tokens=sum(t.prompt_tokens or 0 for t in self.turns),
            completion_tokens=sum(t.completion_tokens or 0 for t in self.turns),
            tool_errors=sum(1 for c in self.calls if not c.success), malformed_calls=self.malformed,
            retries_used=self.retries_used, blocked_repeats=self.blocked, forced_submit=self.forced,
            provider_error=self.provider_error, trace_id=self.trace_id or trajectory.run_id,
        )


def _call_llm(state: _RunState, client: ProviderClient, tracer: AgentTracer, turn: int,
              messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, timeout: float) -> Any | None:
    cfg = state.cfg
    with tracer.span(f"llm_turn_{turn}", "CHAT_MODEL", inputs={"messages": messages[-3:]},
                     attributes={"model": cfg.model, "temperature": cfg.temperature, "turn": turn}) as span:
        try:
            resp = client.call_chat_completion(model_id=cfg.model, messages=messages, tools=tools,
                                               temperature=cfg.temperature, timeout=timeout,
                                               seed=state.seed, max_tokens=cfg.max_tokens)
        except ProviderException as exc:
            state.provider_error = str(exc)
            span.set_attributes({"error": str(exc)})
            return None
        span.set_outputs({"content": resp.content, "tool_calls": resp.tool_calls})
        span.set_attributes({"latency_ms": resp.latency_ms, "prompt_tokens": resp.prompt_tokens,
                             "completion_tokens": resp.completion_tokens})
    state.record_turn(turn, resp)
    return resp


def _run_direct(state: _RunState, client: ProviderClient, tracer: AgentTracer, timeout: float) -> None:
    messages = [{"role": "system", "content": system_prompt(state.cfg)},
                {"role": "user", "content": state.task.prompt}]
    resp = _call_llm(state, client, tracer, 0, messages, None, timeout)
    if resp is not None and resp.content and resp.content.strip():
        state.submit(_first_line(resp.content), llm_output=resp.content)


def _execute_tool(state: _RunState, tracer: AgentTracer, name: str, args: dict[str, Any], turn: int,
                  malformed: bool, raw_args: Any) -> tuple[ToolCallRecord, ToolResult]:
    cfg = state.cfg
    with tracer.span(name or "unknown_tool", "TOOL", inputs=args, attributes={"turn": turn}) as span:
        attempts = 0
        t0 = time.perf_counter()
        while True:
            attempts += 1
            if malformed:
                result = ToolResult(tool_name=name or "unknown", success=False, output=None, latency_ms=0.0,
                                    error=f"Malformed arguments for {name}: {raw_args}")
            elif name not in tool_layer.TOOL_REGISTRY or name == "submit_answer":
                result = ToolResult(tool_name=name or "unknown", success=False, output=None, latency_ms=0.0,
                                    error=f"Tool '{name}' not found.")
            else:
                try:
                    result = tool_layer.TOOL_REGISTRY[name](**args)
                except Exception as exc:  # wrong argument names or types
                    result = ToolResult(tool_name=name, success=False, output=None, latency_ms=0.0,
                                        error=f"Invocation error for {name}: {exc}")
            transient = not result.success and bool(TRANSIENT_ERROR.search(result.error or ""))
            if transient and state.retries_used < cfg.retry_budget:
                state.retries_used += 1
                continue
            break
        elapsed = (time.perf_counter() - t0) * 1000
        span.set_outputs({"success": result.success, "output": result.output, "error": result.error})
        span.set_attributes({"latency_ms": elapsed, "attempts": attempts})
    state.tool_ms += elapsed
    rec = ToolCallRecord(turn=turn, tool=name, arguments=args, success=result.success, output=result.output,
                         error=result.error, latency_ms=elapsed, attempts=attempts)
    return rec, result


def _run_tool_loop(state: _RunState, client: ProviderClient, tracer: AgentTracer, timeout: float) -> None:
    cfg, task = state.cfg, state.task
    tools = tool_schemas(cfg)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(cfg) + _exemplar_block(exemplars(cfg))},
        {"role": "user", "content": task.prompt},
    ]
    seen_calls: set[tuple[str, str]] = set()
    last_text: str | None = None

    for turn in range(cfg.max_steps):
        if cfg.step_guard and turn == cfg.max_steps - 1 and turn > 0:
            messages.append({"role": "user", "content": LAST_TURN_WARNING})
        resp = _call_llm(state, client, tracer, turn, messages, tools, timeout)
        if resp is None:
            return

        if resp.tool_calls:
            calls = resp.tool_calls[:1] if cfg.single_tool_call else resp.tool_calls
            messages.append({"role": "assistant", "content": resp.content, "tool_calls": calls})
            for i, tc in enumerate(calls):
                call_id = tc.get("id", f"call_{turn}_{i}")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                args, malformed = _parse_arguments(raw_args)

                if name == "submit_answer" and not malformed and "answer" in args:
                    state.submit(str(args["answer"]), llm_output=resp.content)
                    return

                if malformed or name == "submit_answer":
                    malformed = True
                    state.had_arg_error = True
                    state.malformed += 1

                key = (name, json.dumps(args, sort_keys=True))
                if cfg.loop_guard and key in seen_calls and not malformed:
                    state.blocked += 1
                    error = (f"You already called {name} with these arguments and the result will not change. "
                             "Use different arguments, a different tool, or call submit_answer.")
                    result = ToolResult(tool_name=name, success=False, output=None, latency_ms=0.0, error=error)
                    rec = ToolCallRecord(turn=turn, tool=name, arguments=args, success=False, output=None,
                                         error=error, latency_ms=0.0, blocked_repeat=True)
                else:
                    seen_calls.add(key)
                    rec, result = _execute_tool(state, tracer, name, args, turn, malformed, raw_args)
                    if not result.success:
                        rejected = bool(result.error) and ("Invocation error" in result.error or "not found." in result.error)
                        if rejected and not malformed:
                            state.had_arg_error = True
                            state.malformed += 1
                        state.had_tool_error = True
                    elif state.had_tool_error:
                        state.had_recovery = True
                state.add_tool_step(rec, result, llm_output=resp.content)

                if result.success:
                    content: Any = result.output
                else:
                    content = {"error": result.error}
                    if cfg.error_feedback and not rec.blocked_repeat:
                        content["hint"] = _hint(name, result.error or "")
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                 "content": json.dumps(content, default=str)})
            continue

        text = (resp.content or "").strip()
        if not text:
            return
        last_text = text
        state.add_text_step(text)
        if ABSTAIN_ANSWER in text.lower():
            state.submit(ABSTAIN_ANSWER, llm_output=text)
            return
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": SUBMIT_NUDGE})

    if cfg.step_guard and state.final_answer is None and last_text:
        state.forced = True
        state.submit(_first_line(last_text), llm_output=last_text)
