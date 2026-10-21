"""
Fault injection for the AgentEvalOps tool layer (Assignment 1, Part D).

The agent loop (code/experiments/agent_executor.py, code/agent/llm_runner.py)
looks tools up in ``code.agent.tools.TOOL_REGISTRY`` at call time.  The
``FaultInjector`` context manager temporarily replaces registry entries with
wrapped versions and restores them on exit, so no agent code has to change.

Fault types
-----------
tool_failure    The call returns success=False with a realistic error message.
                trigger "first_call" (transient: later calls work) or "always"
                (persistent: the tool never works during the run).
slow_response   The call succeeds with identical output, after an added delay.
format_change   The call succeeds with the same information in a different
                representation (CSV <-> JSON records, renamed keys, numbers
                formatted as strings).  ``canonical()`` proves equivalence.

Deliberate vs. natural failures: injected faults never mark themselves in the
tool output (that would tip off the agent).  Instead every injection is
recorded in ``FaultInjector.log`` and the task carries an ``augmentation.fault``
spec, so a trace from a fault-injected task can always be separated from a
naturally failing one.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import io
import json
import time
from typing import Any, Callable

from code.agent import tools as tool_layer
from code.shared.models import ToolResult

FAULT_TYPES = ("tool_failure", "slow_response", "format_change")
TRIGGERS = ("first_call", "always")
FAILURE_MESSAGE = "503 Service Unavailable: upstream {tool} backend did not respond"


@dataclass(frozen=True)
class FaultSpec:
    fault_type: str
    tool: str
    trigger: str = "always"
    delay_ms: int = 0

    def __post_init__(self) -> None:
        if self.fault_type not in FAULT_TYPES:
            raise ValueError(f"unknown fault_type {self.fault_type!r}")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"unknown trigger {self.trigger!r}")
        if self.tool not in tool_layer.TOOL_REGISTRY or self.tool == "submit_answer":
            raise ValueError(f"cannot inject faults into tool {self.tool!r}")
        if self.fault_type == "slow_response" and self.delay_ms <= 0:
            raise ValueError("slow_response needs delay_ms > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Output-format transformations (format_change) and their canonical forms
# ---------------------------------------------------------------------------


def _csv_to_records(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _records_to_csv(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(records)
    return buf.getvalue()


def change_format(tool: str, output: Any) -> Any:
    """Return the same information in a different representation."""
    if tool == "read_file":
        name, content = output["filename"], output["content"]
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list) and parsed and all(isinstance(r, dict) for r in parsed):
            return {"file": name, "format": "csv", "data": _records_to_csv(parsed)}
        if isinstance(parsed, dict):
            return {"file": name, "format": "kv_lines", "data": "\n".join(f"{k}: {v}" for k, v in parsed.items())}
        first_line = content.splitlines()[0] if content else ""
        if parsed is None and "," in first_line:
            return {"file": name, "format": "json_records", "data": json.dumps(_csv_to_records(content))}
        return {"file": name, "format": "text", "data": content}
    if tool == "retrieve":
        return {"results": [{"id": m["fact_id"], "text": m["answer"], "source_question": m["question"], "category": m["topic"]}
                            for m in output["matches"]]}
    if tool == "calculate":
        return {"value": f"{output['result']:,.2f}" if isinstance(output["result"], float) else f"{output['result']:,}",
                "input": output["expression"]}
    if tool == "query_records":
        if isinstance(output, dict) and "rows" in output:
            rows = output["rows"]
            header = list(rows[0].keys()) if rows else []
            return {"columns": header, "values": [[r[h] for h in header] for r in rows], "n": output.get("count", len(rows))}
        return {"aggregate_result": output}
    raise ValueError(f"no format change defined for {tool}")


def canonical(tool: str, output: Any) -> Any:
    """Tool-independent canonical form, used to prove format_change keeps the information."""
    if tool == "read_file":
        if "content" in output:
            fmt, data = None, output["content"]
        else:
            fmt, data = output["format"], output["data"]
        if fmt == "kv_lines":
            pairs = [line.split(": ", 1) for line in data.splitlines() if line]
            return {k: str(v) for k, v in pairs}
        if fmt == "json_records":
            return [{k: str(v) for k, v in r.items()} for r in json.loads(data)]
        if fmt == "csv":
            return [{k: str(v) for k, v in r.items()} for r in _csv_to_records(data)]
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
            if isinstance(parsed, list) and parsed and all(isinstance(r, dict) for r in parsed):
                return [{k: str(v) for k, v in r.items()} for r in parsed]
            return parsed
        except (json.JSONDecodeError, TypeError):
            if data and "," in data.splitlines()[0]:
                return [{k: str(v) for k, v in r.items()} for r in _csv_to_records(data)]
            return data
    if tool == "retrieve":
        items = output["matches"] if "matches" in output else [
            {"fact_id": r["id"], "answer": r["text"], "question": r["source_question"], "topic": r["category"]} for r in output["results"]]
        return sorted((m["fact_id"], m["answer"]) for m in items)
    if tool == "calculate":
        v = output["result"] if "result" in output else float(str(output["value"]).replace(",", ""))
        return round(float(v), 6)
    if tool == "query_records":
        if "rows" in output:
            return [{k: str(v) for k, v in r.items()} for r in output["rows"]]
        if "columns" in output:
            return [{h: str(v) for h, v in zip(output["columns"], vals)} for vals in output["values"]]
        return output.get("aggregate_result", output)
    raise ValueError(tool)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


@dataclass
class FaultInjector:
    """Context manager that wraps TOOL_REGISTRY entries for the duration of one run."""

    specs: list[FaultSpec]
    sleep: Callable[[float], None] = time.sleep
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        tools = [s.tool for s in self.specs]
        if len(tools) != len(set(tools)):
            raise ValueError("at most one fault per tool")
        self._originals: dict[str, Callable[..., ToolResult]] = {}
        self._calls: dict[str, int] = {}

    def _wrap(self, spec: FaultSpec, fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        def wrapped(**kwargs: Any) -> ToolResult:
            n = self._calls[spec.tool] = self._calls.get(spec.tool, 0) + 1
            active = spec.trigger == "always" or (spec.trigger == "first_call" and n == 1)
            if not active:
                return fn(**kwargs)
            entry = {"tool": spec.tool, "call_number": n, "fault_type": spec.fault_type, "arguments": kwargs}
            if spec.fault_type == "tool_failure":
                self.log.append(entry)
                return ToolResult(tool_name=spec.tool, success=False, output=None,
                                  error=FAILURE_MESSAGE.format(tool=spec.tool), latency_ms=0.0)
            res = fn(**kwargs)
            if spec.fault_type == "slow_response":
                self.sleep(spec.delay_ms / 1000.0)
                self.log.append({**entry, "delay_ms": spec.delay_ms})
                return res.model_copy(update={"latency_ms": (res.latency_ms or 0.0) + spec.delay_ms})
            if spec.fault_type == "format_change" and res.success:
                self.log.append(entry)
                return res.model_copy(update={"output": change_format(spec.tool, res.output)})
            return res
        wrapped.__name__ = f"faulty_{spec.tool}"
        return wrapped

    def __enter__(self) -> "FaultInjector":
        for spec in self.specs:
            self._originals[spec.tool] = tool_layer.TOOL_REGISTRY[spec.tool]
            tool_layer.TOOL_REGISTRY[spec.tool] = self._wrap(spec, self._originals[spec.tool])
        return self

    def __exit__(self, *exc: Any) -> None:
        tool_layer.TOOL_REGISTRY.update(self._originals)
        self._originals.clear()
