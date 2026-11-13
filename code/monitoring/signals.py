"""Per-request monitoring signals, computed from a run summary (code/agent/loop.AgentRun.summary()).

Label-free (available for every live request):
  tool_failures   failed tool calls plus harness retries (a retry means a call failed
                  even if the model never saw it)
  malformed       calls the tools rejected for bad arguments (A1 malformed detector)
  loop            A1 loop detector (3+ identical calls, or step limit without an answer)
  steps           trajectory steps
  latency_s       end-to-end wall time
  tokens          prompt + completion tokens

Labelled (only once the true answer is known, e.g. replayed benchmark traffic or a
canary task): correct.

The two label-free signals used for drift detection (chosen in Part E on
validation data) are listed in MONITORED; the dashboard shows all of them.
"""

from __future__ import annotations

from typing import Any

LABEL_FREE = ("tool_failures", "malformed", "loop", "steps", "latency_s", "tokens")
LABELLED = ("correct",)
MONITORED = ("tool_failures", "latency_s")


def signals(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "tool_failures": float(summary.get("tool_errors", 0) + summary.get("retries_used", 0)),
        "malformed": float(bool(summary.get("malformed_any") or summary.get("malformed_calls"))),
        "loop": float(bool(summary.get("loop_extended"))),
        "steps": float(summary.get("steps", 0)),
        "latency_s": float(summary.get("wall_ms", 0.0)) / 1000.0,
        "tokens": float(summary.get("total_tokens", 0)),
        "correct": float(bool(summary.get("correct"))),
    }
