"""AgentEvalOps service (Part D): REST API and Gradio UI on one port.

    uvicorn code.service.app:app --host 0.0.0.0 --port 7860      (the Docker image runs this)

REST endpoints
  POST /predict      {"task": "...", "fault": "none", "model": null}
                     -> answer, trace id, confidence, latency, steps, tool calls, version
  GET  /health       service and local model server status
  GET  /version      git commit, registry version and alias, configuration hash, model digest
  GET  /monitoring   live drift monitor (label-free signals, detector state, alerts)
  POST /monitoring/reset
  GET  /validation   reported metrics next to the latest live validation run
  POST /validation/run   run the documented fixed validation subset now

The deployed configuration is loaded from the bundled registry (alias from
AGENTEVALOPS_ALIAS, default "production"). Every request is logged to
logs/requests.jsonl with version, latency and confidence; the task text is stored
only as a SHA-256 hash and length, and e-mail addresses / phone numbers are masked
in the logged answer, so no personal information is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from code.agent.adhoc import adhoc_task, is_graded
from code.agent.config import REPO_ROOT
from code.agent.loop import run_agent
from code.augmentation.fault_injection import FaultInjector, FaultSpec
from code.evaluation.data import load_split
from code.experiments.local_llm import OllamaClient
from code.monitoring.monitor import LiveMonitor
from code.registry.registry import Registry
from code.service.confidence import ConfidenceModel
from code.tracing.agent_tracer import AgentTracer

ALIAS = os.getenv("AGENTEVALOPS_ALIAS", "production")
LOG_DIR = Path(os.getenv("AGENTEVALOPS_LOG_DIR", str(REPO_ROOT / "logs")))
REPORTED = REPO_ROOT / "supporting_files" / "a2" / "reported_values.json"
DATA_TOOLS = ("retrieve", "query_records", "read_file")
FAULT_PROFILES: dict[str, list[FaultSpec]] = {
    "none": [],
    "transient_failure": [FaultSpec("tool_failure", t, "first_call") for t in DATA_TOOLS],
    "information_outage": [FaultSpec("tool_failure", t, "always") for t in DATA_TOOLS],
    "calculator_outage": [FaultSpec("tool_failure", "calculate", "always")],
    "slow_response": [FaultSpec("slow_response", t, "always", 2000) for t in DATA_TOOLS],
    "format_change": [FaultSpec("format_change", t, "always") for t in DATA_TOOLS],
}
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")


def git_commit() -> str:
    """Commit the image was built from (/app/GIT_COMMIT), else the checkout's HEAD."""
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    baked = REPO_ROOT / "GIT_COMMIT"
    if baked.exists():
        return baked.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mask(text: str | None) -> str | None:
    if text is None:
        return None
    return _PHONE.sub("<phone>", _EMAIL.sub("<email>", text))


class PredictRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000, description="Task text")
    fault: str = Field("none", description=f"Fault profile to inject (drift demo): {list(FAULT_PROFILES)}")
    model: str | None = Field(None, description="Override the model (drift demo: model swap)")


class AgentService:
    """Holds the deployed configuration and everything a request needs."""

    def __init__(self) -> None:
        self.registry = Registry(mlflow_sync=False)
        self.version, self.cfg = self.registry.resolve(ALIAS)
        self.client = OllamaClient()
        self.monitor = LiveMonitor()
        self.confidence = ConfidenceModel()
        self.tracer = AgentTracer(enabled=os.getenv("AGENTEVALOPS_TRACE", "1") == "1")
        self.known = {}
        for split in ("train", "validation", "test", "gold"):
            for item in load_split(split):
                self.known[item.task_id] = item.task
        self.lock = threading.Lock()
        self.commit = git_commit()
        self.last_validation: dict[str, Any] | None = None
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def version_info(self) -> dict[str, Any]:
        info = {"git_commit": self.commit, "registry_alias": ALIAS, "registry_version": self.version,
                "config_name": self.cfg.name, "config_hash": self.cfg.config_hash(), "model": self.cfg.model,
                "prompt_version": self.cfg.prompt_version, "tool_desc_version": self.cfg.tool_desc_version,
                "image_built": os.getenv("IMAGE_BUILD_TIME", "")}
        try:
            info["model_digest"] = self.client.model_digest(self.cfg.model)
            info["ollama_version"] = self.client.server_version()
        except Exception as exc:  # model server not up yet
            info["model_server_error"] = str(exc)
        return info

    def predict(self, req: PredictRequest) -> dict[str, Any]:
        if req.fault not in FAULT_PROFILES:
            raise HTTPException(400, f"unknown fault profile {req.fault!r}")
        task = adhoc_task(req.task, self.known)
        cfg = self.cfg if not req.model else self.cfg.model_copy(update={"model": req.model})
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            with self.lock:
                with FaultInjector(FAULT_PROFILES[req.fault]):
                    run = run_agent(task, cfg, client=self.client, tracer=self.tracer)
        except Exception as exc:  # report the cause as JSON instead of a bare HTTP 500
            logging.getLogger(__name__).exception("agent run failed for request %s", request_id)
            raise HTTPException(500, f"agent run failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        summary = run.summary()
        graded = is_graded(task)
        alerts = self.monitor.update(summary, graded=graded)
        out = {
            "request_id": request_id, "answer": run.final_answer, "trace_id": run.trace_id, "run_id": run.run_id,
            "confidence": self.confidence.predict(summary), "latency_ms": latency_ms,
            "graded": graded, "correct": run.correct if graded else None,
            "gold_answer": task.gold_answer if graded else None, "task_id": task.task_id if graded else None,
            "steps": [{"index": s.step_index, "action": s.action_type.value, "tool": s.tool_name,
                       "arguments": s.arguments,
                       "output": (s.tool_result.output if s.tool_result and s.tool_result.success else None),
                       "error": s.error, "latency_ms": s.latency_ms} for s in run.trajectory.steps],
            "tool_calls": [{"tool": c.tool, "arguments": c.arguments, "success": c.success, "output": c.output,
                            "error": c.error, "attempts": c.attempts, "blocked_repeat": c.blocked_repeat}
                           for c in run.tool_calls],
            "metrics": {k: summary[k] for k in ("steps", "llm_calls", "total_tokens", "tool_errors",
                                                "malformed_calls", "loop_extended", "retries_used")},
            "fault": req.fault, "model": cfg.model, "version": self.version, "git_commit": self.commit,
            "alerts": alerts,
        }
        self._log(task.prompt, out)
        return out

    def _log(self, prompt: str, out: dict[str, Any]) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": out["request_id"],
                 "task_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "task_chars": len(prompt),
                 "bundled_task_id": out["task_id"], "answer": mask(out["answer"]), "confidence": out["confidence"],
                 "latency_ms": out["latency_ms"], "version": out["version"], "model": out["model"],
                 "git_commit": out["git_commit"], "fault": out["fault"], "trace_id": out["trace_id"],
                 "correct": out["correct"], **out["metrics"]}
        with (LOG_DIR / "requests.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry) + "\n")

    def reported(self) -> dict[str, Any]:
        return json.loads(REPORTED.read_text(encoding="utf-8")) if REPORTED.exists() else {}

    def run_validation(self) -> dict[str, Any]:
        """Run the documented fixed subset of the gold set and compare with the reported values."""
        from code.evaluation.gold import verify_gold

        verify_gold()
        reported = self.reported().get("ui_validation_subset", {})
        ids = reported.get("task_ids") or [i.task_id for i in sorted(load_split("gold"), key=lambda i: i.task_id)][:5]
        gold = {i.task_id: i for i in load_split("gold")}
        rows = []
        t0 = time.perf_counter()
        for tid in ids:
            with self.lock:
                run = run_agent(gold[tid].task, self.cfg, client=self.client, tracer=self.tracer)
            rows.append({"task_id": tid, "answer": run.final_answer, "correct": run.correct,
                         "reported_correct": reported.get("per_task", {}).get(tid)})
        live = sum(r["correct"] for r in rows) / len(rows)
        self.last_validation = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(rows),
                                "live_success": round(live, 3), "reported_success": reported.get("success"),
                                "matches_reported": all(r["correct"] == r["reported_correct"] for r in rows
                                                        if r["reported_correct"] is not None),
                                "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}
        return self.last_validation


app = FastAPI(title="AgentEvalOps", description="Tool-using agent with MLOps instrumentation (CS203 A2)")
_service: AgentService | None = None


def service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    return service().predict(req)


@app.get("/health")
def health() -> dict[str, Any]:
    svc = service()
    try:
        ok = svc.client.model_digest(svc.cfg.model) is not None
        return {"status": "ok" if ok else "degraded", "model_server": "up", "model_installed": ok}
    except Exception as exc:
        return {"status": "degraded", "model_server": "down", "error": str(exc)}


@app.get("/version")
def version() -> dict[str, Any]:
    return service().version_info()


@app.get("/monitoring")
def monitoring() -> dict[str, Any]:
    return service().monitor.snapshot()


@app.post("/monitoring/reset")
def monitoring_reset() -> dict[str, Any]:
    service().monitor.reset()
    return {"status": "reset"}


@app.get("/validation")
def validation() -> dict[str, Any]:
    svc = service()
    return {"reported": svc.reported(), "last_live_run": svc.last_validation}


@app.post("/validation/run")
def validation_run() -> dict[str, Any]:
    return service().run_validation()


if os.getenv("AGENTEVALOPS_UI", "1") == "1":
    import gradio as gr

    from code.service.ui import build_ui, ui_theme

    app = gr.mount_gradio_app(app, build_ui(service), path="/ui", theme=ui_theme())


@app.get("/")
def root() -> dict[str, str]:
    return {"ui": "/ui", "docs": "/docs", "health": "/health", "version": "/version"}
