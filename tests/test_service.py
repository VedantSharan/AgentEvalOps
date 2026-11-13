"""Tests for the REST service (code/service/app.py) using the model-free B0 version."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from code.evaluation.data import load_split
from code.monitoring.replay import select_thresholds


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    rows = [{"task_id": f"t{i}", "tool_errors": i % 2, "retries_used": 0, "malformed_calls": 0,
             "loop_extended": False, "steps": 3 + i % 2, "wall_ms": 1000.0 + 50 * i, "total_tokens": 100 + i,
             "correct": True, "slices": {"fault": "none"}} for i in range(12)]
    shifted = [dict(r, tool_errors=4, wall_ms=9000.0) for r in rows]
    thresholds, _ = select_thresholds(rows, shifted)
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps(thresholds), encoding="utf-8")
    monkeypatch.setenv("AGENTEVALOPS_ALIAS", "b0")
    monkeypatch.setenv("AGENTEVALOPS_TRACE", "0")
    monkeypatch.setenv("AGENTEVALOPS_UI", "0")
    monkeypatch.setenv("AGENTEVALOPS_LOG_DIR", str(tmp_path / "logs"))
    import code.monitoring.monitor as monitor
    import code.service.app as app_module

    monkeypatch.setattr(monitor, "THRESHOLDS", thr)
    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module.LiveMonitor.__init__, "__defaults__", (thr, monitor.WINDOW))
    return TestClient(app_module.app)


def test_predict_returns_answer_trace_and_steps(client: TestClient, tmp_path: Path) -> None:
    task = next(i.task for i in load_split("test") if not i.task.answerable)
    res = client.post("/predict", json={"task": task.prompt}).json()
    assert res["answer"] == "unanswerable" and res["graded"] and res["correct"]
    assert res["trace_id"] and res["steps"][0]["action"] == "submit_answer" and res["version"] == "v1"
    log = (tmp_path / "logs" / "requests.jsonl").read_text(encoding="utf-8")
    assert task.prompt not in log and json.loads(log.splitlines()[0])["task_sha256"]


def test_free_text_task_is_not_graded(client: TestClient) -> None:
    res = client.post("/predict", json={"task": "Contact me at someone@example.com please"}).json()
    assert res["graded"] is False and res["correct"] is None


def test_version_health_and_monitoring_endpoints(client: TestClient) -> None:
    v = client.get("/version").json()
    assert v["registry_version"] == "v1" and v["git_commit"] and len(v["config_hash"]) == 64
    assert "status" in client.get("/health").json()
    client.post("/predict", json={"task": "anything"})
    snap = client.get("/monitoring").json()
    assert snap["requests"] == 1 and set(snap["monitored_signals"]) <= set(snap["rolling"])
    assert client.post("/monitoring/reset").json() == {"status": "reset"}


def test_unknown_fault_profile_is_rejected(client: TestClient) -> None:
    assert client.post("/predict", json={"task": "x", "fault": "meteor"}).status_code == 400
