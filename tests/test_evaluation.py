"""Tests for code/evaluation/* and code/registry/registry.py (no Ollama needed)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from code.agent.config import CONFIGS_DIR, load_config
from code.evaluation import gold
from code.evaluation.data import load_split
from code.evaluation.evaluate import _flat_metrics, evaluate, rows_for
from code.evaluation.metrics import bootstrap_ci, paired_comparison, pass_at_k, summarise
from code.experiments.providers import LLMResponse
from code.registry.registry import Registry


def rows(spec: dict[str, list[int]]) -> list[dict[str, Any]]:
    out = []
    for task, results in spec.items():
        for i, c in enumerate(results):
            out.append({"task_id": task, "seed": 203 + i, "correct": bool(c), "steps": 2, "total_tokens": 10,
                        "wall_ms": 100.0, "slices": {"family": task[:2]}})
    return out


def test_bootstrap_is_task_clustered_and_seeded() -> None:
    data = rows({"a": [1, 1], "b": [0, 0], "c": [1, 0]})
    mean, lo, hi = bootstrap_ci(data)
    assert mean == pytest.approx(0.5)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert bootstrap_ci(data) == (mean, lo, hi)


def test_pass_k_needs_every_repeat() -> None:
    assert pass_at_k(rows({"a": [1, 1], "b": [1, 0]})) == (0.5, 2)


def test_summary_has_ci_and_costs() -> None:
    s = summarise(rows({"a": [1], "b": [0]}))
    assert s["success"] == 0.5 and s["n_tasks"] == 2 and "p95_wall_ms" in s


def test_paired_comparison_counts_discordant_pairs() -> None:
    a = rows({"t1": [0], "t2": [0], "t3": [1], "t4": [0]})
    b = rows({"t1": [1], "t2": [1], "t3": [1], "t4": [1]})
    res = paired_comparison(a, b)
    assert res["b_only_correct"] == 3 and res["a_only_correct"] == 0
    assert res["mcnemar_exact_p"] == pytest.approx(0.25)
    assert res["diff_mean"] == pytest.approx(0.75)


def test_split_sizes_match_the_a1_manifests() -> None:
    sizes = {s: len(load_split(s)) for s in ("train", "validation", "test", "test_base", "test_augmented",
                                             "gold", "shifted", "shifted_test")}
    assert sizes == {"train": 45, "validation": 13, "test": 25, "test_base": 10, "test_augmented": 15,
                     "gold": 25, "shifted": 96, "shifted_test": 20}
    assert all(i.faults for i in load_split("shifted_validation"))


def test_gold_checksum_verified_and_tampering_detected(tmp_path: Path) -> None:
    gold.verify_gold()
    shutil.copytree(gold.GOLD_DIR, tmp_path / "g")
    (tmp_path / "g" / "gold_labels.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(gold.GoldSetModified):
        gold.verify_gold(tmp_path / "g")


def test_gold_log_appends_a_row(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    log.write_text("| header |\n", encoding="utf-8")
    gold.log_gold_evaluation("v5", "a" * 64, "qwen2.5:3b", "final", 0.8, "run123", log_path=log)
    assert "| v5 | `aaaaaaaaaaaa` | qwen2.5:3b | final | 0.800 | `run123` |" in log.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        gold.log_gold_evaluation("v5", "a" * 64, None, " ", 0.8, "r", log_path=log)


def test_registry_register_promote_rollback(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry", mlflow_sync=False)
    v1 = reg.register(load_config(CONFIGS_DIR / "b2_original.yaml"), note="baseline", source="configs/b2_original.yaml")
    v2 = reg.register(load_config(CONFIGS_DIR / "proposed.yaml"), note="proposed", source="configs/proposed.yaml")
    assert (v1, v2) == ("v1", "v2")
    assert reg.register(load_config(CONFIGS_DIR / "proposed.yaml"), note="again", source="x") == "v2"
    reg.promote("v1", "production", reason="first")
    reg.promote("v2", "production", reason="better")
    assert reg.resolve("production")[0] == "v2"
    assert reg.rollback("production", reason="regression") == "v1"
    assert reg.resolve("production")[1].name == "b2_original"
    assert [e["event"] for e in reg.history()] == ["register", "register", "promote", "promote", "rollback"]
    proposed = load_config(CONFIGS_DIR / "proposed.yaml")
    assert (tmp_path / "registry" / "prompts" / "v2" / "agent" / f"system_{proposed.prompt_version}.txt").exists()


class AbstainingClient:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def call_chat_completion(self, model_id: str, messages: Any, tools: Any = None, temperature: float = 0.0,
                             timeout: float = 0.0, seed: int | None = None, max_tokens: int | None = None) -> LLMResponse:
        self.calls += 1
        call = {"id": "c", "type": "function", "function": {"name": "submit_answer",
                                                           "arguments": json.dumps({"answer": "unanswerable"})}}
        return LLMResponse(content=None, tool_calls=[call], prompt_tokens=10, completion_tokens=2,
                           resolved_model=model_id, latency_ms=5.0, raw_response={})


def test_evaluate_caches_runs_and_reports_slices(tmp_path: Path) -> None:
    client = AbstainingClient()
    kwargs = dict(split="test_base", trace=False, out_dir=tmp_path / "out", runs_dir=tmp_path / "runs",
                  client=client, registry=Registry(tmp_path / "reg", mlflow_sync=False), log_mlflow=False,
                  progress=False)
    res = evaluate(str(CONFIGS_DIR / "b2_original.yaml"), **kwargs)
    assert res["n_items"] == 10 and res["new_runs"] == 10 and client.calls == 10
    # The 5 unanswerable test tasks are solved by abstaining, the 5 audit tasks are not.
    assert res["metrics"]["success"] == pytest.approx(0.5)
    assert res["slices"]["answerability"]["unanswerable"]["success"] == 1.0
    again = evaluate(str(CONFIGS_DIR / "b2_original.yaml"), **kwargs)
    assert again["new_runs"] == 0 and client.calls == 10
    assert (tmp_path / "out" / "b2_original__test_base.json").exists()


def test_gold_evaluation_requires_a_reason(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        evaluate(str(CONFIGS_DIR / "b0_abstain.yaml"), "gold", reason="", log_mlflow=False, trace=False,
                 registry=Registry(tmp_path / "reg", mlflow_sync=False), out_dir=tmp_path, runs_dir=tmp_path)


def test_mlflow_metric_names_are_sanitised() -> None:
    flat = _flat_metrics({"metrics": {"pass^1": 0.5, "success": 0.5, "k": 1},
                          "slices": {"fault": {"none": {"success": 1.0}}}})
    assert "pass_1" in flat and "slice.fault.none.success" in flat


def test_rows_for_refuses_gold() -> None:
    with pytest.raises(ValueError):
        rows_for(str(CONFIGS_DIR / "b0_abstain.yaml"), "gold")
