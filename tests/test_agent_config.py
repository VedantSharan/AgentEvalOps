"""Tests for code/agent/config.py and the configs/ directory."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from code.agent.config import CONFIGS_DIR, AgentConfig, exemplars, load_config, system_prompt, tool_schemas
from code.experiments.providers import AGENT_TOOLS, SYSTEM_INSTRUCTION

SYSTEMS = ["b0_abstain", "b1_direct", "b2_original", "proposed"]


@pytest.mark.parametrize("name", SYSTEMS)
def test_every_system_config_loads(name: str) -> None:
    cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
    assert cfg.name == name
    assert len(cfg.config_hash()) == 64


def test_hashes_are_distinct_and_stable() -> None:
    hashes = [load_config(CONFIGS_DIR / f"{n}.yaml").config_hash() for n in SYSTEMS]
    assert len(set(hashes)) == len(hashes)
    assert hashes == [load_config(CONFIGS_DIR / f"{n}.yaml").config_hash() for n in SYSTEMS]


def test_original_loop_uses_the_a1_prompt_and_tools_verbatim() -> None:
    cfg = load_config(CONFIGS_DIR / "b2_original.yaml")
    assert system_prompt(cfg) == SYSTEM_INSTRUCTION
    assert tool_schemas(cfg) == AGENT_TOOLS
    assert exemplars(cfg) == []


def test_v2_tools_cover_the_same_tools_as_v1() -> None:
    v2 = tool_schemas(load_config(CONFIGS_DIR / "proposed.yaml"))
    assert [t["function"]["name"] for t in v2] == [t["function"]["name"] for t in AGENT_TOOLS]


def test_model_required_unless_abstaining() -> None:
    AgentConfig(name="x", strategy="abstain")
    with pytest.raises(ValidationError):
        AgentConfig(name="x", strategy="tool_loop")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(name="x", strategy="abstain", typo_field=1)


def test_missing_prompt_file_is_reported(tmp_path: Path) -> None:
    cfg = yaml.safe_load((CONFIGS_DIR / "proposed.yaml").read_text(encoding="utf-8"))
    cfg["prompt_version"] = "v999"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(path)


def test_gate_is_fixed_in_advance() -> None:
    gate = yaml.safe_load((CONFIGS_DIR / "gate.yaml").read_text(encoding="utf-8"))
    assert gate["gold_tolerance"] == 0.04 and gate["min_shifted_gain"] > 0
    assert gate["retrain_pool_parent_split"] == "train" and gate["shifted_eval_parent_split"] == "test"
