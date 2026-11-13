"""Versioned agent configurations for Assignment 2.

Every system the A2 report compares (the baselines and the proposed agent) is a
YAML file in ``configs/``. A configuration names its strategy, model, prompt
version, tool-description version, sampling and step settings, and the
proposed-system features. Prompts and tool descriptions are separate versioned
files under ``prompts/``, so a configuration only refers to them by version.

The configuration hash is the SHA-256 of the canonical JSON of all fields plus
the SHA-256 of every prompt, tool-description and exemplar file the
configuration uses, so editing a prompt file changes the hash even when the
YAML does not change.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"


class AgentConfig(BaseModel):
    """One agent configuration (a baseline or the proposed system)."""

    name: str = Field(..., description="Short identifier, e.g. 'proposed'.")
    description: str = ""
    strategy: Literal["abstain", "direct", "tool_loop"] = Field(
        ..., description="abstain: trivial baseline; direct: one LLM call without tools; tool_loop: function calling."
    )
    model: str | None = Field(None, description="Local Ollama model tag (None for 'abstain').")
    prompt_version: str = Field("v1", description="prompts/agent/system_<v>.txt (direct: direct_<v>.txt).")
    tool_desc_version: str = Field("v1", description="prompts/tools/<v>.json.")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_steps: int = Field(5, ge=1, le=20, description="Maximum LLM turns.")
    max_tokens: int = Field(512, ge=16, le=4096, description="Generation cap per LLM turn.")
    retry_budget: int = Field(0, ge=0, le=5, description="Error-feedback retries allowed after failed tool calls.")
    error_feedback: bool = Field(False, description="Append a corrective hint to failed tool results.")
    loop_guard: bool = Field(False, description="Block repeated identical tool calls and tell the model.")
    step_guard: bool = Field(False, description="Warn before the last turn and force a submission at the end.")
    single_tool_call: bool = Field(False, description="Run only the first tool call of a turn (no planning with placeholders).")
    exemplars: str = Field("none", description="prompts/exemplars/<name>.json, or 'none'.")

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    @model_validator(mode="after")
    def _check(self) -> "AgentConfig":
        if self.strategy != "abstain" and not self.model:
            raise ValueError(f"strategy {self.strategy!r} needs a model")
        return self

    # -- referenced files ------------------------------------------------------

    def prompt_path(self) -> Path | None:
        if self.strategy == "abstain":
            return None
        stem = "direct" if self.strategy == "direct" else "system"
        return PROMPTS_DIR / "agent" / f"{stem}_{self.prompt_version}.txt"

    def tools_path(self) -> Path | None:
        return PROMPTS_DIR / "tools" / f"{self.tool_desc_version}.json" if self.strategy == "tool_loop" else None

    def exemplars_path(self) -> Path | None:
        if self.strategy != "tool_loop" or self.exemplars == "none":
            return None
        return PROMPTS_DIR / "exemplars" / f"{self.exemplars}.json"

    def referenced_files(self) -> list[Path]:
        return [p for p in (self.prompt_path(), self.tools_path(), self.exemplars_path()) if p is not None]

    # -- identity ----------------------------------------------------------------

    def config_hash(self) -> str:
        files = {p.relative_to(REPO_ROOT).as_posix(): _file_sha256(p) for p in self.referenced_files()}
        payload = json.dumps({"config": self.model_dump(), "files": files}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: str | Path) -> AgentConfig:
    """Load a configuration YAML and check that every file it refers to exists."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = AgentConfig.model_validate(data)
    missing = [str(p) for p in cfg.referenced_files() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{path}: missing referenced files {missing}")
    return cfg


@lru_cache(maxsize=None)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def system_prompt(cfg: AgentConfig) -> str:
    path = cfg.prompt_path()
    return _read_text(path) if path else ""


def tool_schemas(cfg: AgentConfig) -> list[dict[str, Any]]:
    path = cfg.tools_path()
    return json.loads(_read_text(path)) if path else []


def exemplars(cfg: AgentConfig) -> list[dict[str, Any]]:
    path = cfg.exemplars_path()
    return json.loads(_read_text(path)) if path else []
