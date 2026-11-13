"""Configuration registry with versioned staging/production aliases (Part B, D, E).

In this project the "model" that gets registered, promoted and rolled back is
an agent *configuration* (model tag + prompt + tool descriptions + settings),
not a set of weights. The registry has two synchronised stores:

* ``registry/`` in the repository, the bundled registry that the Docker image
  reads offline:
    registry/versions/v<N>.yaml   the AgentConfig plus registration metadata
    registry/prompts/v<N>/...     a frozen copy of every prompt file it uses
    registry/aliases.json         {"production": "v5", "staging": "v6", ...}
    registry/history.jsonl        every register / promote / rollback event
* the MLflow Model Registry (registered model ``agentevalops-agent``): each
  version is an MLflow pyfunc model whose artifact is the same YAML, with the
  same aliases, so every registered version links to MLflow runs.

``resolve(ref)`` accepts an alias ("production"), a version ("v3") or a path to
a configs/*.yaml file, and returns (label, AgentConfig).

Commands:
    python -m code.registry.registry list
    python -m code.registry.registry register configs/proposed.yaml --note "..."
    python -m code.registry.registry promote v5 production --reason "..."
    python -m code.registry.registry rollback production --reason "..."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from code.agent.config import PROMPTS_DIR, REPO_ROOT, AgentConfig, load_config

REGISTRY_DIR = Path(os.getenv("AGENTEVALOPS_REGISTRY", str(REPO_ROOT / "registry")))
MLFLOW_MODEL_NAME = "agentevalops-agent"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Registry:
    def __init__(self, root: Path = REGISTRY_DIR, mlflow_sync: bool = True) -> None:
        self.root = Path(root)
        self.mlflow_sync = mlflow_sync

    # -- storage ---------------------------------------------------------------

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    @property
    def aliases_path(self) -> Path:
        return self.root / "aliases.json"

    @property
    def history_path(self) -> Path:
        return self.root / "history.jsonl"

    def aliases(self) -> dict[str, str]:
        if not self.aliases_path.exists():
            return {}
        return json.loads(self.aliases_path.read_text(encoding="utf-8"))

    def _write_aliases(self, aliases: dict[str, str]) -> None:
        self.aliases_path.write_text(json.dumps(dict(sorted(aliases.items())), indent=2) + "\n",
                                     encoding="utf-8", newline="\n")

    def _log(self, event: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"time": _now(), **event}) + "\n")

    def history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line]

    def versions(self) -> list[str]:
        if not self.versions_dir.exists():
            return []
        return sorted((p.stem for p in self.versions_dir.glob("v*.yaml")), key=lambda v: int(v[1:]))

    def entry(self, version: str) -> dict[str, Any]:
        return yaml.safe_load((self.versions_dir / f"{version}.yaml").read_text(encoding="utf-8"))

    # -- operations ------------------------------------------------------------

    def register(self, cfg: AgentConfig, note: str, source: str, metrics: dict[str, Any] | None = None,
                 run_id: str | None = None) -> str:
        """Store ``cfg`` as the next version (or return the existing version with the same hash)."""
        h = cfg.config_hash()
        for v in self.versions():
            if self.entry(v)["config_hash"] == h:
                return v
        version = f"v{len(self.versions()) + 1}"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        frozen = self.root / "prompts" / version
        for path in cfg.referenced_files():
            dest = frozen / path.relative_to(PROMPTS_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        entry = {"version": version, "registered_at": _now(), "config_hash": h, "note": note, "source": source,
                 "mlflow_run_id": run_id, "metrics": metrics or {}, "config": cfg.model_dump()}
        (self.versions_dir / f"{version}.yaml").write_text(yaml.safe_dump(entry, sort_keys=False), encoding="utf-8",
                                                          newline="\n")
        mlflow_version = self._mlflow_register(version, entry)
        if mlflow_version:
            entry["mlflow_model_version"] = mlflow_version
            (self.versions_dir / f"{version}.yaml").write_text(yaml.safe_dump(entry, sort_keys=False),
                                                              encoding="utf-8", newline="\n")
        self._log({"event": "register", "version": version, "config_name": cfg.name, "config_hash": h,
                   "note": note, "mlflow_model_version": mlflow_version})
        return version

    def promote(self, version: str, alias: str, reason: str) -> None:
        """Point ``alias`` at ``version`` (a version such as "v4", or another alias such as "staging")."""
        version = self.aliases().get(version, version)
        if version not in self.versions():
            raise KeyError(f"unknown version {version}")
        aliases = self.aliases()
        previous = aliases.get(alias)
        aliases[alias] = version
        self._write_aliases(aliases)
        self._mlflow_alias(alias, version)
        self._log({"event": "promote", "alias": alias, "version": version, "previous": previous, "reason": reason})

    def rollback(self, alias: str, reason: str) -> str:
        """Point ``alias`` back to the version it had before its latest promotion."""
        events = [e for e in self.history() if e.get("alias") == alias and e["event"] in ("promote", "rollback")]
        if not events:
            raise RuntimeError(f"alias {alias} has never been promoted")
        current = self.aliases().get(alias)
        target = None
        # Walk back through the alias's history to the version it held before `current`.
        for e in reversed(events):
            if e["version"] == current and e.get("previous"):
                target = e["previous"]
                break
        if target is None:
            raise RuntimeError(f"no earlier version to roll {alias} back to")
        aliases = self.aliases()
        aliases[alias] = target
        self._write_aliases(aliases)
        self._mlflow_alias(alias, target)
        self._log({"event": "rollback", "alias": alias, "version": target, "previous": current, "reason": reason})
        return target

    def resolve(self, ref: str) -> tuple[str, AgentConfig]:
        """Alias, version or YAML path -> (label, AgentConfig)."""
        if ref.endswith((".yaml", ".yml")):
            cfg = load_config(ref)
            return cfg.name, cfg
        version = self.aliases().get(ref, ref)
        if version not in self.versions():
            raise KeyError(f"{ref!r} is neither an alias nor a registered version")
        cfg = AgentConfig.model_validate(self.entry(version)["config"])
        frozen = self.root / "prompts" / version
        for path in cfg.referenced_files():
            copy = frozen / path.relative_to(PROMPTS_DIR)
            if copy.exists() and copy.read_bytes() != path.read_bytes():
                raise RuntimeError(f"{path} differs from the copy frozen with {version}; register a new version")
        return version, cfg

    # -- MLflow model registry mirror --------------------------------------------

    def _mlflow_register(self, version: str, entry: dict[str, Any]) -> str | None:
        if not self.mlflow_sync:
            return None
        import mlflow
        from mlflow.pyfunc import PythonModel

        class AgentConfigModel(PythonModel):
            """Pyfunc wrapper: predict() runs the registered configuration on task prompts."""

            def load_context(self, context: Any) -> None:
                from code.agent.config import AgentConfig as _Cfg

                self.cfg = _Cfg.model_validate(yaml.safe_load(Path(context.artifacts["config"]).read_text())["config"])

            def predict(self, context: Any, model_input: Any, params: Any = None) -> list[str | None]:
                from code.agent.loop import run_agent
                from code.agent.adhoc import adhoc_task

                return [run_agent(adhoc_task(str(p)), self.cfg).final_answer for p in list(model_input["prompt"])]

        path = self.versions_dir / f"{version}.yaml"
        mlflow.set_experiment("AgentEvalOps-A2-registry")
        with mlflow.start_run(run_name=f"register-{version}") as run:
            mlflow.log_params({k: v for k, v in entry["config"].items() if k != "description"})
            mlflow.set_tags({"registry_version": version, "config_hash": entry["config_hash"],
                             "source": entry["source"], "source_run_id": entry.get("mlflow_run_id") or ""})
            info = mlflow.pyfunc.log_model(name="agent_config", python_model=AgentConfigModel(),
                                           artifacts={"config": str(path)}, pip_requirements=[])
            mv = mlflow.register_model(info.model_uri, MLFLOW_MODEL_NAME, tags={"registry_version": version})
            del run
        return str(mv.version)

    def _mlflow_alias(self, alias: str, version: str) -> None:
        if not self.mlflow_sync:
            return
        from mlflow import MlflowClient

        mlflow_version = self.entry(version).get("mlflow_model_version")
        if mlflow_version:
            MlflowClient().set_registered_model_alias(MLFLOW_MODEL_NAME, alias, mlflow_version)


def main(argv: list[str] | None = None) -> None:
    from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description="Agent configuration registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("config")
    r.add_argument("--note", default="")
    r.add_argument("--run-id", default=None)
    pr = sub.add_parser("promote")
    pr.add_argument("version")
    pr.add_argument("alias")
    pr.add_argument("--reason", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("alias")
    rb.add_argument("--reason", required=True)
    args = p.parse_args(argv)
    reg = Registry()
    if args.cmd == "list":
        aliases = reg.aliases()
        for v in reg.versions():
            e = reg.entry(v)
            tags = [a for a, t in aliases.items() if t == v]
            print(f"{v:4s} {e['config']['name']:24s} {e['config_hash'][:12]}  {','.join(tags):20s} {e['note']}")
    elif args.cmd == "register":
        print(reg.register(load_config(args.config), note=args.note, source=args.config, run_id=args.run_id))
    elif args.cmd == "promote":
        reg.promote(args.version, args.alias, args.reason)
        print(f"{args.alias} -> {reg.aliases()[args.alias]}")
    elif args.cmd == "rollback":
        print(f"{args.alias} -> {reg.rollback(args.alias, args.reason)}")


if __name__ == "__main__":
    main()
