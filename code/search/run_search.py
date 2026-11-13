"""Part B: systematic configuration search on the validation split.

    python -m code.search.run_search            # uses configs/search_space.yaml

Two Optuna samplers get the same trial budget over the same space:
  * random search (RandomSampler), the control;
  * NSGA-II (NSGAIISampler), multi-objective: maximise success, minimise mean
    tokens and mean latency, with the token/latency budget as a constraint.

Each trial is a full evaluation of one configuration on the validation split
through code/evaluation/evaluate.py, so it is an MLflow run with the configuration,
metrics, data version and git commit, and its agent runs are cached by
configuration hash (a configuration sampled twice is not re-run).

Outputs in supporting_files/a2/search/: trials.csv, summary.json (best
configuration, Pareto front, sampler comparison) and plot data; plots are made by
code/search/plots.py. The selected configuration is written to
configs/selected.yaml and registered with the staging alias.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import optuna
import yaml

from code.agent.config import REPO_ROOT, AgentConfig, load_config
from code.evaluation.evaluate import evaluate
from code.registry.registry import Registry
from code.tracing.mlflow_tracer import DEFAULT_TRACKING_URI

SPACE_FILE = REPO_ROOT / "configs" / "search_space.yaml"
OUT_DIR = REPO_ROOT / "supporting_files" / "a2" / "search"
SELECTED = REPO_ROOT / "configs" / "selected.yaml"
CONFIG_DIR = REPO_ROOT / "configs" / "search"


def objective_j(m: dict[str, Any], w: dict[str, float]) -> float:
    error_rate = max(m["rate_malformed_any"], m["rate_loop_extended"])
    return (w["alpha"] * m["success"] - w["beta"] * m["mean_total_tokens"] / 1000
            - w["gamma"] * m["mean_wall_ms"] / 1000 - w["delta"] * error_rate)


def trial_config(base: AgentConfig, params: dict[str, Any], name: str) -> AgentConfig:
    return base.model_copy(update={**params, "name": name,
                                   "description": f"Part B search trial {name}: {json.dumps(params, sort_keys=True)}"})


class Search:
    def __init__(self, space_file: Path = SPACE_FILE, out_dir: Path = OUT_DIR) -> None:
        self.spec = yaml.safe_load(space_file.read_text(encoding="utf-8"))
        self.base = load_config(REPO_ROOT / self.spec["base_config"])
        self.out_dir = out_dir
        self.records: list[dict[str, Any]] = []

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        return {k: trial.suggest_categorical(k, v) for k, v in self.spec["space"].items()}

    def run_trial(self, sampler_name: str, trial: optuna.Trial) -> dict[str, Any]:
        params = self.suggest(trial)
        name = f"search_{sampler_name}_{trial.number:02d}"
        cfg = trial_config(self.base, params, name)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIG_DIR / f"{name}.yaml"
        path.write_text(f"# {cfg.description}\n" + cfg.to_yaml(), encoding="utf-8", newline="\n")
        res = evaluate(str(path), self.spec["split"], repeats=self.spec["repeats"], label=name,
                       out_dir=self.out_dir / "trials")
        m = res["metrics"]
        budget = self.spec["budget"]
        violation = max(m["mean_total_tokens"] - budget["max_mean_total_tokens"], 0) / budget["max_mean_total_tokens"] \
            + max(m["p95_wall_ms"] / 1000 - budget["max_p95_wall_s"], 0) / budget["max_p95_wall_s"]
        rec = {"sampler": sampler_name, "trial": trial.number, "name": name, **params,
               "config_hash": res["config_hash"], "success": m["success"], "ci_low": m["success_ci_low"],
               "ci_high": m["success_ci_high"], "mean_total_tokens": m["mean_total_tokens"],
               "mean_wall_s": round(m["mean_wall_ms"] / 1000, 2), "p95_wall_s": round(m["p95_wall_ms"] / 1000, 2),
               "mean_steps": m["mean_steps"], "error_rate": max(m["rate_malformed_any"], m["rate_loop_extended"]),
               "J": round(objective_j(m, self.spec["objective_weights"]), 4), "feasible": violation == 0,
               "new_runs": res["new_runs"], "mlflow_run_id": res.get("mlflow_run_id"),
               "config_path": path.relative_to(REPO_ROOT).as_posix()}
        trial.set_user_attr("record", rec)
        trial.set_user_attr("constraint", [violation])
        self.records.append(rec)
        print(f"[search] {name} success={rec['success']:.3f} tokens={rec['mean_total_tokens']:.0f} "
              f"wall={rec['mean_wall_s']}s feasible={rec['feasible']} params={params}", flush=True)
        return rec

    def add_seed_configs(self) -> None:
        """Evaluate the hand-designed Part A configurations as extra candidates ("manual").

        They are scored on the same validation split, so the search can only replace
        them with something better on validation, never with something worse.
        """
        budget = self.spec["budget"]
        for i, rel in enumerate(self.spec.get("seed_configs", [])):
            cfg = load_config(REPO_ROOT / rel)
            if any(r["config_hash"] == cfg.config_hash() for r in self.records):
                continue
            res = evaluate(str(REPO_ROOT / rel), self.spec["split"], repeats=self.spec["repeats"], label=cfg.name,
                           out_dir=self.out_dir / "trials")
            m = res["metrics"]
            feasible = (m["mean_total_tokens"] <= budget["max_mean_total_tokens"]
                        and m["p95_wall_ms"] / 1000 <= budget["max_p95_wall_s"])
            self.records.append({
                "sampler": "manual", "trial": i, "name": cfg.name,
                **{k: getattr(cfg, k) for k in self.spec["space"]}, "config_hash": res["config_hash"],
                "success": m["success"], "ci_low": m["success_ci_low"], "ci_high": m["success_ci_high"],
                "mean_total_tokens": m["mean_total_tokens"], "mean_wall_s": round(m["mean_wall_ms"] / 1000, 2),
                "p95_wall_s": round(m["p95_wall_ms"] / 1000, 2), "mean_steps": m["mean_steps"],
                "error_rate": max(m["rate_malformed_any"], m["rate_loop_extended"]),
                "J": round(objective_j(m, self.spec["objective_weights"]), 4), "feasible": feasible,
                "new_runs": res["new_runs"], "mlflow_run_id": res.get("mlflow_run_id"), "config_path": rel})

    def refine(self) -> None:
        """Stage 2: grid over spec['refine']['grid'] with spec['refine']['fixed'] (validation only)."""
        spec = self.spec["refine"]
        grid = spec["grid"]
        study = optuna.create_study(direction="maximize", study_name="refine",
                                    sampler=optuna.samplers.GridSampler(grid, seed=self.spec["seed"]))
        n = 1
        for values in grid.values():
            n *= len(values)

        def objective(t: optuna.Trial) -> float:
            fixed = spec["fixed"]
            orig = self.suggest
            self.suggest = lambda trial: {**{k: trial.suggest_categorical(k, v) for k, v in grid.items()}, **fixed}
            try:
                return self.run_trial("refine", t)["success"]
            finally:
                self.suggest = orig

        study.optimize(objective, n_trials=n)

    def load_trials(self) -> None:
        """Reload the trials of a finished search from trials.csv (for --reselect)."""
        with (self.out_dir / "trials.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k in ("success", "ci_low", "ci_high", "mean_total_tokens", "mean_wall_s", "p95_wall_s",
                      "mean_steps", "error_rate", "J", "temperature"):
                r[k] = float(r[k])
            for k in ("trial", "max_steps", "retry_budget", "new_runs"):
                r[k] = int(r[k])
            r["feasible"] = r["feasible"] == "True"
            r.setdefault("config_path", f"configs/search/{r['name']}.yaml")
            if not r.get("config_path"):
                r["config_path"] = f"configs/search/{r['name']}.yaml"
        self.records = rows

    def run(self) -> dict[str, Any]:
        seed = self.spec["seed"]
        n_random = self.spec["samplers"]["random"]
        n_nsga = self.spec["samplers"]["nsga2"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        random_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed),
                                           study_name="random")
        random_study.optimize(lambda t: self.run_trial("random", t)["success"], n_trials=n_random)

        nsga = optuna.samplers.NSGAIISampler(seed=seed, population_size=max(4, n_nsga // 2),
                                             constraints_func=lambda t: t.user_attrs["constraint"])
        nsga_study = optuna.create_study(directions=["maximize", "minimize", "minimize"], sampler=nsga,
                                         study_name="nsga2")

        def multi(t: optuna.Trial) -> tuple[float, float, float]:
            rec = self.run_trial("nsga2", t)
            return rec["success"], rec["mean_total_tokens"], rec["mean_wall_s"]

        nsga_study.optimize(multi, n_trials=n_nsga)
        self.add_seed_configs()
        if self.spec.get("refine"):
            self.refine()
        return self.finish()

    def finish(self) -> dict[str, Any]:
        recs = self.records
        feasible = [r for r in recs if r["feasible"]] or recs
        best = sorted(feasible, key=lambda r: (-r["success"], r["mean_total_tokens"], r["mean_wall_s"]))[0]
        pareto = [r for r in recs if not any(
            o is not r and o["success"] >= r["success"] and o["mean_total_tokens"] <= r["mean_total_tokens"]
            and o["mean_wall_s"] <= r["mean_wall_s"] and (o["success"], o["mean_total_tokens"], o["mean_wall_s"])
            != (r["success"], r["mean_total_tokens"], r["mean_wall_s"]) for o in recs)]

        def best_so_far(sampler: str) -> list[float]:
            vals, cur = [], 0.0
            for r in [x for x in recs if x["sampler"] == sampler]:
                cur = max(cur, r["success"] if r["feasible"] else 0.0)
                vals.append(cur)
            return vals

        summary = {
            "space": self.spec["space"], "budget": self.spec["budget"], "split": self.spec["split"],
            "n_trials": len(recs), "n_feasible": sum(r["feasible"] for r in recs),
            "selected": best, "pareto_front": pareto,
            "best_by_sampler": {s: max((r["success"] for r in recs if r["sampler"] == s and r["feasible"]), default=None)
                                for s in ("random", "nsga2", "refine", "manual")},
            "seed_configs": self.spec.get("seed_configs", []),
            "best_so_far": {s: best_so_far(s) for s in ("random", "nsga2", "refine")},
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as fh:
            fields = list(dict.fromkeys(k for r in recs for k in r))
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        return summary


def select_and_register(summary: dict[str, Any]) -> str:
    """Write configs/selected.yaml from the best trial and register it as staging."""
    best = summary["selected"]
    cfg = load_config(REPO_ROOT / best["config_path"]).model_copy(update={
        "name": "selected",
        "description": f"Selected by the Part B search on validation ({best['name']}, success {best['success']:.3f}).",
    })
    SELECTED.write_text("# Selected by python -m code.search.run_search (Part B); see supporting_files/a2/search/\n"
                        + cfg.to_yaml(), encoding="utf-8", newline="\n")
    reg = Registry()
    version = reg.register(load_config(SELECTED), note=f"Part B search winner ({best['name']})",
                           source="configs/selected.yaml", run_id=best.get("mlflow_run_id"),
                           metrics={"validation_success": best["success"],
                                    "validation_mean_tokens": best["mean_total_tokens"]})
    reg.promote(version, "staging", reason=f"best feasible validation success in Part B search ({best['name']})")
    return version


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    p = argparse.ArgumentParser(description="Part B configuration search")
    p.add_argument("--space", type=Path, default=SPACE_FILE)
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--reselect", action="store_true",
                   help="reuse trials.csv of a finished search, add the seed configurations, select again")
    p.add_argument("--refine-only", action="store_true",
                   help="reuse trials.csv, run only stage 2 (refine grid), add seeds, select again")
    args = p.parse_args(argv)
    search = Search(args.space)
    if args.refine_only:
        search.load_trials()
        search.records = [r for r in search.records if r["sampler"] != "refine"]
        search.refine()
        search.add_seed_configs()
        summary = search.finish()
    elif args.reselect:
        search.load_trials()
        search.add_seed_configs()
        summary = search.finish()
    else:
        summary = search.run()
    print(json.dumps({k: summary[k] for k in ("n_trials", "n_feasible", "best_by_sampler", "selected")}, indent=2))
    if not args.no_register:
        print("registered and promoted to staging:", select_and_register(summary))


if __name__ == "__main__":
    main()
