"""Build the exemplar banks the proposed agent can put in its system prompt (Part A).

"Training" this agent means choosing what goes into its prompt from the
training data. Two banks are built, only from train-split data:

train_base
    One worked example per train template family: the shortest A1 trajectory
    that grader v2 marks correct and whose tool calls all succeeded, shown with
    the task's gold answer (the concise format the grader expects).
train_aug_al
    train_base, plus
    * A1 Part D augmentation: for each family, a paraphrased train task paired
      with its parent's successful tool sequence (paraphrases preserve the label,
      so the same calls and answer stay valid);
    * A1 Part D active learning: "pitfall" notes written from the failures that
      the A1 active-learning score ranks highest among labelled train
      trajectories (data/active_learning/human/al_b30.csv), using the human
      failure category.

Part A compares the proposed agent with no bank, train_base and train_aug_al
on the validation split, i.e. performance "with and without augmented and
actively selected Assignment 1 data".

    python -m code.agent.exemplars          # writes prompts/exemplars/{train_base,train_aug_al}.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from code.agent.config import PROMPTS_DIR, REPO_ROOT

TRAJECTORIES = REPO_ROOT / "data" / "experiments" / "multi_provider_run" / "trajectories.jsonl"
REGRADE = REPO_ROOT / "data" / "processed" / "regrade_v2.csv"
TRAIN_SPLIT = REPO_ROOT / "data" / "splits" / "train.csv"
AL_RANKING = REPO_ROOT / "data" / "active_learning" / "human" / "al_b30.csv"
LABELS = [REPO_ROOT / "data" / "labels" / "labels_round1.csv", REPO_ROOT / "data" / "labels" / "labels_round2.csv"]
AUGMENTED = REPO_ROOT / "data" / "augmented" / "tasks_augmented.jsonl"
BASE_TASKS = REPO_ROOT / "data" / "processed" / "tasks_revised.jsonl"
N_PITFALLS = 3
RESULT_CHARS = 180


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _short(value: Any) -> str:
    text = json.dumps(value, default=str, separators=(",", ":"))
    return text if len(text) <= RESULT_CHARS else text[: RESULT_CHARS - 3] + "..."


def _calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"tool": s["tool_name"], "arguments": s.get("arguments") or {},
             "result": _short((s.get("tool_result") or {}).get("output"))}
            for s in traj["steps"] if s["action_type"] == "tool_call"]


def train_successes() -> dict[str, list[dict[str, Any]]]:
    """Successful, error-free train trajectories grouped by template family."""
    train_rows = _csv(TRAIN_SPLIT)
    train_runs = {r["item_id"] for r in train_rows if r["item_type"] == "trajectory"}
    correct = {r["run_id"] for r in _csv(REGRADE) if r["answer_correct_v2"] == "True"}
    tasks = {t["task_id"]: t for t in _jsonl(BASE_TASKS)}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for traj in _jsonl(TRAJECTORIES):
        if traj["run_id"] not in train_runs or traj["run_id"] not in correct:
            continue
        tool_steps = [s for s in traj["steps"] if s["action_type"] == "tool_call"]
        if not tool_steps or any(not (s.get("tool_result") or {}).get("success") for s in tool_steps):
            continue
        family = tasks[traj["task_id"]]["template_family"]
        by_family.setdefault(family, []).append(traj)
    for runs in by_family.values():
        runs.sort(key=lambda t: (len(t["steps"]), t["task_id"], t["run_id"]))
    return by_family


def base_bank() -> list[dict[str, Any]]:
    tasks = {t["task_id"]: t for t in _jsonl(BASE_TASKS)}
    bank = []
    for family, runs in sorted(train_successes().items()):
        best = runs[0]
        bank.append({"type": "example", "family": family, "source_run_id": best["run_id"],
                     "task": tasks[best["task_id"]]["prompt"], "calls": _calls(best),
                     "answer": tasks[best["task_id"]]["gold_answer"]})
    return bank


def paraphrase_examples() -> list[dict[str, Any]]:
    """One paraphrased train task per family, reusing a successful parent trajectory."""
    train_ids = {r["item_id"] for r in _csv(TRAIN_SPLIT) if r["item_type"] == "augmented:paraphrase"}
    paraphrases = {r["augmentation"]["parent_task_id"]: r for r in _jsonl(AUGMENTED)
                   if r["task"]["task_id"] in train_ids}
    out = []
    for family, runs in sorted(train_successes().items()):
        base_task = runs[0]["task_id"]
        # Prefer a different parent task than the base exemplar, for variety.
        for traj in sorted(runs, key=lambda t: (t["task_id"] == base_task, len(t["steps"]), t["run_id"])):
            rec = paraphrases.get(traj["task_id"])
            if rec:
                out.append({"type": "example", "family": family, "source_run_id": traj["run_id"],
                            "augmentation": rec["task"]["task_id"], "task": rec["task"]["prompt"],
                            "calls": _calls(traj), "answer": rec["task"]["gold_answer"]})
                break
    return out


_PITFALL_TEXT = {
    "malformed_arguments": "called {tool} with arguments the tool rejected ({error}). Use only the documented argument names and values.",
    "loop_or_repetition": "repeated {tool} with the same arguments without getting new information. Change the arguments or submit.",
    "premature_stop": "stopped before collecting every fact the task needs. Get each fact with a tool before answering.",
    "correct_process_with_wrong_answer": "found the right facts but submitted '{answer}' instead of '{gold}'. Re-check the calculation with calculate, and the unit and rounding the task asks for.",
    "hallucinated_result": "submitted '{answer}', a value no tool returned. Never state a value that a tool did not give.",
    "wrong_tool": "used {tool} where a different tool held the information. Pick the tool by the kind of fact needed.",
}


def pitfall_notes(n: int = N_PITFALLS) -> list[dict[str, Any]]:
    """Pitfalls from the top of the A1 active-learning ranking, train split only."""
    train_runs = {r["item_id"] for r in _csv(TRAIN_SPLIT) if r["item_type"] == "trajectory"}
    labels = {r["run_id"]: r for path in LABELS for r in _csv(path)}
    trajs = {t["run_id"]: t for t in _jsonl(TRAJECTORIES)}
    tasks = {t["task_id"]: t for t in _jsonl(BASE_TASKS)}
    notes, used_categories = [], set()
    for row in sorted(_csv(AL_RANKING), key=lambda r: int(r["rank"])):
        rid = row["run_id"]
        label = labels.get(rid)
        if rid not in train_runs or not label or label["primary_failure_category"] == "no_failure":
            continue
        cat = label["primary_failure_category"]
        if cat in used_categories or cat not in _PITFALL_TEXT:
            continue
        traj = trajs[rid]
        failed = next((s for s in traj["steps"] if s["action_type"] == "tool_call"
                       and not (s.get("tool_result") or {}).get("success")), None)
        first_tool = next((s for s in traj["steps"] if s["action_type"] == "tool_call"), None)
        step = failed or first_tool or {}
        text = _PITFALL_TEXT[cat].format(tool=step.get("tool_name", "a tool"),
                                         error=((step.get("tool_result") or {}).get("error") or "")[:80],
                                         answer=traj.get("final_answer"), gold=tasks[traj["task_id"]]["gold_answer"])
        notes.append({"type": "pitfall", "category": cat, "source_run_id": rid, "al_rank": int(row["rank"]),
                      "text": f"On the task '{tasks[traj['task_id']]['prompt'][:90]}...', an earlier run {text}"})
        used_categories.add(cat)
        if len(notes) == n:
            break
    return notes


def build(out_dir: Path = PROMPTS_DIR / "exemplars") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    banks = {"train_base": base_bank()}
    banks["train_aug_al"] = banks["train_base"] + paraphrase_examples() + pitfall_notes()
    paths = {}
    for name, bank in banks.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8", newline="\n")
        paths[name] = path
    return paths


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    for name, path in build().items():
        bank = json.loads(path.read_text(encoding="utf-8"))
        print(f"{name}: {len(bank)} entries -> {path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
