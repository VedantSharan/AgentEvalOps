"""
Grader v2 for AgentEvalOps (Assignment 1, Part C finding).

Why this exists
---------------
Round-1 human annotation showed that grader v1 (code/tasks/grader.py) marks
correct answers as wrong when they are not bare strings: ``float("$25,920")``
raises, so "$25,920", "25,920 USD" and "Employee E001 has a salary of $95,000"
were all logged as failures.  For unanswerable tasks v1 demands the literal
word "unanswerable", so clear abstentions such as "Employee ID 'E999' is not
found" were also logged as failures.

Grader v1 is left untouched so the logged outcomes stay reproducible.  v2 is
a separate, more tolerant grader whose decisions are written to new files.

Rules (v2)
----------
* All rules: if the normalised answer equals the normalised gold -> correct.
  Normalisation = lower-case, drop '$' and thousands commas, collapse spaces,
  strip trailing punctuation.
* numeric_tolerance, and exact_match whose gold is a number (with an optional
  unit such as "USD"): extract the stand-alone numbers in the answer (digits
  inside identifiers like "E001" or "PRJ-INF-932" are ignored).  Correct iff
  the gold value is among them (|diff| <= NUMERIC_ABS_TOL).  If the answer
  also contains other numbers, the flag ``multi_number`` is raised so a human
  can confirm the gold value really is the stated answer.
* exact_match with a non-numeric gold: correct iff the normalised gold appears
  as a whole phrase in the normalised answer; ``extra_text`` is flagged when
  the answer contains more than the gold.
* structured_match: v1 behaviour on normalised strings (order matters).
* unanswerable: correct iff the answer contains the v1 signal word, OR it
  contains an explicit abstention / ambiguity phrase (ABSTENTION_PATTERNS)
  and neither is a bare number ("0") nor commits to a value ("... is 0").
  A grader cannot tell a principled abstention from a lucky give-up, so all
  abstention-based credits carry the flag ``abstention_phrase``.

Known limitation: containment rules can over-credit an answer that lists the
gold value among several candidates.  Every such case is flagged for review.

Usage (from the repository root):
    python -m code.tasks.grader_v2 \
        --tasks data/processed/tasks_revised.jsonl \
        --trajectories data/experiments/multi_provider_run/trajectories.jsonl \
        --out data/processed/regrade_v2.csv \
        --summary supporting_files/grader_v2/regrade_summary.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

from code.shared.models import EvaluationRule, TaskRecord
from code.tasks import grader as grader_v1
from code.tasks.fixtures import UNANSWERABLE_SIGNAL

GRADER_VERSION = "2.0.0"
NUMERIC_ABS_TOL = 0.005  # half a cent: answers are money amounts or counts

# A stand-alone number: not glued to letters/digits/underscore/hyphen on the left
# (so "E001", "PRJ-INF-932", "v1.0" are skipped) and not followed by a letter/digit.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_\-.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])|(?<![A-Za-z0-9_\-.,])-?\d+(?:\.\d+)?(?![\d,]*[A-Za-z_])")
_UNIT_WORDS = ("usd", "inr", "dollars", "rupees", "hours", "hrs", "ms", "%")

ABSTENTION_PATTERNS: tuple[str, ...] = (
    r"\bunanswerable\b",
    r"\bcannot\s+(?:be\s+)?(?:determine|determined|calculate|calculated|compute|computed|answer|answered|identify|identified)\b",
    r"\bcan(?:'|no)?t\s+(?:determine|calculate|compute|answer|identify)\b",
    r"\bunable\s+to\s+(?:determine|calculate|compute|answer|identify|find)\b",
    r"\b(?:couldn't|could\s+not|can't|cannot)\s+find\b",
    r"\bdoes\s+not\s+exist\b",
    r"\bnot\s+(?:be\s+)?(?:found|available|present|listed|recorded)\b",
    r"\bno\s+(?:such|matching|data|records?|information|entries|requests|employee)\b",
    r"\binsufficient\s+(?:data|information|evidence)\b",
    r"\b(?:file|log)\s+is\s+empty\b",
    r"\bambiguous\b",
    r"\b(?:two|multiple|several|more\s+than\s+one)\s+(?:employees|people|records|matches|entries|staff)\b",
)


@dataclass
class GradingResultV2:
    task_id: str
    correct: bool
    rule_applied: str
    detail: str
    flags: list[str] = field(default_factory=list)


def normalise(text: str) -> str:
    t = text.strip().lower().replace("$", "")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .;:!")


def extract_numbers(text: str) -> list[float]:
    values = []
    for m in _NUMBER_RE.finditer(text.replace("$", " ")):
        try:
            values.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def gold_as_number(gold: str) -> float | None:
    g = normalise(gold)
    for unit in _UNIT_WORDS:
        if g.endswith(" " + unit) or g.endswith(unit):
            g = g[: -len(unit)].strip()
            break
    try:
        return float(g)
    except ValueError:
        return None


def _is_bare_number(text: str) -> bool:
    return gold_as_number(text) is not None


# "average latency is 0", "answer: 42", "= 3.5" -> the agent committed to a value.
_COMMITTED_VALUE_RE = re.compile(r"\b(?:is|equals|was|answer)\s*[:=]?\s*-?\$?\d|=\s*-?\$?\d")


def grade_v2(task: TaskRecord, proposed: str | None) -> GradingResultV2:
    rule = task.evaluation_rule
    if proposed is None or not proposed.strip():
        return GradingResultV2(task.task_id, False, rule.value, "No answer submitted.")

    if rule == EvaluationRule.unanswerable:
        text = proposed.lower()
        hits = [p for p in ABSTENTION_PATTERNS if re.search(p, text)]
        if UNANSWERABLE_SIGNAL.lower() in text:
            return GradingResultV2(task.task_id, True, rule.value, "contains v1 signal word")
        if _is_bare_number(proposed):
            return GradingResultV2(task.task_id, False, rule.value, "bare value submitted instead of abstaining")
        if _COMMITTED_VALUE_RE.search(text):
            return GradingResultV2(task.task_id, False, rule.value, "abstains but still commits to a value", ["committed_value"])
        if hits:
            return GradingResultV2(task.task_id, True, rule.value, f"abstention phrase matched: {hits[0]}", ["abstention_phrase"])
        return GradingResultV2(task.task_id, False, rule.value, "no abstention or ambiguity signal found")

    gold = task.gold_answer or ""
    if normalise(proposed) == normalise(gold):
        return GradingResultV2(task.task_id, True, rule.value, "normalised strings equal")

    gold_num = gold_as_number(gold)
    if rule == EvaluationRule.numeric_tolerance or (rule == EvaluationRule.exact_match and gold_num is not None):
        if gold_num is None:
            return GradingResultV2(task.task_id, False, rule.value, f"gold {gold!r} is not numeric")
        nums = extract_numbers(proposed)
        matched = [n for n in nums if abs(n - gold_num) <= NUMERIC_ABS_TOL]
        if not matched:
            return GradingResultV2(task.task_id, False, rule.value, f"gold {gold_num} not among numbers {nums}")
        flags = ["multi_number"] if len(set(nums)) > 1 else []
        return GradingResultV2(task.task_id, True, rule.value, f"gold {gold_num} found among numbers {nums}", flags)

    if rule == EvaluationRule.exact_match:
        g = normalise(gold)
        p = normalise(proposed)
        if re.search(r"(?<![a-z0-9])" + re.escape(g) + r"(?![a-z0-9])", p):
            return GradingResultV2(task.task_id, True, rule.value, "gold phrase contained in answer", ["extra_text"])
        return GradingResultV2(task.task_id, False, rule.value, f"{g!r} not found in {p!r}")

    if rule == EvaluationRule.structured_match:
        p_items = [normalise(x) for x in proposed.split(",")]
        g_items = [normalise(x) for x in gold.split(",")]
        return GradingResultV2(task.task_id, p_items == g_items, rule.value, f"{p_items} vs {g_items}")

    raise ValueError(f"Unknown evaluation rule: {rule!r}")


# ---------------------------------------------------------------------------
# Batch re-grading CLI
# ---------------------------------------------------------------------------


def regrade(tasks_path: Path, trajectories_path: Path, out_csv: Path, summary_path: Path) -> dict:
    tasks = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = TaskRecord.model_validate_json(line)
            tasks[t.task_id] = t

    rows = []
    for line in trajectories_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tr = json.loads(line)
        task = tasks.get(tr["task_id"])
        if task is None:
            raise KeyError(f"trajectory {tr['run_id']} references unknown task {tr['task_id']}")
        v1 = grader_v1.grade(task, tr.get("final_answer"))
        v2 = grade_v2(task, tr.get("final_answer"))
        rows.append({
            "run_id": tr["run_id"],
            "task_id": tr["task_id"],
            "template_family": task.template_family,
            "evaluation_rule": task.evaluation_rule.value,
            "logged_outcome": tr["outcome"],
            "answer_correct_v1": v1.correct,
            "answer_correct_v2": v2.correct,
            "changed": v1.correct != v2.correct,
            "flags_v2": ";".join(v2.flags),
            "detail_v2": v2.detail,
            "gold_answer": task.gold_answer,
            "final_answer": tr.get("final_answer"),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    summary = {
        "grader_version": GRADER_VERSION,
        "tasks_file": tasks_path.as_posix(),
        "trajectories_file": trajectories_path.as_posix(),
        "n_trajectories": len(rows),
        "logged_outcome_counts": {o: count(lambda r, o=o: r["logged_outcome"] == o) for o in ("success", "failure", "error")},
        "correct_v1": count(lambda r: r["answer_correct_v1"]),
        "correct_v2": count(lambda r: r["answer_correct_v2"]),
        "changed_false_to_true": count(lambda r: r["changed"] and r["answer_correct_v2"]),
        "changed_true_to_false": count(lambda r: r["changed"] and not r["answer_correct_v2"]),
        "flagged_for_review": count(lambda r: r["changed"] and r["flags_v2"]),
        "v1_correct_matches_logged_success": count(lambda r: r["answer_correct_v1"] == (r["logged_outcome"] == "success")),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Re-grade trajectories with grader v2 (v1 results kept for comparison).")
    p.add_argument("--tasks", type=Path, default=Path("data/processed/tasks_revised.jsonl"))
    p.add_argument("--trajectories", type=Path, default=Path("data/experiments/multi_provider_run/trajectories.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/processed/regrade_v2.csv"))
    p.add_argument("--summary", type=Path, default=Path("supporting_files/grader_v2/regrade_summary.json"))
    a = p.parse_args()
    s = regrade(a.tasks, a.trajectories, a.out, a.summary)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
