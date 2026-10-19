"""
Grader for AgentEvalOps benchmark tasks.

Accepts a TaskRecord and a proposed answer string, then returns a
GradingResult that records whether the answer is correct, the rule that
was applied, and diagnostic detail.

Supported evaluation rules (from EvaluationRule enum):
  - exact_match        : case-insensitive stripped string equality
  - numeric_tolerance  : floating-point comparison within NUMERIC_TOLERANCE
  - structured_match   : order-independent, comma-separated list comparison
  - unanswerable       : answer must contain UNANSWERABLE_SIGNAL (case-insensitive)

An agent's answer is always a plain string (possibly None if not submitted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from code.shared.models import EvaluationRule, TaskRecord
from code.tasks.fixtures import UNANSWERABLE_SIGNAL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute tolerance used for numeric comparison.
# Two values are considered equal if |proposed - gold| <= NUMERIC_TOLERANCE.
NUMERIC_TOLERANCE: float = 1e-6


# ---------------------------------------------------------------------------
# GradingResult
# ---------------------------------------------------------------------------


@dataclass
class GradingResult:
    """Structured outcome from grading a single task response."""

    task_id: str
    correct: bool
    rule_applied: EvaluationRule
    proposed_answer: str | None
    gold_answer: str | None
    detail: str = ""
    # Extra diagnostics (e.g. parsed numeric values, diff)
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal grading helpers
# ---------------------------------------------------------------------------


def _grade_exact_match(proposed: str, gold: str) -> tuple[bool, str]:
    ok = proposed.strip().lower() == gold.strip().lower()
    return ok, f"'{proposed.strip()}' {'==' if ok else '!='} '{gold.strip()}' (case-insensitive)"


def _grade_numeric_tolerance(proposed: str, gold: str) -> tuple[bool, str, dict]:
    """Parse both strings as floats and compare within NUMERIC_TOLERANCE."""
    diagnostics: dict = {}
    try:
        p_val = float(proposed.strip())
    except ValueError:
        return False, f"Could not parse proposed answer as float: {proposed!r}", diagnostics
    try:
        g_val = float(gold.strip())
    except ValueError:
        return False, f"Gold answer is not a valid float: {gold!r}", diagnostics

    diff = abs(p_val - g_val)
    ok = diff <= NUMERIC_TOLERANCE
    diagnostics = {"proposed_float": p_val, "gold_float": g_val, "diff": diff, "tolerance": NUMERIC_TOLERANCE}
    return ok, f"|{p_val} - {g_val}| = {diff:.2e} (tolerance {NUMERIC_TOLERANCE:.2e})", diagnostics


def _grade_structured_match(proposed: str, gold: str) -> tuple[bool, str]:
    """
    Split both strings on ',' and compare element-wise after stripping
    and lowercasing.  Order matters: the agent must list items in the
    same order as the gold answer.
    """
    p_items = [x.strip().lower() for x in proposed.split(",")]
    g_items = [x.strip().lower() for x in gold.split(",")]
    ok = p_items == g_items
    return ok, f"{p_items} {'==' if ok else '!='} {g_items}"


def _grade_unanswerable(proposed: str) -> tuple[bool, str]:
    """
    The agent must include UNANSWERABLE_SIGNAL somewhere in its response.
    This is a necessary but not sufficient condition; it is the agreed
    minimum explicit refusal criterion.
    """
    ok = UNANSWERABLE_SIGNAL.lower() in proposed.strip().lower()
    return ok, (
        f"Response {'contains' if ok else 'does not contain'} "
        f"required signal {UNANSWERABLE_SIGNAL!r}"
    )


# ---------------------------------------------------------------------------
# Public grading function
# ---------------------------------------------------------------------------


def grade(task: TaskRecord, proposed_answer: str | None) -> GradingResult:
    """
    Grade *proposed_answer* against *task* and return a GradingResult.

    Args:
        task:            The TaskRecord being evaluated.
        proposed_answer: The agent's final answer string, or None if the
                         agent did not submit an answer.

    Returns:
        GradingResult with correct=True iff the answer satisfies the rule.
    """
    rule = task.evaluation_rule

    # Handle missing submission uniformly across all rules.
    if proposed_answer is None:
        if rule == EvaluationRule.unanswerable:
            # A None submission is NOT sufficient; the agent must explicitly
            # indicate it cannot answer.
            return GradingResult(
                task_id=task.task_id,
                correct=False,
                rule_applied=rule,
                proposed_answer=None,
                gold_answer=task.gold_answer,
                detail="No answer submitted. Agent must explicitly signal 'unanswerable'.",
            )
        return GradingResult(
            task_id=task.task_id,
            correct=False,
            rule_applied=rule,
            proposed_answer=None,
            gold_answer=task.gold_answer,
            detail="No answer submitted.",
        )

    if rule == EvaluationRule.exact_match:
        ok, detail = _grade_exact_match(proposed_answer, task.gold_answer)  # type: ignore[arg-type]
        return GradingResult(
            task_id=task.task_id,
            correct=ok,
            rule_applied=rule,
            proposed_answer=proposed_answer,
            gold_answer=task.gold_answer,
            detail=detail,
        )

    if rule == EvaluationRule.numeric_tolerance:
        ok, detail, diag = _grade_numeric_tolerance(proposed_answer, task.gold_answer)  # type: ignore[arg-type]
        return GradingResult(
            task_id=task.task_id,
            correct=ok,
            rule_applied=rule,
            proposed_answer=proposed_answer,
            gold_answer=task.gold_answer,
            detail=detail,
            diagnostics=diag,
        )

    if rule == EvaluationRule.structured_match:
        ok, detail = _grade_structured_match(proposed_answer, task.gold_answer)  # type: ignore[arg-type]
        return GradingResult(
            task_id=task.task_id,
            correct=ok,
            rule_applied=rule,
            proposed_answer=proposed_answer,
            gold_answer=task.gold_answer,
            detail=detail,
        )

    if rule == EvaluationRule.unanswerable:
        ok, detail = _grade_unanswerable(proposed_answer)
        return GradingResult(
            task_id=task.task_id,
            correct=ok,
            rule_applied=rule,
            proposed_answer=proposed_answer,
            gold_answer=None,
            detail=detail,
        )

    # Should be unreachable if EvaluationRule is exhaustive.
    raise ValueError(f"Unknown evaluation rule: {rule!r}")
