"""Tests for code/tasks/grader_v2.py. Cases are taken from real round-1 trajectories."""

from __future__ import annotations

import pytest

from code.shared.models import EvaluationRule, TaskRecord
from code.tasks import grader as grader_v1
from code.tasks.grader_v2 import extract_numbers, grade_v2


def _task(rule: EvaluationRule, gold: str | None) -> TaskRecord:
    return TaskRecord(
        task_id="T_test",
        template_family="test_family",
        task_type="test",
        prompt="test prompt",
        answerable=rule != EvaluationRule.unanswerable,
        gold_answer=gold,
        evaluation_rule=rule,
    )


NUM = _task(EvaluationRule.numeric_tolerance, "25920.00")
UNANS = _task(EvaluationRule.unanswerable, None)


@pytest.mark.parametrize("answer", ["$25,920", "25,920 USD", "The book value at the end of Year 2 is $25,920 USD.", "25920"])
def test_numeric_formats_accepted(answer: str) -> None:
    assert grade_v2(NUM, answer).correct


@pytest.mark.parametrize("answer", ["$36,900.00", "25,921", "I'm sorry, I was unable to find the information.", ""])
def test_numeric_wrong_or_missing_rejected(answer: str) -> None:
    assert not grade_v2(NUM, answer).correct


def test_v1_rejects_what_v2_accepts() -> None:
    # Documents the v1 defect that motivated v2.
    assert not grader_v1.grade(NUM, "$25,920").correct
    assert grade_v2(NUM, "$25,920").correct


def test_identifier_digits_ignored() -> None:
    assert extract_numbers("Employee E001 (Alice) has an annual salary of $95,000.") == [95000.0]
    assert extract_numbers("project PRJ-INF-932 under policy v1.0") == []


def test_multi_number_is_flagged() -> None:
    r = grade_v2(_task(EvaluationRule.exact_match, "180 USD"),
                 "Under the superseded 2024 travel policy (v1.0), the allowance was 180 USD.")
    assert r.correct and "multi_number" in r.flags


def test_exact_match_count_wrong() -> None:
    assert not grade_v2(_task(EvaluationRule.exact_match, "5"), "6").correct


def test_exact_match_text_containment() -> None:
    t = _task(EvaluationRule.exact_match, "IngressRouter")
    assert grade_v2(t, "The service is IngressRouter.").correct
    assert not grade_v2(t, "EdgeProxy-A").correct


@pytest.mark.parametrize("answer", [
    "Employee ID 'E999' is not found in the corporate employee records.",
    "The file is empty, so I cannot calculate the average latency of the logged requests.",
    "There are two employees named Rahul Sharma: one in Engineering and one in Sales.",
    "unanswerable",
])
def test_abstentions_accepted(answer: str) -> None:
    assert grade_v2(UNANS, answer).correct


@pytest.mark.parametrize("answer", ["0", "No requests logged; average latency is 0.", "The quantum key was QK-9928-ALPHA.", ""])
def test_fabricated_values_rejected(answer: str) -> None:
    assert not grade_v2(UNANS, answer).correct
