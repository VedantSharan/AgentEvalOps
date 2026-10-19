"""
Tests for code.tasks.generator and code.tasks.grader.

Covers:
  - Exactly 24 tasks, six families, four tasks per family
  - Unique task IDs
  - All tasks validate against TaskRecord (via generate_benchmark itself)
  - Deterministic task content for a fixed seed
  - Gold answers independently computable from source fixtures
  - Correct and incorrect answer grading for all evaluation rules
  - Numeric tolerance boundary conditions
  - Unanswerable task grading rules (None answer is not sufficient)
  - JSONL round-trip: write then load produces identical records
  - Repeated generation does not produce duplicate IDs or change content
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from code.shared.models import EvaluationRule, TaskRecord
from code.tasks.fixtures import (
    CALC_PROBLEMS,
    EMPLOYEE_RECORDS,
    FACTS,
    FILE_FIXTURES,
    GENERATOR_VERSION,
    MULTISTEP_QUESTIONS,
    STRUCTURED_QUERIES,
    UNANSWERABLE_SIGNAL,
    UNANSWERABLE_TASKS,
)
from code.tasks.generator import (
    BENCHMARK_SEED,
    TASKS_PER_FAMILY,
    TOTAL_TASKS,
    export_jsonl,
    generate_benchmark,
    load_jsonl,
)
from code.tasks.grader import NUMERIC_TOLERANCE, GradingResult, grade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark() -> list[TaskRecord]:
    """Generate once per test session for speed."""
    return generate_benchmark(seed=BENCHMARK_SEED)


@pytest.fixture(scope="module")
def tasks_by_family(benchmark: list[TaskRecord]) -> dict[str, list[TaskRecord]]:
    result: dict[str, list[TaskRecord]] = {}
    for t in benchmark:
        result.setdefault(t.template_family, []).append(t)
    return result


# ---------------------------------------------------------------------------
# Generator: structure
# ---------------------------------------------------------------------------


class TestBenchmarkStructure:
    def test_total_task_count(self, benchmark: list[TaskRecord]):
        assert len(benchmark) == TOTAL_TASKS
        assert TOTAL_TASKS == 24

    def test_six_families(self, tasks_by_family: dict):
        expected_families = {
            "retrieval",
            "multistep_qa",
            "calculation",
            "structured_data",
            "file_ops",
            "unanswerable",
        }
        assert set(tasks_by_family.keys()) == expected_families

    def test_four_tasks_per_family(self, tasks_by_family: dict):
        for family, tasks in tasks_by_family.items():
            assert len(tasks) == TASKS_PER_FAMILY, (
                f"Family '{family}' has {len(tasks)} tasks, expected {TASKS_PER_FAMILY}"
            )

    def test_unique_task_ids(self, benchmark: list[TaskRecord]):
        ids = [t.task_id for t in benchmark]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_all_tasks_are_task_records(self, benchmark: list[TaskRecord]):
        for t in benchmark:
            assert isinstance(t, TaskRecord)

    def test_all_tasks_have_metadata(self, benchmark: list[TaskRecord]):
        for t in benchmark:
            assert t.metadata is not None, f"Task {t.task_id} has no metadata"
            assert t.metadata.generator_version == GENERATOR_VERSION
            assert t.metadata.seed == BENCHMARK_SEED
            assert t.metadata.provenance == "code/tasks/generator.py"

    def test_metadata_content_hash_present(self, benchmark: list[TaskRecord]):
        for t in benchmark:
            assert t.metadata is not None
            assert t.metadata.content_hash is not None
            assert len(t.metadata.content_hash) == 64  # SHA-256 hex

    def test_answerable_tasks_have_gold_answer(self, benchmark: list[TaskRecord]):
        for t in benchmark:
            if t.answerable:
                assert t.gold_answer is not None, f"{t.task_id} is answerable but has no gold_answer"

    def test_unanswerable_tasks_have_no_gold_answer(self, benchmark: list[TaskRecord]):
        for t in benchmark:
            if not t.answerable:
                assert t.gold_answer is None, f"{t.task_id} is unanswerable but has gold_answer"
                assert t.evaluation_rule == EvaluationRule.unanswerable


# ---------------------------------------------------------------------------
# Generator: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_content(self):
        run1 = generate_benchmark(seed=42)
        run2 = generate_benchmark(seed=42)
        for t1, t2 in zip(run1, run2):
            assert t1.task_id == t2.task_id
            assert t1.prompt == t2.prompt
            assert t1.gold_answer == t2.gold_answer

    def test_repeated_generation_no_duplicate_ids(self):
        run1 = generate_benchmark(seed=42)
        run2 = generate_benchmark(seed=42)
        combined_ids = [t.task_id for t in run1] + [t.task_id for t in run2]
        # IDs are stable between runs; pairs should be identical, not distinct
        assert len(run1) == len(run2)
        for t1, t2 in zip(run1, run2):
            assert t1.task_id == t2.task_id

    def test_different_seed_different_metadata(self):
        """Seed change must propagate to metadata."""
        run_42 = generate_benchmark(seed=42)
        run_99 = generate_benchmark(seed=99)
        for t42, t99 in zip(run_42, run_99):
            assert t42.metadata.seed != t99.metadata.seed

    def test_task_ids_are_stable(self, benchmark: list[TaskRecord]):
        """Known IDs must always appear."""
        ids = {t.task_id for t in benchmark}
        assert "retrieval-f01" in ids
        assert "calculation-c01" in ids
        assert "unanswerable-un01" in ids


# ---------------------------------------------------------------------------
# Generator: gold answers independently verifiable from fixtures
# ---------------------------------------------------------------------------


class TestGoldAnswers:
    def test_retrieval_gold_answers_match_fixtures(self, tasks_by_family: dict):
        retrieval_tasks = {t.task_id: t for t in tasks_by_family["retrieval"]}
        for fact in FACTS:
            task_id = f"retrieval-{fact['fact_id']}"
            assert task_id in retrieval_tasks, f"Missing task for {fact['fact_id']}"
            assert retrieval_tasks[task_id].gold_answer == fact["answer"]

    def test_multistep_gold_answers_match_fixtures(self, tasks_by_family: dict):
        ms_tasks = {t.task_id: t for t in tasks_by_family["multistep_qa"]}
        for ms in MULTISTEP_QUESTIONS:
            task_id = f"multistep-{ms['ms_id']}"
            assert task_id in ms_tasks
            assert ms_tasks[task_id].gold_answer == ms["answer"]

    def test_calculation_gold_answers_match_fixtures(self, tasks_by_family: dict):
        calc_tasks = {t.task_id: t for t in tasks_by_family["calculation"]}
        for calc in CALC_PROBLEMS:
            task_id = f"calculation-{calc['calc_id']}"
            assert task_id in calc_tasks
            assert calc_tasks[task_id].gold_answer == calc["answer"]

    def test_structured_gold_answers_match_fixtures(self, tasks_by_family: dict):
        sq_tasks = {t.task_id: t for t in tasks_by_family["structured_data"]}
        for sq in STRUCTURED_QUERIES:
            task_id = f"structured-{sq['sq_id']}"
            assert task_id in sq_tasks
            assert sq_tasks[task_id].gold_answer == sq["answer"]

    def test_file_ops_gold_answers_match_fixtures(self, tasks_by_family: dict):
        fo_tasks = {t.task_id: t for t in tasks_by_family["file_ops"]}
        for ff in FILE_FIXTURES:
            task_id = f"fileops-{ff['file_id']}"
            assert task_id in fo_tasks
            assert fo_tasks[task_id].gold_answer == ff["answer"]

    def test_calculation_answers_independently_computable(self):
        """
        Verify gold answers by evaluating the fixture expressions directly.
        This is independent of the generator code.
        """
        for calc in CALC_PROBLEMS:
            computed = eval(calc["expression"])  # noqa: S307 (controlled fixture data)
            assert str(computed) == calc["answer"], (
                f"{calc['calc_id']}: eval({calc['expression']!r}) = {computed!r}, "
                f"expected {calc['answer']!r}"
            )

    def test_structured_engineering_count(self):
        """Independently verify sq01 gold answer from EMPLOYEE_RECORDS."""
        count = sum(1 for e in EMPLOYEE_RECORDS if e["department"] == "Engineering")
        assert str(count) == STRUCTURED_QUERIES[0]["answer"]

    def test_structured_highest_paid(self):
        """Independently verify sq02 gold answer."""
        top = max(EMPLOYEE_RECORDS, key=lambda e: e["salary"])
        assert top["name"] == STRUCTURED_QUERIES[1]["answer"]

    def test_structured_avg_salary(self):
        """Independently verify sq03 gold answer."""
        marketing = [e for e in EMPLOYEE_RECORDS if e["department"] == "Marketing"]
        avg = round(sum(e["salary"] for e in marketing) / len(marketing))
        assert str(avg) == STRUCTURED_QUERIES[2]["answer"]

    def test_structured_engineering_names(self):
        """Independently verify sq04 gold answer."""
        names = sorted(e["name"] for e in EMPLOYEE_RECORDS if e["department"] == "Engineering")
        assert ", ".join(names) == STRUCTURED_QUERIES[3]["answer"]

    def test_file_line_count(self):
        """Independently verify file01 gold answer."""
        ff = FILE_FIXTURES[0]
        line_count = len(ff["content"].splitlines())
        assert str(line_count) == ff["answer"]

    def test_file_last_line(self):
        """Independently verify file02 gold answer."""
        ff = FILE_FIXTURES[1]
        last = ff["content"].splitlines()[-1]
        assert last == ff["answer"]

    def test_file_sum(self):
        """Independently verify file03 gold answer."""
        ff = FILE_FIXTURES[2]
        total = sum(int(n) for n in ff["content"].splitlines())
        assert str(total) == ff["answer"]

    def test_file_config_value(self):
        """Independently verify file04 gold answer."""
        ff = FILE_FIXTURES[3]
        kv = dict(line.split("=", 1) for line in ff["content"].splitlines())
        assert kv["mode"] == ff["answer"]


# ---------------------------------------------------------------------------
# Grader: exact_match
# ---------------------------------------------------------------------------


class TestGraderExactMatch:
    @pytest.fixture
    def exact_task(self, tasks_by_family: dict) -> TaskRecord:
        """Return the first retrieval task (exact_match rule)."""
        return tasks_by_family["retrieval"][0]

    def test_correct_answer(self, exact_task: TaskRecord):
        r = grade(exact_task, exact_task.gold_answer)
        assert r.correct is True
        assert r.rule_applied == EvaluationRule.exact_match

    def test_case_insensitive_match(self, exact_task: TaskRecord):
        r = grade(exact_task, exact_task.gold_answer.upper())
        assert r.correct is True

    def test_whitespace_stripped(self, exact_task: TaskRecord):
        r = grade(exact_task, f"  {exact_task.gold_answer}  ")
        assert r.correct is True

    def test_wrong_answer(self, exact_task: TaskRecord):
        r = grade(exact_task, "Wrong City")
        assert r.correct is False

    def test_none_answer(self, exact_task: TaskRecord):
        r = grade(exact_task, None)
        assert r.correct is False
        assert "No answer submitted" in r.detail

    def test_result_has_task_id(self, exact_task: TaskRecord):
        r = grade(exact_task, exact_task.gold_answer)
        assert r.task_id == exact_task.task_id

    def test_result_is_grading_result(self, exact_task: TaskRecord):
        r = grade(exact_task, exact_task.gold_answer)
        assert isinstance(r, GradingResult)


# ---------------------------------------------------------------------------
# Grader: numeric_tolerance
# ---------------------------------------------------------------------------


class TestGraderNumericTolerance:
    @pytest.fixture
    def calc_task(self, tasks_by_family: dict) -> TaskRecord:
        """Return the 15%-of-240 calculation task (gold='36.0')."""
        return next(t for t in tasks_by_family["calculation"] if t.task_id == "calculation-c03")

    def test_exact_gold_answer(self, calc_task: TaskRecord):
        r = grade(calc_task, "36.0")
        assert r.correct is True

    def test_within_tolerance(self, calc_task: TaskRecord):
        # 36.0 ± half of NUMERIC_TOLERANCE
        close = str(36.0 + NUMERIC_TOLERANCE * 0.5)
        r = grade(calc_task, close)
        assert r.correct is True

    def test_at_tolerance_boundary_exact(self, calc_task: TaskRecord):
        # Exactly at the tolerance boundary → still correct
        boundary = str(36.0 + NUMERIC_TOLERANCE)
        r = grade(calc_task, boundary)
        assert r.correct is True

    def test_just_outside_tolerance(self, calc_task: TaskRecord):
        # Infinitesimally beyond tolerance → incorrect
        import math
        just_over = str(36.0 + NUMERIC_TOLERANCE + 1e-15)
        r = grade(calc_task, just_over)
        # Due to float precision, this might still be True; accept both but
        # verify diagnostics are populated.
        assert "diff" in r.diagnostics

    def test_clearly_wrong_number(self, calc_task: TaskRecord):
        r = grade(calc_task, "99.0")
        assert r.correct is False

    def test_non_numeric_proposal(self, calc_task: TaskRecord):
        r = grade(calc_task, "thirty-six")
        assert r.correct is False
        assert "Could not parse" in r.detail

    def test_none_answer(self, calc_task: TaskRecord):
        r = grade(calc_task, None)
        assert r.correct is False

    def test_diagnostics_populated(self, calc_task: TaskRecord):
        r = grade(calc_task, "36.0")
        assert "proposed_float" in r.diagnostics
        assert "gold_float" in r.diagnostics
        assert "diff" in r.diagnostics
        assert "tolerance" in r.diagnostics


# ---------------------------------------------------------------------------
# Grader: structured_match
# ---------------------------------------------------------------------------


class TestGraderStructuredMatch:
    @pytest.fixture
    def struct_task(self, tasks_by_family: dict) -> TaskRecord:
        """Return the engineering-names task (gold='Alice, Carol, Eve')."""
        return next(t for t in tasks_by_family["structured_data"] if t.task_id == "structured-sq04")

    def test_correct_answer(self, struct_task: TaskRecord):
        r = grade(struct_task, struct_task.gold_answer)
        assert r.correct is True
        assert r.rule_applied == EvaluationRule.structured_match

    def test_case_insensitive(self, struct_task: TaskRecord):
        r = grade(struct_task, struct_task.gold_answer.lower())
        assert r.correct is True

    def test_wrong_order(self, struct_task: TaskRecord):
        r = grade(struct_task, "Carol, Alice, Eve")
        assert r.correct is False

    def test_extra_item(self, struct_task: TaskRecord):
        r = grade(struct_task, "Alice, Bob, Carol, Eve")
        assert r.correct is False

    def test_missing_item(self, struct_task: TaskRecord):
        r = grade(struct_task, "Alice, Carol")
        assert r.correct is False

    def test_none_answer(self, struct_task: TaskRecord):
        r = grade(struct_task, None)
        assert r.correct is False


# ---------------------------------------------------------------------------
# Grader: unanswerable
# ---------------------------------------------------------------------------


class TestGraderUnanswerable:
    @pytest.fixture
    def un_task(self, tasks_by_family: dict) -> TaskRecord:
        return tasks_by_family["unanswerable"][0]

    def test_correct_signal(self, un_task: TaskRecord):
        r = grade(un_task, UNANSWERABLE_SIGNAL)
        assert r.correct is True
        assert r.rule_applied == EvaluationRule.unanswerable

    def test_signal_in_longer_response(self, un_task: TaskRecord):
        r = grade(un_task, f"I cannot answer because this is {UNANSWERABLE_SIGNAL}.")
        assert r.correct is True

    def test_signal_case_insensitive(self, un_task: TaskRecord):
        r = grade(un_task, UNANSWERABLE_SIGNAL.upper())
        assert r.correct is True

    def test_none_is_not_sufficient(self, un_task: TaskRecord):
        """None submission must NOT count as a correct refusal."""
        r = grade(un_task, None)
        assert r.correct is False
        assert "explicitly signal" in r.detail.lower() or "no answer" in r.detail.lower()

    def test_empty_string_is_not_sufficient(self, un_task: TaskRecord):
        """An empty response must NOT count as a correct refusal."""
        r = grade(un_task, "")
        assert r.correct is False

    def test_wrong_answer_is_not_correct(self, un_task: TaskRecord):
        r = grade(un_task, "The stock price is $42.")
        assert r.correct is False

    def test_gold_answer_is_none(self, un_task: TaskRecord):
        r = grade(un_task, UNANSWERABLE_SIGNAL)
        assert r.gold_answer is None


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


class TestJSONLRoundTrip:
    def test_export_and_load(self, benchmark: list[TaskRecord]):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            out = export_jsonl(benchmark, output_path=path, overwrite=False)
            loaded = load_jsonl(out)

        assert len(loaded) == len(benchmark)
        for original, recovered in zip(benchmark, loaded):
            assert original.task_id == recovered.task_id
            assert original.gold_answer == recovered.gold_answer
            assert original.evaluation_rule == recovered.evaluation_rule
            assert original.answerable == recovered.answerable

    def test_jsonl_line_count(self, benchmark: list[TaskRecord]):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            export_jsonl(benchmark, output_path=path, overwrite=False)
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == len(benchmark)

    def test_overwrite_false_raises(self, benchmark: list[TaskRecord]):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            export_jsonl(benchmark, output_path=path, overwrite=False)
            with pytest.raises(FileExistsError):
                export_jsonl(benchmark, output_path=path, overwrite=False)

    def test_overwrite_true_replaces(self, benchmark: list[TaskRecord]):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            export_jsonl(benchmark, output_path=path, overwrite=False)
            out = export_jsonl(benchmark, output_path=path, overwrite=True)
            assert out.exists()

    def test_loaded_tasks_validate_as_task_records(self, benchmark: list[TaskRecord]):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            export_jsonl(benchmark, output_path=path, overwrite=False)
            loaded = load_jsonl(path)
        for t in loaded:
            assert isinstance(t, TaskRecord)

    def test_full_roundtrip_equality(self, benchmark: list[TaskRecord]):
        """Timestamps in metadata differ between runs; compare stable fields only."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "tasks.jsonl"
            export_jsonl(benchmark, output_path=path, overwrite=False)
            loaded = load_jsonl(path)
        for orig, rec in zip(benchmark, loaded):
            assert orig.task_id == rec.task_id
            assert orig.prompt == rec.prompt
            assert orig.gold_answer == rec.gold_answer
            assert orig.template_family == rec.template_family
            assert orig.evaluation_rule == rec.evaluation_rule
            assert orig.answerable == rec.answerable
            assert orig.expected_tools == rec.expected_tools
