"""Free-text tasks (the deployed service accepts any text task).

A user's task has no gold answer, so it is wrapped in a TaskRecord whose
answer cannot be graded (``gold_answer=""``). If the text is exactly the prompt
of a bundled benchmark task, that task is used instead, so its answer can be
graded against the gold value.
"""

from __future__ import annotations

import hashlib

from code.shared.models import EvaluationRule, TaskRecord

ADHOC_FAMILY = "adhoc"


def adhoc_task(prompt: str, known: dict[str, TaskRecord] | None = None) -> TaskRecord:
    text = prompt.strip()
    if known:
        for task in known.values():
            if task.prompt.strip() == text:
                return task
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return TaskRecord(task_id=f"adhoc-{digest}", template_family=ADHOC_FAMILY, task_type="adhoc", prompt=text,
                      answerable=True, gold_answer="", expected_tools=[], evaluation_rule=EvaluationRule.exact_match)


def is_graded(task: TaskRecord) -> bool:
    return task.template_family != ADHOC_FAMILY
