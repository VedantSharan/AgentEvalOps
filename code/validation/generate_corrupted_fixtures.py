"""
Generates deterministic corrupted task and trajectory fixtures for AgentEvalOps (Part B).

These fixtures deliberately introduce specific faults to verify that the
12-check validation suite detects every class of error:
- JSON parsing errors
- Pydantic schema violations
- Duplicate IDs
- Broken reference integrity
- Cryptographic hash mismatches
- Duplicate task content
- Prompt answer leakage
- Invalid tool call argument schemas
- Metric range violations (negative latency, non-sequential steps)
- Prohibited null values
- Grader outcome inconsistencies
"""

import json
from pathlib import Path


def generate_corrupted_fixtures():
    corrupted_dir = Path("data/corrupted")
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    tasks_file = corrupted_dir / "tasks_corrupted.jsonl"
    trajectories_file = corrupted_dir / "trajectories_corrupted.jsonl"

    tasks_lines = [
        # 1. Broken JSON syntax (Check 1 & Check 2)
        '{"task_id": "CORRUPT_JSON_01", "prompt": "Unclosed json string',

        # 2. Schema violation: invalid evaluation_rule enum value (Check 2)
        json.dumps({
            "task_id": "CORRUPT_SCHEMA_01",
            "template_family": "multi_hop_retrieval",
            "task_type": "invalid_schema",
            "prompt": "Test prompt with invalid schema rule",
            "answerable": True,
            "gold_answer": "42",
            "expected_tools": ["calculate"],
            "evaluation_rule": "non_existent_rule_enum"
        }),

        # 3. Base valid task for reference
        json.dumps({
            "task_id": "VALID_TASK_01",
            "template_family": "multi_hop_retrieval",
            "task_type": "valid_base",
            "prompt": "What is the primary server name for regional routing?",
            "answerable": True,
            "gold_answer": "US-EAST-01",
            "expected_tools": ["retrieve"],
            "evaluation_rule": "exact_match",
            "metadata": {
                "provenance": "code/tasks/generator.py",
                "created_at": "2026-10-21T00:00:00+00:00",
                "generator_version": "1.0",
                "content_hash": "a"*64  # intentionally incorrect hash for Check 5
            }
        }),

        # 4. Duplicate task ID (Check 3: duplicate task_id with VALID_TASK_01)
        json.dumps({
            "task_id": "VALID_TASK_01",
            "template_family": "multi_step_calculation",
            "task_type": "duplicate_id",
            "prompt": "Different prompt but duplicate task ID",
            "answerable": True,
            "gold_answer": "100",
            "expected_tools": ["calculate"],
            "evaluation_rule": "exact_match"
        }),

        # 5. Semantic / Exact duplicate prompt (Check 6: duplicate task)
        json.dumps({
            "task_id": "CORRUPT_DUP_TASK_02",
            "template_family": "multi_hop_retrieval",
            "task_type": "duplicate_prompt",
            "prompt": "What is the primary server name for regional routing?",  # same prompt as VALID_TASK_01
            "answerable": True,
            "gold_answer": "US-EAST-01",
            "expected_tools": ["retrieve"],
            "evaluation_rule": "exact_match"
        }),

        # 6. Prompt Answer Leakage (Check 7: gold answer in prompt)
        json.dumps({
            "task_id": "CORRUPT_LEAK_01",
            "template_family": "log_file_audit",
            "task_type": "leaked_answer",
            "prompt": "Given that the secret password is AlphaBravoCharlie99, what is the secret password?",
            "answerable": True,
            "gold_answer": "AlphaBravoCharlie99",
            "expected_tools": ["read_file"],
            "evaluation_rule": "exact_match"
        }),
    ]

    tasks_file.write_text("\n".join(tasks_lines) + "\n", encoding="utf-8")

    trajectories_lines = [
        # 1. Trajectory referencing a non-existent task_id (Check 4: reference integrity)
        json.dumps({
            "run_id": "CORRUPT_RUN_REF_01",
            "task_id": "NON_EXISTENT_TASK_999",
            "configuration_hash": "b"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [],
            "final_answer": "N/A",
            "outcome": "failure",
            "metrics": {
                "total_steps": 0,
                "total_latency_ms": 150.0,
                "total_prompt_tokens": 100,
                "total_completion_tokens": 50
            },
            "errors": []
        }),

        # 2. Base valid trajectory referencing VALID_TASK_01
        json.dumps({
            "run_id": "RUN_VALID_BASE_01",
            "task_id": "VALID_TASK_01",
            "configuration_hash": "c"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [
                {
                    "step_index": 0,
                    "action_type": "tool_call",
                    "tool_name": "retrieve",
                    "arguments": {"query": "server"},
                    "tool_result": {"tool_name": "retrieve", "success": True, "output": "US-EAST-01", "latency_ms": 10.0},
                    "latency_ms": 10.0,
                    "prompt_tokens": 50,
                    "completion_tokens": 20
                }
            ],
            "final_answer": "US-EAST-01",
            "outcome": "success",
            "metrics": {
                "total_steps": 1,
                "total_latency_ms": 200.0,
                "total_prompt_tokens": 50,
                "total_completion_tokens": 20
            },
            "errors": []
        }),

        # 3. Duplicate run_id (Check 3)
        json.dumps({
            "run_id": "RUN_VALID_BASE_01",  # duplicate of previous run_id
            "task_id": "VALID_TASK_01",
            "configuration_hash": "d"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [],
            "final_answer": "US-EAST-01",
            "outcome": "success",
            "metrics": {
                "total_steps": 0,
                "total_latency_ms": 100.0,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 10
            },
            "errors": []
        }),

        # 4. Invalid tool call argument types (Check 9: calculate missing 'expression')
        json.dumps({
            "run_id": "CORRUPT_RUN_ARGS_01",
            "task_id": "VALID_TASK_01",
            "configuration_hash": "e"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [
                {
                    "step_index": 0,
                    "action_type": "tool_call",
                    "tool_name": "calculate",
                    "arguments": {"formula": "2 + 2"},  # wrong argument key! (requires 'expression')
                    "tool_result": {"tool_name": "calculate", "success": False, "error": "Missing expression", "latency_ms": 5.0},
                    "latency_ms": 5.0,
                    "prompt_tokens": 30,
                    "completion_tokens": 10
                }
            ],
            "final_answer": "4",
            "outcome": "failure",
            "metrics": {
                "total_steps": 1,
                "total_latency_ms": 100.0,
                "total_prompt_tokens": 30,
                "total_completion_tokens": 10
            },
            "errors": ["Missing expression"]
        }),

        # 5. Metric range violation: negative latency & total_steps mismatch (Check 10)
        json.dumps({
            "run_id": "CORRUPT_RUN_METRIC_01",
            "task_id": "VALID_TASK_01",
            "configuration_hash": "f"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [],
            "final_answer": "test",
            "outcome": "failure",
            "metrics": {
                "total_steps": 5,  # mismatch: 5 != len(steps)==0
                "total_latency_ms": -45.0,  # negative latency!
                "total_prompt_tokens": -10,  # negative tokens!
                "total_completion_tokens": 10
            },
            "errors": []
        }),

        # 6. Prohibited null values: outcome success but final_answer is null (Check 11)
        json.dumps({
            "run_id": "CORRUPT_RUN_NULL_01",
            "task_id": "VALID_TASK_01",
            "configuration_hash": "1"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [],
            "final_answer": None,  # null final answer on success!
            "outcome": "success",
            "metrics": {
                "total_steps": 0,
                "total_latency_ms": 100.0,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 10
            },
            "errors": []
        }),

        # 7. Gold Answer Grading Consistency: claims success but answer is wrong (Check 12)
        json.dumps({
            "run_id": "CORRUPT_RUN_GRADER_01",
            "task_id": "VALID_TASK_01",
            "configuration_hash": "2"*64,
            "model_version": "test_model:v1",
            "prompt_version": "v1",
            "steps": [],
            "final_answer": "COMPLETELY_WRONG_ANSWER_xyz",  # gold is US-EAST-01
            "outcome": "success",  # claiming success!
            "metrics": {
                "total_steps": 0,
                "total_latency_ms": 100.0,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 10
            },
            "errors": []
        })
    ]

    trajectories_file.write_text("\n".join(trajectories_lines) + "\n", encoding="utf-8")
    print(f"Generated {tasks_file} ({len(tasks_lines)} lines)")
    print(f"Generated {trajectories_file} ({len(trajectories_lines)} lines)")


if __name__ == "__main__":
    generate_corrupted_fixtures()
