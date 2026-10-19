# Shared Data Contracts

## Task record

Every task contains:

- `task_id`: stable unique identifier.
- `template_family`: family identifier used for leakage-free splitting.
- `task_type`: operation or question category.
- `prompt`: instruction given to the agent.
- `answerable`: whether sufficient information exists to answer.
- `gold_answer`: expected answer, null for unanswerable tasks.
- `expected_tools`: tools expected to be useful.
- `evaluation_rule`: `exact_match`, `numeric_tolerance`, `structured_match`, or `unanswerable`.
- `metadata`: provenance, creation time, generator version, seed, and content hash.

## Tool contract

Every tool exposes a unique name, description, argument schema, and execution function. Inputs are validated before execution. Results and errors use consistent JSON-serializable structures.

Tools:
1. `retrieve`
2. `calculate`
3. `query_records`
4. `read_file`
5. `submit_answer`

File operations must remain inside the controlled project data directory. Arbitrary code execution and unrestricted filesystem access are prohibited.

## Trajectory record

Every execution records:

- `run_id`
- `task_id`
- `configuration_hash`
- `model_version`
- `prompt_version`
- `steps`
- `final_answer`
- `outcome`
- `metrics`
- `errors`

Each step records its index, action type, tool name when applicable, arguments, result, error, and available timing information. Unavailable token counts and timings must not be fabricated.

## Execution configuration

Each configuration identifies the model, prompt version, tool-selection strategy, temperature when supported, maximum steps, and retry budget. Hashes are computed from canonical JSON serialization.

## General rules

- All records must be JSON-serializable.
- Task generation uses explicit seeds.
- Raw artifacts are never overwritten by processing scripts.
- Gold answers are generated or verified independently of agent responses.
- Schema changes require corresponding code and test changes.
