"""
Static source fixtures for the AgentEvalOps benchmark generator.

All data is defined here as plain Python constants so that the generator
is fully self-contained and requires no internet access, external APIs,
or LLM calls.  Every entry is deterministic and version-controlled.

Adding or removing rows here changes gold answers; bump GENERATOR_VERSION
and update tests accordingly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Generator identity
# ---------------------------------------------------------------------------

GENERATOR_VERSION: str = "0.1.0"

# ---------------------------------------------------------------------------
# Family 1: Information retrieval
# A local knowledge base of geography and science facts.
# ---------------------------------------------------------------------------

FACTS: list[dict] = [
    {
        "fact_id": "f01",
        "topic": "geography",
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "fact_id": "f02",
        "topic": "geography",
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
    },
    {
        "fact_id": "f03",
        "topic": "science",
        "question": "What is the chemical symbol for water?",
        "answer": "H2O",
    },
    {
        "fact_id": "f04",
        "topic": "science",
        "question": "How many planets are in the Solar System?",
        "answer": "8",
    },
]

# ---------------------------------------------------------------------------
# Family 2: Multi-step question answering
# Each entry requires combining two facts to reach an answer.
# gold_answer is pre-computed from the facts above and stored explicitly
# so tests can verify it independently of the generator logic.
# ---------------------------------------------------------------------------

MULTISTEP_QUESTIONS: list[dict] = [
    {
        "ms_id": "ms01",
        "question": (
            "The capital of France and the capital of Japan are both major cities. "
            "List both capitals in alphabetical order, separated by a comma."
        ),
        "required_facts": ["f01", "f02"],
        "answer": "Paris, Tokyo",
    },
    {
        "ms_id": "ms02",
        "question": (
            "Which topic has more facts in the knowledge base: geography or science? "
            "Answer with the topic name only."
        ),
        "required_facts": ["f01", "f02", "f03", "f04"],
        "answer": "geography",
    },
    {
        "ms_id": "ms03",
        "question": (
            "Combine the chemical symbol for water and the number of planets in the "
            "Solar System into a single string with no separator."
        ),
        "required_facts": ["f03", "f04"],
        "answer": "H2O8",
    },
    {
        "ms_id": "ms04",
        "question": (
            "How many total facts are in the knowledge base? "
            "Answer with the integer only."
        ),
        "required_facts": ["f01", "f02", "f03", "f04"],
        "answer": "4",
    },
]

# ---------------------------------------------------------------------------
# Family 3: Numerical calculation
# Each problem is expressed symbolically so gold answers are computable
# directly from the expression without an LLM.
# ---------------------------------------------------------------------------

CALC_PROBLEMS: list[dict] = [
    {
        "calc_id": "c01",
        "description": "What is 17 multiplied by 6?",
        "expression": "17 * 6",
        "answer": "102",
    },
    {
        "calc_id": "c02",
        "description": "What is the square root of 144, rounded to the nearest integer?",
        "expression": "round(144 ** 0.5)",
        "answer": "12",
    },
    {
        "calc_id": "c03",
        "description": "What is 15 percent of 240?",
        "expression": "0.15 * 240",
        "answer": "36.0",
    },
    {
        "calc_id": "c04",
        "description": "What is the result of (100 - 37) divided by 7, rounded to 2 decimal places?",
        "expression": "round((100 - 37) / 7, 2)",
        "answer": "9.0",
    },
]

# ---------------------------------------------------------------------------
# Family 4: Structured-data operations
# A small in-memory table of employee records.
# ---------------------------------------------------------------------------

EMPLOYEE_RECORDS: list[dict] = [
    {"emp_id": "E001", "name": "Alice", "department": "Engineering", "salary": 95000},
    {"emp_id": "E002", "name": "Bob", "department": "Marketing", "salary": 72000},
    {"emp_id": "E003", "name": "Carol", "department": "Engineering", "salary": 105000},
    {"emp_id": "E004", "name": "Dave", "department": "Marketing", "salary": 68000},
    {"emp_id": "E005", "name": "Eve", "department": "Engineering", "salary": 88000},
]

STRUCTURED_QUERIES: list[dict] = [
    {
        "sq_id": "sq01",
        "question": "How many employees are in the Engineering department?",
        "answer": "3",
    },
    {
        "sq_id": "sq02",
        "question": "What is the name of the highest-paid employee?",
        "answer": "Carol",
    },
    {
        "sq_id": "sq03",
        "question": (
            "What is the average salary of Marketing department employees? "
            "Round to the nearest whole number."
        ),
        "answer": "70000",
    },
    {
        "sq_id": "sq04",
        "question": "List the names of all Engineering employees in alphabetical order, separated by commas.",
        "answer": "Alice, Carol, Eve",
    },
]

# ---------------------------------------------------------------------------
# Family 5: Controlled file operations
# A set of small virtual text files with deterministic contents.
# Agents read these files from the controlled data directory.
# ---------------------------------------------------------------------------

FILE_FIXTURES: list[dict] = [
    {
        "file_id": "file01",
        "filename": "notes_alpha.txt",
        "content": "Line 1: apple\nLine 2: banana\nLine 3: cherry",
        "question": "How many lines does notes_alpha.txt contain?",
        "answer": "3",
    },
    {
        "file_id": "file02",
        "filename": "notes_beta.txt",
        "content": "hello world\nfoo bar\nbaz qux\nend",
        "question": "What is the last line of notes_beta.txt?",
        "answer": "end",
    },
    {
        "file_id": "file03",
        "filename": "numbers.txt",
        "content": "10\n20\n30\n40\n50",
        "question": "What is the sum of all numbers in numbers.txt?",
        "answer": "150",
    },
    {
        "file_id": "file04",
        "filename": "config.txt",
        "content": "key=value\nmode=production\nretries=3",
        "question": "What is the value associated with the key 'mode' in config.txt?",
        "answer": "production",
    },
]

# ---------------------------------------------------------------------------
# Family 6: Ambiguous / unanswerable tasks
# These tasks have insufficient information for a definitive answer.
# Agents should recognize this and decline to answer.
# The UNANSWERABLE_SIGNAL is the canonical phrase agents are expected to use.
# ---------------------------------------------------------------------------

UNANSWERABLE_SIGNAL: str = "unanswerable"

UNANSWERABLE_TASKS: list[dict] = [
    {
        "un_id": "un01",
        "question": (
            "What is the current stock price of ACME Corp? "
            "(No real-time data source is available.)"
        ),
        "reason": "real_time_data_unavailable",
    },
    {
        "un_id": "un02",
        "question": (
            "What will the weather be like in Paris next Tuesday? "
            "(No forecast data is provided.)"
        ),
        "reason": "future_event",
    },
    {
        "un_id": "un03",
        "question": (
            "What is the personal phone number of employee E002? "
            "(Personal contact details are not in the dataset.)"
        ),
        "reason": "data_not_available",
    },
    {
        "un_id": "un04",
        "question": (
            "Who will win the next World Cup? "
            "(Future outcome; no data available.)"
        ),
        "reason": "future_event",
    },
]
