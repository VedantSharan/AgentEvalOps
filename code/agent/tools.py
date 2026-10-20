"""
Typed, deterministic tool implementations for the AgentEvalOps benchmark.

Tools
-----
  retrieve        — look up a fact by topic or keyword from the local knowledge base
  calculate       — evaluate a safe arithmetic expression
  query_records   — aggregate or filter the employee records table
  read_file       — read a controlled virtual benchmark file
  submit_answer   — record the agent's final answer and terminate the loop

Design principles
-----------------
- No internet access, shell execution, or unrestricted filesystem access.
- The calculator uses AST-based safe evaluation; arbitrary code is rejected.
- File operations are confined to the virtual in-memory file store.
- All tools are deterministic for identical inputs.
- Latency is measured with time.perf_counter and returned in milliseconds.
- Results always conform to the shared ToolResult schema.
"""

from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from code.shared.models import ToolResult
from code.tasks.fixtures import EMPLOYEE_RECORDS, FACTS, FILE_FIXTURES

# ---------------------------------------------------------------------------
# Safe arithmetic calculator
# ---------------------------------------------------------------------------

# Permitted AST node types for the safe evaluator.
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Call,      # only math functions allowed
    ast.Attribute, # for math.sqrt etc — handled via whitelist
    ast.Name,      # only whitelisted names
    ast.Load,
)

# Whitelisted names and functions for the safe evaluator.
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}

_SAFE_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_UNOPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_UNOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are permitted.")
        fn = _SAFE_GLOBALS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not permitted: {node.func.id!r}")
        args = [_safe_eval_node(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Name):
        val = _SAFE_GLOBALS.get(node.id)
        if val is None:
            raise ValueError(f"Name not permitted: {node.id!r}")
        return val
    raise ValueError(f"AST node type not permitted: {type(node).__name__}")


def _safe_eval(expression: str) -> float | int:
    """
    Safely evaluate a numeric arithmetic expression.

    Only numeric literals, basic arithmetic operators (+, -, *, /, //, %, **),
    and whitelisted built-in functions (abs, round, sqrt, floor, ceil, min,
    max, sum, pow) are permitted.  Any other construct raises ValueError.

    Args:
        expression: Arithmetic expression string.

    Returns:
        Numeric result (int or float).

    Raises:
        ValueError: If the expression contains forbidden constructs or is
                    syntactically invalid.
    """
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in expression: {exc}") from exc

    # Walk AST and reject any forbidden node types.
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(
                f"Forbidden AST node '{type(node).__name__}' in expression."
            )
    return _safe_eval_node(tree)


# ---------------------------------------------------------------------------
# In-memory virtual file system
# ---------------------------------------------------------------------------

_VIRTUAL_FILES: dict[str, str] = {
    ff["filename"]: ff["content"] for ff in FILE_FIXTURES
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def retrieve(query: str) -> ToolResult:
    """
    Look up facts from the local knowledge base.

    Searches FACTS for entries whose 'question' or 'answer' contain *query*
    (case-insensitive).  Returns all matching facts as a list of dicts.

    Args:
        query: Search keyword or question fragment.

    Returns:
        ToolResult with output={'matches': [...]}.
    """
    t0 = time.perf_counter()
    if not isinstance(query, str) or not query.strip():
        return ToolResult(
            tool_name="retrieve",
            success=False,
            output=None,
            error="query must be a non-empty string.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    q = query.strip().lower()
    tokens = q.split()
    matches = [
        f for f in FACTS
        if (
            q in f["question"].lower()
            or q in f["answer"].lower()
            or q in f["topic"].lower()
            or (
                tokens
                and all(
                    t in f["question"].lower()
                    or t in f["answer"].lower()
                    or t in f["topic"].lower()
                    for t in tokens
                )
            )
        )
    ]
    latency = round((time.perf_counter() - t0) * 1000, 3)
    return ToolResult(
        tool_name="retrieve",
        success=True,
        output={"matches": matches},
        error=None,
        latency_ms=latency,
    )


def calculate(expression: str) -> ToolResult:
    """
    Evaluate a safe arithmetic expression and return the numeric result.

    Only numeric literals and whitelisted operators/functions are permitted.
    Arbitrary Python code, attribute access, imports, and string operations
    are all rejected.

    Args:
        expression: Arithmetic expression string, e.g. "17 * 6" or "sqrt(144)".

    Returns:
        ToolResult with output={'result': <number>, 'expression': <str>}.
    """
    t0 = time.perf_counter()
    if not isinstance(expression, str) or not expression.strip():
        return ToolResult(
            tool_name="calculate",
            success=False,
            output=None,
            error="expression must be a non-empty string.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    try:
        result = _safe_eval(expression)
    except (ValueError, ZeroDivisionError) as exc:
        return ToolResult(
            tool_name="calculate",
            success=False,
            output=None,
            error=str(exc),
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    latency = round((time.perf_counter() - t0) * 1000, 3)
    return ToolResult(
        tool_name="calculate",
        success=True,
        output={"result": result, "expression": expression},
        error=None,
        latency_ms=latency,
    )


def query_records(
    operation: str,
    field: str | None = None,
    value: str | None = None,
    aggregate: str | None = None,
    agg_field: str | None = None,
) -> ToolResult:
    """
    Query or aggregate the employee records table.

    Supported operations:
      "filter"    — filter rows where *field* == *value* (case-insensitive).
                    Returns the filtered list.
      "aggregate" — apply *aggregate* function over *agg_field* on the
                    (optionally filtered) rows.
                    Supported aggregates: count, sum, avg, max, min, list.

    Args:
        operation:  "filter" or "aggregate".
        field:      Column to filter on (required for "filter").
        value:      Value to match (required for "filter").
        aggregate:  Aggregation function (required for "aggregate").
        agg_field:  Column to aggregate over (required for aggregate except "count").

    Returns:
        ToolResult with output depending on the operation.
    """
    t0 = time.perf_counter()
    op = (operation or "").strip().lower()

    def _error(msg: str) -> ToolResult:
        return ToolResult(
            tool_name="query_records",
            success=False,
            output=None,
            error=msg,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    if op not in ("filter", "aggregate"):
        return _error(f"Unsupported operation {operation!r}. Use 'filter' or 'aggregate'.")

    # Start with full table, apply optional filter.
    rows: list[dict] = list(EMPLOYEE_RECORDS)
    if field is not None:
        if field not in EMPLOYEE_RECORDS[0]:
            return _error(f"Unknown field {field!r}.")
        if value is None:
            return _error("'value' is required when 'field' is specified.")
        rows = [r for r in rows if str(r[field]).lower() == str(value).lower()]

    if op == "filter":
        latency = round((time.perf_counter() - t0) * 1000, 3)
        return ToolResult(
            tool_name="query_records",
            success=True,
            output={"rows": rows, "count": len(rows)},
            error=None,
            latency_ms=latency,
        )

    # "aggregate"
    agg = (aggregate or "").strip().lower()
    if agg not in ("count", "sum", "avg", "max", "min", "list"):
        return _error(f"Unsupported aggregate {aggregate!r}. Use count/sum/avg/max/min/list.")

    if agg == "count":
        result: Any = len(rows)
    else:
        if not agg_field:
            return _error(f"'agg_field' is required for aggregate={agg!r}.")
        if agg_field not in EMPLOYEE_RECORDS[0]:
            return _error(f"Unknown agg_field {agg_field!r}.")
        values = [r[agg_field] for r in rows]
        if agg == "list":
            result = sorted(str(v) for v in values)
        elif agg == "sum":
            result = sum(float(v) for v in values)
        elif agg == "avg":
            result = sum(float(v) for v in values) / len(values) if values else None
        elif agg == "max":
            result = max(values)
        elif agg == "min":
            result = min(values)

    latency = round((time.perf_counter() - t0) * 1000, 3)
    return ToolResult(
        tool_name="query_records",
        success=True,
        output={"result": result, "aggregate": agg, "count": len(rows)},
        error=None,
        latency_ms=latency,
    )


def read_file(filename: str) -> ToolResult:
    """
    Read a controlled virtual benchmark file.

    Only filenames present in the FILE_FIXTURES virtual store are accessible.
    Absolute paths, directory traversal (../, /), and filenames not in the
    whitelist are all rejected.

    Args:
        filename: Base filename to read (e.g. "notes_alpha.txt").

    Returns:
        ToolResult with output={'filename': str, 'content': str, 'line_count': int}.
    """
    t0 = time.perf_counter()
    if not isinstance(filename, str) or not filename.strip():
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error="filename must be a non-empty string.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    name = filename.strip()
    # Reject any path separator or traversal attempt.
    if "/" in name or "\\" in name or ".." in name:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error=f"Path traversal or directory separators are not permitted: {name!r}.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    content = _VIRTUAL_FILES.get(name)
    if content is None:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output=None,
            error=f"File not found in virtual store: {name!r}. "
                  f"Available: {sorted(_VIRTUAL_FILES)}",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    lines = content.splitlines()
    latency = round((time.perf_counter() - t0) * 1000, 3)
    return ToolResult(
        tool_name="read_file",
        success=True,
        output={"filename": name, "content": content, "lines": lines, "line_count": len(lines)},
        error=None,
        latency_ms=latency,
    )


def submit_answer(answer: str) -> ToolResult:
    """
    Submit the agent's final answer and signal loop termination.

    Args:
        answer: The proposed answer string.

    Returns:
        ToolResult with output={'submitted_answer': answer}.
        The runner checks for this tool name to terminate the loop.
    """
    t0 = time.perf_counter()
    if not isinstance(answer, str):
        return ToolResult(
            tool_name="submit_answer",
            success=False,
            output=None,
            error=f"answer must be a string, got {type(answer).__name__}.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
    latency = round((time.perf_counter() - t0) * 1000, 3)
    return ToolResult(
        tool_name="submit_answer",
        success=True,
        output={"submitted_answer": answer},
        error=None,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "retrieve": retrieve,
    "calculate": calculate,
    "query_records": query_records,
    "read_file": read_file,
    "submit_answer": submit_answer,
}
