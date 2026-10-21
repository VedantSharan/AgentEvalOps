"""
Export every prompt the code sends to a model into prompts/, exactly as used.

The text is read from the code itself (constants, plus the literal strings in the
agent loop), so prompts/ cannot drift from what actually ran.
tests/test_prompts.py fails if a file in prompts/ differs from the code.

Usage (from the repository root):
    python -m code.reporting.export_prompts
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

OUT = Path("prompts/model_prompts")


def _literal(path: str, startswith: str) -> str:
    """Return the first string literal in `path` that starts with `startswith`."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(startswith):
            return node.value
    raise LookupError(f"{startswith!r} not found in {path}")


def prompt_files() -> dict[str, str]:
    from code.agent import llm_runner
    from code.experiments import providers

    return {
        "collection_agent/system_instruction.txt": providers.SYSTEM_INSTRUCTION,
        "collection_agent/tool_schemas.json": json.dumps(providers.AGENT_TOOLS, indent=2),
        "collection_agent/submit_nudge_user_message.txt": _literal("code/experiments/agent_executor.py", "Please submit your final answer"),
        "collection_agent/smoke_test_user_message.txt": _literal("code/experiments/providers.py", "What is 7 * 6?"),
        "pilot_llm_runner/system_instruction.txt": llm_runner.SYSTEM_INSTRUCTION,
        "pilot_llm_runner/tool_schemas.json": json.dumps(llm_runner.AGENT_TOOLS, indent=2),
        "pilot_llm_runner/submit_nudge_user_message.txt": _literal("code/agent/llm_runner.py", "Please submit your final answer"),
    }


def main() -> None:
    for rel, text in prompt_files().items():
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        print("wrote", p.as_posix())


if __name__ == "__main__":
    main()
