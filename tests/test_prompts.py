"""prompts/model_prompts must match exactly what the code sends to models."""

from pathlib import Path

from code.reporting.export_prompts import OUT, prompt_files


def test_prompt_files_match_code() -> None:
    for rel, text in prompt_files().items():
        on_disk = (OUT / rel).read_text(encoding="utf-8")
        assert on_disk.rstrip("\n") == text.rstrip("\n"), f"{rel} differs from the code; run python -m code.reporting.export_prompts"
