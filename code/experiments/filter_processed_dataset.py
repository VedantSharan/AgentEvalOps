"""
Filters raw execution trajectories into the official processed benchmark dataset (Part A & B).

Per dataset policy:
- Execution errors (such as step exhaustion or fatal API/infrastructure drops) are
  retained in raw experiment logs (data/experiments/multi_provider_run/trajectories.jsonl)
  as historical evidence, but are excluded from all processed evaluation datasets and
  downstream sampling pools.
- The processed benchmark dataset (data/processed/trajectories_processed.jsonl)
  consists strictly of genuine agent completions: outcome in {'success', 'failure'}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def filter_trajectories(
    raw_path: Path,
    processed_path: Path,
) -> tuple[int, int, int]:
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    lines = raw_path.read_text(encoding="utf-8").strip().split("\n")
    total_raw = 0
    valid_count = 0
    error_count = 0

    valid_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        total_raw += 1
        record = json.loads(line)
        outcome = record.get("outcome")

        if outcome == "error":
            error_count += 1
        elif outcome in ("success", "failure"):
            valid_count += 1
            valid_lines.append(line.strip())
        else:
            # Any unrecognized outcome also excluded from processed data
            error_count += 1

    processed_path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    return total_raw, valid_count, error_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw trajectories into processed dataset.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/experiments/multi_provider_run/trajectories.jsonl"),
        help="Path to raw trajectories JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/trajectories_processed.jsonl"),
        help="Path to output processed trajectories JSONL",
    )

    args = parser.parse_args()
    total, valid, errors = filter_trajectories(args.raw, args.output)

    print(f"Processed Trajectory Filtering Complete:")
    print(f"  Raw Trajectories:        {total} ({args.raw})")
    print(f"  Excluded Error Runs:     {errors} (retained in raw only)")
    print(f"  Processed Valid Dataset: {valid} ({args.output})")


if __name__ == "__main__":
    main()
