# Gold Test Set Log

Every evaluation on the gold test set (`data/splits/gold_test/`) is recorded here, as the A2 brief requires, including each Part E gate check. The gold set is never used for training, prompt design, configuration search, threshold tuning or model selection.

Before every evaluation, `code/evaluation/gold.py` (`verify_gold`) checks both files against the SHA-256 values frozen in `HANDOFF_A2.md` and refuses to run if they differ:

| File | SHA-256 |
|---|---|
| `gold_tasks.jsonl` | `f4b15d31aafd6ce4eb76ceaaca7c630cc7c4c08c72e93ca1bcec94b53c20d994` |
| `gold_labels.csv` | `72e8f84fa8363e0f9a0da2dfc408fd3cee454f7befed875e9dba11cbd5ef78bc` |

The evaluation script appends one row per run automatically.

| Date (UTC) | Registry version | Configuration hash | Model | Reason | Success | MLflow run ID |
|---|---|---|---|---|---|---|
| 2026-11-13 04:18 | v1 | `7b7d92628afd` | - | Part C final evaluation of b0 | 0.440 | `c9b9cf46aac04147827e3387b1e65af8` |
| 2026-11-13 04:22 | v2 | `8c3f00e7aee1` | qwen2.5:3b | Part C final evaluation of b1 | 0.440 | `f403727ba3914b34886239ee15696144` |
| 2026-11-13 08:56 | v3 | `9dce03e06f9e` | qwen2.5:3b | Part C final evaluation of b2 | 0.400 | `6fdb52d0500b481db38181f4d69fe1df` |
| 2026-11-13 10:13 | v5 | `ac90f8b182e7` | qwen2.5:3b | Part C final evaluation of production (v5) | 0.400 | `8617aa47cbb244bd908c69cc33bec318` |
