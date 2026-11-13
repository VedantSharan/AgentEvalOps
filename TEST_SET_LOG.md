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
