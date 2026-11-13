# Data Changes after `a1-final`

The A2 brief requires every fix or addition to the A1 data to be versioned with DVC and explained here. The gold test set (`data/splits/gold_test/`) is never edited; its checksums are the ones in `HANDOFF_A2.md`.

| # | Date | Change | Files | Why |
|---|---|---|---|---|
| 1 | 2026-11-13 | Line endings: files are checked out byte-for-byte (`.gitattributes`), and `dvc.lock` was re-hashed | `.gitattributes`, `dvc.lock` | The A1 lock had been hashed from CRLF working copies on Windows, so it did not match a Linux checkout. No data file changed; every data md5 in the `.dvc` pointers is the same |
| 2 | 2026-11-13 | Personal e-mail removed from the raw Label Studio export | `supporting_files/labels/annotator1_raw.json`, `supporting_files/agreement/round1/summary.json` | See below |
| 3 | 2026-11-13 | Three wrong human labels documented, not edited | none | See below |
| 4 | 2026-11-13 | Validation suite now also covers the 96 fault-injected tasks (`augmented_all`, 149 tasks, 12/12 checks pass) | `supporting_files/validation/` | In A1 the suite ran on only 53 of the 149 augmented tasks. Fault-injected tasks reuse their parent's prompt, so the duplicate check now compares prompt plus fault specification. The other datasets give the same results as in A1. The only text change in `corrupted.json` is the pydantic documentation URL inside three error messages (`errors.pydantic.dev/2.14` became `/2.13`), which comes from the installed pydantic build |

## 2. Anonymised Label Studio export

The round 1 export of annotator A1 stored the annotator's personal Gmail address in the `completed_by.email` field of all 45 annotations. `python -m code.annotation.anonymise_exports` replaces every address with the pseudonym `A1@anonymised.invalid`. It is a byte-level string replacement, so the rest of the export, including the numeric annotator `id` the agreement code reads, is unchanged.

| | SHA-256 of `annotator1_raw.json` |
|---|---|
| Original export (not redistributed, because it holds personal data) | `da65349a0b51aca1b6d48152804b2c2c9f9007db732c3077a9baa4033566b6c4` |
| Anonymised export (now in the repository) | `4b2d6c39856b5a45e0ded14f3b04fa64e59a9b868d75ded79adb71a1837f6c8d` |

**Effect on derived files:** re-running the `agreement@round1` and `labels@round1` stages on the anonymised export gives byte-identical labels, kappas, confusion matrices and disagreement tables (line endings aside). The only change is the export hash that `supporting_files/agreement/round1/summary.json` records, which was updated to the new value. `dvc.lock` records the new hashes.

**Limit:** the original export is still in the git history before this commit. Rewriting the shared repository's history would break every clone, so the history was left as it is.

`python -m code.annotation.anonymise_exports --check` exits 1 if any e-mail address is left in `supporting_files/labels/`; the test suite runs it.

## 3. Three wrong human labels (not edited)

Three round 2 trajectories submitted **no answer** (empty `final_answer`) on unanswerable tasks, but both annotators labelled them `success`. Labelling guideline v2 does not cover empty abstentions (`docs/annotation_findings.md` §5). Grader v2 marks all three as wrong ("No answer submitted."), which is the intended behaviour.

| `run_id` | Task | Human label | Grader v2 |
|---|---|---|---|
| `3fcee431-b49a-41f5-b812-b1579b0c7d12` | T6_unans_02 | success | failure |
| `58d66f53-8d57-42b0-862a-939f093ae427` | T6_unans_02 | success | failure |
| `ee77ee60-60d9-482f-97fe-518e8cc6cd8f` | T6_unans_03 | success | failure |

All three are in `data/splits/gold_test/gold_labels.csv`. The brief forbids editing the gold set, so the labels stay as they are. A2 handles them as follows:
- Task success in A2 is always graded by grader v2 against the gold answers in `gold_tasks.jsonl`, not taken from these labels.
- Any agreement figure between an automatic labeller and the human labels (for example the failure-category κ in Part C) leaves these three runs out and says so.
