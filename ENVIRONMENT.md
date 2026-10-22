# Environment and Reproducibility Record

**Recorded:** 22 Oct 2026, from the environment used for Parts C–F.

Python packages are pinned in `requirements.txt`. The complete `pip freeze` of the same virtual environment (266 packages) is in `requirements_freeze.txt`. This file lists everything those two files do not cover.

**Status labels used below:**
- **Observed:** read from the machine or from data files.
- **Not recorded:** the information was not captured at the time and cannot be recovered from the repository.
- **To confirm:** a team member must fill this in.

## 1. Software

| Component | Version | Status / source |
|---|---|---|
| Operating system | Ubuntu 22.04.5 LTS, Linux 6.8.0-138-generic x86_64 | Observed (`lsb_release -d`, `dvc version`) |
| Python | 3.10.12 | Observed |
| DVC command-line tool | 3.67.1 (pip); dvc_data 3.18.3, scmrepo 3.6.2 | Observed (`dvc version`) |
| Label Studio, round 2 (A2's machine) | 1.23.2 | Observed (`pip show label-studio`) |
| Label Studio, round 2 (A1's machine) | **To confirm** | A1 to run `pip show label-studio` |
| Label Studio, round 1 (both annotators) | **Not recorded** | The export files do not store the Label Studio version |
| MLflow | 3.17.2 | Observed; used by the tracer (`code/tracing/mlflow_tracer.py`), benchmark runner (`code/experiments/runner.py`), trace ingestion adapter (`code/tracing/export_trajectories_to_mlflow.py`), and local backend store (`sqlite:///mlruns/mlflow.db`) |
| pytest | 8.4.2 | Observed. Run as `python -m pytest`; `pyproject.toml` adds `-p no:debugging` because the package name `code` shadows a standard-library module |
| Git | **To confirm** | `git --version` |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver 595.99.02 | Observed. **Not used**: no model was run locally; CUDA was not used |

The original 154 trajectories were collected on 21 Oct 2026 (run summary timestamp 2026-10-21T11:06:51Z, runtime 1997 s; `data/experiments/multi_provider_run/summary.json`). The OS and Python version of that collection machine are **not recorded**. The repository's documentation links point to a Windows path, so it may have been a different machine (**to confirm**, Vedant).

## 2. Models that produced data

All models were called through hosted, free-tier, OpenAI-compatible APIs (`code/experiments/providers.py`). Hosted APIs do not expose weight revisions or hashes. The exact identifier is the model string sent to the provider, recorded in every trajectory's `model_version` field.

**Collection configuration** (`code/experiments/trajectory_worker.py`):
- `prompt_version` = v1
- temperature = 0.0
- max_steps = 5 LLM turns
- retry_budget = 1
- function calling: OpenAI style
- 10 distinct `configuration_hash` values (one per model)
- **Telemetry architecture:** Batch runs logged trajectories directly to append-only JSONL files (`trajectories.jsonl`) to isolate external API concurrency and rate-limit backoffs from SQLite database locks. An offline ingestion adapter (`code/tracing/export_trajectories_to_mlflow.py`) converts these records into MLflow traces stored in `mlruns/mlflow.db`.

| Provider | Model string (exact) | Trajectories |
|---|---|---|
| Cohere | `command-a-03-2025` | 19 |
| Cohere | `command-r7b-12-2024` | 20 |
| Google Gemini API | `gemini-3.5-flash` | 2 |
| Google Gemini API | `gemini-3.5-flash-lite` | 17 |
| Groq | `openai/gpt-oss-120b` | 13 |
| Groq | `openai/gpt-oss-20b` | 20 |
| Groq | `qwen/qwen3.8-27b` | 18 |
| NVIDIA NIM | `nvidia/nemotron-3.5-lightning-30b-a3b` | 20 |
| OpenRouter | `minimax/minimax-m3.1-flash:free` | 20 |
| OpenRouter | `nvidia/nemotron-3.5-lightning:free` | 5 |

Source: `supporting_files/dataset_stats.json → trajectories.by_model`.

**Model-version limitation:** the provider-side version behind a model string can change without notice. The date of collection (21 Oct 2026) is the best available version anchor.

**No model produced labels, augmentations or grades:**
- Labels are human, with adjudication drafts that were AI-assisted (see `LABELLING_PROTOCOL.md`).
- Paraphrases are rule-based (`code/augmentation/paraphrase.py`).
- Grading is code (`code/tasks/grader.py`, `grader_v2.py`).
- No API calls were made for Part D: `code/augmentation/run_augmented.py` exists but was **not run**.

## 3. Randomness and determinism

| Step | Seed / rule | Where |
|---|---|---|
| Task generation | `BENCHMARK_SEED = 42` (generation is deterministic; regeneration reproduces identical content hashes) | `code/tasks/revised_generator.py` |
| Collection order (task shuffle) | `--seed 42` | `code/experiments/trajectory_worker.py`, `work_queue.py` |
| Round-1 sample (45) | `seed = 42` | `code/annotation/sample_for_labelling.py` |
| Round-2 sample (40) and presentation order | `SEED = 2026` | `code/annotation/sample_round2.py` |
| Augmentation (paraphrase seeds `203000 + i`; spot-check sample) | `AUG_SEED = 203` | `code/augmentation/build_augmented.py` |
| Active learning vs random (tie-breaks and random baseline, 1000 trials) | `SEED = 203` | `code/active_learning/al_vs_random.py` |
| Splits | No randomness: deterministic family assignment | `code/splits/make_splits.py` |
| LLM sampling during collection | temperature 0.0; provider-side non-determinism is still possible | `trajectory_worker.py` |

## 4. Rebuilding

`dvc repro` reruns the whole derived pipeline (`dvc.yaml`, 14 stages). In testing, a full rerun reproduced every derived file byte-for-byte, except the matplotlib PNG files, whose bytes depend on the matplotlib version. See `README.md` for the commands.
