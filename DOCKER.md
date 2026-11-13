# Docker image: AgentEvalOps service

A self-contained `linux/amd64` image that runs **on CPU only, with no internet, no API keys and no downloads at run time**. It bundles:
- both models, `qwen2.5:3b` (production) and `qwen2.5:1.5b` (drift demo), pulled at build time and checked against the digests in `MODELS.md`;
- the Ollama 0.40.8 CPU backend, copied from the same pinned image digest used for every reported experiment;
- the code, the registry (`registry/`, alias `production` = v7), the prompts and configurations, and the UI;
- the task data, including the gold test set (`data/splits/gold_test/`), whose checksums are verified before every validation.

| Item | Value |
|---|---|
| Image file in the submission | `artefacts/docker_image.tar.gz`. Its SHA-256 is in `artefacts/DOCKER.md`, because the tarball is built from this commit and cannot contain its own checksum |
| Built from | `code/Dockerfile` at tag `a2-final`. The commit is baked in (`/app/GIT_COMMIT`) and shown by `GET /version` and the UI's About tab |
| Base images | `python:3.10-slim-trixie@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d`, `ollama/ollama:0.40.8@sha256:2b8eecdb85aa6ea67c381c8ecffec04b2658496cf2821a8f88b0485bdff198e8` |
| Image size | about 4.1 GB unpacked (models 2.9 GB, Python packages 0.9 GB); 3.2 GB as reported by `docker image inspect` |
| RAM | 3.3 GB peak measured while serving (`supporting_files/a2/loadtest.json`). Give Docker at least 6 GB; well under the 16 GB limit |
| CPU | x86-64 with AVX2. Measured on an Intel Core i7-13700H (Docker Desktop VM, 20 vCPU), 8 inference threads |

## Load or build

```bash
# Load the submitted image (run from the folder holding artefacts/)
gunzip -c artefacts/docker_image.tar.gz | docker load        # -> Loaded image: agentevalops:a2

# Or rebuild it from code/ (no manual steps; the internet is needed only during the build)
cd code && docker build --platform linux/amd64 -t agentevalops:a2 .
```

## Start the service and UI (one command)

```bash
docker run --rm -p 7860:7860 agentevalops:a2
```

The model is loaded into memory before the server accepts requests. When `GET /health` returns `{"status":"ok", ...}`, the service is ready.

- **UI:** http://localhost:7860/ui. Its tabs are:
  - **Ask:** bundled examples, fault injection and model swap for drift demos, the answer, confidence, latency, version and commit, the trace ID, and every step with its tool call, arguments and output.
  - **Monitoring:** live label-free signals against their validation reference, detector state, alerts, and a button that replays shifted traffic.
  - **Validation:** a fixed subset of 5 gold tasks run live next to the reported values.
  - **About:** version, commit and model digest.
- **REST API:**
  - `POST /predict` with `{"task": "...", "fault": "none", "model": null}`;
  - `GET /health`, `GET /version` (commit, registry version, configuration hash, model digest), `GET /monitoring`;
  - `GET /validation` and `POST /validation/run`;
  - OpenAPI docs at `/docs`.

Example request:

```bash
curl -s localhost:7860/predict -H 'content-type: application/json' \
  -d '{"task": "What is the annual salary of employee ID E999 according to the corporate employee records?"}'
```

The response holds:
- the `answer`, `trace_id`, `confidence` and `latency_ms`;
- every step and tool call, with arguments and outputs;
- `graded` and `correct` for bundled benchmark tasks;
- the registry version and the git commit.

Every request is logged to `/app/output/logs/requests.jsonl` with its version, latency and confidence. The task text is stored only as a SHA-256 hash and length, and e-mail addresses and phone numbers are masked in logged answers.

**No outbound calls.** The container runs with only the port published, and the code makes no outbound calls:
- `AGENTEVALOPS_OFFLINE=1` refuses any non-loopback model URL;
- Ollama runs with `OLLAMA_NO_CLOUD=true`;
- MLflow and Gradio telemetry are switched off;
- the UI uses only system fonts.

The page HTML contains two social-preview `<meta>` image URLs, which browsers do not fetch.

## Validate the full gold set offline (one command)

```bash
docker run --rm --network none -v "$PWD/output:/app/output" agentevalops:a2 validate
```

What it does:
1. checks the gold checksums;
2. runs production (v7) on all 25 gold tasks from scratch, with no cached runs (about 12 minutes on the CPU above);
3. prints the metrics and writes `output/validation.json`;
4. compares the result with the reported value in `supporting_files/a2/reported_values.json`.

**Tolerance:** gold success within ±0.04 (one task in 25) of the reported 0.480.
- On the machine that produced the report, every per-task result is identical.
- Another CPU may select a different llama.cpp CPU kernel, which can change a greedy decode.
- The command exits with status 1 if the result is outside the tolerance.

Token counts can differ by a few percent between runs. Ollama counts only prompt tokens that were not served from its prefix cache, so the count depends on which request ran before; answers are unaffected.

Output measured on this machine (`supporting_files/a2/docker_validation.json`):

```text
  "version": "v7",
  "n_tasks": 25,
  "success": 0.48,
  "ci_95": [0.28, 0.68],
  "mean_steps": 2.64,
  "p50_latency_s": 11.46,
  "p95_latency_s": 67.21,
  "reported_success": 0.48,
  "difference": 0.0,
  "tolerance": 0.04,
  "within_tolerance": true,
  "per_task_agreement": 1.0,
VALIDATION PASSED (gold success 0.480, reported 0.48, tolerance +/-0.04)
```

## Smoke test (used by CI)

```bash
docker run --rm --network none agentevalops:a2 smoke
```

It starts the model server and the API, calls `/health`, `/version` and one `/predict`, and prints `SMOKE PASSED`.

## Measured CPU performance

From `supporting_files/a2/loadtest.json`. The load test was `python -m code.service.loadtest --container <name> --requests 10` against the running container, one request at a time; the server handles one model request at a time by design (`OLLAMA_NUM_PARALLEL=1`, so results do not vary).

| Metric | Value |
|---|---|
| p50 / p95 latency per request | 11.7 s / 60.1 s |
| Mean latency | 20.9 s |
| Throughput | 2.9 requests per minute |
| Container memory (peak / mean) | 3.3 GB / 3.2 GB |
| Errors | 0 of 10 |
| Hardware | Intel Core i7-13700H, Windows 11 + Docker Desktop (WSL2), CPU only |

A request is a whole agent run of 1 to 6 LLM turns on CPU. The first request after start-up also processes the long system prompt once, which takes about 45 s.
