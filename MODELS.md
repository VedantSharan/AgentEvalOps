# Models

A2 runs every model locally, as the brief requires: a quantised open-weight LLM of at most 8B parameters, served by Ollama on CPU, with no API keys and no outbound calls.

## Inference server

| Item | Value |
|---|---|
| Server | Ollama 0.40.8 |
| Image (experiments and the source of the binary in our Docker image) | `ollama/ollama:0.40.8@sha256:2b8eecdb85aa6ea67c381c8ecffec04b2658496cf2821a8f88b0485bdff198e8` (linux/amd64) |
| Start command for experiments | `bash scripts/ollama_cpu.sh`, then `export AGENTEVALOPS_OLLAMA_URL=http://127.0.0.1:11435` |
| Hardware used | CPU only. The container is started without `--gpus`, and `OllamaClient` raises `GPUInUse` if the server reports any model in GPU memory |

## Models

| Role | Ollama tag | Parameters | Quantisation | Digest (sha256) | Source |
|---|---|---|---|---|---|
| Main model | `qwen2.5:3b` | 3.1B | Q4_K_M (GGUF) | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | Ollama library build of Qwen2.5-3B-Instruct (Qwen team, Alibaba Cloud) |
| Weaker model (search value; model-swap drift case) | `qwen2.5:1.5b` | 1.5B | Q4_K_M (GGUF) | `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b` | Ollama library build of Qwen2.5-1.5B-Instruct |

Both are pulled once when the image is built and checked against these digests; nothing is downloaded at run time.

## Fixed inference options

Greedy and seeded sampling in llama.cpp is deterministic only for a fixed binary, thread count and context size, so these are pinned like a seed.

| Option | Value | Where it is set |
|---|---|---|
| `seed` | 203 (repeat *r* uses 203 + *r*) | `code/experiments/local_llm.py`, the evaluation script |
| `num_ctx` | 4096 | `AGENTEVALOPS_NUM_CTX` |
| `num_thread` | 8 | `AGENTEVALOPS_NUM_THREAD` |
| Parallel requests | 1 (`OLLAMA_NUM_PARALLEL=1`) | `scripts/ollama_cpu.sh`, Docker entrypoint. Batching requests together changes the numerics |
| `temperature` | from the configuration | `configs/*.yaml` |
