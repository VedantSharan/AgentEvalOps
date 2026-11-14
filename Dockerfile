# AgentEvalOps service image: linux/amd64, CPU only, fully offline at run time.
#
#   docker build --platform linux/amd64 -t agentevalops:a2 .
#   docker run --rm -p 7860:7860 agentevalops:a2                    # API + UI on http://localhost:7860/ui
#   docker run --rm --network none agentevalops:a2 validate        # full gold-set validation
#
# Inference uses the same Ollama build (same image digest) as every reported
# experiment, so validation reproduces the reported answers. Both models are
# pulled and checked against their digests at build time; nothing is downloaded
# when the container runs.

ARG OLLAMA_IMAGE=ollama/ollama:0.40.8@sha256:2b8eecdb85aa6ea67c381c8ecffec04b2658496cf2821a8f88b0485bdff198e8
ARG PYTHON_IMAGE=python:3.10-slim-trixie@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d

FROM ${OLLAMA_IMAGE} AS ollama

# Resolve the git commit of the build context (no git needed).
FROM ${PYTHON_IMAGE} AS commit
COPY .git /src/.git
COPY deploy/git_commit.py /src/git_commit.py
RUN python /src/git_commit.py /src/.git > /commit.txt && cat /commit.txt

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/models \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NOPRUNE=1 \
    OLLAMA_NO_CLOUD=true \
    AGENTEVALOPS_OLLAMA_URL=http://127.0.0.1:11434 \
    AGENTEVALOPS_OFFLINE=1 \
    AGENTEVALOPS_CPU_GUARD=warn \
    AGENTEVALOPS_NUM_THREAD=8 \
    AGENTEVALOPS_NUM_CTX=4096 \
    AGENTEVALOPS_LOG_DIR=/app/output/logs \
    MLFLOW_TRACKING_URI=sqlite:////app/output/mlruns/mlflow.db \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=1 \
    GIT_PYTHON_REFRESH=quiet \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_OFFLINE=1

# CPU inference only: the Ollama binary, llama.cpp CPU backends and runner (no CUDA/Vulkan/MLX).
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama/lib*.so* /usr/lib/ollama/
COPY --from=ollama /usr/lib/ollama/llama-server /usr/lib/ollama/llama-server

WORKDIR /app
COPY deploy/requirements-docker.lock requirements-docker.lock
RUN pip install -r requirements-docker.lock

# Models: pulled once at build time, verified against MODELS.md digests.
COPY deploy/verify_models.py /app/deploy/verify_models.py
RUN (ollama serve > /tmp/ollama-build.log 2>&1 &) \
    && for i in $(seq 1 60); do ollama list > /dev/null 2>&1 && break; sleep 1; done \
    && ollama pull qwen2.5:3b \
    && ollama pull qwen2.5:1.5b \
    && python /app/deploy/verify_models.py

COPY code/ code/
COPY configs/ configs/
COPY prompts/ prompts/
COPY registry/ registry/
COPY deploy/ deploy/
COPY deploy/data/ data/
COPY supporting_files/a2/reported_values.json supporting_files/a2/reported_values.json
COPY scripts/docker_entrypoint.sh scripts/docker_entrypoint.sh
COPY MODELS.md TEST_SET_LOG.md ./
COPY --from=commit /commit.txt /app/GIT_COMMIT
RUN chmod +x scripts/docker_entrypoint.sh && mkdir -p /app/output/logs /app/output/mlruns

ARG IMAGE_BUILD_TIME=""
ENV IMAGE_BUILD_TIME=${IMAGE_BUILD_TIME}
EXPOSE 7860
ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["serve"]
