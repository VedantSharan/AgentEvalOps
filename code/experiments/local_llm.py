"""Local inference for Assignment 2: Ollama on CPU, no API keys, no outbound calls.

The A2 brief requires every model to run locally inside the container. This
module adds an ``OllamaClient`` with the same interface as
``code.experiments.providers.ProviderClient``, so the agent loop works with
either. It talks to Ollama's native ``/api/chat`` endpoint rather than the
OpenAI-compatible one, because only the native endpoint accepts ``num_ctx``
and ``num_thread``. Both are pinned, like the seed, because they change
llama.cpp's arithmetic: host experiments and the Docker image then produce
the same outputs (see MODELS.md).

Two guards enforce the A2 rules:
  * offline mode (``AGENTEVALOPS_OFFLINE=1``, set in the image) refuses any
    call to a non-loopback address;
  * after the first call for each model, the client asks the server which
    models are in GPU memory and raises ``GPUInUse`` if any are, so no
    reported number can come from a GPU run.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

from code.experiments.providers import (
    PROVIDER_SPECS,
    ErrorType,
    LLMResponse,
    ModelSpec,
    ProviderClient,
    ProviderConfig,
    ProviderException,
)

#: Base URL of the local Ollama server. Not OLLAMA_HOST, which Ollama itself
#: reads as its bind address inside the container.
OLLAMA_URL_ENV = "AGENTEVALOPS_OLLAMA_URL"
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
#: When "1", every call must go to a loopback address.
OFFLINE_ENV = "AGENTEVALOPS_OFFLINE"
NUM_CTX = int(os.getenv("AGENTEVALOPS_NUM_CTX", "4096"))
NUM_THREAD = int(os.getenv("AGENTEVALOPS_NUM_THREAD", "8"))
DEFAULT_SEED = 203
#: Models used in A2 (Q4_K_M GGUF builds from the Ollama library; digests in MODELS.md).
LOCAL_MODELS = ("qwen2.5:3b", "qwen2.5:1.5b")

PROVIDER_SPECS.setdefault("ollama", ProviderConfig(
    name="ollama",
    base_url=OLLAMA_DEFAULT_URL,
    env_key="AGENTEVALOPS_OLLAMA_URL",
    default_models=[ModelSpec(m, context_window=NUM_CTX, rpm_limit=10_000, rpd_limit=10_000_000) for m in LOCAL_MODELS],
    endpoint_path="/api/chat",
))


class OfflineViolation(RuntimeError):
    """Raised when offline mode is on and a call would leave the machine."""


class GPUInUse(RuntimeError):
    """Raised when the Ollama server holds a model in GPU memory (A2 runs are CPU-only)."""


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def assert_url_allowed(url: str) -> None:
    """Refuse non-loopback URLs when offline mode is on."""
    if os.getenv(OFFLINE_ENV, "") == "1" and not is_loopback_url(url):
        raise OfflineViolation(f"Offline mode: refusing outbound call to {url}")


def ollama_base_url() -> str:
    url = os.getenv(OLLAMA_URL_ENV, "").strip() or OLLAMA_DEFAULT_URL
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style chat messages to Ollama's native format.

    Ollama expects tool-call arguments as an object, not a JSON string, and has
    no tool_call_id.
    """
    out = []
    for m in messages:
        msg: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"__raw__": args}
                calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = calls
        if m["role"] == "tool" and m.get("name"):
            msg["tool_name"] = m["name"]
        out.append(msg)
    return out


def from_ollama_tool_calls(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Ollama tool calls to the OpenAI shape (string arguments, call ids)."""
    if not raw:
        return None
    calls = []
    for i, tc in enumerate(raw):
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    return calls


class OllamaClient(ProviderClient):
    """Chat client for a local Ollama server, returning OpenAI-shaped responses."""

    def __init__(self, base_url: str | None = None, seed: int = DEFAULT_SEED,
                 num_ctx: int = NUM_CTX, num_thread: int = NUM_THREAD) -> None:
        url = (base_url or ollama_base_url()).rstrip("/")
        super().__init__("ollama", ProviderConfig(
            name="ollama", base_url=url, env_key=OLLAMA_URL_ENV, endpoint_path="/api/chat",
            default_models=PROVIDER_SPECS["ollama"].default_models,
        ))
        self.base_url = url
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_thread = num_thread
        self._cpu_checked: set[str] = set()

    # ProviderClient interface -------------------------------------------------

    def is_configured(self) -> bool:
        return True

    def get_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def get_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def call_chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        timeout: float = 600.0,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = self.get_url()
        assert_url_allowed(url)
        options: dict[str, Any] = {
            "temperature": temperature,
            "seed": self.seed if seed is None else seed,
            "num_ctx": self.num_ctx,
            "num_thread": self.num_thread,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": to_ollama_messages(messages),
            "stream": False,
            "options": options,
            "keep_alive": -1,
        }
        if tools:
            payload["tools"] = tools

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise ProviderException(f"Ollama timeout ({timeout}s): {exc}", ErrorType.NETWORK_TIMEOUT) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderException(f"Cannot reach Ollama at {self.base_url}: {exc}", ErrorType.NETWORK_TIMEOUT) from exc
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        if resp.status_code == 404:
            raise ProviderException(f"Ollama model not found ({model_id}): {resp.text[:200]}", ErrorType.MODEL_NOT_FOUND)
        if resp.status_code != 200:
            raise ProviderException(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}", ErrorType.SERVER_ERROR)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderException(f"Ollama returned non-JSON: {resp.text[:200]}", ErrorType.MALFORMED_RESPONSE) from exc

        if model_id not in self._cpu_checked:
            self.assert_cpu_only()
            self._cpu_checked.add(model_id)

        message = data.get("message") or {}
        return LLMResponse(
            content=message.get("content") or None,
            tool_calls=from_ollama_tool_calls(message.get("tool_calls")),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            resolved_model=data.get("model", model_id),
            latency_ms=latency_ms,
            raw_response={k: v for k, v in data.items() if k != "message"},
            raw_message=dict(message),
        )

    # Server introspection -----------------------------------------------------

    def _get(self, path: str, timeout: float = 10.0) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        assert_url_allowed(url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def server_version(self) -> str:
        return str(self._get("/api/version").get("version", "unknown"))

    def model_digest(self, model_id: str) -> str | None:
        """Return the sha256 digest of an installed model, or None if it is not installed."""
        for m in self._get("/api/tags").get("models", []):
            if model_id in (m.get("name"), m.get("model")):
                return m.get("digest")
        return None

    def assert_cpu_only(self) -> None:
        """Raise GPUInUse if the server holds any model in GPU memory."""
        for m in self._get("/api/ps").get("models", []):
            if int(m.get("size_vram") or 0) > 0:
                raise GPUInUse(
                    f"{m.get('name')} is loaded with {m['size_vram']} bytes in GPU memory. "
                    "A2 runs are CPU-only: start the server with scripts/ollama_cpu.sh."
                )


def make_client(provider_name: str, **kwargs: Any) -> ProviderClient:
    """Return a chat client; "ollama" gives the local client, others the hosted A1 clients.

    Hosted clients are still refused in offline mode.
    """
    if provider_name == "ollama":
        return OllamaClient(**kwargs)
    client = ProviderClient(provider_name)
    assert_url_allowed(client.get_url())
    return client
