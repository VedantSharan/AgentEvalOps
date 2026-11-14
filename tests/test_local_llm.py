"""Tests for code/experiments/local_llm.py (no Ollama server needed)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from code.experiments import local_llm
from code.experiments.local_llm import (
    GPUInUse,
    OfflineViolation,
    OllamaClient,
    assert_url_allowed,
    from_ollama_tool_calls,
    make_client,
    to_ollama_messages,
)


class _Resp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


CHAT_REPLY = {
    "model": "qwen2.5:3b",
    "message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "calculate", "arguments": {"expression": "2+2"}}}]},
    "prompt_eval_count": 120,
    "eval_count": 9,
}


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"posts": [], "vram": 0}

    def post(url: str, json: dict[str, Any], timeout: float) -> _Resp:  # noqa: A002
        state["posts"].append((url, json))
        return _Resp(CHAT_REPLY)

    def get(url: str, timeout: float) -> _Resp:
        if url.endswith("/api/ps"):
            return _Resp({"models": [{"name": "qwen2.5:3b", "size_vram": state["vram"]}]})
        if url.endswith("/api/tags"):
            return _Resp({"models": [{"name": "qwen2.5:3b", "digest": "abc123"}]})
        return _Resp({"version": "0.40.8"})

    monkeypatch.setattr(local_llm.requests, "post", post)
    monkeypatch.setattr(local_llm.requests, "get", get)
    return state


def test_messages_get_object_arguments_for_ollama() -> None:
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "retrieve", "arguments": '{"query": "x"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "name": "retrieve", "content": "ok"},
    ]
    out = to_ollama_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert out[0]["content"] == "" and out[1]["tool_name"] == "retrieve"


def test_malformed_argument_string_is_kept_not_dropped() -> None:
    out = to_ollama_messages([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]}])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"__raw__": "{bad"}


def test_tool_calls_come_back_openai_shaped() -> None:
    calls = from_ollama_tool_calls([{"function": {"name": "calculate", "arguments": {"expression": "1+1"}}}])
    assert calls == [{"id": "call_0", "type": "function",
                      "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}]
    assert from_ollama_tool_calls(None) is None


def test_offline_mode_blocks_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTEVALOPS_OFFLINE", "1")
    assert_url_allowed("http://127.0.0.1:11434/api/chat")
    assert_url_allowed("http://localhost:11434/api/chat")
    with pytest.raises(OfflineViolation):
        assert_url_allowed("https://api.groq.com/openai/v1/chat/completions")
    with pytest.raises(OfflineViolation):
        make_client("groq")


def test_offline_mode_off_allows_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTEVALOPS_OFFLINE", raising=False)
    assert_url_allowed("https://api.groq.com/openai/v1/chat/completions")


def test_request_pins_seed_context_and_threads(fake_server: dict[str, Any]) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:9", seed=7, num_ctx=2048, num_thread=4)
    resp = client.call_chat_completion("qwen2.5:3b", [{"role": "user", "content": "hi"}], temperature=0.0)
    _, payload = fake_server["posts"][0]
    assert payload["options"] == {"temperature": 0.0, "seed": 7, "num_ctx": 2048, "num_thread": 4}
    assert payload["stream"] is False
    assert resp.prompt_tokens == 120 and resp.completion_tokens == 9
    assert resp.tool_calls[0]["function"]["name"] == "calculate"
    assert resp.latency_ms >= 0


def test_per_call_seed_overrides_default(fake_server: dict[str, Any]) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:9", seed=7)
    client.call_chat_completion("qwen2.5:3b", [{"role": "user", "content": "hi"}], seed=99)
    assert fake_server["posts"][0][1]["options"]["seed"] == 99


def test_gpu_resident_model_is_refused(fake_server: dict[str, Any]) -> None:
    fake_server["vram"] = 1_000_000
    client = OllamaClient(base_url="http://127.0.0.1:9")
    with pytest.raises(GPUInUse):
        client.call_chat_completion("qwen2.5:3b", [{"role": "user", "content": "hi"}])


def test_server_introspection(fake_server: dict[str, Any]) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:9")
    assert client.server_version() == "0.40.8"
    assert client.model_digest("qwen2.5:3b") == "abc123"
    assert client.model_digest("missing:1b") is None


def test_gpu_guard_warn_mode_logs_instead_of_failing(fake_server: dict[str, Any],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    fake_server["vram"] = 1_000_000
    monkeypatch.setenv("AGENTEVALOPS_CPU_GUARD", "warn")
    client = OllamaClient(base_url="http://127.0.0.1:9")
    resp = client.call_chat_completion("qwen2.5:3b", [{"role": "user", "content": "hi"}])
    assert resp.prompt_tokens == 120 and len(client.gpu_warnings) == 1