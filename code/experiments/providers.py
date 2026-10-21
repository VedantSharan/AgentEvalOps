"""
Multi-provider API adapters and rate-limit tracking for AgentEvalOps.

Supports configurable OpenAI-compatible providers:
  - Groq
  - Mistral
  - Google Gemini Developer API
  - NVIDIA NIM
  - OpenRouter (with pinned free models)
  - Cohere (compatibility endpoint)
  - Cloudflare Workers AI

Handles authentication via environment variables without logging secrets,
response parsing, rate-limit header extraction, error classification,
and model smoke testing.
"""

from __future__ import annotations

import email.utils
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI-compatible Tool Specifications for Benchmark Agent
# ---------------------------------------------------------------------------

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Look up facts from the local knowledge base matching a query keyword or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or question fragment.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a safe arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate (e.g. '17 * 6' or 'round(100 / 7, 2)').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_records",
            "description": "Query or aggregate the employee records table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["filter", "aggregate"],
                        "description": "Operation type: 'filter' to filter rows, 'aggregate' to compute summary metric.",
                    },
                    "field": {
                        "type": "string",
                        "description": "Column to filter on (emp_id, name, department, salary).",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to match for the filter operation.",
                    },
                    "aggregate": {
                        "type": "string",
                        "enum": ["count", "sum", "avg", "max", "min", "list"],
                        "description": "Aggregation function to apply.",
                    },
                    "agg_field": {
                        "type": "string",
                        "description": "Column to aggregate over (e.g. 'salary' or 'name').",
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a controlled virtual benchmark file. Relative filenames in virtual store permitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative filename to read (e.g. 'server_metrics.csv' or 'logs/gateway_01.json').",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the agent's final answer to conclude the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer string to submit.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You are an AI assistant evaluating tools for the AgentEvalOps benchmark. "
    "Carefully analyze the prompt and use the provided tools to retrieve necessary evidence, "
    "query structured data, or calculate numerical values. "
    "If the prompt explicitly contains all necessary information without requiring tools, "
    "or if requested information is absent or ambiguous across all tools, submit your answer directly. "
    "Always submit your final answer using the submit_answer tool."
)


# ---------------------------------------------------------------------------
# Error Classifications
# ---------------------------------------------------------------------------


class ErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"                    # HTTP 429
    DAILY_QUOTA = "daily_quota"                  # Daily limit reached
    AUTH_FAILURE = "auth_failure"                # HTTP 401 / 403 billing/key
    MODEL_NOT_FOUND = "model_not_found"          # HTTP 404
    SERVER_ERROR = "server_error"                # HTTP 5xx
    NETWORK_TIMEOUT = "network_timeout"          # Timeout / ConnectionError
    MALFORMED_RESPONSE = "malformed_response"    # Bad JSON or schema
    TOOL_FORMAT_ERROR = "tool_format_error"      # Provider does not support tools
    BUDGET_EXHAUSTED = "budget_exhausted"        # Local request cap reached


class ProviderException(Exception):
    """Base exception for provider calls."""

    def __init__(self, message: str, error_type: ErrorType, retry_after: float | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Provider & Model Definitions
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    model_id: str
    is_free_tier: bool = True
    context_window: int = 8192
    supports_tools: bool = True
    rpm_limit: int = 15
    rpd_limit: int = 1500


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    env_key: str
    default_models: list[ModelSpec] = field(default_factory=list)
    custom_headers: dict[str, str] = field(default_factory=dict)
    endpoint_path: str = "/chat/completions"

    def get_api_key(self) -> str | None:
        key = os.getenv(self.env_key, "").strip()
        return key if key else None

    def get_url(self) -> str:
        base = self.base_url.rstrip("/")
        path = self.endpoint_path.lstrip("/")
        return f"{base}/{path}"


PROVIDER_SPECS: dict[str, ProviderConfig] = {
    "gemini": ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        env_key="GEMINI_API_KEY",
        default_models=[
            ModelSpec("gemini-3.5-flash", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
            ModelSpec("gemini-3.8-flash", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
            ModelSpec("gemini-3.5-flash-lite", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
            ModelSpec("gemma-4-26b-a4b-it", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
            ModelSpec("gemma-4-31b-it", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
            ModelSpec("gemini-2.5-flash", is_free_tier=True, rpm_limit=15, rpd_limit=1500),
        ],
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        custom_headers={
            "HTTP-Referer": "https://github.com/AgentEvalOps/AgentEvalOps",
            "X-Title": "AgentEvalOps-Benchmark",
        },
        default_models=[
            ModelSpec("minimax/minimax-m3.1-flash:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("nvidia/nemotron-3.5-lightning:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("openrouter/free", is_free_tier=True, rpm_limit=10),
            ModelSpec("liquid/lfm-2.5-2.6b:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("google/gemma-4-26b-a4b-it:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("google/gemma-4-31b-it:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("nvidia/nemotron-3-ultra-550b-a55b:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("cohere/north-mini-code:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("poolside/laguna-s-2.1:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("poolside/laguna-xs-2.1:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("thinkingmachines/inkling-small:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("thinkingmachines/inkling:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("xingchen-agi/xing4.0-29b-a4b:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("meta-llama/llama-3.3-70b-instruct:free", is_free_tier=True, rpm_limit=10),
            ModelSpec("meta-llama/llama-3.1-8b-instruct:free", is_free_tier=True, rpm_limit=15),
        ],
    ),
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        default_models=[
            ModelSpec("openai/gpt-oss-120b", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("openai/gpt-oss-20b", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("qwen/qwen3.8-27b", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("llama-3.1-8b-instant", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("qwen/qwen3-32b", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("meta-llama/llama-4-scout-17b-16e-instruct", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("llama-3.3-70b-versatile", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("deepseek-r1-distill-llama-70b", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
            ModelSpec("mixtral-8x7b-32768", is_free_tier=True, rpm_limit=30, rpd_limit=14400),
        ],
    ),
    "mistral": ProviderConfig(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        default_models=[
            ModelSpec("ministral-3b-latest", is_free_tier=True, rpm_limit=10),
            ModelSpec("ministral-8b-latest", is_free_tier=True, rpm_limit=10),
            ModelSpec("open-mistral-nemo", is_free_tier=True, rpm_limit=10),
            ModelSpec("mistral-small-latest", is_free_tier=True, rpm_limit=10),
            ModelSpec("mistral-medium-latest", is_free_tier=True, rpm_limit=10),
            ModelSpec("mistral-large-latest", is_free_tier=True, rpm_limit=10),
            ModelSpec("codestral-latest", is_free_tier=True, rpm_limit=10),
        ],
    ),
    "nvidia": ProviderConfig(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key="NVIDIA_API_KEY",
        default_models=[
            ModelSpec("nvidia/nemotron-3.5-lightning-30b-a3b", is_free_tier=True, rpm_limit=40),
            ModelSpec("nv-mistralai/mistral-nemo-12b-instruct", is_free_tier=True, rpm_limit=40),
            ModelSpec("nvidia/nemotron-3.5-super-vl-120b-a12b", is_free_tier=True, rpm_limit=40),
            ModelSpec("nvidia/nemotron-3-ultra-550b-a55b", is_free_tier=True, rpm_limit=40),
            ModelSpec("qwen/qwen3-coder-480b", is_free_tier=True, rpm_limit=40),
            ModelSpec("minimaxai/minimax-m2.7", is_free_tier=True, rpm_limit=40),
            ModelSpec("deepseek-ai/deepseek-v4.1-flash", is_free_tier=True, rpm_limit=40),
            ModelSpec("deepseek-ai/deepseek-v4-flash", is_free_tier=True, rpm_limit=40),
            ModelSpec("meta/llama-4-maverick-17b-128e-instruct", is_free_tier=True, rpm_limit=40),
            ModelSpec("meta/muse-glimmer-30b", is_free_tier=True, rpm_limit=40),
            ModelSpec("z-ai/glm-5.3-flash", is_free_tier=True, rpm_limit=40),
            ModelSpec("poolside/laguna-xs-2.1", is_free_tier=True, rpm_limit=40),
            ModelSpec("meta/llama-3.1-8b-instruct", is_free_tier=True, rpm_limit=40),
        ],
    ),
    "cohere": ProviderConfig(
        name="cohere",
        base_url="https://api.cohere.com/compatibility/v1",
        env_key="COHERE_API_KEY",
        default_models=[
            ModelSpec("command-r7b-12-2024", is_free_tier=True, rpm_limit=20),
            ModelSpec("command-a-03-2025", is_free_tier=True, rpm_limit=20),
            ModelSpec("command-r-08-2024", is_free_tier=True, rpm_limit=20),
            ModelSpec("command-r-plus-08-2024", is_free_tier=True, rpm_limit=20),
        ],
    ),
    "cloudflare": ProviderConfig(
        name="cloudflare",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        env_key="CLOUDFLARE_API_KEY",
        default_models=[
            ModelSpec("@cf/meta/llama-3.2-3b-instruct", is_free_tier=True, rpm_limit=10),
            ModelSpec("@cf/meta/llama-3.1-8b-instruct", is_free_tier=True, rpm_limit=10),
            ModelSpec("@cf/mistralai/mistral-small-3.1-24b-instruct", is_free_tier=True, rpm_limit=10),
            ModelSpec("@cf/qwen/qwen2.5-coder-32b-instruct", is_free_tier=True, rpm_limit=10),
            ModelSpec("@cf/google/gemma-3-12b-it", is_free_tier=True, rpm_limit=10),
            ModelSpec("@cf/zai/glm-4.7-flash", is_free_tier=True, rpm_limit=10),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Rate-Limit & Header Utilities
# ---------------------------------------------------------------------------


def parse_retry_after(headers: dict[str, Any], default: float = 30.0) -> float:
    """Extract retry-after in seconds from response headers (supports seconds or HTTP-date)."""
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        # Check provider-specific reset headers
        for h in ("x-ratelimit-reset-requests", "x-ratelimit-reset", "ratelimit-reset"):
            val = headers.get(h)
            if val:
                try:
                    fval = float(val)
                    if fval > 1000000000:  # timestamp
                        return max(1.0, fval - time.time())
                    return max(1.0, fval)
                except ValueError:
                    pass
        return default

    try:
        return max(1.0, float(raw))
    except ValueError:
        pass

    try:
        parsed_tuple = email.utils.parsedate_to_datetime(raw)
        diff = (parsed_tuple.timestamp() - time.time())
        return max(1.0, diff)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Unified Provider Client
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    prompt_tokens: int | None
    completion_tokens: int | None
    resolved_model: str
    latency_ms: float
    raw_response: dict[str, Any]
    raw_message: dict[str, Any] = field(default_factory=dict)


class ProviderClient:
    """Manages chat completion calls across OpenAI-compatible providers."""

    def __init__(self, provider_name: str, config: ProviderConfig | None = None):
        self.provider_name = provider_name
        self.config = config or PROVIDER_SPECS.get(provider_name)
        if not self.config:
            raise ValueError(f"Unknown provider: {provider_name}")

    def is_configured(self) -> bool:
        """Check if environment credentials exist for this provider."""
        if self.provider_name == "cloudflare":
            return bool(os.getenv("CLOUDFLARE_API_KEY") and os.getenv("CLOUDFLARE_ACCOUNT_ID"))
        return self.config.get_api_key() is not None

    def get_headers(self) -> dict[str, str]:
        key = self.config.get_api_key()
        if not key:
            raise ProviderException(
                f"Missing API key for {self.provider_name} (expected {self.config.env_key}).",
                ErrorType.AUTH_FAILURE,
            )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.custom_headers)
        return headers

    def get_url(self) -> str:
        if self.provider_name == "cloudflare":
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "default")
            base = self.config.base_url.replace("{account_id}", account_id)
            return f"{base.rstrip('/')}/chat/completions"
        return self.config.get_url()

    def call_chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        timeout: float = 35.0,
    ) -> LLMResponse:
        """Execute chat completion request with structured tool definitions."""
        url = self.get_url()
        headers = self.get_headers()

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        # Normalize Cloudflare message content: Cloudflare strictly requires content to be a primitive string
        if self.provider_name == "cloudflare":
            serialized_messages = []
            for m in messages:
                m_copy = dict(m)
                c = m_copy.get("content")
                if c is None:
                    m_copy["content"] = ""
                elif isinstance(c, list):
                    parts = []
                    for part in c:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                        elif isinstance(part, str):
                            parts.append(part)
                        else:
                            parts.append(json.dumps(part))
                    m_copy["content"] = " ".join(parts)
                elif not isinstance(c, str):
                    m_copy["content"] = str(c)
                serialized_messages.append(m_copy)
            payload["messages"] = serialized_messages

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise ProviderException(
                f"Network timeout ({timeout}s) contacting {self.provider_name}: {exc}",
                ErrorType.NETWORK_TIMEOUT,
                retry_after=5.0,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderException(
                f"Connection error contacting {self.provider_name}: {exc}",
                ErrorType.NETWORK_TIMEOUT,
                retry_after=5.0,
            ) from exc

        lat = round((time.perf_counter() - t0) * 1000, 3)

        # Handle rate limits
        if resp.status_code == 429:
            retry_sec = parse_retry_after(dict(resp.headers))
            err_text = resp.text.lower()
            if any(term in err_text for term in ("daily", "day", "quota exceeded", "quota limit", "credit")):
                raise ProviderException(
                    f"{self.provider_name} daily quota reached: {resp.text[:200]}",
                    ErrorType.DAILY_QUOTA,
                    retry_after=retry_sec,
                )
            raise ProviderException(
                f"{self.provider_name} rate limit (429): {resp.text[:200]}",
                ErrorType.RATE_LIMIT,
                retry_after=retry_sec,
            )

        # Handle auth or permission failures
        if resp.status_code in (401, 403):
            raise ProviderException(
                f"{self.provider_name} authentication/authorization error ({resp.status_code}): {resp.text[:200]}",
                ErrorType.AUTH_FAILURE,
            )

        # Handle missing model
        if resp.status_code == 404:
            raise ProviderException(
                f"{self.provider_name} model not found ({model_id}): {resp.text[:200]}",
                ErrorType.MODEL_NOT_FOUND,
            )

        # Handle server errors
        if resp.status_code >= 500:
            raise ProviderException(
                f"{self.provider_name} server error ({resp.status_code}): {resp.text[:200]}",
                ErrorType.SERVER_ERROR,
                retry_after=5.0,
            )

        # Handle Groq model-level tool formatting / validation errors (HTTP 400 with tool_use_failed)
        if resp.status_code == 400 and self.provider_name == "groq":
            try:
                err_data = resp.json()
                err_obj = err_data.get("error", {})
                err_code = err_obj.get("code")
                err_msg = err_obj.get("message", "")
                failed_gen = err_obj.get("failed_generation")

                if err_code == "tool_use_failed" or any(p in err_msg.lower() for p in ("failed to parse tool call", "tool call validation failed", "parsing failed")):
                    tool_name = "unknown_tool"
                    raw_args = ""
                    if failed_gen:
                        name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', failed_gen)
                        if name_match:
                            tool_name = name_match.group(1)
                        args_match = re.search(r'["\']arguments["\']\s*:\s*(.*)', failed_gen, re.DOTALL)
                        if args_match:
                            raw_args = args_match.group(1).rstrip('}')
                        else:
                            raw_args = failed_gen
                    elif "attempted to call tool" in err_msg:
                        match = re.search(r"attempted to call tool ['\"]([^'\"]+)['\"]", err_msg)
                        if match:
                            tool_name = match.group(1)
                            raw_args = "{}"

                    return LLMResponse(
                        content=failed_gen if not raw_args else None,
                        tool_calls=[{
                            "id": "call_groq_malformed",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": raw_args,
                            },
                        }],
                        prompt_tokens=None,
                        completion_tokens=None,
                        resolved_model=model_id,
                        latency_ms=lat,
                        raw_response=err_data,
                        raw_message={"role": "assistant", "error": err_msg},
                    )
            except Exception:
                pass

        if resp.status_code != 200:
            raise ProviderException(
                f"{self.provider_name} API error HTTP {resp.status_code}: {resp.text[:200]}",
                ErrorType.MALFORMED_RESPONSE,
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderException(
                f"{self.provider_name} returned non-JSON body: {resp.text[:200]}",
                ErrorType.MALFORMED_RESPONSE,
            ) from exc

        choices = data.get("choices", [])
        if not choices:
            raise ProviderException(
                f"{self.provider_name} returned empty choices array.",
                ErrorType.MALFORMED_RESPONSE,
            )

        message = choices[0].get("message", {})
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        # Resolve actual model ID (OpenRouter returns resolved model in header or field)
        resolved_model = resp.headers.get("x-openrouter-model") or data.get("model") or model_id

        tool_calls = message.get("tool_calls")
        content = message.get("content")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            resolved_model=resolved_model,
            latency_ms=lat,
            raw_response=data,
            raw_message=dict(message),
        )

    def smoke_test(self, model_id: str) -> tuple[bool, str]:
        """
        Execute a minimal smoke test verifying access, model ID, and tool-calling support.
        Does NOT spend significant quota (uses a minimal 1-turn prompt).
        """
        if not self.is_configured():
            return False, f"Provider {self.provider_name} credentials missing ({self.config.env_key})."

        messages = [
            {"role": "user", "content": "What is 7 * 6? Use the calculate tool to compute the answer."}
        ]
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate arithmetic expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]

        try:
            resp = self.call_chat_completion(
                model_id=model_id,
                messages=messages,
                tools=test_tools,
                temperature=0.0,
                timeout=20.0,
            )
            has_tool_call = bool(resp.tool_calls)
            return True, f"Verified (model={resp.resolved_model}, tool_calling={has_tool_call})"
        except ProviderException as exc:
            return False, f"Smoke test failed ({exc.error_type.value}): {exc}"
        except Exception as exc:
            return False, f"Smoke test error: {exc}"
