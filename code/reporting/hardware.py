"""Hardware and software record attached to every A2 experiment summary (Part A, F)."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import psutil


def _cpu_name() -> str:
    if sys.platform == "win32":
        import subprocess

        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.TimeoutExpired):
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def hardware_record(ollama: Any | None = None) -> dict[str, Any]:
    """CPU, RAM, OS, Python and (if given an OllamaClient) the inference server and model digests."""
    rec: dict[str, Any] = {
        "cpu": _cpu_name(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu_used": False,
        "inference": "Ollama in Docker (CPU only, no --gpus); see MODELS.md",
    }
    if ollama is not None:
        try:
            rec["ollama_version"] = ollama.server_version()
            rec["num_thread"] = ollama.num_thread
            rec["num_ctx"] = ollama.num_ctx
            rec["model_digests"] = {m: ollama.model_digest(m) for m in ("qwen2.5:3b", "qwen2.5:1.5b")}
        except Exception as exc:  # server down: record why
            rec["ollama_error"] = str(exc)
    return rec
