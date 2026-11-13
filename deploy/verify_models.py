"""Fail the image build if a pulled model's digest differs from MODELS.md."""

from __future__ import annotations

import json
import sys
import urllib.request

EXPECTED = {
    "qwen2.5:3b": "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b",
    "qwen2.5:1.5b": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
}

with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as resp:
    installed = {m["name"]: m["digest"] for m in json.load(resp)["models"]}
bad = {m: installed.get(m) for m, d in EXPECTED.items() if installed.get(m) != d}
if bad:
    sys.exit(f"model digest mismatch: {bad}")
print("model digests verified:", ", ".join(EXPECTED))
