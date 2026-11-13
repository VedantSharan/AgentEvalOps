"""Tests for the deployment bundle, Docker files and CI configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from code.service.app import FAULT_PROFILES, mask

REPO = Path(__file__).resolve().parents[1]


def test_bundle_matches_the_dvc_workspace() -> None:
    from deploy.bundle import bundle_paths  # noqa: PLC0415

    if not (REPO / "data" / "splits" / "train.csv").exists():
        pytest.skip("no DVC workspace")
    for rel in bundle_paths():
        assert (REPO / "deploy" / rel).read_bytes() == (REPO / rel).read_bytes(), rel


def test_bundled_gold_set_passes_its_checksums() -> None:
    from code.evaluation.gold import verify_gold

    verify_gold(REPO / "deploy" / "data" / "splits" / "gold_test")


def test_dockerfile_pins_images_and_runs_offline() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "ollama/ollama:0.40.8@sha256:" in text and "python:3.10-slim-trixie@sha256:" in text
    assert "AGENTEVALOPS_OFFLINE=1" in text and "linux/amd64" in text


def test_ci_runs_tests_validation_build_and_offline_smoke() -> None:
    ci = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    steps = " ".join(str(s.get("run", "")) for job in ci["jobs"].values() for s in job["steps"])
    for needed in ("pytest", "run_validation_evidence --sample", "docker build", "--network none"):
        assert needed in steps
    listed = [ln.strip() for ln in (REPO / "deploy" / "ci_tests.txt").read_text().splitlines()
              if ln.strip() and not ln.startswith("#")]
    assert all((REPO / p).exists() for p in listed)


def test_personal_data_is_masked_in_logs() -> None:
    assert mask("mail a.b@c.com or +91 98765 43210") == "mail <email> or <phone>"
    assert set(FAULT_PROFILES) >= {"none", "information_outage", "format_change"}
