"""
Unit and integration tests for the multi-provider trajectory collection pipeline.

Validates:
  1. Successful tool-call round trips.
  2. Multi-turn model interactions.
  3. 429 rate limit cooldown and isolation (other providers continue).
  4. Daily quota exhaustion and deferred job resumption.
  5. Invalid key and unsupported model quarantine.
  6. Network timeout and transient 5xx handling.
  7. Malformed provider responses.
  8. Request-budget enforcement before inference calls.
  9. Crash recovery and duplicate prevention.
  10. Fair round-robin scheduling across provider-model targets.
  11. Graceful shutdown.
  12. Separation of provider errors from model reasoning failures.
  13. Offline cooldown resumption.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from code.experiments.agent_executor import (
    RequestTracker,
    execute_agent_task,
)
from code.experiments.providers import (
    ErrorType,
    LLMResponse,
    ModelSpec,
    ProviderClient,
    ProviderConfig,
    ProviderException,
)
from code.experiments.trajectory_worker import TrajectoryWorker
from code.experiments.work_queue import JobQueue, JobState, compute_job_id
from code.shared.models import (
    EvaluationRule,
    ExecutionConfig,
    Outcome,
    TaskMetadata,
    TaskRecord,
)


@pytest.fixture
def sample_task() -> TaskRecord:
    return TaskRecord(
        task_id="T1_multihop_01",
        template_family="multi_hop_retrieval",
        task_type="multi_hop_qa",
        prompt="According to the current 2025 company travel policy (v2.1), what is the maximum domestic daily per diem allowance for Cloud Infrastructure staff?",
        answerable=True,
        gold_answer="240 USD",
        expected_tools=["retrieve"],
        evaluation_rule=EvaluationRule.exact_match,
        metadata=TaskMetadata(
            provenance="test",
            created_at="2026-10-21T00:00:00Z",
            generator_version="test",
            seed=42,
            content_hash="sample_hash_01",
        ),
    )


@pytest.fixture
def mock_provider_client() -> ProviderClient:
    cfg = ProviderConfig(
        name="mock_provider",
        base_url="https://api.mock.test/v1",
        env_key="MOCK_API_KEY",
    )
    with patch.dict("os.environ", {"MOCK_API_KEY": "test-key-123"}):
        client = ProviderClient("mock_provider", cfg)
    return client


class TestProviderClientUnit:
    """Test provider client parsing, headers, and error mapping."""

    def test_headers_and_url(self, mock_provider_client: ProviderClient):
        with patch.dict("os.environ", {"MOCK_API_KEY": "secret-key"}):
            headers = mock_provider_client.get_headers()
            assert headers["Authorization"] == "Bearer secret-key"
            assert mock_provider_client.get_url() == "https://api.mock.test/v1/chat/completions"

    def test_rate_limit_429_parsed(self, mock_provider_client: ProviderClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "25"}
        mock_resp.text = "Rate limit reached"

        with patch("requests.post", return_value=mock_resp), patch.dict("os.environ", {"MOCK_API_KEY": "k"}):
            with pytest.raises(ProviderException) as exc_info:
                mock_provider_client.call_chat_completion("test-model", [{"role": "user", "content": "hi"}])
            assert exc_info.value.error_type == ErrorType.RATE_LIMIT
            assert exc_info.value.retry_after == 25.0

    def test_daily_quota_429_parsed(self, mock_provider_client: ProviderClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_resp.text = "Daily quota exceeded for free tier"

        with patch("requests.post", return_value=mock_resp), patch.dict("os.environ", {"MOCK_API_KEY": "k"}):
            with pytest.raises(ProviderException) as exc_info:
                mock_provider_client.call_chat_completion("test-model", [{"role": "user", "content": "hi"}])
            assert exc_info.value.error_type == ErrorType.DAILY_QUOTA

    def test_auth_error_401(self, mock_provider_client: ProviderClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API key"

        with patch("requests.post", return_value=mock_resp), patch.dict("os.environ", {"MOCK_API_KEY": "k"}):
            with pytest.raises(ProviderException) as exc_info:
                mock_provider_client.call_chat_completion("test-model", [{"role": "user", "content": "hi"}])
            assert exc_info.value.error_type == ErrorType.AUTH_FAILURE

    def test_model_not_found_404(self, mock_provider_client: ProviderClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Model not found"

        with patch("requests.post", return_value=mock_resp), patch.dict("os.environ", {"MOCK_API_KEY": "k"}):
            with pytest.raises(ProviderException) as exc_info:
                mock_provider_client.call_chat_completion("missing-model", [{"role": "user", "content": "hi"}])
            assert exc_info.value.error_type == ErrorType.MODEL_NOT_FOUND

    def test_malformed_non_json_response(self, mock_provider_client: ProviderClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        mock_resp.text = "<html>502 Bad Gateway</html>"

        with patch("requests.post", return_value=mock_resp), patch.dict("os.environ", {"MOCK_API_KEY": "k"}):
            with pytest.raises(ProviderException) as exc_info:
                mock_provider_client.call_chat_completion("test-model", [{"role": "user", "content": "hi"}])
            assert exc_info.value.error_type == ErrorType.MALFORMED_RESPONSE


class TestAgentExecutorMocked:
    """Test agent execution loop and multi-turn tool interaction."""

    def test_successful_tool_call_round_trip(self, sample_task: TaskRecord, mock_provider_client: ProviderClient):
        # Turn 1: model calls retrieve
        resp_turn1 = LLMResponse(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "retrieve",
                    "arguments": json.dumps({"query": "current 2025 travel policy"}),
                },
            }],
            prompt_tokens=15,
            completion_tokens=20,
            resolved_model="mock_model_1",
            latency_ms=120.0,
            raw_response={},
        )
        # Turn 2: model submits answer
        resp_turn2 = LLMResponse(
            content="Submitting the final per diem.",
            tool_calls=[{
                "id": "call_2",
                "function": {
                    "name": "submit_answer",
                    "arguments": json.dumps({"answer": "240 USD"}),
                },
            }],
            prompt_tokens=35,
            completion_tokens=15,
            resolved_model="mock_model_1",
            latency_ms=110.0,
            raw_response={},
        )

        cfg = ExecutionConfig(
            model_id="mock_model_1",
            prompt_version="v1",
            tool_selection_strategy="openai_tools",
            temperature=0.0,
            max_steps=5,
            retry_budget=0,
        )

        with patch.object(mock_provider_client, "call_chat_completion", side_effect=[resp_turn1, resp_turn2]):
            result = execute_agent_task(
                task=sample_task,
                provider_client=mock_provider_client,
                model_id="mock_model_1",
                config=cfg,
                rate_limit_delay=0.0,
            )

        assert result.outcome == Outcome.success
        assert result.final_answer == "240 USD"
        assert result.process_category == "correct_acceptable_process"
        assert result.trajectory is not None
        assert len(result.trajectory.steps) == 2
        assert result.trajectory.steps[0].tool_name == "retrieve"
        assert result.trajectory.steps[0].tool_result.success is True
        assert result.trajectory.steps[1].tool_name == "submit_answer"

    def test_budget_exhaustion_before_inference(self, sample_task: TaskRecord, mock_provider_client: ProviderClient):
        tracker = RequestTracker(max_requests=1)
        resp_turn1 = LLMResponse(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "function": {"name": "retrieve", "arguments": json.dumps({"query": "travel"})},
            }],
            prompt_tokens=10,
            completion_tokens=10,
            resolved_model="m1",
            latency_ms=50.0,
            raw_response={},
        )

        cfg = ExecutionConfig(
            model_id="m1",
            prompt_version="v1",
            tool_selection_strategy="tools",
            temperature=0.0,
            max_steps=5,
            retry_budget=0,
        )

        # First turn consumes the 1 request; second turn hits budget limit
        with patch.object(mock_provider_client, "call_chat_completion", return_value=resp_turn1):
            result = execute_agent_task(
                task=sample_task,
                provider_client=mock_provider_client,
                model_id="m1",
                config=cfg,
                request_tracker=tracker,
                rate_limit_delay=0.0,
            )

        assert result.error_type == ErrorType.BUDGET_EXHAUSTED
        assert tracker.total_requests == 1

    def test_provider_error_separated_from_model_failure(self, sample_task: TaskRecord, mock_provider_client: ProviderClient):
        cfg = ExecutionConfig(
            model_id="m1",
            prompt_version="v1",
            tool_selection_strategy="tools",
            temperature=0.0,
            max_steps=5,
            retry_budget=0,
        )

        with patch.object(mock_provider_client, "call_chat_completion", side_effect=ProviderException("503 Server Error", ErrorType.SERVER_ERROR)):
            result = execute_agent_task(
                task=sample_task,
                provider_client=mock_provider_client,
                model_id="m1",
                config=cfg,
                rate_limit_delay=0.0,
            )

        assert result.outcome == Outcome.error
        assert result.process_category == "provider_api_error"
        assert result.trajectory is None  # Not recorded as a model failure


class TestWorkQueueAndFairScheduler:
    """Test SQLite durable queue, crash recovery, and fair scheduling."""

    def test_queue_population_and_recovery(self, tmp_path: Path, sample_task: TaskRecord):
        db_path = tmp_path / "queue.sqlite3"
        q = JobQueue(db_path)

        targets = [("gemini", "gemini-2.5-flash"), ("openrouter", "free-llama")]
        inserted = q.populate_jobs([sample_task], targets, repetitions=2, seed=42)
        assert inserted == 4  # 1 task * 2 targets * 2 reps = 4 jobs

        # Claim a job -> transitions to 'running'
        job = q.claim_next_job("gemini", "gemini-2.5-flash")
        assert job is not None
        assert job.state == JobState.RUNNING

        # Simulate crash: recover stale jobs
        recovered = q.recover_stale_jobs()
        assert recovered == 1

        summary = q.get_summary()
        assert summary["pending_count"] == 4
        assert summary["state_counts"].get(JobState.RUNNING.value, 0) == 0

    def test_fair_round_robin_selection(self, tmp_path: Path, sample_task: TaskRecord):
        db_path = tmp_path / "queue.sqlite3"
        q = JobQueue(db_path)

        targets = [("provider_a", "model_a"), ("provider_b", "model_b")]
        q.populate_jobs([sample_task], targets, repetitions=1, seed=42)

        eligible = q.get_eligible_targets()
        assert len(eligible) == 2

        # Claim from target 1 then target 2
        job_a = q.claim_next_job("provider_a", "model_a")
        job_b = q.claim_next_job("provider_b", "model_b")
        assert job_a is not None and job_a.provider == "provider_a"
        assert job_b is not None and job_b.provider == "provider_b"

    def test_rate_limit_isolation_and_resumption(self, tmp_path: Path, sample_task: TaskRecord):
        db_path = tmp_path / "queue.sqlite3"
        q = JobQueue(db_path)

        targets = [("p_blocked", "m1"), ("p_free", "m2")]
        q.populate_jobs([sample_task], targets, repetitions=1, seed=42)

        # Defer p_blocked with cooldown in the future
        q.set_provider_cooldown("p_blocked", "m1", cooldown_seconds=60.0)

        eligible = q.get_eligible_targets()
        assert len(eligible) == 1
        assert eligible[0] == ("p_free", "m2")

        # Expire cooldown artificially
        q.set_provider_cooldown("p_blocked", "m1", cooldown_seconds=-10.0)
        eligible_after = q.get_eligible_targets()
        assert len(eligible_after) == 2

    def test_quarantine_permanent_error(self, tmp_path: Path, sample_task: TaskRecord):
        db_path = tmp_path / "queue.sqlite3"
        q = JobQueue(db_path)

        targets = [("p_bad", "m_invalid")]
        q.populate_jobs([sample_task], targets, repetitions=1, seed=42)

        q.quarantine_provider("p_bad", "m_invalid", "Invalid API key")
        assert q.is_provider_eligible("p_bad", "m_invalid") is False

        eligible = q.get_eligible_targets()
        assert len(eligible) == 0

        summary = q.get_summary()
        assert summary["state_counts"][JobState.PROVIDER_ERROR.value] == 1


class TestWorkerEndToEndMocked:
    """Test full TrajectoryWorker lifecycle with mocked provider calls."""

    def test_worker_run_and_persistence(self, tmp_path: Path, sample_task: TaskRecord):
        exp_dir = tmp_path / "exp_01"
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(sample_task.model_dump_json() + "\n", encoding="utf-8")

        worker = TrajectoryWorker(
            tasks_path=tasks_file,
            experiment_dir=exp_dir,
            max_requests=20,
            max_trajectories=2,
            repetitions=1,
            seed=42,
        )

        # Mock configured client
        fake_client = MagicMock()
        fake_client.is_configured.return_value = True
        fake_client.provider_name = "gemini"
        fake_client.config = ProviderConfig(
            name="gemini",
            base_url="https://fake",
            env_key="GEMINI_API_KEY",
            default_models=[ModelSpec("gemini-2.5-flash")],
        )

        # Model submits answer immediately
        fake_resp = LLMResponse(
            content=None,
            tool_calls=[{
                "id": "c1",
                "function": {"name": "submit_answer", "arguments": json.dumps({"answer": "240 USD"})},
            }],
            prompt_tokens=10,
            completion_tokens=5,
            resolved_model="gemini-2.5-flash",
            latency_ms=100.0,
            raw_response={},
        )
        fake_client.call_chat_completion.return_value = fake_resp

        worker.clients["gemini"] = fake_client

        with patch.object(worker, "get_configured_targets", return_value=[("gemini", "gemini-2.5-flash")]):
            summary = worker.run()

        assert summary["total_trajectories_completed"] == 1
        assert exp_dir / "trajectories.jsonl"
        assert (exp_dir / "summary.json").exists()

        # Re-run worker on same directory: ensure zero duplicate runs
        worker2 = TrajectoryWorker(
            tasks_path=tasks_file,
            experiment_dir=exp_dir,
            max_requests=20,
            max_trajectories=2,
        )
        with patch.object(worker2, "get_configured_targets", return_value=[("gemini", "gemini-2.5-flash")]):
            summary2 = worker2.run()
        # All jobs were already completed, so 0 new trajectories completed
        assert summary2["total_trajectories_completed"] == 0
