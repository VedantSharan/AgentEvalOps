"""
Persistent, multi-provider trajectory collection worker for AgentEvalOps.

Orchestrates:
  - Persistent SQLite work queue
  - Fair round-robin scheduling across providers and models
  - Cooldown management (429 rate limits, daily quotas, auth quarantines)
  - Hard request budgets and duration limits
  - Atomic trajectory persistence to JSONL
  - Graceful shutdown (SIGINT / SIGTERM)
  - Comprehensive progress and diagnostic reporting
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from code.experiments.agent_executor import (
    ExecutionResult,
    RequestTracker,
    execute_agent_task,
)
from code.experiments.providers import (
    ErrorType,
    ProviderClient,
    ProviderConfig,
    PROVIDER_SPECS,
)
from code.experiments.work_queue import JobQueue, JobRecord, JobState
from code.shared.models import ExecutionConfig, TaskRecord, TrajectoryRecord
from code.tasks.revised_generator import load_tasks_jsonl

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrajectoryWorker")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TASKS_PATH = _REPO_ROOT / "data" / "processed" / "tasks_revised.jsonl"
DEFAULT_EXP_DIR = _REPO_ROOT / "data" / "experiments" / "default"


class TrajectoryWorker:
    """Long-running worker coordinating fair multi-provider trajectory collection."""

    def __init__(
        self,
        tasks_path: Path = DEFAULT_TASKS_PATH,
        experiment_dir: Path = DEFAULT_EXP_DIR,
        max_requests: int = 100,
        max_trajectories: int | None = None,
        duration_hours: float | None = None,
        repetitions: int = 1,
        seed: int = 42,
        model_filters: list[str] | None = None,
        max_consecutive_failures: int = 20,
    ):
        self.tasks_path = Path(tasks_path)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.experiment_dir / "work_queue.sqlite3"
        self.trajectories_path = self.experiment_dir / "trajectories.jsonl"
        self.summary_path = self.experiment_dir / "summary.json"

        self.queue = JobQueue(self.db_path)
        self.request_tracker = RequestTracker(max_requests=max_requests)
        self.max_trajectories = max_trajectories
        self.duration_seconds = (duration_hours * 3600.0) if duration_hours else None
        self.repetitions = repetitions
        self.seed = seed
        self.model_filters = model_filters
        self.max_consecutive_failures = max_consecutive_failures

        self.shutdown_requested = False
        self._setup_signals()

        self.clients: dict[str, ProviderClient] = {}
        for pname in PROVIDER_SPECS:
            self.clients[pname] = ProviderClient(pname)

    def _setup_signals(self) -> None:
        """Register graceful signal handlers."""
        def _handler(signum: int, frame: Any) -> None:
            logger.info("Shutdown requested. Finishing active step and saving state...")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def get_configured_targets(self) -> list[tuple[str, str]]:
        """List eligible (provider, model_id) targets with credentials configured."""
        targets: list[tuple[str, str]] = []
        for pname, client in self.clients.items():
            if not client.is_configured():
                continue
            for mod in client.config.default_models:
                if not mod.is_free_tier:
                    continue
                tag = f"{pname}:{mod.model_id}"
                if self.model_filters:
                    if not any(f in tag or f == mod.model_id for f in self.model_filters):
                        continue
                targets.append((pname, mod.model_id))
        return targets

    def initialize_queue(self, tasks: list[TaskRecord]) -> int:
        """Initialize work queue with tasks grid."""
        targets = self.get_configured_targets()
        if not targets:
            logger.warning("No configured providers found! Set environment keys (e.g. GEMINI_API_KEY).")
            return 0

        # Recover any crashed jobs left in 'running'
        recovered = self.queue.recover_stale_jobs()
        if recovered:
            logger.info(f"Recovered {recovered} stale jobs back to 'pending'.")

        # Reconcile with existing trajectories
        existing_run_ids = set()
        if self.trajectories_path.exists():
            with self.trajectories_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            existing_run_ids.add(record.get("run_id"))
                        except Exception:
                            pass
        reconciled = self.queue.reconcile_with_trajectories(existing_run_ids)
        if reconciled:
            logger.info(f"Reconciled {reconciled} jobs with existing trajectories.")

        inserted = self.queue.populate_jobs(
            tasks=tasks,
            targets=targets,
            repetitions=self.repetitions,
            experiment_hash=f"seed_{self.seed}",
            seed=self.seed,
        )
        logger.info(f"Initialized queue: {inserted} new jobs added across {len(targets)} targets.")
        return inserted

    def append_trajectory(self, trajectory: TrajectoryRecord) -> None:
        """Atomically append completed trajectory to JSONL."""
        with self.trajectories_path.open("a", encoding="utf-8") as fh:
            fh.write(trajectory.model_dump_json() + "\n")
            fh.flush()

    def run(self) -> dict[str, Any]:
        """Main execution loop cycling fairly across provider/model targets."""
        tasks = load_tasks_jsonl(self.tasks_path)
        task_map = {t.task_id: t for t in tasks}

        self.initialize_queue(tasks)
        start_time = time.time()
        trajectories_completed = 0
        consecutive_failures = 0

        logger.info("Starting trajectory collection loop...")

        config = ExecutionConfig(
            model_id="dynamic",
            prompt_version="v1",
            tool_selection_strategy="openai_function_calling",
            temperature=0.0,
            max_steps=5,
            retry_budget=1,
        )

        while not self.shutdown_requested:
            # Check duration limit
            if self.duration_seconds and (time.time() - start_time) >= self.duration_seconds:
                logger.info("Configured duration limit reached.")
                break

            # Check trajectory count limit
            if self.max_trajectories and trajectories_completed >= self.max_trajectories:
                logger.info(f"Configured trajectory limit reached ({trajectories_completed}).")
                break

            # Refresh any rate-limited jobs whose cooldown expired
            self.queue.refresh_deferred_jobs()

            # Get eligible targets with pending work
            eligible_targets = self.queue.get_eligible_targets()
            if not eligible_targets:
                # Check if all jobs are done or if any are in cooldown
                summary = self.queue.get_summary()
                pending = summary["pending_count"]
                deferred = summary["deferred_count"]

                if pending == 0 and deferred == 0:
                    logger.info("All jobs in queue completed or exhausted!")
                    break

                if deferred > 0:
                    logger.info(f"All active targets in cooldown. Sleeping 5s ({deferred} deferred jobs)...")
                    for _ in range(5):
                        if self.shutdown_requested:
                            break
                        time.sleep(1.0)
                    continue
                else:
                    logger.info("No eligible jobs remain.")
                    break

            # Cycle fairly through eligible targets
            progress_made_in_cycle = False
            for provider, model_id in eligible_targets:
                if self.shutdown_requested:
                    break

                if self.max_trajectories and trajectories_completed >= self.max_trajectories:
                    logger.info(f"Configured trajectory limit reached ({trajectories_completed}). Stopping collection.")
                    break

                job = self.queue.claim_next_job(provider, model_id)
                if not job:
                    continue

                task = task_map.get(job.task_id)
                if not task:
                    self.queue.mark_job_failed(job.job_id, f"Task {job.task_id} not found in task_map.", error_type="agent_execution_error", is_permanent=False)
                    continue

                logger.info(f"Executing [{job.provider}:{job.model_id}] on {job.task_id} (rep {job.repetition})...")
                client = self.clients[provider]
                config.model_id = model_id

                t0_job = datetime.now(timezone.utc).isoformat()
                result = execute_agent_task(
                    task=task,
                    provider_client=client,
                    model_id=model_id,
                    config=config,
                    request_tracker=self.request_tracker,
                )
                t1_job = datetime.now(timezone.utc).isoformat()

                # Handle outcome
                if result.error_type == ErrorType.RATE_LIMIT:
                    cd = result.retry_after or 30.0
                    logger.warning(f"Rate limit on {provider}:{model_id}. Setting cooldown {cd:.1f}s.")
                    self.queue.set_provider_cooldown(provider, model_id, cd, result.provider_error or "429")
                    self.queue.mark_deferred_rate_limit(job.job_id, cd, result.provider_error or "429", error_type="rate_limit")
                elif result.error_type == ErrorType.DAILY_QUOTA:
                    cd = result.retry_after or 86400.0
                    logger.warning(f"Daily quota reached for {provider}:{model_id}. Deferring {cd:.1f}s.")
                    self.queue.set_provider_cooldown(provider, model_id, cd, result.provider_error or "Daily quota")
                    self.queue.mark_deferred_daily_quota(job.job_id, cd, result.provider_error or "Daily quota")
                elif result.error_type in (ErrorType.AUTH_FAILURE, ErrorType.MODEL_NOT_FOUND):
                    logger.error(f"Quarantining {provider}:{model_id} due to permanent error: {result.provider_error}")
                    self.queue.quarantine_provider(provider, model_id, result.provider_error or "Auth/Model error")
                    self.queue.mark_job_failed(job.job_id, result.provider_error or "Auth error", error_type="provider_error", is_permanent=True)
                elif result.error_type == ErrorType.BUDGET_EXHAUSTED:
                    logger.warning("Global request cap reached! Halting run.")
                    self.queue.mark_deferred_rate_limit(job.job_id, 3600.0, "Budget exhausted", error_type="rate_limit")
                    self.shutdown_requested = True
                    break
                elif result.trajectory:
                    # Successful or completed trajectory
                    consecutive_failures = 0
                    self.append_trajectory(result.trajectory)
                    self.queue.mark_completed(
                        job_id=job.job_id,
                        run_id=result.trajectory.run_id,
                        final_answer=result.final_answer,
                        outcome=result.outcome.value,
                        process_category=result.process_category,
                    )
                    trajectories_completed += 1
                    progress_made_in_cycle = True
                    logger.info(
                        f"Completed {job.task_id} -> {result.outcome.value} "
                        f"(answer={result.final_answer!r}, cat={result.process_category})"
                    )
                    if self.max_trajectories and trajectories_completed >= self.max_trajectories:
                        logger.info(f"Configured trajectory limit reached ({trajectories_completed}). Stopping collection.")
                        break
                elif result.error_type in (ErrorType.SERVER_ERROR, ErrorType.NETWORK_TIMEOUT, ErrorType.MALFORMED_RESPONSE) or result.process_category == "provider_api_error":
                    # Transient provider/API error with bounded retry
                    if job.attempts < job.max_attempts:
                        cd = result.retry_after or 15.0
                        logger.warning(
                            f"Transient provider error on {provider}:{model_id} (attempt {job.attempts}/{job.max_attempts}): "
                            f"{result.provider_error}. Deferring {cd:.1f}s."
                        )
                        self.queue.set_provider_cooldown(provider, model_id, cd, result.provider_error or "Server error")
                        self.queue.mark_deferred_rate_limit(job.job_id, cd, result.provider_error or "Transient provider error", error_type="provider_error")
                    else:
                        logger.error(f"Provider retries exhausted on {job.job_id}: {result.provider_error}")
                        self.queue.mark_job_failed(job.job_id, result.provider_error or "Provider retries exhausted", error_type="provider_error", is_permanent=True)
                else:
                    # General execution error / unhandled failure
                    logger.warning(f"Execution failed on {job.job_id}: {result.provider_error}")
                    self.queue.mark_job_failed(job.job_id, result.provider_error or "Execution failure", error_type="agent_execution_error", is_permanent=False)

                if not result.trajectory:
                    consecutive_failures += 1
                    if self.max_consecutive_failures and consecutive_failures >= self.max_consecutive_failures:
                        logger.error(
                            f"{consecutive_failures} continuous attempts failed to generate a valid trajectory! "
                            "Pausing worker for diagnostic analysis."
                        )
                        self.shutdown_requested = True
                        break

                # Log attempt
                self.queue.log_attempt(
                    job_id=job.job_id,
                    attempt_number=job.attempts,
                    started_at=t0_job,
                    finished_at=t1_job,
                    status=result.outcome.value if result.trajectory else "error",
                    error=result.provider_error,
                    requests_made=result.requests_made,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    duration_ms=result.duration_ms,
                )

        summary = self.export_summary(trajectories_completed, time.time() - start_time)
        logger.info(f"Worker finished. Summary written to {self.summary_path}.")
        return summary

    def export_summary(self, completed_count: int, runtime_sec: float) -> dict[str, Any]:
        """Compile and save progress summary."""
        queue_summary = self.queue.get_summary()
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(runtime_sec, 2),
            "total_trajectories_completed": completed_count,
            "total_requests_made": self.request_tracker.total_requests,
            "max_requests_cap": self.request_tracker.max_requests,
            "queue_state": queue_summary,
        }
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    def print_provider_status(self) -> None:
        """Inspect and display credentials and eligibility for all candidate providers."""
        print("\n" + "=" * 70)
        print("AGENT EVAL OPS - PROVIDER AVAILABILITY & CONFIGURATION")
        print("=" * 70)
        for pname, config in PROVIDER_SPECS.items():
            client = self.clients[pname]
            has_key = client.is_configured()
            if pname == "cloudflare" and os.getenv("CLOUDFLARE_API_KEY") and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
                status_str = "PARTIALLY CONFIGURED (Key present, but CLOUDFLARE_ACCOUNT_ID missing)"
            elif has_key:
                status_str = "CONFIGURED (Key present in env)"
            else:
                status_str = "NOT CONFIGURED (Key missing)"
            print(f"Provider: {pname.upper():<12} | Env: {config.env_key:<20} | Status: {status_str}")
            for mod in config.default_models:
                tier_str = "Free" if mod.is_free_tier else "Paid"
                print(f"   - Model: {mod.model_id:<40} (Tier: {tier_str}, RPM: {mod.rpm_limit})")
        print("=" * 70 + "\n")

    def run_smoke_test(self, limit_tasks: int = 2) -> None:
        """Execute a quick smoke test on configured providers."""
        print("\n" + "=" * 70)
        print(f"RUNNING LIVE SMOKE TEST (Max {limit_tasks} tasks)...")
        print("=" * 70)
        targets = self.get_configured_targets()
        if not targets:
            print("No configured providers found! Set environment variables first.")
            return

        for provider, model_id in targets:
            client = self.clients[provider]
            print(f"\nTesting provider: {provider} with model: {model_id} ...")
            ok, msg = client.smoke_test(model_id)
            status = "PASSED" if ok else "FAILED"
            print(f"[{status}] {msg}")
        print("\nSmoke test complete.\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent AgentEvalOps Trajectory Collection Pipeline.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH, help="Path to tasks.jsonl dataset.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXP_DIR, help="Output directory for experiment.")
    parser.add_argument("--duration-hours", type=float, default=None, help="Maximum execution duration in hours.")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Maximum number of trajectories to complete.")
    parser.add_argument("--max-requests", type=int, default=100, help="Hard cap on total API requests across all models.")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per task (default: 1).")
    parser.add_argument("--models", nargs="*", default=None, help="Filter to specific models (e.g. gemini-2.5-flash).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic task shuffling.")
    parser.add_argument("--max-consecutive-failures", type=int, default=20, help="Pause if N consecutive attempts fail.")
    parser.add_argument("--list-providers", action="store_true", help="List provider configurations and exit.")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal live smoke test on configured providers.")
    parser.add_argument("--status", action="store_true", help="Print current experiment queue status and exit.")

    args = parser.parse_args()

    worker = TrajectoryWorker(
        tasks_path=args.tasks,
        experiment_dir=args.experiment_dir,
        max_requests=args.max_requests,
        max_trajectories=args.max_trajectories,
        duration_hours=args.duration_hours,
        repetitions=args.repetitions,
        seed=args.seed,
        model_filters=args.models,
        max_consecutive_failures=args.max_consecutive_failures,
    )

    if args.list_providers:
        worker.print_provider_status()
        sys.exit(0)

    if args.smoke_test:
        worker.run_smoke_test(limit_tasks=2)
        sys.exit(0)

    if args.status:
        summary = worker.queue.get_summary()
        print("\n" + "=" * 70)
        print(f"EXPERIMENT QUEUE STATUS: {args.experiment_dir}")
        print("=" * 70)
        print(f"Total jobs:      {summary['total_jobs']}")
        print(f"Completed jobs:  {summary['completed_count']}")
        print(f"Pending jobs:    {summary['pending_count']}")
        print(f"Deferred jobs:   {summary['deferred_count']}")
        print(f"Failed/Error:    {summary['error_count']}")
        print("\nState breakdown:")
        for st, cnt in summary["state_counts"].items():
            print(f"  - {st:<22}: {cnt}")
        print("\nOutcomes for completed jobs:")
        for out, cnt in summary["outcomes"].items():
            print(f"  - {out:<22}: {cnt}")
        print("\nProcess categories:")
        for cat, cnt in summary["process_categories"].items():
            print(f"  - {cat:<32}: {cnt}")
        print("=" * 70 + "\n")
        sys.exit(0)

    worker.run()


if __name__ == "__main__":
    main()
