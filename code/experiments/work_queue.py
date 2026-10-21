"""
Persistent SQLite work queue and fair scheduler for AgentEvalOps.

Manages durable state transitions, crash recovery, rate limit cooldowns,
attempt logging, and fair round-robin scheduling across provider-model pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from code.shared.models import TaskRecord


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEFERRED_RATE_LIMIT = "deferred_rate_limit"
    DEFERRED_DAILY_QUOTA = "deferred_daily_quota"
    PROVIDER_ERROR = "provider_error"
    AGENT_EXECUTION_ERROR = "agent_execution_error"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    job_id: str
    task_id: str
    task_content_hash: str
    provider: str
    model_id: str
    repetition: int
    state: JobState
    attempts: int = 0
    max_attempts: int = 3
    cooldown_until: float = 0.0
    last_error: str | None = None
    error_type: str | None = None
    run_id: str | None = None
    final_answer: str | None = None
    outcome: str | None = None
    process_category: str | None = None
    created_at: str = ""
    updated_at: str = ""


def compute_job_id(
    task_id: str,
    task_content_hash: str,
    provider: str,
    model_id: str,
    repetition: int,
    experiment_hash: str,
) -> str:
    """Derive deterministic, collision-free job ID."""
    raw = f"{task_id}:{task_content_hash}:{provider}:{model_id}:{repetition}:{experiment_hash}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    # Sanitise model_id for readable prefix
    clean_model = model_id.split("/")[-1].replace(":", "_").replace("-", "_")[:14]
    return f"{task_id}_{clean_model}_r{repetition}_{digest}"


class JobQueue:
    """Thread-safe SQLite persistent work queue with fair scheduling."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_content_hash TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    repetition INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    cooldown_until REAL DEFAULT 0.0,
                    last_error TEXT,
                    error_type TEXT,
                    run_id TEXT,
                    final_answer TEXT,
                    outcome TEXT,
                    process_category TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_state_target ON jobs(state, provider, model_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_cooldown ON jobs(cooldown_until);

                CREATE TABLE IF NOT EXISTS job_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    requests_made INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    duration_ms REAL DEFAULT 0.0,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );

                CREATE TABLE IF NOT EXISTS provider_status (
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    is_quarantined INTEGER DEFAULT 0,
                    quarantine_reason TEXT,
                    cooldown_until REAL DEFAULT 0.0,
                    requests_consumed INTEGER DEFAULT 0,
                    last_status_code INTEGER,
                    last_error TEXT,
                    daily_quota_limit INTEGER,
                    daily_requests_used INTEGER DEFAULT 0,
                    daily_reset_at REAL DEFAULT 0.0,
                    last_verified_at TEXT,
                    PRIMARY KEY(provider, model_id)
                );
            """)

    # -----------------------------------------------------------------------
    # Population & Initialization
    # -----------------------------------------------------------------------

    def populate_jobs(
        self,
        tasks: list[TaskRecord],
        targets: list[tuple[str, str]],  # (provider, model_id)
        repetitions: int = 1,
        experiment_hash: str = "default_exp",
        seed: int = 42,
    ) -> int:
        """
        Populate jobs table for task × target × repetition grid.
        Shuffles tasks with seed for fair dispersion. Does not overwrite completed jobs.
        """
        rng = random.Random(seed)
        shuffled_tasks = list(tasks)
        rng.shuffle(shuffled_tasks)

        now = datetime.now(timezone.utc).isoformat()
        inserted_count = 0

        with self._get_connection() as conn:
            for rep in range(repetitions):
                for task in shuffled_tasks:
                    thash = task.metadata.content_hash if task.metadata else "no_hash"
                    for provider, model_id in targets:
                        jid = compute_job_id(task.task_id, thash, provider, model_id, rep, experiment_hash)
                        cursor = conn.execute(
                            """
                            INSERT OR IGNORE INTO jobs (
                                job_id, task_id, task_content_hash, provider, model_id, repetition,
                                state, attempts, max_attempts, cooldown_until, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 3, 0.0, ?, ?)
                            """,
                            (
                                jid, task.task_id, thash, provider, model_id, rep,
                                JobState.PENDING.value, now, now
                            ),
                        )
                        if cursor.rowcount > 0:
                            inserted_count += 1
        return inserted_count

    # -----------------------------------------------------------------------
    # Crash Recovery & State Reconcile
    # -----------------------------------------------------------------------

    def recover_stale_jobs(self) -> int:
        """
        Reset any jobs left in 'running' back to 'pending'.
        Called on worker startup to recover cleanly from unexpected crashes.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET state = ?, updated_at = ?
                WHERE state = ?
                """,
                (JobState.PENDING.value, now, JobState.RUNNING.value),
            )
            return cursor.rowcount

    def reconcile_with_trajectories(self, existing_run_ids: set[str]) -> int:
        """Reconcile jobs matching already-persisted trajectories."""
        if not existing_run_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self._get_connection() as conn:
            for run_id in existing_run_ids:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, updated_at = ?
                    WHERE run_id = ? AND state != ?
                    """,
                    (JobState.COMPLETED.value, now, run_id, JobState.COMPLETED.value),
                )
                count += cursor.rowcount
        return count

    # -----------------------------------------------------------------------
    # Cooldown & Quarantine Management
    # -----------------------------------------------------------------------

    def set_provider_cooldown(self, provider: str, model_id: str, cooldown_seconds: float, reason: str = "") -> None:
        """Set cooldown on a provider/model."""
        target_time = time.time() + cooldown_seconds
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO provider_status (provider, model_id, cooldown_until, last_error, last_verified_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider, model_id) DO UPDATE SET
                    cooldown_until = excluded.cooldown_until,
                    last_error = excluded.last_error,
                    last_verified_at = excluded.last_verified_at
                """,
                (provider, model_id, target_time, reason, now),
            )

    def quarantine_provider(self, provider: str, model_id: str, reason: str) -> None:
        """Quarantine a provider/model permanently from further scheduling."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO provider_status (provider, model_id, is_quarantined, quarantine_reason, last_error, last_verified_at)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(provider, model_id) DO UPDATE SET
                    is_quarantined = 1,
                    quarantine_reason = excluded.quarantine_reason,
                    last_error = excluded.last_error,
                    last_verified_at = excluded.last_verified_at
                """,
                (provider, model_id, reason, reason, now),
            )
            # Mark all pending/deferred jobs for this provider as provider_error
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, last_error = ?, error_type = 'auth_or_quarantine', updated_at = ?
                WHERE provider = ? AND model_id = ? AND state IN ('pending', 'deferred_rate_limit', 'deferred_daily_quota')
                """,
                (JobState.PROVIDER_ERROR.value, reason, now, provider, model_id),
            )

    def is_provider_eligible(self, provider: str, model_id: str) -> bool:
        """Check if provider/model is not quarantined and not in active cooldown."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT is_quarantined, cooldown_until FROM provider_status WHERE provider = ? AND model_id = ?",
                (provider, model_id),
            ).fetchone()
            if not row:
                return True
            if row["is_quarantined"]:
                return False
            if row["cooldown_until"] and row["cooldown_until"] > time.time():
                return False
            return True

    def refresh_deferred_jobs(self) -> int:
        """Reopen rate-limited deferred jobs whose cooldown has elapsed."""
        now_ts = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET state = ?, updated_at = ?
                WHERE state = ? AND cooldown_until <= ? AND attempts < max_attempts
                """,
                (JobState.PENDING.value, now_iso, JobState.DEFERRED_RATE_LIMIT.value, now_ts),
            )
            return cursor.rowcount

    # -----------------------------------------------------------------------
    # Fair Scheduling (Round-Robin over Targets)
    # -----------------------------------------------------------------------

    def get_eligible_targets(self) -> list[tuple[str, str]]:
        """Return list of (provider, model_id) pairs with pending work and no active cooldown."""
        now_ts = time.time()
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT j.provider, j.model_id
                FROM jobs j
                LEFT JOIN provider_status ps ON j.provider = ps.provider AND j.model_id = ps.model_id
                WHERE j.state = ?
                  AND (ps.is_quarantined IS NULL OR ps.is_quarantined = 0)
                  AND (ps.cooldown_until IS NULL OR ps.cooldown_until <= ?)
                ORDER BY j.provider, j.model_id
                """,
                (JobState.PENDING.value, now_ts),
            ).fetchall()
            return [(r["provider"], r["model_id"]) for r in rows]

    def claim_next_job(self, provider: str, model_id: str) -> JobRecord | None:
        """Atomically claim the next pending job for a specific target."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE provider = ? AND model_id = ? AND state = ?
                ORDER BY repetition ASC, rowid ASC
                LIMIT 1
                """,
                (provider, model_id, JobState.PENDING.value),
            ).fetchone()

            if not row:
                conn.commit()
                return None

            job_id = row["job_id"]
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, attempts = attempts + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.RUNNING.value, now_iso, job_id),
            )
            conn.commit()

            return JobRecord(
                job_id=row["job_id"],
                task_id=row["task_id"],
                task_content_hash=row["task_content_hash"],
                provider=row["provider"],
                model_id=row["model_id"],
                repetition=row["repetition"],
                state=JobState.RUNNING,
                attempts=row["attempts"] + 1,
                max_attempts=row["max_attempts"],
                cooldown_until=row["cooldown_until"],
                created_at=row["created_at"],
                updated_at=now_iso,
            )

    # -----------------------------------------------------------------------
    # Job Completion & Deferral
    # -----------------------------------------------------------------------

    def mark_completed(
        self,
        job_id: str,
        run_id: str,
        final_answer: str | None,
        outcome: str,
        process_category: str,
    ) -> None:
        """Mark job as successfully completed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, run_id = ?, final_answer = ?, outcome = ?, process_category = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.COMPLETED.value, run_id, final_answer, outcome, process_category, now, job_id),
            )

    def mark_deferred_rate_limit(
        self,
        job_id: str,
        cooldown_seconds: float,
        error_msg: str,
        error_type: str = "rate_limit",
    ) -> None:
        """Defer a job due to rate limit (429) or transient provider failure."""
        now = datetime.now(timezone.utc).isoformat()
        cd_ts = time.time() + cooldown_seconds
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, cooldown_until = ?, last_error = ?, error_type = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.DEFERRED_RATE_LIMIT.value, cd_ts, error_msg, error_type, now, job_id),
            )

    def mark_deferred_daily_quota(
        self,
        job_id: str,
        reset_seconds: float,
        error_msg: str,
    ) -> None:
        """Defer a job due to daily quota exhaustion."""
        now = datetime.now(timezone.utc).isoformat()
        cd_ts = time.time() + reset_seconds
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, cooldown_until = ?, last_error = ?, error_type = 'daily_quota', updated_at = ?
                WHERE job_id = ?
                """,
                (JobState.DEFERRED_DAILY_QUOTA.value, cd_ts, error_msg, now, job_id),
            )

    def mark_job_failed(
        self,
        job_id: str,
        error_msg: str,
        error_type: str = "execution_error",
        is_permanent: bool = False,
    ) -> None:
        """Mark a job as failed (permanent or exhausted retries)."""
        now = datetime.now(timezone.utc).isoformat()
        if is_permanent or error_type == "provider_error":
            target_state = JobState.PROVIDER_ERROR.value
        else:
            target_state = JobState.AGENT_EXECUTION_ERROR.value
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, last_error = ?, error_type = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (target_state, error_msg, error_type, now, job_id),
            )

    def log_attempt(
        self,
        job_id: str,
        attempt_number: int,
        started_at: str,
        finished_at: str,
        status: str,
        error: str | None = None,
        requests_made: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """Record attempt details."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO job_attempts (
                    job_id, attempt_number, started_at, finished_at, status,
                    error, requests_made, prompt_tokens, completion_tokens, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id, attempt_number, started_at, finished_at, status,
                    error, requests_made, prompt_tokens, completion_tokens, duration_ms
                ),
            )

    # -----------------------------------------------------------------------
    # Observability & Progress Summary
    # -----------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Aggregate queue statistics and progress report."""
        with self._get_connection() as conn:
            # Job counts by state
            state_counts: dict[str, int] = {}
            for row in conn.execute("SELECT state, COUNT(*) as cnt FROM jobs GROUP BY state"):
                state_counts[row["state"]] = row["cnt"]

            total_jobs = sum(state_counts.values())

            # Progress by model
            model_stats: dict[str, dict[str, int]] = {}
            for row in conn.execute(
                "SELECT provider, model_id, state, COUNT(*) as cnt FROM jobs GROUP BY provider, model_id, state"
            ):
                key = f"{row['provider']}:{row['model_id']}"
                if key not in model_stats:
                    model_stats[key] = {}
                model_stats[key][row["state"]] = row["cnt"]

            # Outcome stats
            outcomes: dict[str, int] = {}
            for row in conn.execute(
                "SELECT outcome, COUNT(*) as cnt FROM jobs WHERE state = 'completed' GROUP BY outcome"
            ):
                outcomes[row["outcome"]] = row["cnt"]

            # Process categories
            categories: dict[str, int] = {}
            for row in conn.execute(
                "SELECT process_category, COUNT(*) as cnt FROM jobs WHERE process_category IS NOT NULL GROUP BY process_category"
            ):
                categories[row["process_category"]] = row["cnt"]

            # Active provider cooldowns
            cooldowns: list[dict[str, Any]] = []
            now_ts = time.time()
            for row in conn.execute(
                "SELECT provider, model_id, is_quarantined, quarantine_reason, cooldown_until FROM provider_status"
            ):
                rem_cd = max(0.0, row["cooldown_until"] - now_ts) if row["cooldown_until"] else 0.0
                cooldowns.append({
                    "target": f"{row['provider']}:{row['model_id']}",
                    "is_quarantined": bool(row["is_quarantined"]),
                    "quarantine_reason": row["quarantine_reason"],
                    "cooldown_remaining_sec": round(rem_cd, 1),
                })

            return {
                "total_jobs": total_jobs,
                "state_counts": state_counts,
                "completed_count": state_counts.get(JobState.COMPLETED.value, 0),
                "pending_count": state_counts.get(JobState.PENDING.value, 0),
                "deferred_count": (
                    state_counts.get(JobState.DEFERRED_RATE_LIMIT.value, 0)
                    + state_counts.get(JobState.DEFERRED_DAILY_QUOTA.value, 0)
                ),
                "error_count": (
                    state_counts.get(JobState.PROVIDER_ERROR.value, 0)
                    + state_counts.get(JobState.AGENT_EXECUTION_ERROR.value, 0)
                ),
                "model_stats": model_stats,
                "outcomes": outcomes,
                "process_categories": categories,
                "provider_statuses": cooldowns,
            }
