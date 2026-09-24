"""SQLite persistence for JobHunter."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from profile import PROFILE_DIR

DB_PATH = PROFILE_DIR / "jobs.db"


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # The worker and dashboard are separate processes writing to the same file
    # concurrently (progress events during a run, status changes from the UI) —
    # give SQLite room to serialize writers instead of failing immediately.
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_emails (
                msg_uid INTEGER PRIMARY KEY,
                processed_at TIMESTAMP NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                canonical_id TEXT UNIQUE,
                fingerprint TEXT UNIQUE,
                platform TEXT,
                title TEXT,
                company TEXT,
                location TEXT,
                url TEXT,
                raw_description TEXT,
                ai_evaluation JSON,
                overall_score INTEGER,
                status TEXT NOT NULL DEFAULT 'NEW',
                created_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status_score
                ON jobs(status, overall_score DESC);

            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id INTEGER PRIMARY KEY,
                checked_at TIMESTAMP NOT NULL,
                folder TEXT NOT NULL,
                emails_seen INTEGER NOT NULL DEFAULT 0,
                emails_processed INTEGER NOT NULL DEFAULT 0,
                job_links_found INTEGER NOT NULL DEFAULT 0,
                jobs_saved INTEGER NOT NULL DEFAULT 0,
                duplicates INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                worker_version TEXT,
                status TEXT NOT NULL DEFAULT 'done',
                finished_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ingestion_events (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ingestion_events_run
                ON ingestion_events(run_id, id);

            CREATE TABLE IF NOT EXISTS email_jobs (
                msg_uid INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                PRIMARY KEY (msg_uid, job_id)
            );
            CREATE INDEX IF NOT EXISTS idx_email_jobs_job ON email_jobs(job_id);
            """
        )
        for statement in (
            "ALTER TABLE ingestion_runs ADD COLUMN worker_version TEXT",
            "ALTER TABLE ingestion_runs ADD COLUMN status TEXT NOT NULL DEFAULT 'done'",
            "ALTER TABLE ingestion_runs ADD COLUMN finished_at TIMESTAMP",
            "ALTER TABLE processed_emails ADD COLUMN moved_to TEXT",
            # Date the listing was actually posted, not when we saw it — "age of
            # listing" badge. Free text (not a strict date type): SEEK gives a
            # relative string ("Posted 22h ago"), Indeed/LinkedIn give an ISO
            # date via JobSpy. NULL where the source doesn't expose it at all.
            "ALTER TABLE jobs ADD COLUMN date_posted TEXT",
            # Favourite/star flag — independent of status (a NEW or
            # TO_PROCESS job can be starred without changing its workflow state).
            "ALTER TABLE jobs ADD COLUMN favorited INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already exists


def is_email_processed(uid: int) -> bool:
    with connection() as conn:
        return conn.execute(
            "SELECT 1 FROM processed_emails WHERE msg_uid = ?", (uid,)
        ).fetchone() is not None


def mark_email_processed(uid: int) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_emails (msg_uid, processed_at) VALUES (?, ?)",
            (uid, datetime.now(timezone.utc).isoformat()),
        )


def job_exists(canonical_id: str, fingerprint: str) -> bool:
    with connection() as conn:
        return conn.execute(
            "SELECT 1 FROM jobs WHERE canonical_id = ? OR fingerprint = ? LIMIT 1",
            (canonical_id, fingerprint),
        ).fetchone() is not None


def get_job_id(canonical_id: str, fingerprint: str) -> int | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM jobs WHERE canonical_id = ? OR fingerprint = ? LIMIT 1",
            (canonical_id, fingerprint),
        ).fetchone()
        return row["id"] if row else None


def save_job(job_dict: dict[str, Any]) -> int | None:
    """Insert a job. Returns its new id, or None if either deduplication key already exists."""
    evaluation = job_dict.get("ai_evaluation")
    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO jobs (
                canonical_id, fingerprint, platform, title, company, location,
                url, raw_description, ai_evaluation, overall_score, status, created_at, date_posted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_dict["canonical_id"], job_dict["fingerprint"], job_dict["platform"],
                job_dict.get("title", "Unknown title"), job_dict.get("company", "Unknown company"),
                job_dict.get("location", ""), job_dict["url"], job_dict.get("raw_description", ""),
                json.dumps(evaluation) if evaluation is not None else None,
                job_dict.get("overall_score"), job_dict.get("status", "NEW"),
                job_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
                job_dict.get("date_posted"),
            ),
        )
        return cursor.lastrowid if cursor.rowcount == 1 else None


def toggle_favorite(job_id: int) -> bool | None:
    """Flip the favourite flag. Returns the new state, or None if the job
    doesn't exist — kept independent of `status` so starring never disturbs
    a job's NEW/TO_PROCESS/APPLIED workflow position."""
    with connection() as conn:
        row = conn.execute("SELECT favorited FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        new_value = 0 if row["favorited"] else 1
        conn.execute("UPDATE jobs SET favorited = ? WHERE id = ?", (new_value, job_id))
        return bool(new_value)


def link_email_job(uid: int, job_id: int) -> None:
    """Record that `job_id` was found in email `uid` — used to know when every
    job from an email has been actioned, so the email can move out of Inbox."""
    with connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO email_jobs (msg_uid, job_id) VALUES (?, ?)", (uid, job_id)
        )


def emails_ready_to_move() -> list[tuple[int, str]]:
    """Emails not yet moved out of Inbox whose linked jobs are all past NEW.
    Folder priority: any job still under review (TO_PROCESS) keeps it in
    01-process; otherwise 02-applied if anything was applied; otherwise every
    job was discarded, so 03-discarded."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT pe.msg_uid,
                SUM(CASE WHEN j.status = 'NEW' THEN 1 ELSE 0 END) AS new_count,
                SUM(CASE WHEN j.status = 'TO_PROCESS' THEN 1 ELSE 0 END) AS to_process_count,
                SUM(CASE WHEN j.status = 'APPLIED' THEN 1 ELSE 0 END) AS applied_count
            FROM processed_emails pe
            JOIN email_jobs ej ON ej.msg_uid = pe.msg_uid
            JOIN jobs j ON j.id = ej.job_id
            WHERE pe.moved_to IS NULL
            GROUP BY pe.msg_uid
            HAVING new_count = 0
            """
        ).fetchall()
    result = []
    for row in rows:
        if row["to_process_count"] > 0:
            folder = "INBOX.01-process"
        elif row["applied_count"] > 0:
            folder = "INBOX.02-applied"
        else:
            folder = "INBOX.03-discarded"
        result.append((row["msg_uid"], folder))
    return result


def mark_email_moved(uid: int, folder: str) -> None:
    with connection() as conn:
        conn.execute("UPDATE processed_emails SET moved_to = ? WHERE msg_uid = ?", (folder, uid))


def start_ingestion_run(folder: str, worker_version: str) -> int:
    """Create the run row up front so the dashboard can show live progress
    against it while the worker is still going."""
    with connection() as conn:
        cursor = conn.execute(
            """INSERT INTO ingestion_runs (
                checked_at, folder, worker_version, status,
                emails_seen, emails_processed, job_links_found, jobs_saved, duplicates, errors
            ) VALUES (?, ?, ?, 'running', 0, 0, 0, 0, 0, 0)""",
            (datetime.now(timezone.utc).isoformat(), folder, worker_version),
        )
        return cursor.lastrowid


def log_event(run_id: int, level: str, message: str) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO ingestion_events (run_id, created_at, level, message) VALUES (?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(), level, message),
        )


def finish_ingestion_run(run_id: int, counts: dict[str, int], status: str) -> None:
    with connection() as conn:
        conn.execute(
            """UPDATE ingestion_runs SET
                emails_seen = ?, emails_processed = ?, job_links_found = ?,
                jobs_saved = ?, duplicates = ?, errors = ?,
                status = ?, finished_at = ?
            WHERE id = ?""",
            (
                counts["emails_seen"], counts["emails_processed"], counts["job_links_found"],
                counts["jobs_saved"], counts["duplicates"], counts["errors"],
                status, datetime.now(timezone.utc).isoformat(), run_id,
            ),
        )


def latest_ingestion_run() -> sqlite3.Row | None:
    with connection() as conn:
        return conn.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 1").fetchone()


def run_events(run_id: int, since_id: int = 0, limit: int = 200) -> list[sqlite3.Row]:
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM ingestion_events WHERE run_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (run_id, since_id, limit),
        ).fetchall()


init_db()
