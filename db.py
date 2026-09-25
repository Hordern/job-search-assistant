"""SQLite persistence for Job Search Assistant."""

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
            # How this job entered the system: 'email' (inbox job-alert,
            # ingest.py) or 'search' (active JobSpy keyword search,
            # search_ingest.py). NULL for rows saved before this column
            # existed — deliberately left unset rather than guessed, since
            # LinkedIn/Indeed/Glassdoor jobs could have come from either path.
            "ALTER TABLE jobs ADD COLUMN source TEXT",
            # Which configured search_term (search_ingest.py only) found this
            # job — lets us measure a term's keep/discard rate over time and
            # spot ones that are only surfacing noise. NULL for email-sourced
            # jobs and any search jobs saved before this column existed.
            "ALTER TABLE jobs ADD COLUMN search_term TEXT",
            # How many times "View" was clicked, and when last — a job the
            # user opened but never Kept/Discarded is still a real interest
            # signal for search-term review, distinct from the explicit
            # Keep/Discard/Applied/Favourite actions.
            "ALTER TABLE jobs ADD COLUMN click_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE jobs ADD COLUMN last_clicked_at TIMESTAMP",
            # Who/what last set `status` — 'user' (via the real Keep/Discard/
            # Mark Applied/Decline buttons, i.e. genuine judgment) vs
            # 'bulk_cleanup' (a direct DB script, e.g. "discard everything
            # under score N"). NULL for rows never touched this way, or
            # touched before this column existed (unknown/mixed provenance
            # — don't assume either way). Critical for trusting any future
            # preference-learning analysis: a bulk score-based discard says
            # nothing about whether the candidate would have rejected that
            # specific job on its merits.
            "ALTER TABLE jobs ADD COLUMN status_source TEXT",
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
                url, raw_description, ai_evaluation, overall_score, status, created_at, date_posted,
                source, search_term
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_dict["canonical_id"], job_dict["fingerprint"], job_dict["platform"],
                job_dict.get("title", "Unknown title"), job_dict.get("company", "Unknown company"),
                job_dict.get("location", ""), job_dict["url"], job_dict.get("raw_description", ""),
                json.dumps(evaluation) if evaluation is not None else None,
                job_dict.get("overall_score"), job_dict.get("status", "NEW"),
                job_dict.get("created_at", datetime.now(timezone.utc).isoformat()),
                job_dict.get("date_posted"),
                job_dict.get("source"), job_dict.get("search_term"),
            ),
        )
        return cursor.lastrowid if cursor.rowcount == 1 else None


_KEPT_STATUSES = ("TO_PROCESS", "APPLIED")
_DECIDED_STATUSES = ("TO_PROCESS", "APPLIED", "DISCARDED", "DECLINED")


def search_term_stats() -> list[dict[str, Any]]:
    """Per (platform, search_term) performance for active-search-sourced
    jobs — how many it's found, how many are still undecided, and of the
    ones actually decided, what fraction were kept vs discarded/declined.
    Ordered worst-discard-rate first so a noisy term is easy to spot.
    Terms with fewer than 3 decided jobs are excluded — too little signal
    to judge yet, not because they're bad.

    A decision on a job whose employer matched an excluded industry
    (industry_fit.status == 'excluded') is left out of the discard-rate
    math entirely — that discard was about the employer, not whether the
    term/title was a good match, so counting it against the term would
    blame the wrong thing. Still counted in `found` and reported
    separately as `industry_excluded`, so the number isn't just silently
    dropped."""
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT platform, search_term, status, ai_evaluation
            FROM jobs
            WHERE source = 'search' AND search_term IS NOT NULL
            """
        ).fetchall()

    stats: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["platform"], row["search_term"])
        s = stats.setdefault(key, {"found": 0, "kept": 0, "decided": 0, "industry_excluded": 0})
        s["found"] += 1

        industry_excluded = False
        if row["ai_evaluation"]:
            try:
                industry_excluded = (json.loads(row["ai_evaluation"]).get("industry_fit") or {}).get("status") == "excluded"
            except json.JSONDecodeError:
                pass

        if row["status"] in _DECIDED_STATUSES:
            if industry_excluded:
                s["industry_excluded"] += 1
            else:
                s["decided"] += 1
                if row["status"] in _KEPT_STATUSES:
                    s["kept"] += 1

    result = []
    for (platform, term), s in stats.items():
        if s["decided"] < 3:
            continue
        discard_rate = 1 - (s["kept"] / s["decided"])
        result.append({
            "platform": platform, "search_term": term,
            "found": s["found"], "decided": s["decided"], "kept": s["kept"],
            "industry_excluded": s["industry_excluded"],
            "discard_rate": round(discard_rate, 2),
        })
    result.sort(key=lambda r: r["discard_rate"], reverse=True)
    return result


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


def record_click(job_id: int) -> None:
    """Fire-and-forget signal that 'View' was clicked — best-effort, never
    raises on a missing job (a stale card, a race with deletion) since this
    must never block the user's actual navigation to the job posting."""
    with connection() as conn:
        conn.execute(
            "UPDATE jobs SET click_count = click_count + 1, last_clicked_at = datetime('now') WHERE id = ?",
            (job_id,),
        )


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


def latest_ingestion_run_for_email() -> sqlite3.Row | None:
    """Most recent inbox check specifically — 'Last checked' should never
    be silently replaced by a search-ingest run just because it happened
    more recently; the two are different mechanisms with different
    schedules and deserve separate status lines."""
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM ingestion_runs WHERE folder = 'INBOX' ORDER BY id DESC LIMIT 1"
        ).fetchone()


def latest_ingestion_run_for_search() -> sqlite3.Row | None:
    """Most recent active-search run (any connector) — the search_ingest.py
    cycle logs one row per enabled connector each run; this is the latest
    of those, used for a 'Last searched' status line."""
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM ingestion_runs WHERE folder LIKE 'search:%' ORDER BY id DESC LIMIT 1"
        ).fetchone()


def run_events(run_id: int, since_id: int = 0, limit: int = 200) -> list[sqlite3.Row]:
    with connection() as conn:
        return conn.execute(
            "SELECT * FROM ingestion_events WHERE run_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (run_id, since_id, limit),
        ).fetchall()


init_db()
