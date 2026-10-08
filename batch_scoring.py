"""Batch scoring runner - PROTOTYPE, not wired into ingest.py yet.

The Message Batches API bills every token at 50% of the normal price in
exchange for asynchronous processing (most batches finish inside an hour,
24h worst case). Scoring is not latency-sensitive - jobs arrive by email and
you read them later - so the only cost of switching is that a job shows up
scored some minutes later instead of seconds later.

The pipeline is three dependent passes, so each pass becomes its own batch:

    extract  --> hard-disqualified? finish now (1 call total)
             --> audit (only when needs_audit()) --> match --> finish

Each cycle of run_cycle() collects any finished batches, advances those jobs
one stage, and submits the next stage's requests as one new batch. All state
lives in the scoring_queue table, so a crash or reschedule simply resumes.
Parsing, validation, dedupe, the eligibility/industry short-circuit and the
deterministic scoring are the same scorer.py functions the synchronous path
uses (see hard_disqualified_result / merge_audit_and_dedupe / finish_scoring),
so a batch-scored job is identical to a synchronously scored one.

Usage: JOBSEARCH_PROFILE=alice python batch_scoring.py
"""

import fcntl
import json
import sys
from datetime import datetime, timezone

from anthropic import Anthropic, AuthenticationError, BadRequestError, PermissionDeniedError
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

import scorer
from db import connection
from profile import PROFILE, PROFILE_DIR

MAX_ATTEMPTS = 3
LOCK_PATH = PROFILE_DIR / ".batch.lock"
ERROR_PATH = PROFILE_DIR / "batch_last_error.json"
# Default 5-minute TTL on purpose. Measured on a 4-job test batch (2026-10-08):
# requests in a batch run concurrently, so most of them each WRITE the shared
# prefix rather than read it (3 writes + 1 read per stage) - cache hits inside
# a batch are best-effort. A 1-hour TTL would only make those writes cost
# 2x instead of 1.25x, and batches finished in 2-3 minutes anyway.
CACHE_CONTROL = {"type": "ephemeral"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_queue (
    job_id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,            -- extract | audit | match
    batch_id TEXT,                  -- NULL = waiting to be submitted
    extraction TEXT,                -- job_extraction JSON once pass 1 has parsed
    calls INTEGER NOT NULL DEFAULT 1,
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_error TEXT,               -- appended to the prompt on a retry
    created_at TEXT NOT NULL
)"""


def ensure_schema() -> None:
    with connection() as conn:
        conn.execute(SCHEMA)


def enqueue(job_id: int) -> None:
    """Called by ingest in place of score_job(): the job row is already saved
    with ai_evaluation NULL; this just registers it for the next batch."""
    ensure_schema()
    with connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO scoring_queue (job_id, stage, created_at) VALUES (?, 'extract', ?)",
            (job_id, datetime.now(timezone.utc).isoformat()),
        )


def _job(conn, job_id: int):
    return conn.execute("SELECT id, title, company, raw_description FROM jobs WHERE id = ?", (job_id,)).fetchone()


def _prompt(stage: str, job, row):
    if stage == "extract":
        prompt = scorer.build_job_extraction_prompt(job["title"], job["company"], job["raw_description"])
    elif stage == "audit":
        prompt = scorer.build_audit_prompt(
            job["title"], job["company"], job["raw_description"], json.loads(row["extraction"]),
        )
    else:
        prompt = scorer.build_cv_matching_prompt(json.loads(row["extraction"]))
    if row["retry_error"]:
        prompt = scorer._append_retry_hint(prompt, row["retry_error"])
    return prompt


def _request(job_id: int, prompt) -> Request:
    """Same request _anthropic_call sends, plus the 1h cache TTL."""
    if isinstance(prompt, scorer.CachedPrompt):
        content = [
            {"type": "text", "text": prompt.cached, "cache_control": CACHE_CONTROL},
            {"type": "text", "text": prompt.variable},
        ]
    else:
        content = prompt
    return Request(
        custom_id=str(job_id),
        params=MessageCreateParamsNonStreaming(
            model=scorer.MODEL,
            max_tokens=8192,
            thinking=scorer._thinking_off_param(scorer.MODEL),
            metadata={"user_id": f"jobsearch-{PROFILE}-batch"},
            messages=[{"role": "user", "content": content}],
        ),
    )


def submit_pending(client: Anthropic) -> int:
    """One new batch per stage that has jobs waiting. Returns jobs submitted."""
    submitted = 0
    with connection() as conn:
        for stage in ("extract", "audit", "match"):
            rows = conn.execute(
                "SELECT * FROM scoring_queue WHERE stage = ? AND batch_id IS NULL", (stage,)
            ).fetchall()
            if not rows:
                continue
            requests = [_request(r["job_id"], _prompt(stage, _job(conn, r["job_id"]), r)) for r in rows]
            batch = client.messages.batches.create(requests=requests)
            conn.executemany(
                "UPDATE scoring_queue SET batch_id = ? WHERE job_id = ?", [(batch.id, r["job_id"]) for r in rows]
            )
            conn.commit()
            submitted += len(rows)
            print(f"[{PROFILE}] submitted {stage} batch {batch.id}: {len(rows)} job(s)")
    return submitted


def _advance(conn, row, job, text: str | None) -> None:
    stage = row["stage"]
    try:
        if text is None:
            raise ValueError("batch request did not succeed")

        if stage == "extract":
            extraction = scorer.extract_job_requirements(text, job["title"], job["company"], job["raw_description"])
            disqualified = scorer.hard_disqualified_result(extraction, calls_made=row["calls"])
            if disqualified is not None:
                return _finish(conn, row, disqualified)
            if scorer.needs_audit(extraction, job["raw_description"]):
                return _move(conn, row, "audit", extraction)
            scorer.merge_audit_and_dedupe(extraction, None)
            return _move(conn, row, "match", extraction)

        extraction = json.loads(row["extraction"])
        if stage == "audit":
            audit = scorer.extract_audit_findings(text, job["title"], job["company"], job["raw_description"])
            scorer.merge_audit_and_dedupe(extraction, audit)
            return _move(conn, row, "match", extraction)

        expected_ids = {r["requirement_id"] for r in extraction["requirements"] if r["category"] in scorer.CAPABILITY_CATEGORIES}
        matching = scorer.extract_cv_matches(text, expected_ids)
        _finish(conn, row, scorer.finish_scoring(extraction, matching, row["calls"]))

    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        attempts = row["attempts"] + 1
        if attempts >= MAX_ATTEMPTS:
            # Same policy as the synchronous path: stop paying for a job whose
            # content reliably fails to parse; it stays unscored and visible.
            conn.execute("DELETE FROM scoring_queue WHERE job_id = ?", (row["job_id"],))
            print(f"[{PROFILE}] job {row['job_id']} gave up after {attempts} attempts: {exc}")
        else:
            conn.execute(
                "UPDATE scoring_queue SET batch_id = NULL, attempts = ?, retry_error = ? WHERE job_id = ?",
                (attempts, f"{type(exc).__name__}: {exc}", row["job_id"]),
            )


def _move(conn, row, stage: str, extraction: dict) -> None:
    conn.execute(
        "UPDATE scoring_queue SET stage = ?, batch_id = NULL, extraction = ?, calls = calls + 1, "
        "attempts = 0, retry_error = NULL WHERE job_id = ?",
        (stage, json.dumps(extraction), row["job_id"]),
    )


def _finish(conn, row, evaluation: dict) -> None:
    evaluation["description_source"] = "full"
    conn.execute(
        "UPDATE jobs SET ai_evaluation = ?, overall_score = ? WHERE id = ?",
        (json.dumps(evaluation), evaluation["overall_score"], row["job_id"]),
    )
    conn.execute("DELETE FROM scoring_queue WHERE job_id = ?", (row["job_id"],))
    print(f"[{PROFILE}] scored job {row['job_id']}: {evaluation['overall_score']} - {evaluation['recommendation']}")


def collect_finished(client: Anthropic) -> int:
    """Advance every job whose batch has ended. Returns batches processed."""
    done = 0
    with connection() as conn:
        batch_ids = [r["batch_id"] for r in conn.execute(
            "SELECT DISTINCT batch_id FROM scoring_queue WHERE batch_id IS NOT NULL"
        )]
        for batch_id in batch_ids:
            if client.messages.batches.retrieve(batch_id).processing_status != "ended":
                continue
            for result in client.messages.batches.results(batch_id):
                job_id = int(result.custom_id)
                row = conn.execute("SELECT * FROM scoring_queue WHERE job_id = ?", (job_id,)).fetchone()
                if row is None or row["batch_id"] != batch_id:
                    continue  # already handled, or re-queued under a newer batch
                text = None
                if result.result.type == "succeeded":
                    scorer._log_usage(result.result.message.usage, batch=True)
                    text = "".join(b.text for b in result.result.message.content if b.type == "text")
                _advance(conn, row, _job(conn, job_id), text)
                conn.commit()  # keep the write lock short - the next call is a network round trip
            done += 1
    return done


def run_cycle() -> None:
    if not scorer.os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    ensure_schema()
    client = Anthropic()
    try:
        collected = collect_finished(client)
        submitted = submit_pending(client)
    except (AuthenticationError, BadRequestError, PermissionDeniedError) as exc:
        # Key revoked / credit exhausted: the same failures the synchronous path
        # logs to ingestion_events. Leave a marker for watchdog.py and re-raise.
        ERROR_PATH.write_text(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "message": str(exc)[:300]}))
        raise
    ERROR_PATH.unlink(missing_ok=True)
    with connection() as conn:
        waiting = conn.execute("SELECT stage, count(*) n FROM scoring_queue GROUP BY stage").fetchall()
    print(f"[{PROFILE}] collected {collected} batch(es), submitted {submitted} job(s); queue: "
          f"{ {r['stage']: r['n'] for r in waiting} or 'empty' }")


if __name__ == "__main__":
    # One cycle at a time: two overlapping runs would each submit the same
    # waiting rows (batch_id IS NULL) and pay for them twice.
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another batch scoring cycle is already running - skipping.")
        sys.exit(0)
    run_cycle()
