"""Daily housekeeping for one profile, controlled from its preferences.json:

    "daily_auto_clean": true,      # yes/no - nothing below runs when false
    "auto_clean_cutoff": 55        # New jobs scored below this are dismissed

In order, each run:
  1. checks every New / To process listing and marks closed ones EXPIRED
     (check_open_jobs.py)
  2. dismisses scored New jobs below the cutoff as DISCARDED, tagged
     status_source 'auto_clean' so the search-term review does not read them
     as the candidate's own decisions. Favourited jobs are never touched.
  3. looks at every unscored job, works out WHY it is unscored, tries to fix
     it (refetch a missing description, requeue for batch scoring) and gives
     up on a job after it has failed MAX_GIVE_UPS times
  4. writes users/<profile>/daily_clean_report.json, which the dashboard
     shows in its status area

Usage: JOBSEARCH_PROFILE=bob python daily_clean.py [--cutoff N] [--force]
  --cutoff N  use N instead of the configured cutoff for this run only
  --force     run even though daily_auto_clean is off
"""

import collections
import json
import sys
from datetime import datetime, timezone

import batch_scoring
import check_open_jobs
import ingest
from db import connection
from profile import PROFILE, PROFILE_DIR
from scorer import load_prefs

REPORT_PATH = PROFILE_DIR / "daily_clean_report.json"
MAX_GIVE_UPS = 3  # stop paying to retry a job that has failed to score this many times
MIN_DESCRIPTION_CHARS = 300

# (substring of the error message, plain-English reason)
_ERROR_REASONS = (
    ("credit balance", "API credit had run out when it was scored"),
    ("authentication_error", "API key was invalid when it was scored"),
    ("source_quote does not appear", "quote check failed (the model quoted text that is not in the ad)"),
    ("evidence_depth", "model returned an inconsistent match depth"),
    ("Invalid requirement importance", "model returned an invalid field value"),
    ("Unterminated string", "model returned malformed JSON"),
    ("Expecting", "model returned malformed JSON"),
)


def _reason_from_error(message: str) -> str:
    for needle, reason in _ERROR_REASONS:
        if needle in message:
            return reason
    return f"scoring error: {message[:90]}"


def _last_scoring_error(conn, job) -> str | None:
    """The most recent ingest error that names this job - the email path logs
    the title, the search path logs the canonical id."""
    row = conn.execute(
        "SELECT message FROM ingestion_events WHERE level = 'error' AND "
        "(message LIKE ? OR message LIKE ?) ORDER BY id DESC LIMIT 1",
        (f"%{job['canonical_id']}%", f'%"{job["title"]}"%'),
    ).fetchone()
    return row["message"] if row else None


def _refetch_description(job) -> str | None:
    url = job["url"] or ""
    if job["platform"] == "seek":
        result = ingest.fetch_seek_description(url)
    elif job["platform"] == "linkedin":
        match = check_open_jobs.LINKEDIN_ID.search(url)
        result = ingest.fetch_linkedin_description(match.group(1)) if match else None
    else:
        return None
    return result["description"] if result else None


def remove_below_cutoff(cutoff: int) -> int:
    with connection() as conn:
        return conn.execute(
            "UPDATE jobs SET status = 'DISCARDED', status_source = 'auto_clean' "
            "WHERE status = 'NEW' AND overall_score IS NOT NULL AND overall_score < ? AND favorited = 0",
            (cutoff,),
        ).rowcount


def handle_unscored() -> dict:
    """Explains every unscored New / To process job and requeues the ones
    worth another try. Returns {total, reasons: {text: count}, requeued}."""
    reasons: collections.Counter = collections.Counter()
    requeued = 0
    batch_scoring.ensure_schema()
    with connection() as conn:
        jobs = conn.execute(
            "SELECT id, canonical_id, platform, url, title, raw_description FROM jobs "
            "WHERE status IN ('NEW', 'TO_PROCESS') AND ai_evaluation IS NULL ORDER BY id"
        ).fetchall()
        queued = {r["job_id"] for r in conn.execute("SELECT job_id FROM scoring_queue")}
        gave_up = {r["job_id"]: r for r in conn.execute("SELECT * FROM scoring_gave_up")}

    for job in jobs:
        if job["id"] in queued:
            reasons["waiting in the batch scoring queue"] += 1
            continue

        description = job["raw_description"] or ""
        if len(description) < MIN_DESCRIPTION_CHARS:
            fetched = _refetch_description(job)
            if not fetched:
                reasons[f"no description could be fetched ({job['platform']})"] += 1
                continue
            with connection() as conn:
                conn.execute("UPDATE jobs SET raw_description = ? WHERE id = ?", (fetched, job["id"]))
            reasons["description was missing - fetched it and requeued"] += 1
            batch_scoring.enqueue(job["id"])
            requeued += 1
            continue

        record = gave_up.get(job["id"])
        if record and record["gave_up_count"] >= MAX_GIVE_UPS:
            reasons[f"gave up after {record['gave_up_count']} tries: {record['reason'][:90]}"] += 1
            continue

        if record:
            why = f"retrying - last failure: {record['reason'][:80]}"
        else:
            with connection() as conn:
                error = _last_scoring_error(conn, job)
            why = f"retrying - {_reason_from_error(error)}" if error else "retrying - no failure on record"
        reasons[why] += 1
        batch_scoring.enqueue(job["id"])
        requeued += 1

    return {"total": len(jobs), "reasons": dict(reasons.most_common()), "requeued": requeued}


def main() -> None:
    prefs = load_prefs()
    enabled = bool(prefs.get("daily_auto_clean", False)) or "--force" in sys.argv
    cutoff = int(sys.argv[sys.argv.index("--cutoff") + 1]) if "--cutoff" in sys.argv else int(prefs.get("auto_clean_cutoff", 50))
    report = {"ran_at": datetime.now(timezone.utc).isoformat(), "enabled": enabled, "cutoff": cutoff}

    if not enabled:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[{PROFILE}] daily_auto_clean is off - nothing to do.")
        return

    # Marker for the dashboard/watchdog: the closed check can take many minutes
    # (one page fetch per job), and a crash leaves this behind to go stale.
    REPORT_PATH.write_text(json.dumps({**report, "in_progress": True}, indent=2), encoding="utf-8")

    checked = check_open_jobs.check_all(apply=True, status_source="daily_clean", verbose=False)
    report["closed_check"] = {
        "total": checked["total"], "open": checked["open"], "closed": len(checked["closed"]),
        "closed_titles": [f"{j['title'][:50]} - {j['company'][:30]}" for j in checked["closed"]][:15],
        "unknown": len(checked["unknown"]), "unsupported": dict(checked["unsupported"]),
        "rate_limited": checked["rate_limited"], "not_reached": checked["not_reached"],
    }
    report["removed_below_cutoff"] = remove_below_cutoff(cutoff)
    report["unscored"] = handle_unscored()
    report["ran_at"] = datetime.now(timezone.utc).isoformat()  # finish time, not start

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    c, u = report["closed_check"], report["unscored"]
    print(f"[{PROFILE}] cutoff {cutoff}: removed {report['removed_below_cutoff']} below it | "
          f"closed check: {c['open']} open, {c['closed']} closed, {c['unknown']} unknown, "
          f"unsupported {c['unsupported'] or 'none'}"
          f"{', RATE LIMITED, ' + str(c['not_reached']) + ' not reached' if c['rate_limited'] else ''} | "
          f"unscored: {u['total']} ({u['requeued']} requeued)")
    for reason, count in u["reasons"].items():
        print(f"    {count:>3}  {reason}")


if __name__ == "__main__":
    main()
