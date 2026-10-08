"""Active monitoring for Job Search Assistant — the dashboard's status
panels are passive (only mean something when you open the page); this is
the active counterpart. Runs on its own launchd schedule, checks the same
staleness facts the dashboard already shows, plus two things the dashboard
can't see about itself (whether its own process answers HTTP, whether the
launchd jobs are actually loaded), and fires a macOS notification only when
a NEW problem appears — not on every tick while it's still broken, and not
again once it recovers until it breaks again.

Usage: JOBSEARCH_PROFILE=alice python watchdog.py
"""

import json
import os
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from db import connection, latest_ingestion_run_for_email, latest_ingestion_run_for_search
from profile import PROFILE, PROFILE_DIR
from scorer import load_prefs

MELBOURNE = ZoneInfo("Australia/Melbourne")
STATE_PATH = PROFILE_DIR / "watchdog_state.json"

# Mirrors app.py's STALE_THRESHOLD_MINUTES / SEARCH_STALE_THRESHOLD_MINUTES —
# kept as separate constants (not imported) because app.py pulls in FastAPI
# and the rest of the dashboard just to get two integers.
EMAIL_STALE_MINUTES = 30
SEARCH_STALE_MINUTES = 60
# term_review.py is triggered daily at 6:30am but only actually runs about
# every 60h (see term_review._MIN_HOURS_BETWEEN_RUNS) - allow a full extra
# cycle before flagging so a skipped day isn't mistaken for a failure.
TERM_REVIEW_STALE_HOURS = 96
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8000"))
TERM_SUGGESTIONS_PATH = PROFILE_DIR / "term_suggestions.json"

# alice's launchd jobs have no suffix; every other profile's are
# "<label>.<profile>" — matches the existing plists in ~/Library/LaunchAgents.
_SUFFIX = "" if PROFILE == "alice" else f".{PROFILE}"
LAUNCHD_LABELS = {
    "dashboard": f"com.example.jobsearch.dashboard{_SUFFIX}",
    "ingest": f"com.example.jobsearch.ingest{_SUFFIX}",
    "search-ingest": f"com.example.jobsearch.search-ingest{_SUFFIX}",
    "term-review": f"com.example.jobsearch.term-review{_SUFFIX}",
    "batch-score": f"com.example.jobsearch.batch-score{_SUFFIX}",
}
BATCH_QUEUE_STUCK_MINUTES = 120  # batches normally finish in minutes, 24h worst case
BATCH_ERROR_PATH = PROFILE_DIR / "batch_last_error.json"


def _minutes_since(iso_timestamp: str) -> float:
    ts = datetime.fromisoformat(iso_timestamp).astimezone(MELBOURNE)
    return (datetime.now(MELBOURNE) - ts).total_seconds() / 60


def _check_email_stale() -> str | None:
    run = latest_ingestion_run_for_email()
    if not run:
        return "Email ingest has never completed a run for this profile."
    minutes_ago = _minutes_since(run["checked_at"])
    if minutes_ago > EMAIL_STALE_MINUTES:
        return f"Email ingest last ran {minutes_ago:.0f} min ago (threshold {EMAIL_STALE_MINUTES} min)."
    return None


def _check_search_stale() -> str | None:
    if not load_prefs().get("search_enabled", True):
        return None  # off on purpose — not a problem
    run = latest_ingestion_run_for_search()
    if not run:
        return "Active search is enabled but has never completed a run for this profile."
    minutes_ago = _minutes_since(run["checked_at"])
    if minutes_ago > SEARCH_STALE_MINUTES:
        return f"Search ingest last ran {minutes_ago:.0f} min ago (threshold {SEARCH_STALE_MINUTES} min)."
    return None


def _check_dashboard_unreachable() -> str | None:
    """Detection only — recovery lives in sidecar.py, a separate minimal
    always-on server (not FastAPI/uvicorn, the stack that hung in the
    first place) with its own restart button, reachable even when the
    dashboard itself is hung. This check just points you at it."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{DASHBOARD_PORT}/", timeout=5)
        return None
    except Exception as exc:
        sidecar_port = DASHBOARD_PORT + 1000
        return (
            f"Dashboard on port {DASHBOARD_PORT} did not respond: {exc} — "
            f"fix it at http://127.0.0.1:{sidecar_port}/ (or your Tailscale address, same port)."
        )


def _check_scoring_failing() -> str | None:
    """Ingest runs finish 'done' even when every score call fails, so the
    staleness checks and the dashboard both look healthy while nothing is
    being scored — exactly what happened 2026-10-03 to 10-05 when the API
    key was revoked (401 on every call). Looks for key/credit failures in
    the last 90 minutes of ingestion_events."""
    cutoff = (datetime.now(MELBOURNE) - timedelta(minutes=90)).astimezone(timezone.utc).isoformat()
    with connection() as conn:
        row = conn.execute(
            "SELECT message FROM ingestion_events WHERE level = 'error' AND created_at > ? "
            "AND (message LIKE '%authentication_error%' OR message LIKE '%credit balance%' "
            "OR message LIKE '%permission_error%') ORDER BY id DESC LIMIT 1",
            (cutoff,),
        ).fetchone()
    if not row:
        return None
    if "credit balance" in row["message"]:
        return "Scoring is failing: Anthropic credit balance is too low — top up at console.anthropic.com."
    return "Scoring is failing: the Anthropic API key is invalid or revoked — create a new key and update users/*/.env."


def _check_batch_queue_stuck() -> str | None:
    """Jobs saved by ingest but never scored: the batch runner isn't draining
    the scoring_queue table. The table only exists once batch scoring has
    been used."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=BATCH_QUEUE_STUCK_MINUTES)).isoformat()
    try:
        with connection() as conn:
            row = conn.execute(
                "SELECT count(*) n, min(created_at) oldest FROM scoring_queue WHERE created_at < ?", (cutoff,)
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row["n"]:
        return (f"{row['n']} job(s) have been waiting for batch scoring for over {BATCH_QUEUE_STUCK_MINUTES} min "
                f"(oldest queued {row['oldest'][:16]} UTC).")
    return None


def _check_batch_api_failing() -> str | None:
    """batch_scoring.py can't write to ingestion_events (no run), so it leaves
    batch_last_error.json on an auth/credit failure and removes it on the next
    clean cycle."""
    if not BATCH_ERROR_PATH.exists():
        return None
    try:
        err = json.loads(BATCH_ERROR_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if "credit balance" in err.get("message", ""):
        return "Batch scoring is failing: Anthropic credit balance is too low - top up at console.anthropic.com."
    return f"Batch scoring is failing: {err.get('message', 'unknown API error')[:120]}"


def _check_term_review_stale() -> str | None:
    """term_review.py writes term_suggestions.json on every successful run
    (even its own early-exit "not enough signal yet" path), so staleness
    of that file's generated_at is a reliable signal the daily job hasn't
    actually completed — confirmed live (2026-10-02) this catches a real
    failure mode: adaptive thinking consuming the whole max_tokens budget
    before writing any output (see CLAUDE_API_COST_NOTES.md)."""
    if not TERM_SUGGESTIONS_PATH.exists():
        return "term_review.py has never produced term_suggestions.json for this profile."
    try:
        data = json.loads(TERM_SUGGESTIONS_PATH.read_text(encoding="utf-8"))
        generated_at = data["generated_at"]
    except (json.JSONDecodeError, OSError, KeyError):
        return "term_suggestions.json exists but is unreadable or missing generated_at."
    hours_ago = _minutes_since(generated_at) / 60
    if hours_ago > TERM_REVIEW_STALE_HOURS:
        return f"term_review.py last succeeded {hours_ago:.0f}h ago (threshold {TERM_REVIEW_STALE_HOURS}h)."
    return None


def _check_launchd_not_loaded() -> dict[str, str]:
    """A job can be stale (still loaded, just not succeeding — caught by
    the staleness checks above) or entirely unloaded (removed from
    launchd's roster some other way — kickstart can't fix that, it needs
    an explicit `launchctl load`). Distinguishing the two matters because
    the dashboard's recovery button only tries kickstart."""
    problems = {}
    for name, label in LAUNCHD_LABELS.items():
        result = subprocess.run(["launchctl", "list", label], capture_output=True)
        if result.returncode != 0:
            problems[name] = f"launchd job '{label}' is not loaded at all."
    return problems


def _notify(title: str, message: str) -> None:
    script = f'display notification {json.dumps(message)} with title {json.dumps(title)} sound name "Basso"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _load_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    problems: dict[str, str] = {}

    email_problem = _check_email_stale()
    if email_problem:
        problems["email_stale"] = email_problem

    search_problem = _check_search_stale()
    if search_problem:
        problems["search_stale"] = search_problem

    dashboard_problem = _check_dashboard_unreachable()
    if dashboard_problem:
        problems["dashboard_unreachable"] = dashboard_problem

    scoring_problem = _check_scoring_failing()
    if scoring_problem:
        problems["scoring_failing"] = scoring_problem

    for key, check in (("batch_queue_stuck", _check_batch_queue_stuck), ("scoring_failing", _check_batch_api_failing)):
        problem = check()
        if problem:
            problems.setdefault(key, problem)

    term_review_problem = _check_term_review_stale()
    if term_review_problem:
        problems["term_review_stale"] = term_review_problem

    for name, msg in _check_launchd_not_loaded().items():
        problems[f"launchd_{name}_unloaded"] = msg

    previous = _load_state()

    new_problems = {k: v for k, v in problems.items() if k not in previous}
    resolved = {k: v for k, v in previous.items() if k not in problems}

    for key, msg in new_problems.items():
        print(f"[{PROFILE}] NEW problem — {key}: {msg}")
        _notify(f"Job Search Assistant ({PROFILE})", msg)

    for key, msg in resolved.items():
        print(f"[{PROFILE}] RECOVERED — {key}")
        _notify(f"Job Search Assistant ({PROFILE}) — recovered", f"No longer a problem: {msg}")

    if not new_problems and not resolved:
        state = "no active problems" if not problems else f"{len(problems)} problem(s) still active (already notified)"
        print(f"[{PROFILE}] {state}")

    STATE_PATH.write_text(json.dumps(problems, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
