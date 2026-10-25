"""Marks listings that are no longer open as EXPIRED (the dashboard's "Job
closed" status), so they drop out of New / To process without counting as a
dislike in the search-term review.

Checks every NEW or TO_PROCESS SEEK and LinkedIn job by fetching its public
listing page:
  SEEK      closed = the page has no ad body and says "no longer advertised"
  LinkedIn  closed = the guest posting page carries LinkedIn's own closed flag
            ("No longer accepting applications")
Anything it cannot tell (network error, unexpected page) is left alone and
reported as unknown - a wrong "closed" would hide a live job. Other sites
(Indeed, Glassdoor) refuse automated page requests, so they are counted as
unsupported rather than guessed.

Each result is remembered in job_checks: a job checked in the last
RECHECK_AFTER_HOURS is not fetched again, and jobs never checked go first -
so a run that LinkedIn rate-limits part way simply continues next time.

Dry run by default; --apply writes the status changes.

Usage: JOBSEARCH_PROFILE=alice python check_open_jobs.py [--apply]
"""

import collections
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from db import connection
from profile import PROFILE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
SEEK_ID = re.compile(r"/job/(\d+)")
LINKEDIN_ID = re.compile(r"/jobs/view/(\d+)")
PAUSE_SECONDS = {"seek": 1, "linkedin": 2}  # be polite; LinkedIn rate-limits quickly
SUPPORTED_PLATFORMS = set(PAUSE_SECONDS)
RECHECK_AFTER_HOURS = 18

CHECKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_checks (
    job_id INTEGER PRIMARY KEY,
    checked_at TEXT NOT NULL,
    result TEXT NOT NULL            -- open | closed | unknown
)"""


class RateLimited(Exception):
    pass


def _seek_closed(url: str) -> bool | None:
    match = SEEK_ID.search(url)
    if not match:
        return None
    response = requests.get(f"https://www.seek.com.au/job/{match.group(1)}", headers=HEADERS, timeout=20)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    if soup.find(attrs={"data-automation": "jobAdDetails"}):
        return False
    return True if "no longer advertised" in soup.get_text(" ", strip=True).lower() else None


def _linkedin_closed(url: str) -> bool | None:
    match = LINKEDIN_ID.search(url)
    if not match:
        return None
    response = requests.get(
        f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{match.group(1)}", headers=HEADERS, timeout=20,
    )
    if response.status_code == 429:
        raise RateLimited()
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    if soup.select(".closed-job") or "no longer accepting applications" in soup.get_text(" ", strip=True).lower():
        return True
    return False if soup.select_one(".show-more-less-html__markup, .description__text") else None


def is_closed(platform: str, url: str) -> bool | None:
    """True = closed, False = open, None = could not tell."""
    try:
        if platform == "seek":
            return _seek_closed(url)
        if platform == "linkedin":
            return _linkedin_closed(url)
    except requests.RequestException:
        return None
    return None


def check_all(apply: bool, status_source: str = "job_check", verbose: bool = True) -> dict:
    """Checks every NEW / TO_PROCESS job; with apply, marks closed ones
    EXPIRED. Returns counts plus the closed/unknown job rows."""
    with connection() as conn:
        conn.execute(CHECKS_SCHEMA)
        jobs = conn.execute(
            """
            SELECT j.id, j.platform, j.url, j.title, j.company, c.checked_at, c.result
            FROM jobs j LEFT JOIN job_checks c ON c.job_id = j.id
            WHERE j.status IN ('NEW', 'TO_PROCESS')
            ORDER BY c.checked_at IS NOT NULL, c.checked_at, j.id
            """
        ).fetchall()

    fresh_after = (datetime.now(timezone.utc) - timedelta(hours=RECHECK_AFTER_HOURS)).isoformat()
    report = {"total": len(jobs), "open": 0, "closed": [], "unknown": [], "unsupported": collections.Counter(),
              "rate_limited": False, "not_reached": 0}

    for index, job in enumerate(jobs):
        if job["platform"] not in SUPPORTED_PLATFORMS:
            report["unsupported"][job["platform"]] += 1
            continue
        if job["checked_at"] and job["checked_at"] > fresh_after and job["result"] in ("open", "closed"):
            report["open"] += job["result"] == "open"
            continue
        try:
            result = is_closed(job["platform"], job["url"] or "")
        except RateLimited:
            report["rate_limited"] = True
            report["not_reached"] = sum(1 for j in jobs[index:] if j["platform"] in SUPPORTED_PLATFORMS)
            if verbose:
                print(f"[{PROFILE}] LinkedIn rate-limited the checker - stopping; the rest continue next run.")
            break

        label = "closed" if result is True else "open" if result is False else "unknown"
        with connection() as conn:
            conn.execute(
                "INSERT INTO job_checks (job_id, checked_at, result) VALUES (?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET checked_at = excluded.checked_at, result = excluded.result",
                (job["id"], datetime.now(timezone.utc).isoformat(), label),
            )
            if label == "closed" and apply:
                conn.execute(
                    "UPDATE jobs SET status = 'EXPIRED', status_source = ? WHERE id = ? AND status IN ('NEW', 'TO_PROCESS')",
                    (status_source, job["id"]),
                )
        if label == "closed":
            report["closed"].append(dict(job))
            if verbose:
                print(f"closed   {job['id']:>5} {job['platform']:9s} {job['title'][:48]} - {job['company'][:28]}")
        elif label == "open":
            report["open"] += 1
        else:
            report["unknown"].append(dict(job))
            if verbose:
                print(f"unknown  {job['id']:>5} {job['platform']:9s} {job['title'][:48]}")
        time.sleep(PAUSE_SECONDS[job["platform"]])
    return report


def main() -> None:
    apply = "--apply" in sys.argv
    r = check_all(apply)
    unsupported = ", ".join(f"{n} {p}" for p, n in r["unsupported"].items()) or "none"
    print(f"\n[{PROFILE}] {r['total']} jobs: {r['open']} open, {len(r['closed'])} closed"
          f"{' (marked EXPIRED)' if apply else ' (dry run - use --apply)'}, {len(r['unknown'])} unknown, "
          f"unsupported sites: {unsupported}" + (f", {r['not_reached']} not reached (rate limited)" if r["rate_limited"] else ""))


if __name__ == "__main__":
    main()
