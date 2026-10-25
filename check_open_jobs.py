"""Marks listings that are no longer open as EXPIRED (the dashboard's "Job
closed" status), so they drop out of New / To process without counting as a
dislike in the search-term review.

Checks every NEW or TO_PROCESS SEEK and LinkedIn job by fetching its public
listing page:
  SEEK      closed = the page has no ad body and says "no longer advertised"
  LinkedIn  closed = the guest posting page carries LinkedIn's own closed flag
            ("No longer accepting applications")
Anything it cannot tell (network error, unexpected page, other sites) is left
alone and reported as unknown - a wrong "closed" would hide a live job.

Dry run by default; --apply writes the status changes.

Usage: JOBSEARCH_PROFILE=alice python check_open_jobs.py [--apply]
"""

import re
import sys
import time

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


def main() -> None:
    apply = "--apply" in sys.argv
    with connection() as conn:
        jobs = conn.execute(
            "SELECT id, platform, url, title, company FROM jobs WHERE status IN ('NEW', 'TO_PROCESS') ORDER BY id"
        ).fetchall()

    closed, open_count, unknown = [], 0, []
    for job in jobs:
        try:
            result = is_closed(job["platform"], job["url"] or "")
        except RateLimited:
            print(f"[{PROFILE}] LinkedIn rate-limited the checker - stopping here; rerun later for the rest.")
            break
        if result is True:
            closed.append(job)
            print(f"closed   {job['id']:>5} {job['platform']:9s} {job['title'][:48]} - {job['company'][:28]}")
        elif result is False:
            open_count += 1
        else:
            unknown.append(job)
            print(f"unknown  {job['id']:>5} {job['platform']:9s} {job['title'][:48]}")
        time.sleep(PAUSE_SECONDS.get(job["platform"], 0))

    if apply and closed:
        with connection() as conn:
            conn.executemany(
                "UPDATE jobs SET status = 'EXPIRED', status_source = 'job_check' "
                "WHERE id = ? AND status IN ('NEW', 'TO_PROCESS')",
                [(j["id"],) for j in closed],
            )
    print(f"\n[{PROFILE}] checked {open_count + len(closed) + len(unknown)} of {len(jobs)}: "
          f"{open_count} open, {len(closed)} closed{' (marked EXPIRED)' if apply else ' (dry run - use --apply)'}, "
          f"{len(unknown)} unknown")


if __name__ == "__main__":
    main()
