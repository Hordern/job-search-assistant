"""Search-based ingestion via JobSpy — a separate mechanism from the
email-based ingest.py. Runs a keyword+location search directly against a
platform (Indeed, LinkedIn) instead of reading alert emails, which sidesteps
needing a working link-extraction+fetch path for platforms that block
server-side fetching of individual job pages (see LINK_PATTERNS in
ingest.py, which only covers SEEK).

Verified live (2026-09-24): Indeed and LinkedIn both return full description
text this way (LinkedIn needs linkedin_fetch_description=True). Glassdoor
does NOT work — JobSpy's Glassdoor module returns a 400 "location not
parsed" error regardless of location format, on every query tried — this
looks like a library-level bug (likely from Glassdoor's July 2026 merger
into Indeed changing their backend), not something fixable here. Glassdoor
is still a valid, configurable connector (the architecture supports it for
any profile), it just currently returns zero results every run until
JobSpy's library catches up.

Every connector is available to every profile — which ones actually run is
entirely controlled by the "connectors" block in that profile's
preferences.json, never hardcoded here.

Usage:
    JOBSEARCH_PROFILE=bob python search_ingest.py
"""

import re
from typing import Any

from jobspy import scrape_jobs

from db import finish_ingestion_run, get_job_id, log_event, save_job, start_ingestion_run
from ingest import fingerprint
from scorer import load_prefs, score_job
from version import build_version

VERSION = build_version("1.2.0")

SEARCH_CONNECTORS = ("linkedin", "indeed", "glassdoor")

_JOB_URL_ID_PATTERNS = {
    "indeed": re.compile(r"jk=([a-f0-9]+)"),
    "linkedin": re.compile(r"/jobs/view/(\d+)"),
    # Not yet verified against a real result — Glassdoor doesn't currently
    # return anything to test this against (see module docstring).
    "glassdoor": re.compile(r"[?&](?:jobListingId|jl)=(\d+)"),
}

_MIN_DESCRIPTION_LENGTH = 200  # below this, scoring would be closer to a guess than an assessment


def _job_id_from_url(platform: str, url: str) -> str | None:
    pattern = _JOB_URL_ID_PATTERNS.get(platform)
    if not pattern:
        return None
    m = pattern.search(url)
    return m.group(1) if m else None


def _empty_counts() -> dict[str, int]:
    return {"emails_seen": 0, "emails_processed": 0, "job_links_found": 0, "jobs_saved": 0, "duplicates": 0, "errors": 0}


def search_ingest_connector(platform: str, config: dict[str, Any], results_wanted: int = 15) -> dict[str, int]:
    """Run every configured search term for one connector, dedupe and score
    new results. Uses the same counts shape as ingest.ingest_alerts() so the
    dashboard's existing run-history view works unchanged — "emails_seen"
    etc. are a slight misnomer here (there's no email involved) but reusing
    the schema avoids a migration for what's conceptually the same thing:
    candidates considered this run.

    results_wanted defaults to 15/term for production; pass a small number
    when testing — each new (non-duplicate) result triggers a real, billed
    scoring call, so a full-size test run against N search terms can mean
    up to N*results_wanted real API calls on the very first run."""
    counts = _empty_counts()
    search_terms = config.get("search_terms") or []
    location = config.get("search_location") or ""
    if not config.get("enabled") or not search_terms:
        return counts

    run_id = start_ingestion_run(f"search:{platform}", VERSION)
    log_event(run_id, "info", f"Starting {platform} search ingest — terms: {search_terms}")
    status = "ok"

    for term in search_terms:
        try:
            jobs = scrape_jobs(
                site_name=[platform], search_term=term, location=location,
                country_indeed="Australia", results_wanted=results_wanted, description_format="markdown",
                linkedin_fetch_description=(platform == "linkedin"),
            )
        except Exception as exc:
            log_event(run_id, "error", f"{platform} search for {term!r} failed: {type(exc).__name__}: {exc}")
            counts["errors"] += 1
            continue

        log_event(run_id, "info", f"{platform} search {term!r}: {len(jobs)} result(s)")

        for _, row in jobs.iterrows():
            counts["emails_seen"] += 1
            url = str(row.get("job_url") or "")
            job_id = _job_id_from_url(platform, url)
            if not job_id:
                continue
            canonical_id = f"{platform}_{job_id}"
            title = str(row.get("title") or "")
            company = str(row.get("company") or "")
            loc_text = str(row.get("location") or "")
            fp = fingerprint(company, title, loc_text)

            if get_job_id(canonical_id, fp) is not None:
                counts["duplicates"] += 1
                continue

            counts["job_links_found"] += 1
            description = str(row.get("description") or "")
            date_posted = row.get("date_posted")
            job = {
                "canonical_id": canonical_id, "fingerprint": fp, "platform": platform,
                "title": title, "company": company, "location": loc_text, "url": url,
                "raw_description": description, "ai_evaluation": None, "overall_score": None,
                "date_posted": str(date_posted) if date_posted is not None and str(date_posted) != "NaT" else None,
                "source": "search", "search_term": term,
            }

            if len(description) >= _MIN_DESCRIPTION_LENGTH:
                try:
                    evaluation = score_job(title, company, description)
                    job["ai_evaluation"] = evaluation
                    job["overall_score"] = evaluation["overall_score"]
                except Exception as exc:
                    log_event(run_id, "error", f"Scoring failed for {canonical_id}: {type(exc).__name__}: {exc}")
                    counts["errors"] += 1

            counts["emails_processed"] += 1
            new_id = save_job(job)
            if new_id is not None:
                counts["jobs_saved"] += 1
            else:
                counts["duplicates"] += 1

    finish_ingestion_run(run_id, counts, status)
    log_event(run_id, "info", f"Finished {platform} search ingest: {counts}")
    return counts


# Kept conservative for the SCHEDULED run specifically — with up to 4 search
# terms per connector, the full JobSpy default (15/term) means a first run
# against a brand-new profile could trigger up to 60 real scoring calls per
# connector before dedup has anything to work with. Raise this once the
# steady-state (mostly-duplicates) volume per run is a known quantity.
PRODUCTION_RESULTS_WANTED = 6


def search_ingest_all(results_wanted: int = PRODUCTION_RESULTS_WANTED) -> dict[str, dict[str, int]]:
    """Run every enabled search connector for the current profile — unless
    search_enabled is false, a single master switch checked before any
    per-connector config, so turning search off for a profile doesn't mean
    hunting down and flipping every connector's own "enabled" individually
    (and doesn't touch them — they're preserved for whenever it's switched
    back on)."""
    prefs = load_prefs()
    if not prefs.get("search_enabled", True):
        print("Active search is disabled for this profile (search_enabled: false in preferences.json) — skipping.")
        return {}
    connectors = prefs.get("connectors") or {}
    results = {}
    for platform in SEARCH_CONNECTORS:
        config = connectors.get(platform) or {}
        if config.get("enabled"):
            results[platform] = search_ingest_connector(platform, config, results_wanted=results_wanted)
    return results


if __name__ == "__main__":
    results = search_ingest_all()
    if not results:
        print("No search connectors enabled for this profile.")
    for platform, counts in results.items():
        print(f"{platform}: {counts}")
