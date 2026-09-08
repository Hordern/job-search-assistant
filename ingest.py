"""Read job-alert emails safely and persist deduplicated, scored jobs."""

import fcntl
import hashlib
import imaplib
import os
import re
import ssl
from email import message_from_bytes
from email.message import Message
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from db import (
    emails_ready_to_move, finish_ingestion_run, get_job_id, is_email_processed,
    link_email_job, log_event, mark_email_moved, mark_email_processed, save_job,
    start_ingestion_run,
)
from scorer import score_job
from version import build_version

load_dotenv()

VERSION = build_version("1.2.0")

LINK_PATTERNS = {
    "seek": re.compile(r"(?:au\.seek\.com|seek\.com\.au)/job/(\d+)", re.IGNORECASE),
    "linkedin": re.compile(r"linkedin\.com/(?:comm/)?jobs/view/(\d+)", re.IGNORECASE),
}

# SEEK alert emails never contain a direct seek.com.au/job/ID link — every link is a
# click-tracking redirect through this domain that must be followed to find one.
SEEK_TRACKING_HOSTS = {"email.s.seek.com.au"}


def _resolve_seek_redirect(url: str) -> str | None:
    try:
        with requests.get(url, allow_redirects=True, timeout=10, stream=True) as response:
            return response.url
    except requests.RequestException:
        return None


# A browser UA avoids trivial bot-blocking on the job pages we fetch for full
# descriptions; SEEK server-renders and allows this, LinkedIn requires login
# (confirmed by hand) so it isn't attempted there.
JOB_PAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _fetch_seek_description(url: str) -> str | None:
    """Fetch the full ad text from a SEEK job page. Alert emails only ever
    contain title/company/location/salary — this is what lets scoring see
    actual duties and requirements instead of guessing from a thin preview."""
    try:
        response = requests.get(url, headers=JOB_PAGE_HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    details = soup.find(attrs={"data-automation": "jobAdDetails"})
    if not details:
        return None
    text = details.get_text(" ", strip=True)
    return text[:12000] if text else None


def _html_from_message(message: Message) -> str:
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_type() == "text/html" and not part.get_filename():
            payload = part.get_payload(decode=True) or b""
            return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return ""


def _fingerprint(company: str, title: str, location: str) -> str:
    value = f"{company.strip().lower()}{title.strip().lower()}{location.strip().lower()}"
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _platform_and_id(url: str) -> tuple[str, str] | None:
    if urlparse(url).scheme not in {"http", "https"}:
        return None
    for platform, pattern in LINK_PATTERNS.items():
        match = pattern.search(url)
        if match:
            return platform, match.group(1)
    return None


def _card_richness(strings: list[str]) -> int:
    """Rank how likely a set of stripped_strings is to be the actual job card,
    vs. a decoy anchor pointing at the same job ID (logo, digest header text,
    a bare title-only link, "See all jobs" footer nav). The real card is the
    only one where the second string is "Company · Location" — string count
    alone doesn't distinguish it from e.g. a 2-string digest header."""
    if len(strings) >= 2 and " · " in strings[1]:
        return 2
    return 1 if strings else 0


def _extract_jobs(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, dict[str, str]] = {}
    richness: dict[str, int] = {}
    resolved_cache: dict[str, str | None] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        url = href
        host = (urlparse(href).hostname or "").lower()
        if host in SEEK_TRACKING_HOSTS:
            if href not in resolved_cache:
                resolved_cache[href] = _resolve_seek_redirect(href)
            resolved = resolved_cache[href]
            if not resolved:
                continue
            url = resolved

        details = _platform_and_id(url)
        if not details:
            continue
        platform, external_id = details
        canonical_id = f"{platform}_{external_id}"

        # SEEK/LinkedIn job cards render title, company, and location as separate
        # elements inside the anchor — reading them from the anchor's own
        # stripped_strings (not get_text, which glues adjacent elements together
        # with no separator, and not the parent, which can span sibling job cards)
        # keeps each field clean and scoped to this one job. A single job ID can
        # appear in several anchors (LinkedIn digests wrap the same job in a logo
        # link, a header link, and the real card) — keep whichever ranks highest.
        strings = list(a.stripped_strings)
        card_score = _card_richness(strings)
        if canonical_id in found and richness[canonical_id] >= card_score:
            continue

        title = strings[0] if strings else "Untitled role"
        company_location = strings[1] if len(strings) > 1 else ""
        company, _, location_from_card = company_location.partition(" · ")
        company = company[:120]

        # Alert emails carry no duties/requirements text at all — this card's own
        # fields (title/company/location/salary) are genuinely all there is per
        # job here. score_job() may later replace this with a fetched SEEK page.
        found[canonical_id] = {
            "canonical_id": canonical_id,
            "platform": platform,
            "title": title,
            "company": company or "Unknown company",
            "location": location_from_card[:120],
            "url": url,
            "raw_description": " | ".join(strings),
        }
        richness[canonical_id] = card_score
    return list(found.values())


def _sweep_folders(mail: imaplib.IMAP4_SSL, run_id: int, inbox_folder: str) -> None:
    """Move emails out of Inbox once every job they contained has been
    actioned on the dashboard (kept/applied/discarded, none still New). That
    only ever happens after ingest runs, via dashboard clicks — so each run
    re-checks for emails that have since become eligible."""
    ready = emails_ready_to_move()
    if not ready:
        return
    status, _ = mail.select(inbox_folder, readonly=False)
    if status != "OK":
        log_event(run_id, "error", f"Could not reselect {inbox_folder!r} writable for folder moves")
        return
    moved_any = False
    for uid, target_folder in ready:
        mail.create(target_folder)  # no-op (server returns NO) if it already exists
        copy_status, _ = mail.uid("COPY", str(uid), target_folder)
        if copy_status != "OK":
            log_event(run_id, "error", f"Could not copy UID {uid} to {target_folder}")
            continue
        mail.uid("STORE", str(uid), "+FLAGS", "(\\Deleted)")
        mark_email_moved(uid, target_folder)
        log_event(run_id, "info", f"Moved UID {uid} to {target_folder}")
        moved_any = True
    if moved_any:
        mail.expunge()


def _describe_scoring_error(exc: Exception) -> str:
    """Turn an Anthropic SDK / scorer exception into a message that names the
    actual cause (token limit, rate limit, malformed response, ...) instead of
    a bare repr — this is what surfaces on the dashboard's live progress feed."""
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited by the Anthropic API"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error ({exc.status_code}): {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "network error reaching the Anthropic API"
    if isinstance(exc, ValueError):
        return f"model response was malformed or truncated (possibly hit max_tokens): {exc}"
    return str(exc)


def ingest_alerts() -> dict[str, int]:
    """Fetch unprocessed alerts. Failed messages remain unmarked for a later retry."""
    # IMAP_HOST/IMAP_PASSWORD keep this compatible with the earlier digest setup.
    imap_server = os.environ.get("IMAP_SERVER") or os.environ.get("IMAP_HOST")
    imap_user = os.environ.get("IMAP_USER")
    imap_pass = os.environ.get("IMAP_PASS") or os.environ.get("IMAP_PASSWORD")
    missing = [
        name for name, value in (("IMAP_SERVER", imap_server), ("IMAP_USER", imap_user), ("IMAP_PASS", imap_pass))
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    counts = {
        "emails_seen": 0, "emails_processed": 0, "job_links_found": 0,
        "jobs_saved": 0, "duplicates": 0, "errors": 0,
    }
    folder = os.environ.get("IMAP_FOLDER", "INBOX")
    run_id = start_ingestion_run(folder, VERSION)
    log_event(run_id, "info", f"Starting run against {folder} (worker {VERSION})")
    try:
        with imaplib.IMAP4_SSL(imap_server) as mail:
            mail.login(imap_user, imap_pass)
            status, _ = mail.select(folder, readonly=True)
            if status != "OK":
                raise RuntimeError(f"Unable to select IMAP folder {folder!r}")
            status, data = mail.uid("search", None, "ALL")
            if status != "OK":
                raise RuntimeError("Unable to search IMAP folder")

            uids = data[0].split()
            log_event(run_id, "info", f"{len(uids)} message(s) in mailbox")

            # A run of failures in a row — regardless of exception type — is a much
            # stronger signal of a broken connection/session than N unrelated emails
            # each independently failing. Catching specific exception types (below)
            # misses variants like a non-OK fetch status that raises a plain
            # RuntimeError; this backstop stops the doomed-retry loop either way.
            consecutive_failures = 0
            CONSECUTIVE_FAILURE_LIMIT = 5

            for raw_uid in uids:
                counts["emails_seen"] += 1
                uid = int(raw_uid)
                if is_email_processed(uid):
                    continue
                try:
                    status, fetched = mail.uid("fetch", raw_uid, "(BODY.PEEK[])")
                    if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
                        raise RuntimeError("Unable to fetch message body")
                    html = _html_from_message(message_from_bytes(fetched[0][1]))
                    jobs = _extract_jobs(html)
                    counts["job_links_found"] += len(jobs)
                    log_event(run_id, "info", f"UID {uid}: {len(jobs)} job link(s) found")
                    for job in jobs:
                        job["fingerprint"] = _fingerprint(job["company"], job["title"], job["location"])
                        existing_job_id = get_job_id(job["canonical_id"], job["fingerprint"])
                        if existing_job_id is not None:
                            counts["duplicates"] += 1
                            link_email_job(uid, existing_job_id)
                            continue
                        full_description = None
                        if job["platform"] == "seek":
                            full_description = _fetch_seek_description(job["url"])
                            if full_description:
                                job["raw_description"] = full_description

                        if not full_description:
                            # No real description available — scoring on title/company
                            # alone is closer to a guess than an assessment, and it
                            # still costs a real API call. Leave it unscored rather
                            # than spend money on low-information output; it can be
                            # rescored later (LinkedIn bookmarklet, or a future SEEK
                            # fetch that succeeds).
                            job["ai_evaluation"] = None
                            job["overall_score"] = None
                            new_job_id = save_job(job)
                            if new_job_id is not None:
                                counts["jobs_saved"] += 1
                                link_email_job(uid, new_job_id)
                                hint = (
                                    "no full listing available — use the LinkedIn rescore bookmarklet"
                                    if job["platform"] == "linkedin"
                                    else "SEEK fetch failed — left unscored, may succeed on a later run"
                                )
                                log_event(
                                    run_id, "info",
                                    f"Skipped scoring \"{job['title']}\" ({hint})",
                                )
                            continue

                        log_event(run_id, "info", f"Scoring \"{job['title']}\" at {job['company']}…")
                        try:
                            evaluation = score_job(job["title"], job["company"], job["raw_description"])
                        except Exception as exc:
                            log_event(
                                run_id, "error",
                                f"Failed to score \"{job['title']}\": {_describe_scoring_error(exc)}",
                            )
                            raise
                        evaluation["description_source"] = "full"
                        job["ai_evaluation"] = evaluation
                        job["overall_score"] = evaluation["overall_score"]
                        new_job_id = save_job(job)
                        if new_job_id is not None:
                            counts["jobs_saved"] += 1
                            link_email_job(uid, new_job_id)
                            log_event(
                                run_id, "success",
                                f"Scored \"{job['title']}\": {job['overall_score']}/100 — "
                                f"{evaluation.get('recommendation')}",
                            )
                    mark_email_processed(uid)
                    counts["emails_processed"] += 1
                    consecutive_failures = 0
                except (imaplib.IMAP4.abort, OSError, ssl.SSLError) as exc:
                    # The connection itself is dead — every remaining UID would fail
                    # identically. Stop now instead of looping through the rest of
                    # the mailbox one doomed fetch at a time; none of them got
                    # marked processed, so the next scheduled run picks them up.
                    counts["errors"] += 1
                    log_event(run_id, "error", f"IMAP connection lost, stopping this run early: {exc}")
                    print(f"IMAP connection lost, stopping this run early: {exc}")
                    break
                except Exception as exc:
                    counts["errors"] += 1
                    consecutive_failures += 1
                    log_event(run_id, "error", f"UID {uid} was not processed: {exc}")
                    print(f"UID {uid} was not processed: {exc}")
                    if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                        log_event(
                            run_id, "error",
                            f"{consecutive_failures} consecutive failures — assuming the "
                            "connection/session is broken, stopping this run early",
                        )
                        break

            try:
                _sweep_folders(mail, run_id, folder)
            except Exception as exc:
                log_event(run_id, "error", f"Folder sweep failed: {exc}")
        finish_ingestion_run(run_id, counts, "done")
        log_event(run_id, "info", "Run finished")
    except Exception as exc:
        log_event(run_id, "error", f"Run aborted: {exc}")
        finish_ingestion_run(run_id, counts, "error")
        raise
    return counts


LOCK_PATH = Path(__file__).with_name(".ingest.lock")


if __name__ == "__main__":
    # Guards against overlapping runs once this is on a fixed schedule (e.g.
    # launchd every 15 min) — a run that takes longer than the interval
    # (a large backlog, a slow API) would otherwise start a second instance
    # hitting the same mailbox and API concurrently.
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another ingest run is already in progress — skipping.")
    else:
        print(f"JobHunter worker {VERSION}")
        print(ingest_alerts())
