"""Mobile-friendly FastAPI dashboard for JobHunter."""

import csv
import html
import io
import json
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from commute import get_commute
from db import connection, get_job_id, init_db, latest_ingestion_run, run_events, save_job, toggle_favorite
from ingest import fetch_seek_description, fingerprint
from profile import PROFILE, PROFILE_DIR
from scorer import load_prefs, score_job
from version import build_version

VERSION = build_version("1.2.0")
MELBOURNE = ZoneInfo("Australia/Melbourne")
INGEST_SCRIPT = Path(__file__).with_name("ingest.py")
# The bookmarklets (below) run in the user's browser and POST back to this
# dashboard — they need to know which port THIS profile's dashboard actually
# runs on, since two profiles can't share one port on the same machine. Set
# DASHBOARD_PORT in the profile's .env to match whatever port its uvicorn/
# launchd service actually binds; defaults to 8000 (Alice's).
DASHBOARD_PORT = os.environ.get("DASHBOARD_PORT", "8000")

app = FastAPI(title=f"JobHunter v{VERSION} — {PROFILE}")
init_db()

# Scoped narrowly to LinkedIn — the /rescore bookmarklet runs in the context
# of a LinkedIn job page and needs to POST here across origins. No other
# route needs this; the dashboard otherwise has no external callers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.linkedin.com", "https://linkedin.com"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
    allow_private_network=True,
)


class JobAction(BaseModel):
    job_id: int
    action: Literal["DISCARDED", "TO_PROCESS", "APPLIED", "DECLINED"]


class FavoritePayload(BaseModel):
    job_id: int


class RescorePayload(BaseModel):
    canonical_id: str
    raw_description: str


class SeekJobEntry(BaseModel):
    job_id: str
    title: str
    company: str
    location: str = ""


class AddSeekPayload(BaseModel):
    jobs: list[SeekJobEntry]


class AddGlassdoorPayload(BaseModel):
    job_id: str
    title: str
    company: str
    location: str = ""
    url: str
    raw_description: str


# Each section's action buttons: (new status, button label, button CSS classes).
SECTION_ACTIONS: dict[str, list[tuple[str, str, str]]] = {
    "NEW": [
        ("TO_PROCESS", "Keep", "rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white"),
        ("DISCARDED", "Discard", "rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium"),
    ],
    "TO_PROCESS": [
        ("APPLIED", "Mark Applied", "rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white"),
        ("DISCARDED", "Discard", "rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium"),
    ],
    "APPLIED": [
        ("DECLINED", "Decline", "rounded-lg border border-red-300 px-3 py-2 text-sm font-medium text-red-600"),
    ],
}


def _age_badge(date_posted: str | None) -> str:
    """SEEK gives a relative string already ("Posted 22h ago") — display
    verbatim. Indeed/LinkedIn (via JobSpy) give an ISO date ("2026-09-23") —
    compute the relative age ourselves. No badge at all if the source
    didn't expose a posting date (nothing to show, never guessed)."""
    if not date_posted:
        return ""
    if date_posted.lower().startswith("posted"):
        return f'<span class="rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">{html.escape(date_posted)}</span>'
    try:
        posted = datetime.strptime(date_posted[:10], "%Y-%m-%d").date()
    except ValueError:
        return ""
    days = (datetime.now(MELBOURNE).date() - posted).days
    if days <= 0:
        label = "Posted today"
    elif days == 1:
        label = "Posted 1 day ago"
    else:
        label = f"Posted {days} days ago"
    return f'<span class="rounded-full bg-violet-100 px-2 py-0.5 text-violet-700">{html.escape(label)}</span>'


def _eligibility_badges(evaluation: dict) -> str:
    """Only surface eligibility items worth a second look — "met" is the
    default expectation and would just be noise on every card."""
    eligibility = evaluation.get("eligibility") or {}
    cls_by_status = {
        "unmet": "bg-red-100 text-red-700",
        "action_required": "bg-amber-100 text-amber-700",
        "unknown": "bg-slate-100 text-slate-600",
    }
    badges = []
    for etype, status in eligibility.items():
        if status not in cls_by_status:
            continue
        label = etype.replace("_", " ").title()
        badges.append(f'<span class="rounded-full {cls_by_status[status]} px-2 py-0.5">{html.escape(label)}: {html.escape(status)}</span>')
    return "".join(badges)


def _distance_badge(evaluation: dict) -> str:
    location_text = (evaluation.get("job_extraction") or {}).get("location_text")
    if not location_text:
        return ""
    home_suburb = (load_prefs().get("location") or {}).get("home_suburb")
    if not home_suburb:
        return ""
    commute = get_commute(home_suburb, location_text)
    if not commute:
        return ""
    return f'<span class="rounded-full bg-sky-100 px-2 py-0.5 text-sky-700">~{commute["distance_km"]} km (straight-line)</span>'


def _scored_card(row: sqlite3.Row, buttons_html: str) -> str:
    evaluation = json.loads(row["ai_evaluation"])
    gaps = evaluation.get("gaps") or []
    gap_html = "".join(f"<li>{html.escape(str(gap))}</li>" for gap in gaps) or "<li>No gaps listed.</li>"
    fit_tag = lambda label, value, cls: (
        f'<span class="rounded-full {cls} px-2 py-0.5">{label}: {html.escape(str(value))}</span>'
    )
    salary_fit = str(evaluation.get("salary_fit") or "unclear from preview")
    location_fit = str(evaluation.get("location_fit") or "unclear from preview")
    fit_cls = lambda v: (
        "bg-emerald-100 text-emerald-700" if v in ("great", "good")
        else "bg-amber-100 text-amber-700" if v == "consider"
        else "bg-red-100 text-red-700" if v == "poor"
        else "bg-slate-100 text-slate-600"
    )
    status_label = str(row["status"]).replace("_", " ").title()
    is_favorited = bool(row["favorited"])
    star_button = f'''<button onclick="toggleFavorite({row['id']}, this)" data-favorited="{"1" if is_favorited else "0"}" class="shrink-0 text-xl leading-none {"text-rose-500" if is_favorited else "text-slate-300 hover:text-slate-400"}" title="{"Remove from favourites" if is_favorited else "Add to favourites"}">{"♥" if is_favorited else "♡"}</button>'''
    return f'''<article data-scored="1" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><div class="flex items-center gap-2">{star_button}<span class="rounded-full bg-indigo-100 px-3 py-1 text-sm font-bold text-indigo-700">Score {row['overall_score'] or 0}</span></div></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-indigo-700">{html.escape(str(evaluation.get('recommendation') or 'Unrated'))} · {html.escape(str(row['platform'] or ''))} · {html.escape(status_label)}</p>
  <p class="mt-4 text-sm"><span class="font-semibold">Pitch angle:</span> {html.escape(str(evaluation.get('pitch_angle') or 'Not available'))}</p>
  <div class="mt-3 flex flex-wrap gap-2 text-xs">
    <span class="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">CV fit: {evaluation.get('cv_fit_score', '—')}</span>
    {fit_tag('Salary', salary_fit, fit_cls(salary_fit))}
    {fit_tag('Location', location_fit, fit_cls(location_fit))}
    {_distance_badge(evaluation)}
    {_age_badge(row['date_posted'])}
    {_eligibility_badges(evaluation)}
  </div>
  <p class="mt-3 text-sm font-semibold">Gaps</p><ul class="ml-5 list-disc text-sm text-slate-600">{gap_html}</ul>
  <div class="mt-5 flex flex-wrap gap-2"><a class="rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white" href="{html.escape(row['url'] or '', quote=True)}" target="_blank" rel="noopener noreferrer">View</a>{buttons_html}</div>
</article>'''


def _unscored_card(row: sqlite3.Row, buttons_html: str) -> str:
    hint = (
        "No full listing available yet — open it on LinkedIn while logged in and use the "
        "rescore bookmarklet above to get a real score."
        if row["platform"] == "linkedin"
        else "The full listing fetch didn't succeed — a later run may pick it up automatically."
    )
    status_label = str(row["status"]).replace("_", " ").title()
    return f'''<article data-scored="0" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200 border border-dashed border-slate-300">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><span class="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-500">Not scored</span></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{html.escape(str(row['platform'] or ''))} · {html.escape(status_label)}</p>
  <p class="mt-3 text-sm text-slate-500">{html.escape(hint)}</p>
  <div class="mt-5 flex flex-wrap gap-2"><a class="rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white" href="{html.escape(row['url'] or '', quote=True)}" target="_blank" rel="noopener noreferrer">View</a>{buttons_html}</div>
</article>'''


_CARD_FIELDS = "id, title, company, platform, url, ai_evaluation, overall_score, created_at, status, date_posted, favorited"

# capability_score is now uncapped (see scorer.py) — a great skills match on
# a job you're genuinely ineligible for can score higher than a merely-okay
# eligible one, so ranking by score alone would put "Ineligible" jobs above
# real options. Rank by recommendation tier first, score only breaks ties
# within a tier. Unrecognised/missing recommendation (shouldn't happen, but
# don't let it silently sort to the top OR bottom) gets a middling rank.
_RECOMMENDATION_RANK = {"Strong Apply": 4, "Apply": 3, "Stretch": 2, "Pass": 1, "Ineligible": 0}


def _sort_key(row: sqlite3.Row) -> tuple:
    evaluation = json.loads(row["ai_evaluation"])
    rec_rank = _RECOMMENDATION_RANK.get(evaluation.get("recommendation"), 2)
    return (rec_rank, row["overall_score"] or 0, row["created_at"])


def _render_jobs(rows: list[sqlite3.Row], empty_message: str) -> str:
    if not rows:
        return f'<p class="rounded-xl bg-white p-5 text-slate-500 shadow">{html.escape(empty_message)}</p>'

    # Scored jobs (a real listing was actually read) always rank above unscored
    # ones (title/company guesswork we deliberately didn't pay an LLM call for)
    # — never interleaved by a blended score.
    scored = sorted(
        (r for r in rows if r["ai_evaluation"] is not None),
        key=_sort_key, reverse=True,
    )
    unscored = sorted(
        (r for r in rows if r["ai_evaluation"] is None),
        key=lambda r: r["created_at"], reverse=True,
    )

    cards = []
    for row in scored + unscored:
        row_id = row["id"]
        actions = SECTION_ACTIONS.get(row["status"], [])
        buttons_html = "".join(
            f'''<button onclick="act({row_id}, '{action}')" class="{classes}">{label}</button>'''
            for action, label, classes in actions
        )
        if row["ai_evaluation"] is not None:
            cards.append(_scored_card(row, buttons_html))
        else:
            cards.append(_unscored_card(row, buttons_html))
    return "\n".join(cards)


def _cards(status: str, empty_message: str) -> str:
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE status = ?", (status,)).fetchall()
    return _render_jobs(rows, empty_message)


def _cards_all(empty_message: str) -> str:
    """New + To-process combined into one score-ranked list — everything still
    active in the pipeline, i.e. not yet Applied and not archived (Discarded/
    Declined, which are deliberately excluded here so they stay out of the way)."""
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE status IN ('NEW','TO_PROCESS')").fetchall()
    return _render_jobs(rows, empty_message)


def _cards_today(empty_message: str) -> str:
    """Same active set as _cards_all, filtered to jobs created today in
    Melbourne time — done in Python rather than SQL since created_at is
    stored as UTC ISO8601 and SQLite's date functions don't handle the
    Melbourne-local cutoff correctly on their own."""
    today = datetime.now(MELBOURNE).date()
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE status IN ('NEW','TO_PROCESS')").fetchall()
    todays_rows = [r for r in rows if datetime.fromisoformat(r["created_at"]).astimezone(MELBOURNE).date() == today]
    return _render_jobs(todays_rows, empty_message)


def _cards_favorites(empty_message: str) -> str:
    """Starred jobs regardless of status — a favourite you've since applied
    to or discarded should still be findable here, not silently dropped."""
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE favorited = 1").fetchall()
    return _render_jobs(rows, empty_message)


# Ingest is scheduled every 15 minutes (see com.example.jobsearch.ingest.plist) —
# 2x that is a stale/stuck signal without being trigger-happy over one slow run.
STALE_THRESHOLD_MINUTES = 30


def _dashboard_data() -> tuple[dict[str, int], str, str, bool]:
    with connection() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    run = latest_ingestion_run()
    if not run:
        return counts, "Not checked yet", "Run ingest.py to check the mailbox.", True
    checked_at = datetime.fromisoformat(run["checked_at"]).astimezone(MELBOURNE)
    checked_text = checked_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    minutes_ago = (datetime.now(MELBOURNE) - checked_at).total_seconds() / 60
    is_stale = minutes_ago > STALE_THRESHOLD_MINUTES
    detail = (
        f"{run['folder']} · {run['emails_seen']} mailbox messages · "
        f"{run['emails_processed']} newly processed · {run['job_links_found']} job links found · "
        f"{run['jobs_saved']} saved · {run['duplicates']} duplicates · {run['errors']} errors · "
        f"worker {run['worker_version'] or 'unknown'}"
    )
    return counts, checked_text, detail, is_stale


# Dragged to the bookmarks bar and clicked on an open LinkedIn job page. Reads
# the description from the poster's own logged-in session (server-side fetch
# gets LinkedIn's login wall instead). It opens a local receiver page rather
# than fetching /rescore directly: browsers block an HTTPS LinkedIn page from
# fetching an HTTP loopback address as mixed/private-network content. The
# receiver is same-origin with this app, so its POST is permitted.
RESCORE_BOOKMARKLET_JS = """(function(){
var m = location.href.match(/jobs\\/view\\/(\\d+)/);
if(!m){alert('JobHunter: open a LinkedIn job page first');return;}
var id = 'linkedin_' + m[1];
var sels = ['.jobs-description__content','.jobs-box__html-content','#job-details','[class*="jobs-description"]','article'];
var el = null;
for (var i=0;i<sels.length;i++){ el = document.querySelector(sels[i]); if (el && el.innerText && el.innerText.length > 200) break; else el = null; }
var text = (el ? el.innerText : document.body.innerText).slice(0, 12000);
if(!confirm('JobHunter: send ' + text.length + ' characters to rescore this job?\\n\\n' + text.slice(0,300) + '...')) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/rescore-bookmarklet#' + encodeURIComponent(JSON.stringify({canonical_id:id, raw_description:text}));
var popup = window.open(receiver, 'jobhunter-rescore', 'popup,width=480,height=360');
if(!popup) location.href = receiver;
})();"""
RESCORE_BOOKMARKLET_HREF = "javascript:" + quote(RESCORE_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


RESCORE_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JobHunter rescore</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}</style>
</head><body><h1>JobHunter</h1><p id="status">Sending the LinkedIn job description for scoring…</p>
<script>
(async function () {
  var status = document.getElementById('status');
  var payload;
  try { payload = JSON.parse(decodeURIComponent(location.hash.slice(1))); }
  catch (_) { status.textContent = 'Could not read the job description. Return to LinkedIn and run the bookmarklet again.'; status.className = 'error'; return; }
  if (!payload || !payload.canonical_id || !payload.raw_description) {
    status.textContent = 'The job description was empty. Return to LinkedIn and run the bookmarklet again.'; status.className = 'error'; return;
  }
  try {
    var response = await fetch('/rescore', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    var data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unknown error');
    status.textContent = 'Rescored: ' + data.overall_score + '/100 (' + data.recommendation + '). You can close this window and refresh JobHunter.';
    status.className = 'success';
  } catch (error) {
    status.textContent = 'Scoring failed: ' + error.message;
    status.className = 'error';
  }
})();
</script></body></html>"""


# Dragged to the bookmarks bar and clicked on either a single SEEK job page or a
# SEEK search-results page. Unlike LinkedIn, SEEK's own pages are server-fetchable
# without login (see fetch_seek_description in ingest.py) — so this only needs to
# scrape the job ID(s) + card text visible on the page, not a full description.
# Selectors confirmed against live au.seek.com markup (job page: job-detail-title /
# advertiser-name / job-detail-location; search page: [data-testid="job-card"]
# with jobTitle / jobCompany / jobCardLocation inside). Same receiver-page/URL-
# fragment relay as the rescore bookmarklet, for the same mixed-content reason.
ADD_SEEK_BOOKMARKLET_JS = """(function(){
var jobs = [];
var singleMatch = location.pathname.match(/\\/job\\/(\\d+)/);
if (singleMatch) {
  var g = function(sel){ var e = document.querySelector(sel); return e ? e.textContent.trim() : ''; };
  jobs.push({
    job_id: singleMatch[1],
    title: g('[data-automation="job-detail-title"]'),
    company: g('[data-automation="advertiser-name"]'),
    location: g('[data-automation="job-detail-location"]')
  });
} else {
  var cards = document.querySelectorAll('[data-testid="job-card"]');
  cards.forEach(function(card){
    var id = card.getAttribute('data-job-id');
    if (!id) return;
    var gc = function(sel){ var e = card.querySelector(sel); return e ? e.textContent.trim() : ''; };
    jobs.push({
      job_id: id,
      title: gc('[data-automation="jobTitle"]'),
      company: gc('[data-automation="jobCompany"]'),
      location: gc('[data-automation="jobCardLocation"]')
    });
  });
}
if (!jobs.length) { alert('JobHunter: no SEEK job(s) found on this page. Open a SEEK job page or search results page first.'); return; }
if (!confirm('JobHunter: add ' + jobs.length + ' SEEK job(s) to the tracker?\\n\\n' + jobs.slice(0,5).map(function(j){return j.title + ' — ' + j.company;}).join('\\n') + (jobs.length > 5 ? '\\n...' : ''))) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/add-seek-bookmarklet#' + encodeURIComponent(JSON.stringify({jobs: jobs}));
var popup = window.open(receiver, 'jobhunter-add-seek', 'popup,width=480,height=420');
if(!popup) location.href = receiver;
})();"""
ADD_SEEK_BOOKMARKLET_HREF = "javascript:" + quote(ADD_SEEK_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


ADD_SEEK_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JobHunter add from SEEK</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}ul{padding-left:1.2rem}</style>
</head><body><h1>JobHunter</h1><p id="status">Adding SEEK job(s) — fetching full listings and scoring, this can take a little while…</p>
<script>
(async function () {
  var status = document.getElementById('status');
  var payload;
  try { payload = JSON.parse(decodeURIComponent(location.hash.slice(1))); }
  catch (_) { status.textContent = 'Could not read the job list. Return to SEEK and run the bookmarklet again.'; status.className = 'error'; return; }
  if (!payload || !payload.jobs || !payload.jobs.length) {
    status.textContent = 'No jobs were captured. Return to SEEK and run the bookmarklet again.'; status.className = 'error'; return;
  }
  try {
    var response = await fetch('/add-seek', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    var data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unknown error');
    var list = data.jobs.map(function(j){ return '<li>' + j.title + (j.score !== null ? ' — ' + j.score + '/100' : ' — unscored (fetch failed)') + '</li>'; }).join('');
    status.innerHTML = 'Added ' + data.added + ' job(s), ' + data.duplicates + ' already tracked.' + (list ? '<ul>' + list + '</ul>' : '') + 'You can close this window and refresh JobHunter.';
    status.className = 'success';
  } catch (error) {
    status.textContent = 'Adding failed: ' + error.message;
    status.className = 'error';
  }
})();
</script></body></html>"""


# Dragged to the bookmarks bar and clicked on a real Glassdoor job page.
# UNVERIFIED against live markup — Glassdoor requires login even to browse
# anonymously (confirmed 2026-09-24 via the browser tool: their homepage
# shows a mandatory sign-in wall before any content loads), so this couldn't
# be checked against real page structure the way SEEK's and LinkedIn's
# selectors were. Runs in the user's own logged-in session (where the login
# wall doesn't apply), but the exact selectors below are a best-effort
# generic guess and may need adjusting the first time this is actually
# tried on a real page — same multi-selector-fallback pattern as the
# LinkedIn rescore bookmarklet, ending in a full-body-text fallback so it
# degrades rather than fails outright if none of the specific selectors hit.
GLASSDOOR_BOOKMARKLET_JS = """(function(){
var m = location.href.match(/[?&](?:jl|jobListingId)=(\\d+)/) || location.pathname.match(/-JV_[A-Za-z0-9]*?(\\d{6,})\\.htm/);
if(!m){alert('JobHunter: could not find a job ID in this URL — open a specific Glassdoor job listing first');return;}
var id = 'glassdoor_' + m[1];
var g = function(sel){ var e = document.querySelector(sel); return e ? e.textContent.trim() : ''; };
var title = g('[data-test="job-title"]') || g('.JobDetails_jobTitle__') || g('h1');
var company = g('[data-test="employer-name"]') || g('.EmployerProfile_employerName__') || g('[class*="employerName"]');
var sels = ['[data-test="jobDescriptionContent"]', '.JobDetails_jobDescription__', '[class*="jobDescription"]', 'article'];
var el = null;
for (var i=0;i<sels.length;i++){ el = document.querySelector(sels[i]); if (el && el.innerText && el.innerText.length > 200) break; else el = null; }
var text = (el ? el.innerText : document.body.innerText).slice(0, 12000);
if(!confirm('JobHunter: add this Glassdoor job (' + text.length + ' characters)?\\n\\n' + (title||'(title not found)') + ' — ' + (company||'(company not found)') + '\\n\\n' + text.slice(0,300) + '...')) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/add-glassdoor-bookmarklet#' + encodeURIComponent(JSON.stringify({job_id:m[1], title:title, company:company, location:'', url:location.href, raw_description:text}));
var popup = window.open(receiver, 'jobhunter-add-glassdoor', 'popup,width=480,height=380');
if(!popup) location.href = receiver;
})();"""
GLASSDOOR_BOOKMARKLET_HREF = "javascript:" + quote(GLASSDOOR_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


GLASSDOOR_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>JobHunter add from Glassdoor</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}</style>
</head><body><h1>JobHunter</h1><p id="status">Adding the Glassdoor job and scoring it…</p>
<script>
(async function () {
  var status = document.getElementById('status');
  var payload;
  try { payload = JSON.parse(decodeURIComponent(location.hash.slice(1))); }
  catch (_) { status.textContent = 'Could not read the job. Return to Glassdoor and run the bookmarklet again.'; status.className = 'error'; return; }
  if (!payload || !payload.job_id || !payload.raw_description) {
    status.textContent = 'The job details were incomplete. Return to Glassdoor and run the bookmarklet again.'; status.className = 'error'; return;
  }
  try {
    var response = await fetch('/add-glassdoor', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    var data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Unknown error');
    status.textContent = data.duplicate ? 'Already tracked.' : ('Added — ' + data.overall_score + '/100 (' + data.recommendation + '). You can close this window and refresh JobHunter.');
    status.className = 'success';
  } catch (error) {
    status.textContent = 'Adding failed: ' + error.message;
    status.className = 'error';
  }
})();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    counts, checked_text, detail, is_stale = _dashboard_data()
    heartbeat_panel_class = "mb-6 rounded-2xl p-5 text-white " + ("bg-red-900" if is_stale else "bg-slate-900")
    heartbeat_dot_class = "inline-block h-2.5 w-2.5 rounded-full mr-2 " + ("bg-red-400" if is_stale else "bg-emerald-400")
    heartbeat_label = (
        f"⚠ No check in over {STALE_THRESHOLD_MINUTES} min — ingest may be stuck or the launchd job isn't running"
        if is_stale else "Healthy — checking every 15 min"
    )
    run = latest_ingestion_run()
    is_running = bool(run and run["status"] == "running")
    stat_tile = lambda label, value: (
        f'<div class="rounded-xl bg-white p-3 shadow-sm"><p class="text-xs text-slate-500">{label}</p>'
        f'<p class="text-2xl font-bold">{value}</p></div>'
    )
    stats_html = "".join([
        stat_tile("New", counts.get("NEW", 0)),
        stat_tile("To process", counts.get("TO_PROCESS", 0)),
        stat_tile("Applied", counts.get("APPLIED", 0)),
        stat_tile("Discarded", counts.get("DISCARDED", 0)),
        stat_tile("Declined", counts.get("DECLINED", 0)),
    ])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>JobHunter — {html.escape(PROFILE)}</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8"><header class="mb-6"><div class="mb-3 inline-flex items-center gap-2.5 rounded-full bg-stone-200/70 px-5 py-2.5 shadow-sm"><span class="text-base font-semibold tracking-wide text-slate-700">{html.escape(PROFILE.capitalize())}</span><span class="h-1 w-1 rounded-full bg-stone-400"></span><span class="text-xs font-normal text-slate-400">JobHunter v{VERSION}</span></div><h1 class="text-3xl font-bold tracking-tight text-slate-900">Daily job digest</h1><p class="mt-2 text-slate-600">Ranked opportunities from your email alerts. The Today and All tabs are each ranked highest-score-first <em>within themselves</em>.</p><p class="mt-2 text-xs text-slate-500">Drag to your bookmarks bar: <a href="{RESCORE_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Rescore LinkedIn job</a> — open a LinkedIn job page while logged in, then click it to score with the full listing. <a href="{ADD_SEEK_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Add SEEK job(s)</a> — open a SEEK job page or search results page, then click it to add that job (or every job on the page) to the tracker. <a href="{GLASSDOOR_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Add Glassdoor job</a> — open a Glassdoor job page while logged in, then click it to add and score that job (unverified selectors — first try may need a follow-up fix).</p><button onclick="resync()" class="mt-3 rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white">Resync now</button><button id="unscored-toggle" onclick="toggleUnscored()" class="mt-3 ml-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium">Show unscored only</button></header><section id="progress-panel" {"" if is_running else "hidden"} class="mb-6 rounded-2xl bg-slate-900 p-5 text-white"><p class="text-xs font-semibold uppercase tracking-wide text-amber-400">Worker running…</p><div id="progress-events" class="mt-2 max-h-48 overflow-y-auto space-y-0.5 font-mono"></div></section><section class="{heartbeat_panel_class}"><p class="text-xs font-semibold uppercase tracking-wide text-slate-300"><span class="{heartbeat_dot_class}"></span>Last checked · Melbourne time</p><p class="mt-1 text-lg font-semibold">{html.escape(checked_text)}</p><p class="mt-1 text-sm font-semibold">{html.escape(heartbeat_label)}</p><p class="mt-2 text-sm text-slate-300">{html.escape(detail)}</p></section><section class="mb-8 grid grid-cols-2 sm:grid-cols-5 gap-3">{stats_html}</section><div class="mb-4 flex gap-2 border-b border-slate-200"><button id="tab-btn-today" onclick="showTab('today')" class="px-4 py-2 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-700">Today</button><button id="tab-btn-all" onclick="showTab('all')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">All</button><button id="tab-btn-favorites" onclick="showTab('favorites')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">♥ Favourites</button><button id="tab-btn-applied" onclick="showTab('applied')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">✓ Applied</button></div><section id="tab-panel-today" class="mb-8"><div class="space-y-4">{_cards_today("No jobs added today yet.")}</div></section><section id="tab-panel-all" hidden class="mb-8"><div class="space-y-4">{_cards_all("Nothing active right now.")}</div></section><section id="tab-panel-favorites" hidden class="mb-8"><div class="space-y-4">{_cards_favorites("No favourites yet — click the heart on a job to save it here.")}</div></section><section id="tab-panel-applied" hidden class="mb-8"><div class="mb-3 flex gap-2"><input type="search" id="applied-search" oninput="filterApplied()" placeholder="Quick find — company or title (e.g. for a call)…" class="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"><a href="/export/applied.csv" class="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium whitespace-nowrap">Export CSV</a></div><div id="applied-cards" class="space-y-4">{_cards("APPLIED", "Nothing applied to yet.")}</div></section></main><script>
async function act(job_id, action) {{
  const r = await fetch('/action', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{job_id, action}})}});
  if (!r.ok) return alert('Could not update job');
  location.reload();
}}
async function toggleFavorite(job_id, btn) {{
  const r = await fetch('/favorite', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{job_id}})}});
  if (!r.ok) return alert('Could not update favourite');
  // A card can appear on several tabs at once (Today/All/Favourites), and
  // the Favourites panel itself is only rendered server-side at page load —
  // a full reload (same fix as act()) is simpler and more correct than
  // hand-syncing every panel's DOM. The tab-persistence fix means this no
  // longer loses your place.
  location.reload();
}}
async function resync() {{
  const r = await fetch('/resync', {{method:'POST'}});
  if (!r.ok) return alert('Could not start resync');
  location.reload();
}}
let showUnscoredOnly = false;
function toggleUnscored() {{
  showUnscoredOnly = !showUnscoredOnly;
  document.querySelectorAll('article[data-scored]').forEach(function(el) {{
    el.hidden = showUnscoredOnly && el.dataset.scored === '1';
  }});
  document.getElementById('unscored-toggle').textContent = showUnscoredOnly ? 'Show all' : 'Show unscored only';
}}
function filterApplied() {{
  const term = document.getElementById('applied-search').value.trim().toLowerCase();
  document.querySelectorAll('#applied-cards article').forEach(function(el) {{
    el.hidden = term.length > 0 && !el.textContent.toLowerCase().includes(term);
  }});
}}
const ACTIVE_TAB_CLASS = 'px-4 py-2 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-700';
const INACTIVE_TAB_CLASS = 'px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500';
function showTab(name) {{
  ['today', 'all', 'favorites', 'applied'].forEach(function(t) {{
    document.getElementById('tab-panel-' + t).hidden = (t !== name);
    document.getElementById('tab-btn-' + t).className = (t === name) ? ACTIVE_TAB_CLASS : INACTIVE_TAB_CLASS;
  }});
  try {{ sessionStorage.setItem('jobhunter-tab', name); }} catch (_) {{}}
}}
(function restoreTab() {{
  var saved = null;
  try {{ saved = sessionStorage.getItem('jobhunter-tab'); }} catch (_) {{}}
  if (saved && saved !== 'today') showTab(saved);
}})();
let currentRunId = null, lastEventId = 0, wasRunning = {str(is_running).lower()};
async function pollProgress() {{
  try {{
    const res = await fetch('/progress?since=' + lastEventId);
    if (!res.ok) return;
    const data = await res.json();
    const panel = document.getElementById('progress-panel');
    const list = document.getElementById('progress-events');
    if (data.run_id !== currentRunId) {{ currentRunId = data.run_id; lastEventId = 0; list.innerHTML = ''; }}
    for (const ev of data.events) {{
      lastEventId = ev.id;
      const div = document.createElement('div');
      const color = ev.level === 'error' ? 'text-red-400' : ev.level === 'success' ? 'text-emerald-400' : 'text-slate-400';
      div.className = 'text-xs ' + color;
      div.textContent = ev.message;
      list.appendChild(div);
    }}
    list.scrollTop = list.scrollHeight;
    const running = data.status === 'running';
    panel.hidden = !running;
    if (wasRunning && !running) {{ location.reload(); return; }}
    wasRunning = running;
  }} catch (e) {{ /* transient network error, ignore */ }}
}}
pollProgress();
setInterval(pollProgress, 2000);
</script></body></html>'''


@app.get("/rescore-bookmarklet", response_class=HTMLResponse)
def rescore_bookmarklet() -> str:
    """Receive bookmarklet data in the URL fragment and score it same-origin.

    Fragments are never sent in the request, which keeps the full LinkedIn
    description out of server logs and avoids URL length limits. JavaScript on
    this local page reads the fragment and POSTs it to ``/rescore``.
    """
    return RESCORE_RECEIVER_HTML


@app.get("/progress")
def progress(since: int = 0) -> dict:
    run = latest_ingestion_run()
    if not run:
        return {"run_id": None, "status": "idle", "events": []}
    events = run_events(run["id"], since)
    return {
        "run_id": run["id"],
        "status": run["status"],
        "events": [{"id": e["id"], "level": e["level"], "message": e["message"]} for e in events],
    }


@app.post("/action")
def action(payload: JobAction) -> dict[str, bool]:
    with connection() as conn:
        result = conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (payload.action, payload.job_id))
    if result.rowcount != 1:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.post("/favorite")
def favorite(payload: FavoritePayload) -> dict:
    new_state = toggle_favorite(payload.job_id)
    if new_state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "favorited": new_state}


@app.post("/rescore")
def rescore(payload: RescorePayload) -> dict:
    """Re-score a job with a full description pulled from the poster's own
    browser session (LinkedIn bookmarklet) — the only way to get LinkedIn's
    real job text, since a server-side fetch just hits their login wall."""
    with connection() as conn:
        row = conn.execute(
            "SELECT title, company FROM jobs WHERE canonical_id = ?", (payload.canonical_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not tracked yet — run ingest first")
        evaluation = score_job(row["title"], row["company"], payload.raw_description)
        evaluation["description_source"] = "full"
        conn.execute(
            "UPDATE jobs SET ai_evaluation = ?, overall_score = ? WHERE canonical_id = ?",
            (json.dumps(evaluation), evaluation["overall_score"], payload.canonical_id),
        )
    return {"ok": True, "overall_score": evaluation["overall_score"], "recommendation": evaluation["recommendation"]}


@app.get("/export/applied.csv")
def export_applied_csv() -> Response:
    """On-demand CSV of every Applied job, for manual import into Google Sheets
    (File > Import) rather than a live sync — simpler and needs no external
    account/auth."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT title, company, platform, url, overall_score, ai_evaluation, created_at "
            "FROM jobs WHERE status = 'APPLIED' ORDER BY created_at DESC"
        ).fetchall()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Title", "Company", "Platform", "URL", "Score", "Recommendation", "Applied/Added"])
    for row in rows:
        evaluation = json.loads(row["ai_evaluation"]) if row["ai_evaluation"] else {}
        writer.writerow([
            row["title"], row["company"], row["platform"], row["url"],
            row["overall_score"] or "", evaluation.get("recommendation", ""), row["created_at"],
        ])
    return Response(
        content=buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jobhunter_applied.csv"},
    )


@app.get("/add-seek-bookmarklet", response_class=HTMLResponse)
def add_seek_bookmarklet() -> str:
    """Receive add-seek bookmarklet data in the URL fragment and add it
    same-origin, for the same reason as /rescore-bookmarklet above."""
    return ADD_SEEK_RECEIVER_HTML


@app.post("/add-seek")
def add_seek(payload: AddSeekPayload) -> dict:
    """Add job(s) scraped from a SEEK page (job or search-results) that the
    email-alert pipeline never saw. SEEK's own pages are server-fetchable
    without login, so — unlike /rescore — this fetches the full listing and
    scores it itself rather than relying on browser-scraped text."""
    added = []
    duplicates = 0
    for entry in payload.jobs:
        canonical_id = f"seek_{entry.job_id}"
        url = f"https://au.seek.com/job/{entry.job_id}"
        job_fingerprint = fingerprint(entry.company, entry.title, entry.location)
        if get_job_id(canonical_id, job_fingerprint) is not None:
            duplicates += 1
            continue

        job = {
            "canonical_id": canonical_id, "fingerprint": job_fingerprint, "platform": "seek",
            "title": entry.title, "company": entry.company, "location": entry.location,
            "url": url, "raw_description": "", "ai_evaluation": None, "overall_score": None,
        }
        seek_result = fetch_seek_description(url)
        if seek_result:
            description = seek_result["description"]
            job["raw_description"] = description
            job["date_posted"] = seek_result["date_posted"]
            evaluation = score_job(entry.title, entry.company, description)
            evaluation["description_source"] = "full"
            job["ai_evaluation"] = evaluation
            job["overall_score"] = evaluation["overall_score"]

        new_job_id = save_job(job)
        if new_job_id is not None:
            added.append({"title": entry.title, "score": job["overall_score"]})
        else:
            duplicates += 1

    return {"ok": True, "added": len(added), "duplicates": duplicates, "jobs": added}


@app.get("/add-glassdoor-bookmarklet", response_class=HTMLResponse)
def add_glassdoor_bookmarklet() -> str:
    """Receive add-glassdoor bookmarklet data in the URL fragment and add it
    same-origin, for the same reason as /rescore-bookmarklet above."""
    return GLASSDOOR_RECEIVER_HTML


@app.post("/add-glassdoor")
def add_glassdoor(payload: AddGlassdoorPayload) -> dict:
    """Add a job captured from a real Glassdoor page by the bookmarklet.
    Glassdoor blocks server-side fetching entirely (403, confirmed) — unlike
    /add-seek, there's no fallback fetch here; the browser-scraped text
    (from the user's own logged-in session) is the only source of truth."""
    canonical_id = f"glassdoor_{payload.job_id}"
    job_fingerprint = fingerprint(payload.company, payload.title, payload.location)
    if get_job_id(canonical_id, job_fingerprint) is not None:
        return {"ok": True, "duplicate": True}

    job = {
        "canonical_id": canonical_id, "fingerprint": job_fingerprint, "platform": "glassdoor",
        "title": payload.title, "company": payload.company, "location": payload.location,
        "url": payload.url, "raw_description": payload.raw_description,
        "ai_evaluation": None, "overall_score": None,
    }
    evaluation = score_job(payload.title, payload.company, payload.raw_description)
    evaluation["description_source"] = "full"
    job["ai_evaluation"] = evaluation
    job["overall_score"] = evaluation["overall_score"]

    new_job_id = save_job(job)
    if new_job_id is None:
        return {"ok": True, "duplicate": True}
    return {"ok": True, "duplicate": False, "overall_score": evaluation["overall_score"], "recommendation": evaluation["recommendation"]}


@app.post("/resync")
def resync() -> dict[str, bool]:
    """Force-stop any in-progress worker run and launch a fresh one. Useful
    when a run is stuck/hung on a bad connection rather than waiting out the
    next scheduled launchd trigger.

    Matches on "<script path> <profile>" rather than just the script path —
    every profile's ingest.py lives at the identical path, so a bare path
    match would pkill every profile's run indiscriminately, not just this
    one. Both the launchd-scheduled runs and this manual trigger always pass
    the profile as a trailing argument (ingest.py itself ignores it — profile
    resolution is via JOBSEARCH_PROFILE — it exists purely so pkill/pgrep can
    tell profiles' processes apart)."""
    match_pattern = f"{INGEST_SCRIPT} {PROFILE}"
    subprocess.run(["pkill", "-f", match_pattern])
    # SIGTERM isn't instant — wait for the old process to actually release its
    # flock before launching a new one, or it just sees the lock held and skips.
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", match_pattern], capture_output=True).returncode != 0:
            break
        time.sleep(0.1)
    (PROFILE_DIR / ".ingest.lock").unlink(missing_ok=True)
    subprocess.Popen(
        [sys.executable, str(INGEST_SCRIPT), PROFILE],
        cwd=INGEST_SCRIPT.parent,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True}
