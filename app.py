"""Mobile-friendly FastAPI dashboard for Job Search Assistant."""

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
from db import (
    connection, get_job_id, init_db, latest_ingestion_run, latest_ingestion_run_for_email,
    latest_ingestion_run_for_search, record_click, run_events, save_job, search_term_stats,
    toggle_favorite,
)
from ingest import fetch_seek_description, fingerprint
from profile import PROFILE, PROFILE_DIR
import scorer
from scorer import PREFS_PATH, load_prefs, score_job
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

app = FastAPI(title=f"Job Search Assistant v{VERSION} — {PROFILE}")
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


class TermSuggestionApplyPayload(BaseModel):
    add_terms: list[str] = []
    remove_terms: list[str] = []


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


def _source_badge(source: str | None) -> str:
    """No badge for source=None — jobs saved before this column existed,
    left unlabelled rather than guessed (SEEK is always email, but
    LinkedIn/Indeed/Glassdoor rows could be either)."""
    if source == "search":
        return '<span class="rounded-full bg-teal-100 px-2 py-0.5 text-teal-700">🔍 Found by search</span>'
    if source == "email":
        return '<span class="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">📧 Inbox alert</span>'
    return ""


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


_IMPORTANCE_RANK = {"mandatory": 3, "core": 2, "preferred": 1}
_MATCH_STATE_RANK = {"direct": 2, "adjacent": 1}


def _positives(evaluation: dict) -> list[str]:
    """Requirements the CV has real evidence for (direct/adjacent match),
    ranked by how important the requirement is and how strong the evidence
    is — the counterpart to `gaps`, so a card doesn't read as pure deficit."""
    req_by_id = {
        r["requirement_id"]: r for r in (evaluation.get("job_extraction") or {}).get("requirements") or []
    }
    ranked = []
    for m in (evaluation.get("cv_matching") or {}).get("matches") or []:
        state = m.get("match_state")
        if state not in _MATCH_STATE_RANK:
            continue
        req = req_by_id.get(m.get("requirement_id"))
        text = req.get("requirement") if req else None
        if not text:
            continue
        rank = (_IMPORTANCE_RANK.get(req.get("importance"), 0), _MATCH_STATE_RANK[state])
        ranked.append((rank, text))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in ranked[:5]]


def _scored_card(row: sqlite3.Row, buttons_html: str) -> str:
    evaluation = json.loads(row["ai_evaluation"])
    summary = str((evaluation.get("job_extraction") or {}).get("summary") or "").strip()
    summary_html = f'<p class="mt-2 text-sm text-slate-600">{html.escape(summary)}</p>' if summary else ""
    positives = _positives(evaluation)
    positives_html = "".join(f"<li>{html.escape(p)}</li>" for p in positives) or "<li>No confirmed matches yet.</li>"
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
    # Only offered when we have a stored description to rescore from — a
    # LinkedIn stub scored via the bookmarklet's one-off text never gets one
    # saved, so re-running here would just 400.
    # A live status pill sits right next to the button and starts hidden —
    # while a rescore is in flight the button itself is hidden (not just
    # disabled) and this pill shown instead, so there's no interactive-
    # looking element with changing text and no click-race window before
    # the first poll tick lands.
    rescore_button = (
        f'''<button onclick="rescoreJob({row['id']}, this)" data-rescore-btn="{row['id']}" class="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium hover:bg-slate-50" title="Re-run scoring with the latest CV, preferences and stored job description">Rescore</button>'''
        f'''<span data-rescore-pill="{row['id']}" hidden class="rounded-full bg-indigo-100 px-3 py-2 text-sm font-medium text-indigo-700"></span>'''
        if row["raw_description"] else ""
    )
    return f'''<article data-scored="1" data-source="{row['source'] or ''}" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><div class="flex items-center gap-2">{star_button}<span class="rounded-full bg-indigo-100 px-3 py-1 text-sm font-bold text-indigo-700">Score {row['overall_score'] or 0}</span></div></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-indigo-700">{html.escape(str(evaluation.get('recommendation') or 'Unrated'))} · {html.escape(str(row['platform'] or ''))} · {html.escape(status_label)}</p>
  {summary_html}
  <p class="mt-4 text-sm"><span class="font-semibold">Pitch angle:</span> {html.escape(str(evaluation.get('pitch_angle') or 'Not available'))}</p>
  <div class="mt-3 flex flex-wrap gap-2 text-xs">
    <span class="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">CV fit: {evaluation.get('cv_fit_score', '—')}</span>
    {fit_tag('Salary', salary_fit, fit_cls(salary_fit))}
    {fit_tag('Location', location_fit, fit_cls(location_fit))}
    {_distance_badge(evaluation)}
    {_age_badge(row['date_posted'])}
    {_source_badge(row['source'])}
    {_eligibility_badges(evaluation)}
  </div>
  <div class="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
    <div><p class="text-sm font-semibold text-emerald-700">Strengths</p><ul class="ml-5 list-disc text-sm text-slate-600">{positives_html}</ul></div>
    <div><p class="text-sm font-semibold text-slate-700">Gaps</p><ul class="ml-5 list-disc text-sm text-slate-600">{gap_html}</ul></div>
  </div>
  <div class="mt-5 flex flex-wrap gap-2"><a class="rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white" href="{html.escape(row['url'] or '', quote=True)}" target="_blank" rel="noopener noreferrer" onclick="trackClick({row['id']})">View</a>{buttons_html}{rescore_button}</div>
</article>'''


def _unscored_card(row: sqlite3.Row, buttons_html: str) -> str:
    hint = (
        "No full listing available yet — open it on LinkedIn while logged in and use the "
        "rescore bookmarklet above to get a real score."
        if row["platform"] == "linkedin"
        else "The full listing fetch didn't succeed — a later run may pick it up automatically."
    )
    status_label = str(row["status"]).replace("_", " ").title()
    return f'''<article data-scored="0" data-source="{row['source'] or ''}" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200 border border-dashed border-slate-300">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><span class="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-500">Not scored</span></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{html.escape(str(row['platform'] or ''))} · {html.escape(status_label)}</p>
  <p class="mt-3 text-sm text-slate-500">{html.escape(hint)}</p>
  <div class="mt-5 flex flex-wrap gap-2"><a class="rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white" href="{html.escape(row['url'] or '', quote=True)}" target="_blank" rel="noopener noreferrer" onclick="trackClick({row['id']})">View</a>{buttons_html}</div>
</article>'''


_CARD_FIELDS = "id, title, company, platform, url, raw_description, ai_evaluation, overall_score, created_at, status, date_posted, favorited, source"

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
    """The untriaged pipeline — NEW only. Clicking Keep or Discard on a job
    here moves it out to "To process" or the archive respectively, same as
    Today; neither status lingers here once actioned."""
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE status = 'NEW'").fetchall()
    return _render_jobs(rows, empty_message)


def _cards_today(empty_message: str) -> str:
    """Same untriaged set as _cards_all, filtered to jobs created today in
    Melbourne time — done in Python rather than SQL since created_at is
    stored as UTC ISO8601 and SQLite's date functions don't handle the
    Melbourne-local cutoff correctly on their own."""
    today = datetime.now(MELBOURNE).date()
    with connection() as conn:
        rows = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE status = 'NEW'").fetchall()
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
    run = latest_ingestion_run_for_email()
    if not run:
        return counts, "Not checked yet", "Run ingest.py to check the mailbox.", True
    checked_at = datetime.fromisoformat(run["checked_at"]).astimezone(MELBOURNE)
    checked_text = checked_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    minutes_ago = (datetime.now(MELBOURNE) - checked_at).total_seconds() / 60
    is_stale = minutes_ago > STALE_THRESHOLD_MINUTES
    detail = (
        f"{run['emails_seen']} mailbox messages · "
        f"{run['emails_processed']} newly processed · {run['job_links_found']} job links found · "
        f"{run['jobs_saved']} saved · {run['duplicates']} duplicates · {run['errors']} errors · "
        f"worker {run['worker_version'] or 'unknown'}"
    )
    return counts, checked_text, detail, is_stale


# search_ingest.py runs every 30 min (see the search-ingest launchd plist) —
# double that before calling it stale, same 2x-interval margin as email's
# 15-min schedule / 30-min stale threshold.
SEARCH_STALE_THRESHOLD_MINUTES = 60


def _search_status() -> tuple[str, str, bool]:
    """Separate from _dashboard_data()'s email status — active search runs
    on its own schedule via a different mechanism (search_ingest.py, not
    ingest.py), and silently folding the more-recently-run one into a
    single 'Last checked' line made it look like there was only ever one
    ingestion path, which is exactly what prompted this."""
    run = latest_ingestion_run_for_search()
    if not run:
        return "Never run yet", "No active-search connector has completed a run for this profile.", True
    checked_at = datetime.fromisoformat(run["checked_at"]).astimezone(MELBOURNE)
    checked_text = checked_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    minutes_ago = (datetime.now(MELBOURNE) - checked_at).total_seconds() / 60
    is_stale = minutes_ago > SEARCH_STALE_THRESHOLD_MINUTES
    platform = run["folder"].split(":", 1)[-1] if ":" in run["folder"] else run["folder"]
    detail = (
        f"{platform} · {run['job_links_found']} new listing(s) found · {run['jobs_saved']} saved · "
        f"{run['duplicates']} already tracked · {run['errors']} errors · "
        f"worker {run['worker_version'] or 'unknown'}"
    )
    return checked_text, detail, is_stale


TERM_SUGGESTIONS_PATH = PROFILE_DIR / "term_suggestions.json"


def _term_suggestions() -> dict | None:
    """Written by term_review.py's daily run — never generated or edited by
    this process, only read and (via /term-suggestions/apply) cleared."""
    if not TERM_SUGGESTIONS_PATH.exists():
        return None
    try:
        return json.loads(TERM_SUGGESTIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# Shared by the main dashboard's <script> block and the single-job page
# (/job/{id}) — every card, wherever it's rendered, wires its buttons to
# these same functions. Plain (non-f) string: no Python substitution
# needed, so braces are single, not doubled.
CARD_ACTIONS_JS = """
function trackClick(job_id) {
  // Fire-and-forget — must never delay or block the actual navigation to
  // the job posting (target="_blank" already opens it in a new tab
  // regardless of whether this succeeds).
  try { navigator.sendBeacon('/click/' + job_id); } catch (e) { /* best-effort */ }
}
async function act(job_id, action) {
  const r = await fetch('/action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id, action})});
  if (!r.ok) return alert('Could not update job');
  location.reload();
}
// Single source of truth for every rescore in flight for this profile —
// polled regardless of which tab/click started it, so reopening the page
// (or never having clicked anything here at all) still shows accurate
// per-button state and a page-wide badge. A rescore this specific click
// started stays in _locallyHandled until its own fetch resolves, so this
// poller's completion-triggered reload doesn't collide with that call's
// own success/failure handling.
const _locallyHandled = new Set();
let _lastInFlightIds = new Set();

async function pollRescoreQueue() {
  try {
    const r = await fetch('/reprocess-queue');
    if (!r.ok) return;
    const data = await r.json();
    const badge = document.getElementById('rescore-queue-badge');
    const text = document.getElementById('rescore-queue-text');
    const inFlightIds = new Set(data.jobs.map(j => j.job_id));

    // badge/text are absent on single-job pages (no page-wide queue banner
    // there) — button syncing below still applies regardless.
    if (badge && text) {
      if (!data.jobs.length) {
        badge.classList.add('hidden'); badge.classList.remove('inline-flex');
      } else {
        badge.classList.remove('hidden'); badge.classList.add('inline-flex');
        text.textContent = data.jobs.length === 1
          ? 'Rescoring "' + data.jobs[0].title + '" — ' + data.jobs[0].status
          : 'Rescoring ' + data.jobs.length + ' jobs — ' + data.jobs.map(j => j.title).join(', ');
      }
    }

    document.querySelectorAll('[data-rescore-btn]').forEach(btn => {
      const id = parseInt(btn.getAttribute('data-rescore-btn'), 10);
      const pill = document.querySelector('[data-rescore-pill="' + id + '"]');
      const job = data.jobs.find(j => j.job_id === id);
      const wasInFlight = pill && !pill.hidden;
      if (job) {
        // Hide the real button entirely rather than just disabling it — a
        // disabled button with changing text still looks/behaves like a
        // clickable control. Only the pill (no click handler at all) shows
        // while a rescore is running.
        btn.hidden = true;
        if (pill) { pill.hidden = false; pill.textContent = job.status; }
      } else {
        btn.hidden = false;
        if (pill) pill.hidden = true;
        if (wasInFlight && !_locallyHandled.has(id)) {
          // Was in flight, isn't any more, and no local click is already
          // handling its own result — it finished elsewhere/earlier while
          // this page had it open. Reload once to pick up the new score.
          location.reload();
        }
      }
    });

    _lastInFlightIds = inFlightIds;
  } catch (e) { /* best-effort — next poll will pick it back up */ }
}
setInterval(pollRescoreQueue, 1000);
pollRescoreQueue();

async function rescoreJob(job_id, btn) {
  const pill = document.querySelector('[data-rescore-pill="' + job_id + '"]');
  _locallyHandled.add(job_id);
  // Swap immediately, synchronously, before the network round trip —
  // otherwise there's a real window (until the next 1s poll tick) where
  // the button is still visible and clickable.
  btn.hidden = true;
  if (pill) { pill.hidden = false; pill.textContent = 'Waiting…'; }
  let r;
  try {
    r = await fetch('/reprocess/' + job_id, {method:'POST'});
  } finally {
    _locallyHandled.delete(job_id);
  }
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    alert(body.detail || 'Could not rescore job');
    btn.hidden = false;
    if (pill) pill.hidden = true;
    return;
  }
  location.reload();
}
async function toggleFavorite(job_id, btn) {
  const r = await fetch('/favorite', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({job_id})});
  if (!r.ok) return alert('Could not update favourite');
  // A card can appear on several tabs at once (Today/All/Favourites), and
  // the Favourites panel itself is only rendered server-side at page load —
  // a full reload (same fix as act()) is simpler and more correct than
  // hand-syncing every panel's DOM. The tab-persistence fix means this no
  // longer loses your place.
  location.reload();
}
"""


# Dragged to the bookmarks bar and clicked on an open LinkedIn job page. Reads
# the description from the poster's own logged-in session (server-side fetch
# gets LinkedIn's login wall instead). It opens a local receiver page rather
# than fetching /rescore directly: browsers block an HTTPS LinkedIn page from
# fetching an HTTP loopback address as mixed/private-network content. The
# receiver is same-origin with this app, so its POST is permitted.
RESCORE_BOOKMARKLET_JS = """(function(){
var m = location.href.match(/jobs\\/view\\/(\\d+)/);
if(!m){alert('Job Search Assistant: open a LinkedIn job page first');return;}
var id = 'linkedin_' + m[1];
var sels = ['.jobs-description__content','.jobs-box__html-content','#job-details','[class*="jobs-description"]','article'];
var el = null;
for (var i=0;i<sels.length;i++){ el = document.querySelector(sels[i]); if (el && el.innerText && el.innerText.length > 200) break; else el = null; }
var text = (el ? el.innerText : document.body.innerText).slice(0, 12000);
if(!confirm('Job Search Assistant: send ' + text.length + ' characters to rescore this job?\\n\\n' + text.slice(0,300) + '...')) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/rescore-bookmarklet#' + encodeURIComponent(JSON.stringify({canonical_id:id, raw_description:text}));
var popup = window.open(receiver, 'jobhunter-rescore', 'popup,width=480,height=360');
if(!popup) location.href = receiver;
})();"""
RESCORE_BOOKMARKLET_HREF = "javascript:" + quote(RESCORE_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


RESCORE_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Search Assistant rescore</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}</style>
</head><body><h1>Job Search Assistant</h1><p id="status">Sending the LinkedIn job description for scoring…</p>
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
    status.textContent = 'Rescored: ' + data.overall_score + '/100 (' + data.recommendation + '). You can close this window and refresh Job Search Assistant.';
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
if (!jobs.length) { alert('Job Search Assistant: no SEEK job(s) found on this page. Open a SEEK job page or search results page first.'); return; }
if (!confirm('Job Search Assistant: add ' + jobs.length + ' SEEK job(s) to the tracker?\\n\\n' + jobs.slice(0,5).map(function(j){return j.title + ' — ' + j.company;}).join('\\n') + (jobs.length > 5 ? '\\n...' : ''))) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/add-seek-bookmarklet#' + encodeURIComponent(JSON.stringify({jobs: jobs}));
var popup = window.open(receiver, 'jobhunter-add-seek', 'popup,width=480,height=420');
if(!popup) location.href = receiver;
})();"""
ADD_SEEK_BOOKMARKLET_HREF = "javascript:" + quote(ADD_SEEK_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


ADD_SEEK_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Search Assistant add from SEEK</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}ul{padding-left:1.2rem}</style>
</head><body><h1>Job Search Assistant</h1><p id="status">Adding SEEK job(s) — fetching full listings and scoring, this can take a little while…</p>
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
    if (data.job_id) {
      // Single-job click (not a bulk search-results add) — land straight on
      // that job's own page, whether it was newly added or already tracked.
      status.textContent = data.added ? 'Added — opening it now…' : 'Already tracked — opening it now…';
      status.className = 'success';
      location.href = '/job/' + data.job_id;
      return;
    }
    var list = data.jobs.map(function(j){ return '<li>' + j.title + (j.score !== null ? ' — ' + j.score + '/100' : ' — unscored (fetch failed)') + '</li>'; }).join('');
    status.innerHTML = 'Added ' + data.added + ' job(s), ' + data.duplicates + ' already tracked.' + (list ? '<ul>' + list + '</ul>' : '') + '<a href="/">Open Job Search Assistant</a>';
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
if(!m){alert('Job Search Assistant: could not find a job ID in this URL — open a specific Glassdoor job listing first');return;}
var id = 'glassdoor_' + m[1];
var g = function(sel){ var e = document.querySelector(sel); return e ? e.textContent.trim() : ''; };
var title = g('[data-test="job-title"]') || g('.JobDetails_jobTitle__') || g('h1');
var company = g('[data-test="employer-name"]') || g('.EmployerProfile_employerName__') || g('[class*="employerName"]');
var sels = ['[data-test="jobDescriptionContent"]', '.JobDetails_jobDescription__', '[class*="jobDescription"]', 'article'];
var el = null;
for (var i=0;i<sels.length;i++){ el = document.querySelector(sels[i]); if (el && el.innerText && el.innerText.length > 200) break; else el = null; }
var text = (el ? el.innerText : document.body.innerText).slice(0, 12000);
if(!confirm('Job Search Assistant: add this Glassdoor job (' + text.length + ' characters)?\\n\\n' + (title||'(title not found)') + ' — ' + (company||'(company not found)') + '\\n\\n' + text.slice(0,300) + '...')) return;
var receiver = 'http://127.0.0.1:__DASHBOARD_PORT__/add-glassdoor-bookmarklet#' + encodeURIComponent(JSON.stringify({job_id:m[1], title:title, company:company, location:'', url:location.href, raw_description:text}));
var popup = window.open(receiver, 'jobhunter-add-glassdoor', 'popup,width=480,height=380');
if(!popup) location.href = receiver;
})();"""
GLASSDOOR_BOOKMARKLET_HREF = "javascript:" + quote(GLASSDOOR_BOOKMARKLET_JS.replace("__DASHBOARD_PORT__", DASHBOARD_PORT))


GLASSDOOR_RECEIVER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Search Assistant add from Glassdoor</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5;color:#0f172a}#status{padding:1rem;border-radius:.75rem;background:#f1f5f9}.error{background:#fef2f2!important;color:#991b1b}.success{background:#ecfdf5!important;color:#047857}</style>
</head><body><h1>Job Search Assistant</h1><p id="status">Adding the Glassdoor job and scoring it…</p>
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
    status.textContent = data.duplicate ? 'Already tracked — opening it now…' : 'Added — opening it now…';
    status.className = 'success';
    if (data.job_id) { location.href = '/job/' + data.job_id; }
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
    search_checked_text, search_detail, search_is_stale = _search_status()
    search_panel_class = "mb-6 rounded-2xl p-5 text-white " + ("bg-red-900" if search_is_stale else "bg-slate-900")
    search_dot_class = "inline-block h-2.5 w-2.5 rounded-full mr-2 " + ("bg-red-400" if search_is_stale else "bg-emerald-400")
    search_label = (
        f"⚠ No search in over {SEARCH_STALE_THRESHOLD_MINUTES} min — search-ingest may be stuck or the launchd job isn't running"
        if search_is_stale else "Healthy — searching every 30 min"
    )
    suggestions = _term_suggestions()
    has_term_suggestions = bool(suggestions and suggestions.get("has_suggestions"))
    term_suggestion_html = ""
    if has_term_suggestions:
        n_changes = len(suggestions.get("add_terms") or []) + len(suggestions.get("remove_terms") or [])
        term_suggestion_html = (
            f'<a href="/term-suggestions" class="mb-6 block rounded-2xl bg-amber-50 border border-amber-300 p-5 hover:bg-amber-100">'
            f'<p class="text-xs font-semibold uppercase tracking-wide text-amber-700">Search preferences review · {n_changes} suggested change(s)</p>'
            f'<p class="mt-1 text-sm text-amber-900">{html.escape(suggestions.get("summary") or "")}</p>'
            f'<p class="mt-2 text-xs font-semibold text-amber-700">Click to review, edit and apply &rarr;</p>'
            f'</a>'
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
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>Job Search Assistant — {html.escape(PROFILE)}</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8"><header class="mb-6"><div class="mb-3 inline-flex items-center gap-2.5 rounded-full bg-indigo-100 px-5 py-2.5 shadow-sm"><span class="text-base font-semibold tracking-wide text-indigo-900">{html.escape(PROFILE.capitalize())}</span><span class="h-1 w-1 rounded-full bg-indigo-300"></span><span class="text-xs font-normal text-indigo-500">Search Assistant v{VERSION}</span></div><h1 class="text-3xl font-bold tracking-tight text-slate-900">Daily job digest</h1><p class="mt-2 text-slate-600">Ranked opportunities from your email alerts. The Today, All and To process tabs are each ranked highest-score-first <em>within themselves</em>.</p><p class="mt-2 text-xs text-slate-500">Drag to your bookmarks bar: <a href="{RESCORE_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Rescore LinkedIn job</a> — open a LinkedIn job page while logged in, then click it to score with the full listing. <a href="{ADD_SEEK_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Add SEEK job(s)</a> — open a SEEK job page or search results page, then click it to add that job (or every job on the page) to the tracker. <a href="{GLASSDOOR_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Add Glassdoor job</a> — open a Glassdoor job page while logged in, then click it to add and score that job (unverified selectors — first try may need a follow-up fix).</p><button onclick="resync()" class="mt-3 rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white">Resync now</button><button id="unscored-toggle" onclick="toggleUnscored()" class="mt-3 ml-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium">Show unscored only</button><button id="source-filter-toggle" onclick="cycleSourceFilter()" class="mt-3 ml-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium">All sources</button><a href="/search-terms" class="mt-3 ml-2 inline-block rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium">Search term performance</a></header>{term_suggestion_html}<div id="rescore-queue-badge" class="mb-4 hidden items-center gap-2 rounded-full bg-indigo-100 px-4 py-2 text-sm font-medium text-indigo-700"><span class="h-2 w-2 animate-pulse rounded-full bg-indigo-500"></span><span id="rescore-queue-text"></span></div><section id="progress-panel" {"" if is_running else "hidden"} class="mb-6 rounded-2xl bg-slate-900 p-5 text-white"><p class="text-xs font-semibold uppercase tracking-wide text-amber-400">Worker running…</p><div id="progress-events" class="mt-2 max-h-48 overflow-y-auto space-y-0.5 font-mono"></div></section><section class="{heartbeat_panel_class}"><p class="text-xs font-semibold uppercase tracking-wide text-slate-300"><span class="{heartbeat_dot_class}"></span>Last checked (email) · Melbourne time</p><p class="mt-1 text-lg font-semibold">{html.escape(checked_text)}</p><p class="mt-1 text-sm font-semibold">{html.escape(heartbeat_label)}</p><p class="mt-2 text-sm text-slate-300">{html.escape(detail)}</p></section><section class="{search_panel_class}"><p class="text-xs font-semibold uppercase tracking-wide text-slate-300"><span class="{search_dot_class}"></span>Last searched (active search) · Melbourne time</p><p class="mt-1 text-lg font-semibold">{html.escape(search_checked_text)}</p><p class="mt-1 text-sm font-semibold">{html.escape(search_label)}</p><p class="mt-2 text-sm text-slate-300">{html.escape(search_detail)}</p></section><section class="mb-8 grid grid-cols-2 sm:grid-cols-5 gap-3">{stats_html}</section><div class="mb-4 flex gap-2 border-b border-slate-200"><button id="tab-btn-today" onclick="showTab('today')" class="px-4 py-2 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-700">Today</button><button id="tab-btn-all" onclick="showTab('all')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">All</button><button id="tab-btn-toprocess" onclick="showTab('toprocess')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">To process</button><button id="tab-btn-favorites" onclick="showTab('favorites')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">♥ Favourites</button><button id="tab-btn-applied" onclick="showTab('applied')" class="px-4 py-2 text-sm font-semibold border-b-2 border-transparent text-slate-500">✓ Applied</button></div><section id="tab-panel-today" class="mb-8"><div class="space-y-4">{_cards_today("No jobs added today yet.")}</div></section><section id="tab-panel-all" hidden class="mb-8"><div class="space-y-4">{_cards_all("Nothing active right now.")}</div></section><section id="tab-panel-toprocess" hidden class="mb-8"><div class="space-y-4">{_cards("TO_PROCESS", "Nothing kept yet — click Keep on a job in Today or All to shortlist it here.")}</div></section><section id="tab-panel-favorites" hidden class="mb-8"><div class="space-y-4">{_cards_favorites("No favourites yet — click the heart on a job to save it here.")}</div></section><section id="tab-panel-applied" hidden class="mb-8"><div class="mb-3 flex gap-2"><input type="search" id="applied-search" oninput="filterApplied()" placeholder="Quick find — company or title (e.g. for a call)…" class="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"><a href="/export/applied.csv" class="rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium whitespace-nowrap">Export CSV</a></div><div id="applied-cards" class="space-y-4">{_cards("APPLIED", "Nothing applied to yet.")}</div></section></main><script>
{CARD_ACTIONS_JS}
async function resync() {{
  const r = await fetch('/resync', {{method:'POST'}});
  if (!r.ok) return alert('Could not start resync');
  location.reload();
}}
let showUnscoredOnly = false;
let sourceFilter = 'all';
function applyCardFilters() {{
  document.querySelectorAll('article[data-scored]').forEach(function(el) {{
    let hide = false;
    if (showUnscoredOnly && el.dataset.scored === '1') hide = true;
    if (sourceFilter !== 'all' && el.dataset.source !== sourceFilter) hide = true;
    el.hidden = hide;
  }});
}}
function toggleUnscored() {{
  showUnscoredOnly = !showUnscoredOnly;
  document.getElementById('unscored-toggle').textContent = showUnscoredOnly ? 'Show all' : 'Show unscored only';
  applyCardFilters();
}}
function cycleSourceFilter() {{
  sourceFilter = sourceFilter === 'all' ? 'search' : sourceFilter === 'search' ? 'email' : 'all';
  const labels = {{all: 'All sources', search: 'Search only', email: 'Email only'}};
  document.getElementById('source-filter-toggle').textContent = labels[sourceFilter];
  applyCardFilters();
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
  ['today', 'all', 'toprocess', 'favorites', 'applied'].forEach(function(t) {{
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


@app.post("/click/{job_id}")
def click(job_id: int) -> dict:
    """Fired by sendBeacon when 'View' is clicked — signal for the daily
    search-term review, not a workflow action, so it never raises even for
    an unknown id (a beacon send has no way to show the user an error
    anyway)."""
    record_click(job_id)
    return {"ok": True}


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


# In-memory only (not persisted, not shared across profiles' separate
# processes) — a job_id's title + current step label while a manual
# rescore is in flight, polled both by that card's button and by the
# page-wide queue badge (so an in-flight rescore stays visible even from
# a different tab or after a reload). Cleared once the request finishes
# (success or failure), whichever comes first.
_reprocess_status: dict[int, dict[str, str]] = {}


@app.get("/reprocess-status/{job_id}")
def reprocess_status(job_id: int) -> dict:
    entry = _reprocess_status.get(job_id)
    return {"status": entry["status"] if entry else None}


@app.get("/reprocess-queue")
def reprocess_queue() -> dict:
    return {
        "jobs": [
            {"job_id": job_id, "title": entry["title"], "status": entry["status"]}
            for job_id, entry in _reprocess_status.items()
        ]
    }


@app.post("/reprocess/{job_id}")
def reprocess(job_id: int) -> dict:
    """Manually re-run scoring on a job from its already-stored description
    — same production scoring path as a normal ingest, just triggered by
    hand. Resets scorer's module-level caches first so it re-reads the CV,
    preferences and prompt config from disk rather than using whatever this
    long-running process happened to load at startup — code changes still
    need a process restart, but data/prompt edits are picked up immediately."""
    with connection() as conn:
        row = conn.execute("SELECT title, company, raw_description FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if not row["raw_description"]:
        raise HTTPException(
            status_code=400,
            detail="No stored description to rescore from — use the rescore bookmarklet for LinkedIn jobs without one",
        )
    scorer._cv_text = None
    scorer._prefs = None
    scorer._prompt_config = None

    def set_status(label: str) -> None:
        _reprocess_status[job_id] = {"title": row["title"], "status": label}

    set_status("Starting…")
    try:
        evaluation = score_job(
            row["title"], row["company"], row["raw_description"],
            on_step=set_status,
        )
        set_status("Saving…")
        with connection() as conn:
            conn.execute(
                "UPDATE jobs SET ai_evaluation = ?, overall_score = ? WHERE id = ?",
                (json.dumps(evaluation), evaluation["overall_score"], job_id),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rescore failed: {exc}") from exc
    finally:
        _reprocess_status.pop(job_id, None)
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


@app.get("/term-suggestions", response_class=HTMLResponse)
def term_suggestions_page() -> str:
    """Detail view for the amber banner on the dashboard — full reasoning
    from the daily term_review.py run, with each suggested change editable
    (or excludable) before applying. Read-only until /term-suggestions/apply
    is actually submitted."""
    suggestions = _term_suggestions()
    if not suggestions or not suggestions.get("has_suggestions"):
        return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>Job Search Assistant — search preferences review</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8">
<a href="/" class="text-sm text-slate-500 hover:underline">&larr; Back to dashboard</a>
<p class="mt-4 rounded-xl bg-white p-5 text-slate-500 shadow">No suggestions right now.</p>
</main></body></html>'''

    generated_at = suggestions.get("generated_at", "")
    add_rows = "".join(
        f'''<div class="flex items-start gap-3 rounded-lg border border-slate-200 p-3">
<input type="checkbox" checked data-role="add" class="mt-1" value="{html.escape(t['term'])}">
<div class="flex-1"><input type="text" value="{html.escape(t['term'])}" data-role="add-text" class="w-full rounded border border-slate-300 px-2 py-1 text-sm font-medium">
<p class="mt-1 text-xs text-slate-500">{html.escape(t.get('reason', ''))}</p></div>
</div>'''
        for t in (suggestions.get("add_terms") or [])
    ) or '<p class="text-sm text-slate-500">No additions suggested.</p>'
    remove_rows = "".join(
        f'''<div class="flex items-start gap-3 rounded-lg border border-slate-200 p-3">
<input type="checkbox" checked data-role="remove" class="mt-1" value="{html.escape(t['term'])}">
<div class="flex-1"><p class="text-sm font-medium">{html.escape(t['term'])}</p>
<p class="mt-1 text-xs text-slate-500">{html.escape(t.get('reason', ''))}</p></div>
</div>'''
        for t in (suggestions.get("remove_terms") or [])
    ) or '<p class="text-sm text-slate-500">No removals suggested.</p>'

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>Job Search Assistant — search preferences review</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8">
<a href="/" class="text-sm text-slate-500 hover:underline">&larr; Back to dashboard</a>
<h1 class="mt-2 text-2xl font-bold tracking-tight">Search preferences review</h1>
<p class="mt-1 text-xs text-slate-500">Generated {html.escape(generated_at)}</p>
<p class="mt-3 rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-900">{html.escape(suggestions.get("summary", ""))}</p>

<h2 class="mt-6 text-lg font-semibold text-emerald-700">Suggested additions</h2>
<div class="mt-2 space-y-2">{add_rows}</div>

<h2 class="mt-6 text-lg font-semibold text-red-700">Suggested removals</h2>
<div class="mt-2 space-y-2">{remove_rows}</div>

<div class="mt-6 flex gap-2">
<button onclick="applyTermChanges()" class="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white">Apply selected changes</button>
<button onclick="dismissTermChanges()" class="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium">Dismiss</button>
</div>
<p id="apply-status" class="mt-3 text-sm"></p>
</main>
<script>
async function applyTermChanges() {{
  const addTerms = Array.from(document.querySelectorAll('[data-role="add"]:checked')).map(cb => {{
    const row = cb.closest('div.flex');
    return row.querySelector('[data-role="add-text"]').value.trim();
  }}).filter(Boolean);
  const removeTerms = Array.from(document.querySelectorAll('[data-role="remove"]:checked')).map(cb => cb.value);
  const status = document.getElementById('apply-status');
  status.textContent = 'Applying…';
  const r = await fetch('/term-suggestions/apply', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{add_terms: addTerms, remove_terms: removeTerms}})}});
  if (!r.ok) {{ status.textContent = 'Failed to apply changes.'; return; }}
  status.textContent = 'Applied — redirecting…';
  location.href = '/search-terms';
}}
async function dismissTermChanges() {{
  await fetch('/term-suggestions/apply', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{add_terms: [], remove_terms: []}})}});
  location.href = '/';
}}
</script>
</body></html>'''


@app.post("/term-suggestions/apply")
def term_suggestions_apply(payload: TermSuggestionApplyPayload) -> dict:
    """Applies the (possibly edited) selected changes to every connector's
    search_terms symmetrically, then clears the suggestion file either way
    — dismiss is just an apply with empty lists."""
    if payload.add_terms or payload.remove_terms:
        prefs = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        connectors = prefs.get("connectors") or {}
        for platform in ("linkedin", "indeed", "glassdoor"):
            config = connectors.get(platform)
            if not config or "search_terms" not in config:
                continue
            terms = [t for t in config["search_terms"] if t not in payload.remove_terms]
            for term in payload.add_terms:
                if term not in terms:
                    terms.append(term)
            config["search_terms"] = terms
        PREFS_PATH.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")
        scorer._prefs = None  # force a fresh read next time anything loads prefs

    if TERM_SUGGESTIONS_PATH.exists():
        TERM_SUGGESTIONS_PATH.unlink()
    return {"ok": True}


@app.get("/search-terms", response_class=HTMLResponse)
def search_terms_report() -> str:
    """How each configured active-search term is performing — found vs.
    kept vs. discarded, so a noisy term (high discard rate) is easy to
    spot and prune from preferences.json. Read-only: this never edits
    connectors itself, it just surfaces the evidence for a human call."""
    stats = search_term_stats()
    if not stats:
        rows_html = '<p class="rounded-xl bg-white p-5 text-slate-500 shadow">No search-term data yet — needs at least 3 kept/discarded decisions on a term\'s jobs before it shows up here.</p>'
    else:
        row_parts = []
        for s in stats:
            rate_pct = round(s["discard_rate"] * 100)
            rate_cls = (
                "bg-red-100 text-red-700" if rate_pct >= 80
                else "bg-amber-100 text-amber-700" if rate_pct >= 50
                else "bg-emerald-100 text-emerald-700"
            )
            row_parts.append(
                f'<tr class="border-b border-slate-100">'
                f'<td class="py-2 pr-4 font-medium">{html.escape(s["search_term"])}</td>'
                f'<td class="py-2 pr-4 text-slate-500">{html.escape(s["platform"])}</td>'
                f'<td class="py-2 pr-4 text-right">{s["found"]}</td>'
                f'<td class="py-2 pr-4 text-right">{s["decided"]}</td>'
                f'<td class="py-2 pr-4 text-right">{s["kept"]}</td>'
                f'<td class="py-2 text-right"><span class="rounded-full {rate_cls} px-2 py-0.5 text-xs font-semibold">{rate_pct}% discarded</span></td>'
                f'</tr>'
            )
        rows_html = (
            '<div class="overflow-x-auto rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">'
            '<table class="w-full text-sm"><thead><tr class="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">'
            '<th class="py-2 pr-4">Search term</th><th class="py-2 pr-4">Platform</th>'
            '<th class="py-2 pr-4 text-right">Found</th><th class="py-2 pr-4 text-right">Decided</th>'
            '<th class="py-2 pr-4 text-right">Kept</th><th class="py-2 text-right">Discard rate</th>'
            '</tr></thead><tbody>' + "".join(row_parts) + '</tbody></table></div>'
        )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>Job Search Assistant — search term performance</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-3xl p-4 sm:p-8">
<a href="/" class="text-sm text-slate-500 hover:underline">&larr; Back to dashboard</a>
<h1 class="mt-2 text-2xl font-bold tracking-tight">Search term performance</h1>
<p class="mt-2 text-slate-600">How each active-search term ({html.escape(PROFILE.capitalize())}'s LinkedIn/Indeed/Glassdoor searches) is doing, based on your own Keep/Discard/Applied/Declined decisions on the jobs it found. A high discard rate means that term is mostly surfacing noise — worth narrowing or removing from <code>preferences.json</code>. Only terms with at least 3 decided jobs are shown.</p>
<div class="mt-6">{rows_html}</div>
</main></body></html>'''


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_view(job_id: int) -> str:
    """Single-job page — same card rendering as the main dashboard, just
    one row. Used as the landing page after an add/rescore bookmarklet
    action on exactly one job, so the result is immediately visible
    instead of a plain text status message."""
    with connection() as conn:
        row = conn.execute(f"SELECT {_CARD_FIELDS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    body = _render_jobs([row], "")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>Job Search Assistant — {html.escape(row['title'] or 'job')}</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8">
<a href="/" class="text-sm text-slate-500 hover:underline">&larr; Back to dashboard</a>
<div class="mt-4">{body}</div>
</main><script>{CARD_ACTIONS_JS}</script></body></html>'''


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
    last_job_id = None
    for entry in payload.jobs:
        canonical_id = f"seek_{entry.job_id}"
        url = f"https://au.seek.com/job/{entry.job_id}"
        job_fingerprint = fingerprint(entry.company, entry.title, entry.location)
        existing_id = get_job_id(canonical_id, job_fingerprint)
        if existing_id is not None:
            duplicates += 1
            last_job_id = existing_id
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
            last_job_id = new_job_id
        else:
            duplicates += 1
            last_job_id = get_job_id(canonical_id, job_fingerprint)

    # Only meaningful for a single-job bookmarklet click — a search-results
    # page can add many at once, and there's no single job to land on.
    single_job_id = last_job_id if len(payload.jobs) == 1 else None
    return {"ok": True, "added": len(added), "duplicates": duplicates, "jobs": added, "job_id": single_job_id}


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
    existing_id = get_job_id(canonical_id, job_fingerprint)
    if existing_id is not None:
        return {"ok": True, "duplicate": True, "job_id": existing_id}

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
        return {"ok": True, "duplicate": True, "job_id": get_job_id(canonical_id, job_fingerprint)}
    return {
        "ok": True, "duplicate": False, "job_id": new_job_id,
        "overall_score": evaluation["overall_score"], "recommendation": evaluation["recommendation"],
    }


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
