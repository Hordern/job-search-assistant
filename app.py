"""Mobile-friendly FastAPI dashboard for JobHunter."""

import html
import json
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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from db import connection, init_db, latest_ingestion_run, run_events
from scorer import score_job
from version import build_version

VERSION = build_version("1.2.0")
MELBOURNE = ZoneInfo("Australia/Melbourne")
INGEST_SCRIPT = Path(__file__).with_name("ingest.py")

app = FastAPI(title=f"JobHunter v{VERSION}")
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
    action: Literal["DISCARDED", "TO_PROCESS", "APPLIED"]


class RescorePayload(BaseModel):
    canonical_id: str
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
}


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
    return f'''<article data-scored="1" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><span class="rounded-full bg-indigo-100 px-3 py-1 text-sm font-bold text-indigo-700">Score {row['overall_score'] or 0}</span></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-indigo-700">{html.escape(str(evaluation.get('recommendation') or 'Unrated'))} · {html.escape(str(row['platform'] or ''))}</p>
  <p class="mt-4 text-sm"><span class="font-semibold">Pitch angle:</span> {html.escape(str(evaluation.get('pitch_angle') or 'Not available'))}</p>
  <div class="mt-3 flex flex-wrap gap-2 text-xs">
    <span class="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">CV fit: {evaluation.get('cv_fit_score', '—')}</span>
    {fit_tag('Salary', salary_fit, fit_cls(salary_fit))}
    {fit_tag('Location', location_fit, fit_cls(location_fit))}
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
    return f'''<article data-scored="0" class="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200 border border-dashed border-slate-300">
  <div class="flex items-start justify-between gap-4"><div><h2 class="text-lg font-bold">{html.escape(row['title'] or '')}</h2><p class="text-slate-600">{html.escape(row['company'] or '')}</p></div><span class="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-500">Not scored</span></div>
  <p class="mt-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{html.escape(str(row['platform'] or ''))}</p>
  <p class="mt-3 text-sm text-slate-500">{html.escape(hint)}</p>
  <div class="mt-5 flex flex-wrap gap-2"><a class="rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white" href="{html.escape(row['url'] or '', quote=True)}" target="_blank" rel="noopener noreferrer">View</a>{buttons_html}</div>
</article>'''


def _cards(status: str, empty_message: str) -> str:
    actions = SECTION_ACTIONS[status]
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, title, company, platform, url, ai_evaluation, overall_score, created_at "
            "FROM jobs WHERE status = ?",
            (status,),
        ).fetchall()
    if not rows:
        return f'<p class="rounded-xl bg-white p-5 text-slate-500 shadow">{html.escape(empty_message)}</p>'

    # Scored jobs (a real listing was actually read) always rank above unscored
    # ones (title/company guesswork we deliberately didn't pay an LLM call for)
    # — never interleaved by a blended score.
    scored = sorted(
        (r for r in rows if r["ai_evaluation"] is not None),
        key=lambda r: (r["overall_score"] or 0, r["created_at"]), reverse=True,
    )
    unscored = sorted(
        (r for r in rows if r["ai_evaluation"] is None),
        key=lambda r: r["created_at"], reverse=True,
    )

    cards = []
    for row in scored + unscored:
        row_id = row["id"]
        buttons_html = "".join(
            f'''<button onclick="act({row_id}, '{action}')" class="{classes}">{label}</button>'''
            for action, label, classes in actions
        )
        if row["ai_evaluation"] is not None:
            cards.append(_scored_card(row, buttons_html))
        else:
            cards.append(_unscored_card(row, buttons_html))
    return "\n".join(cards)


def _dashboard_data() -> tuple[dict[str, int], str, str]:
    with connection() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    run = latest_ingestion_run()
    if not run:
        return counts, "Not checked yet", "Run ingest.py to check the mailbox."
    checked_at = datetime.fromisoformat(run["checked_at"]).astimezone(MELBOURNE)
    checked_text = checked_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    detail = (
        f"{run['folder']} · {run['emails_seen']} mailbox messages · "
        f"{run['emails_processed']} newly processed · {run['job_links_found']} job links found · "
        f"{run['jobs_saved']} saved · {run['duplicates']} duplicates · {run['errors']} errors · "
        f"worker {run['worker_version'] or 'unknown'}"
    )
    return counts, checked_text, detail


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
var receiver = 'http://127.0.0.1:8000/rescore-bookmarklet#' + encodeURIComponent(JSON.stringify({canonical_id:id, raw_description:text}));
var popup = window.open(receiver, 'jobhunter-rescore', 'popup,width=480,height=360');
if(!popup) location.href = receiver;
})();"""
RESCORE_BOOKMARKLET_HREF = "javascript:" + quote(RESCORE_BOOKMARKLET_JS)


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


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    counts, checked_text, detail = _dashboard_data()
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
    ])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><script src="https://cdn.tailwindcss.com"></script><title>JobHunter</title></head><body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-2xl p-4 sm:p-8"><header class="mb-6"><p class="text-sm font-semibold text-indigo-700">JobHunter v{VERSION}</p><h1 class="text-3xl font-bold">Daily job digest</h1><p class="mt-1 text-slate-600">Ranked opportunities from your email alerts.</p><p class="mt-2 text-xs text-slate-500">Drag to your bookmarks bar: <a href="{RESCORE_BOOKMARKLET_HREF}" class="rounded bg-slate-200 px-2 py-1 font-semibold text-slate-700">Rescore LinkedIn job</a> — open a LinkedIn job page while logged in, then click it to score with the full listing.</p><button onclick="resync()" class="mt-3 rounded-lg bg-slate-800 px-3 py-2 text-sm font-medium text-white">Resync now</button><button id="unscored-toggle" onclick="toggleUnscored()" class="mt-3 ml-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium">Show unscored only</button></header><section id="progress-panel" {"" if is_running else "hidden"} class="mb-6 rounded-2xl bg-slate-900 p-5 text-white"><p class="text-xs font-semibold uppercase tracking-wide text-amber-400">Worker running…</p><div id="progress-events" class="mt-2 max-h-48 overflow-y-auto space-y-0.5 font-mono"></div></section><section class="mb-6 rounded-2xl bg-slate-900 p-5 text-white"><p class="text-xs font-semibold uppercase tracking-wide text-slate-300">Last checked · Melbourne time</p><p class="mt-1 text-lg font-semibold">{html.escape(checked_text)}</p><p class="mt-2 text-sm text-slate-300">{html.escape(detail)}</p></section><section class="mb-8 grid grid-cols-2 sm:grid-cols-4 gap-3">{stats_html}</section><section class="mb-8"><h2 class="mb-3 text-xl font-bold">New opportunities</h2><div class="space-y-4">{_cards("NEW", "No new jobs yet.")}</div></section><section><h2 class="mb-3 text-xl font-bold">To process</h2><div class="space-y-4">{_cards("TO_PROCESS", "Nothing kept for review yet.")}</div></section></main><script>
async function act(job_id, action) {{
  const r = await fetch('/action', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{job_id, action}})}});
  if (!r.ok) return alert('Could not update job');
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


@app.post("/resync")
def resync() -> dict[str, bool]:
    """Force-stop any in-progress worker run and launch a fresh one. Useful
    when a run is stuck/hung on a bad connection rather than waiting out the
    next scheduled launchd trigger."""
    subprocess.run(["pkill", "-f", str(INGEST_SCRIPT)])
    # SIGTERM isn't instant — wait for the old process to actually release its
    # flock before launching a new one, or it just sees the lock held and skips.
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", str(INGEST_SCRIPT)], capture_output=True).returncode != 0:
            break
        time.sleep(0.1)
    (INGEST_SCRIPT.parent / ".ingest.lock").unlink(missing_ok=True)
    subprocess.Popen(
        [sys.executable, str(INGEST_SCRIPT)],
        cwd=INGEST_SCRIPT.parent,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"ok": True}
