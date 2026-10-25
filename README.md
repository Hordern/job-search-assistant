# Job Search Assistant

A personal tool that reads job-alert emails (SEEK, LinkedIn), optionally searches job boards, removes duplicates, and scores each job against your CV with Claude. A small web dashboard lets you review, keep, discard and track applications.

It was built for one household's own job search and runs on a Mac. This repository is a mirror: development happens in a private working repo and this copy is regenerated from it with personal details removed, so the history here is a cleaned replay of the original. It is shared as a worked example rather than as a product; expect rough edges (see [Limitations](#limitations)). Developed with AI coding tools.

## How scoring works

Each job goes through up to three Claude passes. Everything that produces a number or a decision is plain Python, so the model never gets to state a score or judge eligibility.

1. **Extraction** - reads only the job ad and lists atomic requirements, each with a verbatim source quote, plus licences/checks, salary and location facts. No CV is involved, so the same extraction works for any candidate.
2. **Audit** (only when a heuristic suggests the first pass missed something) - re-reads the ad for missed requirements.
3. **CV matching** - for each capability requirement, classifies the match (direct / adjacent / learning / not evidenced / contradicted) and must quote the CV as evidence.

Every quote the model returns is checked against the real text (exact match, then a strict fuzzy match). A quote that is not actually in the ad or CV rejects the response and retries. This is what stops weaker models from inventing plausible-sounding evidence, and why cheaper models were tried and rejected.

Python then computes the capability score from requirement importance and match depth, and sets the recommendation tier (Strong Apply / Apply / Stretch / Pass). Salary, location, scope, down-ranked industries and unmet eligibility **demote the tier** rather than altering the capability score, and excluded industries or unmet mandatory eligibility short-circuit the pipeline after the first call.

## What's in the box

| | |
|---|---|
| `scorer.py` | The three-pass pipeline, validation and deterministic scoring |
| `ingest.py` | Reads alert emails over IMAP, fetches full ads, saves and scores jobs |
| `search_ingest.py` | Optional active search through JobSpy |
| `app.py` | FastAPI dashboard: Today / All / To process / Favourites / Applied, rescoring, "Job closed" |
| `batch_scoring.py` | Scores through the Message Batches API at half price instead of inline |
| `daily_clean.py`, `check_open_jobs.py` | Daily housekeeping: closes expired listings, drops low scores, explains unscored jobs |
| `term_review.py` | Suggests search-term changes from your Keep / Discard decisions |
| `watchdog.py`, `sidecar.py` | Health checks with macOS notifications and a minimal recovery page |
| `cost_report.py` | Estimated API spend from the per-call usage log |

Each person gets their own profile directory (`users/<name>/`) with a separate database, preferences, CV and schedule. `users/example/` shows the preferences format and `users/ci/` is the dummy profile the tests run on.

## Keeping the API bill small

`CLAUDE_API_COST_NOTES.md` records what moved the bill and by how much: turning adaptive thinking off for extraction work, prompt caching, compressing the CV once, skipping calls whose outcome is already decided, compact JSON output, and the Batch API. In this app those took daily spend from about $10-70 to about $1.5.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

mkdir -p users/me
cp users/example/preferences.json users/me/preferences.json   # edit this
cp .env.example users/me/.env                                  # IMAP + ANTHROPIC_API_KEY
# put your CV text in users/me/cv_full.txt (it is gitignored)

JOBSEARCH_PROFILE=me python ingest.py             # one ingest run
JOBSEARCH_PROFILE=me uvicorn app:app --port 8000  # the dashboard
```

On macOS the recurring jobs (ingest every 15 minutes, batch scoring every 3, daily clean at 06:50, watchdog every 15) are meant to run as launchd agents; `watchdog.py` expects labels of the form `com.example.jobsearch.<job>`.

## Tests

```bash
JOBSEARCH_PROFILE=ci CV_PATH=cv_sample.txt python test_scorer.py
```

The same command runs in GitHub Actions on every push. The tests cover the parts that have broken in practice: requirement de-duplication, eligibility rules, quote validation (including look-alike hyphens and Markdown escapes), and the scoring tiers.

## Limitations

- **The dashboard has no authentication.** It binds to all interfaces for use over a private network such as Tailscale. Do not expose it to the internet.
- Time zone (Australia/Melbourne), currency (AUD) and the SEEK / LinkedIn page formats are hard-coded. The page checks depend on those sites' markup and will break when it changes.
- Indeed and Glassdoor block automated page requests, so those jobs cannot be checked for expiry.
- It fetches public job pages. If you run it, you are responsible for complying with each site's terms.
- The watchdog and launchd setup are macOS-specific.
