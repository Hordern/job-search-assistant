"""Resolves the per-user profile directory (users/<profile>/) shared by every
Job Search Assistant module (db.py, scorer.py, ingest.py, app.py). Each profile has its
own .env, cv_scoring_full.txt, preferences.json and jobs.db, so one codebase
can run multiple independent people's job searches — set JOBSEARCH_PROFILE to
pick which one a given process uses.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

PROFILE = os.environ.get("JOBSEARCH_PROFILE")
if not PROFILE:
    sys.exit(
        "JOBSEARCH_PROFILE must be set (e.g. JOBSEARCH_PROFILE=alice) — "
        "see users/alice/ for what a profile directory looks like."
    )

PROFILE_DIR = BASE_DIR / "users" / PROFILE
if not PROFILE_DIR.is_dir():
    sys.exit(
        f"No profile directory at {PROFILE_DIR} — create users/{PROFILE}/ "
        "(.env, cv_scoring_full.txt, preferences.json) before running with "
        f"JOBSEARCH_PROFILE={PROFILE}."
    )

ENV_PATH = PROFILE_DIR / ".env"
