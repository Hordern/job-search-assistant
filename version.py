"""Shared build-identity helper so the worker and dashboard can report exactly
what code they're running — the git hash catches "did the fix actually deploy"
without needing a manual version bump on every change."""

import subprocess
from pathlib import Path


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return "unknown"
        hash_ = result.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            hash_ += "-dirty"
        return hash_
    except Exception:
        return "unknown"


def build_version(version: str) -> str:
    return f"{version}+{git_hash()}"
