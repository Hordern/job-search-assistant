"""Estimated Anthropic API spend for this app, from the per-call usage log
(users/<profile>/api_usage.jsonl). Anthropic's own billing ledger needs an
Admin key we don't have, so check this against the Console's Cost page.

Prices (claude-sonnet-5, $/MTok): input 2.00, cache write 2.50, cache read
0.20, output 10.00. Calls made through the Batch API are billed at 50%.

Usage:
    python cost_report.py                      # default tracking window
    python cost_report.py 2026-10-08T11:10     # from a Melbourne-local time
    python cost_report.py 2026-10-08T11:10 2026-10-10T11:10
"""

import collections
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MEL = ZoneInfo("Australia/Melbourne")
USERS = Path(__file__).parent / "users"
PROFILES = ("alice", "bob")
DEFAULT_START = "2026-10-08T11:10"
BASELINE_PER_DAY = 5.17  # Oct 6-7 average: synchronous scoring, thinking off + caching, before compact JSON / batch / 3-day term review


def _cost(r: dict) -> float:
    c = (r["input"] * 2.0 + r["cache_write"] * 2.5 + r["cache_read"] * 0.2 + r["output"] * 10.0) / 1e6
    return c * 0.5 if r.get("batch") else c


def _kind(r: dict) -> str:
    cached = r["cache_write"] + r["cache_read"]
    if cached == 0 and r["input"] > 20000:
        return "term review"
    if cached == 0:
        return "audit (uncached)"
    return "extraction" if cached < 2500 else "cv matching"


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=MEL)


def _jobs_scored(profile: str, start: datetime, end: datetime) -> int:
    """Jobs that gained a score inside the window. Batch-scored jobs are
    created earlier than they are scored, so this counts jobs created in the
    window that now have an evaluation (close enough over a multi-day window)."""
    db = USERS / profile / "jobs.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT created_at FROM jobs WHERE ai_evaluation IS NOT NULL").fetchall()
    finally:
        conn.close()
    return sum(1 for (c,) in rows if start <= datetime.fromisoformat(c).astimezone(MEL) <= end)


def main() -> None:
    start = _parse(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START)
    end = _parse(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(MEL)
    hours = max((end - start).total_seconds() / 3600, 0.01)
    print(f"API spend {start:%a %d %b %H:%M} -> {end:%a %d %b %H:%M} Melbourne ({hours:.1f}h), estimated\n")

    grand = 0.0
    total_jobs = 0
    by_kind: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0, 0])
    for profile in PROFILES:
        path = USERS / profile / "api_usage.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if start <= datetime.fromisoformat(r["ts"]).astimezone(MEL) <= end:
                    rows.append(r)
        total_jobs += _jobs_scored(profile, start, end)
        cost = sum(_cost(r) for r in rows)
        grand += cost
        out_tok = sum(r["output"] for r in rows)
        total_in = sum(r["input"] + r["cache_write"] + r["cache_read"] for r in rows)
        hit = 100 * sum(r["cache_read"] for r in rows) / total_in if total_in else 0
        batch_share = 100 * sum(1 for r in rows if r.get("batch")) / len(rows) if rows else 0
        print(f"{profile:9s} {len(rows):4d} calls  ${cost:6.2f}  output {out_tok:>8,} tok  "
              f"cache hit {hit:3.0f}%  batch {batch_share:3.0f}% of calls")
        for r in rows:
            k = by_kind[_kind(r)]
            k[0] += 1
            k[1] += _cost(r)
            k[2] += r["output"]

    print(f"{'TOTAL':9s} {'':4s}        ${grand:6.2f}")
    print("\nBy call type:")
    for kind, (n, c, out) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"  {kind:17s} {n:4d} calls  avg output {out / n:6,.0f} tok  ${c:6.2f}  ({100 * c / grand:.0f}%)")

    per_day = grand / hours * 24
    print(f"\nRun rate: ${per_day:.2f}/day  (~${per_day * 30:.0f}/month)  vs ${BASELINE_PER_DAY:.2f}/day on Oct 6-7 (sync scoring) "
          f"-> {100 * (1 - per_day / BASELINE_PER_DAY):.0f}% lower")

    print(f"Jobs scored in window: {total_jobs}  ->  ${grand / total_jobs:.3f} per job" if total_jobs else "No jobs scored in window.")


if __name__ == "__main__":
    main()
