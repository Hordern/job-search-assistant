"""Straight-line ("as the crow flies") distance from home to a job's
location — free, no API key or account required. Uses OpenStreetMap's
Nominatim geocoding service to resolve a place name to coordinates, cached
to disk (a suburb's coordinates never change, so re-geocoding on every
dashboard load would be pointless network traffic and would violate
Nominatim's usage policy, which requires results to be cached and requests
kept to roughly 1/second).

This is explicitly NOT a real commute time — no roads, no traffic, no public
transport. It's a rough distance badge only. Fails soft: every public
function returns None (never raises) when a place can't be resolved or the
network call fails, so a badge just doesn't render.
"""

import json
import math
import time
from typing import Any

import requests

from profile import PROFILE_DIR

CACHE_PATH = PROFILE_DIR / "commute_cache.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and roughly
# 1 request/second max — cached lookups never hit this, only new places do.
USER_AGENT = "JobSearchAssistantPersonalTool/1.0 (personal use, not for redistribution)"
_MIN_REQUEST_INTERVAL_S = 1.1

_cache: dict[str, Any] | None = None
_last_request_time = 0.0
# A failed lookup (network error, or the geocoder rate-limiting us) is cached
# for this long before we retry it — without this, every uncached location on
# every page load re-attempts and re-fails the same request, which is both
# slow (each failure still costs a real network round-trip) and makes the
# rate-limiting worse, not better.
_FAILURE_RETRY_AFTER_S = 3600


def _load_cache() -> dict[str, Any]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            try:
                _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        CACHE_PATH.write_text(json.dumps(_cache, indent=2), encoding="utf-8")


def _geocode(place: str) -> tuple[float, float] | None:
    global _last_request_time
    cache = _load_cache()
    key = f"geocode:{place.strip().lower()}"
    if key in cache:
        cached = cache[key]
        if isinstance(cached, dict) and cached.get("_failed_at") is not None:
            if time.time() - cached["_failed_at"] < _FAILURE_RETRY_AFTER_S:
                return None
            # retry-after window elapsed — fall through and re-attempt
        else:
            return tuple(cached) if cached else None

    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL_S:
        time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": place, "format": "json", "limit": 1, "countrycodes": "au"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        _last_request_time = time.monotonic()
        response.raise_for_status()
        results = response.json()
        if not results:
            cache[key] = None
            _save_cache()
            return None
        coords = (float(results[0]["lat"]), float(results[0]["lon"]))
    except (requests.RequestException, KeyError, IndexError, ValueError):
        _last_request_time = time.monotonic()
        cache[key] = {"_failed_at": time.time()}
        _save_cache()
        return None

    cache[key] = list(coords)
    _save_cache()
    return coords


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def get_commute(origin: str, destination: str) -> dict[str, Any] | None:
    """Returns {"distance_km": 14.2} (straight-line, not a real route) or
    None if either place can't be resolved or a lookup fails."""
    if not origin or not destination:
        return None
    origin_coords = _geocode(f"{origin}, Australia")
    dest_coords = _geocode(f"{destination}, Australia")
    if not origin_coords or not dest_coords:
        return None
    return {"distance_km": round(_haversine_km(origin_coords, dest_coords), 1)}
