"""Nominatim wrapper for resolving marine place names not in the gazetteer.

Public Nominatim instance, free, rate-limited to 1 req/sec per their ToS,
cached on disk to respect that limit AND survive boat-laptop offline
periods.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "marine-router (boat-voice; https://github.com/cbc76am-hue/marine-router)"
HTTP_TIMEOUT_S = 10.0
MIN_REQUEST_INTERVAL_S = 1.1
CACHE_DIR = Path.home() / ".cache" / "marine-router" / "nominatim"
CACHE_STALE_AFTER_S = 24 * 3600

SALISH_LAT_MIN = 47.0
SALISH_LAT_MAX = 49.0
SALISH_LON_MIN = -124.5
SALISH_LON_MAX = -122.0

_last_request_at: float = 0.0


@dataclass(frozen=True)
class ExternalCandidate:
    name: str
    short_name: str
    lat: float
    lon: float
    source: str = "osm"
    importance: float = 0.0
    osm_type: str = ""
    osm_class: str = ""


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_path(normalized: str) -> Path:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _cache_load(normalized: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_cache_path(normalized)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_save(normalized: str, blob: Dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp = _cache_path(normalized)
        tmp = fp.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(blob, f)
        tmp.replace(fp)
    except OSError as e:
        log.warning("external_geocode: cache write failed for %r (%s)", normalized, e)


def _cache_age_s(blob: Optional[Dict[str, Any]]) -> Optional[float]:
    if not blob:
        return None
    ts = blob.get("fetched_at")
    if not ts:
        return None
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - when).total_seconds()


def _rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
    _last_request_at = time.monotonic()


def _http_get(url: str) -> List[Dict[str, Any]]:
    _rate_limit()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError(f"expected list, got {type(data).__name__}")
    return data


def _to_candidate(item: Dict[str, Any]) -> ExternalCandidate:
    display = str(item["display_name"])
    return ExternalCandidate(
        name=display,
        short_name=display.split(",", 1)[0].strip(),
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        importance=float(item.get("importance") or 0.0),
        osm_type=str(item.get("osm_type") or ""),
        osm_class=str(item.get("class") or ""),
    )


def lookup(query: str, max_candidates: int = 5) -> List[ExternalCandidate]:
    """Look up a place name in Nominatim, constrained to the Salish Sea bbox.

    Returns candidates sorted by importance (highest first).  Cached on disk
    for 24 h; failures (network, malformed, rate-limit) return [].
    """
    log.info("external_geocode: lookup(%r)", query)
    normalized = _normalize(query)
    if not normalized:
        return []

    cached = _cache_load(normalized)
    age = _cache_age_s(cached)
    if cached and age is not None and age <= CACHE_STALE_AFTER_S:
        raw = cached.get("results") or []
        log.debug("external_geocode: cache hit for %r (%d candidates)", query, len(raw))
        try:
            return [_to_candidate(item) for item in raw]
        except (KeyError, TypeError, ValueError) as e:
            log.warning("external_geocode: cached payload for %r unusable (%s)", query, e)

    params = {
        "q": query,
        "format": "json",
        "addressdetails": "1",
        "viewbox": f"{SALISH_LON_MIN},{SALISH_LAT_MIN},{SALISH_LON_MAX},{SALISH_LAT_MAX}",
        "bounded": "1",
        "limit": str(max_candidates),
    }
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"

    try:
        results = _http_get(url)
        candidates = [_to_candidate(item) for item in results]
    except (urllib.error.URLError, OSError, json.JSONDecodeError,
            KeyError, TypeError, ValueError) as e:
        log.warning("external_geocode: lookup failed for %r (%s)", query, e)
        return []

    candidates.sort(key=lambda c: c.importance, reverse=True)
    log.info("external_geocode: nominatim returned %d candidates for %r",
             len(candidates), query)
    _cache_save(normalized, {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query,
        "normalized": normalized,
        "results": results,
    })
    return candidates
