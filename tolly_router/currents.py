"""Tidal-current predictions for spatio-temporal routing.

Standalone, importable data layer over NOAA CO-OPS tidal-current predictions.
The spatio-temporal A* calls ``current_field_at`` from inside its hot loop, so
this module is designed for vectorized evaluation over many query points at
one timestamp.

NOAA CO-OPS API
---------------
Endpoint::

    https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
        ?product=currents_predictions
        &interval=h
        &time_zone=gmt
        &units=english
        &format=json
        &station=<id>
        &begin_date=YYYYMMDD&end_date=YYYYMMDD
        &application=<your-tag>

The hourly response (``interval=h``) returns a list of records under
``response["current_predictions"]["cp"]``.  Each record is shaped::

    {
        "Time":          "YYYY-MM-DD HH:MM",   # GMT when time_zone=gmt
        "Velocity_Major": float,                # SIGNED knots along the major axis
        "meanFloodDir":   int,                  # degrees true, direction current
                                                #   flows TOWARD on flood
        "meanEbbDir":     int,                  # degrees true, direction current
                                                #   flows TOWARD on ebb
        "Bin":            "...",  "Depth": "..."
    }

NOAA's ``Velocity_Major`` is signed: positive = flood (flowing toward
``meanFloodDir``), negative = ebb (flowing toward ``meanEbbDir``).  This is
*not* a ``(speed, direction)`` pair as the brief described; we convert it to
``(u_east, v_north)`` at ingest.

Direction convention
--------------------
All vectors in this module are in metres/second:
  * ``u_east``  is positive *eastward*  (toward increasing longitude)
  * ``v_north`` is positive *northward* (toward increasing latitude)

NOAA's ``meanFloodDir`` / ``meanEbbDir`` are compass-true headings expressing
the direction water is *flowing TOWARDS* (not the direction it comes FROM,
which is the meteorological wind convention).  So a current with
``meanFloodDir = 90`` flowing on a flood at +2 kts is moving due east; we
convert to ``(u_east = +1.029 m/s, v_north = 0)``.

Cache
-----
  ~/.cache/marine-router/currents/<station_id>.json
      Each file holds a ~72 h window (now - 36 h .. now + 36 h) of hourly
      predictions, plus the fetched timestamp.  ``CACHE_STALE_AFTER_S = 21600``
      (6 h) — predictions are stable, NOAA is the slow part.

Public API
----------
- ``load_station_catalog()``            -> list of {id, name, lat, lon}
- ``ensure_station_cache(station_id)``  -> path to fresh on-disk cache
- ``parse_predictions(payload)``        -> structured array of
                                           (time_utc, u_east, v_north)
- ``current_at(lat, lon, t)``           -> (u_east, v_north) scalar
- ``current_field_at(query_pts, t)``    -> ndarray[N, 2] of (u_east, v_north)
                                           for many query points at one time
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .polar import KTS_TO_MS

log = logging.getLogger(__name__)

# ----- config ---------------------------------------------------------------

NOAA_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_APP = "marine-router-currents"
HTTP_TIMEOUT_S = 20.0

CACHE_DIR = Path(os.path.expanduser("~/.cache/marine-router/currents"))
CACHE_STALE_AFTER_S = 6 * 3600          # 6 h — predictions don't drift fast
CACHE_WINDOW_HALF_HOURS = 36            # cache "now ± 36 h"

# Hardcoded fallback station list used if the data file is absent.  The
# canonical list lives at ``data/current_stations.json`` (verified IDs +
# coordinates); keeping a fallback here means ``current_at`` still works in
# an unusual install where the data dir isn't shipped.  Verified against
# NOAA mdapi 2026-05-18.
_FALLBACK_STATIONS: List[Dict[str, Any]] = [
    {"id": "PUG1701", "name": "Deception Pass (Narrows)",
     "lat": 48.4062, "lon": -122.6431},
    {"id": "PUG1702", "name": "Rosario Strait",
     "lat": 48.4581, "lon": -122.7501},
    {"id": "PUG1616", "name": "Admiralty Inlet (off Bush Point)",
     "lat": 48.0335, "lon": -122.6376},
    {"id": "PUG1703", "name": "San Juan Channel, south entrance",
     "lat": 48.4610, "lon": -122.9520},
    {"id": "PUG1718", "name": "Haro Strait, 1.2 nm west of Kellett Bluff",
     "lat": 48.5887, "lon": -123.2258},
    {"id": "PUG1628", "name": "Skagit Bay channel, SW of Hope Island",
     "lat": 48.3978, "lon": -122.5796},
    {"id": "PUG1605", "name": "Possession Sound Entrance",
     "lat": 47.8993, "lon": -122.3601},
    {"id": "PUG1603", "name": "Hood Canal Bridge",
     "lat": 47.8547, "lon": -122.6294},
    {"id": "PUG1629", "name": "Yokeko Point, Deception Pass",
     "lat": 48.4127, "lon": -122.6132},
    {"id": "PUG1710", "name": "Hale Passage, east of Lummi Point",
     "lat": 48.7349, "lon": -122.6802},
]

# Path to the curated data file.  Resolved relative to the repo root so the
# module works whether installed or run from a source checkout.
_DATA_FILE_REL = Path(__file__).resolve().parent.parent / "data" / "current_stations.json"


# ----- station catalog ------------------------------------------------------

def load_station_catalog(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the list of station records.

    Reads ``data/current_stations.json`` if present, otherwise falls back to
    the hardcoded list.  Each record has ``id``, ``name``, ``lat``, ``lon``.
    """
    fp = Path(path) if path is not None else _DATA_FILE_REL
    if fp.is_file():
        try:
            with open(fp) as f:
                blob = json.load(f)
            stations = blob.get("stations") or []
            if stations:
                return list(stations)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("station catalog at %s unreadable (%s); using fallback",
                        fp, e)
    return list(_FALLBACK_STATIONS)


# ----- NOAA fetch -----------------------------------------------------------

def _http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": NOAA_APP})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"NOAA error: {data['error']}")
    return data


def fetch_current_predictions(station_id: str,
                              begin: datetime,
                              end: datetime) -> Dict[str, Any]:
    """Pull hourly tidal-current predictions for ``[begin, end]`` (UTC).

    Returns the raw NOAA JSON payload.
    """
    params = {
        "product": "currents_predictions",
        "application": NOAA_APP,
        "begin_date": begin.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station_id,
        "time_zone": "gmt",
        "units": "english",
        "interval": "h",
        "format": "json",
    }
    url = f"{NOAA_BASE}?{urllib.parse.urlencode(params)}"
    return _http_get_json(url)


# ----- shaping & conversion -------------------------------------------------

def _signed_kts_to_uv(signed_kts: float,
                      flood_dir_deg: float,
                      ebb_dir_deg: float) -> Tuple[float, float]:
    """Convert NOAA's ``Velocity_Major`` (signed knots) into ``(u_east, v_north)``.

    Positive ``signed_kts`` = current flowing toward ``flood_dir_deg`` at that
    magnitude; negative = flowing toward ``ebb_dir_deg`` at ``|signed_kts|``.
    Directions are compass-true (degrees clockwise from north).

    Returns (u_east, v_north) in metres/second.
    """
    if signed_kts >= 0.0:
        speed_kts = signed_kts
        dir_deg = flood_dir_deg
    else:
        speed_kts = -signed_kts
        dir_deg = ebb_dir_deg
    speed_ms = speed_kts * KTS_TO_MS
    # Compass true: 0 = north, 90 = east.  u_east = sin(dir), v_north = cos(dir).
    rad = math.radians(dir_deg)
    return speed_ms * math.sin(rad), speed_ms * math.cos(rad)


def parse_predictions(payload: Dict[str, Any]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a raw NOAA payload into three parallel arrays.

    Returns:
        times_utc_s: float64[N]  -- UTC epoch seconds, sorted ascending
        u_east:      float64[N]  -- m/s, eastward positive
        v_north:     float64[N]  -- m/s, northward positive

    Empty arrays returned if the payload has no records.
    """
    inner = payload.get("current_predictions") or {}
    items = inner.get("cp") or []
    if not items:
        return (np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64))

    times: List[float] = []
    us: List[float] = []
    vs: List[float] = []
    for p in items:
        try:
            t_str = p["Time"]
            # NOAA returns 'YYYY-MM-DD HH:MM' in GMT (we asked for time_zone=gmt).
            t = datetime.strptime(t_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc)
            signed = float(p.get("Velocity_Major") or 0.0)
            flood_dir = float(p.get("meanFloodDir") or 0.0)
            ebb_dir = float(p.get("meanEbbDir") or 0.0)
        except (KeyError, ValueError, TypeError):
            continue
        u, v = _signed_kts_to_uv(signed, flood_dir, ebb_dir)
        times.append(t.timestamp())
        us.append(u)
        vs.append(v)

    t_arr = np.asarray(times, dtype=np.float64)
    u_arr = np.asarray(us, dtype=np.float64)
    v_arr = np.asarray(vs, dtype=np.float64)
    order = np.argsort(t_arr)
    return t_arr[order], u_arr[order], v_arr[order]


# ----- cache layer ----------------------------------------------------------

def _cache_path(station_id: str) -> Path:
    return CACHE_DIR / f"{station_id}.json"


def _cache_load(station_id: str) -> Optional[Dict[str, Any]]:
    fp = _cache_path(station_id)
    try:
        with open(fp) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_save(station_id: str, blob: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _cache_path(station_id)
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(blob, f)
    tmp.replace(fp)


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


def ensure_station_cache(station_id: str,
                         force_refresh: bool = False,
                         now: Optional[datetime] = None,
                         ) -> Dict[str, Any]:
    """Return a fresh on-disk cache blob for ``station_id``.

    Re-fetches from NOAA if the existing cache is older than
    ``CACHE_STALE_AFTER_S`` or doesn't exist or ``force_refresh`` is set.
    The cached blob contains the raw NOAA payload plus a precomputed
    parsed-arrays view (as plain lists, for JSON serializability).
    """
    now = now or datetime.now(timezone.utc)
    cached = _cache_load(station_id)
    age = _cache_age_s(cached)
    if cached and not force_refresh and age is not None and age <= CACHE_STALE_AFTER_S:
        return cached

    begin = (now - timedelta(hours=CACHE_WINDOW_HALF_HOURS)).date()
    end = (now + timedelta(hours=CACHE_WINDOW_HALF_HOURS)).date()
    begin_dt = datetime.combine(begin, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time()).replace(tzinfo=timezone.utc)
    payload = fetch_current_predictions(station_id, begin_dt, end_dt)
    t_arr, u_arr, v_arr = parse_predictions(payload)

    blob = {
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "station_id": station_id,
        "begin_date": begin.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "raw": payload,
        "times_utc_s": t_arr.tolist(),
        "u_east_ms": u_arr.tolist(),
        "v_north_ms": v_arr.tolist(),
    }
    _cache_save(station_id, blob)
    return blob


# ----- interpolation --------------------------------------------------------

@dataclass
class _StationSeries:
    """In-memory hourly series for one station: parallel numpy arrays."""
    station_id: str
    lat: float
    lon: float
    times_utc_s: np.ndarray   # float64[N]
    u_east_ms: np.ndarray     # float64[N]
    v_north_ms: np.ndarray    # float64[N]


def _series_from_cache(blob: Dict[str, Any],
                       lat: float, lon: float) -> _StationSeries:
    return _StationSeries(
        station_id=blob.get("station_id", "?"),
        lat=lat, lon=lon,
        times_utc_s=np.asarray(blob.get("times_utc_s", []), dtype=np.float64),
        u_east_ms=np.asarray(blob.get("u_east_ms", []), dtype=np.float64),
        v_north_ms=np.asarray(blob.get("v_north_ms", []), dtype=np.float64),
    )


def _temporal_interp(series: _StationSeries, t_s: float
                     ) -> Tuple[float, float, bool]:
    """Linearly interpolate (u, v) at ``t_s`` (UTC epoch seconds).

    Returns (u_east, v_north, in_window).  ``in_window`` is False if ``t_s``
    falls outside [series.times[0], series.times[-1]]; in that case (0, 0)
    is returned.
    """
    ts = series.times_utc_s
    if ts.size == 0:
        return 0.0, 0.0, False
    if t_s < ts[0] or t_s > ts[-1]:
        return 0.0, 0.0, False
    idx = int(np.searchsorted(ts, t_s))
    if idx == 0:
        return float(series.u_east_ms[0]), float(series.v_north_ms[0]), True
    if idx >= ts.size:
        return float(series.u_east_ms[-1]), float(series.v_north_ms[-1]), True
    t0 = ts[idx - 1]
    t1 = ts[idx]
    if t1 == t0:
        return float(series.u_east_ms[idx]), float(series.v_north_ms[idx]), True
    frac = (t_s - t0) / (t1 - t0)
    u = series.u_east_ms[idx - 1] + frac * (
        series.u_east_ms[idx] - series.u_east_ms[idx - 1])
    v = series.v_north_ms[idx - 1] + frac * (
        series.v_north_ms[idx] - series.v_north_ms[idx - 1])
    return float(u), float(v), True


# Approximate metres-per-degree at the centre of the Salish Sea.  IDW is
# scale-invariant in the weighting (1/d^2 normalised) so the exact scale
# only matters when *comparing* IDW vs. some absolute distance threshold —
# we don't.  But we want lat-vs-lon to be commensurate so the "nearest
# stations" set isn't biased.  At lat 48: 1 deg lat ≈ 111.32 km;
# 1 deg lon ≈ 111.32 * cos(48) ≈ 74.50 km.
_REF_LAT_DEG = 48.0
_LON_SCALE = math.cos(math.radians(_REF_LAT_DEG))


def _approx_xy_km(lat: float, lon: float) -> Tuple[float, float]:
    """Cheap planar approximation around the Salish Sea.  km."""
    return lon * 111.32 * _LON_SCALE, lat * 111.32


class CurrentField:
    """Spatial+temporal current field over a curated station catalog.

    Construct once at service start (Phase 2 will hold this on the Router).
    Each call to ``field_at`` is O(N_query * 3) for k=3 IDW; the loop over
    stations is over a small constant (the catalog size).

    Usage::

        cf = CurrentField.from_default_catalog()
        u, v = cf.current_at(48.4, -122.6, dt)                  # scalar
        uv = cf.field_at(pts, dt)                                # ndarray[N,2]
    """

    K_NEAREST = 3                # IDW over the 3 nearest stations
    IDW_POWER = 2                # inverse-distance squared weighting
    EPS_KM = 1e-3                # avoid div-by-zero when query == station

    def __init__(self, series: Sequence[_StationSeries]):
        self.series: List[_StationSeries] = list(series)
        if not self.series:
            raise ValueError("CurrentField requires at least one station")
        # Pre-stack station coords for vectorized distance.
        self._station_xy_km = np.asarray(
            [_approx_xy_km(s.lat, s.lon) for s in self.series],
            dtype=np.float64,
        )  # [S, 2]

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_default_catalog(cls,
                             stations: Optional[Iterable[Dict[str, Any]]] = None,
                             force_refresh: bool = False,
                             now: Optional[datetime] = None,
                             skip_on_fetch_error: bool = True,
                             ) -> "CurrentField":
        """Build a ``CurrentField`` from the curated station catalog.

        Fetches (or reads from cache) each station's hourly prediction window.
        Stations that fail to fetch are skipped with a warning unless
        ``skip_on_fetch_error`` is False.
        """
        catalog = list(stations) if stations is not None else load_station_catalog()
        out: List[_StationSeries] = []
        for st in catalog:
            sid = st["id"]
            try:
                blob = ensure_station_cache(sid, force_refresh=force_refresh, now=now)
            except (OSError, urllib.error.URLError, json.JSONDecodeError,
                    RuntimeError, KeyError) as e:
                if skip_on_fetch_error:
                    log.warning("station %s: cache/fetch failed (%s) — skipping",
                                sid, e)
                    continue
                raise
            out.append(_series_from_cache(blob, st["lat"], st["lon"]))
        return cls(out)

    @classmethod
    def from_series(cls, series: Sequence[_StationSeries]) -> "CurrentField":
        """Construct directly from synthetic series (used by tests)."""
        return cls(series)

    # ---- queries ---------------------------------------------------------

    def current_at(self, lat: float, lon: float, t: datetime
                   ) -> Tuple[float, float]:
        """Scalar query: return (u_east, v_north) at one (lat, lon, t).

        Returns (0, 0) and emits a warning if ``t`` is outside the cached
        prediction window for ALL contributing stations.
        """
        pts = np.asarray([[lat, lon]], dtype=np.float64)
        field = self.field_at(pts, t)
        return float(field[0, 0]), float(field[0, 1])

    def field_at(self, query_pts: np.ndarray, t: datetime) -> np.ndarray:
        """Vectorized query: return current vectors at N points at one time.

        Args:
            query_pts: ndarray[N, 2] of (lat, lon)
            t:         single datetime (timezone-aware preferred; naive is
                       treated as UTC)

        Returns:
            ndarray[N, 2] of (u_east, v_north) in m/s

        For the A* hot loop the typical N is "all cells in the prune bbox" =
        O(10^5).  k=3 IDW with S ~ 10 stations is O(N * S) for distance and
        O(N) for the final weighting — pure numpy.
        """
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        t_s = t.timestamp()

        N = int(query_pts.shape[0])
        if N == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Per-station current at this timestamp (S, 2).  Stations whose
        # window doesn't cover t contribute zero and are excluded by the
        # weight mask below.
        S = len(self.series)
        uv_per_station = np.zeros((S, 2), dtype=np.float64)
        valid_mask = np.zeros(S, dtype=bool)
        for i, s in enumerate(self.series):
            u, v, in_window = _temporal_interp(s, t_s)
            uv_per_station[i, 0] = u
            uv_per_station[i, 1] = v
            valid_mask[i] = in_window

        if not np.any(valid_mask):
            log.warning(
                "current_field_at: t=%s is outside the cached prediction "
                "window for all %d stations; returning zero current.",
                t.isoformat(), S)
            return np.zeros((N, 2), dtype=np.float64)

        # Distances from query pts to each station, planar km.
        qxy = np.empty((N, 2), dtype=np.float64)
        qxy[:, 0] = query_pts[:, 1] * 111.32 * _LON_SCALE
        qxy[:, 1] = query_pts[:, 0] * 111.32
        # dx[N, S], dy[N, S]
        dx = qxy[:, 0:1] - self._station_xy_km[None, :, 0]
        dy = qxy[:, 1:2] - self._station_xy_km[None, :, 1]
        d2 = dx * dx + dy * dy + self.EPS_KM  # [N, S]

        # For each query point pick the K nearest VALID stations.  We do
        # this with np.argpartition on the distance array after marking
        # invalid stations as infinite distance.
        d2_masked = np.where(valid_mask[None, :], d2, np.inf)

        K = min(self.K_NEAREST, int(np.sum(valid_mask)))
        if K <= 0:
            return np.zeros((N, 2), dtype=np.float64)

        # argpartition gives the K smallest indices unsorted; that's fine
        # for weighted averaging.
        idx_partitioned = np.argpartition(d2_masked, kth=K - 1, axis=1)[:, :K]
        # Gather the per-neighbor distance and vectors.
        rows = np.arange(N)[:, None]
        d2_k = d2_masked[rows, idx_partitioned]                      # [N, K]
        weights = 1.0 / np.power(d2_k, self.IDW_POWER / 2.0)         # IDW
        weights /= weights.sum(axis=1, keepdims=True)
        u_k = uv_per_station[idx_partitioned, 0]                     # [N, K]
        v_k = uv_per_station[idx_partitioned, 1]
        u = np.sum(weights * u_k, axis=1)
        v = np.sum(weights * v_k, axis=1)
        return np.stack([u, v], axis=1)


# ----- module-level convenience --------------------------------------------

_DEFAULT_FIELD: Optional[CurrentField] = None


def get_default_field() -> CurrentField:
    """Lazily build (and cache) a default ``CurrentField`` from the catalog.

    Calls NOAA on first use (or just reads cache files).  Intended for ad-hoc
    use from probe scripts + the Tolly tool path; the routing service holds
    its own instance with explicit refresh control.
    """
    global _DEFAULT_FIELD
    if _DEFAULT_FIELD is None:
        _DEFAULT_FIELD = CurrentField.from_default_catalog()
    return _DEFAULT_FIELD


def current_at(lat: float, lon: float, t: datetime) -> Tuple[float, float]:
    """Module-level scalar shortcut.  Builds the default field on first call."""
    return get_default_field().current_at(lat, lon, t)


def current_field_at(query_pts: np.ndarray, t: datetime) -> np.ndarray:
    """Module-level vectorized shortcut.  Builds the default field on first call."""
    return get_default_field().field_at(query_pts, t)
