#!/usr/bin/env python3
"""Probe NOAA tidal-current predictions for one station + time.

Used to manually verify that what ``tolly_router.currents`` parses matches
the NOAA website. Prints both the (u_east, v_north) vector AND the
equivalent (speed_kts, direction_true_deg) so you can cross-check the
station page on tidesandcurrents.noaa.gov.

Usage::

    python3 scripts/probe_currents.py --station PUG1535 \
        --time '2026-05-18T14:00:00Z'

If --time is omitted, uses 'now'.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import currents as cu  # noqa: E402
from tolly_router.polar import MS_TO_KTS  # noqa: E402


def _parse_time(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _uv_to_speed_dir(u_ms: float, v_ms: float):
    """(u_east, v_north) m/s -> (speed_kts, compass_true_deg) flowing toward."""
    speed_ms = math.hypot(u_ms, v_ms)
    if speed_ms == 0.0:
        return 0.0, 0.0
    # atan2(east, north) gives bearing where 0=N, 90=E
    dir_rad = math.atan2(u_ms, v_ms)
    dir_deg = (math.degrees(dir_rad) + 360.0) % 360.0
    return speed_ms * MS_TO_KTS, dir_deg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", required=True,
                    help="NOAA station ID (e.g. PUG1701).")
    ap.add_argument("--time", default=None,
                    help="ISO 8601 query time (UTC); defaults to now.")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Bypass the on-disk cache and re-fetch from NOAA.")
    args = ap.parse_args(argv)

    when = _parse_time(args.time) if args.time else datetime.now(timezone.utc)

    # Try to find the station's coords in the catalog so the IDW/spatial layer
    # can locate it. If unknown, treat lat/lon as the same as the query — but
    # then the spatial step is a no-op anyway because we only have one station.
    catalog = cu.load_station_catalog()
    match = next((s for s in catalog if s["id"] == args.station), None)
    if match is None:
        print(f"# station {args.station} not in catalog; spatial info unavailable",
              file=sys.stderr)
        lat, lon = 0.0, 0.0
    else:
        lat, lon = match["lat"], match["lon"]
        print(f"# station: {match['id']}  '{match['name']}'  "
              f"({lat}, {lon})")

    # Build a single-station CurrentField so the spatial interp returns this
    # station's vector verbatim at the query coord.
    blob = cu.ensure_station_cache(args.station, force_refresh=args.force_refresh)
    series = cu._series_from_cache(blob, lat, lon)
    cf = cu.CurrentField.from_series([series])

    print(f"# query time: {when.isoformat()}")
    print(f"# cache fetched_at: {blob.get('fetched_at')}")
    print(f"# cache window: {blob.get('begin_date')} .. {blob.get('end_date')}")

    u, v = cf.current_at(lat or 48.0, lon or -122.5, when)
    speed_kts, dir_true = _uv_to_speed_dir(u, v)
    print(f"u_east = {u:+.4f} m/s")
    print(f"v_north= {v:+.4f} m/s")
    print(f"speed  = {speed_kts:.3f} kts")
    print(f"dir    = {dir_true:.1f} deg true (flowing toward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
