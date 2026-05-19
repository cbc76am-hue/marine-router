"""Unit tests for tolly_router.currents."""
from __future__ import annotations

import logging
import socket
import sys
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import currents as cu  # noqa: E402


def _make_synthetic_series(station_id, lat, lon, t0_utc, hours,
                           u_const_ms=0.0, v_const_ms=0.0):
    """Build a _StationSeries with constant (u, v) over ``hours``."""
    times = np.array(
        [(t0_utc + timedelta(hours=h)).timestamp() for h in range(hours)],
        dtype=np.float64,
    )
    u = np.full(hours, u_const_ms, dtype=np.float64)
    v = np.full(hours, v_const_ms, dtype=np.float64)
    return cu._StationSeries(
        station_id=station_id, lat=lat, lon=lon,
        times_utc_s=times, u_east_ms=u, v_north_ms=v,
    )


def _net_available(host="api.tidesandcurrents.noaa.gov", timeout=2.0):
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


class DirectionConventionTest(unittest.TestCase):
    """NOAA's Velocity_Major sign + meanFloodDir/meanEbbDir -> (u, v) m/s."""

    def test_direction_convention_flood_east(self):
        # Flood at 2.0 kts flowing toward compass 90 (east). Ebb dir is set
        # to 270 (west) — unused for positive signed_kts.
        u, v = cu._signed_kts_to_uv(2.0, flood_dir_deg=90.0, ebb_dir_deg=270.0)
        # 2 kts * 0.5144 m/s = 1.0289 m/s east, 0 north
        self.assertAlmostEqual(u, 1.0289, places=3)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_direction_convention_ebb_uses_ebb_dir(self):
        # Negative signed_kts -> magnitude flows TOWARD meanEbbDir.
        u, v = cu._signed_kts_to_uv(-2.0, flood_dir_deg=90.0, ebb_dir_deg=270.0)
        # 2 kts toward 270 = west = -u, 0 v
        self.assertAlmostEqual(u, -1.0289, places=3)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_direction_convention_north(self):
        u, v = cu._signed_kts_to_uv(1.0, flood_dir_deg=0.0, ebb_dir_deg=180.0)
        # 1 kt north => 0 u, +0.5144 v
        self.assertAlmostEqual(u, 0.0, places=6)
        self.assertAlmostEqual(v, 0.5144, places=3)


class ParsePredictionsTest(unittest.TestCase):
    """End-to-end shape: a synthetic NOAA payload parses correctly."""

    def test_parse_payload_signed_kts(self):
        payload = {
            "current_predictions": {
                "cp": [
                    {"Time": "2026-05-18 00:00",
                     "Velocity_Major": 2.0,
                     "meanFloodDir": 90,
                     "meanEbbDir": 270},
                    {"Time": "2026-05-18 01:00",
                     "Velocity_Major": -3.0,
                     "meanFloodDir": 90,
                     "meanEbbDir": 270},
                ]
            }
        }
        t, u, v = cu.parse_predictions(payload)
        self.assertEqual(t.size, 2)
        # First entry: 2 kts east -> ~+1.029 m/s east, 0 north
        self.assertAlmostEqual(u[0], 1.0289, places=3)
        self.assertAlmostEqual(v[0], 0.0, places=6)
        # Second: -3 kts (ebb dir 270 = west) -> -1.543 m/s east, 0 north
        self.assertAlmostEqual(u[1], -1.5433, places=3)
        self.assertAlmostEqual(v[1], 0.0, places=6)


class TemporalInterpolationTest(unittest.TestCase):

    def test_temporal_interpolation_midpoint(self):
        t0 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=1)
        series = cu._StationSeries(
            station_id="TEST",
            lat=48.0, lon=-122.0,
            times_utc_s=np.array([t0.timestamp(), t1.timestamp()], dtype=np.float64),
            u_east_ms=np.array([1.0, 3.0], dtype=np.float64),
            v_north_ms=np.array([0.5, -0.5], dtype=np.float64),
        )
        midpoint = t0 + timedelta(minutes=30)
        u, v, ok = cu._temporal_interp(series, midpoint.timestamp())
        self.assertTrue(ok)
        # Midpoint of (1,3) is 2.0; midpoint of (0.5,-0.5) is 0.0.
        self.assertAlmostEqual(u, 2.0, places=6)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_temporal_interpolation_at_exact_sample(self):
        t0 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(hours=1)
        series = cu._StationSeries(
            station_id="TEST",
            lat=48.0, lon=-122.0,
            times_utc_s=np.array([t0.timestamp(), t1.timestamp()], dtype=np.float64),
            u_east_ms=np.array([1.0, 3.0], dtype=np.float64),
            v_north_ms=np.array([0.5, -0.5], dtype=np.float64),
        )
        u, v, ok = cu._temporal_interp(series, t1.timestamp())
        self.assertTrue(ok)
        self.assertAlmostEqual(u, 3.0, places=6)
        self.assertAlmostEqual(v, -0.5, places=6)


class SpatialIDWTest(unittest.TestCase):

    def test_spatial_idw_three_equidistant_stations(self):
        # Three stations at the corners of an equilateral-ish triangle on a
        # local tangent plane around 48 N. Their current vectors are different;
        # the query point sits at the geographic centroid. With three
        # equidistant stations IDW reduces to a uniform mean.
        # Build a 3-station catalog where the IDW point is equidistant to all.
        # Place at angles 0/120/240 around centre (48.0, -122.5) at 0.05 deg.
        t0 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        cx, cy = 48.0, -122.5
        # Triangle vertices on lat/lon
        import math as _m
        verts = []
        cosphi = _m.cos(_m.radians(cx))
        for ang_deg in (0.0, 120.0, 240.0):
            ang = _m.radians(ang_deg)
            # 0.1 deg of "equivalent" radius — adjust lon by cos lat so the
            # planar approximation in CurrentField treats them as equidistant.
            dlat = 0.1 * _m.cos(ang)
            dlon = (0.1 * _m.sin(ang)) / cosphi
            verts.append((cx + dlat, cy + dlon))

        # Three synthetic series, each with constant (u, v) at hours [0..3].
        uv_vals = [(1.0, 0.0), (0.0, 2.0), (-3.0, 1.0)]
        all_series = []
        for (lat, lon), (uu, vv) in zip(verts, uv_vals):
            all_series.append(_make_synthetic_series(
                "S?", lat, lon, t0, hours=3,
                u_const_ms=uu, v_const_ms=vv,
            ))
        cf = cu.CurrentField.from_series(all_series)

        # Query at the centroid at the second hour (in-window).
        query = np.array([[cx, cy]], dtype=np.float64)
        when = t0 + timedelta(hours=1)
        uv = cf.field_at(query, when)
        # Average of the three: u = (1+0-3)/3 = -0.6667; v = (0+2+1)/3 = 1.0
        self.assertAlmostEqual(uv[0, 0], -2.0 / 3.0, places=3)
        self.assertAlmostEqual(uv[0, 1], 1.0, places=3)


class OutsideWindowTest(unittest.TestCase):

    def test_outside_window_returns_zero_with_warning(self):
        t0 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        series = _make_synthetic_series(
            "TEST", 48.0, -122.5, t0, hours=2, u_const_ms=1.0, v_const_ms=0.5,
        )
        cf = cu.CurrentField.from_series([series])
        future = t0 + timedelta(days=365)
        with self.assertLogs("tolly_router.currents", level=logging.WARNING) as cm:
            u, v = cf.current_at(48.0, -122.5, future)
        self.assertEqual(u, 0.0)
        self.assertEqual(v, 0.0)
        self.assertTrue(any("outside the cached prediction" in line for line in cm.output))


@unittest.skipUnless(_net_available(), "no network to api.tidesandcurrents.noaa.gov")
class LiveNOAAFetchSmokeTest(unittest.TestCase):
    """Hits NOAA. Skipped automatically when offline."""

    def test_live_noaa_fetch_smoke(self):
        import time as _t
        t0 = _t.monotonic()
        now = datetime.now(timezone.utc)
        # Bypass any pre-existing cache so we exercise the live path.
        blob = cu.ensure_station_cache("PUG1701", force_refresh=True, now=now)
        elapsed = _t.monotonic() - t0
        self.assertLess(elapsed, 10.0, f"NOAA fetch took {elapsed:.1f}s")
        self.assertIn("raw", blob)
        # Parsed arrays should be present and non-empty.
        self.assertGreater(len(blob.get("times_utc_s") or []), 0)
        # u, v entries should be finite numbers, not all zero.
        u = blob["u_east_ms"]
        v = blob["v_north_ms"]
        self.assertTrue(any(abs(x) > 0.01 for x in u + v),
                        "Deception Pass should have non-zero current values")


if __name__ == "__main__":
    unittest.main()
