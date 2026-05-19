"""Unit tests for tolly_router.polar."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import polar  # noqa: E402

KTS_TO_MS = 0.5144444444


class NoCurrentTest(unittest.TestCase):
    """SOG should equal STW when current is zero."""

    def test_no_current_effective_sog_equals_stw(self):
        sog, fuel = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=14.0, current_uv_ms=(0.0, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(sog, 14.0, places=4)
        self.assertAlmostEqual(fuel, 28.0, places=4)

    def test_no_current_north(self):
        sog, _ = polar.effective_sog_and_fuel(
            heading_deg=0.0, stw_kts=7.0, current_uv_ms=(0.0, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(sog, 7.0, places=4)


class CurrentEffectTest(unittest.TestCase):

    def test_following_current_adds(self):
        # Heading east at 14 kts, current 2 kts east => SOG ≈ 16 kts
        sog, _ = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=14.0,
            current_uv_ms=(2.0 * KTS_TO_MS, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(sog, 16.0, places=4)

    def test_opposing_current_subtracts(self):
        # Heading east at 14 kts, current 2 kts west => SOG ≈ 12 kts
        sog, _ = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=14.0,
            current_uv_ms=(-2.0 * KTS_TO_MS, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(sog, 12.0, places=4)

    def test_beam_current_crab_angle(self):
        # Heading east at 14 kts, current 1 m/s north
        # boat_through_water_ms = 14 kts * 0.5144 = 7.202 east
        # boat_over_ground_ms = (7.202 east, 1 north)
        # magnitude = sqrt(7.202^2 + 1) = sqrt(52.87) = 7.271 m/s
        # in kts = 7.271 / 0.5144 = 14.135 kts
        # So SOG is slightly GREATER than 14 (orthogonal component adds in
        # quadrature) — the brief said "slightly less" but Pythagoras
        # disagrees: a beam current always increases magnitude over the bare
        # STW. The boat does crab off course, that's the cost of beam current.
        sog, _ = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=14.0,
            current_uv_ms=(0.0, 1.0),
            polar=polar.VesselPolar(),
        )
        stw_ms = 14.0 * KTS_TO_MS
        expected_ms = math.hypot(stw_ms, 1.0)
        expected_kts = expected_ms / KTS_TO_MS
        self.assertAlmostEqual(sog, expected_kts, places=4)
        # Sanity: it's larger than STW, not smaller.
        self.assertGreater(sog, 14.0)


class FuelModeTest(unittest.TestCase):

    def test_mode_threshold_cruise(self):
        _, fuel = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=14.0, current_uv_ms=(0.0, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(fuel, 28.0, places=4)

    def test_mode_threshold_displacement(self):
        _, fuel = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=7.0, current_uv_ms=(0.0, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(fuel, 8.0, places=4)

    def test_mode_threshold_at_boundary(self):
        # threshold_kts = 9.0, exactly-9 uses displacement (strict >).
        _, fuel = polar.effective_sog_and_fuel(
            heading_deg=90.0, stw_kts=9.0, current_uv_ms=(0.0, 0.0),
            polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(fuel, 8.0, places=4)


class VectorizedTest(unittest.TestCase):
    """The vectorized form must agree with the scalar form."""

    def test_vector_matches_scalar(self):
        headings = np.array([0.0, 45.0, 90.0, 180.0, 270.0])
        stw = 14.0
        # Mixed current across headings
        currents = np.array([
            [0.0, 0.0],
            [1.0 * KTS_TO_MS, 0.0],
            [-2.0 * KTS_TO_MS, 0.0],
            [0.0, 1.0],
            [0.5, -0.5],
        ])
        sog_v, fuel_v = polar.effective_sog_and_fuel_vec(
            headings, stw, currents, polar=polar.VesselPolar(),
        )
        for i, h in enumerate(headings):
            sog_s, fuel_s = polar.effective_sog_and_fuel(
                heading_deg=float(h), stw_kts=stw,
                current_uv_ms=(float(currents[i, 0]), float(currents[i, 1])),
                polar=polar.VesselPolar(),
            )
            self.assertAlmostEqual(sog_v[i], sog_s, places=6)
            self.assertAlmostEqual(fuel_v[i], fuel_s, places=6)

    def test_vector_per_element_stw_picks_per_element_fuel(self):
        # Mixed STW: cruise + displacement in the same call.
        headings = np.array([90.0, 90.0])
        stw = np.array([14.0, 7.0])
        currents = np.zeros((2, 2))
        sog, fuel = polar.effective_sog_and_fuel_vec(
            headings, stw, currents, polar=polar.VesselPolar(),
        )
        self.assertAlmostEqual(fuel[0], 28.0, places=4)
        self.assertAlmostEqual(fuel[1], 8.0, places=4)

    def test_vector_broadcasts_scalar_current(self):
        headings = np.array([0.0, 90.0])
        stw = 14.0
        current = np.array([1.0 * KTS_TO_MS, 0.0])  # (2,) shape
        sog, fuel = polar.effective_sog_and_fuel_vec(
            headings, stw, current, polar=polar.VesselPolar(),
        )
        # heading=0: STW north + current east => sqrt(14^2 + 1^2) = 14.0357
        # heading=90: STW east + current east => 14 + 1 = 15
        self.assertAlmostEqual(sog[1], 15.0, places=4)
        self.assertAlmostEqual(sog[0], math.hypot(14.0, 1.0), places=4)


class PolarConfigLoadTest(unittest.TestCase):
    """``VesselPolar.load()`` precedence: user-config > learner > defaults.

    These tests force ``learned_path`` to a non-existent file so they
    exercise the pure user-config + defaults precedence layers, regardless
    of whether ``tolly-polar-learner.service`` has populated the cache on
    this machine.
    """

    NO_FILE = Path("/tmp/this-file-does-not-exist.json")

    def test_load_defaults_when_missing(self):
        p = polar.VesselPolar.load(self.NO_FILE, learned_path=self.NO_FILE)
        self.assertEqual(p, polar.VesselPolar())

    def test_load_partial_override(self):
        # Write a partial file, load it, verify only known keys are picked up.
        import json
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"cruise_stw_kts": 12.5,
                       "bogus_key": "ignored"}, f)
            path = Path(f.name)
        try:
            p = polar.VesselPolar.load(path, learned_path=self.NO_FILE)
            self.assertAlmostEqual(p.cruise_stw_kts, 12.5)
            # Defaults for the rest
            self.assertAlmostEqual(p.cruise_fuel_gph, 28.0)
            self.assertAlmostEqual(p.mode_threshold_kts, 9.0)
        finally:
            path.unlink()

    def test_load_learner_fills_when_no_user_config(self):
        """Learner JSON supplies values when the user file is absent."""
        import json
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"cruise_fuel_gph": 30.5,
                       "displacement_fuel_gph": 9.1,
                       "_source": "tolly-polar-learner"}, f)
            learned = Path(f.name)
        try:
            p = polar.VesselPolar.load(self.NO_FILE, learned_path=learned)
            self.assertAlmostEqual(p.cruise_fuel_gph, 30.5)
            self.assertAlmostEqual(p.displacement_fuel_gph, 9.1)
            # untouched fields stay at defaults
            self.assertAlmostEqual(p.cruise_stw_kts, 14.0)
        finally:
            learned.unlink()

    def test_load_user_config_overrides_learner_per_field(self):
        """User config wins per-key; unspecified keys come from learner."""
        import json
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"cruise_fuel_gph": 26.5}, f)  # user pins one
            user = Path(f.name)
        with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"cruise_fuel_gph": 30.5,
                       "displacement_fuel_gph": 9.1}, f)
            learned = Path(f.name)
        try:
            p = polar.VesselPolar.load(user, learned_path=learned)
            # user wins for cruise_fuel_gph
            self.assertAlmostEqual(p.cruise_fuel_gph, 26.5)
            # learner provides displacement_fuel_gph (no user override)
            self.assertAlmostEqual(p.displacement_fuel_gph, 9.1)
        finally:
            user.unlink()
            learned.unlink()


if __name__ == "__main__":
    unittest.main()
