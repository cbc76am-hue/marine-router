"""Unit tests for tolly_router.polar_log."""
from __future__ import annotations

import random
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import polar_log  # noqa: E402


def _balanced_sample(ts: int, sog: float, total_fuel: float,
                     rpm: float = 2400.0) -> dict:
    """Build a sample dict that passes the engaged-balanced filter and
    has a known total_fuel split evenly between port and starboard."""
    return {
        "ts": ts,
        "sog_kts": sog,
        "port_rpm": rpm,
        "port_fuel_gph": total_fuel / 2.0,
        "port_load": 0.5,
        "stbd_rpm": rpm,
        "stbd_fuel_gph": total_fuel / 2.0,
        "stbd_load": 0.5,
    }


def _sample_to_tuple(s: dict) -> tuple:
    """Match the column ordering used by polar_log._SAMPLE_COLS."""
    return tuple(s.get(c) for c in polar_log._SAMPLE_COLS)


class SchemaTest(unittest.TestCase):

    def test_schema_init(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "p.db"
            conn = polar_log.connect(db_path)
            try:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' ORDER BY name"
                )
                tables = [r[0] for r in cur.fetchall()]
                self.assertIn("samples", tables)
                self.assertIn("learned_runs", tables)

                # ts index is present
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
                indexes = [r[0] for r in cur.fetchall()]
                self.assertIn("idx_samples_ts", indexes)
            finally:
                conn.close()

    def test_connect_is_idempotent(self):
        """Calling connect twice on the same file should not error."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "p.db"
            c1 = polar_log.connect(db_path)
            c1.close()
            c2 = polar_log.connect(db_path)
            c2.close()


class InsertQueryTest(unittest.TestCase):

    def test_insert_then_query(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "p.db"
            conn = polar_log.connect(db_path)
            try:
                now = int(time.time())
                row = _balanced_sample(now, sog=7.5, total_fuel=10.0)
                polar_log.insert_sample(conn, row)

                got = polar_log.recent_samples(conn, hours=1)
                self.assertEqual(len(got), 1)
                d = dict(zip(polar_log._SAMPLE_COLS, got[0]))
                self.assertEqual(d["ts"], now)
                self.assertAlmostEqual(d["sog_kts"], 7.5)
                self.assertAlmostEqual(d["port_fuel_gph"], 5.0)
                self.assertAlmostEqual(d["stbd_fuel_gph"], 5.0)
            finally:
                conn.close()

    def test_recent_samples_respects_cutoff(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "p.db"
            conn = polar_log.connect(db_path)
            try:
                now = int(time.time())
                # one fresh, one old (2h ago)
                polar_log.insert_sample(
                    conn, _balanced_sample(now, 7.5, 10.0))
                polar_log.insert_sample(
                    conn, _balanced_sample(now - 7200, 7.5, 10.0))

                got = polar_log.recent_samples(conn, hours=1)
                self.assertEqual(len(got), 1)
            finally:
                conn.close()


class FitTest(unittest.TestCase):

    def test_fit_too_few_samples(self):
        # 5 samples per bin, far below MIN_SAMPLES_PER_BIN=30 -> None
        rows = []
        ts = int(time.time())
        for i in range(5):
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + i, sog=7.5, total_fuel=10.0)))
        for i in range(5):
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + 100 + i, sog=14.0, total_fuel=30.0)))
        self.assertIsNone(polar_log.fit_polar(rows))

    def test_fit_clear_signal(self):
        """100 samples per bin, no noise — median should land exactly."""
        rows = []
        ts = int(time.time())
        # displacement: SOG 6.5-8.5, fuel 10.0
        for i in range(100):
            sog = 6.5 + (i % 5) * 0.4  # 6.5..8.1
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + i, sog=sog, total_fuel=10.0)))
        # cruise: SOG 13-15, fuel 30.0
        for i in range(100):
            sog = 13.0 + (i % 5) * 0.4
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + 1000 + i, sog=sog, total_fuel=30.0)))

        fit = polar_log.fit_polar(rows)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["displacement_fuel_gph"], 10.0, delta=0.5)
        self.assertAlmostEqual(fit["cruise_fuel_gph"], 30.0, delta=0.5)
        self.assertEqual(fit["_samples_displacement"], 100)
        self.assertEqual(fit["_samples_cruise"], 100)
        self.assertEqual(fit["_source"], "tolly-polar-learner")
        # reference speeds preserved
        self.assertAlmostEqual(fit["displacement_stw_kts"], 7.5)
        self.assertAlmostEqual(fit["cruise_stw_kts"], 14.0)
        self.assertAlmostEqual(fit["mode_threshold_kts"], 9.0)

    def test_fit_outliers_handled(self):
        """5% extreme outliers shouldn't move the median much."""
        rng = random.Random(42)
        rows = []
        ts = int(time.time())
        true_disp = 10.0
        true_cruise = 30.0
        n_per = 100

        for i in range(n_per):
            sog = rng.uniform(6.0, 8.5)
            # base fuel + small noise
            fuel = true_disp + rng.gauss(0, 0.2)
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + i, sog=sog, total_fuel=fuel)))
        # 5% outliers in the displacement bin (5 of 100): wildly high
        for i in range(5):
            sog = rng.uniform(6.0, 8.5)
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + 500 + i, sog=sog, total_fuel=80.0)))

        for i in range(n_per):
            sog = rng.uniform(13.0, 15.5)
            fuel = true_cruise + rng.gauss(0, 0.4)
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + 1000 + i, sog=sog, total_fuel=fuel)))
        for i in range(5):
            sog = rng.uniform(13.0, 15.5)
            rows.append(_sample_to_tuple(
                _balanced_sample(ts + 1500 + i, sog=sog, total_fuel=200.0)))

        fit = polar_log.fit_polar(rows)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["displacement_fuel_gph"], true_disp,
                               delta=1.0)
        self.assertAlmostEqual(fit["cruise_fuel_gph"], true_cruise,
                               delta=1.0)


class FilterTest(unittest.TestCase):
    """Sanity that the filter keeps engaged-balanced and drops everything else."""

    def test_filter_drops_idle(self):
        # 500 idle-engine samples shouldn't produce a fit.
        rows = []
        ts = int(time.time())
        for i in range(500):
            rows.append(_sample_to_tuple(_balanced_sample(
                ts + i, sog=7.5, total_fuel=10.0, rpm=500.0)))
        self.assertIsNone(polar_log.fit_polar(rows))

    def test_filter_drops_one_engine(self):
        # 500 samples with port engaged + stbd idle (huge asymmetry) -> None.
        rows = []
        ts = int(time.time())
        for i in range(500):
            s = _balanced_sample(ts + i, sog=7.5, total_fuel=10.0)
            s["port_rpm"] = 2400.0
            s["stbd_rpm"] = 700.0  # >600 so it passes the min, but
            # the asymmetry filter should fire.
            rows.append(_sample_to_tuple(s))
        self.assertIsNone(polar_log.fit_polar(rows))


class LearnedRunsTest(unittest.TestCase):

    def test_insert_learned_run_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "p.db"
            conn = polar_log.connect(db_path)
            try:
                fit = {
                    "displacement_stw_kts": 7.5,
                    "displacement_fuel_gph": 9.7,
                    "cruise_stw_kts": 14.0,
                    "cruise_fuel_gph": 29.6,
                    "mode_threshold_kts": 9.0,
                    "_calibrated_at": "2026-05-18T00:00:00Z",
                    "_samples_displacement": 42,
                    "_samples_cruise": 38,
                    "_source": "tolly-polar-learner",
                }
                polar_log.insert_learned_run(conn, fit)
                cur = conn.execute(
                    "SELECT n_samples, displacement_fuel_gph, cruise_fuel_gph "
                    "FROM learned_runs"
                )
                rows = cur.fetchall()
                self.assertEqual(len(rows), 1)
                n, d_fuel, c_fuel = rows[0]
                self.assertEqual(n, 80)
                self.assertAlmostEqual(d_fuel, 9.7)
                self.assertAlmostEqual(c_fuel, 29.6)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
