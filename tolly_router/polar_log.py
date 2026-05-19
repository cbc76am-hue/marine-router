"""Storage + curve-fit for the polar learner service.

Owns the SQLite schema (`samples` and `learned_runs`), provides read/write
helpers, and contains the curve-fit logic that turns a stream of engine
samples into a two-point polar (displacement + cruise fuel rates).

The fit is conservative on purpose:

  * filter for "engaged, balanced cruising" so we're not learning from
    idle / one-engine / docking samples
  * bin by SOG into the two regimes the existing polar model uses
  * require >= 30 samples per bin
  * use the median per bin (robust to outliers like fuel-rail glitches)

Reference speeds stay at the configured defaults; only the fuel rate at those
speeds is learned.  Regressing fuel vs RPM (and learning the reference STW
itself) needs many more samples to disentangle prop-pitch effects.

Schema (created on first connect, idempotent):

    CREATE TABLE samples (
        ts             INTEGER NOT NULL,  -- UTC seconds since epoch
        sog_kts        REAL,
        port_rpm       REAL,
        port_fuel_gph  REAL,
        port_load      REAL,
        stbd_rpm       REAL,
        stbd_fuel_gph  REAL,
        stbd_load      REAL
    );
    CREATE INDEX idx_samples_ts ON samples(ts);

    CREATE TABLE learned_runs (
        ts                       INTEGER NOT NULL,
        n_samples                INTEGER,
        displacement_stw_kts     REAL,
        displacement_fuel_gph    REAL,
        cruise_stw_kts           REAL,
        cruise_fuel_gph          REAL,
        mode_threshold_kts       REAL,
        raw_json                 TEXT
    );
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Reference speeds used as the bin "center" the learner labels samples
# against.  Held in code (not config) so the learner's output schema is
# self-describing and reproducible.  Match polar.py's defaults.
DISPLACEMENT_STW_KTS_REF = 7.5
CRUISE_STW_KTS_REF = 14.0
MODE_THRESHOLD_KTS_DEFAULT = 9.0

# SOG bin edges for the curve fit.
DISPLACEMENT_BIN = (5.5, 9.0)
CRUISE_BIN = (12.0, 16.0)

# Filter thresholds for which raw samples count toward the fit.
MIN_SOG_KTS = 1.0          # boat must be moving
MIN_RPM = 600.0            # engine engaged (above idle)
MAX_RPM_ASYMMETRY = 500.0  # |port-stbd| RPM, drop one-engine / load-asym

# Minimum samples per bin to produce a learned polar.
MIN_SAMPLES_PER_BIN = 30


# -------------------------------------------------------------------------
# Schema + connect
# -------------------------------------------------------------------------

_SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS samples (
        ts            INTEGER NOT NULL,
        sog_kts       REAL,
        port_rpm      REAL,
        port_fuel_gph REAL,
        port_load     REAL,
        stbd_rpm      REAL,
        stbd_fuel_gph REAL,
        stbd_load     REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts)",
    """
    CREATE TABLE IF NOT EXISTS learned_runs (
        ts                    INTEGER NOT NULL,
        n_samples             INTEGER,
        displacement_stw_kts  REAL,
        displacement_fuel_gph REAL,
        cruise_stw_kts        REAL,
        cruise_fuel_gph       REAL,
        mode_threshold_kts    REAL,
        raw_json              TEXT
    )
    """,
]


def connect(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path``, create parent dir + schema if missing.  Returns the
    connection (caller closes).  Uses WAL so the learner can write while a
    human runs `sqlite3 ... "SELECT ..."` on the same DB."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for stmt in _SCHEMA_SQL:
        conn.execute(stmt)
    conn.commit()
    return conn


# -------------------------------------------------------------------------
# Read/write helpers
# -------------------------------------------------------------------------

# Column order for inserts + queries.  Keep these symmetric so the tuple
# layout returned by `recent_samples` matches the dict accepted by
# `insert_sample`.
_SAMPLE_COLS = (
    "ts", "sog_kts",
    "port_rpm", "port_fuel_gph", "port_load",
    "stbd_rpm", "stbd_fuel_gph", "stbd_load",
)


def insert_sample(conn: sqlite3.Connection, sample: dict) -> None:
    """Write one row.  Missing keys land as SQL NULL.  ``ts`` is required."""
    if "ts" not in sample:
        raise ValueError("insert_sample: 'ts' is required")
    values = tuple(sample.get(c) for c in _SAMPLE_COLS)
    placeholders = ",".join("?" for _ in _SAMPLE_COLS)
    cols = ",".join(_SAMPLE_COLS)
    conn.execute(
        f"INSERT INTO samples ({cols}) VALUES ({placeholders})", values
    )
    conn.commit()


def recent_samples(conn: sqlite3.Connection, hours: int) -> list[tuple]:
    """Return all sample rows newer than ``hours`` ago, ordered by ts."""
    cutoff = int(datetime.now(timezone.utc).timestamp()) - hours * 3600
    cur = conn.execute(
        f"SELECT {','.join(_SAMPLE_COLS)} FROM samples "
        f"WHERE ts >= ? ORDER BY ts ASC",
        (cutoff,),
    )
    return list(cur.fetchall())


def insert_learned_run(conn: sqlite3.Connection, fit: dict) -> None:
    """Append a learned_runs row from the dict produced by ``fit_polar``."""
    ts = int(datetime.now(timezone.utc).timestamp())
    conn.execute(
        """INSERT INTO learned_runs (
            ts, n_samples,
            displacement_stw_kts, displacement_fuel_gph,
            cruise_stw_kts, cruise_fuel_gph,
            mode_threshold_kts, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ts,
            (fit.get("_samples_displacement") or 0)
            + (fit.get("_samples_cruise") or 0),
            fit.get("displacement_stw_kts"),
            fit.get("displacement_fuel_gph"),
            fit.get("cruise_stw_kts"),
            fit.get("cruise_fuel_gph"),
            fit.get("mode_threshold_kts"),
            json.dumps(fit, separators=(",", ":"), sort_keys=True),
        ),
    )
    conn.commit()


# -------------------------------------------------------------------------
# Curve fit
# -------------------------------------------------------------------------

def _row_to_dict(row: tuple) -> dict:
    return dict(zip(_SAMPLE_COLS, row))


def _passes_filter(s: dict) -> bool:
    """Sample is 'engaged balanced cruising' suitable for the fit.

    Drop:
      * not moving
      * either engine idle or off
      * one-engine / large load asymmetry
      * missing critical fields
    """
    sog = s.get("sog_kts")
    p_rpm = s.get("port_rpm")
    s_rpm = s.get("stbd_rpm")
    p_fuel = s.get("port_fuel_gph")
    s_fuel = s.get("stbd_fuel_gph")
    if None in (sog, p_rpm, s_rpm, p_fuel, s_fuel):
        return False
    if sog < MIN_SOG_KTS:
        return False
    if p_rpm <= MIN_RPM or s_rpm <= MIN_RPM:
        return False
    if abs(p_rpm - s_rpm) >= MAX_RPM_ASYMMETRY:
        return False
    return True


def fit_polar(samples: Iterable[tuple]) -> dict | None:
    """Compute a two-point polar from raw samples.

    Returns ``None`` if either bin has fewer than ``MIN_SAMPLES_PER_BIN``
    qualifying rows.  Returns a JSON-serializable dict on success — see the
    `_source: "tolly-polar-learner"` shape in the brief.
    """
    disp_fuels: list[float] = []
    cruise_fuels: list[float] = []
    disp_lo, disp_hi = DISPLACEMENT_BIN
    cruise_lo, cruise_hi = CRUISE_BIN

    for row in samples:
        s = _row_to_dict(row)
        if not _passes_filter(s):
            continue
        total = (s["port_fuel_gph"] or 0.0) + (s["stbd_fuel_gph"] or 0.0)
        sog = s["sog_kts"]
        if disp_lo <= sog <= disp_hi:
            disp_fuels.append(total)
        elif cruise_lo <= sog <= cruise_hi:
            cruise_fuels.append(total)

    n_disp = len(disp_fuels)
    n_cruise = len(cruise_fuels)
    if n_disp < MIN_SAMPLES_PER_BIN or n_cruise < MIN_SAMPLES_PER_BIN:
        return None

    disp_median = statistics.median(disp_fuels)
    cruise_median = statistics.median(cruise_fuels)

    return {
        "displacement_stw_kts": DISPLACEMENT_STW_KTS_REF,
        "displacement_fuel_gph": round(disp_median, 4),
        "cruise_stw_kts": CRUISE_STW_KTS_REF,
        "cruise_fuel_gph": round(cruise_median, 4),
        "mode_threshold_kts": MODE_THRESHOLD_KTS_DEFAULT,
        "_calibrated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "_samples_displacement": n_disp,
        "_samples_cruise": n_cruise,
        "_source": "tolly-polar-learner",
    }
