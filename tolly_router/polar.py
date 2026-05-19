"""Vessel speed + fuel model for time/fuel route optimization.

Standalone, importable.  The spatio-temporal A* calls
``effective_sog_and_fuel`` per (edge, time) pair to compute edge cost, so the
public API is vectorized: pass numpy arrays of headings and current vectors,
get back arrays of SOG and fuel rate.

Vessel defaults
---------------
Defaults match the 1974 Tollycraft 34 with twin MerCruiser 357 MAG MPI EFI
engines:

  cruise_stw_kts        14.0   speed through water at ~3000 RPM
  cruise_fuel_gph       28.0   both engines combined, on plane
  displacement_stw_kts   7.5   hull-speed economy mode at ~1500 RPM
  displacement_fuel_gph  8.0   both engines combined at displacement
  mode_threshold_kts     9.0   STW > threshold => cruise fuel rate

These are config knobs, not constants — refined from real engine CAN data
via the polar learner once samples accumulate.

Config file precedence
----------------------
``VesselPolar.load()`` merges three sources, key-by-key, in this priority
order (highest wins):

  1. ``~/.config/marine-router/polar.json`` — hand-edited user override.
     Highest priority so a human can pin a single value (e.g. force the
     mode threshold) without disabling the learner.
  2. ``~/.cache/marine-router/polar-learned.json`` — written by the
     ``tolly-polar-learner.service`` (see ``scripts/tolly_polar_learner.py``)
     after fitting a curve to recent engine CAN data.
  3. Built-in defaults — last resort.

Each file may set any subset of keys; unset keys fall through to the next
layer.  If both override files exist, ``VesselPolar.load`` logs one INFO
line at load showing which fields came from which source.

File schema (all keys optional; missing keys fall through to the next layer)::

    {
        "cruise_stw_kts":          14.0,
        "cruise_fuel_gph":         28.0,
        "displacement_stw_kts":     7.5,
        "displacement_fuel_gph":    8.0,
        "mode_threshold_kts":       9.0
    }

The learner file may include extra ``_calibrated_at`` / ``_samples_*`` /
``_source`` metadata fields; ``load`` ignores anything not in the dataclass.

Direction convention
--------------------
``heading_deg`` is the boat's heading in compass-true degrees (0 = north,
90 = east, etc.) — the direction the bow is pointing.  Current vectors are
``(u_east, v_north)`` in metres/second as documented in
``tolly_router.currents``.

Vector math
-----------
Given heading (deg true), STW (kts), and current (u, v) in m/s::

    boat_through_water_kts = STW * (sin(heading_rad), cos(heading_rad))
    boat_through_water_ms  = boat_through_water_kts * KTS_TO_MS
    boat_over_ground_ms    = boat_through_water_ms + current_ms
    SOG_kts                = |boat_over_ground_ms| / KTS_TO_MS

Fuel rate depends on STW only (a flat polar — the right shape for a planing
powerboat run at one of two governed RPM bands). Cruise vs displacement
selection: ``STW > mode_threshold_kts`` => cruise rate, else displacement.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

KTS_TO_MS = 0.5144444444
MS_TO_KTS = 1.0 / KTS_TO_MS

CONFIG_PATH = Path(os.path.expanduser("~/.config/marine-router/polar.json"))
LEARNED_PATH = Path(os.path.expanduser(
    "~/.cache/marine-router/polar-learned.json"
))


def _read_blob(path: Path) -> dict | None:
    """Read + JSON-parse ``path``.  Returns None if missing/unreadable.

    Soft-fails (logs a warning) on malformed JSON so a stale/corrupt cache
    can't take the routing service down.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("polar config %s unreadable (%s); skipping layer", p, e)
        return None
    if not isinstance(blob, dict):
        log.warning("polar config %s top-level not an object; skipping", p)
        return None
    return blob


def _coerce_known(blob: dict, known_fields: set[str]) -> dict:
    """Keep only known dataclass fields; coerce values to float."""
    out: dict = {}
    for k, v in blob.items():
        if k not in known_fields:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            log.warning("polar config field %r=%r not coercible to float; "
                        "skipping", k, v)
    return out


@dataclass(frozen=True)
class VesselPolar:
    """Speed + fuel model. All units knots / gph; see module docstring."""
    cruise_stw_kts: float = 14.0
    cruise_fuel_gph: float = 28.0
    displacement_stw_kts: float = 7.5
    displacement_fuel_gph: float = 8.0
    mode_threshold_kts: float = 9.0

    @classmethod
    def load(cls,
             path: Path = CONFIG_PATH,
             learned_path: Path = LEARNED_PATH) -> "VesselPolar":
        """Load with layered precedence: user-config > learner > defaults.

        ``path`` is the manual override (``~/.config/marine-router/polar.json``);
        ``learned_path`` is the learner-written cache.  Per-key merge: a user
        may pin one field and let the learner provide the rest.

        If both override files exist, logs one INFO line describing which
        fields landed where.  Anything not coercible to float is dropped
        with a warning, never crashes.
        """
        known = {f.name for f in fields(cls)}
        defaults = asdict(cls())

        user_blob = _read_blob(Path(path))
        learner_blob = _read_blob(Path(learned_path))

        user_kept = _coerce_known(user_blob, known) if user_blob else {}
        learner_kept = (
            _coerce_known(learner_blob, known) if learner_blob else {}
        )

        # Merge: defaults < learner < user.
        merged = {**defaults, **learner_kept, **user_kept}

        if user_blob is not None and learner_blob is not None:
            # Both files present: show provenance once at load.  Helpful
            # when troubleshooting "why does the router think fuel is 30 gph?"
            default_fields = sorted(
                k for k in known
                if k not in user_kept and k not in learner_kept
            )
            log.info(
                "polar: user-config %s, learner-data %s, defaults %s",
                sorted(user_kept.keys()),
                sorted(k for k in learner_kept if k not in user_kept),
                default_fields,
            )

        return cls(**merged)

    def fuel_rate_for_stw(self, stw_kts: float) -> float:
        """Pick the cruise or displacement fuel rate based on STW."""
        return (self.cruise_fuel_gph if stw_kts > self.mode_threshold_kts
                else self.displacement_fuel_gph)

    def fuel_rate_for_stw_array(self, stw_kts) -> np.ndarray:
        """Vectorized fuel-rate lookup."""
        stw = np.asarray(stw_kts, dtype=np.float64)
        return np.where(stw > self.mode_threshold_kts,
                        self.cruise_fuel_gph,
                        self.displacement_fuel_gph)


# ----- core math: scalars + vectors ----------------------------------------

def effective_sog_and_fuel(heading_deg: float,
                           stw_kts: float,
                           current_uv_ms: Tuple[float, float],
                           polar: VesselPolar = None
                           ) -> Tuple[float, float]:
    """Scalar form. Return (effective_sog_kts, fuel_rate_gph).

    See module docstring for the vector math.
    """
    polar = polar or VesselPolar.load()
    u_c, v_c = current_uv_ms
    rad = math.radians(heading_deg)
    stw_ms = stw_kts * KTS_TO_MS
    u_b = stw_ms * math.sin(rad) + u_c
    v_b = stw_ms * math.cos(rad) + v_c
    sog_kts = math.hypot(u_b, v_b) * MS_TO_KTS
    fuel_gph = polar.fuel_rate_for_stw(stw_kts)
    return sog_kts, fuel_gph


def effective_sog_and_fuel_vec(headings_deg,
                               stw_kts,
                               current_uv_ms,
                               polar: VesselPolar = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized form for A* edge cost.

    Args:
        headings_deg:   array-like[N] of headings, degrees true
        stw_kts:        scalar or array-like[N] of speed through water (kts)
        current_uv_ms:  ndarray[N, 2] or (2,) of (u_east, v_north) m/s
        polar:          VesselPolar (or default-loaded)

    Returns:
        sog_kts:  ndarray[N]
        fuel_gph: ndarray[N]

    All inputs are broadcast against each other. Fuel rate is selected per-
    element via the configured ``mode_threshold_kts``.
    """
    polar = polar or VesselPolar.load()
    headings = np.asarray(headings_deg, dtype=np.float64)
    stw = np.asarray(stw_kts, dtype=np.float64)
    uv = np.asarray(current_uv_ms, dtype=np.float64)
    if uv.ndim == 1:
        # (2,) -> apply same current to every heading.
        uv = np.broadcast_to(uv, (headings.shape[0], 2))
    elif uv.shape[-1] != 2:
        raise ValueError(f"current_uv_ms last axis must be 2; got {uv.shape}")

    rad = np.radians(headings)
    stw_ms = stw * KTS_TO_MS
    # Broadcast stw_ms (scalar OR (N,)) against the headings.
    u_b = stw_ms * np.sin(rad) + uv[..., 0]
    v_b = stw_ms * np.cos(rad) + uv[..., 1]
    sog_kts = np.hypot(u_b, v_b) * MS_TO_KTS
    # Fuel rate is per-element on stw; broadcast against headings if scalar.
    if stw.ndim == 0:
        fuel_gph = np.full_like(sog_kts,
                                polar.fuel_rate_for_stw(float(stw)))
    else:
        fuel_gph = polar.fuel_rate_for_stw_array(stw)
        # ensure same shape as sog_kts
        fuel_gph = np.broadcast_to(fuel_gph, sog_kts.shape).astype(np.float64,
                                                                    copy=True)
    return sog_kts, fuel_gph
