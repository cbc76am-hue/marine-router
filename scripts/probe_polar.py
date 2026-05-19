#!/usr/bin/env python3
"""Probe the vessel polar / fuel model with a single sample.

Usage::

    python3 scripts/probe_polar.py --heading 90 --stw 14 --current 0,0
    python3 scripts/probe_polar.py --heading 270 --stw 14 --current 1.0,0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import polar  # noqa: E402


def _parse_uv(s: str):
    a, b = s.split(",")
    return float(a), float(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heading", type=float, required=True,
                    help="Boat heading, degrees true (0=N, 90=E).")
    ap.add_argument("--stw", type=float, required=True,
                    help="Speed through water, knots.")
    ap.add_argument("--current", type=_parse_uv, required=True,
                    help="Current vector as u_east,v_north in m/s "
                         "(e.g. '0,0' or '0.5,-0.25').")
    args = ap.parse_args(argv)

    p = polar.VesselPolar.load()
    sog_kts, fuel_gph = polar.effective_sog_and_fuel(
        heading_deg=args.heading,
        stw_kts=args.stw,
        current_uv_ms=args.current,
        polar=p,
    )
    print(f"# polar config: {p}")
    print(f"heading_deg={args.heading}  stw_kts={args.stw}  current_uv_ms={args.current}")
    print(f"effective_sog_kts = {sog_kts:.3f}")
    print(f"fuel_rate_gph     = {fuel_gph:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
