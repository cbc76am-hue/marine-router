#!/usr/bin/env python3
"""Offline routing CLI: run a single route or the acceptance suite against
``data/graph.npz``.

Usage::

    python3 scripts/route.py --start 48.4045,-122.5062 --end 48.5363,-123.0168
    python3 scripts/route.py --suite                            # all 6 cases
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router.routing import RouteResult, Router  # noqa: E402

DEFAULT_GRAPH = ROOT / "data" / "graph.npz"


# --------------------------------------------------------------------------
# Pretty-printer
# --------------------------------------------------------------------------

def _print_result(idx: int, label: str, result: RouteResult,
                  start: Tuple[float, float],
                  end: Tuple[float, float]) -> None:
    print("=" * 78)
    print(f"Test {idx}: {label}")
    print(f"  start={start}  end={end}")
    if result.ok:
        print(f"  result: OK")
        print(f"  distance_nm:   {result.distance_nm:.2f}")
        print(f"  waypoints:     {len(result.waypoints)}")
        print(f"  hazards_near:  {result.hazards_near}")
        print(f"  runtime_s:     {result.runtime_s:.3f}")
        print(f"  nodes_expanded:{result.nodes_expanded}")
        if result.warnings:
            for w in result.warnings:
                print(f"  warning:       {w}")
        else:
            print(f"  warnings:      (none)")
        print(f"  waypoint list:")
        for i, wp in enumerate(result.waypoints):
            name = f"  [{wp.name}]" if wp.name else ""
            print(f"    {i:3d}: ({wp.lat:.5f}, {wp.lon:.5f}){name}")
    else:
        print(f"  result: ERROR  ({result.error!r})")
        print(f"  runtime_s:     {result.runtime_s:.3f}")
        if result.warnings:
            for w in result.warnings:
                print(f"  warning:       {w}")


# --------------------------------------------------------------------------
# Test suite
# --------------------------------------------------------------------------

# Each suite entry: (label, start, end, expected_ok, optional_error_substring).
# expected_ok=True  → route must return ok=True
# expected_ok=False → route must return ok=False AND the error must contain
#                     the given substring (or any error if substring is None)
SUITE: List[Tuple[str, Tuple[float, float], Tuple[float, float],
                  bool, Optional[str]]] = [
    # Known limitation: Swinomish Channel is too narrow for the 50 m raster
    # (LNDARE hems it to <50 m clear in places), so the start basin is
    # disconnected from the destination basin.  Expected outcome until we
    # add adaptive resolution in known narrow passages.
    ("Shelter Bay -> Friday Harbor (Swinomish + Rosario)",
     (48.4045, -122.5062), (48.5363, -123.0168),
     False, "disconnected"),
    ("Mid Rosario -> Anacortes",
     (48.6300, -122.7850), (48.5167, -122.6131),
     True, None),
    ("Mid Puget Sound -> Bremerton",
     (47.7000, -122.4500), (47.5673, -122.6326),
     True, None),
    ("Downtown Seattle pier -> Bainbridge (expects start-nudge)",
     (47.6062, -122.3321), (47.6230, -122.5110),
     True, None),
    ("Mid Rosario -> Coupeville/Whidbey (expects destination-on-land)",
     (48.6300, -122.7850), (48.1944, -122.6440),
     False, "destination is not on water"),
    ("Outside scope (BC) -> Friday Harbor (expects out-of-coverage)",
     (50.0000, -123.0000), (48.5363, -123.0168),
     False, "outside routing coverage area"),
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_latlon(s: str) -> Tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH,
                    help="Path to graph.npz")
    ap.add_argument("--start", type=_parse_latlon,
                    help="Start lat,lon (decimal)")
    ap.add_argument("--end", type=_parse_latlon,
                    help="End lat,lon (decimal)")
    ap.add_argument("--suite", action="store_true",
                    help="Run all 6 acceptance cases.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.graph.is_file():
        print(f"graph not found: {args.graph}", file=sys.stderr)
        return 2

    t_load = time.perf_counter()
    router = Router.from_npz(args.graph)
    print(f"# graph loaded in {time.perf_counter() - t_load:.2f}s "
          f"(shape={router.graph.shape}, "
          f"navigable={router.graph.cells_navigable} cells)")

    if args.suite:
        failed: List[str] = []
        for idx, (label, start, end, exp_ok, exp_err) in enumerate(SUITE, 1):
            result = router.route(start, end)
            _print_result(idx, label, result, start, end)
            # Check the expectation
            if result.ok != exp_ok:
                failed.append(
                    f"#{idx} '{label}': expected ok={exp_ok}, got ok={result.ok}"
                )
            elif (not exp_ok) and exp_err is not None:
                if exp_err.lower() not in (result.error or "").lower():
                    failed.append(
                        f"#{idx} '{label}': error should contain "
                        f"'{exp_err}', got '{result.error}'"
                    )
        if failed:
            print()
            print(f"FAIL: {len(failed)} of {len(SUITE)} acceptance cases failed:")
            for msg in failed:
                print(f"  - {msg}")
            return 1
        print()
        print(f"PASS: all {len(SUITE)} acceptance cases met expectations.")
        return 0

    if not (args.start and args.end):
        print("error: pass --start LAT,LON --end LAT,LON or --suite",
              file=sys.stderr)
        return 2

    result = router.route(args.start, args.end)
    _print_result(0, "ad-hoc", result, args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
