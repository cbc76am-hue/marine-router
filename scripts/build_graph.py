#!/usr/bin/env python3
"""Rasterize the no-go MultiPolygon to a 50 m grid in UTM 10N, compute an A*
cost surface, write a compressed .npz under data/.

Runnable as either:
  python3 scripts/build_graph.py
  python3 -m tolly_router.build_graph

The threshold and the input GeoJSON are parametrized so a multi-threshold
build loop can iterate over (threshold, geojson) pairs without changing this
script.  See /home/boat/MARINE_ROUTING_PLAN.md §13.1.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# Allow running from a checkout without installing the package.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tolly_router import raster  # noqa: E402
from tolly_router.enc import DEFAULT_DEPTH_THRESHOLD_M  # noqa: E402

log = logging.getLogger("build_graph")

DEFAULT_NOGO = ROOT / "data" / "nogo.geojson"
DEFAULT_OUT_DIR = ROOT / "data"
DEFAULT_THRESHOLD_M = DEFAULT_DEPTH_THRESHOLD_M
DEFAULT_RESOLUTION_M = 50.0
# The default-threshold artifact also lives at this stable name so the
# service can pick up "the current graph" without knowing about the
# multi-threshold lookup logic.
DEFAULT_ALIAS = "graph.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nogo-geojson", type=Path, default=DEFAULT_NOGO,
                    help="Phase 1 no-go GeoJSON input.")
    ap.add_argument("--depth-threshold", type=float, default=DEFAULT_THRESHOLD_M,
                    metavar="M",
                    help="Min navigable depth in meters; metadata only, the "
                         "no-go GeoJSON is the actual depth filter.")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M,
                    metavar="M", help="Cell size in meters.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Directory to write graph_dNNN.npz into.")
    ap.add_argument("--alias", default=DEFAULT_ALIAS,
                    help=(
                        "Also write/copy the threshold-specific .npz to this "
                        "stable filename in --out-dir.  Use '' to skip."
                    ))
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    help="Override WGS84 scope; default is whatever Phase 1 "
                         "wrote into the GeoJSON properties.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.nogo_geojson.is_file():
        log.error("nogo geojson not found: %s", args.nogo_geojson)
        return 2

    t0 = time.time()
    result = raster.build_raster(
        nogo_geojson_path=args.nogo_geojson,
        depth_threshold_m=args.depth_threshold,
        resolution_m=args.resolution,
        bbox_wgs84_override=tuple(args.bbox) if args.bbox else None,
    )
    log.info("Build done in %.1fs", time.time() - t0)

    out_name = raster.threshold_filename(args.depth_threshold)
    out_path = args.out_dir / out_name
    raster.save_npz(result, out_path)
    sz_mb = out_path.stat().st_size / 1e6
    log.info("Wrote %s (%.1f MB)", out_path, sz_mb)

    if args.alias:
        alias_path = args.out_dir / args.alias
        if alias_path.resolve() != out_path.resolve():
            shutil.copyfile(out_path, alias_path)
            log.info("Aliased default graph at %s", alias_path)

    log.info(
        "Summary: shape=%s navigable=%d/%d (%.1f%%) bbox_utm=%s built_at=%s",
        result.nogo.shape,
        int((~result.nogo).sum()),
        result.nogo.size,
        100.0 * int((~result.nogo).sum()) / result.nogo.size,
        result.bbox_utm,
        result.built_at,
    )
    log.info("Total: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
