#!/usr/bin/env python3
"""Walk NOAA ENC cells, build a no-go multipolygon, write data/nogo.geojson.

Runnable as either:
  python3 scripts/build_nogo.py
  python3 -m tolly_router.build_nogo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# Allow running from a checkout without installing the package.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapely.geometry import mapping  # noqa: E402

from tolly_router import enc, nogo  # noqa: E402
from tolly_router.enc import DEFAULT_DEPTH_THRESHOLD_M  # noqa: E402

log = logging.getLogger("build_nogo")

DEFAULT_CHARTS = Path("/home/boat/Documents/Charts/ENC/US_REGION15")
DEFAULT_OUT = ROOT / "data" / "nogo.geojson"

# Puget Sound + San Juans scope: (minlon, minlat, maxlon, maxlat).
SCOPE_BBOX = (-124.5, 47.0, -122.0, 49.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--charts", type=Path, default=DEFAULT_CHARTS,
                    help="Root directory of NOAA ENC cells.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output GeoJSON path.")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MINLAT","MAXLON","MAXLAT"),
                    default=SCOPE_BBOX,
                    help="Scope bbox in WGS84 lon/lat.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N in-scope cells (debug).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    t0 = time.time()
    bbox = tuple(args.bbox)
    log.info("Scanning %s for ENC cells intersecting %s", args.charts, bbox)
    cells = enc.cells_in_bbox(args.charts, bbox)
    log.info("%d cells in scope", len(cells))
    if args.limit:
        cells = cells[: args.limit]
        log.info("Limiting to %d cells", len(cells))

    if not cells:
        log.error("No cells matched the scope; aborting.")
        return 2

    all_hits: list[enc.CellHit] = []
    per_layer: dict[str, int] = defaultdict(int)
    failures = 0

    for i, cell in enumerate(cells, 1):
        try:
            cell_hits = list(enc.extract_cell(cell, clip_bbox_wgs84=bbox))
        except Exception as exc:
            failures += 1
            log.warning("Cell %s failed cleanly: %s", cell.name, exc)
            continue
        if not cell_hits:
            log.debug("Cell %s contributed no features", cell.name)
            continue
        for hit in cell_hits:
            per_layer[hit.layer_name] += len(hit.geometries)
        all_hits.extend(cell_hits)
        log.info("[%d/%d] %s -> %s features across %d layers",
                 i, len(cells), cell.name,
                 sum(len(h.geometries) for h in cell_hits),
                 len(cell_hits))

    log.info("Extraction done in %.1fs. Per-layer feature counts:", time.time() - t0)
    for k in sorted(per_layer):
        log.info("  %s: %d", k, per_layer[k])
    if failures:
        log.warning("%d cells failed to parse and were skipped", failures)

    log.info("Building union no-go multipolygon (this may take a minute)...")
    t1 = time.time()
    result = nogo.build_nogo(all_hits)
    log.info("Union done in %.1fs", time.time() - t1)

    g_wgs84 = result.geometry_wgs84
    g_utm = result.geometry
    area_km2 = g_utm.area / 1e6
    bb = g_wgs84.bounds
    log.info("Result: type=%s area=%.1f km^2 bbox=(%.4f, %.4f, %.4f, %.4f)",
             g_wgs84.geom_type, area_km2, *bb)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feature = {
        "type": "Feature",
        "properties": {
            "phase": 1,
            "scope_bbox": list(bbox),
            "cells_in_scope": len(cells),
            "cells_contributing": result.cell_count,
            "cells_failed": failures,
            "layer_counts": result.layer_counts,
            "area_km2": area_km2,
            "min_navigable_depth_m": DEFAULT_DEPTH_THRESHOLD_M,
            "buffer_m": {
                "UWTROC": 50,
                "OBSTRN": 30,
                "WRECKS": 75,
            },
            "source_crs": "EPSG:4326 (via UTM 10N for buffering)",
        },
        "geometry": mapping(g_wgs84),
    }
    geojson = {"type": "FeatureCollection", "features": [feature]}
    with args.out.open("w") as f:
        json.dump(geojson, f)
    log.info("Wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    log.info("Total: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
