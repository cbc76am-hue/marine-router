"""Build the union no-go multipolygon from ENC layer hits.

Two-step pipeline:
  1. buffer per-layer geometries (in UTM 10N meters) and accumulate
  2. unary_union to merge overlapping no-go zones into one MultiPolygon
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .coords import to_wgs84
from .enc import CellHit

log = logging.getLogger(__name__)


@dataclass
class NogoResult:
    geometry: BaseGeometry          # in source CRS (UTM 10N)
    geometry_wgs84: BaseGeometry    # reprojected to EPSG:4326
    layer_counts: dict[str, int]
    cell_count: int


def _coerce_polygonal(g: BaseGeometry) -> list[Polygon]:
    """Reduce any geometry to a list of polygons.  Points and lines have
    already been buffered upstream; anything still non-polygonal at this stage
    is dropped."""
    if g.is_empty:
        return []
    gt = g.geom_type
    if gt == "Polygon":
        return [g]
    if gt == "MultiPolygon":
        return list(g.geoms)
    if gt == "GeometryCollection":
        out: list[Polygon] = []
        for sub in g.geoms:
            out.extend(_coerce_polygonal(sub))
        return out
    return []


def build_nogo(cell_hits: Iterable[CellHit]) -> NogoResult:
    layer_polys: dict[str, list[Polygon]] = defaultdict(list)
    layer_counts: dict[str, int] = defaultdict(int)
    cells_seen: set[str] = set()

    for hit in cell_hits:
        cells_seen.add(str(hit.cell_path))
        for geom in hit.geometries:
            layer_counts[hit.layer_name] += 1
            if hit.buffer_m > 0:
                buffered = geom.buffer(hit.buffer_m)
            else:
                # Polygons stay polygons; bare points/lines with zero buffer
                # are not useful as no-go shapes so they get dropped.
                if geom.geom_type in ("Point", "LineString",
                                       "MultiPoint", "MultiLineString"):
                    continue
                buffered = geom
            for poly in _coerce_polygonal(buffered):
                if poly.is_valid:
                    layer_polys[hit.layer_name].append(poly)
                else:
                    fixed = poly.buffer(0)
                    layer_polys[hit.layer_name].extend(_coerce_polygonal(fixed))

    # Union per layer first to keep the final union manageable, then
    # union of unions.
    per_layer_union: list[BaseGeometry] = []
    for layer, polys in layer_polys.items():
        log.info("Unioning %d polygons from %s", len(polys), layer)
        per_layer_union.append(unary_union(polys))

    if per_layer_union:
        merged = unary_union(per_layer_union)
    else:
        merged = MultiPolygon()

    merged_wgs84 = to_wgs84(merged)
    # Per-vertex reprojection can introduce micro self-touches; clean up so
    # the GeoJSON parses as a valid OGC polygon set.
    if not merged_wgs84.is_valid:
        log.info("Reprojected geometry not OGC-valid; running make_valid()")
        merged_wgs84 = make_valid(merged_wgs84)
        # make_valid can return non-polygonal mixes; drop the lines/points.
        if merged_wgs84.geom_type == "GeometryCollection":
            polys: list[Polygon] = []
            for sub in merged_wgs84.geoms:
                polys.extend(_coerce_polygonal(sub))
            merged_wgs84 = unary_union(polys) if polys else MultiPolygon()

    return NogoResult(
        geometry=merged,
        geometry_wgs84=merged_wgs84,
        layer_counts=dict(layer_counts),
        cell_count=len(cells_seen),
    )
