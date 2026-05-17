"""Rasterize the no-go MultiPolygon into a grid-A* navigability + cost surface.

Input: a no-go MultiPolygon in WGS84 (built by the ENC ingest step at a given
depth threshold).  Output: a numpy bool mask + float32 cost array, plus the
affine transform that maps grid (col, row) to UTM Zone 10N (easting, northing).

The depth threshold itself is not used here — it's only carried through as
metadata so a multi-raster lookup can pick the right pre-built graph.  See
/home/boat/MARINE_ROUTING_PLAN.md §13.1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from .coords import UTM10N_EPSG, WGS84_EPSG, to_utm10n, wgs84_to_utm10n_transformer

log = logging.getLogger(__name__)

# Reserved cost range is documented in /home/boat/MARINE_ROUTING_PLAN.md;
# currently only 1.0, 1.5, and inf are emitted.  2.0 is reserved for soft
# penalty zones (CTNARE etc.).
COST_OPEN = np.float32(1.0)
COST_EDGE = np.float32(1.5)
COST_SOFT = np.float32(2.0)  # reserved, not emitted
COST_BLOCKED = np.float32(np.inf)

EDGE_DISTANCE_CELLS = 2  # cells from no-go that count as "edge-hugging"


@dataclass
class RasterResult:
    nogo: np.ndarray              # bool[H, W]
    cost: np.ndarray              # float32[H, W]
    transform: Affine             # grid (col, row) -> UTM (e, n)
    crs: str                      # "EPSG:32610"
    resolution_m: float
    depth_threshold_m: float
    bbox_wgs84: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    bbox_utm: tuple[float, float, float, float]    # (e_min, n_min, e_max, n_max)
    built_at: str


# --------------------------------------------------------------------------
# Geometry handling
# --------------------------------------------------------------------------

def load_nogo_geojson(path: Path) -> tuple[BaseGeometry, tuple[float, float, float, float]]:
    """Load the Phase 1 GeoJSON and return (geometry_wgs84, scope_bbox_wgs84).

    The Phase 1 output is a FeatureCollection of one Feature whose properties
    include ``scope_bbox`` — we use that as the canonical scope, falling back
    to the geometry's own bounds if missing.
    """
    with path.open() as f:
        gj = json.load(f)

    if gj.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: expected FeatureCollection, got {gj.get('type')!r}")
    features = gj.get("features") or []
    if not features:
        raise ValueError(f"{path}: no features")
    feat = features[0]
    geom = shape(feat["geometry"])

    props = feat.get("properties") or {}
    scope = props.get("scope_bbox")
    if scope and len(scope) == 4:
        bbox_wgs84 = tuple(float(v) for v in scope)
    else:
        bbox_wgs84 = tuple(float(v) for v in geom.bounds)

    return geom, bbox_wgs84


def reproject_wgs84_to_utm10n(geom: BaseGeometry) -> BaseGeometry:
    """Project a WGS84 geometry into UTM Zone 10N (EPSG:32610)."""
    return to_utm10n(geom)


def utm_bbox_for_scope(bbox_wgs84: tuple[float, float, float, float]
                       ) -> tuple[float, float, float, float]:
    """Project the four corners of a WGS84 bbox into UTM 10N, then take the
    envelope.  Going corner-by-corner under-counts the curvature of the
    meridians, so we sample the edges as well — small overshoot is fine, the
    rasterize step only burns inside the polygon.
    """
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    transformer = wgs84_to_utm10n_transformer()
    # Sample 11 points per edge for the lon/lat envelope.
    n = 11
    lons = np.concatenate([
        np.full(n, minlon),
        np.full(n, maxlon),
        np.linspace(minlon, maxlon, n),
        np.linspace(minlon, maxlon, n),
    ])
    lats = np.concatenate([
        np.linspace(minlat, maxlat, n),
        np.linspace(minlat, maxlat, n),
        np.full(n, minlat),
        np.full(n, maxlat),
    ])
    es, ns = transformer.transform(lons, lats)
    return (float(np.min(es)), float(np.min(ns)),
            float(np.max(es)), float(np.max(ns)))


# --------------------------------------------------------------------------
# Rasterization + cost surface
# --------------------------------------------------------------------------

def grid_dimensions(bbox_utm: tuple[float, float, float, float],
                    resolution_m: float
                    ) -> tuple[int, int, tuple[float, float, float, float]]:
    """Snap a UTM bbox out to whole-cell boundaries at ``resolution_m`` and
    return (height, width, snapped_bbox)."""
    e_min, n_min, e_max, n_max = bbox_utm
    # Snap minimums down and maximums up so the entire requested bbox is
    # covered by complete cells.
    e_min_s = np.floor(e_min / resolution_m) * resolution_m
    n_min_s = np.floor(n_min / resolution_m) * resolution_m
    e_max_s = np.ceil(e_max / resolution_m) * resolution_m
    n_max_s = np.ceil(n_max / resolution_m) * resolution_m
    width = int(round((e_max_s - e_min_s) / resolution_m))
    height = int(round((n_max_s - n_min_s) / resolution_m))
    return height, width, (float(e_min_s), float(n_min_s),
                            float(e_max_s), float(n_max_s))


def build_transform(bbox_utm_snapped: tuple[float, float, float, float],
                    resolution_m: float) -> Affine:
    """Build a north-up Affine: row 0 is the top of the bbox, column 0 the
    left edge.  Pixel size is +resolution east, -resolution north."""
    e_min, _n_min, _e_max, n_max = bbox_utm_snapped
    return Affine(resolution_m, 0.0, e_min,
                  0.0, -resolution_m, n_max)


def rasterize_nogo(geom_utm: BaseGeometry,
                   transform: Affine,
                   shape_hw: tuple[int, int]) -> np.ndarray:
    """Burn the no-go geometry into a bool[H, W] mask. True = blocked."""
    out = rasterio.features.rasterize(
        [(mapping(geom_utm), 1)],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,   # any cell the polygon touches is blocked
        dtype="uint8",
    )
    return out.astype(bool)


def build_cost_surface(nogo: np.ndarray,
                       edge_iterations: int = EDGE_DISTANCE_CELLS) -> np.ndarray:
    """Return a float32[H, W] cost surface:
       * 1.0 in open water,
       * 1.5 within ``edge_iterations`` cells of a no-go,
       * inf on no-go cells.
    """
    from scipy.ndimage import binary_dilation

    cost = np.full(nogo.shape, COST_OPEN, dtype=np.float32)

    # 3x3 structuring element → 8-connected dilation (diagonals included).
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(nogo, structure=struct,
                              iterations=edge_iterations)
    edge_band = dilated & ~nogo
    cost[edge_band] = COST_EDGE
    cost[nogo] = COST_BLOCKED
    return cost


# --------------------------------------------------------------------------
# Top-level pipeline
# --------------------------------------------------------------------------

def build_raster(nogo_geojson_path: Path,
                 depth_threshold_m: float,
                 resolution_m: float = 50.0,
                 bbox_wgs84_override: Optional[tuple[float, float, float, float]] = None,
                 ) -> RasterResult:
    """End-to-end: GeoJSON in → RasterResult out.

    ``bbox_wgs84_override`` lets callers force a scope independent of what's
    in the GeoJSON properties (rarely useful; v1 just reuses Phase 1's scope).
    """
    log.info("Loading no-go geometry from %s", nogo_geojson_path)
    geom_wgs84, scope_bbox = load_nogo_geojson(nogo_geojson_path)
    if bbox_wgs84_override is not None:
        scope_bbox = tuple(float(v) for v in bbox_wgs84_override)
    log.info("Scope bbox (WGS84): %s", scope_bbox)
    log.info("No-go geometry: type=%s parts=%s",
             geom_wgs84.geom_type,
             len(geom_wgs84.geoms) if hasattr(geom_wgs84, "geoms") else 1)

    log.info("Reprojecting no-go geometry to UTM 10N (EPSG:%d)", UTM10N_EPSG)
    geom_utm = reproject_wgs84_to_utm10n(geom_wgs84)

    bbox_utm = utm_bbox_for_scope(scope_bbox)
    height, width, bbox_utm_snapped = grid_dimensions(bbox_utm, resolution_m)
    log.info("UTM bbox %s → snapped %s → grid %d x %d cells @ %.0f m",
             bbox_utm, bbox_utm_snapped, height, width, resolution_m)

    transform = build_transform(bbox_utm_snapped, resolution_m)
    log.info("Affine transform: %s", transform)

    log.info("Rasterizing no-go polygons...")
    nogo = rasterize_nogo(geom_utm, transform, (height, width))
    n_blocked = int(nogo.sum())
    log.info("nogo: %d / %d cells blocked (%.1f%%)",
             n_blocked, nogo.size, 100.0 * n_blocked / nogo.size)

    log.info("Building cost surface...")
    cost = build_cost_surface(nogo)
    n_edge = int(np.count_nonzero(cost == COST_EDGE))
    log.info("cost: %d edge-hug cells (%.1f%%), %d navigable",
             n_edge, 100.0 * n_edge / nogo.size, int((~nogo).sum()))

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return RasterResult(
        nogo=nogo,
        cost=cost,
        transform=transform,
        crs=f"EPSG:{UTM10N_EPSG}",
        resolution_m=float(resolution_m),
        depth_threshold_m=float(depth_threshold_m),
        bbox_wgs84=tuple(float(v) for v in scope_bbox),
        bbox_utm=bbox_utm_snapped,
        built_at=built_at,
    )


def save_npz(result: RasterResult, out_path: Path) -> Path:
    """Serialize a RasterResult to a compressed .npz file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        nogo=result.nogo,
        cost=result.cost,
        transform=np.asarray(result.transform, dtype=np.float64),
        crs=np.asarray(result.crs),
        resolution_m=np.float64(result.resolution_m),
        depth_threshold_m=np.float64(result.depth_threshold_m),
        bbox_wgs84=np.asarray(result.bbox_wgs84, dtype=np.float64),
        bbox_utm=np.asarray(result.bbox_utm, dtype=np.float64),
        built_at=np.asarray(result.built_at),
        cells_total=np.int64(result.nogo.size),
        cells_navigable=np.int64(int((~result.nogo).sum())),
    )
    return out_path


def threshold_filename(depth_threshold_m: float) -> str:
    """`2.44 m` → `graph_d244.npz` (cm as an int)."""
    cm = int(round(depth_threshold_m * 100))
    return f"graph_d{cm}.npz"
