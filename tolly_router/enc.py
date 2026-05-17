"""S-57 ENC ingestion helpers.

Walks a NOAA ENC directory tree, filters cells whose footprint intersects a
WGS84 bounding box, and yields per-cell layer features as Shapely geometries
already reprojected to a target CRS (UTM 10N by default).

Per /home/boat/MARINE_ROUTING_PLAN.md §4b, we care about:
  LNDARE, DEPARE, DRGARE, UWTROC, OBSTRN, WRECKS, RESARE, CTNARE
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from osgeo import ogr, osr
from shapely import wkb as shapely_wkb
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .coords import UTM10N_EPSG, WGS84_EPSG

log = logging.getLogger(__name__)

ogr.UseExceptions()
# S-57 driver: skip the catalog files, force loading of all layers.
# Setting via env so it persists across Open() calls in this process.
os.environ.setdefault("OGR_S57_OPTIONS",
                      "RETURN_PRIMITIVES=OFF,RETURN_LINKAGES=OFF,LNAM_REFS=OFF,UPDATES=APPLY")

DEFAULT_DEPTH_THRESHOLD_M = 2.44

# Layer extraction rules.  Buffer is meters, applied in the UTM CRS.
@dataclass(frozen=True)
class LayerRule:
    name: str
    buffer_m: float = 0.0
    # Optional predicate on the OGR feature; return True to keep.
    predicate: Optional[callable] = field(default=None, compare=False)


def _depth_field_predicate(feat: "ogr.Feature", *, field: str,
                           min_depth_m: float, default_block: bool) -> bool:
    """Keep (i.e. block) a polygon when its ``field`` depth value is below
    ``min_depth_m``.  When the field is missing, return ``default_block`` —
    callers choose conservative-block vs trust-as-navigable per layer."""
    idx = feat.GetFieldIndex(field)
    if idx < 0 or not feat.IsFieldSet(idx):
        return default_block
    val = feat.GetFieldAsDouble(idx)
    return val < min_depth_m


def _depare_predicate(feat: "ogr.Feature",
                      min_depth_m: float = DEFAULT_DEPTH_THRESHOLD_M) -> bool:
    """Keep DEPARE polygons whose DRVAL2 (max depth) is below the minimum
    navigable depth.  Missing DRVAL2 → conservatively block."""
    return _depth_field_predicate(feat, field="DRVAL2",
                                  min_depth_m=min_depth_m,
                                  default_block=True)


# S-57 CATREA codes worth treating as a hard no-go.  Skipped: 23 (no-wake),
# None (unclassified — typically the giant VTS / TSS advisory zones that
# swallow open water like Rosario Strait).  RESARE is officially a *soft*
# penalty per /home/boat/MARINE_ROUTING_PLAN.md §4b; here we keep only the
# physically-restricted subset.
_RESARE_HARDBLOCK_CATEGORIES = {
    "4",   # nature reserve
    "5",   # bird sanctuary
    "8",   # degaussing range
    "9",   # military area
    "10",  # historic wreck area
    "18",  # dredging area
    "19",  # fish sanctuary
    "20",  # ecological reserve
}


def _resare_predicate(feat: "ogr.Feature") -> bool:
    idx = feat.GetFieldIndex("CATREA")
    if idx < 0 or not feat.IsFieldSet(idx):
        return False
    raw = feat.GetField(idx)
    if raw is None:
        return False
    cats = raw if isinstance(raw, list) else [raw]
    return any(str(c) in _RESARE_HARDBLOCK_CATEGORIES for c in cats)


# CTNARE is intentionally excluded.  /home/boat/MARINE_ROUTING_PLAN.md §4b
# lists it as a *soft* penalty, and inspection of REGION 15 shows CTNARE
# features are advisory (TSS lanes, "weak current", "submerged ops",
# chart-omission notes) that span entire straits — including them as hard
# no-go would render most of Puget Sound un-routable.

def _drgare_predicate(feat: "ogr.Feature",
                      min_depth_m: float = DEFAULT_DEPTH_THRESHOLD_M) -> bool:
    """Keep (block) a DRGARE polygon only when its minimum depth (DRVAL1) is
    set AND is shallower than the vessel's minimum.

    Critical difference from `_depare_predicate`: missing depth values on a
    *dredged* area mean "we don't know the maintained depth from this chart,"
    not "this is shallow water."  NOAA Region 15 cells commonly leave DRVAL1/
    DRVAL2 unset on DRGARE features — including the Swinomish Channel cells.
    Treating those as no-go (the old DEPARE-style rule) wrongly blocks
    maintained channels and disconnects whole basins.  The safer default
    here is "trust that a charted dredged area is navigable for our draft."
    """
    return _depth_field_predicate(feat, field="DRVAL1",
                                  min_depth_m=min_depth_m,
                                  default_block=False)


DEFAULT_RULES: tuple[LayerRule, ...] = (
    LayerRule("LNDARE", buffer_m=0.0),
    LayerRule("DEPARE", buffer_m=0.0, predicate=_depare_predicate),
    LayerRule("DRGARE", buffer_m=0.0, predicate=_drgare_predicate),
    LayerRule("UWTROC", buffer_m=50.0),
    LayerRule("OBSTRN", buffer_m=30.0),
    LayerRule("WRECKS", buffer_m=75.0),
    LayerRule("RESARE", buffer_m=0.0, predicate=_resare_predicate),
)


@dataclass
class CellHit:
    """One layer's features pulled from one ENC cell, in the target CRS."""
    cell_path: Path
    layer_name: str
    geometries: list[BaseGeometry]
    buffer_m: float


# NOAA ENC cell-naming convention: the digit after the `US` prefix is the
# usage band — 1 overview, 2 general, 3 coastal, 4 approach, 5 harbour.
# Higher numbers = larger scale = more detail.  For routing we want at least
# coastal-and-finer; overview/general charts often carry continent-scale
# polygons (caution areas, generalised coastlines) that swallow everything
# at our scope.
DEFAULT_USAGE_BANDS: tuple[int, ...] = (3, 4, 5)


def cell_usage_band(cell_path: Path) -> Optional[int]:
    """Pull the usage-band digit from a cell name like `US4WA11M.000`.
    Returns None if the cell doesn't match the convention."""
    stem = cell_path.stem  # e.g. "US4WA11M"
    if len(stem) < 3 or not stem.startswith("US"):
        return None
    ch = stem[2]
    if not ch.isdigit():
        return None
    return int(ch)


def find_cells(root: Path,
               usage_bands: Iterable[int] = DEFAULT_USAGE_BANDS) -> list[Path]:
    """Return every `*.000` ENC base cell under `root` whose NOAA usage band
    is in `usage_bands`."""
    bands = set(usage_bands)
    out = []
    for path in sorted(root.rglob("*.000")):
        band = cell_usage_band(path)
        if band is None or band in bands:
            out.append(path)
    return out


def cell_extent_wgs84(cell_path: Path) -> Optional[tuple[float, float, float, float]]:
    """Return (minlon, minlat, maxlon, maxlat) for the cell, or None if it
    can't be read.  Prefers M_COVR; falls back to any layer with features."""
    ds = ogr.Open(str(cell_path))

    for cand in ("M_COVR", "DEPARE", "LNDARE"):
        lyr = ds.GetLayerByName(cand)
        if lyr is None:
            continue
        if lyr.GetFeatureCount() == 0:
            continue
        minx, maxx, miny, maxy = lyr.GetExtent()
        return (minx, miny, maxx, maxy)
    return None


def cells_in_bbox(root: Path,
                  bbox_wgs84: tuple[float, float, float, float]) -> list[Path]:
    """Filter cells to those whose extent intersects `bbox_wgs84`
    (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    scope = box(minlon, minlat, maxlon, maxlat)
    out: list[Path] = []
    for cell in find_cells(root):
        ext = cell_extent_wgs84(cell)
        if ext is None:
            log.info("Skipping unreadable cell %s", cell.name)
            continue
        cmin_lon, cmin_lat, cmax_lon, cmax_lat = ext
        cell_box = box(cmin_lon, cmin_lat, cmax_lon, cmax_lat)
        if cell_box.intersects(scope):
            out.append(cell)
    return out


def _make_transform(src_epsg: int, dst_epsg: int) -> osr.CoordinateTransformation:
    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src, dst)


def extract_cell(cell_path: Path,
                 rules: Iterable[LayerRule] = DEFAULT_RULES,
                 dst_epsg: int = UTM10N_EPSG,
                 clip_bbox_wgs84: Optional[tuple[float, float, float, float]] = None
                 ) -> Iterator[CellHit]:
    """Open one ENC cell, yield CellHit per requested layer.  Missing layers
    are skipped silently (some cells legitimately don't have every layer).

    If `clip_bbox_wgs84` is provided as (minlon, minlat, maxlon, maxlat),
    each feature's WGS84 geometry is intersected with the bbox before being
    reprojected.  This keeps the union from dragging in continental-scale
    land polygons that overview cells carry.
    """
    ds = ogr.Open(str(cell_path))

    transform = _make_transform(WGS84_EPSG, dst_epsg)
    clip_geom = None
    if clip_bbox_wgs84 is not None:
        minlon, minlat, maxlon, maxlat = clip_bbox_wgs84
        clip_geom = box(minlon, minlat, maxlon, maxlat)

    for rule in rules:
        lyr = ds.GetLayerByName(rule.name)
        if lyr is None:
            continue
        geoms: list[BaseGeometry] = []
        lyr.ResetReading()
        for feat in lyr:
            if feat is None:
                continue
            if rule.predicate is not None and not rule.predicate(feat):
                continue
            ogr_geom = feat.GetGeometryRef()
            if ogr_geom is None or ogr_geom.IsEmpty():
                continue
            # Read in WGS84 first; clip then reproject.  This is cheaper than
            # reprojecting million-vertex Alaska just to throw it away.
            try:
                shp = shapely_wkb.loads(bytes(ogr_geom.ExportToWkb()))
            except Exception as exc:
                log.debug("WKB->Shapely failed on %s/%s: %s",
                          cell_path.name, rule.name, exc)
                continue
            if shp.is_empty:
                continue

            if clip_geom is not None:
                if not shp.intersects(clip_geom):
                    continue
                # Polygon-vs-bbox intersection; points stay as-is (already
                # passed the intersects test).
                if shp.geom_type in ("Polygon", "MultiPolygon",
                                      "GeometryCollection"):
                    try:
                        shp = shp.intersection(clip_geom)
                    except Exception:
                        # Bad ENC geometry; one buffer(0) repair attempt.
                        shp = shp.buffer(0).intersection(clip_geom)
                if shp.is_empty:
                    continue

            # Reproject the (possibly clipped) geometry to the target CRS.
            ogr_clipped = ogr.CreateGeometryFromWkb(shp.wkb)
            if ogr_clipped is None:
                continue
            ogr_clipped.Transform(transform)
            try:
                shp = shapely_wkb.loads(bytes(ogr_clipped.ExportToWkb()))
            except Exception as exc:
                log.debug("WKB->Shapely (post-transform) failed on %s/%s: %s",
                          cell_path.name, rule.name, exc)
                continue
            if shp.is_empty:
                continue
            geoms.append(shp)
        if geoms:
            yield CellHit(cell_path=cell_path,
                          layer_name=rule.name,
                          geometries=geoms,
                          buffer_m=rule.buffer_m)
