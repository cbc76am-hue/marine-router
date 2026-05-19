"""Coordinate transform helpers for Phase 3 routing.

The routing graph lives on a north-up grid in UTM Zone 10N (EPSG:32610).  A
rasterio-style 6-parameter affine maps grid (col, row) → UTM (easting,
northing); inverting it (and chaining with a pyproj WGS84↔UTM transformer)
gets us between lat/lon and grid indices.

Phase 4 (the HTTP service) imports this module without pulling in A* so the
service layer can validate inputs cheaply.

Affine convention (matches what Phase 2 writes into ``graph.npz``)::

    e = a*col + b*row + c
    n = d*col + e_*row + f
    transform = (a, b, c, d, e_, f)              # rasterio 6-tuple
    transform = (a, b, c, d, e_, f, 0, 0, 1)     # the 9-element flat form

For a north-up grid built by Phase 2: ``a = +res``, ``e_ = -res``, ``b = d = 0``,
``c = e_min`` (left edge of column 0), ``f = n_max`` (top edge of row 0).
Pixel (row, col) covers UTM ``[c + a*col, c + a*(col+1)]`` east and
``[f + e_*(row+1), f + e_*row]`` north.  We use cell *centers* for round-trip
math so the WGS84 → grid → WGS84 roundtrip stays inside half a cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pyproj import Transformer
from shapely.ops import transform as shp_transform

WGS84_EPSG = 4326
UTM10N_EPSG = 32610


# --------------------------------------------------------------------------
# Affine helpers (numpy-only, no rasterio dependency at runtime)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GridAffine:
    """A flat-tuple, north-up affine.  All math is plain numpy.

    Stored as the 6 active rasterio coefficients (a, b, c, d, e, f).
    """
    a: float   # +res for north-up
    b: float
    c: float   # e_min (UTM east of left edge of col 0)
    d: float
    e: float   # -res for north-up
    f: float   # n_max (UTM north of top edge of row 0)

    @classmethod
    def from_array(cls, arr) -> "GridAffine":
        """Accept either the 6-tuple or 9-tuple flat form Phase 2 writes."""
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size >= 6:
            return cls(float(arr[0]), float(arr[1]), float(arr[2]),
                       float(arr[3]), float(arr[4]), float(arr[5]))
        raise ValueError(f"affine must have >= 6 elements, got {arr.size}")

    def cell_center_utm(self, row: float, col: float) -> Tuple[float, float]:
        """Return (easting, northing) at the *center* of cell (row, col)."""
        e = self.a * (col + 0.5) + self.b * (row + 0.5) + self.c
        n = self.d * (col + 0.5) + self.e * (row + 0.5) + self.f
        return e, n

    def utm_to_rowcol(self, easting: float, northing: float) -> Tuple[float, float]:
        """Inverse of the center-cell mapping; returns floats so callers can
        decide rounding (we always round-half-to-even via numpy.round for the
        integer cell)."""
        # Solve [[a b][d e]] * [col+0.5, row+0.5]^T = [e_-c, n_-f]^T
        a, b, d, e = self.a, self.b, self.d, self.e
        det = a * e - b * d
        if det == 0.0:
            raise ValueError("degenerate affine (det=0)")
        de = easting - self.c
        dn = northing - self.f
        col_h = (e * de - b * dn) / det
        row_h = (-d * de + a * dn) / det
        return row_h - 0.5, col_h - 0.5


# --------------------------------------------------------------------------
# pyproj-backed WGS84 ↔ UTM10N
# --------------------------------------------------------------------------

# Module-level transformers — they're thread-safe and we want to pay
# initialization cost (~ms) only once per process.
_WGS84_TO_UTM = Transformer.from_crs(WGS84_EPSG, UTM10N_EPSG, always_xy=True)
_UTM_TO_WGS84 = Transformer.from_crs(UTM10N_EPSG, WGS84_EPSG, always_xy=True)


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float]:
    """WGS84 (lat, lon) → UTM 10N (easting, northing) in meters."""
    e, n = _WGS84_TO_UTM.transform(lon, lat)  # always_xy: (lon, lat)
    return float(e), float(n)


def utm_to_latlon(easting: float, northing: float) -> Tuple[float, float]:
    """UTM 10N (easting, northing) → WGS84 (lat, lon)."""
    lon, lat = _UTM_TO_WGS84.transform(easting, northing)
    return float(lat), float(lon)


def _project_xy(transformer):
    def fn(xs, ys, zs=None):
        x2, y2 = transformer.transform(xs, ys)
        return (x2, y2) if zs is None else (x2, y2, zs)
    return fn


def to_utm10n(geom):
    """Reproject a Shapely WGS84 geometry into UTM 10N using the cached
    module-level transformer."""
    return shp_transform(_project_xy(_WGS84_TO_UTM), geom)


def to_wgs84(geom):
    """Reproject a Shapely UTM 10N geometry into WGS84 using the cached
    module-level transformer."""
    return shp_transform(_project_xy(_UTM_TO_WGS84), geom)


def wgs84_to_utm10n_transformer() -> Transformer:
    """Return the cached WGS84 → UTM 10N Transformer.  For batch numpy
    transforms where the Shapely path is overkill."""
    return _WGS84_TO_UTM


def utm10n_to_wgs84_transformer() -> Transformer:
    """Return the cached UTM 10N → WGS84 Transformer.  For batch numpy
    transforms (lon, lat ordering since always_xy=True)."""
    return _UTM_TO_WGS84


def latlon_to_grid(lat: float, lon: float, transform) -> Tuple[int, int]:
    """WGS84 (lat, lon) → integer (row, col) grid cell.

    Accepts the 6- or 9-tuple flat-form affine that Phase 2 writes into
    ``graph.npz``.
    """
    aff = transform if isinstance(transform, GridAffine) else GridAffine.from_array(transform)
    e, n = latlon_to_utm(lat, lon)
    row_h, col_h = aff.utm_to_rowcol(e, n)
    return int(round(row_h)), int(round(col_h))


def grid_to_latlon(row: int, col: int, transform) -> Tuple[float, float]:
    """Integer (row, col) → WGS84 (lat, lon) of cell center."""
    aff = transform if isinstance(transform, GridAffine) else GridAffine.from_array(transform)
    e, n = aff.cell_center_utm(row, col)
    return utm_to_latlon(e, n)


# --------------------------------------------------------------------------
# Vectorized batch helpers used by A* setup
# --------------------------------------------------------------------------

def utm_grid_for_bbox(transform, row0: int, col0: int,
                      n_rows: int, n_cols: int):
    """Return (E[n_rows, n_cols], N[n_rows, n_cols]) of cell-center UTM coords
    for a sub-window of the full grid starting at (row0, col0)."""
    aff = transform if isinstance(transform, GridAffine) else GridAffine.from_array(transform)
    cols = np.arange(col0, col0 + n_cols, dtype=np.float64)
    rows = np.arange(row0, row0 + n_rows, dtype=np.float64)
    # Keep the general form for robustness against off-axis affines.
    e_1d = aff.a * (cols + 0.5) + aff.c
    n_1d = aff.e * (rows + 0.5) + aff.f
    if aff.b != 0.0 or aff.d != 0.0:
        # General case: 2D combine.
        E = aff.a * (cols[None, :] + 0.5) + aff.b * (rows[:, None] + 0.5) + aff.c
        N = aff.d * (cols[None, :] + 0.5) + aff.e * (rows[:, None] + 0.5) + aff.f
        return E, N
    # Broadcast 1D vectors to (n_rows, n_cols).
    E = np.broadcast_to(e_1d[None, :], (n_rows, n_cols))
    N = np.broadcast_to(n_1d[:, None], (n_rows, n_cols))
    return E, N


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    R_NM = 3440.065  # mean earth radius in nm
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2.0 * R_NM * np.arcsin(np.sqrt(a)))
