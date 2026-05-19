r"""Grid A* router on the navigability + cost surface.

Algorithm overview:

1. **Bbox prune.**  Build a sub-window of the cost array around the
   straight line from start to end, padded by 5 km (100 cells).  Everything
   else is left in the full grid as a memory cushion — we only allocate
   ``closed`` + ``g_score`` over the prune bbox so the hot loop touches
   ~10⁵-10⁶ cells instead of 17M.

2. **A\* heuristic.**  We precompute UTM (E, N) coordinates for every cell in
   the prune bbox once (vectorized), then the heuristic for a candidate cell
   is just ``hypot(E - E_end, N - N_end) / 50 m`` — Euclidean in UTM, scaled
   to cell-cost units.  UTM 10N distortion across our scope is <0.5 %, well
   below the admissibility wiggle room we have from the 1.0/1.5 cost
   distinction.  This is ~10× faster than per-step haversine.

3. **A\* main loop.**  Standard binary-heap A* in pure Python.  Hot loop is
   the bottleneck — we keep it as tight as possible by:
   * hoisting ``cost``, ``h``, ``came_from``, ``g_score`` arrays to locals;
   * unrolling the 8-neighbor offsets and per-neighbor base edge cost
     (1.0 cardinal, √2 diagonal) into module constants;
   * using a monotonic counter as the second heap key so ties don't fall back
     on numpy array comparison.

4. **Post-processing.**  Collect (row, col) path from the came-from
   dictionary, project the cell centers to UTM, run
   ``shapely.LineString(...).simplify(100.0)`` for Douglas-Peucker, project
   the simplified line back to WGS84.

Edge cases handled explicitly:
* start or end outside the grid bbox  → error.
* start cell is no-go  → soft: try a spiral search for a nearby navigable
  cell within 1 km; nudge.  Emit a warning so the caller knows.
* end cell is no-go  → error (refuse to plan onto land).
* great-circle distance > 100 nm  → error.
* A* exhausts the frontier without reaching the end  → error.
* path passes within 10 cells of the prune bbox edge  → warning.
"""
from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString

from .coords import (
    GridAffine,
    grid_to_latlon,
    haversine_nm,
    latlon_to_grid,
    latlon_to_utm,
    utm10n_to_wgs84_transformer,
    utm_grid_for_bbox,
    utm_to_latlon,
)
from .currents import CurrentField, get_default_field
from .polar import KTS_TO_MS, MS_TO_KTS, VesselPolar, effective_sog_and_fuel
from .raster import COST_EDGE

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

# Bbox prune: A* search is constrained to a rectangle around the straight-
# line start→end with this much padding on every side.  Sized to cover
# realistic Salish Sea routes where the actual path wraps around an
# island (e.g. HOME → Padilla Bay is 6 nm straight but ~80 nm sailable
# around the south end of Whidbey).  25 km pad accommodates Whidbey,
# Camano, and Bainbridge wrap-arounds; broader scope routes (Hood Canal
# bridge, BC border) may still feel the cap and surface a "route hugs
# prune-bbox edge" warning.
PRUNE_PAD_M = 25000.0
SIMPLIFY_TOL_M = 100.0
# Nudge radius — how far the spiral search will reach to find a navigable
# cell in the same connected basin as the other endpoint.  Sized for two
# realistic cases:
#   1. Named-harbor destinations (Friday Harbor, Anacortes) where the
#      pier coord lands just inside a dock at 50 m resolution; observed
#      offsets up to ~1 km.
#   2. Marina/slip START coords (e.g. Shelter Bay, Edmonds) where the
#      raster blocks the slip itself plus a narrow channel exit; the
#      nearest navigable cell can be 1-3 km away.  At 3 km we cover
#      virtually every marina in the Salish Sea while still refusing
#      to plan routes from genuinely land-locked points (mid-Whidbey,
#      mid-mainland) where the nudge would have to span >3 km.
# Every nudge >0 raises a clear warning the caller sees ("start nudged
# X m to nearest water at lat, lon"), so we don't silently move the
# user's intent.
START_NUDGE_RADIUS_M = 3000.0
START_NUDGE_PREFERRED_M = 500.0     # don't warn if we found water within this
END_NUDGE_RADIUS_M = 3000.0
EDGE_WARN_CELLS = 10
MAX_ROUTE_NM = 100.0                # plan §5c hard limit
SQRT2 = math.sqrt(2.0)

# Cardinal first then diagonal so the heap tends to explore axis-aligned
# moves first all-else-equal.
_NEIGHBORS_8: Tuple[Tuple[int, int, float], ...] = (
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, SQRT2),
    (-1,  1, SQRT2),
    ( 1, -1, SQRT2),
    ( 1,  1, SQRT2),
)


# --------------------------------------------------------------------------
# Public dataclasses
# --------------------------------------------------------------------------

@dataclass
class Waypoint:
    lat: float
    lon: float
    name: Optional[str] = None

    def as_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class RouteResult:
    ok: bool
    waypoints: List[Waypoint] = field(default_factory=list)
    distance_nm: float = 0.0
    min_depth_m: Optional[float] = None
    hazards_near: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Diagnostics (not part of the public API contract; useful for the CLI).
    runtime_s: float = 0.0
    nodes_expanded: int = 0
    # v2 (spatio-temporal) fields, populated only when optimize != "safe".
    optimize_mode: Optional[str] = None
    departure_time: Optional[str] = None     # ISO8601 UTC
    arrival_time: Optional[str] = None       # ISO8601 UTC
    duration_minutes: Optional[float] = None
    fuel_gallons: Optional[float] = None
    legs: Optional[List[Dict[str, Any]]] = None


# --------------------------------------------------------------------------
# Graph loading
# --------------------------------------------------------------------------

@dataclass
class Graph:
    nogo: np.ndarray            # bool[H, W]
    cost: np.ndarray            # float32[H, W]
    transform: GridAffine
    crs: str
    resolution_m: float
    bbox_wgs84: Tuple[float, float, float, float]
    bbox_utm: Tuple[float, float, float, float]
    built_at: str
    depth_threshold_m: float
    cells_navigable: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nogo.shape


def load_graph(path: Path) -> Graph:
    """Load a .npz into a ``Graph``.  Call once at service start."""
    path = Path(path)
    with np.load(path) as d:
        nogo = np.asarray(d["nogo"], dtype=bool)
        cost = np.asarray(d["cost"], dtype=np.float32)
        transform = GridAffine.from_array(d["transform"])
        crs = str(d["crs"])
        resolution_m = float(d["resolution_m"])
        bbox_wgs84 = tuple(float(v) for v in d["bbox_wgs84"])
        bbox_utm = tuple(float(v) for v in d["bbox_utm"])
        built_at = str(d["built_at"])
        depth_threshold_m = float(d["depth_threshold_m"])
    cells_navigable = int(nogo.size - np.count_nonzero(nogo))
    log.info("graph loaded: shape=%s navigable=%d/%d res=%.1fm",
             nogo.shape, cells_navigable, nogo.size, resolution_m)
    return Graph(nogo=nogo, cost=cost, transform=transform, crs=crs,
                 resolution_m=resolution_m, bbox_wgs84=bbox_wgs84,
                 bbox_utm=bbox_utm, built_at=built_at,
                 depth_threshold_m=depth_threshold_m,
                 cells_navigable=cells_navigable)


# --------------------------------------------------------------------------
# In-bbox helpers
# --------------------------------------------------------------------------

def _latlon_in_bbox(lat: float, lon: float,
                    bbox_wgs84: Sequence[float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


def _nudge_to_water(nogo: np.ndarray, row: int, col: int,
                    max_radius_cells: int,
                    components: Optional[np.ndarray] = None,
                    target_component: Optional[int] = None
                    ) -> Optional[Tuple[int, int]]:
    """Spiral outward from (row, col) up to ``max_radius_cells`` looking for
    a cell where ``nogo`` is False.  Returns the nearest navigable cell or
    None if none exist within the radius.

    If ``components`` and ``target_component`` are given, only cells in that
    component are considered acceptable.  This matters in real charts:
    marinas often sit just inside a disconnected inlet (e.g. Padilla Bay
    relative to the Salish Sea main basin), and the geometrically-nearest
    water cell may be in an unreachable pocket.

    Implementation uses concentric L∞ rings; small radii (<=20) are <1 ms.
    """
    H, W = nogo.shape

    def _ok(rr: int, cc: int) -> bool:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if nogo[rr, cc]:
            return False
        if components is not None and target_component is not None:
            return bool(components[rr, cc] == target_component)
        return True

    if _ok(row, col):
        return row, col
    for r in range(1, max_radius_cells + 1):
        for dc in range(-r, r + 1):
            for dr in (-r, r):
                rr, cc = row + dr, col + dc
                if _ok(rr, cc):
                    return rr, cc
        for dr in range(-r + 1, r):
            for dc in (-r, r):
                rr, cc = row + dr, col + dc
                if _ok(rr, cc):
                    return rr, cc
    return None


def _prune_bbox(start_rc: Tuple[int, int], end_rc: Tuple[int, int],
                pad_cells: int, grid_shape: Tuple[int, int]
                ) -> Tuple[int, int, int, int]:
    """Return (row0, row1, col0, col1) of the sub-window containing both
    endpoints with ``pad_cells`` padding, clipped to the full grid."""
    r0, c0 = start_rc
    r1, c1 = end_rc
    row_min = max(0, min(r0, r1) - pad_cells)
    row_max = min(grid_shape[0], max(r0, r1) + pad_cells + 1)
    col_min = max(0, min(c0, c1) - pad_cells)
    col_max = min(grid_shape[1], max(c0, c1) + pad_cells + 1)
    return row_min, row_max, col_min, col_max


# --------------------------------------------------------------------------
# A* core
# --------------------------------------------------------------------------

def _astar(cost: np.ndarray, h: np.ndarray,
           start_local: Tuple[int, int], end_local: Tuple[int, int]
           ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    """Run A* on a sub-window cost array.  Returns (path_or_None,
    nodes_expanded).  Coordinates are in the local sub-window frame.

    Both ``cost`` and ``h`` are 2D float32 arrays of equal shape; ``h`` is
    the precomputed admissible heuristic (cell-cost units).

    Implementation note: the hot loop indexes 1D *flat* views (``cost1d``
    etc.) keyed by ``idx = r * W + c`` — flat-array numpy indexing is ~3×
    faster than 2D ``arr[r, c]`` in CPython, which matters a lot for
    100k-node routes.  No-go cells are detected via a flat bool mask
    (``blocked1d``) which is faster than ``math.isfinite`` on a float.
    """
    H, W = cost.shape
    size = H * W
    cost1d = cost.reshape(-1)
    h1d = h.reshape(-1)
    blocked1d = ~np.isfinite(cost1d)
    closed1d = np.zeros(size, dtype=bool)
    INF = math.inf
    g_score1d = np.full(size, INF, dtype=np.float32)
    # came_dir[idx] encodes the parent direction as 1..8 (0 = no parent).
    came_dir1d = np.zeros(size, dtype=np.int8)

    sr, sc = start_local
    er, ec = end_local
    start_idx = sr * W + sc
    end_idx = er * W + ec
    g_score1d[start_idx] = 0.0
    counter = 0
    # Each heap entry: (f_score, counter, flat_idx, row, col).  Carrying
    # (row, col) avoids a divmod on every pop.
    open_heap: List[Tuple[float, int, int, int, int]] = [
        (float(h1d[start_idx]), counter, start_idx, sr, sc)
    ]
    counter += 1

    heappush = heapq.heappush
    heappop = heapq.heappop
    # Per-direction (delta_row, delta_col, flat_offset, edge_multiplier).
    # The flat offset is precomputed from the sub-window width so the
    # inner loop avoids any per-iteration arithmetic.
    nbrs = (
        (-1,  0, -W,     1.0),    # N
        ( 1,  0,  W,     1.0),    # S
        ( 0, -1, -1,     1.0),    # W
        ( 0,  1,  1,     1.0),    # E
        (-1, -1, -W - 1, SQRT2),  # NW
        (-1,  1, -W + 1, SQRT2),  # NE
        ( 1, -1,  W - 1, SQRT2),  # SW
        ( 1,  1,  W + 1, SQRT2),  # SE
    )

    nodes_expanded = 0

    while open_heap:
        _, _, idx, r, c = heappop(open_heap)
        if closed1d[idx]:
            continue
        closed1d[idx] = True
        nodes_expanded += 1
        if idx == end_idx:
            break

        g_cur = g_score1d[idx]
        for k, (dr, dc, offset, edge_mul) in enumerate(nbrs):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = idx + offset
            if blocked1d[nidx] or closed1d[nidx]:
                continue
            # Block diagonal corner-cutting: a diagonal move (dr, dc both
            # nonzero) is only legal when both adjacent cardinals are also
            # navigable. Otherwise A* could squeeze through where two land
            # cells touch at a corner — a route that crosses land.
            if dr != 0 and dc != 0:
                if blocked1d[idx + dr * W]:
                    continue
                if blocked1d[idx + dc]:
                    continue
            tentative_g = g_cur + edge_mul * cost1d[nidx]
            if tentative_g < g_score1d[nidx]:
                g_score1d[nidx] = tentative_g
                came_dir1d[nidx] = k + 1
                heappush(open_heap,
                         (tentative_g + h1d[nidx], counter,
                          nidx, nr, nc))
                counter += 1
    else:
        return None, nodes_expanded

    path: List[Tuple[int, int]] = []
    idx = end_idx
    while True:
        r, c = divmod(idx, W)
        path.append((r, c))
        if idx == start_idx:
            break
        k = came_dir1d[idx]
        idx = idx - nbrs[k - 1][2]
    path.reverse()
    return path, nodes_expanded


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------

def _path_to_utm(path_local: List[Tuple[int, int]],
                 transform: GridAffine,
                 row_off: int, col_off: int) -> List[Tuple[float, float]]:
    """Convert local sub-window (row, col) cells to UTM (E, N) cell centers."""
    out = []
    for r, c in path_local:
        e, n = transform.cell_center_utm(r + row_off, c + col_off)
        out.append((e, n))
    return out


def _simplify_utm(utm_pts: List[Tuple[float, float]],
                  tol_m: float) -> List[Tuple[float, float]]:
    """Douglas-Peucker in UTM.  Returns list of (E, N).  Drops single-point
    cases gracefully."""
    if len(utm_pts) <= 2:
        return list(utm_pts)
    ls = LineString(utm_pts)
    simp = ls.simplify(tol_m, preserve_topology=False)
    return list(simp.coords)


def _line_clear_grid(a: Tuple[int, int], b: Tuple[int, int],
                     nogo: np.ndarray,
                     row_off: int = 0, col_off: int = 0) -> bool:
    """Bresenham line from (sub-grid) cell ``a`` to ``b``.  Returns True if
    every cell on the line is navigable in ``nogo`` after applying the
    ``row_off``/``col_off`` translation back to full-grid coords.

    Enforces the same diagonal corner-cut rule as ``_astar``: when the
    Bresenham step is diagonal, BOTH adjacent cardinal cells must also be
    navigable.  Otherwise a simplified leg could cross a blocked corner
    where two land cells touch, even though every cell on the line itself
    is navigable.
    """
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if nogo[r + row_off, c + col_off]:
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        step_r = e2 > -dc
        step_c = e2 < dr
        # Diagonal step: enforce A*'s corner-cut rule — both cardinals open.
        if step_r and step_c:
            if (nogo[r + sr + row_off, c + col_off]
                    or nogo[r + row_off, c + sc + col_off]):
                return False
        if step_r:
            err -= dc
            r += sr
        if step_c:
            err += dr
            c += sc


def _simplify_validated(path_local: List[Tuple[int, int]],
                        transform: GridAffine,
                        row_off: int, col_off: int,
                        tol_m: float,
                        nogo: np.ndarray) -> List[Tuple[float, float]]:
    """Greedy string-pulling / visibility simplification of the A* grid path.

    Walks the path forward.  From each anchor, looks as far ahead as possible
    while the straight Bresenham line from anchor to candidate stays inside
    navigable cells AND honors A*'s corner-cut rule.  Jumps to that farthest
    candidate.  Repeats until the end is reached.

    Replaces the old "Douglas-Peucker then splice raw cells on failure"
    pipeline, which produced 30+°-per-leg zig-zag because the splice fallback
    dumped dense 25 m grid stair-steps back into the public route around
    narrows.  String-pulling gives a polyline that's both (1) provably never
    crosses no-go, by construction, and (2) as straight as the geometry
    allows — typically dropping waypoint counts 5-10× on open-water routes.

    ``tol_m`` is kept in the signature for API compatibility but is unused;
    the visibility check is the only constraint.
    """
    if len(path_local) <= 2:
        return _path_to_utm(path_local, transform, row_off, col_off)

    n = len(path_local)
    out_idxs: List[int] = [0]
    i = 0
    while i < n - 1:
        # Binary search over the candidate window for the farthest visible
        # index.  Visibility isn't strictly monotone (open->blocked->open is
        # possible in some chart topologies) but it's nearly always monotone
        # in practice, so binary search finds an answer ~log(n) faster than
        # the linear scan, and we fall back to a short linear sweep around
        # the boundary to catch the non-monotone case.
        lo, hi = i + 1, n - 1
        best = i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _line_clear_grid(path_local[i], path_local[mid],
                                nogo, row_off, col_off):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        # Edge-case sweep: if there's a longer-reach visible point past
        # ``best`` (non-monotone visibility), prefer it.
        for j in range(best + 1, min(n, best + 8)):
            if _line_clear_grid(path_local[i], path_local[j],
                                nogo, row_off, col_off):
                best = j
        out_idxs.append(best)
        i = best

    log.debug("simplify: %d input cells -> %d visibility waypoints",
              n, len(out_idxs))
    return [transform.cell_center_utm(path_local[k][0] + row_off,
                                      path_local[k][1] + col_off)
            for k in out_idxs]


def _utm_to_waypoints(utm_pts: List[Tuple[float, float]]) -> List[Waypoint]:
    return [Waypoint(*utm_to_latlon(e, n)) for (e, n) in utm_pts]


def _waypoints_distance_nm(wps: List[Waypoint]) -> float:
    total = 0.0
    for a, b in zip(wps[:-1], wps[1:]):
        total += haversine_nm(a.lat, a.lon, b.lat, b.lon)
    return total


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Time-dependent A* (spatio-temporal)
# --------------------------------------------------------------------------

# Per-direction (dr, dc, flat_offset_factor, edge_dist_multiplier, heading_deg).
# Heading is the compass-true direction of travel for this neighbor step
# (0=N, 90=E, 180=S, 270=W).  This matches the polar/currents convention
# (u_east, v_north).  Diagonal headings are mid-points (45°, 135°, etc.).
_NBR_TIME = (
    (-1,  0,  1.0,    0.0),    # N
    ( 1,  0,  1.0,  180.0),    # S
    ( 0, -1,  1.0,  270.0),    # W
    ( 0,  1,  1.0,   90.0),    # E
    (-1, -1,  SQRT2, 315.0),   # NW
    (-1,  1,  SQRT2,  45.0),   # NE
    ( 1, -1,  SQRT2, 225.0),   # SW
    ( 1,  1,  SQRT2, 135.0),   # SE
)


def _astar_time(cost: np.ndarray, h_arr_s: np.ndarray,
                start_local: Tuple[int, int],
                end_local: Tuple[int, int],
                cf_tensor: np.ndarray, field_ds: int,
                stw_ms: float, resolution_m: float,
                ) -> Tuple[Optional[List[Tuple[int, int]]], int,
                           List[float], List[Tuple[float, float]]]:
    """Time-dependent A* on the sub-window.  Edge cost is seconds of
    actual travel time given the precomputed current field.

    Arguments:
        cost:        float32[H, W] -- v1 cost surface; inf = no-go.
                     We don't *multiply* by it (cost is unitless), but
                     we DO use it to detect no-go via isfinite(), AND
                     to penalise edge-band cells by inflating the
                     traversal time by their factor (so the time-mode
                     A* still routes around hazards just like v1).
        h_arr_s:     float32[H, W] -- admissible heuristic in seconds.
        cf_tensor:   float32[T, ds_h, ds_w, 2] -- precomputed (u, v) m/s.
        field_ds:    int -- spatial down-sample factor (cells per ds cell).
        stw_ms:      vessel speed through water, m/s (scalar; cruise).
        resolution_m: grid resolution.

    Returns:
        (path_cells, nodes_expanded, edge_times_s, edge_currents)
        edge_times_s[i] is the time of the edge leading INTO path_cells[i+1].
        edge_currents[i] is the (u, v) m/s used for that edge.
    """
    H, W = cost.shape
    size = H * W
    cost1d = cost.reshape(-1)
    blocked1d = ~np.isfinite(cost1d)
    closed1d = np.zeros(size, dtype=bool)
    g_time_s = np.full(size, math.inf, dtype=np.float64)
    came_dir1d = np.zeros(size, dtype=np.int8)

    # Precompute the per-direction unit heading vector (sin, cos) so we
    # don't repeatedly call math.radians/sin/cos inside the hot loop.
    nbrs_pre: Tuple[Tuple[int, int, int, float, float, float, float], ...] = tuple(
        (dr, dc, dr * W + dc, dist_mul,
         math.sin(math.radians(hd)),    # = unit east
         math.cos(math.radians(hd)),    # = unit north
         dist_mul * resolution_m)       # edge_distance_m
        for (dr, dc, dist_mul, hd) in _NBR_TIME
    )

    h_arr_s_1d = h_arr_s.reshape(-1)

    T = int(cf_tensor.shape[0])
    ds_h = int(cf_tensor.shape[1])
    ds_w = int(cf_tensor.shape[2])
    # 10-minute slices hard-coded in route_time(); seconds-to-slice
    # conversion factor:
    SLICE_S = 600.0

    sr, sc = start_local
    er, ec = end_local
    start_idx = sr * W + sc
    end_idx = er * W + ec
    g_time_s[start_idx] = 0.0
    counter = 0
    open_heap: List[Tuple[float, int, int, int, int]] = [
        (float(h_arr_s_1d[start_idx]), counter, start_idx, sr, sc)
    ]
    counter += 1

    heappush = heapq.heappush
    heappop = heapq.heappop

    nodes_expanded = 0
    # 0.5 kt floor on along-track SOG.  Keeps edge_time finite when adverse
    # current overpowers STW, and stays under (STW + 5 kt) so the heuristic
    # remains admissible.
    SOG_FLOOR_MS = 0.5 * KTS_TO_MS

    while open_heap:
        _, _, idx, r, c = heappop(open_heap)
        if closed1d[idx]:
            continue
        closed1d[idx] = True
        nodes_expanded += 1
        if idx == end_idx:
            break

        g_cur = g_time_s[idx]
        ti = int(g_cur / SLICE_S)
        if ti < 0:
            ti = 0
        elif ti >= T:
            ti = T - 1
        cf_slice = cf_tensor[ti]   # [ds_h, ds_w, 2] view, free

        for k, (dr, dc, offset, dist_mul,
                u_unit, v_unit, edge_dist_m) in enumerate(nbrs_pre):
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            nidx = idx + offset
            if blocked1d[nidx] or closed1d[nidx]:
                continue
            if dr != 0 and dc != 0:
                if blocked1d[idx + dr * W]:
                    continue
                if blocked1d[idx + dc]:
                    continue

            dsr = nr // field_ds
            dsc = nc // field_ds
            if dsr >= ds_h:
                dsr = ds_h - 1
            if dsc >= ds_w:
                dsc = ds_w - 1
            u = float(cf_slice[dsr, dsc, 0])
            v = float(cf_slice[dsr, dsc, 1])

            u_b = stw_ms * u_unit + u
            v_b = stw_ms * v_unit + v
            sog_along = u_b * u_unit + v_b * v_unit
            if sog_along < SOG_FLOOR_MS:
                sog_along = SOG_FLOOR_MS
            c_pen = float(cost1d[nidx])
            edge_time_s = edge_dist_m * c_pen / sog_along

            tentative_g = g_cur + edge_time_s
            if tentative_g < g_time_s[nidx]:
                g_time_s[nidx] = tentative_g
                came_dir1d[nidx] = k + 1
                heappush(open_heap,
                         (tentative_g + float(h_arr_s_1d[nidx]),
                          counter, nidx, nr, nc))
                counter += 1
    else:
        return None, nodes_expanded, [], []

    # Reconstruct path; capture each step's direction index so the metadata
    # pass below doesn't have to linear-search nbrs_pre per edge.
    path: List[Tuple[int, int]] = []
    dir_indices: List[int] = []      # direction taken INTO path[i+1]
    idx = end_idx
    while True:
        r, c = divmod(idx, W)
        path.append((r, c))
        if idx == start_idx:
            break
        k = came_dir1d[idx]
        dir_indices.append(k - 1)
        idx = idx - nbrs_pre[k - 1][2]
    path.reverse()
    dir_indices.reverse()

    edge_times: List[float] = []
    edge_currents: List[Tuple[float, float]] = []
    t_prev_s = 0.0   # running accumulator — avoid quadratic sum(edge_times)
    for i in range(1, len(path)):
        nr, nc = path[i]
        _, _, _, _, u_unit, v_unit, edge_dist_m = nbrs_pre[dir_indices[i - 1]]
        ti = int(t_prev_s / SLICE_S)
        if ti < 0:
            ti = 0
        elif ti >= T:
            ti = T - 1
        dsr = nr // field_ds
        dsc = nc // field_ds
        if dsr >= ds_h:
            dsr = ds_h - 1
        if dsc >= ds_w:
            dsc = ds_w - 1
        u = float(cf_tensor[ti, dsr, dsc, 0])
        v = float(cf_tensor[ti, dsr, dsc, 1])
        u_b = stw_ms * u_unit + u
        v_b = stw_ms * v_unit + v
        sog_along = u_b * u_unit + v_b * v_unit
        if sog_along < SOG_FLOOR_MS:
            sog_along = SOG_FLOOR_MS
        c_pen = float(cost1d[nr * W + nc])
        edge_t = edge_dist_m * c_pen / sog_along
        edge_times.append(edge_t)
        edge_currents.append((u, v))
        t_prev_s += edge_t

    return path, nodes_expanded, edge_times, edge_currents


def _build_leg_metadata(wps: List[Waypoint],
                        cf_tensor: np.ndarray, field_ds: int,
                        step_min: int,
                        stw_kts: float, polar: VesselPolar,
                        departure_time: datetime,
                        field_obj: Optional[CurrentField] = None,
                        ) -> List[Dict[str, Any]]:
    """Per-leg current + duration breakdown for the public response.

    Walks each (wp[i], wp[i+1]) leg in WGS84, computes great-circle distance
    + bearing, samples the IDW current at the leg's midpoint at mid-time,
    and projects (boat-through-water + current) onto the heading for the
    along-track SOG.  Used only for a small (5-20) waypoint list — perf is
    not a concern at this scale.
    """
    SLICE_S = step_min * 60.0
    stw_ms = stw_kts * KTS_TO_MS
    cf = field_obj if field_obj is not None else get_default_field()

    legs: List[Dict[str, Any]] = []
    t_elapsed_s = 0.0
    for i in range(len(wps) - 1):
        a = wps[i]
        b = wps[i + 1]
        d_nm = haversine_nm(a.lat, a.lon, b.lat, b.lon)
        d_m = d_nm * 1852.0
        lat1 = math.radians(a.lat)
        lat2 = math.radians(b.lat)
        dlon = math.radians(b.lon - a.lon)
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        brg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

        mid_lat = 0.5 * (a.lat + b.lat)
        mid_lon = 0.5 * (a.lon + b.lon)
        leg_mid_s = t_elapsed_s + (d_m / max(stw_ms, 0.1)) * 0.5
        mid_dt = departure_time + timedelta(seconds=leg_mid_s)
        uv = cf.field_at(np.asarray([[mid_lat, mid_lon]], dtype=np.float64),
                         mid_dt)
        u = float(uv[0, 0])
        v = float(uv[0, 1])
        sog_kts, _ = effective_sog_and_fuel(brg, stw_kts, (u, v), polar=polar)
        # effective_sog_and_fuel returns vector magnitude; for leg-time we want
        # along-track SOG (projection onto heading) so a beam current doesn't
        # falsely shorten the leg.  Recover via simple projection.
        rad = math.radians(brg)
        u_unit = math.sin(rad)
        v_unit = math.cos(rad)
        sog_along_ms = max((stw_ms * u_unit + u) * u_unit
                            + (stw_ms * v_unit + v) * v_unit,
                           0.25)
        leg_dur_s = d_m / sog_along_ms
        leg_min = leg_dur_s / 60.0

        cur_kts = math.hypot(u, v) * MS_TO_KTS
        cur_dir = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0

        legs.append({
            "from_idx": i,
            "to_idx": i + 1,
            "distance_nm": round(d_nm, 3),
            "minutes": round(leg_min, 2),
            "heading_deg": round(brg, 1),
            "current_kts": round(cur_kts, 2),
            "current_dir_deg": round(cur_dir, 1),
        })
        t_elapsed_s += leg_dur_s
    return legs


# --------------------------------------------------------------------------
# Router class
# --------------------------------------------------------------------------

class Router:
    """Loaded-once, called-many wrapper around the routing graph."""

    def __init__(self, graph: Graph):
        self.graph = graph
        # Connected-component labelling of the navigable mask.  Computed
        # lazily on first call to ``route`` so simple "is graph loaded"
        # queries (Phase 4 /health) stay cheap.  ~17M cells: ~250 ms with
        # scipy.ndimage.label.
        self._components: Optional[np.ndarray] = None
        self._main_component: int = 0

    @classmethod
    def from_npz(cls, path: Path) -> "Router":
        return cls(load_graph(path))

    # ---- Connectivity precompute ------------------------------------------

    def _ensure_components(self) -> None:
        if self._components is not None:
            return
        from scipy.ndimage import label
        t = time.perf_counter()
        # 4-connectivity matches A*'s legal-move set exactly: every diagonal
        # A* allows requires both adjacent cardinals to be open (no corner-
        # cutting — see _astar), and if those cardinals are open, the 4-conn
        # label already connects the four cells via the cardinal path. So
        # 4-conn label == A*-connectivity in this grid, and 8-conn would
        # over-merge across blocked corners that A* refuses to cross.
        labels, _n = label(~self.graph.nogo)
        sizes = np.bincount(labels.ravel())
        if sizes.size > 1:
            sizes[0] = 0
            main = int(sizes.argmax())
        else:
            main = 0
        self._components = labels
        self._main_component = main
        log.info("connected components computed in %.2fs; main = %d "
                 "(%d cells)", time.perf_counter() - t, main,
                 int(sizes[main]) if sizes.size > 1 else 0)

    # ---- Public entrypoint -------------------------------------------------

    def route(self,
              start: Tuple[float, float],
              end: Tuple[float, float],
              *,
              departure_time: Optional[str] = None,
              optimize: str = "safe",
              ) -> RouteResult:
        """Plan a distance-optimal route from ``start`` to ``end`` (lat, lon).

        ``departure_time`` and ``optimize`` are part of the public contract
        but unused on this path; the time-aware modes live in
        ``route_time``.
        """
        t0 = time.perf_counter()
        start_lat, start_lon = start
        end_lat, end_lon = end

        bbox = self.graph.bbox_wgs84
        if not _latlon_in_bbox(start_lat, start_lon, bbox):
            return RouteResult(ok=False,
                               error="coordinates outside routing coverage area",
                               runtime_s=time.perf_counter() - t0)
        if not _latlon_in_bbox(end_lat, end_lon, bbox):
            return RouteResult(ok=False,
                               error="coordinates outside routing coverage area",
                               runtime_s=time.perf_counter() - t0)

        gc_nm = haversine_nm(start_lat, start_lon, end_lat, end_lon)
        if gc_nm > MAX_ROUTE_NM:
            return RouteResult(ok=False,
                               error="route distance exceeds 100 nm limit",
                               runtime_s=time.perf_counter() - t0)

        nogo = self.graph.nogo
        cost = self.graph.cost
        H, W = nogo.shape

        s_row, s_col = latlon_to_grid(start_lat, start_lon, self.graph.transform)
        e_row, e_col = latlon_to_grid(end_lat, end_lon, self.graph.transform)
        # Bounds: latlon_in_bbox above gates the obvious off-grid case but
        # rounding to whole cells can land just off the edge — clip.
        if not (0 <= s_row < H and 0 <= s_col < W and
                0 <= e_row < H and 0 <= e_col < W):
            return RouteResult(ok=False,
                               error="coordinates outside routing coverage area",
                               runtime_s=time.perf_counter() - t0)

        warnings_out: List[str] = []

        # Precompute connected components on first use; lets the nudge
        # logic skip isolated water pockets that would A* down to an
        # immediate "no navigable path found".
        self._ensure_components()
        components = self._components
        main = self._main_component

        # Destination: small nudge allowed (named-harbor coords often
        # land on a pier at 50 m resolution; see END_NUDGE_RADIUS_M).  We
        # prefer a cell in the main connected component if possible.  If
        # the nudge fails, the destination is land-locked → refuse.
        end_nudge_used = False
        if nogo[e_row, e_col] or (components is not None and
                                  components[e_row, e_col] != main):
            end_radius_cells = int(math.ceil(
                END_NUDGE_RADIUS_M / self.graph.resolution_m))
            end_nudged = _nudge_to_water(
                nogo, e_row, e_col, end_radius_cells,
                components=components, target_component=main)
            if end_nudged is None:
                # Couldn't reach the main basin; try without component
                # constraint so we can still surface a sensible error
                # (it'll be either "destination is not on water" or
                # "no navigable path found").
                end_nudged = _nudge_to_water(
                    nogo, e_row, e_col, end_radius_cells)
            if end_nudged is None:
                return RouteResult(ok=False,
                                   error="destination is not on water",
                                   runtime_s=time.perf_counter() - t0)
            new_row, new_col = end_nudged
            if (new_row, new_col) != (e_row, e_col):
                end_nudge_used = True
                nlat, nlon = grid_to_latlon(new_row, new_col,
                                            self.graph.transform)
                dist_m = haversine_nm(end_lat, end_lon, nlat, nlon) * 1852.0
                warnings_out.append(
                    f"destination nudged to nearest water ({dist_m:.0f} m "
                    f"away at {nlat:.4f}, {nlon:.4f})"
                )
            e_row, e_col = new_row, new_col

        # Start: try to nudge if it's no-go (or in a disconnected
        # pocket relative to the destination).  Target the destination's
        # component so we don't strand the A* on an island.
        target_for_start = (int(components[e_row, e_col])
                            if components is not None else None)
        nudge_used = False
        need_start_nudge = (
            nogo[s_row, s_col] or
            (components is not None and target_for_start is not None
             and components[s_row, s_col] != target_for_start)
        )
        if need_start_nudge:
            radius_cells = int(math.ceil(
                START_NUDGE_RADIUS_M / self.graph.resolution_m))
            nudged = _nudge_to_water(
                nogo, s_row, s_col, radius_cells,
                components=components, target_component=target_for_start)
            if nudged is None:
                # If the constrained nudge fails, the start is either
                # land-locked entirely OR it's in a water pocket
                # disconnected from the destination's basin.  Distinguish
                # by retrying without the component constraint: if THAT
                # finds water, we know the start is reachable in some
                # local water area that just doesn't connect to the
                # destination — don't run A* on a doomed start.
                any_water = _nudge_to_water(nogo, s_row, s_col, radius_cells)
                if any_water is None:
                    return RouteResult(
                        ok=False,
                        error="start is not on water and no nearby water",
                        runtime_s=time.perf_counter() - t0)
                return RouteResult(
                    ok=False,
                    error=("start is in a water basin disconnected from "
                           "the destination; no route possible at this "
                           "chart resolution"),
                    runtime_s=time.perf_counter() - t0)
            new_row, new_col = nudged
            if (new_row, new_col) != (s_row, s_col):
                dist_m = math.hypot((new_row - s_row), (new_col - s_col)) \
                    * self.graph.resolution_m
                nudge_used = True
                nudged_lat, nudged_lon = grid_to_latlon(
                    new_row, new_col, self.graph.transform)
                warnings_out.append(
                    f"start nudged to nearest water ({dist_m:.0f} m away "
                    f"at {nudged_lat:.4f}, {nudged_lon:.4f})"
                )
            s_row, s_col = new_row, new_col

        pad_cells = int(math.ceil(PRUNE_PAD_M / self.graph.resolution_m))
        row0, row1, col0, col1 = _prune_bbox(
            (s_row, s_col), (e_row, e_col), pad_cells, (H, W))
        sub_cost = np.ascontiguousarray(cost[row0:row1, col0:col1])
        sub_h, sub_w = sub_cost.shape

        E_grid, N_grid = utm_grid_for_bbox(self.graph.transform,
                                           row0, col0, sub_h, sub_w)
        # End coordinate in UTM (use the grid cell center, not raw input, so
        # the heuristic is consistent with the path).
        e_E, e_N = self.graph.transform.cell_center_utm(e_row, e_col)
        # Distance in meters → cell-cost units = meters / 50 m.
        h_arr = np.ascontiguousarray(
            np.hypot(E_grid - e_E, N_grid - e_N).astype(np.float32)
            / np.float32(self.graph.resolution_m)
        )

        start_local = (s_row - row0, s_col - col0)
        end_local = (e_row - row0, e_col - col0)
        path_local, nodes_expanded = _astar(sub_cost, h_arr,
                                            start_local, end_local)
        if path_local is None:
            return RouteResult(ok=False,
                               error="no navigable path found",
                               warnings=warnings_out,
                               runtime_s=time.perf_counter() - t0,
                               nodes_expanded=nodes_expanded)

        path_arr = np.asarray(path_local, dtype=np.int64)
        path_costs = sub_cost[path_arr[:, 0], path_arr[:, 1]]
        hazards_near = int(np.count_nonzero(path_costs >= COST_EDGE))
        if hazards_near > 0:
            warnings_out.append(
                f"route passes close to hazards ({hazards_near} edge-buffer cells)"
            )
        rmin, cmin = path_arr.min(axis=0)
        rmax, cmax = path_arr.max(axis=0)
        margin = min(rmin, cmin, sub_h - 1 - rmax, sub_w - 1 - cmax)
        # Only warn if the prune actually clipped on a side that the start
        # or end wasn't already on; otherwise we'd flag every route.
        if margin < EDGE_WARN_CELLS and pad_cells > EDGE_WARN_CELLS:
            warnings_out.append(
                "route hugs prune-bbox edge; consider a wider search"
            )

        # Visibility-validated Douglas-Peucker simplify in UTM, project
        # back to WGS84.  DP gives clean geometric simplification; the
        # validation step ensures no simplified leg crosses a no-go cell.
        simp_utm = _simplify_validated(
            path_local, self.graph.transform, row0, col0,
            SIMPLIFY_TOL_M, nogo,
        )
        wps = _utm_to_waypoints(simp_utm)
        # Tag endpoints for friendlier downstream consumption.
        if wps:
            wps[0] = Waypoint(wps[0].lat, wps[0].lon,
                              "Start (nudged)" if nudge_used else "Start")
            wps[-1] = Waypoint(wps[-1].lat, wps[-1].lon, "End")

        distance_nm = _waypoints_distance_nm(wps)

        result = RouteResult(
            ok=True,
            waypoints=wps,
            distance_nm=distance_nm,
            min_depth_m=None,
            hazards_near=hazards_near,
            warnings=warnings_out,
            error=None,
            runtime_s=time.perf_counter() - t0,
            nodes_expanded=nodes_expanded,
        )
        log.debug("route ok: %.1f nm, %d wps, %d nodes expanded in %.2fs",
                  distance_nm, len(wps), nodes_expanded, result.runtime_s)
        return result

    # ------------------------------------------------------------------
    # Spatio-temporal routing (time / fuel / depart_window)
    # ------------------------------------------------------------------

    def route_time(self,
                   start: Tuple[float, float],
                   end: Tuple[float, float],
                   *,
                   departure_time: datetime,
                   optimize: str = "time",
                   current_field: Optional[CurrentField] = None,
                   polar: Optional[VesselPolar] = None,
                   ) -> RouteResult:
        """Plan a time-optimal route against the tidal-current field.

        ``optimize`` is ``"time"`` or ``"fuel"``.  At a fixed cruise STW the
        two collapse to the same path (fuel = constant_rate * time); the
        flag exists so callers don't need to migrate once the polar grows
        a mixed cruise/displacement optimum.

        ``current_field`` and ``polar`` default to
        ``CurrentField.from_default_catalog()`` / ``VesselPolar.load()``;
        callers running a sweep should pass pre-built instances to skip
        repeated NOAA fetches.
        """
        t0 = time.perf_counter()
        polar = polar if polar is not None else VesselPolar.load()
        stw_kts = float(polar.cruise_stw_kts)
        stw_ms = stw_kts * KTS_TO_MS

        # ---- shared prep (validation, nudges, prune bbox) ----
        prep = self._prepare_search(start, end, t0)
        if prep.get("error"):
            return prep["error_result"]
        nogo = self.graph.nogo
        s_row, s_col = prep["s_row"], prep["s_col"]
        e_row, e_col = prep["e_row"], prep["e_col"]
        row0, row1, col0, col1 = prep["bbox"]
        sub_cost = prep["sub_cost"]
        sub_h, sub_w = sub_cost.shape
        warnings_out = list(prep["warnings"])
        nudge_used = prep["nudge_used"]

        # ---- precompute current field tensor over the sub-window ----
        # Departure-window upper bound: best-case time to destination = great-
        # circle / (STW + 5 kt); we then triple it as a safe upper for the A*
        # window (currents can stretch out a trip materially).  Cap at 12 h.
        start_lat, start_lon = start
        end_lat, end_lon = end
        gc_nm = haversine_nm(start_lat, start_lon, end_lat, end_lon)
        best_case_h = gc_nm / (stw_kts + 5.0)        # admissible-style lower bound
        window_h = max(2.0, min(12.0, best_case_h * 3.0))
        STEP_MIN = 10                                # 10-minute slices
        n_steps = max(2, int(math.ceil(window_h * 60 / STEP_MIN)) + 1)

        # Build a coarse spatial down-sample for the field sampling: we
        # don't need a per-25 m current vector.  ~500 m is well below
        # the spatial scale of NOAA station spacing.  Index lookups in
        # the hot loop use the down-sampled cell.
        FIELD_DS = max(1, int(round(500.0 / self.graph.resolution_m)))
        ds_h = max(1, (sub_h + FIELD_DS - 1) // FIELD_DS)
        ds_w = max(1, (sub_w + FIELD_DS - 1) // FIELD_DS)

        # Sample the (lat, lon) coords for each down-sampled cell once.
        ds_rows = np.minimum(np.arange(ds_h, dtype=np.int64) * FIELD_DS,
                             sub_h - 1) + row0
        ds_cols = np.minimum(np.arange(ds_w, dtype=np.int64) * FIELD_DS,
                             sub_w - 1) + col0
        ds_row_grid, ds_col_grid = np.meshgrid(ds_rows, ds_cols, indexing="ij")
        flat_rows = ds_row_grid.reshape(-1)
        flat_cols = ds_col_grid.reshape(-1)
        # Vectorized cell-center -> UTM -> WGS84
        ds_e = (self.graph.transform.a * (flat_cols.astype(np.float64) + 0.5)
                + self.graph.transform.c)
        ds_n = (self.graph.transform.e * (flat_rows.astype(np.float64) + 0.5)
                + self.graph.transform.f)
        lon_arr, lat_arr = utm10n_to_wgs84_transformer().transform(ds_e, ds_n)
        ds_latlon = np.empty((flat_rows.size, 2), dtype=np.float64)
        ds_latlon[:, 0] = lat_arr
        ds_latlon[:, 1] = lon_arr

        try:
            field_obj = (current_field if current_field is not None
                         else get_default_field())
        except (ValueError, OSError) as e:
            # Empty station catalog (offline install / cold cache) or NOAA
            # unreachable.  Surface as a clean ok=false instead of a 500.
            return RouteResult(ok=False,
                               error=f"tidal-current data unavailable: {e}",
                               warnings=warnings_out,
                               runtime_s=time.perf_counter() - t0,
                               optimize_mode=optimize)
        # Tensor shape: [T, ds_h, ds_w, 2].  Each slice is the IDW field at
        # a fixed time over the down-sampled cells.
        cf_tensor = np.zeros((n_steps, ds_h, ds_w, 2), dtype=np.float32)
        for ti in range(n_steps):
            t_slice = departure_time + timedelta(minutes=ti * STEP_MIN)
            uv_flat = field_obj.field_at(ds_latlon, t_slice)   # [N, 2] m/s
            cf_tensor[ti] = uv_flat.reshape(ds_h, ds_w, 2).astype(np.float32)

        t_field_done = time.perf_counter()

        # ---- run time-dependent A* ----
        start_local = (s_row - row0, s_col - col0)
        end_local = (e_row - row0, e_col - col0)

        # Admissible heuristic: h(cell) = euclidean_m / ((STW + 5 kt) * KTS_TO_MS).
        # The +5 kt cap is above any Salish Sea reference-station max (~4 kt),
        # so actual SOG can never exceed it — heuristic stays admissible under
        # the time-varying edge costs of the spatio-temporal A*.
        e_E, e_N = self.graph.transform.cell_center_utm(e_row, e_col)
        E_grid, N_grid = utm_grid_for_bbox(self.graph.transform,
                                           row0, col0, sub_h, sub_w)
        h_max_speed_ms = (stw_kts + 5.0) * KTS_TO_MS
        h_arr_s = np.ascontiguousarray(
            (np.hypot(E_grid - e_E, N_grid - e_N) / h_max_speed_ms).astype(np.float32)
        )

        path_local, nodes_expanded, edge_times_s, edge_currents = _astar_time(
            sub_cost, h_arr_s,
            start_local, end_local,
            cf_tensor, FIELD_DS,
            stw_ms, self.graph.resolution_m,
        )
        t_astar_done = time.perf_counter()

        if path_local is None:
            return RouteResult(ok=False,
                               error="no navigable path found",
                               warnings=warnings_out,
                               runtime_s=time.perf_counter() - t0,
                               nodes_expanded=nodes_expanded,
                               optimize_mode=optimize)

        path_arr = np.asarray(path_local, dtype=np.int64)
        path_costs = sub_cost[path_arr[:, 0], path_arr[:, 1]]
        hazards_near = int(np.count_nonzero(path_costs >= COST_EDGE))
        if hazards_near > 0:
            warnings_out.append(
                f"route passes close to hazards ({hazards_near} edge-buffer cells)"
            )

        rmin, cmin = path_arr.min(axis=0)
        rmax, cmax = path_arr.max(axis=0)
        pad_cells = int(math.ceil(PRUNE_PAD_M / self.graph.resolution_m))
        margin = min(rmin, cmin, sub_h - 1 - rmax, sub_w - 1 - cmax)
        if margin < EDGE_WARN_CELLS and pad_cells > EDGE_WARN_CELLS:
            warnings_out.append(
                "route hugs prune-bbox edge; consider a wider search"
            )

        # Per-leg duration accumulation in seconds: edge_times_s[k] is the
        # cost of the edge ENDING at path_local[k+1] (so the start cell has
        # no edge time).
        total_s = float(sum(edge_times_s)) if edge_times_s else 0.0

        simp_utm = _simplify_validated(
            path_local, self.graph.transform, row0, col0,
            SIMPLIFY_TOL_M, nogo,
        )
        wps = _utm_to_waypoints(simp_utm)
        if wps:
            wps[0] = Waypoint(wps[0].lat, wps[0].lon,
                              "Start (nudged)" if nudge_used else "Start")
            wps[-1] = Waypoint(wps[-1].lat, wps[-1].lon, "End")
        distance_nm = _waypoints_distance_nm(wps)

        # Per-leg metadata for the public response.  Skip on dense paths so
        # the response stays small.
        legs: Optional[List[Dict[str, Any]]] = None
        if len(wps) < 50:
            legs = _build_leg_metadata(
                wps, cf_tensor, FIELD_DS, STEP_MIN, stw_kts, polar,
                departure_time, field_obj=field_obj,
            )
        else:
            warnings_out.append("per-leg detail omitted: too many waypoints")

        fuel_rate_gph = polar.fuel_rate_for_stw(stw_kts)
        duration_h = total_s / 3600.0
        fuel_gal = fuel_rate_gph * duration_h
        arrival = departure_time + timedelta(seconds=total_s)

        result = RouteResult(
            ok=True,
            waypoints=wps,
            distance_nm=distance_nm,
            min_depth_m=None,
            hazards_near=hazards_near,
            warnings=warnings_out,
            error=None,
            runtime_s=time.perf_counter() - t0,
            nodes_expanded=nodes_expanded,
            optimize_mode=optimize,
            departure_time=departure_time.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            arrival_time=arrival.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            duration_minutes=round(total_s / 60.0, 2),
            fuel_gallons=round(fuel_gal, 3),
            legs=legs,
        )
        log.debug("route_time(%s) ok: %.1f nm, %.1f min, %d wps; "
                  "field=%.2fs astar=%.2fs total=%.2fs",
                  optimize, distance_nm, total_s / 60.0, len(wps),
                  t_field_done - t0, t_astar_done - t_field_done,
                  result.runtime_s)
        return result

    # ---- internal shared prep helper -------------------------------------

    def _prepare_search(self,
                        start: Tuple[float, float],
                        end: Tuple[float, float],
                        t0: float) -> Dict[str, Any]:
        """Shared validation + nudge + prune-bbox setup for both ``route``
        and ``route_time``.  Returns a dict with keys:

            error          (None | str)         -- if set, abort
            error_result   (RouteResult)        -- ready-to-return failure
            s_row, s_col, e_row, e_col          -- final cell coords
            bbox           (row0, row1, col0, col1)
            sub_cost       float32[sub_h, sub_w]
            warnings       list[str]
            nudge_used     bool
        """
        start_lat, start_lon = start
        end_lat, end_lon = end
        bbox = self.graph.bbox_wgs84
        if not _latlon_in_bbox(start_lat, start_lon, bbox):
            return {"error": "out_of_scope",
                    "error_result": RouteResult(
                        ok=False,
                        error="coordinates outside routing coverage area",
                        runtime_s=time.perf_counter() - t0)}
        if not _latlon_in_bbox(end_lat, end_lon, bbox):
            return {"error": "out_of_scope",
                    "error_result": RouteResult(
                        ok=False,
                        error="coordinates outside routing coverage area",
                        runtime_s=time.perf_counter() - t0)}

        gc_nm = haversine_nm(start_lat, start_lon, end_lat, end_lon)
        if gc_nm > MAX_ROUTE_NM:
            return {"error": "too_long",
                    "error_result": RouteResult(
                        ok=False,
                        error="route distance exceeds 100 nm limit",
                        runtime_s=time.perf_counter() - t0)}

        nogo = self.graph.nogo
        cost = self.graph.cost
        H, W = nogo.shape

        s_row, s_col = latlon_to_grid(start_lat, start_lon, self.graph.transform)
        e_row, e_col = latlon_to_grid(end_lat, end_lon, self.graph.transform)
        if not (0 <= s_row < H and 0 <= s_col < W and
                0 <= e_row < H and 0 <= e_col < W):
            return {"error": "out_of_scope",
                    "error_result": RouteResult(
                        ok=False,
                        error="coordinates outside routing coverage area",
                        runtime_s=time.perf_counter() - t0)}

        warnings_out: List[str] = []
        self._ensure_components()
        components = self._components
        main = self._main_component

        # End nudge (same logic as v1)
        if nogo[e_row, e_col] or (components is not None and
                                  components[e_row, e_col] != main):
            end_radius_cells = int(math.ceil(
                END_NUDGE_RADIUS_M / self.graph.resolution_m))
            end_nudged = _nudge_to_water(
                nogo, e_row, e_col, end_radius_cells,
                components=components, target_component=main)
            if end_nudged is None:
                end_nudged = _nudge_to_water(
                    nogo, e_row, e_col, end_radius_cells)
            if end_nudged is None:
                return {"error": "dest_not_water",
                        "error_result": RouteResult(
                            ok=False,
                            error="destination is not on water",
                            runtime_s=time.perf_counter() - t0)}
            new_row, new_col = end_nudged
            if (new_row, new_col) != (e_row, e_col):
                nlat, nlon = grid_to_latlon(new_row, new_col,
                                            self.graph.transform)
                dist_m = haversine_nm(end_lat, end_lon, nlat, nlon) * 1852.0
                warnings_out.append(
                    f"destination nudged to nearest water ({dist_m:.0f} m "
                    f"away at {nlat:.4f}, {nlon:.4f})"
                )
            e_row, e_col = new_row, new_col

        # Start nudge (same logic as v1)
        target_for_start = (int(components[e_row, e_col])
                            if components is not None else None)
        nudge_used = False
        need_start_nudge = (
            nogo[s_row, s_col] or
            (components is not None and target_for_start is not None
             and components[s_row, s_col] != target_for_start)
        )
        if need_start_nudge:
            radius_cells = int(math.ceil(
                START_NUDGE_RADIUS_M / self.graph.resolution_m))
            nudged = _nudge_to_water(
                nogo, s_row, s_col, radius_cells,
                components=components, target_component=target_for_start)
            if nudged is None:
                any_water = _nudge_to_water(nogo, s_row, s_col, radius_cells)
                if any_water is None:
                    return {"error": "start_landlocked",
                            "error_result": RouteResult(
                                ok=False,
                                error="start is not on water and no nearby water",
                                runtime_s=time.perf_counter() - t0)}
                return {"error": "start_disconnected",
                        "error_result": RouteResult(
                            ok=False,
                            error=("start is in a water basin disconnected from "
                                   "the destination; no route possible at this "
                                   "chart resolution"),
                            runtime_s=time.perf_counter() - t0)}
            new_row, new_col = nudged
            if (new_row, new_col) != (s_row, s_col):
                dist_m = math.hypot((new_row - s_row), (new_col - s_col)) \
                    * self.graph.resolution_m
                nudge_used = True
                nudged_lat, nudged_lon = grid_to_latlon(
                    new_row, new_col, self.graph.transform)
                warnings_out.append(
                    f"start nudged to nearest water ({dist_m:.0f} m away "
                    f"at {nudged_lat:.4f}, {nudged_lon:.4f})"
                )
            s_row, s_col = new_row, new_col

        pad_cells = int(math.ceil(PRUNE_PAD_M / self.graph.resolution_m))
        row0, row1, col0, col1 = _prune_bbox(
            (s_row, s_col), (e_row, e_col), pad_cells, (H, W))
        sub_cost = np.ascontiguousarray(cost[row0:row1, col0:col1])
        return {
            "error": None,
            "s_row": s_row, "s_col": s_col,
            "e_row": e_row, "e_col": e_col,
            "bbox": (row0, row1, col0, col1),
            "sub_cost": sub_cost,
            "warnings": warnings_out,
            "nudge_used": nudge_used,
        }
