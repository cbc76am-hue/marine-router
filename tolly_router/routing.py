r"""Grid A* router on the navigability + cost surface.

Algorithm overview (per /home/boat/MARINE_ROUTING_PLAN.md §5):

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
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString

from .coords import (
    GridAffine,
    grid_to_latlon,
    haversine_nm,
    latlon_to_grid,
    latlon_to_utm,
    utm_grid_for_bbox,
    utm_to_latlon,
)
from .raster import COST_EDGE

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables (plan §5)
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
    """Douglas-Peucker simplify in UTM, then validate each surviving leg's
    Bresenham line against the full-grid ``nogo`` mask.  If a leg crosses
    blocked cells, splice in the original grid-path cells between the
    matching anchors so the public waypoint list never crosses land/hazards.
    """
    if len(path_local) <= 2:
        return _path_to_utm(path_local, transform, row_off, col_off)

    utm_pts = _path_to_utm(path_local, transform, row_off, col_off)
    simp_utm = _simplify_utm(utm_pts, tol_m)
    if len(simp_utm) <= 1:
        return simp_utm

    # Map each simplified UTM point back to its original path index.  DP
    # preserves the input vertices, so each simplified point matches one
    # cell-center in ``utm_pts`` within sub-meter tolerance.
    simp_idxs: List[int] = []
    last_i = 0
    for e, n in simp_utm:
        for i in range(last_i, len(utm_pts)):
            ee, nn = utm_pts[i]
            if abs(ee - e) < 0.5 and abs(nn - n) < 0.5:
                simp_idxs.append(i)
                last_i = i
                break

    # Validate every leg against the no-go raster; splice on failure.
    out_idxs: List[int] = [simp_idxs[0]]
    repaired = 0
    for prev_i, cur_i in zip(simp_idxs[:-1], simp_idxs[1:]):
        prev_cell = path_local[prev_i]
        cur_cell = path_local[cur_i]
        if _line_clear_grid(prev_cell, cur_cell, nogo, row_off, col_off):
            out_idxs.append(cur_i)
        else:
            # Bad leg — fall back to the original grid path between these
            # anchors.  This keeps the route navigable at the cost of more
            # waypoints around the offending segment.
            out_idxs.extend(range(prev_i + 1, cur_i + 1))
            repaired += 1

    if repaired:
        log.debug("simplify: repaired %d leg(s) that crossed no-go cells "
                  "(%d -> %d waypoints)",
                  repaired, len(simp_idxs), len(out_idxs))

    return [utm_pts[i] for i in out_idxs]


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
              departure_time: Optional[str] = None,   # accepted, ignored v1
              optimize: str = "safe",                  # accepted, ignored v1
              ) -> RouteResult:
        """Plan a route from ``start`` to ``end``.  Each endpoint is a
        ``(lat, lon)`` tuple.

        ``departure_time`` and ``optimize`` are accepted but currently
        ignored — see /home/boat/MARINE_ROUTING_PLAN.md §6, §13.
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
