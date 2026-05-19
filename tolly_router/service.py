"""aiohttp HTTP service wrapping ``Router``.

Endpoints:

* ``POST /route``    — plan a route.  200 on success or recoverable error.
                       400 only for malformed JSON / missing fields / unsupported
                       ``optimize`` value.
* ``GET  /health``   — liveness + graph metadata.  Always 200; ``ok`` boolean.
* ``POST /reload``   — re-read ``data/graph.npz`` from disk in the background.
                       Returns 202 immediately; new ``Router`` swaps in atomically.

One Python process, one ``Router`` instance, shared across handlers.
``Router.route`` is CPU-bound; we run it via ``asyncio.to_thread`` so the
event loop stays responsive.  The ``Router`` reference is guarded by an
``asyncio.Lock`` during ``/reload``; reads pick up the new instance, in-flight
``route`` calls keep their captured reference.  Listens on ``127.0.0.1:8090``;
local-only, no auth.

Optimize modes:

* ``safe`` — distance-optimal A*; ignores currents.
* ``time`` / ``fuel`` — spatio-temporal A* against tidal currents.
* ``depart_window`` — sweep candidate departure times in ``time`` mode and
  return the best.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from .currents import CurrentField
from .external_geocode import lookup as external_lookup
from .gazetteer import Destination, load_destinations
from .polar import VesselPolar
from .routing import RouteResult, Router
from .worker_pool import RouteTimeWorkerPool

# OSM importance threshold.  Towns/cities score ~0.3+; natural features
# (passes, bays, points) score 0.05-0.2.  Since Nominatim's `bounded=1`
# with our Salish Sea viewbox already constrains results to the marine
# area, accept anything above noise (0.05).
EXTERNAL_AUTO_USE_MIN_IMPORTANCE = 0.05

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants / config
# --------------------------------------------------------------------------

# Default to a path relative to this file so the service works from any
# checkout.  Override at runtime with TOLLY_ROUTER_GRAPH=<path> or
# make_app(graph_path=).
DEFAULT_GRAPH = Path(__file__).resolve().parent.parent / "data" / "graph.npz"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090

SUPPORTED_OPTIMIZE = {"safe", "time", "fuel", "depart_window"}

# 24 candidates × ~6 s/route ≈ 2.5 min worst-case wall time for a 10 nm route.
DEPART_WINDOW_MAX_SWEEPS = 24

# Max concurrent A* sweep candidates.  The route_time hot loop is mixed
# Python heap + numpy; numpy releases the GIL on tensor ops but the heap
# manipulation does not.  3 concurrent workers on a 4-core laptop gives
# ~2-3× wall-time speedup vs serial without thrashing the rest of the
# system (boat-voice webserver, polar_learner, healthz polls).
DEPART_WINDOW_MAX_PARALLEL = int(
    os.environ.get("TOLLY_ROUTER_SWEEP_PARALLELISM", "3")
)

APP_STARTED_KEY = web.AppKey("started_monotonic", float)
# The Router reference lives inside a one-slot mutable container so the
# /reload swap doesn't mutate the aiohttp Application after startup (which
# triggers a DeprecationWarning in aiohttp 3.9+).
ROUTER_BOX_KEY = web.AppKey("router_box", dict)
GRAPH_PATH_KEY = web.AppKey("graph_path", Path)
LOCK_KEY = web.AppKey("reload_lock", asyncio.Lock)
LAST_DIAG_KEY = web.AppKey("last_diagnostics_box", dict)
WORKER_POOL_KEY = web.AppKey("worker_pool", RouteTimeWorkerPool)


def _get_router(app: web.Application) -> Optional[Router]:
    return app[ROUTER_BOX_KEY].get("router")


def _set_router(app: web.Application, router: Router) -> Optional[Router]:
    """Atomically swap the router reference; returns the previous one."""
    box = app[ROUTER_BOX_KEY]
    old = box.get("router")
    box["router"] = router
    return old


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _result_to_public_json(r: RouteResult) -> Dict[str, Any]:
    """Convert a ``RouteResult`` into the public JSON shape.

    Strips diagnostic fields (``runtime_s``, ``nodes_expanded``); those live
    behind ``GET /diagnostics``.  Time-aware fields are added only when the
    result populated them — ``safe`` responses are unchanged.
    """
    if not r.ok:
        # Per spec: failure is 200 with ok=false so the caller can render
        # the error to the user.  Include warnings if any (e.g. nudge note).
        out: Dict[str, Any] = {"ok": False, "error": r.error or "unknown error"}
        if r.warnings:
            out["warnings"] = list(r.warnings)
        if r.optimize_mode:
            out["optimize_mode"] = r.optimize_mode
        return out
    out = {
        "ok": True,
        "waypoints": [
            {"lat": w.lat, "lon": w.lon, "name": w.name}
            if w.name is not None
            else {"lat": w.lat, "lon": w.lon}
            for w in r.waypoints
        ],
        "distance_nm": round(r.distance_nm, 3),
        "min_depth_m": r.min_depth_m,
        "hazards_near": r.hazards_near,
        "warnings": list(r.warnings),
    }
    if r.optimize_mode is not None:
        out["optimize_mode"] = r.optimize_mode
        if r.departure_time is not None:
            out["departure_time"] = r.departure_time
        if r.arrival_time is not None:
            out["arrival_time"] = r.arrival_time
        if r.duration_minutes is not None:
            out["duration_minutes"] = r.duration_minutes
        if r.fuel_gallons is not None:
            out["fuel_gallons"] = r.fuel_gallons
        if r.legs is not None:
            out["legs"] = r.legs
    if r.auto_exit is not None:
        out["auto_exit"] = r.auto_exit
    return out


def _parse_extras(raw: Any) -> List[Destination]:
    """Coerce a JSON `extra_destinations` array into Destination dataclasses.

    Silently drops malformed entries — extras are user-saved waypoints,
    and a single bad row shouldn't fail the whole route.
    """
    out: List[Destination] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        try:
            lat = float(entry["lat"])
            lon = float(entry["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            continue
        aliases = entry.get("aliases") or ()
        if isinstance(aliases, list):
            aliases = tuple(str(a) for a in aliases)
        else:
            aliases = ()
        out.append(Destination(
            name=name.strip(),
            lat=lat,
            lon=lon,
            region=entry.get("region", "user"),
            aliases=aliases,
        ))
    return out


def _parse_iso8601(s: str) -> datetime:
    """Parse an ISO8601 timestamp, accepting trailing 'Z'.  Naive input is
    treated as UTC.  Raises ValueError on bad format."""
    if not isinstance(s, str):
        raise ValueError(f"expected ISO8601 string, got {type(s).__name__}")
    txt = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _result_to_diagnostics(r: RouteResult) -> Dict[str, Any]:
    return {
        "ok": r.ok,
        "error": r.error,
        "distance_nm": round(r.distance_nm, 3) if r.ok else None,
        "n_waypoints": len(r.waypoints),
        "hazards_near": r.hazards_near,
        "runtime_s": round(r.runtime_s, 4),
        "nodes_expanded": r.nodes_expanded,
    }


def _validate_latlon(obj: Any, field: str) -> tuple[float, float]:
    """Validate an inbound ``{"lat": ..., "lon": ...}`` object.  Raises
    ``web.HTTPBadRequest`` with a useful JSON body on failure."""
    if not isinstance(obj, dict):
        raise web.HTTPBadRequest(
            text=f'{{"error": "missing or invalid field: {field}"}}',
            content_type="application/json",
        )
    try:
        lat = float(obj["lat"])
        lon = float(obj["lon"])
    except (KeyError, TypeError, ValueError):
        raise web.HTTPBadRequest(
            text=f'{{"error": "field {field} must be {{lat, lon}} numbers"}}',
            content_type="application/json",
        )
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise web.HTTPBadRequest(
            text=f'{{"error": "field {field} lat/lon out of range"}}',
            content_type="application/json",
        )
    return lat, lon


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

async def handle_route(request: web.Request) -> web.Response:
    """``POST /route`` — plan a route.

    400 on bad input; otherwise 200 with ``ok`` boolean per the contract.
    """
    try:
        body = await request.json()
    except ValueError:
        return web.json_response(
            {"error": "request body is not valid JSON"}, status=400)

    if not isinstance(body, dict):
        return web.json_response(
            {"error": "request body must be a JSON object"}, status=400)

    # ---- name resolution ----
    # destination_name / start_name are optional shortcuts.  Resolution chain:
    #   1. builtin gazetteer (verified NOAA-chart coords)
    #   2. extras passed by caller (typically the operator's Signal K marks)
    #   3. external geocoder (Nominatim) — only auto-used if exactly one
    #      high-confidence in-area candidate; otherwise 400 with candidates
    gaz = load_destinations()
    extras = _parse_extras(body.get("extra_destinations") or [])
    extra_warnings: List[str] = []

    dest_name = body.get("destination_name")
    if dest_name:
        resolved_dest = gaz.resolve(dest_name, extras=extras)
        if resolved_dest is None:
            ext = await asyncio.to_thread(external_lookup, dest_name, 5)
            good = [c for c in ext
                    if c.importance >= EXTERNAL_AUTO_USE_MIN_IMPORTANCE]
            if len(good) == 1:
                c = good[0]
                body.setdefault("end", {"lat": c.lat, "lon": c.lon})
                extra_warnings.append(
                    f"destination resolved via OpenStreetMap: "
                    f"{c.short_name} at ({c.lat:.4f}, {c.lon:.4f}) — "
                    "verify on chart"
                )
                log.info("destination resolved externally: %r -> %s (%.4f,%.4f)",
                         dest_name, c.short_name, c.lat, c.lon)
            else:
                sugg = gaz.suggestions(dest_name, k=3, extras=extras)
                return web.json_response(
                    {"error": f"unknown destination: {dest_name!r}",
                     "suggestions": [s.name for s in sugg],
                     "external_candidates": [
                         {"name": c.short_name, "full_name": c.name,
                          "lat": c.lat, "lon": c.lon}
                         for c in good[:3]
                     ]},
                    status=400)
        else:
            body.setdefault(
                "end", {"lat": resolved_dest.lat, "lon": resolved_dest.lon})

    start_name = body.get("start_name")
    if start_name:
        resolved_start = gaz.resolve(start_name, extras=extras)
        if resolved_start is None:
            sugg = gaz.suggestions(start_name, k=3, extras=extras)
            return web.json_response(
                {"error": f"unknown start: {start_name!r}",
                 "suggestions": [s.name for s in sugg]},
                status=400)
        body.setdefault(
            "start", {"lat": resolved_start.lat, "lon": resolved_start.lon})

    if "start" not in body:
        return web.json_response({"error": "missing field: start"}, status=400)
    if "end" not in body:
        return web.json_response({"error": "missing field: end"}, status=400)

    try:
        start_lat, start_lon = _validate_latlon(body.get("start"), "start")
        end_lat, end_lon = _validate_latlon(body.get("end"), "end")
    except web.HTTPBadRequest as exc:
        # _validate_latlon raises with the proper body/content_type already.
        return exc

    optimize = body.get("optimize", "safe")
    if not isinstance(optimize, str):
        return web.json_response(
            {"error": "optimize must be a string"}, status=400)
    if optimize not in SUPPORTED_OPTIMIZE:
        return web.json_response(
            {"error": f"unknown optimize value: {optimize!r}"}, status=400)

    departure_time_raw = body.get("departure_time")
    depart_window = body.get("depart_window")

    router = _get_router(request.app)
    if router is None:
        # /health says ok=false; /route gives a clean 503 so callers can
        # differentiate "service up but graph missing" from "service down".
        return web.json_response(
            {"ok": False, "error": "graph not loaded"}, status=503)

    # ---- dispatch by optimize mode -------------------------------------
    t0 = time.perf_counter()

    if optimize == "safe":
        result: RouteResult = await asyncio.to_thread(
            router.route,
            (start_lat, start_lon),
            (end_lat, end_lon),
            departure_time=departure_time_raw,
            optimize=optimize,
        )
        wall_s = time.perf_counter() - t0
        request.app[LAST_DIAG_KEY].clear()
        request.app[LAST_DIAG_KEY].update({
            **_result_to_diagnostics(result),
            "wall_time_s": round(wall_s, 4),
            "request": {
                "start": {"lat": start_lat, "lon": start_lon},
                "end": {"lat": end_lat, "lon": end_lon},
                "optimize": optimize,
                "departure_time": departure_time_raw,
            },
        })
        log.debug("route safe %s: ok=%s err=%r dist=%.2f wps=%d nodes=%d wall=%.2fs",
                  f"{start_lat:.4f},{start_lon:.4f}->{end_lat:.4f},{end_lon:.4f}",
                  result.ok, result.error,
                  result.distance_nm if result.ok else 0.0,
                  len(result.waypoints), result.nodes_expanded, wall_s)
        for w in reversed(extra_warnings):
            result.warnings.insert(0, w)
        return web.json_response(_result_to_public_json(result), status=200)

    if optimize in ("time", "fuel"):
        # departure_time defaults to "now" if missing.
        try:
            dt = (_parse_iso8601(departure_time_raw)
                  if departure_time_raw else datetime.now(timezone.utc))
        except ValueError as e:
            return web.json_response(
                {"error": f"departure_time invalid: {e}"}, status=400)
        result = await asyncio.to_thread(
            router.route_time,
            (start_lat, start_lon),
            (end_lat, end_lon),
            departure_time=dt,
            optimize=optimize,
        )
        wall_s = time.perf_counter() - t0
        request.app[LAST_DIAG_KEY].clear()
        request.app[LAST_DIAG_KEY].update({
            **_result_to_diagnostics(result),
            "wall_time_s": round(wall_s, 4),
            "request": {
                "start": {"lat": start_lat, "lon": start_lon},
                "end": {"lat": end_lat, "lon": end_lon},
                "optimize": optimize,
                "departure_time": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        })
        for w in reversed(extra_warnings):
            result.warnings.insert(0, w)
        log.debug("route %s %s: ok=%s err=%r dist=%.2f dur=%.1fmin "
                  "wps=%d wall=%.2fs",
                  optimize,
                  f"{start_lat:.4f},{start_lon:.4f}->{end_lat:.4f},{end_lon:.4f}",
                  result.ok, result.error,
                  result.distance_nm if result.ok else 0.0,
                  result.duration_minutes or 0.0,
                  len(result.waypoints), wall_s)
        return web.json_response(_result_to_public_json(result), status=200)

    # optimize == "depart_window"
    if not isinstance(depart_window, dict):
        return web.json_response(
            {"error": "depart_window requires {earliest, latest, step_minutes}"},
            status=400)
    try:
        earliest = _parse_iso8601(depart_window["earliest"])
        latest = _parse_iso8601(depart_window["latest"])
    except (KeyError, ValueError) as e:
        return web.json_response(
            {"error": f"depart_window earliest/latest invalid: {e}"},
            status=400)
    try:
        step_minutes = int(depart_window.get("step_minutes", 15))
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "depart_window.step_minutes must be an integer"},
            status=400)
    if step_minutes < 1:
        return web.json_response(
            {"error": "step_minutes must be >= 1"}, status=400)
    if latest <= earliest:
        return web.json_response(
            {"error": "depart_window latest must be after earliest"}, status=400)

    span_min = (latest - earliest).total_seconds() / 60.0
    n_sweeps = int(span_min // step_minutes) + 1
    n_sweeps = max(1, min(n_sweeps, DEPART_WINDOW_MAX_SWEEPS))

    try:
        field = await asyncio.to_thread(CurrentField.from_default_catalog)
    except (ValueError, OSError) as e:
        return web.json_response(
            {"ok": False, "optimize_mode": "depart_window",
             "error": f"tidal-current data unavailable: {e}"},
            status=200)
    polar = VesselPolar.load()

    # Run sweep candidates across the worker-process pool.  Each worker is
    # a separate OS process so the GIL doesn't serialize them.  Falls back
    # to in-process serial if the pool failed to spawn.
    pool: RouteTimeWorkerPool | None = request.app.get(WORKER_POOL_KEY)
    times = [earliest + timedelta(minutes=i * step_minutes) for i in range(n_sweeps)]

    if pool is not None:
        log.info("depart_window: %d candidates via worker pool (n=%d)",
                 n_sweeps, pool.n_workers)
        candidate_results = await asyncio.gather(*(
            pool.route_time(
                (start_lat, start_lon),
                (end_lat, end_lon),
                dt_i,
                "time",
            )
            for dt_i in times
        ))
        results: List[Tuple[datetime, RouteResult]] = list(zip(times, candidate_results))
    else:
        log.warning("depart_window: worker pool unavailable, running serial")
        results = []
        for dt_i in times:
            ri = await asyncio.to_thread(
                router.route_time,
                (start_lat, start_lon),
                (end_lat, end_lon),
                departure_time=dt_i,
                optimize="time",
                current_field=field,
                polar=polar,
            )
            results.append((dt_i, ri))

    successes = [(dt_i, ri) for dt_i, ri in results if ri.ok]
    wall_s = time.perf_counter() - t0
    if not successes:
        # All failed (probably out-of-coverage / no path).  Surface the
        # first error so the caller has something to render.
        first_err = next(ri.error for _, ri in results if not ri.ok)
        request.app[LAST_DIAG_KEY].clear()
        request.app[LAST_DIAG_KEY].update({
            "ok": False, "wall_time_s": round(wall_s, 4),
            "candidates_evaluated": len(results),
            "first_error": first_err,
        })
        return web.json_response(
            {"ok": False, "optimize_mode": "depart_window",
             "candidates_evaluated": len(results),
             "error": first_err},
            status=200)

    # Best = smallest duration_minutes.  Ties broken by earlier departure.
    successes.sort(key=lambda x: (x[1].duration_minutes or float("inf"),
                                   x[0]))
    best_dt, best = successes[0]
    others = successes[1:]

    body_out = {
        "ok": True,
        "optimize_mode": "depart_window",
        "candidates_evaluated": len(results),
        "best": {
            "departure_time": best.departure_time,
            "arrival_time": best.arrival_time,
            "duration_minutes": best.duration_minutes,
            "fuel_gallons": best.fuel_gallons,
            "distance_nm": round(best.distance_nm, 3),
            "waypoints": [
                {"lat": w.lat, "lon": w.lon, "name": w.name}
                if w.name is not None
                else {"lat": w.lat, "lon": w.lon}
                for w in best.waypoints
            ],
            "hazards_near": best.hazards_near,
            "warnings": list(extra_warnings) + list(best.warnings),
        },
        "alternatives": [
            {"departure_time": ri.departure_time,
             "duration_minutes": ri.duration_minutes,
             "fuel_gallons": ri.fuel_gallons}
            for _, ri in others
        ],
    }
    if best.auto_exit is not None:
        body_out["best"]["auto_exit"] = best.auto_exit
        body_out["auto_exit"] = best.auto_exit
    if best.legs is not None:
        body_out["best"]["legs"] = best.legs

    request.app[LAST_DIAG_KEY].clear()
    request.app[LAST_DIAG_KEY].update({
        "ok": True,
        "wall_time_s": round(wall_s, 4),
        "candidates_evaluated": len(results),
        "best_departure_time": best.departure_time,
        "best_duration_minutes": best.duration_minutes,
        "request": {
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "optimize": "depart_window",
            "earliest": earliest.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "latest": latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step_minutes": step_minutes,
        },
    })
    log.debug("depart_window %s: n=%d best=%s dur=%.1fmin wall=%.2fs",
              f"{start_lat:.4f},{start_lon:.4f}->{end_lat:.4f},{end_lon:.4f}",
              len(results), best.departure_time,
              best.duration_minutes or 0.0, wall_s)
    return web.json_response(body_out, status=200)


async def handle_health(request: web.Request) -> web.Response:
    """``GET /health`` — liveness + graph metadata.

    Always 200 (HA-style); caller inspects the ``ok`` boolean.
    """
    router = _get_router(request.app)
    started: float = request.app[APP_STARTED_KEY]
    uptime_s = round(time.monotonic() - started, 2)

    if router is None:
        return web.json_response(
            {"ok": False, "graph_loaded": False, "uptime_s": uptime_s,
             "version": "v2", "error": "graph not loaded"},
            status=200,
        )

    g = router.graph
    body = {
        "ok": True,
        "graph_loaded": True,
        "scope_bbox": list(g.bbox_wgs84),
        "graph_built_at": g.built_at,
        "resolution_m": g.resolution_m,
        "depth_threshold_m": g.depth_threshold_m,
        "uptime_s": uptime_s,
        "version": "v2",
    }
    return web.json_response(body, status=200)


async def handle_reload(request: web.Request) -> web.Response:
    """``POST /reload`` — async reload of ``data/graph.npz``.

    Returns 202 immediately; the actual load happens in a background
    task.  New ``Router`` swaps in atomically under the reload lock.
    """
    app = request.app
    graph_path: Path = app[GRAPH_PATH_KEY]

    async def _do_reload() -> None:
        lock: asyncio.Lock = app[LOCK_KEY]
        if lock.locked():
            log.warning("reload already in progress; ignoring overlap")
            return
        async with lock:
            log.info("reload: loading graph from %s", graph_path)
            try:
                new_router = await asyncio.to_thread(Router.from_npz, graph_path)
            except (OSError, ValueError, KeyError) as exc:
                log.error("reload failed: %s", exc, exc_info=True)
                return
            old = _set_router(app, new_router)
            log.info("reload: graph swapped (shape=%s, built_at=%s); "
                     "old=%s", new_router.graph.shape,
                     new_router.graph.built_at, "dropped" if old else "n/a")

    # Schedule and return immediately.
    asyncio.create_task(_do_reload())
    return web.json_response(
        {"accepted": True, "graph_path": str(graph_path)}, status=202)


async def handle_diagnostics(request: web.Request) -> web.Response:
    """``GET /diagnostics`` — last route's diagnostic fields.

    In-memory single slot, no history.  Useful for debugging slow
    routes from the CLI without exposing diagnostics on the public
    /route response.
    """
    diag = request.app[LAST_DIAG_KEY]
    if not diag:
        return web.json_response(
            {"available": False,
             "note": "no route has been planned since startup"},
            status=200)
    return web.json_response({"available": True, **diag}, status=200)


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

async def _on_startup(app: web.Application) -> None:
    graph_path: Path = app[GRAPH_PATH_KEY]
    log.info("loading graph from %s ...", graph_path)
    t0 = time.perf_counter()
    try:
        router = await asyncio.to_thread(Router.from_npz, graph_path)
    except (OSError, ValueError, KeyError) as exc:
        # Startup must not abort — /health is documented to return ok=false
        # and /reload exists precisely to recover from this state.
        log.error("graph load failed at startup (%s); service starting "
                  "without a router. POST /reload after fixing %s.",
                  exc, graph_path, exc_info=True)
        return
    _set_router(app, router)
    log.info("graph loaded in %.2fs (shape=%s, navigable=%d cells, "
             "depth_thresh=%.2fm)",
             time.perf_counter() - t0, router.graph.shape,
             router.graph.cells_navigable, router.graph.depth_threshold_m)
    t1 = time.perf_counter()
    await asyncio.to_thread(router._ensure_components)
    log.info("connectivity precomputed in %.2fs", time.perf_counter() - t1)

    # Spawn the depart_window worker pool — each worker is a separate OS
    # process that pre-loads the graph + components + CurrentField, so
    # subsequent sweeps run truly in parallel (threads serialize on the
    # GIL inside the A* heap loop; processes don't).
    t2 = time.perf_counter()
    try:
        pool = await asyncio.to_thread(
            RouteTimeWorkerPool, graph_path, DEPART_WINDOW_MAX_PARALLEL,
        )
        app[WORKER_POOL_KEY] = pool
        log.info("worker pool ready in %.2fs (%d workers)",
                 time.perf_counter() - t2, pool.n_workers)
    except Exception as exc:
        # Service stays up even if the pool fails to spawn — depart_window
        # will fall back to in-process serial sweep with a warning.
        log.error("worker pool init failed (%s); depart_window will run serial",
                  exc, exc_info=True)


def make_app(graph_path: Path | None = None) -> web.Application:
    """Construct the aiohttp application.  Exposed for tests."""
    if graph_path is None:
        env_p = os.environ.get("TOLLY_ROUTER_GRAPH")
        graph_path = Path(env_p) if env_p else DEFAULT_GRAPH

    app = web.Application()
    app[APP_STARTED_KEY] = time.monotonic()
    app[GRAPH_PATH_KEY] = Path(graph_path)
    app[LOCK_KEY] = asyncio.Lock()
    app[ROUTER_BOX_KEY] = {}     # one-slot box; mutated in place, not the app
    app[LAST_DIAG_KEY] = {}

    app.router.add_post("/route", handle_route)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/reload", handle_reload)
    app.router.add_get("/diagnostics", handle_diagnostics)

    app.on_startup.append(_on_startup)
    return app


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=os.environ.get("TOLLY_ROUTER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.environ.get("TOLLY_ROUTER_HOST", DEFAULT_HOST)
    port = int(os.environ.get("TOLLY_ROUTER_PORT", str(DEFAULT_PORT)))
    app = make_app()
    log.info("starting tolly-router on http://%s:%d  graph=%s",
             host, port, app[GRAPH_PATH_KEY])
    web.run_app(app, host=host, port=port, print=None,
                access_log=logging.getLogger("aiohttp.access"))


if __name__ == "__main__":
    main()
