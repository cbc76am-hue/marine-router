"""aiohttp HTTP service wrapping ``Router``.

Endpoints (spec: /home/boat/MARINE_ROUTING_PLAN.md §6):

* ``POST /route``    — plan a route.  200 on success or recoverable error.
                       400 only for malformed JSON / missing fields / unsupported
                       ``optimize`` value.
* ``GET  /health``   — liveness + graph metadata.  Always 200; ``ok`` boolean.
* ``POST /reload``   — re-read ``data/graph.npz`` from disk in the background.
                       Returns 202 immediately; new ``Router`` swaps in atomically.

Design notes:

* One Python process, one ``Router`` instance, shared across handlers.
* ``Router.route`` is CPU-bound 1-3 s of pure-Python A*; we run it via
  ``asyncio.to_thread`` so the event loop stays responsive to ``/health`` etc.
* The ``Router`` reference is guarded by an ``asyncio.Lock`` during ``/reload``
  swap.  Reads after the swap pick up the new instance; in-flight ``route``
  calls keep their captured reference (no in-flight invalidation needed).
* Listens on ``127.0.0.1:8090``; local-only, no auth.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from .routing import RouteResult, Router

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

SUPPORTED_OPTIMIZE = {"safe"}
# Reject loudly — these are spec'd but not yet implemented.
RESERVED_OPTIMIZE = {"time", "fuel", "depart_window"}

APP_STARTED_KEY = web.AppKey("started_monotonic", float)
# The Router reference lives inside a one-slot mutable container so the
# /reload swap doesn't mutate the aiohttp Application after startup (which
# triggers a DeprecationWarning in aiohttp 3.9+).
ROUTER_BOX_KEY = web.AppKey("router_box", dict)
GRAPH_PATH_KEY = web.AppKey("graph_path", Path)
LOCK_KEY = web.AppKey("reload_lock", asyncio.Lock)
LAST_DIAG_KEY = web.AppKey("last_diagnostics_box", dict)


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
    """Convert a ``RouteResult`` into the public JSON shape (plan §6).

    Strips diagnostic fields (``runtime_s``, ``nodes_expanded``) — those
    live behind ``GET /diagnostics`` instead.
    """
    if not r.ok:
        # Per spec: failure is 200 with ok=false so the caller can render
        # the error to the user.  Include warnings if any (e.g. nudge note).
        out: Dict[str, Any] = {"ok": False, "error": r.error or "unknown error"}
        if r.warnings:
            out["warnings"] = list(r.warnings)
        return out
    return {
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
    if optimize in RESERVED_OPTIMIZE:
        return web.json_response(
            {"error": "optimize value not supported in v1"}, status=400)
    if optimize not in SUPPORTED_OPTIMIZE:
        return web.json_response(
            {"error": f"unknown optimize value: {optimize!r}"}, status=400)

    departure_time = body.get("departure_time")

    router = _get_router(request.app)
    if router is None:
        # /health says ok=false; /route gives a clean 503 so callers can
        # differentiate "service up but graph missing" from "service down".
        return web.json_response(
            {"ok": False, "error": "graph not loaded"}, status=503)

    # Router.route is CPU-bound 1-3 s; push it off the event loop.
    t0 = time.perf_counter()
    result: RouteResult = await asyncio.to_thread(
        router.route,
        (start_lat, start_lon),
        (end_lat, end_lon),
        departure_time=departure_time,
        optimize=optimize,
    )
    wall_s = time.perf_counter() - t0

    # Stash diagnostics for the /diagnostics endpoint (last route only).
    request.app[LAST_DIAG_KEY].clear()
    request.app[LAST_DIAG_KEY].update({
        **_result_to_diagnostics(result),
        "wall_time_s": round(wall_s, 4),
        "request": {
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "optimize": optimize,
            "departure_time": departure_time,
        },
    })

    log.info("route %s: ok=%s err=%r dist=%.2f wps=%d nodes=%d wall=%.2fs",
             f"{start_lat:.4f},{start_lon:.4f}->{end_lat:.4f},{end_lon:.4f}",
             result.ok, result.error,
             result.distance_nm if result.ok else 0.0,
             len(result.waypoints), result.nodes_expanded, wall_s)

    return web.json_response(_result_to_public_json(result), status=200)


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
             "version": "v1", "error": "graph not loaded"},
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
        "version": "v1",
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
