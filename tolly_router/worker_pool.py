"""ProcessPoolExecutor-based worker pool for depart_window sweep candidates.

The single-route A* path is GIL-bound — the inner loop's Python heap ops
dominate, so ``asyncio.to_thread`` doesn't actually parallelize sweep
candidates (verified: 5 candidates serially = 5 candidates via
``to_thread``).  Worker processes sidestep the GIL entirely.

Each worker pre-loads the graph + connectivity + CurrentField at spawn
time so subsequent ``route_time`` calls only pay the A* compute cost.
Used only by the ``depart_window`` path; single-route /route calls stay
in-process (no benefit to a pool for one call).
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from .routing import RouteResult

LOGGER = logging.getLogger(__name__)


# Module-level state per worker process.  Each worker is a separate OS
# process so these globals are isolated; the parent process never reads them.
_WORKER_ROUTER: Any = None
_WORKER_FIELD: Any = None


def _worker_init(graph_path: str) -> None:
    """One-time init per worker process: load graph + components + field."""
    global _WORKER_ROUTER, _WORKER_FIELD
    from .currents import CurrentField
    from .routing import Router

    _WORKER_ROUTER = Router.from_npz(Path(graph_path))
    _WORKER_ROUTER._ensure_components()
    try:
        _WORKER_FIELD = CurrentField.from_default_catalog()
    except (ValueError, OSError) as e:
        # Cache cold + offline = no currents.  Workers will return ok=false
        # routes per route_time's existing handling.
        LOGGER.warning("worker: CurrentField init failed (%s); deferring", e)
        _WORKER_FIELD = None


def _worker_probe() -> bool:
    """Used during pool warm-up to confirm a worker has initialized."""
    return _WORKER_ROUTER is not None


def _worker_route_time(
    start: Tuple[float, float],
    end: Tuple[float, float],
    departure_time_iso: str,
    optimize: str,
) -> RouteResult:
    """Worker entry: route one candidate.  Returns a fully-populated
    RouteResult that pickles cleanly back to the parent."""
    if departure_time_iso.endswith("Z"):
        dt = datetime.fromisoformat(departure_time_iso.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(departure_time_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    # Re-load CurrentField lazily if init couldn't (offline at startup,
    # cache populated later).
    global _WORKER_FIELD
    if _WORKER_FIELD is None:
        try:
            from .currents import CurrentField
            _WORKER_FIELD = CurrentField.from_default_catalog()
        except (ValueError, OSError):
            pass
    return _WORKER_ROUTER.route_time(
        start, end,
        departure_time=dt,
        optimize=optimize,
        current_field=_WORKER_FIELD,
    )


class RouteTimeWorkerPool:
    """Async wrapper around a ProcessPoolExecutor specialized for route_time."""

    def __init__(self, graph_path: Path, n_workers: int = 3):
        self._executor = cf.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(str(graph_path),),
        )
        self._n_workers = n_workers
        LOGGER.info("worker pool: spawning %d workers (graph=%s)",
                    n_workers, graph_path)
        # Force-warm all workers — ProcessPoolExecutor spawns lazily on
        # first submit, but we want all the graph-load cost paid up front.
        probes = [self._executor.submit(_worker_probe) for _ in range(n_workers)]
        for p in probes:
            p.result(timeout=120)
        LOGGER.info("worker pool: %d workers ready", n_workers)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    async def route_time(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        departure_time: datetime,
        optimize: str = "time",
    ) -> RouteResult:
        """Submit one candidate to the pool, await result.  Concurrent
        calls are scheduled across workers automatically."""
        future = self._executor.submit(
            _worker_route_time, start, end,
            departure_time.isoformat(), optimize,
        )
        return await asyncio.wrap_future(future)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
