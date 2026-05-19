"""Place-name → coordinates resolution for Salish Sea destinations.

The router accepts numeric coordinates as the wire contract, but callers
(LLMs especially) hallucinate harbor coordinates with depressing
frequency.  This module loads a curated list of NOAA-charted entrance
positions from ``data/destinations.json`` and resolves spoken names + a
small alias set to verified coords.

Public API:
    load_destinations(path=None) -> list[Destination]
    resolve(name) -> Destination | None
    suggestions(query, k=3) -> list[Destination]
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "destinations.json"


@dataclass(frozen=True)
class Destination:
    name: str
    lat: float
    lon: float
    region: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = ()

    def matches(self, query_norm: str) -> bool:
        """True if the normalized query equals this destination's name or any alias."""
        if query_norm == self.name.lower():
            return True
        return query_norm in (a.lower() for a in self.aliases)


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace for matching."""
    return " ".join(s.lower().split())


class Gazetteer:
    """In-memory catalog of named destinations."""

    def __init__(self, destinations: List[Destination]):
        self.destinations = destinations
        # Pre-build a flat lookup: normalized-name/alias -> Destination
        self._lookup: dict[str, Destination] = {}
        for d in destinations:
            self._lookup[d.name.lower()] = d
            for alias in d.aliases:
                # Aliases can collide; first wins (canonical destination order
                # in JSON is intentional — keep it stable).
                self._lookup.setdefault(alias.lower(), d)
        self._all_keys = list(self._lookup.keys())

    def resolve(
        self,
        name: str,
        extras: Optional[List[Destination]] = None,
    ) -> Optional[Destination]:
        """Return the Destination for an exact (case-insensitive) match on
        name or alias.  Checks the builtin catalog first, then the per-call
        ``extras`` list (typically user-saved Signal K waypoints).  None on
        miss."""
        if not name:
            return None
        norm = _normalize(name)
        hit = self._lookup.get(norm)
        if hit is not None:
            return hit
        if extras:
            for d in extras:
                if d.name.lower() == norm:
                    return d
                if any(a.lower() == norm for a in d.aliases):
                    return d
        return None

    def suggestions(
        self,
        query: str,
        k: int = 3,
        extras: Optional[List[Destination]] = None,
    ) -> List[Destination]:
        """Return up to k closest matches by string similarity.  Searches
        builtin catalog + per-call extras."""
        if not query:
            return []
        q = _normalize(query)
        # Build the key universe: builtin + extras' names + aliases.
        extras = extras or []
        extra_keys: dict[str, Destination] = {}
        for d in extras:
            extra_keys.setdefault(d.name.lower(), d)
            for a in d.aliases:
                extra_keys.setdefault(a.lower(), d)
        all_keys = list(self._all_keys) + list(extra_keys.keys())
        close = difflib.get_close_matches(q, all_keys, n=k, cutoff=0.5)
        seen: set[str] = set()
        out: List[Destination] = []
        for key in close:
            d = self._lookup.get(key) or extra_keys.get(key)
            if d is None or d.name in seen:
                continue
            seen.add(d.name)
            out.append(d)
        return out


# Module-level singleton, loaded lazily.
_GAZETTEER: Optional[Gazetteer] = None


def load_destinations(path: Optional[Path] = None) -> Gazetteer:
    """Load and cache the gazetteer.  Subsequent calls return the cached
    instance unless ``path`` is provided (forces reload — used by tests)."""
    global _GAZETTEER
    if path is not None or _GAZETTEER is None:
        fp = Path(path) if path is not None else _DATA_PATH
        with open(fp) as f:
            blob = json.load(f)
        items = blob.get("destinations", [])
        dests = []
        for item in items:
            dests.append(Destination(
                name=item["name"],
                lat=float(item["lat"]),
                lon=float(item["lon"]),
                region=item.get("region", ""),
                notes=item.get("notes", ""),
                aliases=tuple(item.get("aliases", [])),
            ))
        g = Gazetteer(dests)
        if path is None:
            _GAZETTEER = g
        log.info("gazetteer loaded: %d destinations from %s", len(dests), fp)
        return g
    return _GAZETTEER


def resolve(name: str) -> Optional[Destination]:
    """Convenience: load default gazetteer if needed, then resolve."""
    return load_destinations().resolve(name)


def suggestions(query: str, k: int = 3) -> List[Destination]:
    """Convenience: load default gazetteer if needed, then suggest."""
    return load_destinations().suggestions(query, k)
