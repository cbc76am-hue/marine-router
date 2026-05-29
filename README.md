# marine-router

Chart-aware route planning for the Salish Sea (Puget Sound + San Juans, NOAA
ENC Region 15).  Resolves named destinations and computes navigable routes
that route around land, shallows, and charted hazards.  Designed to power
voice or chartplotter integrations where a human reviews the route before
navigating from it — **routes are drafts; this is not a substitute for chart
inspection or onboard judgment.**

## Capabilities

- **Distance-optimal A\*** (`optimize=safe`) — classic shortest-path over
  the navigability + cost grid.
- **Time-optimal spatio-temporal A\*** (`optimize=time` / `fuel`) — edge
  costs are seconds against the live NOAA CO-OPS tidal-current field,
  using a per-vessel polar (cruise + displacement fuel rates).
- **Departure-window optimization** (`optimize=depart_window`) — sweeps
  candidate departure times, returns the best one + alternatives.
  Parallelized across a process-pool of worker A* engines (3 by default).
- **Name-based destinations** — built-in 46-entry gazetteer of NOAA-charted
  harbor entrances + the operator's Signal K waypoints (passed as
  `extra_destinations`) + OpenStreetMap fallback for marine features
  outside the curated set.
- **Multi-leg routing** — `via_destinations` chains legs through named
  via-points, threading each leg's arrival time forward as the next leg's
  departure.
- **Swinomish channel auto-routing** — when start is inside the channel,
  picks north or south exit by tidal current direction at departure time.
- **Polar learner** — background SK subscriber fits a real fuel curve from
  engine CAN data, replacing the configured defaults.

## Architecture

| Module | Role |
|---|---|
| `tolly_router/enc.py` | Walk NOAA S-57 ENC cells; extract LNDARE, DEPARE, DRGARE, UWTROC, OBSTRN, WRECKS, RESARE |
| `tolly_router/nogo.py` | Per-class buffer + `shapely.ops.unary_union` of no-go polygons |
| `tolly_router/raster.py` | Rasterize to a 25 m UTM 10N navigability + cost grid |
| `tolly_router/coords.py` | WGS84 ↔ UTM 10N ↔ grid (row, col) helpers |
| `tolly_router/currents.py` | NOAA CO-OPS tidal-current fetcher + IDW field interpolator |
| `tolly_router/polar.py` | Vessel speed + fuel model (cruise / displacement) |
| `tolly_router/routing.py` | Grid A* + spatio-temporal A* + Swinomish-exit picker |
| `tolly_router/gazetteer.py` | Place-name → coords resolver (builtin + extras + suggestions) |
| `tolly_router/external_geocode.py` | OSM Nominatim wrapper for non-gazetteer marine features |
| `tolly_router/worker_pool.py` | ProcessPoolExecutor for parallel depart_window candidates |
| `tolly_router/service.py` | aiohttp HTTP service on `127.0.0.1:8090` |

## Quick start

Requires Python 3.10+, GDAL with S-57 driver, and a local copy of NOAA ENC charts.

```bash
# system deps (Debian/Ubuntu)
sudo apt install gdal-bin python3-gdal python3-shapely python3-pyproj \
                 python3-rasterio python3-numpy python3-scipy python3-aiohttp

# 1) parse charts → no-go GeoJSON  (~80 s for ~210 cells)
python3 scripts/build_nogo.py \
    --charts /path/to/NOAA/ENC/US_REGION15 \
    --out data/nogo.geojson

# 2) rasterize → graph.npz  (~7 s)
python3 scripts/build_graph.py

# 3) ad-hoc route via CLI
python3 scripts/route.py \
    --start 48.6300,-122.7850 \
    --end   48.5167,-122.6131

# 4) or run the HTTP service (default 127.0.0.1:8090)
python3 -m tolly_router.service
curl -sS http://127.0.0.1:8090/health
curl -sS -X POST http://127.0.0.1:8090/route \
    -H 'Content-Type: application/json' \
    -d '{"start":{"lat":48.6300,"lon":-122.7850},
         "end":{"lat":48.5167,"lon":-122.6131}}'

# acceptance suite (6 cases)
python3 scripts/route.py --suite
```

## Vessel parameters

Defaults are tuned for a small motoryacht with 3 ft draft + 5 ft safety margin
(min navigable depth 2.44 m). To use a different threshold, regenerate the
graph with the appropriate value:

```bash
python3 scripts/build_nogo.py --charts <...> --depth-threshold 3.0 \
    --out data/nogo_d300.geojson
python3 scripts/build_graph.py --depth-threshold 3.0 \
    --nogo-geojson data/nogo_d300.geojson
```

## HTTP API contract

`POST /route` — full schema (most fields optional):

```json
{
  "destination_name":   "Roche Harbor",
  "start_name":         "Swinomish North Exit",
  "start":              {"lat": 48.4045, "lon": -122.5062},
  "end":                {"lat": 48.5363, "lon": -123.0168},
  "extra_destinations": [{"name": "fish hole", "lat": 48.55, "lon": -122.95}],
  "via_destinations":   ["Sucia Island"],
  "departure_time":     "2026-05-19T18:00:00Z",
  "optimize":           "time",
  "depart_window":      {"earliest": "...", "latest": "...", "step_minutes": 15}
}
```

- If `destination_name` is set, it resolves against gazetteer → extras → OSM.
  Either name OR explicit `end` lat/lon is required.
- Same for `start_name` / `start`.
- `optimize`: `safe` | `time` | `fuel` | `depart_window`.
- `via_destinations` works with `time` and `fuel` (not `depart_window`).

200 success (single route, modes `safe`/`time`/`fuel`):
```json
{
  "ok": true,
  "waypoints": [{"lat": ..., "lon": ..., "name": "Start"}, ...],
  "distance_nm": 10.804,
  "duration_minutes": 47.2,
  "fuel_gallons": 23.1,
  "arrival_time": "2026-05-19T18:47:12Z",
  "departure_time": "2026-05-19T18:00:00Z",
  "auto_exit": "north",
  "hazards_near": 2,
  "warnings": ["..."],
  "optimize_mode": "time"
}
```

200 success (`depart_window` mode) — `best` plus stripped `alternatives` list.

200 recoverable failure (`ok=false`):
- `"destination is not on water"`
- `"start is in a water basin disconnected from the destination"`
- `"coordinates outside routing coverage area"`
- `"no navigable path found"`
- `"route distance exceeds 100 nm limit"`
- `"tidal-current data unavailable: ..."` (offline + cache cold)

400 with structured error + suggestions:
```json
{
  "error": "unknown destination: 'rosche'",
  "suggestions": ["Roche Harbor", "Reid Harbor"],
  "external_candidates": [{"name": "Lopez Pass", "lat": ..., "lon": ...}]
}
```

`GET /health` — liveness + graph metadata + version.
`POST /reload` — async swap to a freshly-built `data/graph.npz`.

## Resolution + scope tuning

The default graph build is 50 m / scope WGS84 (-124.5, 47.0, -122.0, 49.0).
That keeps the resident memory ~80 MB and per-route A* under a couple of
seconds. For tighter nearshore routing — narrow maintained channels like
Swinomish, route planning from a marina slip — rebuild at 25 m:

```bash
python3 scripts/build_graph.py --resolution 25
```

Cost: graph build is ~13 s instead of 7 s; resident memory rises to
~340 MB; per-route A* on wrap-around routes (HOME → Padilla via the
south of Whidbey) is 5-15 s. Worth it if you boat from a slip; not
necessary if you only route from offshore start points.

The `PRUNE_PAD_M = 25000.0` and `START_NUDGE_RADIUS_M = END_NUDGE_RADIUS_M =
3000.0` defaults are sized for the Salish Sea: 25 km prune accommodates
wrap-around-Whidbey routes; 3 km nudge accommodates marina slips. Tune
in `tolly_router/routing.py` for other scopes.

## Known limitations

- **Sub-25m channels can't be routed.** Notable casualties: the Ballard
  Locks + Lake Washington Ship Canal (Salmon Bay → Fremont Cut → Lake
  Union) are below grid resolution, so destinations past the locks are
  unreachable from salt water.  Plan to Shilshole and navigate the locks
  + canal manually.  The Swinomish channel has the same issue but the
  router auto-handles it by routing from one of the channel exits.

- **Tide heights ignored.** Depths are at chart datum (MLLW). The router
  consults tidal CURRENTS for time-mode optimization but not tide
  HEIGHTS — so it won't tell you a passage is closed at low water.

- **Scope is Puget Sound + San Juans** (lon -124.5 to -122.0, lat 47.0 to
  49.0). Cross-border (US/Canada) routing is out of scope at the graph
  level, but the gazetteer + external geocoder can resolve names like
  Bedwell Harbour; routes to BC destinations that exit the scope bbox
  will return "outside routing coverage area".

- **No autopilot integration.** This service plans routes; it does not
  steer or hand off to autopilot. Output is a draft you inspect on a
  chartplotter before navigating from it.

## Polar learner

A small background service (`scripts/tolly_polar_learner.py`) listens to
Signal K for engine + speed data, logs samples to SQLite, and every 30 min
fits a fuel-rate curve that the routing engine uses for `optimize=time`
and `optimize=fuel` modes. Designed for the boat: the synthetic engine
emitter or a real CAN tap → SK → learner — same path either way.

What it produces:

- `~/.cache/marine-router/polar-log.db` — SQLite history (`samples` +
  `learned_runs`). WAL mode so you can query it live.
- `~/.cache/marine-router/polar-learned.json` — the active learned polar.
  Rewritten atomically after each successful fit cycle.

Precedence used by `VesselPolar.load()` (per-key merge, highest wins):

1. `~/.config/marine-router/polar.json` — hand-edited override.
2. `~/.cache/marine-router/polar-learned.json` — learner output.
3. Built-in defaults (cruise 14 kt / 28 gph, displacement 7.5 kt / 8 gph).

Inspect the most recent learned run:

```bash
sqlite3 ~/.cache/marine-router/polar-log.db \
  "SELECT datetime(ts,'unixepoch'), n_samples,
          displacement_fuel_gph, cruise_fuel_gph
   FROM learned_runs ORDER BY ts DESC LIMIT 5"

cat ~/.cache/marine-router/polar-learned.json
```

Pause / disable the service:

```bash
systemctl --user stop tolly-polar-learner.service       # one-shot
systemctl --user disable --now tolly-polar-learner.service  # don't auto-start
```

The service config lives at `~/.config/marine-router/learner.json`
(optional — sensible defaults are baked in). The Signal K admin token
goes in `~/.config/marine-router/sk-token` (mode 600).

## Used by

- [boat-voice](https://github.com/cbc76am-hue/boat-voice) — a Whisper +
  Claude + Piper voice assistant for a 1974 Tollycraft 34 — calls this
  service for chart-aware route planning by name ("plan a route to Friday
  Harbor with a stop at Sucia").

## Status

Single-developer hobby project. Built against NOAA ENC charts current to
2026-05. Acceptance suite at `scripts/route.py --suite`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, style notes,
and open work.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
