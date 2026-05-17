# marine-router

Chart-aware A* route planning for the Salish Sea (Puget Sound + San Juans, NOAA ENC Region 15).

Given a start and a destination on the water, returns a route as a list of (lat, lon)
waypoints that stays in water ≥8 ft deep, routes around land, and avoids charted
hazards (rocks, wrecks, restricted areas, submarine cables, traffic separation lanes).

Designed to power voice or chartplotter integrations where a human reviews the route
before navigating from it — **routes are drafts; this is not a substitute for
chart inspection or onboard judgment.**

## Architecture

Five phases, each a separate module:

| Module | Role |
|---|---|
| `tolly_router/enc.py` | Walk NOAA S-57 ENC cells; extract LNDARE, DEPARE, DRGARE, UWTROC, OBSTRN, WRECKS, RESARE |
| `tolly_router/nogo.py` | Per-class buffer + `shapely.ops.unary_union` of no-go polygons |
| `tolly_router/raster.py` | Rasterize to a 50 m UTM 10N navigability + cost grid |
| `tolly_router/coords.py` | WGS84 ↔ UTM 10N ↔ grid (row, col) helpers |
| `tolly_router/routing.py` | Grid A* with corner-cut block + Bresenham-validated Douglas-Peucker simplify |
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

`POST /route`

```json
{
  "start":          {"lat": 48.4045, "lon": -122.5062},
  "end":            {"lat": 48.5363, "lon": -123.0168},
  "vessel":         {"draft_m": 0.91, "safety_m": 1.52},   // optional
  "departure_time": "2026-05-17T09:30:00-07:00",           // accepted, ignored in v1
  "optimize":       "safe"                                  // v1: only "safe"
}
```

200 success:
```json
{
  "ok": true,
  "waypoints": [{"lat": 48.4045, "lon": -122.5062, "name": "Start"}, ...],
  "distance_nm": 10.804,
  "min_depth_m": null,
  "hazards_near": 2,
  "warnings": ["..."]
}
```

200 recoverable failure (`ok=false` with a readable error string):
- `"destination is not on water"`
- `"start is in a water basin disconnected from the destination; no route possible at this chart resolution"`
- `"coordinates outside routing coverage area"`
- `"no navigable path found"`
- `"route distance exceeds 100 nm limit"`

400 only for malformed JSON, missing required fields, or `optimize` not in the
v1 supported set.

`GET /health` — liveness + graph metadata.
`POST /reload` — async swap to a freshly-built `data/graph.npz`.

## Known limitations

- **50 m grid loses narrow channels.** Some maintained channels (e.g. Swinomish
  out of Padilla Bay) are <50 m wide at points, which severs them in the
  rasterized graph. Affected routes return `"start is in a water basin
  disconnected from the destination"` rather than producing an unsafe path.
  Adaptive resolution (25 m in known narrow passages) is a follow-up.

- **Tides ignored.** Depths are at chart datum (MLLW). The service API
  accepts `departure_time` for forward compatibility; v1 does not consult tide
  predictions.

- **Currents ignored.** The router minimizes distance, not time. The Salish
  Sea has 4-8 kt tidal currents at narrows; a time-optimal route may differ
  substantially from a distance-optimal route. Current-aware spatio-temporal
  A* is the planned v2 feature.

- **Scope is Puget Sound + San Juans** (lon -124.5 to -122.0, lat 47.0 to
  49.0). Cross-border (US/Canada) routing is out of scope.

- **No autopilot integration.** This service plans routes; it does not steer
  or hand off to autopilot. Use the output as a draft you inspect on a
  chartplotter.

## Status

v1, single-developer hobby project. Built against NOAA ENC charts current to
2026-05. Acceptance suite passes 6/6 (`scripts/route.py --suite`).
