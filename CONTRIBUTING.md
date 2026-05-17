# Contributing to marine-router

Thanks for your interest. This is a small open-source project; contributions
that improve route quality, broaden chart coverage, or add the planned v1.1+
features (tide-adjusted depths, tidal currents, weather routing) are
especially welcome. See the **Open work** section below for ideas.

## Ground rules

- **Routes are drafts.** This project plans navigable polylines for human
  review. PRs that move toward autopilot integration or remove the human-
  in-the-loop posture will not be merged.
- **No safety-of-life claims.** Don't add language to docs that implies the
  router is safe to follow without independent chart verification.
- **Apache 2.0.** By submitting a PR you agree your contribution is
  Apache-2.0 licensed. No CLA is required.

## Setting up a dev environment

```bash
# system deps (Debian / Ubuntu)
sudo apt install gdal-bin python3-gdal python3-shapely python3-pyproj \
                 python3-rasterio python3-numpy python3-scipy python3-aiohttp

# clone + install editable
git clone https://github.com/cbc76am-hue/marine-router.git
cd marine-router
pip install -e .
```

You'll need a local copy of NOAA Electronic Navigational Charts to build
the routing graph. Region 15 (Pacific Northwest) is the default scope; see
[NOAA's chart download portal](https://www.charts.noaa.gov/ENCs/ENCs.shtml).

## Running the acceptance suite

```bash
python3 scripts/build_nogo.py --charts /path/to/ENC/US_REGION15
python3 scripts/build_graph.py
python3 scripts/route.py --suite
```

The suite must pass 6/6 before a PR is mergeable. Output ends with
`PASS: all 6 acceptance cases met expectations.` on success.

## Style + design conventions

A few habits the codebase tries to follow:

- **No defensive coding against impossible states.** `try/except` should
  catch only at real boundaries (filesystem, network, parsing untrusted
  input). Don't wrap calls to internal helpers in broad excepts to "be safe."
- **Comments explain WHY, not WHAT.** If well-named code already says
  what it does, no comment is needed. Reserve comments for hidden
  constraints, subtle invariants, workarounds with citations.
- **No build-history or task-history references in source.** Things like
  `# Phase 3 added this` belong in commit messages, not code.
- **Match the existing module patterns.** The five core modules
  (`enc`, `nogo`, `raster`, `coords`, `routing`, `service`) each have a
  focused role; new functionality should fit one of them or get its own
  module rather than cross-cutting them.

## How to submit a PR

1. Fork the repo and create a topic branch off `main`.
2. Make the change. Run the acceptance suite locally.
3. If the change touches the routing engine, add or update a suite case
   in `scripts/route.py` that exercises it.
4. Open the PR against `main`. Describe the motivation, the approach, and
   any chart-data caveats. If the change affects route shape, include a
   before/after sample for one or two named routes (e.g. Shelter Bay →
   Friday Harbor) in the PR body.
5. Expect review notes; the reviewer may ask for narrower exception
   handling, fewer comments, or simpler abstractions per the style
   conventions above.

## Open work

The README's *Known limitations* section lists the biggest gaps. Concrete
PR-friendly items, in roughly increasing difficulty:

- **Coastline / channel fixtures.** A small curated set of canonical
  waypoints for narrow channels (Swinomish, Hood Canal sill, Deception
  Pass approach) that the router can use as breadcrumbs when the 50m
  raster severs them.
- **Adaptive resolution.** Cut the rasterizer at 25 m inside a hand-picked
  set of bounding boxes around known narrow passages.
- **Tide-adjusted depths (v1.1).** Pull NOAA CO-OPS station predictions,
  build a multi-threshold raster lookup, dispatch by departure_time.
- **Current-aware spatio-temporal A* (v2).** The Salish Sea killer
  feature; see plan doc fragments in commit history for the rough sketch.
- **Wind / wave gating (v2.1).** NWS NDFD + Wavewatch III as a "safe to
  leave at this time?" filter rather than a route shaper.

## Code of conduct

Be kind. Be specific. Assume good intent. If you're new to marine
navigation or to A* — both fine — say so in your PR; reviewers will help
orient.
