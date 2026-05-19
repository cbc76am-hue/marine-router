#!/usr/bin/env python3
"""Polar learner — subscribes to engine + nav data on Signal K, logs samples,
fits a fuel-rate curve every 30 min, and writes the result where
``tolly_router.polar`` can pick it up.

Run as a long-running systemd user service.  See the unit file in the commit
message that ships this script (``~/.config/systemd/user/tolly-polar-learner.service``).

Config file
-----------
``~/.config/marine-router/learner.json`` (all keys optional)::

    {
        "sk_ws_url":     "ws://127.0.0.1:3000/signalk/v1/stream?subscribe=none",
        "sk_token_file": "/home/boat/.config/marine-router/sk-token",
        "db_path":       "/home/boat/.cache/marine-router/polar-log.db",
        "learned_path":  "/home/boat/.cache/marine-router/polar-learned.json",
        "subscribe_period_ms": 5000,
        "sample_max_age_s":    10,
        "fit_interval_s":      1800,
        "fit_window_hours":    168
    }

Tokens go in a separate file (mode 600).  The script itself stays in the
repo so the file path can be tracked and reviewed without leaking the token.

Env-var equivalents:
    POLAR_LEARNER_SK_WS_URL
    POLAR_LEARNER_SK_TOKEN_FILE
    POLAR_LEARNER_DB_PATH
    POLAR_LEARNER_LEARNED_PATH

Subscribes to these Signal K paths at ``period: 5000``:

    navigation.speedOverGround           m/s   -> kts (* 1.94384)
    propulsion.port.revolutions          Hz    -> RPM (* 60)
    propulsion.starboard.revolutions     Hz    -> RPM (* 60)
    propulsion.port.fuel.rate            m^3/s -> gph (* 951019.388)
    propulsion.starboard.fuel.rate       m^3/s -> gph
    propulsion.port.engineLoad           ratio (0-1)
    propulsion.starboard.engineLoad      ratio (0-1)

Sample insertion rule: a row is written when ALL of (sog, port_rpm,
port_fuel, stbd_rpm, stbd_fuel) have been received within
``sample_max_age_s`` of each other.  Loads are best-effort (allowed to be
missing/NULL).

Every ``fit_interval_s`` seconds, runs ``polar_log.fit_polar`` against the
most recent ``fit_window_hours`` of samples.  If the fit succeeds, writes
``learned_path`` atomically (tmp + rename) and inserts a ``learned_runs``
row for history.

SIGTERM/SIGINT clean shutdown.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("tolly_polar_learner")

# Make the in-repo `tolly_router` package importable when this script is
# launched directly (e.g. via systemd ExecStart).  The repo root is the
# directory two levels up from this file: <repo>/scripts/<this>.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tolly_router import polar_log  # noqa: E402
from tolly_router.polar import MS_TO_KTS  # noqa: E402

try:
    import websocket  # python3-websocket (websocket-client 1.7.0)
    from websocket import WebSocketTimeoutException
except ImportError:
    websocket = None  # type: ignore
    WebSocketTimeoutException = TimeoutError  # type: ignore


# ----- unit conversions ----------------------------------------------------
HZ_TO_RPM = 60.0
# Mercury / Signal K canonical: m^3/s.  1 m^3 × 264.172 gal × 3600 s/h.
M3S_TO_GPH = 951019.388


# ----- defaults ------------------------------------------------------------
DEFAULTS = {
    "sk_ws_url": "ws://127.0.0.1:3000/signalk/v1/stream?subscribe=none",
    "sk_token_file": str(Path.home() / ".config/marine-router/sk-token"),
    "db_path": str(Path.home() / ".cache/marine-router/polar-log.db"),
    "learned_path": str(Path.home() / ".cache/marine-router/polar-learned.json"),
    "subscribe_period_ms": 5000,
    "sample_max_age_s": 10,
    "fit_interval_s": 1800,         # 30 min
    "fit_window_hours": 24 * 7,     # 7 days
}

CONFIG_PATH = Path(os.path.expanduser("~/.config/marine-router/learner.json"))


# ----- the 7 SK paths we listen to + how to convert each value ------------
# Each entry: SK path -> (state-key, converter)
SK_PATHS: dict[str, tuple[str, callable]] = {
    "navigation.speedOverGround":      ("sog_kts",        lambda v: v * MS_TO_KTS),
    "propulsion.port.revolutions":     ("port_rpm",       lambda v: v * HZ_TO_RPM),
    "propulsion.starboard.revolutions":("stbd_rpm",       lambda v: v * HZ_TO_RPM),
    "propulsion.port.fuel.rate":       ("port_fuel_gph",  lambda v: v * M3S_TO_GPH),
    "propulsion.starboard.fuel.rate":  ("stbd_fuel_gph",  lambda v: v * M3S_TO_GPH),
    "propulsion.port.engineLoad":      ("port_load",      lambda v: float(v)),
    "propulsion.starboard.engineLoad": ("stbd_load",      lambda v: float(v)),
}

# Required keys before we'll write a row.  Loads are nice-to-have.
REQUIRED_KEYS = ("sog_kts", "port_rpm", "stbd_rpm",
                 "port_fuel_gph", "stbd_fuel_gph")


# ----- config + token ------------------------------------------------------

def load_config() -> dict:
    """Merge defaults <- file <- env vars."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH) as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                cfg.update({k: v for k, v in blob.items() if k in cfg})
        except (OSError, json.JSONDecodeError) as e:
            LOGGER.warning("config %s unreadable (%s); using defaults",
                           CONFIG_PATH, e)
    # env-var overrides
    env_map = {
        "POLAR_LEARNER_SK_WS_URL":     "sk_ws_url",
        "POLAR_LEARNER_SK_TOKEN_FILE": "sk_token_file",
        "POLAR_LEARNER_DB_PATH":       "db_path",
        "POLAR_LEARNER_LEARNED_PATH":  "learned_path",
    }
    for env, key in env_map.items():
        v = os.environ.get(env)
        if v:
            cfg[key] = v
    return cfg


def load_token(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        LOGGER.warning("token load fail (%s); SK subscribe disabled", e)
        return None


# ----- atomic JSON write ---------------------------------------------------

def atomic_write_json(path: Path, blob: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2, sort_keys=True)
    tmp.replace(path)


# ----- SK WebSocket subscribe ---------------------------------------------

def open_sk(url: str, token: str | None) -> "websocket.WebSocket | None":
    if websocket is None:
        LOGGER.warning("websocket-client not installed; cannot subscribe")
        return None
    if token is None:
        LOGGER.warning("no SK token; cannot subscribe")
        return None
    try:
        ws = websocket.create_connection(
            url,
            header=[f"Authorization: Bearer {token}"],
            timeout=10,
        )
        LOGGER.info("SK connected %s", url)
        return ws
    except (OSError, websocket.WebSocketException) as e:
        LOGGER.warning("SK connect fail (%s)", e)
        return None


def send_subscribe(ws: "websocket.WebSocket", period_ms: int) -> bool:
    """Send a subscribe message for the 7 paths at the configured cadence."""
    sub = {
        "context": "vessels.self",
        "subscribe": [
            {"path": p, "period": period_ms, "format": "delta",
             "policy": "instant", "minPeriod": period_ms}
            for p in SK_PATHS
        ],
    }
    try:
        ws.send(json.dumps(sub))
        return True
    except (OSError, websocket.WebSocketException) as e:
        LOGGER.warning("subscribe send fail (%s)", e)
        return False


# ----- sample assembly -----------------------------------------------------

class SampleBuilder:
    """Rolling per-path state; emits a row when all required keys are fresh.

    A row is emitted when each required key has been seen within
    ``max_age_s`` of *the most recent received value*.  This means at a
    5 s subscribe period and max_age 10 s, any group of 7 values arriving
    within ~2 ticks of each other produces one row.

    To avoid spamming duplicate rows when one channel updates faster than
    the others, we also enforce a minimum emission gap (`min_emit_gap_s`).
    """

    def __init__(self, max_age_s: float, min_emit_gap_s: float = 4.0):
        self.max_age_s = max_age_s
        self.min_emit_gap_s = min_emit_gap_s
        self.state: dict[str, tuple[float, float]] = {}  # key -> (ts, value)
        self.last_emit_ts: float = 0.0

    def update(self, key: str, value: float) -> None:
        self.state[key] = (time.time(), value)

    def try_emit(self) -> dict | None:
        """Return a row dict if a complete sample is ready; else None.

        Side-effect: stamps ``last_emit_ts`` on emit.
        """
        now = time.time()
        if now - self.last_emit_ts < self.min_emit_gap_s:
            return None
        for key in REQUIRED_KEYS:
            entry = self.state.get(key)
            if entry is None:
                return None
            ts, _ = entry
            if now - ts > self.max_age_s:
                return None

        # All required are fresh — also pull in optional loads if fresh.
        row: dict = {"ts": int(now)}
        for key in REQUIRED_KEYS:
            ts, value = self.state[key]
            row[key] = value
        for opt_key in ("port_load", "stbd_load"):
            entry = self.state.get(opt_key)
            if entry is not None and now - entry[0] <= self.max_age_s:
                row[opt_key] = entry[1]
        self.last_emit_ts = now
        return row


# ----- delta parsing -------------------------------------------------------

def handle_delta(blob: dict, builder: SampleBuilder) -> None:
    """Apply any matching path updates from one delta message."""
    for upd in blob.get("updates", []) or []:
        for v in upd.get("values", []) or []:
            path = v.get("path")
            val = v.get("value")
            if path not in SK_PATHS or val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            key, conv = SK_PATHS[path]
            try:
                builder.update(key, conv(fval))
            except (TypeError, ValueError):
                continue


# ----- main loop -----------------------------------------------------------

_running = True


def _stop(*_: object) -> None:
    global _running
    _running = False


def run_fit_cycle(conn, cfg: dict) -> dict | None:
    """Read recent samples, fit, write learned JSON + history row.  Returns
    the fit dict on success; None if there weren't enough samples."""
    rows = polar_log.recent_samples(conn, hours=cfg["fit_window_hours"])
    fit = polar_log.fit_polar(rows)
    if fit is None:
        LOGGER.info("fit skipped — samples=%d (need >= 30 per bin)", len(rows))
        return None
    atomic_write_json(Path(cfg["learned_path"]), fit)
    polar_log.insert_learned_run(conn, fit)
    LOGGER.info(
        "fit OK disp=%.2fgph (n=%d) cruise=%.2fgph (n=%d) wrote=%s",
        fit['displacement_fuel_gph'], fit['_samples_displacement'],
        fit['cruise_fuel_gph'], fit['_samples_cruise'], cfg['learned_path'],
    )
    return fit


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("POLAR_LEARNER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    cfg = load_config()
    token = load_token(cfg["sk_token_file"])

    LOGGER.info(
        "starting db=%s learned=%s period=%dms fit_every=%ds",
        cfg["db_path"], cfg["learned_path"],
        cfg["subscribe_period_ms"], cfg["fit_interval_s"],
    )

    conn = polar_log.connect(Path(cfg["db_path"]))
    builder = SampleBuilder(max_age_s=cfg["sample_max_age_s"])

    ws: "websocket.WebSocket | None" = None
    last_fit_at = time.monotonic()  # don't fit immediately on boot
    backoff_s = 1.0

    while _running:
        # Connect (with backoff) if not connected
        if ws is None:
            ws = open_sk(cfg["sk_ws_url"], token)
            if ws is None:
                # Sleep with periodic _running checks so SIGTERM is responsive
                wait = backoff_s
                while wait > 0 and _running:
                    time.sleep(min(1.0, wait))
                    wait -= 1.0
                backoff_s = min(backoff_s * 2.0, 30.0)
                continue
            if not send_subscribe(ws, cfg["subscribe_period_ms"]):
                ws.close()
                ws = None
                continue
            backoff_s = 1.0  # reset on a successful connect+subscribe

        # Receive one message, with a short timeout so the fit-cycle clock
        # gets evaluated even when nothing is arriving from SK.
        raw: str | None = None
        try:
            ws.settimeout(2.0)
            raw = ws.recv()
        except WebSocketTimeoutException:
            raw = None
        except (OSError, websocket.WebSocketException) as e:
            LOGGER.warning("SK recv error (%s); reconnecting", e)
            ws.close()
            ws = None
            continue

        if raw:
            try:
                blob = json.loads(raw)
            except json.JSONDecodeError:
                blob = None
            if isinstance(blob, dict):
                handle_delta(blob, builder)
                row = builder.try_emit()
                if row is not None:
                    polar_log.insert_sample(conn, row)

        # Fit-cycle clock
        if time.monotonic() - last_fit_at >= cfg["fit_interval_s"]:
            run_fit_cycle(conn, cfg)
            last_fit_at = time.monotonic()

    # Clean shutdown
    if ws is not None:
        ws.close()
    conn.close()
    LOGGER.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
