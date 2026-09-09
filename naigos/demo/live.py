"""Live simulation server: steps the env forever and streams it to a browser.

    python -m naigos.demo.live --aoi tehran_basin --checkpoint checkpoints/theatre_1000.pkl

Serves three things on http://localhost:8765 :

    GET /            the CesiumJS viewer
    GET /scene       static scene description (georef, terrain bounds, threats)
    GET /stream      Server-Sent Events, one JSON frame per sim tick

The simulation is genuinely live, not a replay. A worker thread steps
`NaigosEnv` continuously with the trained policy, and aircraft are re-tasked
through `env.respawn` the moment a sortie ends, so the theatre never empties.
Nothing is interpolated server-side -- the browser gets the logged state and
Cesium interpolates between samples for display.

Stdlib only: `ThreadingHTTPServer` plus SSE. A websocket framework would buy
nothing here (the stream is one-directional) and would be another dependency
between a reader and running the thing.

The globe is two layers and they are not interchangeable:

    imagery   Copernicus Sentinel-2 via Cesium ion (`naigos.demo.imagery`).
              Cosmetic. Never observed by the policy.
    terrain   `/terrain` -- the env's own heightmap, the surface every
              line-of-sight ray was computed against.

`--visual` picks between two postures, and the difference is exactly which of
those two supplies the surface:

    physics          the default, and the only evidence-grade one. Surface from
                     `/terrain`; skin from Sentinel-2 or OpenStreetMap.
    photorealistic   Google Photorealistic 3D Tiles through CesiumJS. The
                     tileset brings its own geometry, so the drawn surface stops
                     being the modelled one and nothing on screen is evidence
                     about terrain masking. `VisualConfig.evidence_grade` says so.

Credentials are read from explicit environment variables only --
NAIGOS_CESIUM_ION_TOKEN (or CESIUM_ION_TOKEN), NAIGOS_GOOGLE_MAPS_API_KEY (or
GOOGLE_MAPS_API_KEY) -- and the ion token is substituted into the page at serve
time. With none of them the viewer falls back to keyless OpenStreetMap over the
simulation's own terrain, and every measured claim is unchanged.
"""

from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..data.geodetic import GeoRef
from ..env.flight_env import NaigosEnv
from ..env import detection as det_mod
from ..env import terrain as terrain_mod
from ..env.threats import per_threat_params
from ..env.terrain import sample_height
from ..rl.ppo import greedy_policy
from . import imagery as imagery_mod

ASSETS = Path(__file__).parent / "assets"



def build_terrain_grid(hmap, tcfg, georef: GeoRef, geo_bounds: dict, n: int = 512):
    """Resample the env's OWN heightmap onto a regular lat/lon grid.

    Two choices here are load-bearing.

    Source: this samples `state.hmap` -- the surface the simulation actually
    computed every line-of-sight ray against -- NOT the cached 30 m DEM it was
    derived from. Sampling the raw DEM instead was measured to diverge from the
    modelled surface by up to 2094 m, which is exactly the defect (next-steps
    E-9) this endpoint exists to fix. Sampling the env grid gives mean 1.5 m /
    p95 8.6 m / max 40.7 m agreement at n=512 over tehran_basin.

    Frame: lat/lon rather than the native UTM/ENU, so the browser can do a
    plain bilinear lookup and needs no proj4. `tests/test_geodetic_live.py`
    asserts the Cesium page loads exactly one external script; shipping a
    projection library to the client would break that, and should.

    Edge handling: `sample_height` clamps outside the grid, so the skirt
    extends the boundary elevation outward. Filling with zero instead put a
    2 km bilinear seam along the AOI edge.
    """
    lons = np.linspace(geo_bounds["west"], geo_bounds["east"], n)
    lats = np.linspace(geo_bounds["south"], geo_bounds["north"], n)  # row 0 = south, as ENU
    lo, la = np.meshgrid(lons, lats)
    x, y = georef.from_wgs84(lo, la)

    h = np.asarray(sample_height(jnp.asarray(hmap), tcfg, jnp.asarray(x), jnp.asarray(y)))
    inside = (x >= 0) & (x <= tcfg.extent_x) & (y >= 0) & (y <= tcfg.extent_y)

    meta = {
        "west": geo_bounds["west"], "east": geo_bounds["east"],
        "south": geo_bounds["south"], "north": geo_bounds["north"],
        "nx": n, "ny": n,
        "min_m": float(h.min()), "max_m": float(h.max()),
        "inside_fraction": round(float(inside.mean()), 4),
        "cell_m": float(tcfg.cell),
        "grid": f"{tcfg.nx}x{tcfg.ny}",
    }
    return np.round(h).astype("<i2").tobytes(), meta

@dataclass
class Counters:
    """Cumulative tallies across the whole live run, not per episode."""

    sorties: int = 0
    reached: int = 0
    shot_down: int = 0
    terrain: int = 0
    bounds: int = 0
    fuel: int = 0

    def as_dict(self) -> dict:
        lost = self.shot_down + self.terrain + self.bounds + self.fuel
        done = self.reached + lost
        return {
            "sorties_launched": self.sorties,
            "objectives_reached": self.reached,
            "shot_down": self.shot_down,
            "terrain_losses": self.terrain,
            "bounds_losses": self.bounds,
            "fuel_losses": self.fuel,
            "success_rate": round(self.reached / done, 3) if done else None,
        }


@dataclass
class Simulation:
    """Owns the env and the worker thread. Publishes a snapshot other threads read."""

    env: NaigosEnv
    policy: object
    georef: GeoRef
    speed: float = 10.0          # sim seconds per wall-clock second
    use_cbf: bool = False
    seed: int = 0
    # Re-draw the threat field on this cadence. Without it the whole session
    # reports one threat layout: a benign draw showed 100% success against 55%
    # measured over 24 sampled layouts.
    reroll_s: float = 1200.0

    snapshot: dict = field(default_factory=dict)
    counters: Counters = field(default_factory=Counters)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        cfg = self.env.cfg
        self.key = jax.random.PRNGKey(self.seed)
        self.key, k0 = jax.random.split(self.key)
        self.state, self.obs = self.env.reset(k0)
        self.counters.sorties = cfg.n_blue
        self.sim_t = 0.0

        self._afilter = None
        if self.use_cbf:
            from ..rl.cbf import CBFConfig, make_policy_filter

            self._afilter = make_policy_filter(CBFConfig(), cfg)

        # jit once, up front, so the first frame is not a 2 s stall
        self._step = jax.jit(self._raw_step)
        self._respawn = jax.jit(self.env.respawn)
        self._reroll = jax.jit(self.env.reroll_threats)
        self._next_reroll = self.reroll_s
        self.threat_draws = 1
        self._terrain_cache = None  # static for a given cell_m; built on first request
        self._publish(np.ones(cfg.n_blue, dtype=bool), np.zeros(cfg.n_blue))

    def _raw_step(self, state, obs, key):
        k_pol, k_next = jax.random.split(key)
        action = self.policy(obs, k_pol)
        feasible = jnp.ones((self.env.cfg.n_blue,), dtype=bool)
        if self._afilter is not None:
            action, feasible = self._afilter(state, obs, action)
        state2, obs2, terms, done, info = self.env.step(state, action)
        return state2, obs2, terms, info, feasible, k_next

    # ------------------------------------------------------------------ loop
    def run(self):
        cfg = self.env.cfg
        dt_wall = cfg.dt / max(self.speed, 1e-6)
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self.key, k = jax.random.split(self.key)
            state2, obs2, terms, info, feasible, _ = self._step(self.state, self.obs, k)

            t = jax.device_get(terms)
            self.counters.shot_down += int(np.asarray(t.shotdown).sum())
            self.counters.terrain += int(np.asarray(t.terrain_violation).sum())
            self.counters.bounds += int(np.asarray(t.bounds_violation).sum())
            self.counters.fuel += int(np.asarray(t.out_of_fuel).sum())
            self.counters.reached += int(np.asarray(t.arrived).sum())

            self.state, self.obs = state2, obs2
            self.sim_t += cfg.dt

            # re-task anything that finished this tick
            done_mask = np.asarray(jax.device_get(info["agent_done"]))
            if done_mask.any():
                self.key, kr = jax.random.split(self.key)
                self.state = self._respawn(self.state, kr, jnp.asarray(done_mask))
                self.obs = self.env.observe(self.state)
                self.counters.sorties += int(done_mask.sum())

            if self.reroll_s > 0 and self.sim_t >= self._next_reroll:
                self.key, kt = jax.random.split(self.key)
                self.state = self._reroll(self.state, kt)
                self.obs = self.env.observe(self.state)
                self._next_reroll += self.reroll_s
                self.threat_draws += 1

            self._publish(np.asarray(jax.device_get(feasible)), np.asarray(t.exposure), done_mask)

            next_tick += dt_wall
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def stop(self):
        self._stop.set()

    # --------------------------------------------------------------- publish
    def _publish(self, feasible, exposure, retasked=None):
        cfg = self.env.cfg
        st = jax.device_get(self.state)
        pos = np.asarray(st.air.pos)
        lon, lat = self.georef.to_wgs84(pos[:, 0], pos[:, 1])
        psi = np.asarray(st.air.psi)
        lock = np.asarray(st.lock).max(axis=0)

        ground = np.asarray(sample_height(jnp.asarray(st.hmap), cfg.terrain, pos[:, 0], pos[:, 1]))

        tpos = np.asarray(st.threats.pos)
        tlon, tlat = self.georef.to_wgs84(tpos[:, 0], tpos[:, 1])

        # Which threat's ray to draw, and whether terrain is cutting it.
        # Selected by DETECTION PROBABILITY, not by lock. Lock decays toward zero
        # the moment an aircraft is masked, so selecting on it hid the ray in
        # exactly the situation the ray exists to show: the threat that would see
        # you if the ridge were not there. pd already carries the LOS term, so
        # fall back to geometric proximity when nothing can see the aircraft.
        lock_arr = np.asarray(st.lock)
        tparams = per_threat_params(cfg, st.threats.kind)
        det = det_mod.detection_probability(
            jnp.asarray(st.hmap), cfg.terrain, cfg.detection,
            jnp.asarray(pos), jnp.asarray(np.asarray(st.air.psi)),
            jnp.asarray(tpos), tparams, jnp.asarray(np.asarray(st.threats.active)),
        )
        pd_arr = np.asarray(det["pd"])
        slant = np.asarray(det["slant"])
        active = np.asarray(st.threats.active)
        # rank: any real detection wins; otherwise the nearest active threat
        rank = np.where(active[:, None], pd_arr + 1e-6 / np.maximum(slant, 1.0), -1.0)
        tracker = rank.argmax(axis=0)
        emitter = jnp.asarray(tpos[tracker] + np.array([0.0, 0.0, 10.0]))
        clearance = np.asarray(
            terrain_mod.los_clearance(
                jnp.asarray(st.hmap), cfg.terrain, cfg.detection, emitter, jnp.asarray(pos)
            )
        )
        trk_lon, trk_lat = self.georef.to_wgs84(tpos[tracker, 0], tpos[tracker, 1])

        obj = np.asarray(st.objective)
        olon, olat = self.georef.to_wgs84(obj[:, 0], obj[:, 1])

        frame = {
            "t": round(self.sim_t, 1),
            "aircraft": [
                {
                    "id": int(i),
                    "lon": round(float(lon[i]), 6),
                    "lat": round(float(lat[i]), 6),
                    "alt": round(float(pos[i, 2]), 1),
                    "agl": round(float(pos[i, 2] - ground[i]), 1),
                    # Cesium wants a compass heading; psi is CCW from east
                    "heading": round(float((90.0 - np.degrees(psi[i])) % 360.0), 1),
                    "speed": round(float(st.air.speed[i]), 1),
                    "fuel": round(float(st.air.fuel[i] / cfg.airframe.fuel_init), 3),
                    "lock": round(float(lock[i]), 3),
                    "exposure": round(float(exposure[i]), 3),
                    "alive": bool(st.alive[i]),
                    "reached": bool(st.reached[i]),
                    "cbf_infeasible": bool(not feasible[i]),
                    "objective": [round(float(olon[i]), 6), round(float(olat[i]), 6),
                                  round(float(obj[i, 2]), 1)],
                    "retasked": bool(retasked[i]) if retasked is not None else False,
                    # ray to draw: [lon, lat, alt, clearance_m, pd]. clearance < 0
                    # means terrain is cutting it -- the aircraft is masked from
                    # the one threat that would otherwise have the best look.
                    "tracker": [
                        round(float(trk_lon[i]), 6), round(float(trk_lat[i]), 6),
                        round(float(tpos[tracker[i], 2]), 1), round(float(clearance[i]), 1),
                        round(float(pd_arr[tracker[i], i]), 3),
                    ],
                }
                for i in range(cfg.n_blue)
            ],
            # only the movers need re-sending every tick
            "threats": [
                {"i": int(i), "lon": round(float(tlon[i]), 6), "lat": round(float(tlat[i]), 6),
                 "alt": round(float(tpos[i, 2]), 1)}
                for i in range(cfg.n_threat)
                if bool(st.threats.active[i]) and cfg.threat_kinds[int(st.threats.kind[i])].speed > 0
            ],
            "counters": {**self.counters.as_dict(), "threat_draws": self.threat_draws},
            # the viewer re-reads /scene when this changes, so a re-rolled field
            # does not leave stale envelopes drawn over the map
            "threat_draw": self.threat_draws,
        }
        with self._lock:
            self.snapshot = frame

    def latest(self) -> dict:
        with self._lock:
            return self.snapshot

    # --------------------------------------------------------------- terrain
    def terrain_grid(self, notes: dict, n: int = 512):
        if self._terrain_cache is None:
            self._terrain_cache = build_terrain_grid(
                np.asarray(self.state.hmap), self.env.cfg.terrain, self.georef,
                notes["geo_bounds"], n=n,
            )
        return self._terrain_cache


    # ----------------------------------------------------------------- scene
    def scene(self, notes: dict) -> dict:
        cfg = self.env.cfg
        st = jax.device_get(self.state)
        tpos = np.asarray(st.threats.pos)
        lon, lat = self.georef.to_wgs84(tpos[:, 0], tpos[:, 1])
        return {
            "theatre": notes["theatre"],
            "bounds": notes["geo_bounds"],
            "georef": notes["georef"],
            "extent_m": [cfg.terrain.extent_x, cfg.terrain.extent_y],
            "n_blue": cfg.n_blue,
            "dt_s": cfg.dt,
            "speed": self.speed,
            "cbf": self.use_cbf,
            "mode": "live",
            "assumptions": notes["assumptions"],
            "terrain": self.terrain_grid(notes)[1],
            "threats": [
                {
                    "i": int(i),
                    "lon": round(float(lon[i]), 6),
                    "lat": round(float(lat[i]), 6),
                    "alt": round(float(tpos[i, 2]), 1),
                    "label": cfg.threat_kinds[int(st.threats.kind[i])].label,
                    "lethal_m": float(cfg.threat_kinds[int(st.threats.kind[i])].lethal_range
                                      * cfg.red_lethal_scale),
                    "detect_m": float(cfg.threat_kinds[int(st.threats.kind[i])].detect_range
                                      * cfg.red_detect_scale),
                    "alt_max_m": float(cfg.threat_kinds[int(st.threats.kind[i])].alt_max),
                    "mobile": cfg.threat_kinds[int(st.threats.kind[i])].speed > 0,
                    "active": bool(st.threats.active[i]),
                }
                for i in range(cfg.n_threat)
            ],
        }


def replay_payload(path: Path, georef: GeoRef, cfg, aoi: str | None = None) -> dict:
    """Convert a recorded demo.json into the same geodetic shape the live stream
    emits, so one page renders both.

    The conversion happens here rather than in the browser for the same reason the
    terrain grid does: keeping projections server-side is what lets the Cesium page
    stay at exactly one external script.
    """
    d = json.loads(path.read_text())

    # A recording is positions in a LOCAL ENU frame. Rendering it against the
    # wrong theatre's georef puts the whole sortie on the wrong continent, and
    # nothing about the result looks broken. Refuse rather than guess.
    rec_theatre = d.get("theatre")
    if rec_theatre is None:
        raise SystemExit(
            f"{path} predates theatre tagging and cannot be placed on the globe safely.\n"
            f"Regenerate it: python -m naigos.demo.replay --checkpoint <ckpt>"
        )
    if aoi and rec_theatre != aoi:
        raise SystemExit(
            f"{path} was recorded on theatre {rec_theatre!r} but the server is running "
            f"{aoi!r}. Rendering it here would place every aircraft over the wrong ground.\n"
            f"Use: --aoi {rec_theatre} --replay {path}"
        )
    if d.get("georef"):
        georef = GeoRef(**d["georef"])   # the recording's own frame wins

    # The globe must show the surface THIS ROLLOUT flew over. Serving the live
    # env's terrain instead draws a landscape the recording never saw, and the
    # logged AGL then disagrees with the drawn ground by hundreds of metres.
    terrain = None
    rt = d.get("terrain")
    if rt and d.get("geo_bounds"):
        from ..env.config import TerrainConfig

        tcfg = TerrainConfig(nx=rt["nx"], ny=rt["ny"], cell=rt["cell_m"])
        hmap = np.asarray(rt["heights"], dtype=np.float32).reshape(rt["ny"], rt["nx"])
        terrain = build_terrain_grid(hmap, tcfg, georef, d["geo_bounds"])

    out = {"policies": [], "frames": {}, "dt_s": d.get("dt_s", cfg.dt),
           "theatre": rec_theatre, "cell_m": d.get("cell_m"),
           "_terrain": terrain, "_bounds": d.get("geo_bounds"),
           "_threats": d.get("threats"), "_threat_pos": None}
    for name, w in d["worlds"].items():
        pos = np.asarray(w["pos"])                    # (S, B, 3) local ENU
        alive = np.asarray(w["alive"])
        reached = np.asarray(w["reached"])
        lock = np.asarray(w["lock"])
        agl = np.asarray(w["agl"])
        lon, lat = georef.to_wgs84(pos[..., 0], pos[..., 1])
        out["policies"].append(name)
        out["frames"][name] = [
            [
                {
                    "id": b,
                    "lon": round(float(lon[t, b]), 6), "lat": round(float(lat[t, b]), 6),
                    "alt": round(float(pos[t, b, 2]), 1), "agl": round(float(agl[t, b]), 1),
                    "lock": round(float(lock[t, b]), 3),
                    "alive": bool(alive[t, b]), "reached": bool(reached[t, b]),
                }
                for b in range(pos.shape[1])
            ]
            for t in range(pos.shape[0])
        ]
    out["summaries"] = d.get("summaries", {})
    return out


def make_handler(sim: Simulation, notes: dict, ion_token: str | None,
                 replay: dict | None = None, visual=None,
                 google_api_key: str | None = None):
    """Build the request handler, baking the token and the visual config into the page.

    The token is substituted here, at serve time, from an environment variable --
    it is never written into `assets/cesium.html` and never committed. It travels
    through exactly one substitution point, `__ION_TOKEN__`, and the Google Maps
    key -- needed only on the `google_maps_api` tileset route -- through exactly
    one more, `__GOOGLE_API_KEY__`. The two config blobs below are credential-free
    by construction (`VisualConfig` holds booleans, not secrets), so the sensitive
    strings in this function are those two and they are greppable.

    The visual block is a separate substitution from the terrain endpoint on
    purpose: imagery is a cosmetic layer, terrain is the surface the model
    computed against, and the two must not be able to be confused for one another.
    """
    visual = visual or imagery_mod.resolve_visual_config(ion_token=ion_token)
    html = (ASSETS / "cesium.html").read_text().replace(
        "/*__ION_TOKEN__*/null", json.dumps(ion_token)
    ).replace(
        "/*__GOOGLE_API_KEY__*/null", json.dumps(google_api_key)
    ).replace(
        '/*__IMAGERY__*/{mode: "osm", osm_url: "https://tile.openstreetmap.org/"}',
        json.dumps(visual.to_page()),
    ).replace(
        '/*__VISUAL__*/{mode: "physics", evidence_grade: true}',
        json.dumps(visual.as_dict()),
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet; the sim loop owns stdout
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(html.encode(), "text/html; charset=utf-8")
            if self.path == "/scene":
                sc = sim.scene(notes)
                # The viewer states what it is showing, and whether that is
                # evidence. Credential-free: see VisualConfig.
                sc["visual"] = visual.as_dict()
                if replay is not None:
                    sc["mode"] = "replay"
                    sc["policies"] = replay["policies"]
                    sc["n_frames"] = len(replay["frames"][replay["policies"][0]])
                    sc["dt_s"] = replay["dt_s"]
                    if replay.get("_terrain"):
                        sc["terrain"] = replay["_terrain"][1]
                    if replay.get("_bounds"):
                        sc["bounds"] = replay["_bounds"]
                return self._send(json.dumps(sc).encode(), "application/json")
            if self.path == "/frames":
                if replay is None:
                    return self.send_error(404, "not running in replay mode")
                pub = {k: v for k, v in replay.items() if not k.startswith("_")}
                return self._send(json.dumps(pub).encode(), "application/json")
            if self.path == "/terrain":
                if replay is not None and replay.get("_terrain"):
                    return self._send(replay["_terrain"][0], "application/octet-stream")
                body, _ = sim.terrain_grid(notes)
                return self._send(body, "application/octet-stream")
            if self.path == "/stream":
                return self._stream()
            self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_t = None
            try:
                while True:
                    frame = sim.latest()
                    if frame and frame.get("t") != last_t:
                        last_t = frame["t"]
                        self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                        self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the tab closed; not an error

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="naigos.demo.live", description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/theatre_1000.pkl")
    ap.add_argument("--aoi", default="tehran_basin", help="component snapshot under components/aoi/")
    ap.add_argument("--threats", type=int, default=14)
    ap.add_argument("--cell-m", type=float, default=500.0,
                    help="terrain grid cell size in metres (finer = better looking globe)")
    ap.add_argument("--blue", type=int, default=6)
    ap.add_argument("--speed", type=float, default=10.0, help="sim seconds per wall-clock second")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cbf", action="store_true", help="run the HOCBF-QP backstop")
    ap.add_argument("--red-level", type=float, default=0.2, help="red curriculum level, 0-1")
    ap.add_argument("--reroll", type=float, default=1200.0,
                    help="re-draw the threat field every N sim seconds (0 disables)")
    ap.add_argument("--ion-token", default=None,
                    help="Cesium ion token. Prefer the env var NAIGOS_CESIUM_ION_TOKEN "
                         "(or CESIUM_ION_TOKEN); this flag only overrides it for one run.")
    # No --google-api-key flag on purpose: a key on argv is a key in the shell
    # history and in every `ps` listing. Environment variable only.
    ap.add_argument("--visual", choices=imagery_mod.VISUAL_MODES,
                    default=imagery_mod.DEFAULT_VISUAL_MODE,
                    help="physics (default): the simulation's own DEM under a Sentinel-2 or "
                         "OSM skin -- the only evidence-grade mode. photorealistic: Google "
                         "Photorealistic 3D Tiles via CesiumJS, which replaces the drawn "
                         "surface with the provider's geometry (needs NAIGOS_CESIUM_ION_TOKEN "
                         "or NAIGOS_GOOGLE_MAPS_API_KEY; falls back to physics without one).")
    ap.add_argument("--imagery", choices=imagery_mod.IMAGERY_MODES, default=None,
                    help="base-layer skin: Sentinel-2 via Cesium ion (needs a token) or "
                         "keyless OpenStreetMap. Defaults to sentinel2 under --visual physics "
                         "and osm under --visual photorealistic. In physics mode the terrain "
                         "comes from the simulation's DEM either way.")
    ap.add_argument("--replay", default=None,
                    help="scrub a recorded rollout (runs/demo/demo.json) instead of streaming live")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    # Validate the visual request before the DEM, the checkpoint and the JIT --
    # a run that spends 30 s starting up and then draws the wrong globe is worse
    # than one that refuses in the first millisecond. argparse's `choices` covers
    # each flag alone; this covers the combination.
    try:
        imagery_mod.validate_cli(a.visual, a.imagery)
    except imagery_mod.VisualConfigError as e:
        raise SystemExit(f"{ap.prog}: {e}")

    from ..env.theatre_bridge import describe, env_from_theatre
    from ..rl.red_team import RedCurriculum

    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, n_blue=a.blue, n_threat=a.threats, cell_m=a.cell_m)
    cfg = RedCurriculum().apply(cfg, a.red_level)
    print(describe(notes))

    ckpt = Path(a.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found. Train one, or use the shipped checkpoints/theatre_1000.pkl")
    with open(ckpt, "rb") as f:
        params = pickle.load(f)["actor"]

    env = NaigosEnv(cfg, hmap=hmap)
    sim = Simulation(
        env=env,
        policy=greedy_policy(params, cfg),
        georef=GeoRef(**notes["georef"]),
        speed=a.speed,
        use_cbf=a.cbf,
        reroll_s=a.reroll,
    )
    replay = None
    if a.replay:
        rp = Path(a.replay)
        if not rp.exists():
            raise SystemExit(f"{rp} not found. Generate it with `python -m naigos.demo.replay`.")
        replay = replay_payload(rp, sim.georef, cfg, aoi=a.aoi)
        print(f"replay: {rp} -- {len(replay['policies'])} policies x "
              f"{len(replay['frames'][replay['policies'][0]])} frames")
    else:
        threading.Thread(target=sim.run, daemon=True).start()

    # Credentials come from the environment (the ion token may be overridden for
    # a single run by --ion-token). Nothing token-shaped is written to disk, into
    # the page template, or into the VisualConfig below.
    ion_token = imagery_mod.resolve_ion_token(a.ion_token)
    google_api_key = imagery_mod.resolve_google_api_key()
    visual = imagery_mod.resolve_visual_config(
        a.visual, ion_token=ion_token, google_api_key=google_api_key, imagery=a.imagery,
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", a.port),
        make_handler(sim, notes, ion_token, replay, visual=visual,
                     google_api_key=google_api_key),
    )
    url = f"http://127.0.0.1:{a.port}/"
    mode = "replay" if replay else f"live, {a.speed:g}x real time"
    print(f"\n{url}   ({mode}, {cfg.n_blue} aircraft, "
          f"{cfg.n_threat_active} threats, CBF {'on' if a.cbf else 'off'})")
    # Two layers, said separately every time, because conflating them is the bug
    # this viewer already shipped once.
    print(f"terrain: the simulation's own {cfg.terrain.nx}x{cfg.terrain.ny} @ {cfg.terrain.cell:.0f} m "
          "heightmap, served to the globe -- what occludes on screen is what occluded in the model")
    print(imagery_mod.describe(visual))
    if visual.fallback_reason:
        print(f"visual: --visual {a.visual} unavailable -- {visual.fallback_reason}")
    if not visual.evidence_grade:
        # The one mode where the picture is not the model. Said on stdout as well
        # as in the HUD, so it is in the terminal scrollback of any screen capture.
        print(f"WARNING: {imagery_mod.PHOTOREALISTIC_EVIDENCE_WARNING}")
        # The viewer says the same thing, continuously, in a banner it cannot be
        # left without: the mode carries a runtime toggle back to physics terrain,
        # and lands there by itself if the provider fails.
        print("the browser shows a persistent visual-only banner in this mode, and the "
              "'physics terrain' button returns to the simulation's own surface")
    if visual.base_imagery == "sentinel2":
        print(f"attribution: {imagery_mod.SENTINEL2_ATTRIBUTION}")
    if visual.tileset_attribution:
        print(f"attribution: {visual.tileset_attribution}")
    print("ctrl-c to stop\n")
    if a.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        sim.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
