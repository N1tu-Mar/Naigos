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
from ..rl.ppo import greedy_policy

ASSETS = Path(__file__).parent / "assets"


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

        from ..env.terrain import sample_height

        ground = np.asarray(sample_height(jnp.asarray(st.hmap), cfg.terrain, pos[:, 0], pos[:, 1]))

        tpos = np.asarray(st.threats.pos)
        tlon, tlat = self.georef.to_wgs84(tpos[:, 0], tpos[:, 1])

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
                    "objective": [round(float(olon[i]), 6), round(float(olat[i]), 6)],
                    "retasked": bool(retasked[i]) if retasked is not None else False,
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
            "assumptions": notes["assumptions"],
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


def make_handler(sim: Simulation, notes: dict, ion_token: str | None):
    html = (ASSETS / "cesium.html").read_text().replace(
        "/*__ION_TOKEN__*/null", json.dumps(ion_token)
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
                return self._send(json.dumps(sim.scene(notes)).encode(), "application/json")
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
    ap.add_argument("--blue", type=int, default=6)
    ap.add_argument("--speed", type=float, default=10.0, help="sim seconds per wall-clock second")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cbf", action="store_true", help="run the HOCBF-QP backstop")
    ap.add_argument("--red-level", type=float, default=0.2, help="red curriculum level, 0-1")
    ap.add_argument("--reroll", type=float, default=1200.0,
                    help="re-draw the threat field every N sim seconds (0 disables)")
    ap.add_argument("--ion-token", default=None, help="Cesium ion token (optional; OSM used without)")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    from ..env.theatre_bridge import describe, env_from_theatre
    from ..rl.red_team import RedCurriculum

    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, n_blue=a.blue, n_threat=a.threats)
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
    worker = threading.Thread(target=sim.run, daemon=True)
    worker.start()

    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sim, notes, a.ion_token))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"\nlive on {url}   ({a.speed:g}x real time, {cfg.n_blue} aircraft, "
          f"{cfg.n_threat_active} threats, CBF {'on' if a.cbf else 'off'})")
    if not a.ion_token:
        print("no --ion-token: using OpenStreetMap imagery on the WGS84 ellipsoid.\n"
              "  pass --ion-token to get Cesium World Terrain (free key at ion.cesium.com).")
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
