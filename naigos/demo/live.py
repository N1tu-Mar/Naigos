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

`--visual` picks between three postures, and the difference is exactly which
of those two supplies the surface -- and what, if anything, stands on it:

    physics          the default, and the only evidence-grade one. Surface from
                     `/terrain`; skin from Sentinel-2 or OpenStreetMap.
    photorealistic   Google Photorealistic 3D Tiles through CesiumJS. The
                     tileset brings its own geometry, so the drawn surface stops
                     being the modelled one and nothing on screen is evidence
                     about terrain masking. `VisualConfig.evidence_grade` says so.
    urban-presentation
                     a dense 3D city for context: the provider's tiles when a
                     credential exists, else OSM buildings and roads from the
                     local visual cache (`naigos.demo.urban`, served at
                     `/urban`) extruded over the simulation's own DEM.
                     Presentation only; building geometry is never used by LOS.

Credentials are read from explicit environment variables only --
NAIGOS_CESIUM_ION_TOKEN (or CESIUM_ION_TOKEN), NAIGOS_GOOGLE_MAPS_API_KEY (or
GOOGLE_MAPS_API_KEY) -- and the ion token is substituted into the page at serve
time. With none of them the viewer falls back to keyless OpenStreetMap over the
simulation's own terrain, and every measured claim is unchanged.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import threading
import time
from collections import deque
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
from . import attitude as att_mod
from . import camera as camera_mod
from . import events as events_mod
from . import imagery as imagery_mod
from . import los as los_mod
from . import models as models_mod
from . import urban as urban_mod
from . import ambience as ambience_mod
from . import cities as cities_mod
from . import presentation as presentation_mod
from . import atmosphere as atmosphere_mod
from . import scenario as scenario_mod

ASSETS = Path(__file__).parent / "assets"

#: The scene/frame schema the page is written against. 2 added model registry,
#: camera presets, aircraft attitude (heading/pitch/roll), per-frame threat
#: state (heading, DEM ground height, tracked aircraft, sensor yaw) and visual
#: events. The page refuses a payload it does not recognise rather than
#: drawing half of one.
SCENE_SCHEMA = 2

#: Said wherever an effect is drawn. The effect is the picture; this is the claim.
EVENT_HONESTY_NOTE = (
    "Tracers, launch flashes and impacts visualise a shootdown the simulation "
    "already resolved (its kill-hazard draw); no projectile or missile is "
    "simulated. Sensor/turret slew shows which aircraft the threat's track "
    "matrix currently favours.")

#: How many recent events each live frame carries, so a client that misses a
#: tick still sees an event once. The page de-duplicates by event id.
LIVE_EVENT_BUFFER = 48



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


# ---- visual state shared by the live stream and the replay export -------------------
#
# Both paths hand the page the same shapes, built by the same functions below,
# from the same simulation state -- which is what keeps "one renderer" true once
# the renderer draws orientation and events as well as positions.

#: Field order of one packed threat state in a frame. Packed, not keyed: a
#: replay carries one per active threat per frame per policy, and keys would
#: triple the static artifact. `/scene` publishes this list so the page decodes
#: by name.
THREAT_STATE_FIELDS = ("i", "lon", "lat", "alt", "ground_m", "heading", "track", "sensor_yaw")


def aircraft_attitude(georef: GeoRef, pos, psi, gamma, phi) -> dict:
    """Heading (true north), pitch and roll, in degrees, for arrays of aircraft.

    Straight from the airframe state: see `naigos.demo.attitude` for the
    conventions and why nothing here is inferred from successive positions.
    """
    pos = np.asarray(pos, dtype=np.float64)
    return {
        "heading": att_mod.true_heading_deg(georef.to_wgs84, pos[..., 0], pos[..., 1], psi),
        "pitch": att_mod.pitch_deg(np.asarray(gamma, dtype=np.float64)),
        "roll": att_mod.roll_deg(np.asarray(phi, dtype=np.float64)),
    }


def threat_state_series(georef: GeoRef, hmap, tcfg, tpos, tpsi, active, track, blue_pos) -> list:
    """Packed threat state for S frames at once: a list of S lists of THREAT_STATE_FIELDS.

    `tpos` (S, T, 3), `tpsi` (S, T), `track` (S, T) with -1 for no track,
    `blue_pos` (S, B, 3); `active` (T,) or None for all.

    `ground_m` is the SIMULATION's DEM under the threat -- `sample_height`, the
    function the env pins ground units with -- and is what a ground model is
    anchored to. `track` is the aircraft this threat's track matrix favours
    (`events.track_selection`), or -1; `sensor_yaw` turns the turret/sensor node
    toward it, and is 0 (facing the platform heading) when there is no track.
    """
    tpos = np.asarray(tpos, dtype=np.float64)
    blue_pos = np.asarray(blue_pos, dtype=np.float64)
    track = np.asarray(track, dtype=int)
    S, T = tpos.shape[:2]
    idx = (np.arange(T) if active is None
           else np.flatnonzero(np.asarray(active, dtype=bool)))
    ground = np.asarray(sample_height(jnp.asarray(hmap), tcfg,
                                      jnp.asarray(tpos[..., 0]), jnp.asarray(tpos[..., 1])))
    lon, lat = (np.asarray(v).reshape(S, T) for v in georef.to_wgs84(tpos[..., 0], tpos[..., 1]))
    heading = np.asarray(att_mod.true_heading_deg(
        georef.to_wgs84, tpos[..., 0], tpos[..., 1], tpsi)).reshape(S, T)
    blon, blat = (np.asarray(v).reshape(blue_pos.shape[:2])
                  for v in georef.to_wgs84(blue_pos[..., 0], blue_pos[..., 1]))
    b = np.clip(track, 0, None)
    rows = np.arange(S)[:, None]
    brg = att_mod.bearing_deg(lon, lat, blon[rows, b], blat[rows, b])
    yaw = np.where(track >= 0, att_mod.sensor_yaw_deg(heading, brg), 0.0)
    return [
        [[int(i), round(float(lon[s, i]), 5), round(float(lat[s, i]), 5),
          round(float(tpos[s, i, 2])), round(float(ground[s, i])),
          round(float(heading[s, i])), int(track[s, i]), round(float(yaw[s, i]))]
         for i in idx]
        for s in range(S)
    ]


def threat_states(georef: GeoRef, hmap, tcfg, tpos, tpsi, active, lock, in_flight, blue_pos) -> list:
    """`threat_state_series` for one live frame, with the track chosen from `lock` (T, B)."""
    track = events_mod.track_selection(lock, in_flight)
    return threat_state_series(georef, hmap, tcfg, np.asarray(tpos)[None], np.asarray(tpsi)[None],
                               active, track[None], np.asarray(blue_pos)[None])[0]


def geo_event(georef: GeoRef, ev: dict) -> dict:
    """An `events` record with its ENU positions converted to [lon, lat, alt].

    `at` is where the aircraft was at the step the event came from; `from` is
    the attributed threat's position at that step, or None.
    """
    out = {k: v for k, v in ev.items() if k not in ("pos", "threat_pos")}
    for src, dst in (("pos", "at"), ("threat_pos", "from")):
        p = ev.get(src)
        if p is None:
            out[dst] = None
            continue
        lon, lat = georef.to_wgs84(p[0], p[1])
        out[dst] = [round(float(lon), 6), round(float(lat), 6), round(float(p[2]), 1)]
    return out


def threat_scene_entry(i: int, kind, lon: float, lat: float, alt: float, ground_m: float,
                       heading: float, active: bool, cfg=None, mobile: bool | None = None,
                       rec: dict | None = None) -> dict:
    """One `/scene` threat, from a live ThreatKindConfig or a recording's threat record."""
    if rec is not None:
        airborne = bool(rec.get("airborne", False))
        label, lethal, detect, alt_max = rec["label"], rec["lethal_m"], rec["detect_m"], rec["alt_max_m"]
    else:
        airborne = bool(kind.airborne)
        label = kind.label
        lethal = float(kind.lethal_range * cfg.red_lethal_scale)
        detect = float(kind.detect_range * cfg.red_detect_scale)
        alt_max = float(kind.alt_max)
        mobile = kind.speed > 0 if mobile is None else mobile
    return {
        "i": int(i),
        "lon": round(float(lon), 6), "lat": round(float(lat), 6), "alt": round(float(alt), 1),
        "label": label, "lethal_m": float(lethal), "detect_m": float(detect),
        "alt_max_m": float(alt_max), "mobile": bool(mobile), "airborne": airborne,
        "model": models_mod.threat_model_key(airborne, bool(mobile)),
        "ground_m": round(float(ground_m), 1), "heading": round(float(heading), 1),
        "active": bool(active),
    }


def grid_height(terrain_bytes: bytes, meta: dict, lon: float, lat: float) -> float:
    """Bilinear height from the served `/terrain` grid -- the page's sampleGrid(), inside the box.

    The city camera aims at the ground under the densest urban chunk; this is
    the height of the surface the globe actually draws there.
    """
    h = np.frombuffer(terrain_bytes, dtype="<i2").reshape(meta["ny"], meta["nx"]).astype(np.float64)
    u = (lon - meta["west"]) / (meta["east"] - meta["west"])
    v = (lat - meta["south"]) / (meta["north"] - meta["south"])
    fx = min(max(u, 0.0), 1.0) * (meta["nx"] - 1)
    fy = min(max(v, 0.0), 1.0) * (meta["ny"] - 1)
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, meta["nx"] - 1), min(y0 + 1, meta["ny"] - 1)
    tx, ty = fx - x0, fy - y0
    return float((h[y0, x0] * (1 - tx) + h[y0, x1] * tx) * (1 - ty)
                 + (h[y1, x0] * (1 - tx) + h[y1, x1] * tx) * ty)


def view_fields(bounds: dict, terrain_meta: dict, terrain_bytes: bytes | None,
                urban_status=None, camera_key: str | None = None,
                theatre: str | None = None) -> dict:
    """The camera presets (with the city focus) and the urban block, for `/scene`.

    Identical for live, served replay and the static export, so `--camera
    urban-overview` frames the same city in every mode -- including physics,
    which draws no buildings but can still be pointed at the basin.

    A theatre with a city config (``naigos.demo.cities``) declares its own city
    views; they are checked here against the grid this page draws -- above the
    DEM, over the safe region, clear of every protected zone -- and a preset
    that breaks the contract raises rather than opening somewhere unintended.
    """
    focus = None
    if urban_status is not None and urban_status.focus:
        f = urban_status.focus
        ground = (grid_height(terrain_bytes, terrain_meta, f["lon"], f["lat"])
                  if terrain_bytes else None)
        focus = {"lon": f["lon"], "lat": f["lat"], "ground_m": ground}
    city = cities_mod.get_city(theatre or (urban_status.aoi if urban_status is not None else None))
    sample = (cities_mod.TerrainSampler.from_bytes(terrain_bytes, terrain_meta)
              if (city is not None and terrain_bytes) else None)
    return {
        "camera": camera_mod.presets(bounds, terrain_meta, urban=focus, default=camera_key,
                                     city=city, sample=sample),
        "urban": urban_status.summary() if urban_status is not None else None,
    }


def visual_scene_fields(bounds: dict, terrain_meta: dict, inline_models: bool) -> dict:
    """The schema-2 additions to `/scene`, identical for live and replay."""
    return {
        "schema_version": SCENE_SCHEMA,
        "models": models_mod.page_registry(inline=inline_models),
        "camera": camera_mod.presets(bounds, terrain_meta),
        "lighting": camera_mod.lighting(),
        "threat_state_fields": list(THREAT_STATE_FIELDS),
        "event_note": EVENT_HONESTY_NOTE,
    }


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
    # Opt-in fictional ambience (naigos.demo.ambience.LiveAmbience), or None.
    # Handed plain coordinates each publish; it can read nothing else and
    # nothing it returns is fed back into the step.
    ambience: object = None

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
        # Visual events: derived from each step's state, never fed back into it.
        self.tick = 0
        self.events = events_mod.EventDeriver(n_blue=cfg.n_blue,
                                               lock_threshold=cfg.detection.lock_threshold)
        self._event_log: deque = deque(maxlen=LIVE_EVENT_BUFFER)
        self._ambience_log: deque = deque(maxlen=LIVE_EVENT_BUFFER)
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
            state_before = self.state
            state2, obs2, terms, info, feasible, _ = self._step(self.state, self.obs, k)

            t = jax.device_get(terms)
            self.counters.shot_down += int(np.asarray(t.shotdown).sum())
            self.counters.terrain += int(np.asarray(t.terrain_violation).sum())
            self.counters.bounds += int(np.asarray(t.bounds_violation).sum())
            self.counters.fuel += int(np.asarray(t.out_of_fuel).sum())
            self.counters.reached += int(np.asarray(t.arrived).sum())

            self.state, self.obs = state2, obs2
            self.sim_t += cfg.dt
            # Before the respawn below: a kill's position is where the aircraft
            # was when the env resolved it, not the start line it is re-tasked to.
            self._derive_events(state_before, state2, t, info)

            # re-task anything that finished this tick
            done_mask = np.asarray(jax.device_get(info["agent_done"]))
            if done_mask.any():
                self.key, kr = jax.random.split(self.key)
                self.state = self._respawn(self.state, kr, jnp.asarray(done_mask))
                self.obs = self.env.observe(self.state)
                self.counters.sorties += int(done_mask.sum())
                # a new sortie: nothing from the old one can be emitted again
                self.events.retask(done_mask)

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

    # ---------------------------------------------------------------- events
    def _derive_events(self, before, after, terms, info) -> None:
        """Visual events for the step that took `before` to `after`.

        Reads the step's own outputs -- terms, the engagement firing matrix,
        the lock matrix, the tracker ray's clearance -- and appends what it
        derives to a ring buffer the frames carry. Writes nothing back: the
        next step is computed from `self.state`, which this never touches.
        """
        cfg = self.env.cfg
        self.tick += 1
        b = jax.device_get(before)
        a = jax.device_get(after)
        trk = self._tracking(a)
        p_kill = np.asarray(per_threat_params(cfg, a.threats.kind)["p_kill"], dtype=np.float64)
        evs = self.events.step(
            step=self.tick, t_s=self.sim_t,
            in_flight=np.asarray(b.alive) & ~np.asarray(b.reached),
            alive=np.asarray(a.alive),
            shotdown=np.asarray(terms.shotdown) > 0.5,
            lock=np.asarray(a.lock), exposure=np.asarray(terms.exposure),
            blue_pos=np.asarray(a.air.pos), threat_pos=np.asarray(a.threats.pos),
            firing=np.asarray(jax.device_get(info["firing"])),
            kill_weight=1.0 - (1.0 - np.clip(p_kill, 0.0, 0.999)) ** cfg.dt,
            masked=trk["clearance"] < 0.0,
        )
        for ev in evs:
            self._event_log.append(geo_event(self.georef, ev))

    def _tracking(self, st) -> dict:
        """Which threat's ray to draw for each aircraft, and whether terrain cuts it.

        Selected by DETECTION PROBABILITY, not by lock. Lock decays toward zero
        the moment an aircraft is masked, so selecting on it hid the ray in
        exactly the situation the ray exists to show: the threat that would see
        you if the ridge were not there. pd already carries the LOS term, so
        fall back to geometric proximity when nothing can see the aircraft.
        """
        cfg = self.env.cfg
        pos = np.asarray(st.air.pos)
        tpos = np.asarray(st.threats.pos)
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
        emitter_enu = tpos[tracker] + np.array([0.0, 0.0, 10.0])
        clearance = np.asarray(
            terrain_mod.los_clearance(
                jnp.asarray(st.hmap), cfg.terrain, cfg.detection,
                jnp.asarray(emitter_enu), jnp.asarray(pos),
            )
        )
        return {"tracker": tracker, "emitter_enu": emitter_enu, "clearance": clearance,
                "pd": pd_arr}

    # --------------------------------------------------------------- publish
    def _publish(self, feasible, exposure, retasked=None):
        cfg = self.env.cfg
        st = jax.device_get(self.state)
        pos = np.asarray(st.air.pos)
        lon, lat = self.georef.to_wgs84(pos[:, 0], pos[:, 1])
        lock = np.asarray(st.lock).max(axis=0)
        att = aircraft_attitude(self.georef, pos, np.asarray(st.air.psi),
                                np.asarray(st.air.gamma), np.asarray(st.air.phi))

        ground = np.asarray(sample_height(jnp.asarray(st.hmap), cfg.terrain, pos[:, 0], pos[:, 1]))

        tpos = np.asarray(st.threats.pos)
        trk = self._tracking(st)
        tracker, emitter_enu, clearance, pd_arr = (
            trk["tracker"], trk["emitter_enu"], trk["clearance"], trk["pd"])
        trk_lon, trk_lat = self.georef.to_wgs84(tpos[tracker, 0], tpos[tracker, 1])
        los_rays = self._los_rays(st, emitter_enu, pos)
        in_flight = np.asarray(st.alive) & ~np.asarray(st.reached)

        obj = np.asarray(st.objective)
        olon, olat = self.georef.to_wgs84(obj[:, 0], obj[:, 1])

        frame = {
            "t": round(self.sim_t, 1),
            "tick": self.tick,
            "aircraft": [
                {
                    "id": int(i),
                    "lon": round(float(lon[i]), 6),
                    "lat": round(float(lat[i]), 6),
                    "alt": round(float(pos[i, 2]), 1),
                    "agl": round(float(pos[i, 2] - ground[i]), 1),
                    # attitude straight from the airframe state; conventions in
                    # naigos.demo.attitude. heading is compass, TRUE north.
                    "heading": round(float(att["heading"][i]), 2),
                    "pitch": round(float(att["pitch"][i]), 2),
                    "roll": round(float(att["roll"][i]), 2),
                    "sortie": int(self.events.sortie[i]),
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
                    # the same ray as a drawable curve: see _los_rays
                    "los": los_rays[i],
                }
                for i in range(cfg.n_blue)
            ],
            # Every active threat, packed as THREAT_STATE_FIELDS: movers move,
            # and any threat's sensor can slew toward the aircraft it tracks.
            "threats": threat_states(self.georef, st.hmap, cfg.terrain, tpos,
                                     np.asarray(st.threats.psi), np.asarray(st.threats.active),
                                     np.asarray(st.lock), in_flight, pos),
            # the last few visual events, de-duplicated by id in the page
            "events": list(self._event_log),
            "counters": {**self.counters.as_dict(), "threat_draws": self.threat_draws},
            # the viewer re-reads /scene when this changes, so a re-rolled field
            # does not leave stale envelopes drawn over the map
            "threat_draw": self.threat_draws,
            # fictional ambience effects starting in the current bucket, by sim
            # time; empty unless --ambience was asked for
            "ambience": self._ambience_events(lon, lat, tpos, st),
        }
        with self._lock:
            self.snapshot = frame

    def _ambience_events(self, lon, lat, tpos, st) -> list:
        """Advance the presentation-only ambience stream. Reads copies; writes nothing back."""
        if self.ambience is None:
            return []
        alive = np.asarray(st.alive)
        active = np.asarray(st.threats.active)
        tlon, tlat = self.georef.to_wgs84(tpos[:, 0], tpos[:, 1])
        pts = ([(float(lon[i]), float(lat[i])) for i in range(len(alive)) if alive[i]]
               + [(float(tlon[i]), float(tlat[i])) for i in range(len(active)) if active[i]])
        self._ambience_log.extend(self.ambience.advance(self.sim_t, pts))
        return list(self._ambience_log)

    def latest(self) -> dict:
        with self._lock:
            return self.snapshot

    # ------------------------------------------------------------- LOS rays
    def _los_rays(self, st, emitter_enu, pos) -> list[dict]:
        """The tracker ray as the curve the model actually marched.

        The frame used to carry the emitter, the aircraft and one clearance
        number, and the viewer drew a straight line between the two points. The
        model does not use a straight line: `los_clearance` drops every sample
        for 4/3-earth refraction, which reaches ~100 m at the midpoint of an
        80 km ray. On grazing geometry the drawn line therefore cleared ridges
        the model said it did not -- the picture disagreeing with the number
        printed beside it (next-steps E-13).

        So the ray is sent as the sampled, dropped polyline, plus the index of
        the sample where clearance is worst. `naigos.demo.los` owns the
        geometry and imports nothing from the env; the DEM lookup stays here,
        because the surface belongs to the simulation and the renderer's job is
        to draw the one it used.
        """
        cfg = self.env.cfg
        dcfg = cfg.detection
        n = cfg.n_blue

        s = los_mod.sample_fractions(dcfg.los_samples)[None, :]      # (1, S)
        seg = pos - emitter_enu                                      # (B, 3)
        gx = emitter_enu[:, None, 0] + s * seg[:, None, 0]
        gy = emitter_enu[:, None, 1] + s * seg[:, None, 1]
        ground = np.asarray(sample_height(
            jnp.asarray(st.hmap), cfg.terrain, jnp.asarray(gx), jnp.asarray(gy)))

        prof = los_mod.profile(
            emitter_enu, pos, ground,
            los_samples=dcfg.los_samples, earth_radius_eff=dcfg.earth_radius_eff,
        )

        out = []
        for i in range(n):
            pinch = int(prof["pinch"][i])
            idx = los_mod.draw_indices(dcfg.los_samples, pinch)
            lon, lat = self.georef.to_wgs84(prof["x"][i, idx], prof["y"][i, idx])
            alt = prof["z"][i, idx]
            plon, plat = self.georef.to_wgs84(
                float(prof["x"][i, pinch]), float(prof["y"][i, pinch]))
            cl = float(prof["min_clearance"][i])
            out.append({
                # the ray itself, refracted, in draw order
                "pts": [[round(float(a), 6), round(float(b), 6), round(float(c), 1)]
                        for a, b, c in zip(lon, lat, alt)],
                # where it is pinched: the sample the minimum came from. Its
                # third element is the RAY's height there, so the marker sits on
                # the line; `ground` is the surface it is being measured against.
                "pinch": [round(float(plon), 6), round(float(plat), 6),
                          round(float(prof["z"][i, pinch]), 1)],
                "pinch_ground": round(float(prof["ground"][i, pinch]), 1),
                "clearance": round(cl, 1),
                "blocked": bool(cl < 0.0),
                "band": los_mod.band(cl),
                # peak refraction drop on this ray -- the amount a straight line
                # would have lied by
                "drop_m": round(float(prof["drop"][i].max()), 1),
            })
        return out

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
        ground = np.asarray(sample_height(jnp.asarray(st.hmap), cfg.terrain,
                                          jnp.asarray(tpos[:, 0]), jnp.asarray(tpos[:, 1])))
        heading = att_mod.true_heading_deg(self.georef.to_wgs84, tpos[:, 0], tpos[:, 1],
                                           np.asarray(st.threats.psi))
        terrain = self.terrain_grid(notes)[1]
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
            "terrain": terrain,
            "threats": [
                threat_scene_entry(
                    i, cfg.threat_kinds[int(st.threats.kind[i])], lon[i], lat[i], tpos[i, 2],
                    ground[i], heading[i], bool(st.threats.active[i]), cfg=cfg)
                for i in range(cfg.n_threat)
            ],
            **visual_scene_fields(notes["geo_bounds"], terrain, inline_models=False),
        }


def recording_positions(rp: dict):
    """``positions_at(t)`` over a converted recording: every aircraft and threat, all policies.

    For the ambience stream's standoff test only. Built from the logged frames
    the page draws, nearest logged frame to ``t``; returns plain (lon, lat).
    """
    dt = float(rp["dt_s"] or 1.0)
    li, la = THREAT_STATE_FIELDS.index("lon"), THREAT_STATE_FIELDS.index("lat")

    def positions_at(t: float) -> list[tuple[float, float]]:
        pts = []
        for pol in rp["policies"]:
            frames = rp["frames"][pol]
            f = min(len(frames) - 1, max(0, int(round(t / dt))))
            pts += [(a["lon"], a["lat"]) for a in frames[f] if a.get("alive")]
            tf = rp["threat_frames"].get(pol) or []
            if f < len(tf):
                pts += [(row[li], row[la]) for row in tf[f]]
        return pts

    return positions_at


def replay_payload(path: Path, georef: GeoRef, cfg=None, aoi: str | None = None,
                   presentation=None) -> dict:
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

    # A model has a nose and wings, and a recording without the attitude state
    # cannot say where they point. Refuse rather than infer it from successive
    # positions, which would draw every banked turn wings-level.
    stale = [n for n, w in d.get("worlds", {}).items()
             if not all(k in w for k in ("psi", "gamma", "phi", "threat_psi", "track"))]
    if stale:
        raise SystemExit(
            f"{path} predates the 3D viewer schema (no logged attitude for {', '.join(stale)}), "
            f"so its aircraft cannot be drawn banking or climbing as they flew.\n"
            f"Regenerate it: python -m naigos.demo.replay --checkpoint <ckpt>"
        )

    # The globe must show the surface THIS ROLLOUT flew over. Serving the live
    # env's terrain instead draws a landscape the recording never saw, and the
    # logged AGL then disagrees with the drawn ground by hundreds of metres.
    terrain = None
    tcfg = hmap = None
    rt = d.get("terrain")
    if rt:
        from ..env.config import TerrainConfig

        tcfg = TerrainConfig(nx=rt["nx"], ny=rt["ny"], cell=rt["cell_m"])
        hmap = np.asarray(rt["heights"], dtype=np.float32).reshape(rt["ny"], rt["nx"])
        if d.get("geo_bounds"):
            terrain = build_terrain_grid(hmap, tcfg, georef, d["geo_bounds"])

    active = [bool(t["active"]) for t in d.get("threats", [])]
    out = {"policies": [], "frames": {}, "threat_frames": {}, "events": {},
           "dt_s": d.get("dt_s") or getattr(cfg, "dt", None),
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
        att = aircraft_attitude(georef, pos, np.asarray(w["psi"]), np.asarray(w["gamma"]),
                                np.asarray(w["phi"]))
        out["policies"].append(name)
        out["frames"][name] = [
            [
                {
                    "id": b,
                    "lon": round(float(lon[t, b]), 6), "lat": round(float(lat[t, b]), 6),
                    "alt": round(float(pos[t, b, 2]), 1), "agl": round(float(agl[t, b]), 1),
                    "heading": round(float(att["heading"][t, b]), 2),
                    "pitch": round(float(att["pitch"][t, b]), 2),
                    "roll": round(float(att["roll"][t, b]), 2),
                    "lock": round(float(lock[t, b]), 3),
                    "alive": bool(alive[t, b]), "reached": bool(reached[t, b]),
                }
                for b in range(pos.shape[1])
            ]
            for t in range(pos.shape[0])
        ]
        # Per-frame threat state, packed as THREAT_STATE_FIELDS -- the same
        # shape the live stream sends in `frame["threats"]`. `track` is the
        # choice the recording made from the full lock matrix.
        out["threat_frames"][name] = (
            threat_state_series(georef, hmap, tcfg, np.asarray(w["threat_pos"]),
                                np.asarray(w["threat_psi"]), active or None,
                                np.asarray(w["track"], dtype=int), pos)
            if hmap is not None else [[] for _ in range(pos.shape[0])])
        # Derived by `naigos.demo.events` from the FULL-resolution trace when the
        # recording was made, each keyed to the logged frame that first shows it.
        out["events"][name] = [geo_event(georef, ev) for ev in w.get("events", [])]
    out["summaries"] = d.get("summaries", {})
    # The fictional ambience stream, generated once for the whole recording from
    # the visual seed and exported with it, so every play and every scrub shows
    # the same effects at the same sim times. None when --ambience is off.
    if presentation is not None and out["policies"]:
        n = len(out["frames"][out["policies"][0]])
        out["ambience"] = presentation.replay_ambience(n * float(out["dt_s"] or 1.0),
                                                       recording_positions(out))
    # The recording's own scene, for the served replay mode: its threats, its
    # terrain, its bounds -- never the running env's.
    if terrain is not None:
        out["_scene"] = replay_scene(d, georef, terrain[1], out["dt_s"], inline_models=False)
    return out


def replay_scene(d: dict, georef: GeoRef, terrain_meta: dict, dt_s: float,
                 inline_models: bool = True) -> dict:
    """The `/scene` a RECORDING describes, built from the recording itself.

    The served replay mode used to hand the page `sim.scene(notes)` -- the LIVE
    env's threat field -- and patch only the terrain and the bounds over it. The
    envelopes drawn were therefore whichever sites the running simulation
    happened to hold, over a rollout that flew against different ones. It went
    unnoticed because a threat field looks like a threat field.

    A static export has no live env to borrow from, which is what forced the
    question. Everything here comes out of the recording.

    Threat positions live in `worlds[*].threat_pos`, not in `threats[]`, so the
    first frame supplies them; whether a site moved over the rollout is read off
    that same array rather than from a `speed` field the recording does not
    carry.
    """
    ref = "trained" if "trained" in d["worlds"] else next(iter(d["worlds"]))
    tp = np.asarray(d["worlds"][ref]["threat_pos"], dtype=np.float64)   # (S, T, 3)
    lon, lat = georef.to_wgs84(tp[0, :, 0], tp[0, :, 1])
    moved = np.abs(tp - tp[0]).max(axis=(0, 2)) > 1.0                   # (T,)

    # The ground under each threat, from the recording's own heightmap -- the
    # surface the rollout's env pinned its ground units to.
    rt = d["terrain"]
    from ..env.config import TerrainConfig

    tcfg = TerrainConfig(nx=rt["nx"], ny=rt["ny"], cell=rt["cell_m"])
    hmap = np.asarray(rt["heights"], dtype=np.float32).reshape(rt["ny"], rt["nx"])
    ground = np.asarray(sample_height(jnp.asarray(hmap), tcfg,
                                      jnp.asarray(tp[0, :, 0]), jnp.asarray(tp[0, :, 1])))
    tpsi = np.asarray(d["worlds"][ref].get("threat_psi", np.zeros(tp.shape[:2])))
    heading = att_mod.true_heading_deg(georef.to_wgs84, tp[0, :, 0], tp[0, :, 1], tpsi[0])

    threats = []
    for i, t in enumerate(d.get("threats", [])):
        # the recording says whether the kind moves; older ones fall back to
        # whether it was seen to move
        mobile = bool(t["mobile"]) if "mobile" in t else bool(moved[i])
        threats.append(threat_scene_entry(
            i, None, lon[i], lat[i], tp[0, i, 2], ground[i], heading[i], bool(t["active"]),
            mobile=mobile, rec=t))

    return {
        "theatre": d.get("theatre"),
        "bounds": d["geo_bounds"],
        "georef": d.get("georef"),
        "extent_m": d.get("extent_m"),
        "n_blue": int(d["n_blue"]),
        "dt_s": dt_s,
        "speed": None,
        "cbf": False,
        "mode": "replay",
        # A recording carries no assumptions block; saying so is better than an
        # empty list that reads as "nothing was assumed".
        "assumptions": d.get("assumptions", {}),
        "terrain": terrain_meta,
        "threats": threats,
        **visual_scene_fields(d["geo_bounds"], terrain_meta, inline_models=inline_models),
    }


def static_payload(path: Path, aoi: str | None = None, urban_status=None,
                   camera_key: str | None = None, embed_urban: bool = False,
                   presentation=None) -> dict:
    """Everything `cesium.html` fetches, for a page that will fetch nothing.

    Keyed by the route it stands in for, so the static artifact and the served
    session hand the renderer the same three shapes. `/terrain` is the same
    int16 buffer the endpoint serves, base64'd, rather than a JSON array of
    floats: 512x512 is 512 KB of bytes and about 2 MB of digits.

    Reuses `replay_payload` -- the same conversion, the same theatre guard, the
    same refusal to place a recording on a georef it was not recorded against.
    """
    d = json.loads(path.read_text())
    georef = GeoRef(**d["georef"]) if d.get("georef") else None
    if georef is None:
        raise SystemExit(
            f"{path} carries no georef and cannot be placed on the globe.\n"
            f"Regenerate it: python -m naigos.demo.replay --checkpoint <ckpt>"
        )
    rp = replay_payload(path, georef, None, aoi=aoi, presentation=presentation)
    if not rp.get("_terrain"):
        raise SystemExit(
            f"{path} carries no terrain grid, so the globe would draw a surface "
            f"this rollout never flew over.\n"
            f"Regenerate it: python -m naigos.demo.replay --checkpoint <ckpt>"
        )
    terrain_bytes, terrain_meta = rp["_terrain"]
    # Models inlined as data URIs: the artifact fetches nothing but CesiumJS.
    scene = replay_scene(d, georef, terrain_meta, rp["dt_s"], inline_models=True)
    scene["policies"] = rp["policies"]
    scene["n_frames"] = len(rp["frames"][rp["policies"][0]])
    scene.update(view_fields(scene["bounds"], terrain_meta, terrain_bytes, urban_status, camera_key,
                             theatre=rp.get("theatre")))
    # Atmosphere, ambience settings, scenario framing and checkpoint disclosure:
    # the same block the live server hands the page.
    if presentation is not None:
        scene.update(presentation.scene_fields())
    else:
        scene.update(presentation_mod.resolve(rp.get("theatre"), "physics",
                                              checkpoint=d.get("checkpoint")).scene_fields())
    out = {
        "/scene": scene,
        "/frames": {k: v for k, v in rp.items() if not k.startswith("_")},
        "/terrain": base64.b64encode(terrain_bytes).decode("ascii"),
    }
    # The city layer travels inside the artifact, so an offline viewer draws
    # it without a single request for data.
    if embed_urban and urban_status is not None and urban_status.available:
        out["/urban"] = urban_mod.page_payload(urban_status)
    return out


def render_page(visual, ion_token: str | None = None,
                google_api_key: str | None = None) -> str:
    """Substitute the credentials and the visual config into the viewer page.

    Credentials are injected here, at serve time, from environment variables --
    never written into `assets/cesium.html` and never committed. Each travels
    through exactly one substitution point: the ion token through `__ION_TOKEN__`,
    the Google Maps key through `__GOOGLE_API_KEY__`. The two config blobs are
    credential-free by construction (`VisualConfig` holds booleans, not secrets),
    so the sensitive strings in this function are those two and they are greppable.

    A credential the page cannot use is a credential that should not be in it.
    `/` is served to whoever can reach the port, so the Google key enters the
    document only on the one route that talks to Google directly; on the ion
    route, and in physics mode, CesiumJS never contacts Google at all and the
    page gets `null`.

    Split out of `make_handler` so this -- the one function in the server that
    handles secrets -- can be tested directly, without an env, a checkpoint or a
    JIT standing between the assertion and the string.
    """
    page_google_key = (
        google_api_key if visual.tileset_route == "google_maps_api" else None
    )
    return (ASSETS / "cesium.html").read_text().replace(
        "/*__ION_TOKEN__*/null", json.dumps(ion_token)
    ).replace(
        "/*__GOOGLE_API_KEY__*/null", json.dumps(page_google_key)
    ).replace(
        '/*__IMAGERY__*/{mode: "osm", osm_url: "https://tile.openstreetmap.org/"}',
        json.dumps(visual.to_page()),
    ).replace(
        '/*__VISUAL__*/{mode: "physics", evidence_grade: true}',
        json.dumps(visual.as_dict()),
    )


def make_handler(sim: Simulation, notes: dict, ion_token: str | None,
                 replay: dict | None = None, visual=None,
                 google_api_key: str | None = None,
                 urban_status=None, camera_key: str | None = None,
                 presentation=None):
    """Build the request handler, baking the token and the visual config into the page.

    The page is rendered once, here, by `render_page` -- which owns every
    credential substitution and the rule about which of them the page is allowed
    to see.

    The visual block is a separate substitution from the terrain endpoint on
    purpose: imagery is a cosmetic layer, terrain is the surface the model
    computed against, and the two must not be able to be confused for one another.
    """
    visual = visual or imagery_mod.resolve_visual_config(ion_token=ion_token)
    html = render_page(visual, ion_token, google_api_key)
    # The local city layer is served only to the mode that draws it; physics
    # and photorealistic pages get a 404 and never hold the bytes.
    urban_body = None
    if visual.mode == imagery_mod.URBAN_MODE and urban_status is not None:
        page = urban_mod.page_payload(urban_status)
        if page is not None:
            urban_body = json.dumps(page, separators=(",", ":")).encode()
    view_cache: dict = {}

    def terrain_pair():
        if replay is not None and replay.get("_terrain"):
            return replay["_terrain"]
        return sim.terrain_grid(notes)

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
                # A recording describes its own scene; the running env's threat
                # field has nothing to do with it.
                sc = (dict(replay["_scene"]) if replay is not None and replay.get("_scene")
                      else sim.scene(notes))
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
                # Camera presets and the urban block: computed once, from the
                # grid this page is drawing.
                if "v" not in view_cache:
                    tb, tm = terrain_pair()
                    view_cache["v"] = view_fields(sc["bounds"], tm, tb, urban_status, camera_key,
                                                  theatre=sc.get("theatre"))
                sc.update(view_cache["v"])
                # Atmosphere, ambience settings, scenario and checkpoint
                # disclosure, resolved once at startup (naigos.demo.presentation).
                if presentation is not None:
                    sc.update(presentation.scene_fields())
                return self._send(json.dumps(sc).encode(), "application/json")
            if self.path == "/urban":
                if urban_body is None:
                    return self.send_error(404, "no local urban layer in this visual mode")
                return self._send(urban_body, "application/json")
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
            if self.path.startswith(models_mod.MODEL_ROUTE):
                # registry filenames only; anything else is a 404 without a
                # filesystem lookup (see models.served_file)
                f = models_mod.served_file(self.path)
                if f is not None and f.exists():
                    return self._send(f.read_bytes(), "model/gltf-binary")
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


def smoke_report(visual, notes: dict, terrain_meta: dict, replay_path: str | None = None,
                 urban_status=None, camera_key: str | None = None,
                 terrain_bytes: bytes | None = None, presentation=None) -> dict:
    """What the viewer WOULD draw, resolved without a browser. Diagnostic only.

    Everything here is decided before the first pixel: which surface, which
    skin, which models load, which camera the page opens on, and whether a
    screenshot of it would be evidence. It deliberately does not claim more --
    `rendered_pixels` is False, because nothing was rendered. A headless check
    that said "rendered OK" would be the viewer equivalent of a test that
    passes on nothing.
    """
    audit = models_mod.audit()
    view = view_fields(notes["geo_bounds"], terrain_meta, terrain_bytes, urban_status, camera_key,
                       theatre=notes.get("theatre"))
    cam = view["camera"]
    urban_summary = view["urban"] or {}
    counts = urban_summary.get("counts") or {}
    draws_local = (visual.mode == imagery_mod.URBAN_MODE
                   and urban_summary.get("state") == "available")
    return {
        "rendered_pixels": False,
        "note": ("diagnostic only: resolves configuration, assets and camera; it does not "
                 "open a browser, create a WebGL context or render anything"),
        "scene_schema": SCENE_SCHEMA,
        "theatre": notes.get("theatre"),
        "mode": "replay" if replay_path else "live",
        "terrain_source": visual.terrain_source,
        "terrain_provider": ("CustomHeightmapTerrainProvider over /terrain (the simulation's DEM)"
                             if visual.terrain_source == imagery_mod.TERRAIN_SOURCE_SIMULATION
                             else "provider 3D tiles (globe hidden)"),
        "terrain_grid": terrain_meta.get("grid"),
        "terrain_range_m": [terrain_meta.get("min_m"), terrain_meta.get("max_m")],
        "visual_mode": visual.mode,
        "requested_visual_mode": visual.requested_mode,
        "imagery": visual.imagery,
        "base_imagery": visual.base_imagery,
        "fallback_reason": visual.fallback_reason,
        "evidence_grade": visual.evidence_grade,
        "models": audit,
        "model_fallback_count": sum(a["fallback"] for a in audit),
        "camera_mode": cam["default"],
        "camera_preset": cam["default"],
        "camera": cam[cam["default"]],
        "credentials_present": {"ion_token": visual.ion_token_present,
                                "google_api_key": visual.google_api_key_present},
        # --- the city layer (urban-presentation) -------------------------------
        "presentation_only": visual.presentation_only,
        "requested_geometry_source": visual.requested_geometry_source,
        "geometry_source": visual.geometry_source,
        # Never "active" here: only the browser can see a tile drawn.
        "provider_readiness": visual.provider_state,
        "local_urban_state": urban_summary.get("state") if urban_summary else None,
        "local_urban_reason": urban_summary.get("reason") if urban_summary else None,
        "local_cache_id": urban_summary.get("cache_id"),
        "local_cache_sha256": ((urban_summary.get("source") or {}).get("raw_sha256")),
        "building_count": counts.get("buildings", 0) if draws_local else 0,
        "road_count": counts.get("roads", 0) if draws_local else 0,
        "cached_building_count": counts.get("buildings", 0),
        "cached_road_count": counts.get("roads", 0),
        "render_only_building_occlusion": visual.building_occlusion_default,
        "building_note": visual.building_note,
        # --- the city standard: presentation, framing, exclusions ---------------
        "evidence_state": ("evidence-grade: the drawn surface is the simulation DEM and nothing "
                           "presentation-only is drawn" if visual.evidence_grade else
                           "presentation only: not evidence about terrain masking"),
        "camera_presets_available": cam.get("available"),
        **({"presentation": presentation.summary()} if presentation is not None else {}),
        "protected_zones": _zone_report(notes.get("theatre"), cam, urban_summary, presentation),
    }


def _zone_report(theatre: str | None, cam: dict, urban_summary: dict, presentation) -> dict:
    """How each protected zone was enforced in this configuration. Names and counts only."""
    city = cities_mod.get_city(theatre)
    if city is None:
        return {"zones": 0, "note": "this theatre declares no protected zones"}
    zones = city.aoi_def.protected_zones
    rep = {
        "zones": len(zones),
        "names": [z.name for z in zones],
        "camera_presets_checked": [k for k in cam.get("available", []) if k in city.presets],
        "camera_zones": len(city.zones("camera")),
        "render_cutout_zones": len(city.zones("render_cutout")),
        "extraction_zones": len(city.zones("extraction")),
        "urban_payload_exclusions": (urban_summary or {}).get("exclusions"),
        "ambience_zones": len(city.zones("ambience")),
    }
    if presentation is not None and presentation.ambience.enabled:
        mask = ambience_mod.build_mask(city, presentation.ambience.seed)
        rep["ambience_mask_cells_in_a_zone"] = sum(
            1 for lon, lat in mask.cells for z in city.zones("ambience") if z.contains(lon, lat))
    return rep


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
                         "or NAIGOS_GOOGLE_MAPS_API_KEY; falls back to physics without one). "
                         "urban-presentation: a dense 3D city for context -- provider tiles "
                         "when a credential exists, else the local OSM building cache from "
                         "`python -m naigos.demo.urban`; presentation only.")
    ap.add_argument("--camera", choices=list(camera_mod.CLI_PRESETS), default=None,
                    help="opening camera. Default: urban-overview under --visual "
                         "urban-presentation, terrain-overview otherwise.")
    ap.add_argument("--imagery", choices=imagery_mod.IMAGERY_MODES, default=None,
                    help="base-layer skin: Sentinel-2 via Cesium ion (needs a token) or "
                         "keyless OpenStreetMap. Defaults to sentinel2 under --visual physics "
                         "and osm under --visual photorealistic. In physics mode the terrain "
                         "comes from the simulation's DEM either way.")
    ap.add_argument("--atmosphere", default="auto",
                    choices=("auto",) + tuple(sorted(atmosphere_mod.PROFILES)),
                    help="presentation-only atmosphere profile from the allowlist. auto (default): "
                         "the theatre's own profile under a presentation mode, neutral under "
                         "physics. Never touches detection, LOS or any simulation result.")
    ap.add_argument("--ambience", default="off", choices=presentation_mod.AMBIENCE_CHOICES,
                    help="opt-in conflict_ambience: recurring FICTIONAL distant flashes, smoke and "
                         "dust as art-directed VFX, generated deterministically from --visual-seed. "
                         "Presentation modes only; not simulated events, nothing is hit.")
    ap.add_argument("--ambience-setting", default=ambience_mod.DEFAULT_SETTING,
                    choices=tuple(sorted(ambience_mod.SETTINGS)),
                    help="ambience density: sparse or sustained (default)")
    ap.add_argument("--visual-seed", type=int, default=0,
                    help="seed for the presentation-only ambience stream; same seed, same effects")
    ap.add_argument("--replay", default=None,
                    help="scrub a recorded rollout (runs/demo/demo.json) instead of streaming live")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--smoke-render", action="store_true",
                    help="print what the viewer would draw -- terrain source, imagery, model "
                         "registry and fallback count, opening camera, evidence grade -- as JSON "
                         "and exit. Diagnostic only: renders nothing and needs no checkpoint.")
    a = ap.parse_args(argv)

    # Validate the visual request before the DEM, the checkpoint and the JIT --
    # a run that spends 30 s starting up and then draws the wrong globe is worse
    # than one that refuses in the first millisecond. argparse's `choices` covers
    # each flag alone; this covers the combination.
    try:
        imagery_mod.validate_cli(a.visual, a.imagery)
    except imagery_mod.VisualConfigError as e:
        raise SystemExit(f"{ap.prog}: {e}")
    camera_key = camera_mod.preset_key(a.camera, a.visual == imagery_mod.URBAN_MODE)
    # How the scene is presented beyond the provider: atmosphere, ambience,
    # scenario framing and the checkpoint disclosure. Refusals happen here, in
    # the first millisecond, like the visual ones above.
    rec_ckpt = None
    if a.replay and Path(a.replay).exists():
        rec_ckpt = json.loads(Path(a.replay).read_text()).get("checkpoint")
    try:
        presentation = presentation_mod.resolve(
            a.aoi, a.visual, atmosphere=a.atmosphere, ambience=a.ambience,
            ambience_setting=a.ambience_setting, visual_seed=a.visual_seed,
            checkpoint=rec_ckpt or scenario_mod.checkpoint_theatre(a.checkpoint))
    except presentation_mod.PresentationError as e:
        raise SystemExit(f"{ap.prog}: {e}")
    # The local city layer: read from the visual cache, never fetched here. A
    # missing cache is a labelled state, not an error -- starting the server
    # must not depend on the network.
    urban_status = urban_mod.load(a.aoi)

    from ..env.theatre_bridge import describe, env_from_theatre
    from ..rl.red_team import RedCurriculum

    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, n_blue=a.blue, n_threat=a.threats, cell_m=a.cell_m)

    if a.smoke_render:
        # Same resolution path as a real launch -- credentials from the
        # environment, consumed into booleans -- minus the policy and the server.
        visual = imagery_mod.resolve_visual_config(
            a.visual, ion_token=imagery_mod.resolve_ion_token(a.ion_token),
            google_api_key=imagery_mod.resolve_google_api_key(), imagery=a.imagery,
            local_urban=urban_status.available)
        if a.replay:
            d = json.loads(Path(a.replay).read_text())
            georef = GeoRef(**d["georef"])
            notes = {**notes, "theatre": d.get("theatre"), "geo_bounds": d["geo_bounds"]}
            rt = d["terrain"]
            from ..env.config import TerrainConfig

            tb, meta = build_terrain_grid(
                np.asarray(rt["heights"], dtype=np.float32).reshape(rt["ny"], rt["nx"]),
                TerrainConfig(nx=rt["nx"], ny=rt["ny"], cell=rt["cell_m"]), georef, d["geo_bounds"])
        else:
            tb, meta = build_terrain_grid(np.asarray(hmap), cfg.terrain, GeoRef(**notes["georef"]),
                                          notes["geo_bounds"])
        print(json.dumps(smoke_report(visual, notes, meta, a.replay, urban_status, camera_key, tb,
                                      presentation=presentation), indent=2))
        return 0
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
        ambience=None if a.replay else presentation.live_ambience(),
    )
    replay = None
    if a.replay:
        rp = Path(a.replay)
        if not rp.exists():
            raise SystemExit(f"{rp} not found. Generate it with `python -m naigos.demo.replay`.")
        replay = replay_payload(rp, sim.georef, cfg, aoi=a.aoi, presentation=presentation)
        print(f"replay: {rp} -- {len(replay['policies'])} policies x "
              f"{len(replay['frames'][replay['policies'][0]])} frames")
    else:
        threading.Thread(target=sim.run, daemon=True).start()

    # Credentials come from the environment (the ion token may be overridden for
    # a single run by --ion-token). Nothing token-shaped is written to disk or
    # into the VisualConfig below, and `make_handler` puts each one into the
    # served page only on the route that uses it.
    ion_token = imagery_mod.resolve_ion_token(a.ion_token)
    google_api_key = imagery_mod.resolve_google_api_key()
    visual = imagery_mod.resolve_visual_config(
        a.visual, ion_token=ion_token, google_api_key=google_api_key, imagery=a.imagery,
        local_urban=urban_status.available,
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", a.port),
        make_handler(sim, notes, ion_token, replay, visual=visual,
                     google_api_key=google_api_key, urban_status=urban_status,
                     camera_key=camera_key, presentation=presentation),
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
        # The modes where the picture is not the model. Said on stdout as well
        # as in the HUD, so it is in the terminal scrollback of any screen capture.
        print(f"WARNING: {visual.separation_note}")
        # The viewer says the same thing, continuously, in a banner it cannot be
        # left without: the mode carries a runtime toggle back to physics terrain,
        # and lands there by itself if the provider fails.
        print("the browser shows a persistent visual-only banner in this mode, and the "
              "'physics terrain' button returns to the simulation's own surface")
    if visual.mode == imagery_mod.URBAN_MODE:
        if urban_status.available:
            c = urban_status.payload["counts"]
            print(f"urban: local cache {urban_status.payload['cache_id']} -- {c['buildings']:,} "
                  f"buildings, {c['roads']:,} road pieces (OSM, ODbL); "
                  + ("runtime fallback under the provider tiles"
                     if visual.geometry_source == imagery_mod.GEOMETRY_PROVIDER
                     else "drawn over the simulation DEM"))
            print(f"attribution: {imagery_mod.OSM_BUILDINGS_ATTRIBUTION}")
        else:
            print(f"urban: {urban_status.reason}")
        print(f"urban: {imagery_mod.BUILDING_LOS_NOTE} Render-only building occlusion "
              "starts OFF.")
    print(f"camera: {camera_key}")
    print(f"scenario: {presentation.scenario['label']}")
    print(presentation.scenario["checkpoint"]["text"])
    print(f"atmosphere: {presentation.atmosphere.key} (presentation only)")
    if presentation.ambience.enabled:
        print(f"ambience: conflict_ambience ({presentation.ambience.setting}, visual seed "
              f"{presentation.ambience.seed}) -- {ambience_mod.AMBIENCE_NOTE}")
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
