"""The Naigos environment: vectorized JAX 3D contested-airspace flight.

Contract (copied from Nomos): pure `reset` / `step` over **one** world, both
`jit`-able and `vmap`-able over a batch of worlds. Every array is fixed-size and
padded; nothing is data-dependent in shape; no Python control flow on traced
values.

    env = NaigosEnv(cfg)
    state, obs = env.reset(key)
    state, obs, terms, done, info = env.step(state, action)

`terms` is a `RewardTerms` of *raw, unweighted* quantities. Weighting lives in
`naigos.rl.reward` so the curriculum can re-weight without touching the physics,
and so `naigos.rl.verifier` can recheck the constraint channel independently.

BLUE HAS NO WEAPON. `action` is (n_blue, 3) = (bank, flight-path angle,
throttle). See config.BLUE_ACTION_NAMES and tests/test_invariant.py.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import airframe as af_mod
from . import detection as det_mod
from . import obs as obs_mod
from . import terrain as terrain_mod
from . import threats as threats_mod
from .airframe import AircraftState
from .config import ROUTE_FAMILY_NAMES, TRAIN_ROUTE_CENTERS_DEG, EnvConfig, route_protocol_problems
from .threats import ThreatState

# EnvState.route for the legacy placement: due east, family west_east, legacy flag.
_LEGACY_ROUTE = (0.0, float(ROUTE_FAMILY_NAMES.index("west_east")), 1.0)


class EnvState(NamedTuple):
    air: AircraftState
    threats: ThreatState
    lock: jax.Array  # (T, B) track quality in [0, 1]
    dwell: jax.Array  # (T, B) seconds of continuous lock
    alive: jax.Array  # (B,) bool
    reached: jax.Array  # (B,) bool
    objective: jax.Array  # (B, 3)
    prev_dist: jax.Array  # (B,) horizontal range to objective at the last step
    hmap: jax.Array  # (ny, nx) DEM in metres AMSL
    bounds: jax.Array  # (4,) x_min, x_max, y_min, y_max -- this world's play area
    t: jax.Array  # () int32 step counter
    key: jax.Array
    # (3,) float32: route bearing (rad, direction of travel, psi convention),
    # family index into config.ROUTE_FAMILY_NAMES, and 1.0 if the episode used
    # the legacy west-to-east placement. Read only by respawn, which re-tasks
    # along the same route, and by per-family evaluation. Nothing in step,
    # reward or observation depends on it.
    route: jax.Array


class RewardTerms(NamedTuple):
    """Raw, unweighted per-agent quantities. Signs are applied in reward.py."""

    progress: jax.Array  # (B,) metres closed on the objective this step
    exposure: jax.Array  # (B,) max detection probability against this aircraft
    lock_level: jax.Array  # (B,) max track quality against this aircraft
    envelope_dwell: jax.Array  # (B,) 1 if inside any lethal envelope
    arrived: jax.Array  # (B,) 1 on the step the objective is reached
    shotdown: jax.Array  # (B,) 1 on the step the aircraft is killed
    fuel_used: jax.Array  # (B,) kg burned this step
    g_excess: jax.Array  # (B,) load factor above 1
    terrain_violation: jax.Array  # (B,) 1 if below the AGL floor (or underground)
    ceiling_violation: jax.Array  # (B,)
    stall_violation: jax.Array  # (B,)
    bounds_violation: jax.Array  # (B,)
    out_of_fuel: jax.Array  # (B,)
    edge_proximity: jax.Array  # (B,) 0 inside, ramps to 1 at the map boundary


class NaigosEnv:
    def __init__(self, cfg: EnvConfig, hmap: jax.Array | None = None, red_policy=None):
        """`hmap` pins a fixed DEM (the real-data path). If None, each reset
        generates synthetic terrain -- useful for tests and for domain
        randomisation, but the headline result should be run on a real DEM.
        """
        problems = route_protocol_problems(cfg)
        if problems:
            raise ValueError("invalid route configuration:\n" + "\n".join(f"  - {p}" for p in problems))
        self.cfg = cfg
        self.fixed_hmap = hmap
        if red_policy is None:
            from ..rl.red_team import scripted_red

            red_policy = scripted_red
        self.red_policy = red_policy

    # ---------------------------------------------------------------- reset --
    def reset(self, key: jax.Array) -> tuple[EnvState, obs_mod.Observation]:
        """A fresh episode, with the route drawn as `cfg.route_mode` says."""
        return self._reset(key, None)

    def reset_route(self, key: jax.Array, route: jax.Array) -> tuple[EnvState, obs_mod.Observation]:
        """A fresh episode along a GIVEN route: `route` is (2,) [bearing rad, family id].

        This is how the frozen route bank is played (`naigos.rl.train.
        evaluate_route_bank`). It ignores `route_mode`: the bearing comes from the
        caller, everything else -- terrain, box, offsets, threats -- from `key`.
        jit/vmap-safe; `route` may be traced.
        """
        route = jnp.asarray(route, dtype=jnp.float32)
        return self._reset(key, jnp.stack([route[0], route[1], jnp.float32(0.0)]))

    def _reset(self, key: jax.Array, route: jax.Array | None) -> tuple[EnvState, obs_mod.Observation]:
        cfg = self.cfg
        k_map, k_start, k_obj, k_thr, k_next = jax.random.split(key, 5)

        hmap = self.fixed_hmap if self.fixed_hmap is not None else terrain_mod.synthetic_terrain(k_map, cfg.terrain)

        if cfg.map_randomize:
            # its own split, so the flag-off key stream is untouched
            k_next, k_box = jax.random.split(k_next)
            bounds = self._sample_bounds(k_box)
        else:
            bounds = jnp.array([0.0, cfg.terrain.extent_x, 0.0, cfg.terrain.extent_y], dtype=jnp.float32)
        x0, x1, y0, y1 = self._box(bounds)
        w, h = x1 - x0, y1 - y0
        B = cfg.n_blue

        if route is None and cfg.route_mode == "legacy":
            # start on one edge, objective on the far edge: the direct line crosses
            # the whole threat field, so the naive baseline is genuinely exposed.
            lateral = y0 + jnp.linspace(0.25, 0.75, B) * h
            jitter = jax.random.uniform(k_start, (B,), minval=-0.04, maxval=0.04) * h
            inset = cfg.spawn_inset_frac
            start_xy = jnp.stack(
                [
                    jnp.full((B,), x0 + inset * w),
                    jnp.clip(lateral + jitter, y0 + 1.5 * inset * h, y0 + (1 - 1.5 * inset) * h),
                ],
                axis=-1,
            )

            obj_lat = y0 + jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h
            obj_xy = jnp.stack([jnp.full((B,), x0 + (1.0 - inset) * w), obj_lat], axis=-1)
            route = jnp.array(_LEGACY_ROUTE, dtype=jnp.float32)
        else:
            if route is None:
                # its own split again: neither the legacy nor the map_randomize
                # stream moves when the route is drawn
                k_next, k_route = jax.random.split(k_next)
                route = self._sample_route(k_route)
            start_xy, obj_xy = self._route_endpoints(route[0], (x0, x1, y0, y1), k_start, k_obj)

        g_start = terrain_mod.sample_height(hmap, cfg.terrain, start_xy[:, 0], start_xy[:, 1])
        g_obj = terrain_mod.sample_height(hmap, cfg.terrain, obj_xy[:, 0], obj_xy[:, 1])
        start = jnp.concatenate([start_xy, (g_start + 2_500.0)[:, None]], axis=-1)
        objective = jnp.concatenate([obj_xy, (g_obj + 1_000.0)[:, None]], axis=-1)

        corridor = jnp.stack([jnp.mean(start, axis=0), jnp.mean(objective, axis=0)], axis=0)
        tstate = threats_mod.spawn(k_thr, cfg, hmap, corridor)

        to_obj = objective - start
        air = AircraftState(
            pos=start,
            speed=jnp.full((B,), cfg.airframe.v_init),
            psi=jnp.arctan2(to_obj[:, 1], to_obj[:, 0]),
            gamma=jnp.zeros((B,)),
            phi=jnp.zeros((B,)),
            fuel=jnp.full((B,), cfg.airframe.fuel_init),
        )

        state = EnvState(
            air=air,
            threats=tstate,
            lock=jnp.zeros((cfg.n_threat, B)),
            dwell=jnp.zeros((cfg.n_threat, B)),
            alive=jnp.ones((B,), dtype=bool),
            reached=jnp.zeros((B,), dtype=bool),
            objective=objective,
            prev_dist=jnp.linalg.norm(to_obj[:, :2], axis=-1),
            hmap=hmap,
            bounds=bounds,
            t=jnp.int32(0),
            key=k_next,
            route=route,
        )
        return state, self.observe(state)

    def _sample_route(self, key: jax.Array) -> jax.Array:
        """One training route: a family from `route_train_families`, uniformly,
        and a bearing uniform inside that family's +/- jitter sector.

        Only training centres can come out of here -- the config validator has
        already refused any family or jitter that would reach held-out geometry.
        """
        cfg = self.cfg
        k_fam, k_jit = jax.random.split(key)
        fams = cfg.route_train_families
        centres = jnp.deg2rad(jnp.array([TRAIN_ROUTE_CENTERS_DEG[f] for f in fams], dtype=jnp.float32))
        ids = jnp.array([ROUTE_FAMILY_NAMES.index(f) for f in fams], dtype=jnp.float32)
        i = jax.random.randint(k_fam, (), 0, len(fams))
        j = jnp.deg2rad(cfg.route_bearing_jitter_deg)
        bearing = centres[i] + jax.random.uniform(k_jit, (), minval=-j, maxval=j)
        return jnp.stack([jnp.arctan2(jnp.sin(bearing), jnp.cos(bearing)), ids[i], jnp.float32(0.0)])

    def _route_endpoints(self, bearing: jax.Array, box, k_start: jax.Array, k_obj: jax.Array):
        """Start and objective positions (B, 2) for a route flying along `bearing`.

        Geometry, in the "inset box" -- the play box shrunk by spawn_inset_frac on
        EVERY side, so both ends clear the boundary ramp (edge_margin_frac is
        smaller) whatever the direction:

          * each aircraft's start lies on the line through the inset box's centre
            offset cross-track by s_i, at the point where that line ENTERS the
            box; its objective on a line offset by o_i, where it LEAVES. For due
            east that is the legacy picture: west inset edge to east inset edge.
          * |s_i|, |o_i| <= route_lateral_frac * m, m the inset half short side,
            so every line passes through the box's inscribed disk and each end is
            at least m*sqrt(1 - f^2) from the centre along track (the length
            bound in config.route_min_length_guarantee).
          * the along-track reach is capped at the inset box's half LONG side, so a
            diagonal is no longer than a cardinal route along the long axis; the
            cap never binds on a cardinal route.

        Starts are spread across the track like the legacy starts (evenly plus
        jitter); objectives are drawn independently, also like legacy.
        """
        cfg = self.cfg
        B = cfg.n_blue
        x0, x1, y0, y1 = box
        inset = cfg.spawn_inset_frac
        sx0, sx1 = x0 + inset * (x1 - x0), x1 - inset * (x1 - x0)
        sy0, sy1 = y0 + inset * (y1 - y0), y1 - inset * (y1 - y0)
        cx, cy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
        a, b = 0.5 * (sx1 - sx0), 0.5 * (sy1 - sy0)
        m = jnp.minimum(a, b)
        reach = jnp.maximum(a, b)

        dx, dy = jnp.cos(bearing), jnp.sin(bearing)
        nx, ny = -dy, dx
        lim = cfg.route_lateral_frac * m

        spread = jnp.linspace(-0.8, 0.8, B) if B > 1 else jnp.zeros((1,))
        s = jnp.clip(spread + jax.random.uniform(k_start, (B,), minval=-0.1, maxval=0.1), -1.0, 1.0) * lim
        o = jax.random.uniform(k_obj, (B,), minval=-1.0, maxval=1.0) * lim

        def chord(off):
            # slab test of the line (c + off*n) + t*d against the inset box
            px, py = cx + off * nx, cy + off * ny
            inv_x = 1.0 / jnp.where(jnp.abs(dx) < 1e-9, 1e-9, dx)
            inv_y = 1.0 / jnp.where(jnp.abs(dy) < 1e-9, 1e-9, dy)
            tx0, tx1 = (sx0 - px) * inv_x, (sx1 - px) * inv_x
            ty0, ty1 = (sy0 - py) * inv_y, (sy1 - py) * inv_y
            t_in = jnp.maximum(jnp.minimum(tx0, tx1), jnp.minimum(ty0, ty1))
            t_out = jnp.minimum(jnp.maximum(tx0, tx1), jnp.maximum(ty0, ty1))
            return px, py, t_in, t_out

        spx, spy, t_in, _ = chord(s)
        opx, opy, _, t_out = chord(o)
        t_s = jnp.maximum(t_in, -reach)
        t_o = jnp.minimum(t_out, reach)
        start_xy = jnp.stack([spx + t_s * dx, spy + t_s * dy], axis=-1)
        obj_xy = jnp.stack([opx + t_o * dx, opy + t_o * dy], axis=-1)
        return start_xy, obj_xy

    def _sample_bounds(self, key: jax.Array) -> jax.Array:
        """Draw a random rectangular play area inside the fixed DEM grid.

        The grid is a static shape, so map-size variety has to live inside it:
        width uniform in [map_min_extent_m, extent_x], height = width * aspect
        clipped to [map_min_extent_m, extent_y], placed at a random offset that
        keeps the whole box on the grid.
        """
        cfg = self.cfg
        ex, ey = cfg.terrain.extent_x, cfg.terrain.extent_y
        k_w, k_a, k_x, k_y = jax.random.split(key, 4)
        w = jax.random.uniform(k_w, (), minval=min(cfg.map_min_extent_m, ex), maxval=ex)
        lo, hi = cfg.map_aspect_range
        aspect = jax.random.uniform(k_a, (), minval=lo, maxval=hi)
        h = jnp.clip(w * aspect, min(cfg.map_min_extent_m, ey), ey)
        x0 = jax.random.uniform(k_x, ()) * (ex - w)
        y0 = jax.random.uniform(k_y, ()) * (ey - h)
        return jnp.stack([x0, x0 + w, y0, y0 + h]).astype(jnp.float32)

    def _box(self, bounds: jax.Array):
        """(x_min, x_max, y_min, y_max) of the play area.

        With map_randomize off the box IS the grid, so it comes back as the same
        Python constants the env used before bounds existed: the default path
        stays bit-identical instead of picking up float32 rounding.
        """
        if not self.cfg.map_randomize:
            return 0.0, self.cfg.terrain.extent_x, 0.0, self.cfg.terrain.extent_y
        return bounds[0], bounds[1], bounds[2], bounds[3]

    # -------------------------------------------------------------- observe --
    def _geometry(self, state: EnvState):
        cfg = self.cfg
        tparams = threats_mod.per_threat_params(cfg, state.threats.kind)
        det = det_mod.detection_probability(
            state.hmap,
            cfg.terrain,
            cfg.detection,
            state.air.pos,
            state.air.psi,
            state.threats.pos,
            tparams,
            state.threats.active,
        )
        # a dead or arrived aircraft stops generating returns
        live = (state.alive & ~state.reached).astype(jnp.float32)
        det["pd"] = det["pd"] * live[None, :]
        return tparams, det

    def observe(self, state: EnvState) -> obs_mod.Observation:
        cfg = self.cfg
        tparams, det = self._geometry(state)
        return obs_mod.build(
            cfg,
            state.air.pos,
            state.air.psi,
            state.air.gamma,
            state.air.phi,
            state.air.speed,
            state.air.fuel,
            af_mod.velocity(state.air),
            state.alive,
            state.objective,
            det["alt_agl"],
            state.threats,
            tparams,
            det["pd"],
            state.lock,
            det["slant"],
            det["vis"],
            bounds=state.bounds,
        )

    # ----------------------------------------------------------------- step --
    def step(self, state: EnvState, action: jax.Array):
        """One env step. `action` is (n_blue, 3) in [-1, 1]."""
        cfg = self.cfg
        dt = cfg.dt
        key, k_kill, k_red = jax.random.split(state.key, 3)

        frozen = (~state.alive) | state.reached  # dead/arrived aircraft stop flying
        fuel_before = state.air.fuel

        # sub-step the airframe: the decision rate (dt) is coarser than the rate
        # at which a 7 g turn actually needs integrating.
        h = dt / cfg.substeps
        air_next = state.air
        for _ in range(cfg.substeps):
            air_next = af_mod.step(cfg.airframe, air_next, action, h)
        air = jax.tree.map(lambda new, old: jnp.where(_bcast(frozen, new), old, new), air_next, state.air)

        # --- red team moves, using the picture it had at the start of the step -
        psi_cmd, speed_cmd = self.red_policy(
            k_red, cfg, state.threats, state.air.pos, af_mod.velocity(state.air), state.alive & ~state.reached, state.lock
        )
        tstate = threats_mod.step(cfg, state.threats, state.hmap, psi_cmd, speed_cmd)

        # --- detection against the new geometry -------------------------------
        tparams = threats_mod.per_threat_params(cfg, tstate.kind)
        det = det_mod.detection_probability(
            state.hmap, cfg.terrain, cfg.detection, air.pos, air.psi, tstate.pos, tparams, tstate.active
        )
        live = (state.alive & ~state.reached).astype(jnp.float32)
        pd = det["pd"] * live[None, :]

        lock = det_mod.update_track(state.lock, pd, tparams, dt)
        locked = lock >= cfg.detection.lock_threshold
        dwell = jnp.where(locked, state.dwell + dt, 0.0)

        in_env, firing, hazard = det_mod.engagement(
            cfg.detection, lock, dwell, det["slant"], det["alt_agl"], air.pos[:, 2], tparams, dt
        )
        hazard = hazard * live

        # --- outcomes ----------------------------------------------------------
        shot = (jax.random.uniform(k_kill, (cfg.n_blue,)) < hazard) & state.alive & ~state.reached

        ground = terrain_mod.sample_height(state.hmap, cfg.terrain, air.pos[:, 0], air.pos[:, 1])
        alt_agl = air.pos[:, 2] - ground
        terrain_hit = (alt_agl < cfg.airframe.floor_agl) & ~frozen
        ceiling_hit = (air.pos[:, 2] >= cfg.airframe.ceiling - 1e-3) & ~frozen
        stall = (air.speed <= cfg.airframe.v_stall + 1e-3) & ~frozen
        # out of bounds means out of THIS world's play area, not off the DEM grid
        x0, x1, y0, y1 = self._box(state.bounds)
        oob = (
            (air.pos[:, 0] < x0)
            | (air.pos[:, 0] > x1)
            | (air.pos[:, 1] < y0)
            | (air.pos[:, 1] > y1)
        ) & ~frozen
        dry = (air.fuel <= 0.0) & ~frozen

        # Soft boundary ramp. The terminal bounds penalty is a cliff with no
        # gradient: by the time it fires the aircraft is already gone, so the
        # policy learns nothing about *approaching* the edge. MEASURED: without
        # this, out-of-bounds losses climbed to 30% as the policy learned to
        # dodge threats by leaving the map.
        # Extent-relative so it stays narrower than the spawn inset on any map
        # size. See EnvConfig.edge_margin; with map_randomize it scales with the box.
        margin = cfg.edge_margin_frac * jnp.minimum(x1 - x0, y1 - y0) if cfg.map_randomize else cfg.edge_margin
        dx_edge = jnp.minimum(air.pos[:, 0] - x0, x1 - air.pos[:, 0])
        dy_edge = jnp.minimum(air.pos[:, 1] - y0, y1 - air.pos[:, 1])
        edge = jnp.clip(1.0 - jnp.minimum(dx_edge, dy_edge) / margin, 0.0, 1.0)

        dist = jnp.linalg.norm((state.objective - air.pos)[:, :2], axis=-1)
        arrive = (dist <= cfg.objective_radius) & state.alive & ~state.reached

        alive = state.alive & ~shot & ~terrain_hit & ~oob
        reached = state.reached | arrive

        terms = RewardTerms(
            progress=jnp.where(frozen, 0.0, state.prev_dist - dist),
            exposure=jnp.max(pd, axis=0),
            lock_level=jnp.max(lock, axis=0),
            envelope_dwell=jnp.max(in_env * (lock > 0.05), axis=0) * live,
            arrived=arrive.astype(jnp.float32),
            shotdown=shot.astype(jnp.float32),
            fuel_used=jnp.where(frozen, 0.0, fuel_before - air.fuel),
            g_excess=jnp.where(frozen, 0.0, af_mod.load_factor(air.phi) - 1.0),
            terrain_violation=terrain_hit.astype(jnp.float32),
            ceiling_violation=ceiling_hit.astype(jnp.float32),
            stall_violation=stall.astype(jnp.float32),
            bounds_violation=oob.astype(jnp.float32),
            out_of_fuel=dry.astype(jnp.float32),
            edge_proximity=jnp.where(frozen, 0.0, edge),
        )

        t = state.t + 1
        new_state = EnvState(
            air=air,
            threats=tstate,
            lock=lock,
            dwell=dwell,
            alive=alive,
            reached=reached,
            objective=state.objective,
            prev_dist=jnp.where(frozen, state.prev_dist, dist),
            hmap=state.hmap,
            bounds=state.bounds,
            t=t,
            key=key,
            route=state.route,
        )

        agent_done = (~alive) | reached
        episode_done = jnp.all(agent_done) | (t >= cfg.max_steps)

        info = {
            "hazard": hazard,
            "firing": firing,
            "alt_agl": alt_agl,
            "dist_to_objective": dist,
            "agent_done": agent_done,
            "n_alive": jnp.sum(alive),
            "n_reached": jnp.sum(reached),
        }
        return new_state, self.observe(new_state), terms, episode_done, info

    # ------------------------------------------------------------- respawn --
    def respawn(self, state: EnvState, key: jax.Array, mask: jax.Array) -> EnvState:
        """Re-task the masked aircraft: fresh start, fresh objective, cleared track.

        prompt.md s2 offers a choice between a finite cohort that freezes and
        continuous respawn. Training uses the finite cohort, because an episode
        return has to mean "one sortie"; this is the continuous-tasking variant,
        used by the live viewer so the theatre never empties out.

        Crucially it also clears each respawned aircraft's COLUMN of the lock and
        dwell matrices. Leaving them would hand a brand-new aircraft the track
        history of the one that just died at the far end of the map -- it would
        be shot down seconds after spawning, for reasons invisible in the trace.
        """
        cfg = self.cfg
        k_lat, k_obj, k_next = jax.random.split(key, 3)
        B = cfg.n_blue
        # re-task inside this world's own play area
        x0, x1, y0, y1 = self._box(state.bounds)
        w, h = x1 - x0, y1 - y0
        inset = cfg.spawn_inset_frac

        # Both placements are computed and the episode's own flag picks one, so a
        # world keeps its route however it was reset: legacy, drawn, or from the
        # route bank. `where` only selects, so legacy worlds are bit-identical.
        lat = y0 + jax.random.uniform(k_lat, (B,), minval=1.5 * inset, maxval=1 - 1.5 * inset) * h
        start_legacy = jnp.stack([jnp.full((B,), x0 + inset * w), lat], axis=-1)
        obj_lat = y0 + jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h
        obj_legacy = jnp.stack([jnp.full((B,), x0 + (1.0 - inset) * w), obj_lat], axis=-1)
        # same bearing the episode began with: the threat field was built around
        # that corridor, so a re-task along a new bearing would fly a route the
        # threats were never placed against.
        start_route, obj_route = self._route_endpoints(state.route[0], (x0, x1, y0, y1), k_lat, k_obj)
        legacy = state.route[2] > 0.5
        start_xy = jnp.where(legacy, start_legacy, start_route)
        obj_xy = jnp.where(legacy, obj_legacy, obj_route)

        g_s = terrain_mod.sample_height(state.hmap, cfg.terrain, start_xy[:, 0], start_xy[:, 1])
        g_o = terrain_mod.sample_height(state.hmap, cfg.terrain, obj_xy[:, 0], obj_xy[:, 1])
        start = jnp.concatenate([start_xy, (g_s + 2_500.0)[:, None]], axis=-1)
        objective = jnp.concatenate([obj_xy, (g_o + 1_000.0)[:, None]], axis=-1)
        to_obj = objective - start

        m = mask[:, None]
        air = AircraftState(
            pos=jnp.where(m, start, state.air.pos),
            speed=jnp.where(mask, cfg.airframe.v_init, state.air.speed),
            psi=jnp.where(mask, jnp.arctan2(to_obj[:, 1], to_obj[:, 0]), state.air.psi),
            gamma=jnp.where(mask, 0.0, state.air.gamma),
            phi=jnp.where(mask, 0.0, state.air.phi),
            fuel=jnp.where(mask, cfg.airframe.fuel_init, state.air.fuel),
        )
        clear = ~mask[None, :]
        return state._replace(
            air=air,
            objective=jnp.where(m, objective, state.objective),
            prev_dist=jnp.where(mask, jnp.linalg.norm(to_obj[:, :2], axis=-1), state.prev_dist),
            alive=state.alive | mask,
            reached=state.reached & ~mask,
            lock=state.lock * clear,
            dwell=state.dwell * clear,
            key=k_next,
        )

    def reroll_threats(self, state: EnvState, key: jax.Array) -> EnvState:
        """Draw a fresh threat field over the same terrain, clearing all tracks.

        MEASURED NEED: `respawn` re-tasks aircraft but leaves the threat layout
        fixed, so a long live session measures one draw rather than the policy.
        A benign draw showed a 100% success rate against 55% over 24 sampled
        layouts -- the counters looked like a result and were an artefact of one
        map.
        """
        k_thr, k_next = jax.random.split(key)
        # the corridor is drawn from positions and objectives inside state.bounds,
        # and the play area itself is kept. Threats may land outside the box on
        # purpose: an emitter past the border still sees in.
        corridor = jnp.stack([jnp.mean(state.air.pos, axis=0), jnp.mean(state.objective, axis=0)], axis=0)
        tstate = threats_mod.spawn(k_thr, self.cfg, state.hmap, corridor)
        return state._replace(
            threats=tstate,
            lock=jnp.zeros_like(state.lock),
            dwell=jnp.zeros_like(state.dwell),
            key=k_next,
        )

    # ------------------------------------------------------------- rollouts --
    def rollout(self, key, policy, n_steps: int | None = None, action_filter=None, route=None):
        """Scan a whole episode. `policy(obs, key) -> (n_blue, 3)`.

        `route`, if given, is a (2,) [bearing rad, family id] and the episode
        starts from `reset_route` instead of `reset` -- how the route bank runs.

        `action_filter(state, obs, action) -> (action, feasible)` is the optional
        runtime safety backstop (see `naigos.rl.cbf.make_policy_filter`). It sits
        between the policy and the env exactly as `a_exec = filter(a_policy,
        state)` describes, and its per-step feasibility flags are logged so the
        infeasibility rate can be reported rather than hidden.

        Returns `(final_state, traj)` where traj is a pytree stacked on a leading
        time axis -- the trace the verifier and the demo replay consume.
        """
        n_steps = n_steps or self.cfg.max_steps

        def body(carry, _):
            state, obs, k = carry
            k, ka = jax.random.split(k)
            action = policy(obs, ka)
            if action_filter is None:
                feasible = jnp.ones((self.cfg.n_blue,), dtype=bool)
            else:
                action, feasible = action_filter(state, obs, action)
            state2, obs2, terms, done, info = self.step(state, action)
            out = {
                "pos": state2.air.pos,
                "psi": state2.air.psi,
                # attitude state the airframe integrated, logged as-is so a
                # replay can draw bank and climb instead of inferring them
                "gamma": state2.air.gamma,
                "phi": state2.air.phi,
                "speed": state2.air.speed,
                "alt_agl": info["alt_agl"],
                "alive": state2.alive,
                "reached": state2.reached,
                "action": action,
                "cbf_feasible": feasible,
                "terms": terms,
                "done": done,
                "threat_pos": state2.threats.pos,
                "threat_psi": state2.threats.psi,
                "lock": state2.lock,
                # (T, B) engagement firing solutions this step -- which threat a
                # logged shootdown can be attributed to
                "firing": info["firing"],
            }
            return (state2, obs2, k), out

        k0, k1 = jax.random.split(key)
        state, obs = self.reset(k0) if route is None else self.reset_route(k0, route)
        (final, _, _), traj = jax.lax.scan(body, (state, obs, k1), None, length=n_steps)
        return final, traj


def _bcast(mask: jax.Array, like: jax.Array) -> jax.Array:
    """Broadcast a (B,) boolean over a (B, ...) leaf."""
    return mask.reshape(mask.shape + (1,) * (like.ndim - 1))
