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
from .config import EnvConfig
from .threats import ThreatState


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
    t: jax.Array  # () int32 step counter
    key: jax.Array


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
        self.cfg = cfg
        self.fixed_hmap = hmap
        if red_policy is None:
            from ..rl.red_team import scripted_red

            red_policy = scripted_red
        self.red_policy = red_policy

    # ---------------------------------------------------------------- reset --
    def reset(self, key: jax.Array) -> tuple[EnvState, obs_mod.Observation]:
        cfg = self.cfg
        k_map, k_start, k_obj, k_thr, k_next = jax.random.split(key, 5)

        hmap = self.fixed_hmap if self.fixed_hmap is not None else terrain_mod.synthetic_terrain(k_map, cfg.terrain)

        ex, ey = cfg.terrain.extent_x, cfg.terrain.extent_y
        B = cfg.n_blue

        # start on one edge, objective on the far edge: the direct line crosses
        # the whole threat field, so the naive baseline is genuinely exposed.
        lateral = jnp.linspace(0.25, 0.75, B) * ey
        jitter = jax.random.uniform(k_start, (B,), minval=-0.04, maxval=0.04) * ey
        inset = cfg.spawn_inset_frac
        start_xy = jnp.stack(
            [jnp.full((B,), inset * ex), jnp.clip(lateral + jitter, 1.5 * inset * ey, (1 - 1.5 * inset) * ey)],
            axis=-1,
        )

        obj_lat = jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * ey
        obj_xy = jnp.stack([jnp.full((B,), (1.0 - inset) * ex), obj_lat], axis=-1)

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
            t=jnp.int32(0),
            key=k_next,
        )
        return state, self.observe(state)

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
        oob = (
            (air.pos[:, 0] < 0.0)
            | (air.pos[:, 0] > cfg.terrain.extent_x)
            | (air.pos[:, 1] < 0.0)
            | (air.pos[:, 1] > cfg.terrain.extent_y)
        ) & ~frozen
        dry = (air.fuel <= 0.0) & ~frozen

        # Soft boundary ramp. The terminal bounds penalty is a cliff with no
        # gradient: by the time it fires the aircraft is already gone, so the
        # policy learns nothing about *approaching* the edge. MEASURED: without
        # this, out-of-bounds losses climbed to 30% as the policy learned to
        # dodge threats by leaving the map.
        # Extent-relative so it stays narrower than the spawn inset on any map
        # size. See EnvConfig.edge_margin.
        margin = cfg.edge_margin
        dx_edge = jnp.minimum(air.pos[:, 0], cfg.terrain.extent_x - air.pos[:, 0])
        dy_edge = jnp.minimum(air.pos[:, 1], cfg.terrain.extent_y - air.pos[:, 1])
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
            t=t,
            key=key,
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

    # ------------------------------------------------------------- rollouts --
    def rollout(self, key, policy, n_steps: int | None = None):
        """Scan a whole episode. `policy(obs, key) -> (n_blue, 3)`.

        Returns `(final_state, traj)` where traj is a pytree stacked on a leading
        time axis -- the trace the verifier and the demo replay consume.
        """
        n_steps = n_steps or self.cfg.max_steps

        def body(carry, _):
            state, obs, k = carry
            k, ka = jax.random.split(k)
            action = policy(obs, ka)
            state2, obs2, terms, done, info = self.step(state, action)
            out = {
                "pos": state2.air.pos,
                "psi": state2.air.psi,
                "speed": state2.air.speed,
                "alt_agl": info["alt_agl"],
                "alive": state2.alive,
                "reached": state2.reached,
                "action": action,
                "terms": terms,
                "done": done,
                "threat_pos": state2.threats.pos,
                "lock": state2.lock,
            }
            return (state2, obs2, k), out

        k0, k1 = jax.random.split(key)
        state, obs = self.reset(k0)
        (final, _, _), traj = jax.lax.scan(body, (state, obs, k1), None, length=n_steps)
        return final, traj


def _bcast(mask: jax.Array, like: jax.Array) -> jax.Array:
    """Broadcast a (B,) boolean over a (B, ...) leaf."""
    return mask.reshape(mask.shape + (1,) * (like.ndim - 1))
