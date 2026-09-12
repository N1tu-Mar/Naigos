"""Red-side state: the threat field that blue must survive.

Threats are *the environment* from blue's point of view (prompt.md s0). Blue
never controls them and never attacks them. This module owns their state, their
kinematics and the per-kind parameter table; the behaviour that picks their
commands lives in `naigos.rl.red_team`.

Airborne kinds fly a bounded 3D track: horizontal pursuit still comes from the
red policy's `psi_cmd` / `speed_cmd`, and the vertical channel is a separate,
deterministic command built here from the SAME track picture red already uses
(`vertical_command`). Ground kinds are unaffected and stay pinned to the DEM.
See docs/interceptor-3d-dynamics.md for the modelling assumptions.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig, ThreatKindConfig
from . import terrain as terrain_mod

# every scalar field of ThreatKindConfig, gathered per-threat into (T,) arrays
# only the numeric fields go into the per-threat jax table; `label` is a string
# and `airborne` is folded in as a float so it can be indexed inside jit.
KIND_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(ThreatKindConfig) if f.type in ("float", "bool")
)


class ThreatState(NamedTuple):
    pos: jax.Array  # (T, 3) local ENU metres
    psi: jax.Array  # (T,) ground/air track heading
    speed: jax.Array  # (T,)
    kind: jax.Array  # (T,) int32
    active: jax.Array  # (T,) bool
    vz: jax.Array  # (T,) m/s vertical velocity, +up. 0 for ground kinds.


def kind_param_table(cfg: EnvConfig) -> dict[str, jax.Array]:
    """(N_THREAT_KINDS,) arrays, one per ThreatKindConfig field, with the red
    curriculum scales already applied.

    The curriculum (prompt.md s4 v2) turns red difficulty up by scaling these:
    longer detection reach, tighter lethal envelope, faster reaction, faster
    interceptors. Annealing happens by rebuilding EnvConfig, so the table is
    static per-config and costs nothing at runtime.
    """
    table = {f: jnp.array([float(getattr(k, f)) for k in cfg.threat_kinds], dtype=jnp.float32) for f in KIND_FIELDS}
    table["detect_range"] = table["detect_range"] * cfg.red_detect_scale
    table["lethal_range"] = table["lethal_range"] * cfg.red_lethal_scale
    table["reaction_latency"] = table["reaction_latency"] * cfg.red_latency_scale
    table["speed"] = table["speed"] * cfg.red_speed_scale
    return table


def per_threat_params(cfg: EnvConfig, kind: jax.Array) -> dict[str, jax.Array]:
    """Broadcast the kind table out to (T,) arrays indexed by each threat's kind."""
    table = kind_param_table(cfg)
    return {f: table[f][kind] for f in table}


SPAWN_AGL = 3_000.0
"""m AGL an airborne kind is placed at, before its own envelope clamp."""

CONTACT_THRESHOLD = 0.05
"""Track quality that counts as "this threat has a contact".

Deliberately the same number `naigos.rl.red_team.scripted_red` uses to pick a
horizontal target. The vertical channel must not see a blue the horizontal
channel cannot -- that would be omniscience, not pursuit.
"""


def _clamp_altitude(z: jax.Array, ground: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Clamp an airborne altitude into [ground + clearance, ceiling].

    TIE-BREAK: where terrain rises so far that `ground + terrain_clearance`
    exceeds `vehicle_ceiling` the two bounds cross. Terrain wins -- the ceiling
    is lifted to the floor -- because a threat inside a mountain is a worse
    modelling failure than one briefly above its published ceiling, and `clip`
    needs a well-ordered interval to stay jit-safe.
    """
    lo = ground + params["terrain_clearance"]
    hi = jnp.maximum(params["vehicle_ceiling"], lo)
    return jnp.clip(z, lo, hi)


def spawn(key: jax.Array, cfg: EnvConfig, hmap: jax.Array, corridor: jax.Array) -> ThreatState:
    """Place `cfg.n_threat_active` threats; the remainder spawn inactive (padding).

    Threats are biased toward the straight-line corridor between the start and
    the objective so that the naive-direct-route baseline is genuinely punished
    and the policy has something to route around. `corridor` is (2, 3): the mean
    start point and the mean objective.
    """
    T = cfg.n_threat
    k_pos, k_kind, k_off, k_psi = jax.random.split(key, 4)

    t = jax.random.uniform(k_pos, (T,), minval=0.12, maxval=0.88)
    base = corridor[0][None, :2] + t[:, None] * (corridor[1] - corridor[0])[None, :2]
    # lateral scatter around the corridor, wide enough that a dog-leg can work
    offset = jax.random.normal(k_off, (T, 2)) * 22_000.0
    xy = base + offset
    xy = jnp.stack(
        [
            jnp.clip(xy[:, 0], 2_000.0, cfg.terrain.extent_x - 2_000.0),
            jnp.clip(xy[:, 1], 2_000.0, cfg.terrain.extent_y - 2_000.0),
        ],
        axis=-1,
    )

    # kind mix comes from each kind's spawn_weight, so a theatre-derived set of
    # five classes needs no code change here.
    w = jnp.array([k.spawn_weight for k in cfg.threat_kinds], dtype=jnp.float32)
    kind = jax.random.choice(k_kind, jnp.arange(cfg.n_threat_kinds, dtype=jnp.int32), shape=(T,), p=w / w.sum())

    ground = terrain_mod.sample_height(hmap, cfg.terrain, xy[:, 0], xy[:, 1])
    params = per_threat_params(cfg, kind)
    airborne = params["airborne"] > 0.5

    # airborne kinds start on station at SPAWN_AGL, but only where the vehicle
    # could actually be: clamped into its own flight envelope over the local
    # terrain. Ground kinds keep the 5 m pedestal they always had.
    z_air = _clamp_altitude(ground + SPAWN_AGL, ground, params)
    z = jnp.where(airborne, z_air, ground + 5.0)

    active = jnp.arange(T) < cfg.n_threat_active
    psi = jax.random.uniform(k_psi, (T,), minval=-jnp.pi, maxval=jnp.pi)
    return ThreatState(
        pos=jnp.stack([xy[:, 0], xy[:, 1], z], axis=-1),
        psi=psi,
        speed=params["speed"],
        kind=kind,
        active=active,
        # level on station: nothing is tracked yet, so there is nothing to climb
        # or dive toward.
        vz=jnp.zeros((T,), dtype=jnp.float32),
    )


def vertical_command(
    cfg: EnvConfig,
    st: ThreatState,
    blue_pos: jax.Array,  # (B, 3)
    blue_alive: jax.Array,  # (B,) bool
    lock: jax.Array,  # (T, B) track quality at the START of the step
) -> jax.Array:
    """(T,) commanded vertical rate, m/s, +up. Zero for anything not flying.

    This is the vertical half of the scripted red pursuit, kept here rather than
    in `red_team.py` so the RedPolicy interface (psi_cmd, speed_cmd) is
    unchanged and a learned red can be dropped in later without inheriting the
    physics.

    INFORMATION DISCIPLINE. The contact is chosen exactly the way
    `scripted_red` chooses its horizontal target: the highest-track LIVING blue,
    and only if that track clears `CONTACT_THRESHOLD`. With no track the command
    is 0 -- the interceptor holds altitude. It must never climb toward a blue it
    has not detected; that would be omniscience and would make altitude a
    useless axis for blue rather than a contested one.

    The law is a one-step closure, `dz / dt`, saturated by the vehicle's own
    climb and descent limits. Deterministic, no state, no lead term: vertical
    lead prediction is Phase 2.
    """
    params = per_threat_params(cfg, st.kind)

    score = jnp.where(blue_alive[None, :], lock, -1.0)  # (T, B)
    tgt = jnp.argmax(score, axis=-1)  # (T,)
    has_contact = jnp.max(score, axis=-1) > CONTACT_THRESHOLD

    dz = blue_pos[tgt, 2] - st.pos[:, 2]
    vz_des = dz / cfg.dt
    vz = jnp.clip(vz_des, -params["descent_rate_max"], params["climb_rate_max"])

    flying = st.active & (params["airborne"] > 0.5)
    return jnp.where(flying & has_contact, vz, 0.0)


def step(
    cfg: EnvConfig,
    st: ThreatState,
    hmap: jax.Array,
    psi_cmd: jax.Array,
    speed_cmd: jax.Array,
    vz_cmd: jax.Array | None = None,
) -> ThreatState:
    """Advance the threat field one step under commanded heading, speed and
    vertical rate.

    Ground units are pinned to the DEM surface. Airborne units integrate a
    bounded vertical rate over `cfg.dt` and are then clamped into their own
    flight envelope at the UPDATED horizontal position, so running at a ridge
    pushes the vehicle up rather than through it. `vz_cmd` defaults to level
    flight, which keeps every caller that predates the 3D model working.
    """
    params = per_threat_params(cfg, st.kind)
    turn_lim = params["turn_rate"] * cfg.dt
    dpsi = jnp.arctan2(jnp.sin(psi_cmd - st.psi), jnp.cos(psi_cmd - st.psi))
    psi = st.psi + jnp.clip(dpsi, -turn_lim, turn_lim)
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    speed = jnp.clip(speed_cmd, 0.0, params["speed"])
    dx = speed * jnp.cos(psi) * cfg.dt
    dy = speed * jnp.sin(psi) * cfg.dt

    moved = st.active & (params["speed"] > 0.0)
    x = jnp.where(moved, jnp.clip(st.pos[:, 0] + dx, 0.0, cfg.terrain.extent_x), st.pos[:, 0])
    y = jnp.where(moved, jnp.clip(st.pos[:, 1] + dy, 0.0, cfg.terrain.extent_y), st.pos[:, 1])

    # --- vertical channel --------------------------------------------------
    # sampled at the NEW (x, y): the clearance floor that matters is the ground
    # the vehicle is about to be over, not the one it just left.
    ground = terrain_mod.sample_height(hmap, cfg.terrain, x, y)
    airborne = params["airborne"] > 0.5

    if vz_cmd is None:
        vz_cmd = jnp.zeros_like(st.pos[:, 2])
    vz_lim = jnp.clip(vz_cmd, -params["descent_rate_max"], params["climb_rate_max"])
    z_air = _clamp_altitude(st.pos[:, 2] + vz_lim * cfg.dt, ground, params)

    z_new = jnp.where(airborne, z_air, ground + 5.0)
    # an inactive (padded) threat is inert in every axis
    z = jnp.where(st.active, z_new, st.pos[:, 2])

    # report the rate actually ACHIEVED, so a clamp against terrain or ceiling
    # is visible in the state instead of being silently commanded away.
    vz = jnp.where(st.active & airborne, (z - st.pos[:, 2]) / cfg.dt, 0.0)

    return st._replace(
        pos=jnp.stack([x, y, z], axis=-1),
        psi=jnp.where(moved, psi, st.psi),
        speed=jnp.where(moved, speed, 0.0),
        vz=vz,
    )


def kind_onehot(kind: jax.Array, n_kinds: int) -> jax.Array:
    return jax.nn.one_hot(kind, n_kinds, dtype=jnp.float32)
