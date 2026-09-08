"""Red-side state: the threat field that blue must survive.

Threats are *the environment* from blue's point of view (prompt.md s0). Blue
never controls them and never attacks them. This module owns their state, their
kinematics and the per-kind parameter table; the behaviour that picks their
commands lives in `naigos.rl.red_team`.
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
    params0 = per_threat_params(cfg, kind)
    airborne = params0["airborne"]
    z = ground + 5.0 + airborne * 3_000.0

    active = jnp.arange(T) < cfg.n_threat_active
    psi = jax.random.uniform(k_psi, (T,), minval=-jnp.pi, maxval=jnp.pi)
    params = per_threat_params(cfg, kind)
    return ThreatState(
        pos=jnp.stack([xy[:, 0], xy[:, 1], z], axis=-1),
        psi=psi,
        speed=params["speed"],
        kind=kind,
        active=active,
    )


def step(cfg: EnvConfig, st: ThreatState, hmap: jax.Array, psi_cmd: jax.Array, speed_cmd: jax.Array) -> ThreatState:
    """Advance the threat field one step under commanded heading and speed.

    Ground units are pinned to the DEM surface; interceptors hold their commanded
    altitude implicitly by flying level (a 2.5D pursuer -- see next-steps.md, the
    3D interceptor is a known gap).
    """
    params = per_threat_params(cfg, st.kind)
    turn_lim = params["turn_rate"] * cfg.dt
    dpsi = jnp.arctan2(jnp.sin(psi_cmd - st.psi), jnp.cos(psi_cmd - st.psi))
    psi = st.psi + jnp.clip(dpsi, -turn_lim, turn_lim)
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    speed = jnp.clip(speed_cmd, 0.0, params["speed"])
    dx = speed * jnp.cos(psi) * cfg.dt
    dy = speed * jnp.sin(psi) * cfg.dt
    x = jnp.clip(st.pos[:, 0] + dx, 0.0, cfg.terrain.extent_x)
    y = jnp.clip(st.pos[:, 1] + dy, 0.0, cfg.terrain.extent_y)

    ground = terrain_mod.sample_height(hmap, cfg.terrain, x, y)
    airborne = params["airborne"] > 0.5
    z = jnp.where(airborne, jnp.maximum(st.pos[:, 2], ground + 200.0), ground + 5.0)

    moved = st.active & (params["speed"] > 0.0)
    pos = jnp.where(
        moved[:, None],
        jnp.stack([x, y, z], axis=-1),
        st.pos,
    )
    return st._replace(pos=pos, psi=jnp.where(moved, psi, st.psi), speed=jnp.where(moved, speed, 0.0))


def kind_onehot(kind: jax.Array, n_kinds: int) -> jax.Array:
    return jax.nn.one_hot(kind, n_kinds, dtype=jnp.float32)
