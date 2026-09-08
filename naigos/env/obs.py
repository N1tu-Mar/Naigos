"""Per-agent local observation: ego vector + variable-length neighbour sets.

This is a Dec-POMDP. Each aircraft sees only what it can sense: a threat that is
beyond the RWR horizon, or currently terrain-masked and never tracked, simply is
not in the observation. The threat and friendly sets are padded to fixed width
with a boolean mask, and the network (Deep Sets + attention) is permutation
invariant over them -- so the policy does not care whether 1 or 6 threats are up.

All spatial features are expressed in the **ego body frame** (x forward along
the velocity vector's ground track, y left, z up). Ego-frame features are what
make the learned behaviour transfer across map locations and headings.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import EnvConfig
from . import spatial_hash as sh
from . import threats as threats_mod

# feature normalisers (metres / m s^-1). Kept explicit so the verifier and the
# demo can un-normalise without guessing.
POS_SCALE = 50_000.0
VEL_SCALE = 300.0
RANGE_SCALE = 90_000.0


class Observation(NamedTuple):
    ego: jax.Array  # (B, ego_dim)
    threats: jax.Array  # (B, K, threat_feat_dim)
    threat_mask: jax.Array  # (B, K) bool
    friends: jax.Array  # (B, M, friend_feat_dim)
    friend_mask: jax.Array  # (B, M) bool


def to_ego_frame(vec: jax.Array, psi: jax.Array) -> jax.Array:
    """Rotate world (..., 3) into the body frame of an aircraft heading `psi`."""
    c, s = jnp.cos(psi), jnp.sin(psi)
    x = vec[..., 0] * c + vec[..., 1] * s
    y = -vec[..., 0] * s + vec[..., 1] * c
    return jnp.stack([x, y, vec[..., 2]], axis=-1)


def sensed_mask(cfg: EnvConfig, slant: jax.Array, vis: jax.Array, lock: jax.Array) -> jax.Array:
    """(B, T) -- can this aircraft currently perceive this threat?

    Two channels, matching how a real cockpit builds the picture:
      * geometric: within the RWR horizon AND with an unbroken line of sight
        (if terrain masks the radar's beam, it masks the warning receiver too);
      * memory: a threat that already has a track on you has been emitting, so
        it stays on the display while that track lives.
    """
    geometric = (slant.T <= cfg.hash.query_radius) & (vis.T > 0.2)
    remembered = lock.T > 0.05
    return geometric | remembered


def threat_features(
    cfg: EnvConfig,
    blue_pos: jax.Array,  # (B, 3)
    blue_psi: jax.Array,  # (B,)
    tstate: threats_mod.ThreatState,
    tparams: dict[str, jax.Array],
    pd: jax.Array,  # (T, B)
    lock: jax.Array,  # (T, B)
    idx: jax.Array,  # (B, K)
) -> jax.Array:
    """Gather the K selected threats into ego-frame feature vectors."""
    B, K = idx.shape
    tp = tstate.pos[idx]  # (B, K, 3)
    rel = tp - blue_pos[:, None, :]
    rel_e = to_ego_frame(rel, blue_psi[:, None])

    tv = jnp.stack(
        [tstate.speed * jnp.cos(tstate.psi), tstate.speed * jnp.sin(tstate.psi), jnp.zeros_like(tstate.speed)],
        axis=-1,
    )[idx]
    tv_e = to_ego_frame(tv, blue_psi[:, None])

    rng = jnp.linalg.norm(rel, axis=-1)
    envelope = tparams["lethal_range"][idx]

    b = jnp.arange(B)[:, None]
    pd_sel = pd.T[b, idx]
    lock_sel = lock.T[b, idx]

    return jnp.concatenate(
        [
            rel_e / POS_SCALE,  # 3
            tv_e[..., :2] / VEL_SCALE,  # 2
            (rng / RANGE_SCALE)[..., None],  # 1
            (envelope / RANGE_SCALE)[..., None],  # 1
            pd_sel[..., None],  # 1
            lock_sel[..., None],  # 1
            threats_mod.kind_onehot(tstate.kind[idx], cfg.n_threat_kinds),  # n_threat_kinds
        ],
        axis=-1,
    )


def friend_features(
    blue_pos: jax.Array,
    blue_psi: jax.Array,
    blue_vel: jax.Array,
    alive: jax.Array,
    idx: jax.Array,  # (B, M)
) -> jax.Array:
    rel = blue_pos[idx] - blue_pos[:, None, :]
    rel_e = to_ego_frame(rel, blue_psi[:, None])
    v_e = to_ego_frame(blue_vel[idx], blue_psi[:, None])
    rng = jnp.linalg.norm(rel, axis=-1)
    return jnp.concatenate(
        [
            rel_e / POS_SCALE,  # 3
            v_e[..., :2] / VEL_SCALE,  # 2
            (rng / RANGE_SCALE)[..., None],  # 1
            alive[idx][..., None].astype(jnp.float32),  # 1
        ],
        axis=-1,
    )


def build(
    cfg: EnvConfig,
    blue_pos: jax.Array,
    blue_psi: jax.Array,
    blue_gamma: jax.Array,
    blue_phi: jax.Array,
    blue_speed: jax.Array,
    blue_fuel: jax.Array,
    blue_vel: jax.Array,
    alive: jax.Array,
    objective: jax.Array,  # (B, 3)
    alt_agl: jax.Array,  # (B,)
    tstate: threats_mod.ThreatState,
    tparams: dict[str, jax.Array],
    pd: jax.Array,  # (T, B)
    lock: jax.Array,  # (T, B)
    slant: jax.Array,  # (T, B)
    vis: jax.Array,  # (T, B)
) -> Observation:
    af = cfg.airframe
    to_obj = objective - blue_pos
    dist = jnp.linalg.norm(to_obj[:, :2], axis=-1)
    bearing = jnp.arctan2(to_obj[:, 1], to_obj[:, 0])
    herr = jnp.arctan2(jnp.sin(bearing - blue_psi), jnp.cos(bearing - blue_psi))
    phi_max = jnp.arccos(1.0 / af.n_max)

    ego = jnp.stack(
        [
            (blue_speed - af.v_stall) / (af.v_max - af.v_stall),
            blue_pos[:, 2] / af.ceiling,
            jnp.clip(alt_agl / 5_000.0, -1.0, 2.0),
            blue_fuel / af.fuel_init,
            jnp.sin(herr),
            jnp.cos(herr),
            blue_gamma / af.gamma_max,
            blue_phi / phi_max,
            jnp.clip(dist / RANGE_SCALE, 0.0, 4.0),
            jnp.max(lock, axis=0),
        ],
        axis=-1,
    )

    sensed = sensed_mask(cfg, slant, vis, lock)  # (B, T)
    t_idx, t_mask = sh.topk_neighbours(
        cfg.hash,
        cfg.terrain,
        blue_pos[:, :2],
        tstate.pos[:, :2],
        tstate.active,
        cfg.k_threat_obs,
        pair_valid=sensed,
    )
    tf = threat_features(cfg, blue_pos, blue_psi, tstate, tparams, pd, lock, t_idx)
    tf = tf * t_mask[..., None]

    # friendlies: exclude self by marking the query point itself invalid per-row
    not_self = ~jnp.eye(cfg.n_blue, dtype=bool)
    f_idx, f_mask = sh.topk_neighbours(
        cfg.hash,
        cfg.terrain,
        blue_pos[:, :2],
        blue_pos[:, :2],
        alive,
        cfg.k_friend_obs,
        pair_valid=not_self,
    )
    ff = friend_features(blue_pos, blue_psi, blue_vel, alive, f_idx) * f_mask[..., None]

    return Observation(ego=ego, threats=tf, threat_mask=t_mask, friends=ff, friend_mask=f_mask)


def flatten_global(obs: Observation, extra: jax.Array | None = None) -> jax.Array:
    """Centralised-critic input: the whole scene, flattened and shared.

    CTDE -- this is only ever fed to the critic at training time. The actor never
    sees it.
    """
    parts = [obs.ego.reshape(-1), obs.threats.reshape(-1), obs.threat_mask.reshape(-1).astype(jnp.float32)]
    if extra is not None:
        parts.append(extra.reshape(-1))
    return jnp.concatenate(parts)


def known_envelopes(cfg: EnvConfig, obs: Observation, blue_pos: jax.Array, blue_psi: jax.Array):
    """Reconstruct world-frame lethal envelopes from what each aircraft can SEE.

    The CBF filter must not be handed ground truth -- a backstop that keeps you
    out of envelopes you have not detected is not a backstop, it is an oracle.
    So the envelopes come back out of the observation the policy was given, in
    the same padded (B, K) layout, and masked slots carry `active=False`.

    Returns `(centers (B, K, 3), radii (B, K), active (B, K))`.
    """
    rel_e = obs.threats[..., :3] * POS_SCALE  # ego frame
    c, s = jnp.cos(blue_psi)[:, None], jnp.sin(blue_psi)[:, None]
    wx = rel_e[..., 0] * c - rel_e[..., 1] * s
    wy = rel_e[..., 0] * s + rel_e[..., 1] * c
    centers = blue_pos[:, None, :] + jnp.stack([wx, wy, rel_e[..., 2]], axis=-1)
    radii = obs.threats[..., 6] * RANGE_SCALE
    # a zero-radius class (pure surveillance) imposes no keep-out
    return centers, radii, obs.threat_mask & (radii > 1.0)
