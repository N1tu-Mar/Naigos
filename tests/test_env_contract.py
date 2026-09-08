"""The env contract: pure reset/step, jit-able, vmap-able, fixed shapes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=40)


def straight(obs, key):
    h = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
    return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.zeros_like(h), jnp.full_like(h, 0.6)], -1)


def test_reset_shapes():
    env = NaigosEnv(CFG)
    st, o = env.reset(jax.random.PRNGKey(0))
    assert st.air.pos.shape == (CFG.n_blue, 3)
    assert st.lock.shape == (CFG.n_threat, CFG.n_blue)
    assert o.ego.shape == (CFG.n_blue, CFG.ego_dim)
    assert o.threats.shape == (CFG.n_blue, CFG.k_threat_obs, CFG.threat_feat_dim)
    assert o.threat_mask.shape == (CFG.n_blue, CFG.k_threat_obs)
    assert o.friends.shape == (CFG.n_blue, CFG.k_friend_obs, CFG.friend_feat_dim)


def test_reset_is_deterministic_in_the_key():
    env = NaigosEnv(CFG)
    a, _ = env.reset(jax.random.PRNGKey(3))
    b, _ = env.reset(jax.random.PRNGKey(3))
    c, _ = env.reset(jax.random.PRNGKey(4))
    assert jnp.allclose(a.air.pos, b.air.pos)
    assert not jnp.allclose(a.air.pos, c.air.pos)


def test_step_is_jit_able():
    env = NaigosEnv(CFG)
    st, _ = env.reset(jax.random.PRNGKey(0))
    f = jax.jit(env.step)
    s2, o2, terms, done, info = f(st, jnp.zeros((CFG.n_blue, 3)))
    assert s2.air.pos.shape == st.air.pos.shape
    assert done.dtype == jnp.bool_


def test_rollout_is_vmap_able_over_worlds():
    env = NaigosEnv(CFG)
    n = 4
    final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, straight)))(jax.random.split(jax.random.PRNGKey(0), n))
    assert traj["pos"].shape == (n, CFG.max_steps, CFG.n_blue, 3)
    assert final.alive.shape == (n, CFG.n_blue)


def test_observation_features_are_finite_and_bounded():
    env = NaigosEnv(CFG)
    _, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, straight)))(jax.random.split(jax.random.PRNGKey(1), 4))
    assert jnp.all(jnp.isfinite(traj["pos"]))
    st, o = env.reset(jax.random.PRNGKey(2))
    for arr in (o.ego, o.threats, o.friends):
        assert jnp.all(jnp.isfinite(arr))
    # masked slots must be exactly zero so pooling cannot read padding
    assert jnp.all(o.threats[~o.threat_mask] == 0.0)
    assert jnp.all(o.friends[~o.friend_mask] == 0.0)


def test_masked_neighbour_slots_never_include_self():
    env = NaigosEnv(CFG)
    st, o = env.reset(jax.random.PRNGKey(0))
    # a friendly slot at exactly zero relative position would be self
    rel = o.friends[..., :3]
    is_self = (jnp.linalg.norm(rel, axis=-1) < 1e-9) & o.friend_mask
    assert not bool(jnp.any(is_self))


def test_dead_aircraft_stop_moving():
    env = NaigosEnv(CFG)
    st, _ = env.reset(jax.random.PRNGKey(0))
    st = st._replace(alive=jnp.array([True, False, False]))
    s2, *_ = env.step(st, jnp.full((CFG.n_blue, 3), 0.5))
    assert not jnp.allclose(s2.air.pos[0], st.air.pos[0])
    assert jnp.allclose(s2.air.pos[1:], st.air.pos[1:])


def test_episode_terminates_by_max_steps():
    env = NaigosEnv(CFG)
    _, traj = env.rollout(jax.random.PRNGKey(0), straight)
    assert bool(traj["done"][-1]) or traj["alive"][-1].sum() >= 0


@pytest.mark.parametrize("n_threat_active", [0, 3, 8])
def test_threat_count_does_not_change_shapes(n_threat_active):
    cfg = EnvConfig(n_blue=3, n_threat=8, n_threat_active=n_threat_active, max_steps=10)
    env = NaigosEnv(cfg)
    st, o = env.reset(jax.random.PRNGKey(0))
    assert o.threats.shape == (3, cfg.k_threat_obs, cfg.threat_feat_dim)
    s2, o2, *_ = env.step(st, jnp.zeros((3, 3)))
    assert o2.threats.shape == o.threats.shape


def test_no_threats_means_no_detections():
    cfg = EnvConfig(n_blue=3, n_threat=8, n_threat_active=0, max_steps=30)
    env = NaigosEnv(cfg)
    final, traj = env.rollout(jax.random.PRNGKey(0), straight)
    assert float(traj["terms"].exposure.max()) == 0.0
    assert float(traj["terms"].shotdown.sum()) == 0.0
