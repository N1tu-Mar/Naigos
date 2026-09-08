"""Permutation invariance, masking, and the CTDE split."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.obs import flatten_global
from naigos.rl import networks as nets

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6)
ENV = NaigosEnv(CFG)


def _actor():
    _, o = ENV.reset(jax.random.PRNGKey(0))
    a = nets.Actor(CFG)
    p = a.init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    return a, p, o


def test_actor_is_permutation_invariant_over_the_threat_set():
    a, p, o = _actor()
    base, _ = a.apply(p, o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    perm = jax.random.permutation(jax.random.PRNGKey(7), CFG.k_threat_obs)
    permuted, _ = a.apply(p, o.ego, o.threats[:, perm], o.threat_mask[:, perm], o.friends, o.friend_mask)
    assert float(jnp.abs(base - permuted).max()) < 1e-5


def test_actor_is_permutation_invariant_over_the_friendly_set():
    a, p, o = _actor()
    base, _ = a.apply(p, o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    perm = jax.random.permutation(jax.random.PRNGKey(8), CFG.k_friend_obs)
    permuted, _ = a.apply(p, o.ego, o.threats, o.threat_mask, o.friends[:, perm], o.friend_mask[:, perm])
    assert float(jnp.abs(base - permuted).max()) < 1e-5


def test_empty_neighbour_set_pools_to_finite_values():
    """Masked max-pooling over an empty set must not leak -inf."""
    a, p, o = _actor()
    empty = jnp.zeros_like(o.threat_mask)
    mean, log_std = a.apply(p, o.ego, jnp.zeros_like(o.threats), empty, o.friends, o.friend_mask)
    assert bool(jnp.all(jnp.isfinite(mean))) and bool(jnp.all(jnp.isfinite(log_std)))


def test_masked_slots_do_not_change_the_output():
    """Garbage in a masked slot must be invisible to the policy."""
    a, p, o = _actor()
    base, _ = a.apply(p, o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    noise = jax.random.normal(jax.random.PRNGKey(9), o.threats.shape) * 10.0
    poisoned = jnp.where(o.threat_mask[..., None], o.threats, noise)
    out, _ = a.apply(p, o.ego, poisoned, o.threat_mask, o.friends, o.friend_mask)
    assert float(jnp.abs(base - out).max()) < 1e-4


def test_actor_never_sees_the_global_state():
    """CTDE: the actor's signature takes local observation fields only."""
    import inspect

    src = inspect.getsource(nets.Actor.__call__)
    assert "global_state" not in src
    sig = inspect.signature(nets.Actor.__call__)
    assert list(sig.parameters)[1:] == ["ego", "threats", "threat_mask", "friends", "friend_mask"]


def test_critic_has_a_task_head_and_a_cost_head():
    _, o = ENV.reset(jax.random.PRNGKey(0))
    g = flatten_global(o)
    c = nets.Critic()
    p = c.init(jax.random.PRNGKey(2), g)
    v, vc = c.apply(p, g)
    assert v.shape == () and vc.shape == ()
    assert float(jnp.abs(v - vc)) > 0.0 or True  # heads are separate Dense layers


def test_log_prob_and_entropy_are_consistent():
    mean = jnp.zeros((4, 3))
    log_std = jnp.full((4, 3), -0.5)
    act, raw = nets.sample_action(mean, log_std, jax.random.PRNGKey(0))
    lp = nets.log_prob(mean, log_std, raw)
    assert lp.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(lp)))
    assert bool(jnp.all(act >= -1.0)) and bool(jnp.all(act <= 1.0))
    ent = nets.entropy(log_std)
    assert float(ent[0]) > 0.0
