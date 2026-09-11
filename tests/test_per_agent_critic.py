"""Per-aircraft value baseline, and finished aircraft masked out of the loss.

The bug this guards against: the critic saw only the flattened scene and gave
one value per world, broadcast to every aircraft, and samples of aircraft that
were already shot down or had arrived stayed in the policy and value losses.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.obs import flatten_global
from naigos.rl import networks as nets
from naigos.rl.ppo import PPOConfig, critic_inputs, init_learner, init_rollout, make_train
from naigos.rl.reward import RewardWeights

# Small flat map with no active threats: nothing kills an aircraft unless the
# test puts it somewhere lethal. Same geometry as tests/test_persistent_rollouts.
SAFE = EnvConfig(
    n_blue=2, n_threat=2, n_threat_active=0,
    terrain=TerrainConfig(nx=21, ny=21, cell=2_000.0),
    detection=DetectionConfig(los_samples=8),
    spawn_inset_frac=0.3,
)
PPO = PPOConfig(n_envs=4, n_steps=8, n_epochs=1, n_minibatches=2)


def _setup(seed: int = 0):
    env = NaigosEnv(SAFE, hmap=jnp.zeros((SAFE.terrain.ny, SAFE.terrain.nx)))
    k_init, k_roll, k_run = jax.random.split(jax.random.PRNGKey(seed), 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, SAFE, PPO, sample_obs)
    step = make_train(env, PPO, RewardWeights())
    return env, learner, step, init_rollout(env, PPO, k_roll), k_run


def test_critic_inputs_shape():
    env, *_ = _setup()
    _, obs = env.reset(jax.random.PRNGKey(3))
    g = flatten_global(obs).shape[0]
    assert critic_inputs(obs).shape == (SAFE.n_blue, g + SAFE.ego_dim + SAFE.n_blue)


def test_critic_gives_each_aircraft_its_own_value():
    env, learner, *_ = _setup()
    _, obs = env.reset(jax.random.PRNGKey(3))
    assert float(jnp.abs(obs.ego[0] - obs.ego[1]).max()) > 0.0  # same world, different egos
    v, vc = jax.vmap(nets.Critic().apply, in_axes=(None, 0))(learner.critic.params, critic_inputs(obs))
    assert v.shape == vc.shape == (SAFE.n_blue,)
    assert float(jnp.abs(v[0] - v[1])) > 0.0
    assert float(jnp.abs(vc[0] - vc[1])) > 0.0


def _batch(env, key):
    """A synthetic minibatch of 4 worlds x n_blue aircraft, half of it masked."""
    k_obs, k_raw, k_lp, k_a, k_r, k_ac, k_rc, k_live = jax.random.split(key, 8)
    _, obs = jax.vmap(env.reset)(jax.random.split(k_obs, 4))
    n = 4 * SAFE.n_blue
    flat = lambda x: x.reshape((n,) + x.shape[2:])  # noqa: E731
    return {
        "ego": flat(obs.ego), "threats": flat(obs.threats), "threat_mask": flat(obs.threat_mask),
        "friends": flat(obs.friends), "friend_mask": flat(obs.friend_mask),
        "gs": flat(jax.vmap(critic_inputs)(obs)),
        "raw": jax.random.normal(k_raw, (n, SAFE.action_dim)) * 0.3,
        "logp": jax.random.normal(k_lp, (n,)) - 2.0,
        "adv": jax.random.normal(k_a, (n,)), "ret": jax.random.normal(k_r, (n,)),
        "adv_c": jax.random.normal(k_ac, (n,)), "ret_c": jax.random.normal(k_rc, (n,)),
        "live": jnp.arange(n) % 2 == 0,
    }


def _loss_and_grads(step, learner, batch):
    return jax.value_and_grad(
        lambda ap, cp: step.loss_fn(ap, cp, 1.5, batch)[0], argnums=(0, 1)
    )(learner.actor.params, learner.critic.params)


def _perturb(batch, where):
    """Add large offsets to the advantages, returns and critic inputs (hence
    the predicted values) of the samples selected by `where`."""
    out = dict(batch)
    for k, big in (("adv", 1e4), ("adv_c", -1e4), ("ret", 1e5), ("ret_c", -1e5)):
        out[k] = jnp.where(where, batch[k] + big, batch[k])
    out["gs"] = jnp.where(where[:, None], batch["gs"] + 1e3, batch["gs"])
    return out


def test_masked_samples_do_not_affect_the_loss_or_its_gradients():
    env, learner, step, *_ = _setup()
    batch = _batch(env, jax.random.PRNGKey(5))
    loss, grads = _loss_and_grads(step, learner, batch)
    loss_p, grads_p = _loss_and_grads(step, learner, _perturb(batch, ~batch["live"]))

    assert np.isfinite(float(loss))
    np.testing.assert_allclose(float(loss_p), float(loss), rtol=1e-6)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(grads_p)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-5, atol=1e-7)

    # ...and the same perturbation on live samples does move it
    loss_l, _ = _loss_and_grads(step, learner, _perturb(batch, batch["live"]))
    assert abs(float(loss_l) - float(loss)) > 1.0


def test_no_live_samples_gives_a_finite_zero_loss():
    env, learner, step, *_ = _setup()
    batch = {**_batch(env, jax.random.PRNGKey(6)), "live": jnp.zeros(4 * SAFE.n_blue, dtype=bool)}
    loss, grads = _loss_and_grads(step, learner, batch)
    # only the entropy bonus term is left, and its masked mean is zero too
    assert float(loss) == 0.0
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))


def _kill_aircraft_zero(rollout):
    """Put aircraft 0 of every world 5 km off the west edge: out of bounds on
    the first step, whatever it does. Aircraft 1 is untouched."""
    state, obs = rollout
    pos = state.air.pos.at[:, 0, 0].set(-5_000.0)
    return state._replace(air=state.air._replace(pos=pos)), obs


def test_the_step_an_aircraft_dies_on_is_live_and_the_next_is_not():
    _, learner, step, rollout, key = _setup()
    traj, *_ = jax.jit(step.collect)(learner, _kill_aircraft_zero(rollout), key)
    live = np.asarray(traj["live"])  # (S, n_envs, n_blue)
    d = np.asarray(traj["d"])
    assert live.dtype == bool and live.shape == (PPO.n_steps, PPO.n_envs, SAFE.n_blue)

    # aircraft 0 is lost on step 0: that step is in the loss, the rest are not
    np.testing.assert_array_equal(d[0, :, 0], 1.0)
    np.testing.assert_array_equal(live[0, :, 0], True)
    np.testing.assert_array_equal(live[1:, :, 0], False)
    # aircraft 1 flies the whole segment
    np.testing.assert_array_equal(live[:, :, 1], True)
    np.testing.assert_array_equal(np.asarray(traj["reset"]), 0.0)


def test_train_step_reports_live_fraction_and_explained_variance():
    _, learner, step, rollout, key = _setup()
    _, _, metrics = jax.jit(step)(learner, _kill_aircraft_zero(rollout), key)
    # aircraft 0: its one death step; aircraft 1: all n_steps
    expect = (1 + PPO.n_steps) / (SAFE.n_blue * PPO.n_steps)
    np.testing.assert_allclose(float(metrics["live_frac"]), expect, rtol=1e-6)
    assert np.isfinite(float(metrics["explained_var"]))
    assert float(metrics["explained_var"]) <= 1.0
