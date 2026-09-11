"""Persistent PPO rollouts: episodes run across iteration boundaries.

The bug this guards against: `collect()` used to reset every world at the start
of every iteration and scan only `n_steps`, so no step past `n_steps` of any
episode was ever trained and the training arrival rate was 0.0 for a whole run.

Everything runs on a small flat map with no active threats, so the only way an
episode ends is the clock, leaving the map or reaching the objective -- which
makes "this world cannot be done yet" a matter of geometry, not luck.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import DetectionConfig, EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.ppo import PPOConfig, gae, init_learner, init_rollout, make_train
from naigos.rl.reward import RewardWeights

# 40 km square, spawn and objective 30% in from the x edges, lateral spawn
# clipped to the middle 10% of y. In 16 steps (32 s) an aircraft covers < 10 km,
# so it can reach neither an edge (>= 12 km) nor the objective (>= 13 km), and
# starting 2500 m over flat ground it cannot descend to the 30 m floor.
SAFE = EnvConfig(
    n_blue=2, n_threat=2, n_threat_active=0,
    terrain=TerrainConfig(nx=21, ny=21, cell=2_000.0),
    detection=DetectionConfig(los_samples=8),
    spawn_inset_frac=0.3,
)
# 16 km square with the default 10% inset: the objective is >= 9.8 km past the
# arrival radius, i.e. more than one 8-step segment away at any airspeed. Only
# a rollout that persists across iterations can ever see an arrival here.
REACH = dataclasses.replace(
    SAFE, terrain=TerrainConfig(nx=17, ny=17, cell=1_000.0), spawn_inset_frac=0.10, max_steps=60,
)
PPO = PPOConfig(n_envs=4, n_steps=8, n_epochs=1, n_minibatches=2)


def _flat(cfg: EnvConfig) -> NaigosEnv:
    return NaigosEnv(cfg, hmap=jnp.zeros((cfg.terrain.ny, cfg.terrain.nx)))


def _setup(cfg: EnvConfig, ppo: PPOConfig = PPO, seed: int = 0):
    env = _flat(cfg)
    key = jax.random.PRNGKey(seed)
    k_init, k_roll, k_run = jax.random.split(key, 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo, sample_obs)
    step = make_train(env, ppo, RewardWeights())
    return env, learner, step, init_rollout(env, ppo, k_roll), k_run


def test_consecutive_train_steps_continue_the_same_episode():
    _, learner, step, rollout, key = _setup(SAFE)
    train_step = jax.jit(step)
    np.testing.assert_array_equal(np.asarray(rollout[0].t), 0)
    for i in range(2):
        key, k = jax.random.split(key)
        learner, rollout, metrics = train_step(learner, rollout, k)
        assert float(metrics["episodes_completed"]) == 0.0, i
    state = rollout[0]
    np.testing.assert_array_equal(np.asarray(state.t), 2 * PPO.n_steps)
    assert bool(np.all(np.asarray(state.alive)))


def _timeout_segment():
    """One 8-step segment on a 6-step episode: every world times out at index 5."""
    cfg = dataclasses.replace(SAFE, max_steps=6)
    _, learner, step, rollout, key = _setup(cfg)
    traj, v_last, _, rollout = jax.jit(step.collect)(learner, rollout, key)
    return cfg, traj, v_last, rollout


def test_a_finished_world_is_reset_inside_the_segment():
    cfg, traj, _, (state, _) = _timeout_segment()
    reset = np.asarray(traj["reset"])  # (S, n_envs)
    np.testing.assert_array_equal(reset[cfg.max_steps - 1], 1.0)
    assert reset.sum() == PPO.n_envs  # exactly once per world
    # the clock restarted and the segment kept going on the fresh episode
    np.testing.assert_array_equal(np.asarray(state.t), PPO.n_steps - cfg.max_steps)
    assert bool(np.all(np.asarray(state.alive)))
    assert not bool(np.any(np.asarray(state.reached)))
    # two steps (4 s) out from the spawn line, not eight
    spawn_x = cfg.spawn_inset_frac * cfg.terrain.extent_x
    assert float(np.max(np.abs(np.asarray(state.air.pos[..., 0]) - spawn_x))) < 4.0 * cfg.airframe.v_max


def test_done_flag_is_set_for_every_agent_when_the_episode_times_out():
    cfg, traj, v_last, _ = _timeout_segment()
    d = np.asarray(traj["d"])  # (S, n_envs, n_blue)
    end = cfg.max_steps - 1
    # every aircraft is alive and short of the objective, yet its stream ends
    np.testing.assert_array_equal(d[end], 1.0)
    np.testing.assert_array_equal(d[:end], 0.0)
    np.testing.assert_array_equal(d[end + 1:], 0.0)

    # ...so GAE does not bootstrap the next episode's value into this one
    values = traj["v"][..., None] * jnp.ones_like(traj["r"])
    adv, _ = gae(traj["r"], values, traj["d"], v_last[:, None] * jnp.ones(traj["r"].shape[-1]),
                 PPO.gamma, PPO.gae_lambda)
    np.testing.assert_allclose(np.asarray(adv[end]), np.asarray(traj["r"][end] - values[end]), rtol=1e-5, atol=1e-4)


def test_arrivals_are_trained_when_the_objective_is_beyond_one_segment():
    _, learner, step, rollout, key = _setup(REACH)
    train_step = jax.jit(step)
    arrivals, episodes = [], []
    for _ in range(8):
        key, k = jax.random.split(key)
        learner, rollout, metrics = train_step(learner, rollout, k)
        arrivals.append(float(metrics["arrival_rate"]))
        episodes.append(float(metrics["episodes_completed"]))
    assert arrivals[0] == 0.0  # geometrically out of reach in the first segment
    assert max(arrivals) > 0.0, arrivals
    assert sum(episodes) > 0.0, episodes
