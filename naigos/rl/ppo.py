"""MAPPO under CTDE, with a PPO-Lagrangian constraint channel.

Shared-parameter actor over local observations; one centralized critic over the
whole scene with two heads (task value, cost value), evaluated once per aircraft
(see `critic_inputs`). Samples of aircraft already killed or arrived are masked
out of every loss. The Lagrange multiplier
prices the CMDP cost channel so "do not get shot down" is a *learned* constraint
rather than a reward weight somebody guessed.

    L_actor = -(A_reward - lambda * A_cost) * ratio  (clipped)
    lambda  <- lambda + lr_lambda * (J_cost - budget)   [projected to >= 0]

The multiplier is stored as a softplus pre-activation so it stays non-negative
without a hard projection step fighting the optimizer.

Everything here is jit-compiled end to end: rollout collection uses
`env.rollout` under `vmap`, so a training iteration is one XLA program.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from ..env.config import EnvConfig
from ..env.flight_env import NaigosEnv
from ..env.obs import Observation, flatten_global
from . import networks as nets
from .reward import RewardWeights, compute as compute_reward, constraint_cost


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 64
    n_steps: int = 128  # rollout length per iteration
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 3e-3
    max_grad_norm: float = 0.5

    # --- CMDP / Lagrangian ---
    # Expected airframe losses per agent-episode. The cost channel is terminal
    # only (see reward.constraint_cost), so this is a rate in [0, 1] and 0.10
    # means "lose at most one sortie in ten". A budget the policy cannot meet
    # turns dual ascent into an ever-growing penalty weight.
    cost_budget: float = 0.10
    lr_lambda: float = 5e-3
    lambda_init: float = 1.0
    cost_gamma: float = 1.0  # terminal cost: no discounting, count the event
    # Proportional term on the constraint violation (Stooke et al., PID
    # Lagrangian). Pure integral control on a constraint that starts far outside
    # the feasible set overshoots and then unwinds slowly; the proportional term
    # responds to the current violation instead of its history.
    lambda_kp: float = 0.5
    lambda_max: float = 25.0  # hard cap, so a mis-set budget cannot run away


class Learner(NamedTuple):
    actor: TrainState
    critic: TrainState
    lam_raw: jnp.ndarray  # softplus pre-activation of the Lagrange multiplier
    lam_opt: Any


def lam_value(lam_raw):
    return jax.nn.softplus(lam_raw)


def critic_inputs(obs: Observation) -> jnp.ndarray:
    """Per-aircraft critic input, `(n_blue, G + ego_dim + n_blue)`.

    Row i is the whole scene (`flatten_global`, width G), aircraft i's own ego
    vector and a one-hot of i. The scene alone is identical for every aircraft
    in a world, so a critic fed only that can give one value per world at best;
    the ego vector and index give each aircraft its own baseline.
    """
    n_blue = obs.ego.shape[0]
    g = flatten_global(obs)
    return jnp.concatenate(
        [jnp.broadcast_to(g, (n_blue, g.shape[0])), obs.ego, jnp.eye(n_blue, dtype=g.dtype)], axis=-1
    )


def masked_mean(x, mask):
    """Mean of `x` over `mask`; zero when nothing is masked in."""
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def init_learner(key, cfg: EnvConfig, ppo: PPOConfig, sample_obs: Observation) -> Learner:
    k_a, k_c = jax.random.split(key)
    actor = nets.Actor(cfg)
    critic = nets.Critic()

    ap = actor.init(k_a, sample_obs.ego, sample_obs.threats, sample_obs.threat_mask,
                    sample_obs.friends, sample_obs.friend_mask)
    cp = critic.init(k_c, critic_inputs(sample_obs))

    tx = optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm), optax.adam(ppo.lr))
    txc = optax.chain(optax.clip_by_global_norm(ppo.max_grad_norm), optax.adam(ppo.lr))
    lam_raw = jnp.asarray(jnp.log(jnp.expm1(jnp.maximum(ppo.lambda_init, 1e-4))), dtype=jnp.float32)
    lam_opt = optax.adam(ppo.lr_lambda)
    return Learner(
        actor=TrainState.create(apply_fn=actor.apply, params=ap, tx=tx),
        critic=TrainState.create(apply_fn=critic.apply, params=cp, tx=txc),
        lam_raw=lam_raw,
        lam_opt=lam_opt.init(lam_raw),
    )


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over a (S, ...) trajectory."""

    def body(carry, x):
        adv, next_value = carry
        r, v, d = x
        nonterm = 1.0 - d
        delta = r + gamma * next_value * nonterm - v
        adv = delta + gamma * lam * nonterm * adv
        return (adv, v), adv

    (_, _), advs = jax.lax.scan(
        body,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    return advs, advs + values


def init_rollout(env: NaigosEnv, ppo: PPOConfig, key):
    """Fresh batched (n_envs) `(EnvState, Observation)` for a persistent rollout.

    `train_step` carries this across iterations, so an episode longer than
    `n_steps` continues into the next segment instead of being cut off.
    """
    return jax.vmap(env.reset)(jax.random.split(key, ppo.n_envs))


def make_train(env: NaigosEnv, ppo: PPOConfig, weights: RewardWeights):
    """Build a jit-able `train_step(learner, rollout, key) -> (learner, rollout, metrics)`."""
    cfg = env.cfg
    actor_apply = nets.Actor(cfg).apply
    critic_apply = nets.Critic().apply
    # one value per aircraft: the critic runs on each row of `critic_inputs`
    critic_per_agent = jax.vmap(critic_apply, in_axes=(None, 0))

    def policy_step(params, obs: Observation, key):
        mean, log_std = actor_apply(params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        act, raw = nets.sample_action(mean, log_std, key)
        return act, raw, nets.log_prob(mean, log_std, raw)

    def collect(learner: Learner, rollout, key):
        """One vectorised rollout segment across `n_envs` worlds, continuing
        from `rollout`. A world whose episode ends is reset in place."""

        def one_world(state, obs, k):
            def body(carry, _):
                st, ob, kk = carry
                kk, ka, kr = jax.random.split(kk, 3)
                act, raw, lp = policy_step(learner.actor.params, ob, ka)
                gs = critic_inputs(ob)
                v, vc = critic_per_agent(learner.critic.params, gs)
                # Recorded BEFORE the step, so the step on which an aircraft is
                # killed or arrives is live; every later sample of it is not.
                live = st.alive & ~st.reached
                st2, ob2, terms, done, info = env.step(st, act)
                r = compute_reward(terms, weights)
                c = constraint_cost(terms)
                # The max_steps timeout is treated as terminal (not bootstrapped).
                agent_done = (info["agent_done"] | done).astype(jnp.float32)
                # auto-reset: a finished world starts a fresh episode in place
                st_r, ob_r = env.reset(kr)
                st2, ob2 = jax.tree.map(lambda a, b: jnp.where(done, a, b), (st_r, ob_r), (st2, ob2))
                out = dict(
                    ego=ob.ego, threats=ob.threats, threat_mask=ob.threat_mask,
                    friends=ob.friends, friend_mask=ob.friend_mask,
                    gs=gs, raw=raw, logp=lp, v=v, vc=vc, r=r, c=c, d=agent_done, live=live,
                    shot=terms.shotdown, arrived=terms.arrived, exposure=terms.exposure,
                    progress=terms.progress, reset=done.astype(jnp.float32),
                )
                return (st2, ob2, kk), out

            (st_f, ob_f, _), traj = jax.lax.scan(body, (state, obs, k), None, length=ppo.n_steps)
            v_last, vc_last = critic_per_agent(learner.critic.params, critic_inputs(ob_f))
            return traj, v_last, vc_last, (st_f, ob_f)

        state, obs = rollout
        keys = jax.random.split(key, ppo.n_envs)
        traj, v_last, vc_last, rollout = jax.vmap(one_world)(state, obs, keys)
        # vmap puts env on axis 0; scan put time on axis 1 -> move time first
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return traj, v_last, vc_last, rollout

    def loss_fn(actor_params, critic_params, lam, batch):
        mean, log_std = actor_apply(
            actor_params, batch["ego"], batch["threats"], batch["threat_mask"],
            batch["friends"], batch["friend_mask"],
        )
        logp = nets.log_prob(mean, log_std, batch["raw"])
        ratio = jnp.exp(logp - batch["logp"])

        # samples of aircraft already killed or arrived carry no signal
        live = batch["live"]
        adv = batch["adv"] - lam * batch["adv_c"]
        adv_mean = masked_mean(adv, live)
        adv_std = jnp.sqrt(masked_mean((adv - adv_mean) ** 2, live))
        adv = (adv - adv_mean) / (adv_std + 1e-8)

        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * adv
        pg_loss = -masked_mean(jnp.minimum(pg1, pg2), live)
        ent = masked_mean(nets.entropy(log_std), live)

        v, vc = critic_apply(critic_params, batch["gs"])
        v_loss = masked_mean((v - batch["ret"]) ** 2, live) + masked_mean((vc - batch["ret_c"]) ** 2, live)

        total = pg_loss + ppo.vf_coef * v_loss - ppo.ent_coef * ent
        return total, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent, "ratio": jnp.mean(ratio)}

    def train_step(learner: Learner, rollout, key):
        k_roll, k_shuf = jax.random.split(key)
        traj, v_last, vc_last, rollout = collect(learner, rollout, k_roll)

        # advantages, computed per (env, agent) stream against per-agent values
        adv, ret = gae(traj["r"], traj["v"], traj["d"], v_last, ppo.gamma, ppo.gae_lambda)
        adv_c, ret_c = gae(traj["c"], traj["vc"], traj["d"], vc_last, ppo.cost_gamma, ppo.gae_lambda)

        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]) if x.ndim >= 3 else x.reshape(-1), {
            **{k: traj[k] for k in ("ego", "threats", "threat_mask", "friends", "friend_mask", "raw", "logp",
                                    "gs", "live")},
            "adv": adv, "ret": ret, "adv_c": adv_c, "ret_c": ret_c,
        })

        n = flat["adv"].shape[0]
        mb = n // ppo.n_minibatches

        def epoch(carry, ek):
            actor, critic = carry
            perm = jax.random.permutation(ek, n)

            def minibatch(carry2, i):
                actor, critic = carry2
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                batch = jax.tree.map(lambda x: x[idx], flat)
                grads, aux = jax.grad(
                    lambda ap, cp: loss_fn(ap, cp, lam_value(learner.lam_raw), batch),
                    argnums=(0, 1), has_aux=True,
                )(actor.params, critic.params)
                actor = actor.apply_gradients(grads=grads[0])
                critic = critic.apply_gradients(grads=grads[1])
                return (actor, critic), aux

            (actor, critic), aux = jax.lax.scan(minibatch, (actor, critic), jnp.arange(ppo.n_minibatches))
            return (actor, critic), aux

        (actor, critic), aux = jax.lax.scan(
            epoch, (learner.actor, learner.critic), jax.random.split(k_shuf, ppo.n_epochs)
        )

        # --- Lagrange multiplier: PI dual ascent on the constraint violation ---
        j_cost = jnp.mean(jnp.sum(traj["c"], axis=0))  # expected losses per agent-segment
        violation = j_cost - ppo.cost_budget
        lam_grad = -violation * jax.nn.sigmoid(learner.lam_raw)  # d(-lam*viol)/d(raw)
        opt = optax.adam(ppo.lr_lambda)
        updates, lam_opt = opt.update(lam_grad, learner.lam_opt, learner.lam_raw)
        lam_raw = optax.apply_updates(learner.lam_raw, updates)
        # proportional term, applied outside the optimizer so it does not
        # accumulate state -- it must be able to vanish the moment the
        # constraint is satisfied.
        lam_eff = jnp.clip(lam_value(lam_raw) + ppo.lambda_kp * jnp.maximum(violation, 0.0), 0.0, ppo.lambda_max)
        lam_raw = jnp.clip(lam_raw, -10.0, jnp.log(jnp.expm1(ppo.lambda_max)))

        # reward-head fit on the live samples, with the values the rollout used
        live = traj["live"]
        ret_var = masked_mean((ret - masked_mean(ret, live)) ** 2, live)
        err = ret - traj["v"]
        err_var = masked_mean((err - masked_mean(err, live)) ** 2, live)

        metrics = {
            "reward": jnp.mean(jnp.sum(traj["r"], axis=0)),
            "cost": j_cost,
            "lambda": lam_eff,
            "lambda_integral": lam_value(lam_raw),
            "cost_violation": violation,
            "shootdown_rate": jnp.mean(jnp.sum(traj["shot"], axis=0)),
            "arrival_rate": jnp.mean(jnp.sum(traj["arrived"], axis=0)),
            "exposure": jnp.mean(traj["exposure"]),
            "progress_km": jnp.mean(jnp.sum(traj["progress"], axis=0)) / 1000.0,
            "episodes_completed": jnp.sum(traj["reset"]),  # world resets in this segment
            "live_frac": jnp.mean(live.astype(jnp.float32)),
            # 0.0 when the live returns have no variance to explain
            "explained_var": jnp.where(ret_var > 0, 1.0 - err_var / jnp.where(ret_var > 0, ret_var, 1.0), 0.0),
            **jax.tree.map(jnp.mean, aux),
        }
        return Learner(actor=actor, critic=critic, lam_raw=lam_raw, lam_opt=lam_opt), rollout, metrics

    train_step.collect = collect  # exposed so tests can inspect the raw segment
    train_step.loss_fn = loss_fn  # exposed so tests can check the live mask
    return train_step


def greedy_policy(actor_params, cfg: EnvConfig):
    """Deterministic policy for evaluation and for the demo replay."""
    apply = nets.Actor(cfg).apply

    def pol(obs: Observation, key):
        mean, _ = apply(actor_params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        return mean

    return pol
