"""MAPPO under CTDE, with a PPO-Lagrangian constraint channel.

Shared-parameter actor over local observations; one centralized critic over the
whole scene with two heads (task value, cost value). The Lagrange multiplier
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


def init_learner(key, cfg: EnvConfig, ppo: PPOConfig, sample_obs: Observation) -> Learner:
    k_a, k_c = jax.random.split(key)
    actor = nets.Actor(cfg)
    critic = nets.Critic()

    ap = actor.init(k_a, sample_obs.ego, sample_obs.threats, sample_obs.threat_mask,
                    sample_obs.friends, sample_obs.friend_mask)
    gs = flatten_global(sample_obs)
    cp = critic.init(k_c, gs)

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


def make_train(env: NaigosEnv, ppo: PPOConfig, weights: RewardWeights):
    """Build a jit-able `train_step(learner, key) -> (learner, metrics)`."""
    cfg = env.cfg
    actor_apply = nets.Actor(cfg).apply
    critic_apply = nets.Critic().apply

    def policy_step(params, obs: Observation, key):
        mean, log_std = actor_apply(params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        act, raw = nets.sample_action(mean, log_std, key)
        return act, raw, nets.log_prob(mean, log_std, raw)

    def collect(learner: Learner, key):
        """One vectorised rollout segment across `n_envs` worlds."""

        def one_world(k):
            k_reset, k_roll = jax.random.split(k)
            state, obs = env.reset(k_reset)

            def body(carry, _):
                st, ob, kk = carry
                kk, ka = jax.random.split(kk)
                act, raw, lp = policy_step(learner.actor.params, ob, ka)
                gs = flatten_global(ob)
                v, vc = critic_apply(learner.critic.params, gs)
                st2, ob2, terms, done, info = env.step(st, act)
                r = compute_reward(terms, weights)
                c = constraint_cost(terms)
                agent_done = info["agent_done"].astype(jnp.float32)
                out = dict(
                    ego=ob.ego, threats=ob.threats, threat_mask=ob.threat_mask,
                    friends=ob.friends, friend_mask=ob.friend_mask,
                    gs=gs, raw=raw, logp=lp, v=v, vc=vc, r=r, c=c, d=agent_done,
                    shot=terms.shotdown, arrived=terms.arrived, exposure=terms.exposure,
                    progress=terms.progress,
                )
                return (st2, ob2, kk), out

            (st_f, ob_f, _), traj = jax.lax.scan(body, (state, obs, k_roll), None, length=ppo.n_steps)
            v_last, vc_last = critic_apply(learner.critic.params, flatten_global(ob_f))
            return traj, v_last, vc_last, st_f

        keys = jax.random.split(key, ppo.n_envs)
        traj, v_last, vc_last, st_f = jax.vmap(one_world)(keys)
        # vmap puts env on axis 0; scan put time on axis 1 -> move time first
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return traj, v_last, vc_last, st_f

    def loss_fn(actor_params, critic_params, lam, batch):
        mean, log_std = actor_apply(
            actor_params, batch["ego"], batch["threats"], batch["threat_mask"],
            batch["friends"], batch["friend_mask"],
        )
        logp = nets.log_prob(mean, log_std, batch["raw"])
        ratio = jnp.exp(logp - batch["logp"])

        adv = batch["adv"] - lam * batch["adv_c"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1 - ppo.clip_eps, 1 + ppo.clip_eps) * adv
        pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))
        ent = jnp.mean(nets.entropy(log_std))

        v, vc = critic_apply(critic_params, batch["gs"])
        v_loss = jnp.mean((v - batch["ret"]) ** 2) + jnp.mean((vc - batch["ret_c"]) ** 2)

        total = pg_loss + ppo.vf_coef * v_loss - ppo.ent_coef * ent
        return total, {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent, "ratio": jnp.mean(ratio)}

    def train_step(learner: Learner, key):
        k_roll, k_shuf = jax.random.split(key)
        traj, v_last, vc_last, _ = collect(learner, k_roll)

        # advantages, computed per (env, agent) stream
        adv, ret = gae(traj["r"], traj["v"][..., None] * jnp.ones_like(traj["r"]),
                       traj["d"], v_last[:, None] * jnp.ones(traj["r"].shape[-1]),
                       ppo.gamma, ppo.gae_lambda)
        adv_c, ret_c = gae(traj["c"], traj["vc"][..., None] * jnp.ones_like(traj["c"]),
                           traj["d"], vc_last[:, None] * jnp.ones(traj["c"].shape[-1]),
                           ppo.cost_gamma, ppo.gae_lambda)

        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]) if x.ndim >= 3 else x.reshape(-1), {
            **{k: traj[k] for k in ("ego", "threats", "threat_mask", "friends", "friend_mask", "raw", "logp")},
            "adv": adv, "ret": ret, "adv_c": adv_c, "ret_c": ret_c,
        })
        # the critic is per-world, not per-agent: broadcast its targets to agents
        n_agents = cfg.n_blue
        flat["gs"] = jnp.repeat(traj["gs"].reshape((-1,) + traj["gs"].shape[2:]), n_agents, axis=0)

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
            **jax.tree.map(jnp.mean, aux),
        }
        return Learner(actor=actor, critic=critic, lam_raw=lam_raw, lam_opt=lam_opt), metrics

    return train_step


def greedy_policy(actor_params, cfg: EnvConfig):
    """Deterministic policy for evaluation and for the demo replay."""
    apply = nets.Actor(cfg).apply

    def pol(obs: Observation, key):
        mean, _ = apply(actor_params, obs.ego, obs.threats, obs.threat_mask, obs.friends, obs.friend_mask)
        return mean

    return pol
