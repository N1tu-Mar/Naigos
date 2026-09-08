"""Training loop: curriculum-annealed MAPPO-Lagrangian, checkpointed.

Runs identically on a laptop and on a Modal GPU worker -- `modal_train.py` is a
thin wrapper that calls `run()`. Keeping the loop here means the thing that
produces the headline learning curve is the same code path that is unit tested.

Two curricula run at once and they are deliberately coupled:
  * RED difficulty (naigos.rl.red_team.RedCurriculum) rises only once blue is
    surviving, so the task never outruns the policy.
  * REWARD weights (naigos.rl.reward.RewardCurriculum) shift from survival to
    efficiency only once the shootdown rate has fallen.
Both are driven by the same measured statistic (shootdown / survival rate), so
they cannot disagree about how well training is going.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..env.config import EnvConfig
from ..env.flight_env import NaigosEnv
from .ppo import PPOConfig, greedy_policy, init_learner, make_train
from .red_team import RedCurriculum
from .reward import RewardCurriculum, RewardWeights


@dataclasses.dataclass
class TrainConfig:
    use_cbf: bool = False  # run the HOCBF-QP filter inside evaluation rollouts
    iterations: int = 400
    seed: int = 0
    eval_every: int = 20
    eval_worlds: int = 64
    checkpoint_every: int = 50
    out_dir: str = "runs/dev"
    curriculum_every: int = 10


def _exposure_metrics(traj, final, n_worlds: int, n_blue: int) -> dict:
    """Exposure statistics that are not biased by how early a policy dies.

    `mean_exposure` conditioned on being alive FLATTERS a policy that gets shot
    down early: its surviving steps are all near the start of the route, where
    nothing can see it yet. Comparing a trained policy against a baseline on that
    number alone is not a fair comparison, so three statistics are reported:

      * `mean_exposure`      -- over live steps. Kept for continuity; biased.
      * `exposure_early`     -- over the first 100 steps for EVERY agent, alive
                                or not. Same window for every policy, so it is a
                                like-for-like comparison.
      * `exposure_successful`-- over live steps of sorties that reached the
                                objective. Answers "when it works, how exposed
                                was it", which is the operationally useful one.
    """
    # vmap puts the world axis first, then scan's time axis: (W, S, B).
    pd = np.asarray(traj["terms"].exposure)
    live = np.asarray(traj["alive"]).astype(np.float32)
    reached = np.asarray(final.reached).astype(np.float32)  # (W, B)

    win = min(100, pd.shape[1])
    early = pd[:, :win]
    succ_mask = live * reached[:, None, :]
    return {
        "mean_exposure": float((pd * live).sum() / max(live.sum(), 1)),
        "exposure_early": float(early.sum() / max(win * n_worlds * n_blue, 1)),
        "exposure_successful": float((pd * succ_mask).sum() / max(succ_mask.sum(), 1)),
        "cumulative_exposure_per_sortie": float(pd.sum() / max(n_worlds * n_blue, 1)),
    }


def evaluate(env: NaigosEnv, actor_params, n_worlds: int, key, use_cbf: bool = False) -> dict:
    """Deterministic evaluation over full-length episodes."""
    pol = greedy_policy(actor_params, env.cfg)
    afilter = None
    if use_cbf:
        from .cbf import CBFConfig, make_policy_filter

        afilter = make_policy_filter(CBFConfig(), env.cfg)
    final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, pol, action_filter=afilter)))(
        jax.random.split(key, n_worlds)
    )
    live = traj["alive"].astype(np.float32)
    return {
        "survival_rate": float(final.alive.mean()),
        "objective_rate": float(final.reached.mean()),
        "shootdown_rate": float(np.asarray(traj["terms"].shotdown).sum() / (n_worlds * env.cfg.n_blue)),
        **_exposure_metrics(traj, final, n_worlds, env.cfg.n_blue),
        "mean_lock": float((np.asarray(traj["terms"].lock_level) * live).sum() / max(live.sum(), 1)),
        "mean_min_agl": float(np.asarray(traj["alt_agl"]).min(axis=0).mean()),
        "mean_agl_live": float((np.asarray(traj["alt_agl"]) * live).sum() / max(live.sum(), 1)),
        # death-cause breakdown. Without this a falling shootdown rate reads as
        # progress even when the policy has merely swapped being shot down for
        # flying into a hill.
        "terrain_rate": float(np.asarray(traj["terms"].terrain_violation).sum() / (n_worlds * env.cfg.n_blue)),
        "bounds_rate": float(np.asarray(traj["terms"].bounds_violation).sum() / (n_worlds * env.cfg.n_blue)),
        "timeout_rate": float((final.alive & ~final.reached).mean()),
        # A filter that is infeasible most of the time is not a backstop. Report
        # it either way; see next-steps.md S-2.
        "cbf_infeasible_rate": float(
            ((~np.asarray(traj["cbf_feasible"])) * live).sum() / max(live.sum(), 1)
        ),
    }


def baseline(env: NaigosEnv, n_worlds: int, key) -> dict:
    """Both reference controllers: the naive direct route and a competent
    hand-written avoid-plus-nap-of-the-earth heuristic."""

    def direct(obs, k):
        herr = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
        return jnp.stack([jnp.clip(herr * 2.0, -1, 1), jnp.zeros_like(herr), jnp.full_like(herr, 0.6)], -1)

    def avoid_nap(obs, k):
        """A competent hand-written heuristic: route around sensed envelopes and
        fly low. Beating the naive direct route is a weak claim; this is the one
        worth beating (see next-steps.md E-7)."""
        herr = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
        agl = obs.ego[:, 2] * 5000.0
        rng = obs.threats[..., 5] * 90_000.0 + 1e3
        env_r = obs.threats[..., 6] * 90_000.0
        danger = jnp.clip((env_r * 1.8 - rng) / (env_r + 1e-3), 0.0, 1.0) * obs.threat_mask
        push = jnp.sum(-jnp.sign(obs.threats[..., 1]) * danger, axis=-1)
        return jnp.stack([
            jnp.clip(herr * 2.0 + 3.0 * push, -1, 1),
            jnp.clip((400.0 - agl) / 300.0, -1, 1),
            jnp.full_like(herr, 0.6),
        ], -1)

    out = {}
    for name, pol in (("direct", direct), ("avoid_nap", avoid_nap)):
        final, traj = jax.jit(jax.vmap(lambda k2: env.rollout(k2, pol)))(jax.random.split(key, n_worlds))
        m = {
            "survival_rate": float(final.alive.mean()),
            "objective_rate": float(final.reached.mean()),
            "shootdown_rate": float(np.asarray(traj["terms"].shotdown).sum() / (n_worlds * env.cfg.n_blue)),
            **_exposure_metrics(traj, final, n_worlds, env.cfg.n_blue),
        }
        out.update({f"{name}_{k}": v for k, v in m.items()})
    return out


def run(
    env_cfg: EnvConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
    train_cfg: TrainConfig | None = None,
    hmap=None,
):
    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()
    train_cfg = train_cfg or TrainConfig()

    out = Path(train_cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    red_cur = RedCurriculum()
    rew_cur = RewardCurriculum()
    base_w = RewardWeights()

    level = 0.0
    shootdown_rate = 1.0
    survival_rate = 0.0

    cfg = red_cur.apply(env_cfg, level)
    env = NaigosEnv(cfg, hmap=hmap)
    weights, sw, ew = rew_cur.weights_for(base_w, shootdown_rate)

    key = jax.random.PRNGKey(train_cfg.seed)
    key, k_init, k_base = jax.random.split(key, 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo_cfg, sample_obs)

    base = baseline(env, train_cfg.eval_worlds, k_base)
    history = [{"iter": 0, "phase": "baseline", **base}]
    for tag in ("direct", "avoid_nap"):
        print(
            f"[baseline {tag:9s} @ level 0] surv {base[tag+'_survival_rate']:.3f} "
            f"obj {base[tag+'_objective_rate']:.3f} shot {base[tag+'_shootdown_rate']:.3f} "
            f"exp_early {base[tag+'_exposure_early']:.3f}"
        )

    train_step = jax.jit(make_train(env, ppo_cfg, weights))
    t0 = time.time()

    for it in range(1, train_cfg.iterations + 1):
        key, k_step = jax.random.split(key)
        learner, metrics = train_step(learner, k_step)
        metrics = {k: float(v) for k, v in metrics.items()}

        if it % train_cfg.eval_every == 0 or it == 1:
            key, k_eval = jax.random.split(key)
            ev = evaluate(env, learner.actor.params, train_cfg.eval_worlds, k_eval, train_cfg.use_cbf)
            shootdown_rate = ev["shootdown_rate"]
            survival_rate = ev["survival_rate"]
            row = {
                "iter": it,
                "wall_s": round(time.time() - t0, 1),
                "red_level": level,
                "survival_w": sw,
                "efficiency_w": ew,
                **metrics,
                **ev,
            }
            history.append(row)
            print(
                f"[{it:4d}] R {metrics['reward']:8.1f} cost {metrics['cost']:6.3f} "
                f"lam {metrics['lambda']:5.2f} | surv {ev['survival_rate']:.3f} "
                f"obj {ev['objective_rate']:.3f} shot {ev['shootdown_rate']:.3f} "
                f"exp_e {ev['exposure_early']:.3f} agl {ev['mean_agl_live']:.0f} terr {ev['terrain_rate']:.3f} "
                f"oob {ev['bounds_rate']:.3f} | red {level:.2f}"
            )
            (out / "history.json").write_text(json.dumps(history, indent=2))

        # --- curricula ------------------------------------------------------
        if it % train_cfg.curriculum_every == 0:
            # Promote on MEASURED SURVIVAL, not on (1 - shootdown_rate). BUG
            # FOUND IN TRAINING: those are not the same quantity, and a policy
            # that has swapped being shot down for flying into a ridge has a
            # *low* shootdown rate. The curriculum promoted it to level 0.5,
            # survival collapsed from 0.51 to 0.10, and the run never recovered.
            new_level = red_cur.update(level, survival_rate)
            weights, sw, ew = rew_cur.weights_for(base_w, shootdown_rate)
            if new_level != level:
                level = new_level
                cfg = red_cur.apply(env_cfg, level)
                env = NaigosEnv(cfg, hmap=hmap)
                # config changed shape-compatibly, so the params carry over;
                # the jitted step must be rebuilt because cfg is static.
                train_step = jax.jit(make_train(env, ppo_cfg, weights))
            else:
                train_step = jax.jit(make_train(env, ppo_cfg, weights))

        if it % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations:
            with open(out / f"ckpt_{it:06d}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "actor": jax.device_get(learner.actor.params),
                        "critic": jax.device_get(learner.critic.params),
                        "lam_raw": float(learner.lam_raw),
                        "red_level": level,
                        "iter": it,
                        "env_cfg": dataclasses.asdict(env_cfg),
                    },
                    f,
                )

    (out / "history.json").write_text(json.dumps(history, indent=2))
    return learner, history
