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
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..env.config import EnvConfig
from ..env.flight_env import NaigosEnv
from . import checkpoint as ckpt
from . import runmeta
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


def device_report() -> dict:
    """What this process is actually going to run on.

    Printed at the top of every run and written into `perf.json`, because the
    failure mode that matters most on a remote worker is a run that quietly
    executes on CPU: JAX falls back silently, the loop still completes, and the
    resulting wall-clock number is then reported as a GPU number.
    """
    devs = jax.devices()
    return {
        "platform": jax.default_backend(),
        "device_count": len(devs),
        "devices": [f"{d.device_kind}" for d in devs],
        "jax": jax.__version__,
        "python": sys.version.split()[0],
        "host": platform.platform(),
    }


def peak_memory_bytes() -> int | None:
    """Peak device memory across devices, or None if the backend does not report it.

    The CPU backend does not implement `memory_stats`, so this is None for every
    local run. None is reported as null rather than as zero: an unmeasured
    quantity and a measured zero are not the same thing.
    """
    best: int | None = None
    for d in jax.devices():
        fn = getattr(d, "memory_stats", None)
        if fn is None:
            continue
        try:
            stats = fn() or {}
        except Exception:  # pragma: no cover - backend dependent
            continue
        v = stats.get("peak_bytes_in_use")
        if v is not None:
            best = int(v) if best is None else max(best, int(v))
    return best


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


def rollout_metrics(final, traj, n_worlds: int, n_blue: int) -> dict:
    """Every evaluation statistic, from one batch of full-length episodes.

    Split out of `evaluate` so a policy that is not the learner's -- a baseline,
    or a previous champion re-evaluated on the same episodes -- is measured by
    exactly the same arithmetic. `naigos.pipeline.evaluation` depends on that:
    a regression gate comparing numbers produced by two different formulas
    would be comparing the formulas.
    """
    # The expressions below are the ones `evaluate` has always used, verbatim
    # (JAX float32 reductions), so a history produced before this split and one
    # produced after it are bit-identical.
    live = traj["alive"].astype(np.float32)
    return {
        "survival_rate": float(final.alive.mean()),
        "objective_rate": float(final.reached.mean()),
        "shootdown_rate": float(np.asarray(traj["terms"].shotdown).sum() / (n_worlds * n_blue)),
        **_exposure_metrics(traj, final, n_worlds, n_blue),
        "mean_lock": float((np.asarray(traj["terms"].lock_level) * live).sum() / max(live.sum(), 1)),
        "mean_min_agl": float(np.asarray(traj["alt_agl"]).min(axis=0).mean()),
        "mean_agl_live": float((np.asarray(traj["alt_agl"]) * live).sum() / max(live.sum(), 1)),
        # death-cause breakdown. Without this a falling shootdown rate reads as
        # progress even when the policy has merely swapped being shot down for
        # flying into a hill.
        "terrain_rate": float(np.asarray(traj["terms"].terrain_violation).sum() / (n_worlds * n_blue)),
        "bounds_rate": float(np.asarray(traj["terms"].bounds_violation).sum() / (n_worlds * n_blue)),
        "timeout_rate": float((final.alive & ~final.reached).mean()),
        # A filter that is infeasible most of the time is not a backstop. Report
        # it either way; see next-steps.md S-2.
        "cbf_infeasible_rate": float(
            ((~np.asarray(traj["cbf_feasible"])) * live).sum() / max(live.sum(), 1)
        ),
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
    return rollout_metrics(final, traj, n_worlds, env.cfg.n_blue)


def direct_route_policy(obs, k):
    """The naive reference: fly straight at the objective."""
    herr = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
    return jnp.stack([jnp.clip(herr * 2.0, -1, 1), jnp.zeros_like(herr), jnp.full_like(herr, 0.6)], -1)


def avoid_nap_policy(obs, k):
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


BASELINE_POLICIES = {"direct": direct_route_policy, "avoid_nap": avoid_nap_policy}


def baseline(env: NaigosEnv, n_worlds: int, key) -> dict:
    """Both reference controllers: the naive direct route and a competent
    hand-written avoid-plus-nap-of-the-earth heuristic."""
    out = {}
    for name, pol in BASELINE_POLICIES.items():
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
    meta: dict | None = None,
    on_persist: Callable[[], None] | None = None,
    resume_from: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
):
    """Train, and record what it cost.

    `meta`, when given, is written once as an immutable `run.json` (see
    `runmeta.write_metadata`); a second run whose specification differs is
    refused rather than allowed to overwrite it. `on_persist` is called after
    every `history.json`, `perf.json` and checkpoint write -- a Modal Volume
    needs an explicit `commit()`, and without one a six-hour run that hits its
    timeout leaves nothing behind at all. `on_progress` receives the same
    moments as a dict, so a caller can keep a status manifest current without
    the loop knowing what a manifest is.

    `resume_from` is a checkpoint path. Resuming does NOT re-seed, re-baseline
    or reset the curricula: the RNG key, both optimizer states, the multiplier's
    optimizer state, the curriculum level, the measured rates that drive both
    curricula and the accumulated history all come off the checkpoint, so the
    continued run is the same sample path the interrupted one was on. The caller
    is responsible for having decided the resume is legitimate --
    `runmeta.resume_compatibility` and `checkpoint.curriculum_compatibility` are
    the checks, and `scripts/modal_runs.py resume` is where they are applied.
    """
    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()
    train_cfg = train_cfg or TrainConfig()

    out = Path(train_cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def persist(progress: dict | None = None):
        if progress is not None and on_progress is not None:
            try:
                on_progress(progress)
            except Exception as e:  # pragma: no cover - remote storage dependent
                print(f"[progress] FAILED: {e!r}")
        if on_persist is None:
            return
        try:
            on_persist()
        except Exception as e:  # pragma: no cover - remote storage dependent
            # Never lose the training run to a storage error; the next call
            # retries, and `verify_run_dir` catches a truncated result later.
            print(f"[persist] FAILED: {e!r}")

    dev = device_report()
    print(
        f"[device] jax {dev['jax']} backend={dev['platform']} "
        f"x{dev['device_count']} {', '.join(dev['devices']) or 'unknown'}"
    )
    if meta is not None:
        meta = {**meta, "runtime": {**(meta.get("runtime") or {}), **dev}}
        meta, fresh = runmeta.write_metadata(out, meta)
        print(f"[run] {'wrote' if fresh else 'matched existing'} {out / runmeta.META_FILENAME}")
        persist()

    steps_per_iter = ppo_cfg.n_envs * ppo_cfg.n_steps
    perf = {
        "device": dev,
        "n_envs": ppo_cfg.n_envs,
        "n_steps": ppo_cfg.n_steps,
        "n_blue": env_cfg.n_blue,
        "env_steps_per_iteration": steps_per_iter,
        "agent_steps_per_iteration": steps_per_iter * env_cfg.n_blue,
    }
    compile_s: list[float] = []  # iteration 1, then one entry per jit rebuild
    steady_s: list[float] = []  # every iteration that did not recompile
    recompiled = True  # iteration 1 always pays for compilation

    # Filled in below if this is a resume. Everything a resumed run writes is
    # stamped with it, because a `perf.json` whose timings cover only the second
    # half of a run must not read as the timings of the whole run.
    resume_info: dict = {"resumed": False}
    wall_offset = 0.0

    def perf_snapshot(wall_s: float) -> dict:
        # The median over the last 50 steady iterations, not the mean over all
        # of them: a single stall (eval, checkpoint write, a noisy neighbour on
        # a shared host) would otherwise set the reported throughput.
        window = steady_s[-50:]
        med = float(np.median(window)) if window else None
        peak = peak_memory_bytes()
        return {
            **perf,
            **resume_info,
            "wall_s": round(wall_s, 1),
            "first_iteration_s": round(compile_s[0], 3) if compile_s else None,
            "recompiles": max(len(compile_s) - 1, 0),
            "recompile_s_total": round(sum(compile_s[1:]), 2),
            "recompile_s_mean": round(float(np.mean(compile_s[1:])), 3) if len(compile_s) > 1 else None,
            "iterations_timed": len(steady_s),
            "steady_iteration_s_median": round(med, 4) if med else None,
            "steady_iteration_s_p90": (
                round(float(np.percentile(window, 90)), 4) if window else None
            ),
            "env_steps_per_s": round(steps_per_iter / med, 1) if med else None,
            "agent_steps_per_s": round(steps_per_iter * env_cfg.n_blue / med, 1) if med else None,
            "peak_mem_bytes": peak,
            "peak_mem_mb": round(peak / 2**20, 1) if peak is not None else None,
        }

    red_cur = RedCurriculum()
    rew_cur = RewardCurriculum()
    base_w = RewardWeights()

    level = 0.0
    shootdown_rate = 1.0
    survival_rate = 0.0
    start_iter = 0

    blob = ckpt.load(resume_from) if resume_from is not None else None
    if blob is not None:
        rec = blob["recovery"]
        # Restore the curriculum state BEFORE building the env, because the env
        # is a function of the red level and the reward weights are a function
        # of the measured shootdown rate. Rebuilding either from its iteration-1
        # default and then loading parameters into it would resume the policy
        # into a different task.
        level = float(rec["red_level"])
        shootdown_rate = float(rec["shootdown_rate"])
        survival_rate = float(rec["survival_rate"])
        start_iter = int(rec["iteration"])
        wall_offset = float(rec.get("wall_s") or 0.0)

    cfg = red_cur.apply(env_cfg, level)
    env = NaigosEnv(cfg, hmap=hmap)
    weights, sw, ew = rew_cur.weights_for(base_w, shootdown_rate)

    # `k_init` is derived from the seed the same way on a fresh run and on a
    # resume, so the learner a resume grafts onto is structurally the learner the
    # original run built. The key chain itself is then replaced wholesale below.
    key = jax.random.PRNGKey(train_cfg.seed)
    key, k_init, k_base = jax.random.split(key, 3)
    _, sample_obs = env.reset(k_init)
    learner = init_learner(k_init, cfg, ppo_cfg, sample_obs)

    if blob is None:
        base = baseline(env, train_cfg.eval_worlds, k_base)
        history = [{"iter": 0, "phase": "baseline", **base}]
        for tag in ("direct", "avoid_nap"):
            print(
                f"[baseline {tag:9s} @ level 0] surv {base[tag+'_survival_rate']:.3f} "
                f"obj {base[tag+'_objective_rate']:.3f} shot {base[tag+'_shootdown_rate']:.3f} "
                f"exp_early {base[tag+'_exposure_early']:.3f}"
            )
    else:
        learner = ckpt.restore_learner(learner, blob)
        key = ckpt.restore_key(blob)
        # The checkpoint's history, not the one on disk. `history.json` is
        # rewritten at every eval boundary and checkpoints are less frequent, so
        # the file can be AHEAD of the checkpoint -- those iterations are about
        # to be executed again, and keeping their old rows would leave the curve
        # non-monotonic and double-counted.
        history = list(rec["history"])
        chain = list(rec.get("resumed_from") or [])
        chain.append({
            "checkpoint": Path(resume_from).name,
            "from_iteration": start_iter,
            "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": ((meta or {}).get("code") or {}).get("commit"),
        })
        resume_info = {
            "resumed": True,
            "resumed_from_iteration": start_iter,
            "resume_chain": chain,
            "segment_first_iteration": start_iter + 1,
        }
        print(
            f"[resume] {Path(resume_from).name}: continuing at iteration {start_iter + 1} "
            f"of {train_cfg.iterations}, red level {level:.2f}, "
            f"shootdown {shootdown_rate:.3f}, {len(history)} history rows carried forward"
        )
        if start_iter >= train_cfg.iterations:
            print(
                f"[resume] the checkpoint is already at iteration {start_iter} of "
                f"{train_cfg.iterations}: nothing to continue."
            )
            return learner, history

    train_step = jax.jit(make_train(env, ppo_cfg, weights))
    t0 = time.time() - wall_offset

    for it in range(start_iter + 1, train_cfg.iterations + 1):
        key, k_step = jax.random.split(key)
        t_it = time.perf_counter()
        learner, metrics = train_step(learner, k_step)
        # JAX dispatch is asynchronous, so the iteration is not over until a
        # value is pulled back to the host. This conversion is that sync point,
        # which is why the timer closes after it and not before.
        metrics = {k: float(v) for k, v in metrics.items()}
        it_s = time.perf_counter() - t_it
        if recompiled:
            compile_s.append(it_s)
            recompiled = False
            if it == 1:
                print(f"[compile] first iteration {it_s:.1f} s (XLA compile + one step)")
        else:
            steady_s.append(it_s)

        # The final iteration always evaluates, so `history.json` ends at the
        # iteration the run claims to have reached. Without it a completed run
        # and one that stopped after its last eval boundary look identical.
        if it % train_cfg.eval_every == 0 or it == 1 or it == train_cfg.iterations:
            key, k_eval = jax.random.split(key)
            ev = evaluate(env, learner.actor.params, train_cfg.eval_worlds, k_eval, train_cfg.use_cbf)
            shootdown_rate = ev["shootdown_rate"]
            survival_rate = ev["survival_rate"]
            snap = perf_snapshot(time.time() - t0)
            row = {
                "iter": it,
                # Every row produced after a resume says so, so a spliced curve
                # is legible as one in the raw file and not only in `perf.json`.
                **({"resumed_from_iteration": start_iter} if resume_info["resumed"] else {}),
                "wall_s": round(time.time() - t0, 1),
                "red_level": level,
                "survival_w": sw,
                "efficiency_w": ew,
                # cost of the run, logged next to the result of the run, so a
                # curve and the throughput that produced it cannot drift apart
                "iter_s": snap["steady_iteration_s_median"],
                "env_steps_per_s": snap["env_steps_per_s"],
                "peak_mem_mb": snap["peak_mem_mb"],
                "recompiles": snap["recompiles"],
                **metrics,
                **ev,
            }
            history.append(row)
            med, eps = snap["steady_iteration_s_median"], snap["env_steps_per_s"]
            # iteration 1 has only compiled, so there is no steady-state sample
            # yet; say so rather than printing a nan.
            rate = f"{med:.2f} s/it {eps / 1e3:.1f}k env-step/s" if med and eps else "still compiling"
            print(
                f"[{it:4d}] R {metrics['reward']:8.1f} cost {metrics['cost']:6.3f} "
                f"lam {metrics['lambda']:5.2f} | surv {ev['survival_rate']:.3f} "
                f"obj {ev['objective_rate']:.3f} shot {ev['shootdown_rate']:.3f} "
                f"exp_e {ev['exposure_early']:.3f} agl {ev['mean_agl_live']:.0f} terr {ev['terrain_rate']:.3f} "
                f"oob {ev['bounds_rate']:.3f} | red {level:.2f} | {rate}"
            )
            (out / runmeta.HISTORY_FILENAME).write_text(json.dumps(history, indent=2))
            runmeta.write_json(out / runmeta.PERF_FILENAME, snap)
            persist({"last_iteration": it, "iterations_declared": train_cfg.iterations,
                     "event": "eval", **{k: v for k, v in resume_info.items() if k != "resume_chain"}})

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
            # Every rebuild costs a full XLA compile (next-steps.md G-4). Mark
            # the next iteration so its time is booked as compilation rather
            # than silently inflating the reported throughput.
            recompiled = True

        if it % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations:
            # `key` is saved at the END of the iteration, after every split this
            # iteration performed, so a resume continues the same stream rather
            # than replaying part of it. Everything else needed to continue --
            # both optimizer states, the multiplier's optimizer state, both
            # curricula and the accumulated history -- goes with it; see
            # `naigos/rl/checkpoint.py` for why each one is not optional.
            ckpt.save(
                out / f"ckpt_{it:06d}.pkl",
                learner=learner,
                key=key,
                iteration=it,
                red_level=level,
                shootdown_rate=shootdown_rate,
                survival_rate=survival_rate,
                survival_w=sw,
                efficiency_w=ew,
                history=history,
                env_cfg=env_cfg,
                ppo_cfg=ppo_cfg,
                red_curriculum=red_cur,
                reward_curriculum=rew_cur,
                reward_weights_base=base_w,
                wall_s=time.time() - t0,
                run_name=(meta or {}).get("run_name"),
                code_commit=((meta or {}).get("code") or {}).get("commit"),
                resumed_from=resume_info.get("resume_chain"),
            )
            persist({"last_iteration": it, "iterations_declared": train_cfg.iterations,
                     "event": "checkpoint"})

    (out / runmeta.HISTORY_FILENAME).write_text(json.dumps(history, indent=2))
    final_perf = perf_snapshot(time.time() - t0)
    runmeta.write_json(out / runmeta.PERF_FILENAME, final_perf)
    persist({"last_iteration": train_cfg.iterations,
             "iterations_declared": train_cfg.iterations, "event": "final"})
    print(
        f"[perf] {final_perf['iterations_timed']} timed iterations, "
        f"median {final_perf['steady_iteration_s_median']} s/it, "
        f"{final_perf['env_steps_per_s']} env-step/s, "
        f"first iteration {final_perf['first_iteration_s']} s, "
        f"{final_perf['recompiles']} recompiles costing {final_perf['recompile_s_total']} s, "
        f"peak device memory {final_perf['peak_mem_mb']} MB"
    )
    return learner, history
