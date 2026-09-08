"""Learning-delta demo: replay REAL logged rollouts, untrained vs trained.

Nomos's honesty rule, kept: nothing here is scripted. Both trajectories come out
of `env.rollout` with the same env, the same seeds and the same threat field --
only the actor parameters differ. The untrained side is the freshly initialised
network, not a hand-written strawman.

    python -m naigos.demo.replay --checkpoint runs/theatre1/ckpt_000700.pkl

Produces `demo.json` (the raw trajectories + counters) and, if matplotlib is
available, a side-by-side plan view over the DEM with the threat envelopes drawn.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..env.flight_env import NaigosEnv
from ..rl.networks import Actor
from ..rl.ppo import greedy_policy


def direct_route_policy(obs, key):
    """The naive baseline everything is measured against."""
    h = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
    return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.zeros_like(h), jnp.full_like(h, 0.6)], -1)


def _summary(final, traj, cfg):
    live = np.asarray(traj["alive"]).astype(np.float32)
    denom = max(live.sum(), 1.0)
    return {
        "survival_rate": float(final.alive.mean()),
        "objective_rate": float(final.reached.mean()),
        "shootdowns": int(np.asarray(traj["terms"].shotdown).sum()),
        "terrain_losses": int(np.asarray(traj["terms"].terrain_violation).sum()),
        "bounds_losses": int(np.asarray(traj["terms"].bounds_violation).sum()),
        "mean_detection_prob": float((np.asarray(traj["terms"].exposure) * live).sum() / denom),
        "mean_track_quality": float((np.asarray(traj["terms"].lock_level) * live).sum() / denom),
        "mean_min_agl_m": float(np.asarray(traj["alt_agl"]).min(axis=0).mean()),
        "cbf_infeasible_rate": float(((~np.asarray(traj["cbf_feasible"])) * live).sum() / denom),
    }


def run(env: NaigosEnv, trained_params, n_worlds: int, seed: int, use_cbf: bool = False):
    """Roll out untrained, trained and the direct-route baseline on IDENTICAL seeds."""
    cfg = env.cfg
    keys = jax.random.split(jax.random.PRNGKey(seed), n_worlds)

    untrained = Actor(cfg).init(
        jax.random.PRNGKey(12345),
        *(lambda o: (o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask))(env.reset(keys[0])[1]),
    )

    afilter = None
    if use_cbf:
        from ..rl.cbf import CBFConfig, make_policy_filter

        afilter = make_policy_filter(CBFConfig(), cfg)

    out = {}
    for name, pol in (
        ("untrained", greedy_policy(untrained, cfg)),
        ("trained", greedy_policy(trained_params, cfg)),
        ("direct_route_baseline", direct_route_policy),
    ):
        final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, pol, action_filter=afilter)))(keys)
        out[name] = {"summary": _summary(final, traj, cfg), "traj": traj, "final": final}
    return out


def to_json(results, cfg, path: Path, world: int = 0):
    """Dump one world's trajectories plus every summary. This is the artifact the
    viewer reads, so it must contain the ACTUAL logged positions."""
    payload = {"summaries": {k: v["summary"] for k, v in results.items()}, "worlds": {}}
    for name, r in results.items():
        payload["worlds"][name] = {
            "pos": np.asarray(r["traj"]["pos"][world]).tolist(),
            "alive": np.asarray(r["traj"]["alive"][world]).tolist(),
            "reached": np.asarray(r["traj"]["reached"][world]).tolist(),
            "lock": np.asarray(r["traj"]["lock"][world]).max(axis=1).tolist(),
            "threat_pos": np.asarray(r["traj"]["threat_pos"][world][0]).tolist(),
        }
    payload["threat_kinds"] = [k.label for k in cfg.threat_kinds]
    payload["extent_m"] = [cfg.terrain.extent_x, cfg.terrain.extent_y]
    path.write_text(json.dumps(payload))
    return path


def plot(results, env: NaigosEnv, path: Path, world: int = 0):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (pip install -e '.[demo]'); skipping the figure")
        return None

    cfg = env.cfg
    hmap = np.asarray(env.fixed_hmap if env.fixed_hmap is not None else results["trained"]["final"].hmap[world])
    ex, ey = cfg.terrain.extent_x / 1000.0, cfg.terrain.extent_y / 1000.0

    names = ["untrained", "trained", "direct_route_baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    for ax, name in zip(axes, names):
        r = results[name]
        ax.imshow(hmap, origin="lower", extent=[0, ex, 0, ey], cmap="terrain", alpha=0.85)

        tp = np.asarray(r["traj"]["threat_pos"][world][0])
        kinds = np.asarray(results["trained"]["final"].threats.kind[world])
        active = np.asarray(results["trained"]["final"].threats.active[world])
        for i, (p, k) in enumerate(zip(tp, kinds)):
            if not active[i]:
                continue
            lethal = cfg.threat_kinds[int(k)].lethal_range * cfg.red_lethal_scale / 1000.0
            if lethal > 0:
                ax.add_patch(plt.Circle((p[0] / 1000, p[1] / 1000), lethal, color="red", alpha=0.13, lw=0))
            ax.plot(p[0] / 1000, p[1] / 1000, "r^", ms=5)

        pos = np.asarray(r["traj"]["pos"][world])  # (S, B, 3)
        alive = np.asarray(r["traj"]["alive"][world])
        for b in range(cfg.n_blue):
            n = int(alive[:, b].sum()) if not alive[:, b].all() else len(alive)
            ax.plot(pos[:n, b, 0] / 1000, pos[:n, b, 1] / 1000, lw=1.8)
            if n < len(alive):
                ax.plot(pos[n - 1, b, 0] / 1000, pos[n - 1, b, 1] / 1000, "kx", ms=9, mew=2)

        s = r["summary"]
        ax.set_title(
            f"{name}\nsurvived {s['survival_rate']:.0%}   objective {s['objective_rate']:.0%}   "
            f"shootdowns {s['shootdowns']}\nmean detection prob {s['mean_detection_prob']:.3f}"
        )
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
        ax.set_xlim(0, ex)
        ax.set_ylim(0, ey)

    fig.suptitle("Naigos learning delta -- logged rollouts, identical seeds and threat field", fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic terrain instead of the cited DEM")
    ap.add_argument("--cbf", action="store_true", help="run the HOCBF-QP safety backstop")
    a = ap.parse_args(argv)

    with open(a.checkpoint, "rb") as f:
        ck = pickle.load(f)

    if a.synthetic:
        from ..env.config import EnvConfig

        cfg, hmap = EnvConfig(), None
    else:
        from ..env.theatre_bridge import describe, env_from_theatre

        cfg, hmap, notes = env_from_theatre()
        print(describe(notes))

    from ..rl.red_team import RedCurriculum

    cfg = RedCurriculum().apply(cfg, ck.get("red_level", 0.0))
    env = NaigosEnv(cfg, hmap=hmap)

    results = run(env, ck["actor"], a.worlds, a.seed, use_cbf=a.cbf)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, r in results.items():
        print(f"{name:24s} {r['summary']}")
    print("wrote", to_json(results, cfg, out / "demo.json"))
    p = plot(results, env, out / "learning_delta.png")
    if p:
        print("wrote", p)

    (out / "summary.json").write_text(json.dumps({k: v["summary"] for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
