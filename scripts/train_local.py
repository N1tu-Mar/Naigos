#!/usr/bin/env python
"""Short local training run -- the smoke test for the whole training stack.

Synthetic ridged terrain: fast, needs no cache, and is not a real-data result.
Writes the same `run.json` / `perf.json` a Modal run does, so
`scripts/modal_runs.py verify runs/local` checks it the same way.
"""
import argparse
from pathlib import Path

from naigos.env.config import EnvConfig
from naigos.rl import runmeta
from naigos.rl.ppo import PPOConfig
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights
from naigos.rl.train import TrainConfig, run


def _resume_checkpoint(out, meta, overrides):
    """Resolve `--resume` into a checkpoint path, or refuse.

    Same rules as the Modal path (`scripts/modal_runs.py resume`): the run must
    already describe itself, the requested configuration must be a continuation
    of the recorded one, and each accepted difference must be named. A resume
    that quietly changes the theatre or the seed produces a `history.json` whose
    halves came from different experiments.
    """
    from naigos.rl import checkpoint as ckpt

    existing = runmeta.read_metadata(out)
    compat = runmeta.resume_compatibility(existing, meta, overrides=overrides)
    if not compat["ok"]:
        raise SystemExit(
            "refusing to resume:\n"
            + "\n".join(f"  - {b}" for b in compat["blocking"])
            + "\n\nIf a change is intentional, name each key: "
            + " ".join(f"--override-resume {k}" for k in sorted(compat["changes"]))
        )
    for note in compat["overridden"]:
        print(f"[resume] OVERRIDDEN: {note}")
    path = ckpt.latest_resumable(out)
    if path is None:
        raise SystemExit(
            f"no resumable checkpoint in {out}. Checkpoints written before resume support "
            "hold the policy but not the optimizer state, the RNG stream or the curriculum "
            "state, so this run has to be restarted rather than continued."
        )
    problems = ckpt.curriculum_compatibility(
        ckpt.load(path),
        red_curriculum=RedCurriculum(),
        reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(),
    )
    if problems and "curriculum" not in overrides:
        raise SystemExit(
            "refusing to resume:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nPass --override-resume curriculum to accept it."
        )
    for p in problems:
        print(f"[resume] OVERRIDDEN: {p}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", default="runs/local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue --out from its most recent valid checkpoint")
    ap.add_argument("--override-resume", action="append", default=[], metavar="KEY",
                    help=f"accept one named incompatibility. Keys: "
                         f"{', '.join(runmeta.RESUME_BLOCKING_KEYS)}")
    a = ap.parse_args()
    cfg = EnvConfig()
    profile = runmeta.RunProfile(
        name="local-synthetic", purpose="synthetic-terrain smoke test of the training stack",
        iterations=a.iterations, n_envs=a.envs, n_steps=a.steps, n_threat=cfg.n_threat,
        n_blue=cfg.n_blue, cell_m=cfg.terrain.cell,
        eval_every=10, eval_worlds=64, checkpoint_every=30,
    )
    meta = runmeta.build_metadata(
        profile, run_name=Path(a.out).name, seed=a.seed, synthetic=True,
        code=runmeta.git_info(Path(__file__).resolve().parents[1]), launcher="local",
    )
    resume_from = _resume_checkpoint(Path(a.out), meta, a.override_resume) if a.resume else None
    try:
        runmeta.write_metadata(Path(a.out), meta)
    except runmeta.RunCollision as e:
        raise SystemExit(str(e))
    run(
        EnvConfig(),
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=10, checkpoint_every=30),
        meta=meta,
        resume_from=resume_from,
    )
