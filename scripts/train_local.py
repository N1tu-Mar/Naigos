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
from naigos.rl.train import TrainConfig, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", default="runs/local")
    ap.add_argument("--seed", type=int, default=0)
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
    run(
        EnvConfig(),
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=10, checkpoint_every=30),
        meta=meta,
    )
