#!/usr/bin/env python
"""Short local training run -- the smoke test for the whole training stack."""
import argparse

from naigos.env.config import EnvConfig
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
    run(
        EnvConfig(),
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=10, checkpoint_every=30),
    )
