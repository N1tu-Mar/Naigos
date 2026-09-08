#!/usr/bin/env python
"""Training run on the REAL theatre (cited DEM + calibrated radar classes).

This is the path any reported number must come through. `scripts/train_local.py`
runs on synthetic ridged terrain, which is correct for tests and CI and is not a
real-data result.
"""
import argparse
import json
from pathlib import Path

from naigos.env.theatre_bridge import describe, env_from_theatre
from naigos.rl.ppo import PPOConfig
from naigos.rl.train import TrainConfig, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--threats", type=int, default=16)
    ap.add_argument("--out", default="runs/theatre")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cbf", action="store_true", help="run the HOCBF-QP backstop during evaluation")
    a = ap.parse_args()

    cfg, hmap, notes = env_from_theatre(n_threat=a.threats)
    print(describe(notes))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "theatre.json").write_text(json.dumps(notes, indent=2, default=str))

    run(
        cfg,
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=20,
                    checkpoint_every=100, use_cbf=a.cbf),
        hmap=hmap,
    )
