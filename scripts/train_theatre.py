#!/usr/bin/env python
"""Training run on the REAL theatre (cited DEM + calibrated radar classes).

This is the path any reported number must come through. `scripts/train_local.py`
runs on synthetic ridged terrain, which is correct for tests and CI and is not a
real-data result.

Every flag and default that existed before still means what it meant. What is
new is that the run writes an immutable `run.json` describing itself and a
`perf.json` recording device, iteration time, throughput and peak memory, so a
local run and a Modal run are verified by the same command:

    uv run python scripts/modal_runs.py verify runs/theatre
"""
import argparse
import json
from pathlib import Path

from naigos.env.theatre_bridge import describe, env_from_theatre
from naigos.rl import runmeta
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
    ap.add_argument("--aoi", default=None, help="theatre to train on (default: the packaged one)")
    ap.add_argument("--cell-m", type=float, default=1500.0,
                    help="terrain grid cell size (m). Published numbers use 500")
    a = ap.parse_args()

    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, n_threat=a.threats, cell_m=a.cell_m)
    print(describe(notes))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / runmeta.THEATRE_FILENAME).write_text(json.dumps(notes, indent=2, default=str))

    # The run's identity is the directory it writes into. `run.json` is written
    # once; pointing a *different* configuration at an existing run directory is
    # refused rather than allowed to interleave two runs' checkpoints.
    profile = runmeta.RunProfile(
        name="local", purpose="local CPU run on the cited theatre",
        iterations=a.iterations, n_envs=a.envs, n_steps=a.steps, n_threat=a.threats,
        n_blue=cfg.n_blue, cell_m=a.cell_m,
        eval_every=20, eval_worlds=64, checkpoint_every=100,
    )
    meta = runmeta.build_metadata(
        profile, run_name=out.name, seed=a.seed, synthetic=False, aoi=a.aoi,
        use_cbf=a.cbf, code=runmeta.git_info(Path(__file__).resolve().parents[1]),
        launcher="local",
    )

    run(
        cfg,
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=20,
                    checkpoint_every=100, use_cbf=a.cbf),
        hmap=hmap,
        meta=meta,
    )
