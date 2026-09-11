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

from naigos.env.config import EnvConfig
from naigos.env.theatre_bridge import describe, env_from_theatre
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
    ap.add_argument("--edge-obs", action="store_true",
                    help="append body-frame map-edge distances to the ego observation "
                         "(EnvConfig.obs_edge_features; the checkpoint records it)")
    ap.add_argument("--resume", action="store_true",
                    help="continue --out from its most recent valid checkpoint")
    ap.add_argument("--override-resume", action="append", default=[], metavar="KEY",
                    help=f"accept one named incompatibility. Keys: "
                         f"{', '.join(runmeta.RESUME_BLOCKING_KEYS)}")
    a = ap.parse_args()

    out = Path(a.out)

    # The run's identity is the directory it writes into. `run.json` is written
    # once; pointing a *different* configuration at an existing run directory is
    # refused rather than allowed to interleave two runs' checkpoints. That check
    # runs BEFORE the theatre is built, so a refused run leaves nothing behind
    # and does not spend a minute loading a DEM first.
    cfg_probe = EnvConfig()
    profile = runmeta.RunProfile(
        name="local", purpose="local CPU run on the cited theatre",
        iterations=a.iterations, n_envs=a.envs, n_steps=a.steps, n_threat=a.threats,
        n_blue=cfg_probe.n_blue, cell_m=a.cell_m,
        eval_every=20, eval_worlds=64, checkpoint_every=100,
    )
    meta = runmeta.build_metadata(
        profile, run_name=out.name, seed=a.seed, synthetic=False, aoi=a.aoi,
        use_cbf=a.cbf, code=runmeta.git_info(Path(__file__).resolve().parents[1]),
        launcher="local",
    )
    resume_from = _resume_checkpoint(out, meta, a.override_resume) if a.resume else None
    try:
        runmeta.write_metadata(out, meta)
    except runmeta.RunCollision as e:
        raise SystemExit(str(e))

    env_overrides = {"obs_edge_features": True} if a.edge_obs else {}
    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, n_threat=a.threats, cell_m=a.cell_m, **env_overrides)
    print(describe(notes))
    (out / runmeta.THEATRE_FILENAME).write_text(json.dumps(notes, indent=2, default=str))

    run(
        cfg,
        PPOConfig(n_envs=a.envs, n_steps=a.steps),
        TrainConfig(iterations=a.iterations, out_dir=a.out, seed=a.seed, eval_every=20,
                    checkpoint_every=100, use_cbf=a.cbf),
        hmap=hmap,
        meta=meta,
        resume_from=resume_from,
    )
