"""Modal GPU wrapper around `naigos.rl.train.run`.

Deliberately thin: the loop that produces the headline learning curve is the
same code path the tests exercise locally. This file only owns the image, the
Volume and the entrypoint.

    modal run naigos/rl/modal_train.py --iterations 3000

NOT YET RUN ON MODAL. See next-steps.md gap C-1: the image below installs the
dependencies and mounts a Volume, but no GPU run has been executed, so the
throughput and wall-clock numbers a training report would need do not exist yet.
The local CPU path is what every measured number in this repo comes from.
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError:  # pragma: no cover - modal is not a hard dependency
    modal = None

APP_NAME = "naigos"
VOLUME_NAME = "naigos-runs"

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "jax[cuda12]>=0.4.34",
            "flax>=0.10",
            "optax>=0.2.3",
            "numpy>=1.26",
            extra_index_url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        )
        # the cited cache and the component specs travel with the image so the
        # GPU worker never needs network access -- the same offline guarantee
        # the local path has.
        .add_local_dir("naigos", remote_path="/root/naigos")
        .add_local_dir("components", remote_path="/root/components")
        .add_local_dir("data_cache", remote_path="/root/data_cache")
    )
    app = modal.App(APP_NAME, image=image)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(gpu="A10G", timeout=60 * 60 * 6, volumes={"/runs": volume})
    def train_remote(
        iterations: int = 2000,
        n_envs: int = 256,
        n_steps: int = 128,
        n_threat: int = 16,
        seed: int = 0,
        synthetic: bool = False,
        run_name: str = "modal",
    ):
        import json

        os.chdir("/root")
        from naigos.rl.ppo import PPOConfig
        from naigos.rl.train import TrainConfig, run

        out = f"/runs/{run_name}"
        if synthetic:
            from naigos.env.config import EnvConfig

            cfg, hmap = EnvConfig(n_threat=n_threat, n_threat_active=n_threat), None
        else:
            from naigos.env.theatre_bridge import describe, env_from_theatre

            cfg, hmap, notes = env_from_theatre(n_threat=n_threat)
            print(describe(notes))
            os.makedirs(out, exist_ok=True)
            with open(f"{out}/theatre.json", "w") as f:
                json.dump(notes, f, indent=2, default=str)

        _, history = run(
            cfg,
            PPOConfig(n_envs=n_envs, n_steps=n_steps),
            TrainConfig(iterations=iterations, out_dir=out, seed=seed, eval_every=25, checkpoint_every=200),
            hmap=hmap,
        )
        volume.commit()
        return history[-1] if history else {}

    @app.local_entrypoint()
    def main(iterations: int = 2000, n_envs: int = 256, seed: int = 0, synthetic: bool = False):
        print(train_remote.remote(iterations=iterations, n_envs=n_envs, seed=seed, synthetic=synthetic))
