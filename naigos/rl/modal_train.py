"""Modal GPU wrapper around `naigos.rl.train.run`.

Deliberately thin: the loop that produces the headline learning curve is the
same code path the tests exercise locally. This file owns the image, the Volume,
the run identity and the entrypoint -- nothing about the algorithm.

    modal run naigos/rl/modal_train.py --profile smoke        # do this first
    modal run naigos/rl/modal_train.py --profile short
    modal run naigos/rl/modal_train.py --profile full
    uv run python scripts/modal_runs.py fetch <run-name>      # retrieve + verify

NO GPU RUN HAS HAPPENED YET, so this repo contains no GPU throughput number and
no speedup claim. Every measured number in it comes from CPU. What changed is
that the instrumentation which would produce such a number now exists and is
written to `perf.json` on the Volume: device, first-iteration compile time,
steady-state seconds per iteration, env-steps/s, recompile cost and peak device
memory. `scripts/modal_runs.py verify` refuses to call a run a GPU run if
`perf.json` says the backend was CPU. See next-steps.md C-1.

Credentials: none are stored in this repository and none are needed by it.
Modal authenticates from `~/.modal.toml` (written by `modal token new`) or from
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the environment. The image copies
only `naigos/`, `components/` and `data_cache/`, so no dotfile, no `.env` and no
shell profile is ever uploaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import runmeta

try:
    import modal
except ImportError:  # pragma: no cover - modal is not a hard dependency
    modal = None

APP_NAME = "naigos"
VOLUME_NAME = "naigos-runs"
RUNS_ROOT = "/runs"

# Static at decoration time, so they are environment variables rather than
# flags. A10G is the default because the model is small and the bottleneck is
# the batched environment rollout, not matmul width -- which is a hypothesis
# the first run's `perf.json` will confirm or refute.
GPU_KIND = os.environ.get("NAIGOS_MODAL_GPU", "A10G")
TIMEOUT_S = int(os.environ.get("NAIGOS_MODAL_TIMEOUT_S", 6 * 60 * 60))

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "jax[cuda12]>=0.4.34",
            "flax>=0.10",
            "optax>=0.2.3",
            "distrax>=0.1.5",
            "numpy>=1.26",
            extra_index_url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        )
        # the cited cache and the component specs travel with the image so the
        # GPU worker never needs network access -- the same offline guarantee
        # the local path has. `data_cache/` is ~170 MB; it is an image layer, so
        # it is uploaded once and reused until it changes.
        .add_local_dir("naigos", remote_path="/root/naigos")
        .add_local_dir("components", remote_path="/root/components")
        .add_local_dir("data_cache", remote_path="/root/data_cache")
    )
    app = modal.App(APP_NAME, image=image)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(gpu=GPU_KIND, timeout=TIMEOUT_S, volumes={RUNS_ROOT: volume})
    def train_remote(spec: dict, meta: dict) -> dict:
        """Run one training job into `/runs/<run_name>` on the Volume.

        `spec` is a `RunProfile` as a dict plus the run's seed/theatre choices;
        `meta` is the immutable `run.json` payload built on the launching
        machine, where the git commit is actually knowable.
        """
        os.chdir("/root")
        from naigos.rl import runmeta as rm
        from naigos.rl.ppo import PPOConfig
        from naigos.rl.train import TrainConfig, run

        profile = rm.RunProfile(**spec["profile"])
        out = rm.run_dir(RUNS_ROOT, spec["run_name"])
        out.mkdir(parents=True, exist_ok=True)

        if spec["synthetic"]:
            from naigos.env.config import EnvConfig

            cfg = EnvConfig(
                n_blue=profile.n_blue,
                n_threat=profile.n_threat,
                n_threat_active=profile.n_threat,
            )
            hmap = None
        else:
            from naigos.env.theatre_bridge import describe, env_from_theatre

            cfg, hmap, notes = env_from_theatre(aoi=spec["aoi"], **profile.theatre_kwargs())
            print(describe(notes))
            rm.write_json(out / rm.THEATRE_FILENAME, notes)

        # Commit after every history/checkpoint write. Without this a run that
        # hits TIMEOUT_S leaves an empty Volume: the container's filesystem is
        # discarded and only committed bytes survive.
        volume.commit()
        _, history = run(
            cfg,
            PPOConfig(**profile.ppo_kwargs()),
            TrainConfig(
                out_dir=str(out),
                seed=spec["seed"],
                use_cbf=spec["use_cbf"],
                **profile.train_kwargs(),
            ),
            hmap=hmap,
            meta=meta,
            on_persist=volume.commit,
        )

        summary = rm.summarize_run(out)
        problems = rm.verify_run_dir(out)
        summary["problems"] = problems
        if profile.name == "smoke" and not problems:
            # The gate that lets an expensive profile launch. Written only when
            # the cheap run actually verified, and stamped with the commit it
            # verified, so a code change re-arms it.
            rm.write_json(
                Path(RUNS_ROOT) / rm.SMOKE_MARKER,
                rm.smoke_marker_payload(spec["run_name"], meta.get("code") or {},
                                        (rm.read_metadata(out) or {}).get("runtime")),
            )
        volume.commit()
        return summary

    def _read_smoke_marker() -> dict | None:
        """Read `/runs/.smoke_ok` off the Volume from the launching machine."""
        try:
            blob = b"".join(volume.read_file(runmeta.SMOKE_MARKER))
        except Exception:  # pragma: no cover - absent marker, or no volume yet
            return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None

    @app.local_entrypoint()
    def main(
        profile: str = "smoke",
        run_name: str = "",
        seed: int = 0,
        synthetic: bool = False,
        aoi: str = "",
        cbf: bool = False,
        iterations: int = 0,
        n_envs: int = 0,
        n_steps: int = 0,
        n_threat: int = 0,
        cell_m: float = 0.0,
        skip_smoke_gate: bool = False,
    ):
        prof = runmeta.resolve_profile(
            profile,
            iterations=iterations or None,
            n_envs=n_envs or None,
            n_steps=n_steps or None,
            n_threat=n_threat or None,
            cell_m=cell_m or None,
        )
        name = runmeta.validate_run_name(run_name) if run_name else runmeta.default_run_name(
            prof.name, seed
        )
        code = runmeta.git_info(Path(__file__).resolve().parents[2])

        marker = _read_smoke_marker()
        refusal = runmeta.smoke_gate(prof.name, marker, code.get("commit"))
        if refusal and not skip_smoke_gate:
            raise SystemExit(f"refusing to launch --profile {prof.name}: {refusal}")
        if refusal:
            print(f"[gate] OVERRIDDEN: {refusal}")
        if code.get("dirty"):
            print("[gate] working tree is dirty; this run will not be reproducible from its commit")

        meta = runmeta.build_metadata(
            prof,
            run_name=name,
            seed=seed,
            synthetic=synthetic,
            aoi=aoi or None,
            use_cbf=cbf,
            code=code,
            launcher="modal",
        )
        print(
            f"[launch] {name} profile={prof.name} gpu={GPU_KIND} "
            f"{prof.iterations} iterations x {prof.env_steps_per_iteration} env-steps "
            f"= {prof.total_env_steps:,} env-steps"
        )
        spec = {
            "profile": prof.as_dict(),
            "run_name": name,
            "seed": seed,
            "synthetic": synthetic,
            "aoi": aoi or None,
            "use_cbf": cbf,
        }
        summary = train_remote.remote(spec, meta)
        print(json.dumps(summary, indent=2, default=str))
        if summary.get("problems"):
            print("[verify] problems reported by the worker; see above")
        print(
            f"\nretrieve it with:\n"
            f"  uv run python scripts/modal_runs.py fetch {name}"
        )
