"""Modal GPU wrapper around `naigos.rl.train.run`.

Deliberately thin: the loop that produces the headline learning curve is the
same code path the tests exercise locally. This file owns the image, the Volume,
the run identity, the status manifest and the entrypoints -- nothing about the
algorithm.

The operational flow, in the order it has to happen:

    modal token new                                          # 1. authenticate
    modal deploy naigos/rl/modal_train.py                     # 2. publish the app
    uv run python scripts/modal_runs.py submit --profile smoke # 3. detached smoke
    uv run python scripts/modal_runs.py status <run-name>      # 4. verify it
    uv run python scripts/modal_runs.py submit --profile short # 5. detached run
    uv run python scripts/modal_runs.py fetch <run-name>       # 6. retrieve + verify
    uv run python scripts/modal_runs.py resume <run-name>      # 7. only if needed

`modal run naigos/rl/modal_train.py --profile smoke` still works and still
blocks; it is kept because it is the shortest way to see a traceback while
debugging the image. It is NOT the detached path: closing the laptop kills it.
Detachment comes from a *deployed* function plus `.spawn()`, which puts the call
in Modal's queue and returns a call id that outlives the client entirely.

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
shell profile is ever uploaded. Everything the worker prints into `train.log` on
the shared Volume passes through `runmeta.redact` first, because a traceback
from a client library can quote the environment it was handed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import runmeta

try:
    import modal
except ImportError:  # pragma: no cover - modal is not a hard dependency
    modal = None

APP_NAME = "naigos"
VOLUME_NAME = "naigos-runs"
RUNS_ROOT = "/runs"
LOG_FILENAME = "train.log"

# --- budget-oriented defaults ------------------------------------------------
# All four are explicit because all four cost money, and because a default that
# is not written down is a default nobody reviews.
#
# GPU: A10G, the cheapest current-generation card that fits the model, chosen
# because the bottleneck is expected to be the batched environment rollout
# rather than matmul width -- a hypothesis the first run's `perf.json` will
# confirm or refute, at which point this is the number to revisit.
GPU_KIND = os.environ.get("NAIGOS_MODAL_GPU", "A10G")
# Timeout: two of them, because one timeout for both sizes is a budget hazard in
# whichever direction it is set. A smoke run that hangs must not be allowed to
# bill six hours of GPU to prove the image works.
TIMEOUT_S = int(os.environ.get("NAIGOS_MODAL_TIMEOUT_S", 6 * 60 * 60))
SMOKE_TIMEOUT_S = int(os.environ.get("NAIGOS_MODAL_SMOKE_TIMEOUT_S", 30 * 60))
# Retries: ZERO, deliberately. Modal's retry restarts the container from
# scratch, so an automatic retry of a long run silently pays for the same
# iterations twice and writes a second run into a directory that already has a
# writer. Resume is an operator decision made against a named checkpoint
# (`modal_runs.py resume`), not something that happens while nobody is looking.
MAX_RETRIES = int(os.environ.get("NAIGOS_MODAL_RETRIES", 0))
# Checkpoint cadence is per profile (`runmeta.PROFILES`). It is the ceiling on
# how much GPU time a crash can destroy, so it is a budget number too: `full`
# checkpoints every 200 iterations of 3000.

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

    class _Tee(io.TextIOBase):
        """Duplicate the worker's output into `<run>/train.log` on the Volume.

        Modal keeps container logs, but they are addressed by app and not by
        run, they expire, and they are gone the moment someone rotates a
        workspace. A per-run log committed next to the checkpoints is the copy
        that can still be fetched a week later, alongside the artifacts it
        describes. Redacted on the way in: the Volume is shared storage.
        """

        def __init__(self, stream, path: Path):
            self._stream = stream
            self._fh = open(path, "a", buffering=1, errors="replace")

        def write(self, text: str) -> int:  # noqa: D102
            self._stream.write(text)
            try:
                self._fh.write(runmeta.redact(text))
            except Exception:  # pragma: no cover - never lose a run to logging
                pass
            return len(text)

        def flush(self) -> None:  # noqa: D102
            self._stream.flush()
            with contextlib.suppress(Exception):
                self._fh.flush()

        def close(self) -> None:  # noqa: D102
            with contextlib.suppress(Exception):
                self._fh.close()

    def _utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _train_impl(spec: dict, meta: dict) -> dict:
        """Run one training job into `/runs/<run_name>` on the Volume.

        `spec` is a `RunProfile` as a dict plus the run's seed/theatre/resume
        choices; `meta` is the immutable `run.json` payload built on the
        launching machine, where the git commit is actually knowable.

        Everything that can be persisted is persisted before anything that can
        fail: the manifest is written and committed first, so a container that
        dies during image warm-up still leaves a record saying which job it was.
        """
        os.chdir("/root")
        from naigos.rl import runmeta as rm

        # The launching machine writes a `queued` manifest before the container
        # exists; without a reload this container would not see it and would
        # start the run's status history over.
        with contextlib.suppress(Exception):
            volume.reload()

        out = rm.run_dir(RUNS_ROOT, spec["run_name"])
        out.mkdir(parents=True, exist_ok=True)

        # One writer per run directory. `history.json` is rewritten whole at
        # every eval boundary, so a second writer does not merge with the first,
        # it erases it.
        rm.acquire_writer_lock(
            out, owner=f"modal:{os.environ.get('MODAL_TASK_ID', 'unknown')}",
            force=bool(spec.get("force_unlock")),
        )
        rm.update_manifest(
            out,
            status=rm.RUNNING,
            started_utc=_utc(),
            termination_reason=None,
            job_id=spec.get("job_id") or os.environ.get("MODAL_TASK_ID"),
            attempt={
                "kind": "resume" if spec.get("resume_from") else "start",
                "started_utc": _utc(),
                "task_id": os.environ.get("MODAL_TASK_ID"),
                "resume_from": spec.get("resume_from"),
                "resume_overrides": spec.get("resume_overrides") or [],
            },
        )
        volume.commit()

        tee = _Tee(sys.stdout, out / LOG_FILENAME)
        reason, status, summary = rm.REASON_COMPLETED, rm.COMPLETED, {}
        try:
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                summary = _run_training(spec, meta, out, rm)
        except BaseException as e:  # noqa: BLE001 - every exit path must be recorded
            status = rm.FAILED
            reason = rm.REASON_EXCEPTION
            # Redacted, and the type plus message only -- a full traceback goes
            # to `train.log`, which is also redacted, rather than into the
            # manifest that every `status` call prints to a terminal.
            summary = {"error": rm.redact(f"{type(e).__name__}: {e}")}
            tee.write(rm.redact(traceback.format_exc()))
            raise
        finally:
            tee.flush()
            ckpts = [rm.checkpoint_iteration(p) for p in rm.checkpoint_paths(out)]
            rm.update_manifest(
                out,
                status=status,
                termination_reason=reason,
                finished_utc=_utc(),
                actual_backend=(summary.get("platform") if isinstance(summary, dict) else None),
                actual_devices=(summary.get("device") if isinstance(summary, dict) else None),
                last_iteration=(summary.get("iterations_done") if isinstance(summary, dict) else None),
                checkpoints=[c for c in ckpts if c is not None],
                error=(summary.get("error") if isinstance(summary, dict) else None),
            )
            rm.release_writer_lock(out)
            tee.close()
            volume.commit()
        return summary

    def _preflight(out: Path, rm) -> dict:
        """Answer, by name, each thing a smoke run exists to prove.

        Run before training on every profile, and *reported* on the smoke
        profile, where a named failure is the entire deliverable. Each check is
        the cheapest thing that actually distinguishes working from broken --
        importing the stack rather than trusting the image built, reading a byte
        back off the Volume rather than trusting the mount, asking JAX what
        backend it resolved rather than assuming the `gpu=` argument took.
        """
        report: dict = {}

        def check(name: str, fn):
            try:
                detail = fn()
                report[name] = {"ok": True, "detail": rm.redact(str(detail))}
            except BaseException as e:  # noqa: BLE001 - a failed check is data
                report[name] = {"ok": False, "detail": rm.redact(f"{type(e).__name__}: {e}")}

        def _image():
            import flax, jax, optax  # noqa: F401

            from naigos.rl.ppo import PPOConfig  # noqa: F401
            from naigos.rl.train import device_report  # noqa: F401

            return f"jax {jax.__version__} flax {flax.__version__} optax {optax.__version__}"

        def _gpu():
            import jax

            backend = jax.default_backend()
            kinds = [d.device_kind for d in jax.devices()]
            if backend != "gpu":
                # The dangerous failure: JAX falls back silently, the loop
                # completes, and a CPU number is later published as a GPU number.
                raise RuntimeError(
                    f"requested gpu={GPU_KIND} but JAX resolved backend {backend!r} "
                    f"on {kinds}"
                )
            return f"{backend} x{len(kinds)} {', '.join(kinds)}"

        def _cache():
            cache = Path("/root/data_cache")
            if not cache.is_dir():
                raise FileNotFoundError("/root/data_cache is not in the image")
            files = sum(1 for _ in cache.rglob("*") if _.is_file())
            if files == 0:
                raise FileNotFoundError("/root/data_cache is present but empty")
            specs = Path("/root/components")
            if not specs.is_dir():
                raise FileNotFoundError("/root/components is not in the image")
            return f"{files} cached files, {sum(1 for _ in specs.glob('*.json'))} component specs"

        def _volume():
            probe = out / ".preflight"
            probe.write_text(_utc())
            volume.commit()
            if not probe.read_text():
                raise OSError("wrote a probe file to the Volume and read back nothing")
            probe.unlink()
            volume.commit()
            return f"{out} is writable and commits"

        check("image", _image)
        check("gpu_backend", _gpu)
        check("data_cache", _cache)
        check("volume_write", _volume)
        for name, entry in report.items():
            print(f"[preflight] {'ok  ' if entry['ok'] else 'FAIL'} {name}: {entry['detail']}")
        return report

    def _run_training(spec: dict, meta: dict, out: Path, rm) -> dict:
        """The part inside the log tee and the status bookkeeping."""
        from naigos.rl import checkpoint as ckpt
        from naigos.rl.ppo import PPOConfig
        from naigos.rl.train import TrainConfig, run

        profile = rm.RunProfile(**spec["profile"])
        preflight = _preflight(out, rm)
        rm.update_manifest(out, preflight=preflight)
        volume.commit()
        failed = rm.preflight_problems({**preflight, "checkpoint": {"ok": True},
                                        "artifact_fetch": {"ok": True}})
        if failed:
            # Fail here rather than after paying for a compile: every one of
            # these means the run cannot produce a usable result.
            raise RuntimeError("preflight failed: " + "; ".join(failed))

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

        resume_path = None
        if spec.get("resume_from"):
            # The compatibility decision was made on the launching machine,
            # where the git commit is knowable; this side re-derives the actual
            # file, because "most recent valid checkpoint" is a question only the
            # Volume can answer and the answer may have moved since submission.
            resume_path = ckpt.latest_resumable(out)
            if resume_path is None:
                raise RuntimeError(
                    f"resume was requested for {spec['run_name']} but no checkpoint on the "
                    "Volume carries a recovery block. Older checkpoints hold the policy only; "
                    "this run has to be restarted, not resumed."
                )
            print(f"[resume] selected {resume_path.name} from {out}")

        def progress(info: dict) -> None:
            rm.update_manifest(
                out,
                last_iteration=info.get("last_iteration"),
                iterations_declared=info.get("iterations_declared"),
                resumed=info.get("resumed"),
                resumed_from_iteration=info.get("resumed_from_iteration"),
            )

        # Commit after every history/checkpoint write. Without this a run that
        # hits its timeout leaves an empty Volume: the container's filesystem is
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
            on_progress=progress,
            resume_from=resume_path,
        )

        # The last two preflight checks can only be answered after training:
        # a checkpoint has to exist and read back, and the directory has to be
        # complete enough for `fetch` to retrieve something verifiable.
        from naigos.rl import checkpoint as _ck

        latest = _ck.latest_resumable(out)
        preflight["checkpoint"] = (
            {"ok": True, "detail": f"{latest.name} reads back with a recovery block"}
            if latest is not None
            else {"ok": False, "detail": "no checkpoint on the Volume carries recovery state"}
        )
        summary = rm.summarize_run(out)
        problems = rm.verify_run_dir(out)
        preflight["artifact_fetch"] = (
            {"ok": True, "detail": f"{len(list(out.iterdir()))} files, verify_run_dir clean"}
            if not [p for p in problems if not p.startswith("preflight check")]
            else {"ok": False, "detail": "; ".join(problems)}
        )
        rm.update_manifest(out, preflight=preflight)
        problems = rm.verify_run_dir(out)
        summary["problems"] = problems
        summary["preflight"] = preflight
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

    @app.function(gpu=GPU_KIND, timeout=TIMEOUT_S, retries=MAX_RETRIES,
                  volumes={RUNS_ROOT: volume})
    def train_remote(spec: dict, meta: dict) -> dict:
        """The long-run entry point. Six-hour default ceiling."""
        return _train_impl(spec, meta)

    @app.function(gpu=GPU_KIND, timeout=SMOKE_TIMEOUT_S, retries=MAX_RETRIES,
                  volumes={RUNS_ROOT: volume})
    def smoke_remote(spec: dict, meta: dict) -> dict:
        """Same code, same GPU, a thirty-minute ceiling.

        A separate function only because Modal fixes the timeout at decoration
        time. A smoke run exists to fail fast; letting it inherit the long
        timeout would let a hung plumbing check bill six hours.
        """
        return _train_impl(spec, meta)

    def read_volume_json(remote_path: str) -> dict | None:
        """Read one JSON file off the Volume from the launching machine."""
        try:
            blob = b"".join(volume.read_file(remote_path))
        except Exception:  # pragma: no cover - absent file, or no volume yet
            return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None

    def read_volume_text(remote_path: str, *, tail_bytes: int | None = None) -> str | None:
        try:
            blob = b"".join(volume.read_file(remote_path))
        except Exception:  # pragma: no cover - absent file, or no volume yet
            return None
        if tail_bytes is not None and len(blob) > tail_bytes:
            blob = blob[-tail_bytes:]
        return runmeta.redact(blob.decode("utf-8", errors="replace"))

    def _read_smoke_marker() -> dict | None:
        """Read `/runs/.smoke_ok` off the Volume from the launching machine."""
        return read_volume_json(runmeta.SMOKE_MARKER)

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
        """Attached run. Blocks until the job finishes; closing the laptop kills it.

        Kept for debugging the image, where watching a traceback arrive is the
        point. For anything longer than a smoke run use
        `scripts/modal_runs.py submit`, which spawns a deployed function and
        returns immediately.
        """
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
            "resume_from": None,
            "resume_overrides": [],
            "force_unlock": False,
            "job_id": None,
        }
        fn = smoke_remote if prof.name == "smoke" else train_remote
        summary = fn.remote(spec, meta)
        print(json.dumps(summary, indent=2, default=str))
        if summary.get("problems"):
            print("[verify] problems reported by the worker; see above")
        print(
            f"\nretrieve it with:\n"
            f"  uv run python scripts/modal_runs.py fetch {name}"
        )
