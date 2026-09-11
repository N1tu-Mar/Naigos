"""Modal deployment of the cloud learning pipeline: crons, workers, admin actions.

    modal token new                                   # once, on any machine
    modal deploy naigos/rl/modal_pipeline.py          # the only step that needs the laptop
    uv run python scripts/pipeline.py status          # from anywhere, any time after

After ``modal deploy`` the schedules belong to Modal. The three ``modal.Cron``
functions below fire on Modal's infrastructure whether or not any client is
running; closing the laptop changes nothing. (``modal run`` would not do this:
schedules are only active for a *deployed* app.)

What runs where:

  * ``snapshot_tick`` / ``nightly_tick`` / ``weekly_tick`` -- small CPU
    containers, one at a time (``max_containers=1``), that call
    ``naigos.pipeline.coordinator.tick``. They do no costly work themselves.
  * ``snapshot_worker`` -- the research image (requests, py3dep, rasterio ...),
    network allowed, but the research build runs in a child process whose
    DNS is restricted to the fixed allowlist (``naigos.pipeline.egress``).
  * ``train_candidate`` -- the GPU image, same stack as ``modal_train.py`` but
    with no data baked in: it trains on a verified, container-local copy of one
    snapshot. ``block_network=True``: the offline training invariant is enforced
    by the platform, not only by the code path.
  * ``evaluate_candidate`` / ``promote_candidate`` / ``rollback_champion`` -- the
    CPU JAX image, also ``block_network=True``. Evaluation runs on CPU so a
    promotion-time re-run reproduces the recorded numbers exactly.
  * ``admin`` -- status, pause/resume, retry, unlock, prune, config: every
    write the CLI asks for happens here, in the cloud, with the same code.

Budget and safety defaults (environment variables are read at deploy time):

  ``NAIGOS_MODAL_GPU``                 A10G      same variable as modal_train.py
  ``NAIGOS_PIPELINE_TRAIN_TIMEOUT_S``  4 h       hard ceiling per candidate
  ``NAIGOS_PIPELINE_EVAL_TIMEOUT_S``   1 h
  ``NAIGOS_PIPELINE_BLOCK_NETWORK``    1         set 0 only if Modal Dict access
                                                 fails under block_network
  retries                              0         always; see modal_train.py

No credential is stored in this repository or passed as an argument. Modal
authenticates from ``~/.modal.toml`` or ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``.
The pipeline's sources need no key; if one ever does, it belongs in a
``modal.Secret`` attached to ``snapshot_worker`` only.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from . import runmeta

try:
    import modal
except ImportError:  # pragma: no cover - modal is not a hard dependency
    modal = None

REPO = Path(__file__).resolve().parents[2]
APP_NAME = "naigos-pipeline"
VOLUME_NAME = "naigos-runs"          # the same Volume as modal_train.py, under /pipeline
LEASE_DICT_NAME = "naigos-pipeline-leases"
VOL_MOUNT = "/vol"

GPU_KIND = os.environ.get("NAIGOS_MODAL_GPU", "A10G")
TRAIN_TIMEOUT_S = int(os.environ.get("NAIGOS_PIPELINE_TRAIN_TIMEOUT_S", 4 * 3600))
EVAL_TIMEOUT_S = int(os.environ.get("NAIGOS_PIPELINE_EVAL_TIMEOUT_S", 3600))
BLOCK_NETWORK = os.environ.get("NAIGOS_PIPELINE_BLOCK_NETWORK", "1") != "0"
TICK_TIMEOUT_S = 600
MAX_RETRIES = 0


def deployed_code() -> dict:
    """The commit this deployment runs.

    Computed from git on the deploying machine and baked into every image as
    environment variables, because a container has no ``.git`` to ask. The same
    function reads it back inside the container, so both sides agree.
    """
    commit = os.environ.get("NAIGOS_CODE_COMMIT")
    if commit:
        dirty = os.environ.get("NAIGOS_CODE_DIRTY")
        return {"commit": commit, "dirty": {"0": False, "1": True}.get(dirty)}
    return runmeta.git_info(REPO)


CODE = deployed_code()


def _code_env() -> dict:
    return {"NAIGOS_CODE_COMMIT": CODE.get("commit") or "",
            "NAIGOS_CODE_DIRTY": {True: "1", False: "0"}.get(CODE.get("dirty"), "")}


def _schedule() -> dict:
    from naigos.pipeline import config as pcfg

    return pcfg.deployed_schedule()


if modal is not None:
    _ignore = ["**/__pycache__", "**/*.pyc"]
    _base = modal.Image.debian_slim(python_version="3.12")

    def _finish(image):
        # The commit is baked in last, after the dependency layers, so a new
        # commit re-uses the cached pip layers instead of rebuilding CUDA JAX.
        # The package is mounted last because Modal requires add_local_* to be
        # the final step.
        return image.env(_code_env()).add_local_dir(
            str(REPO / "naigos"), remote_path="/root/naigos", ignore=_ignore)

    coordinator_image = _finish(_base.pip_install("numpy>=1.26"))
    # Research dependencies only -- no JAX, no training stack.
    snapshot_image = _finish(_base.pip_install(
        "numpy>=1.26", "requests>=2.31", "py3dep>=0.16", "rasterio>=1.3", "rioxarray>=0.15",
        "xarray>=2023.1", "pyproj>=3.6"))
    # The training stack of modal_train.py, minus the baked data_cache/components:
    # a candidate trains on its snapshot, never on whatever the image carried.
    train_image = _finish(_base.pip_install(
        "jax[cuda12]>=0.4.34", "flax>=0.10", "optax>=0.2.3", "distrax>=0.1.5", "numpy>=1.26",
        extra_index_url="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"))
    eval_image = _finish(_base.pip_install(
        "jax>=0.4.34", "flax>=0.10", "optax>=0.2.3", "distrax>=0.1.5", "numpy>=1.26",
    ).env({"JAX_PLATFORMS": "cpu"}))

    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    lease_dict = modal.Dict.from_name(LEASE_DICT_NAME, create_if_missing=True)
    VOLUMES = {VOL_MOUNT: volume}
    SCHEDULE = _schedule()

    # --- services bound to Modal ----------------------------------------------------

    def _commit() -> None:
        with contextlib.suppress(Exception):
            volume.commit()

    def _reload() -> None:
        with contextlib.suppress(Exception):
            volume.reload()

    def _remote_state(call_id: str | None):
        """Modal's account of a call; the same probe as scripts/modal_runs.py."""
        if not call_id:
            return None
        try:
            fc = modal.FunctionCall.from_id(call_id)
        except Exception:  # noqa: BLE001
            return None
        try:
            fc.get(timeout=0)
            return runmeta.COMPLETED
        except TimeoutError:
            return runmeta.RUNNING
        except BaseException as e:  # noqa: BLE001 - the exception type is the state
            return runmeta.RUNNING if type(e).__name__ == "TimeoutError" else e

    def _call_id() -> str | None:
        with contextlib.suppress(Exception):
            return modal.current_function_call_id()
        return None

    def _smoke_marker() -> dict | None:
        path = Path(VOL_MOUNT) / runmeta.SMOKE_MARKER
        with contextlib.suppress(Exception):
            return json.loads(path.read_text())
        return None

    def _spawn(fn_name: str, payload: dict) -> str:
        fn = {"snapshot_worker": snapshot_worker, "train_candidate": train_candidate,
              "evaluate_candidate": evaluate_candidate}[fn_name]
        return fn.spawn(payload).object_id

    def _services():
        from naigos.pipeline import layout, leases, worker

        lay = layout.Layout(Path(VOL_MOUNT))
        store = leases.DictLeaseStore(lease_dict, mirror_dir=lay.root / "locks")
        mgr = leases.LeaseManager(store, remote_state=_remote_state)
        return worker.Services(lay=lay, leases=mgr, code=CODE, spawn=_spawn,
                               remote_state=_remote_state, smoke_marker=_smoke_marker,
                               commit=_commit, reload=_reload, call_id=_call_id)

    class _Tee(io.TextIOBase):
        """Copy output into a redacted log file on the Volume, as modal_train.py does."""

        def __init__(self, stream, path: Path):
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream, self._fh = stream, open(path, "a", buffering=1, errors="replace")

        def write(self, text: str) -> int:  # noqa: D102
            self._stream.write(text)
            with contextlib.suppress(Exception):
                self._fh.write(runmeta.redact(text))
            return len(text)

        def flush(self) -> None:  # noqa: D102
            self._stream.flush()
            with contextlib.suppress(Exception):
                self._fh.flush()

    @contextlib.contextmanager
    def _logged(path: Path):
        tee = _Tee(sys.stdout, path)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                yield
            except BaseException:
                tee.write(runmeta.redact(traceback.format_exc()))
                raise
            finally:
                tee.flush()

    # --- the crons ------------------------------------------------------------------

    def _tick(kind: str, manual: bool = False) -> dict:
        os.chdir("/root")
        from naigos.pipeline import coordinator

        summary = coordinator.tick(_services(), kind, manual=manual)
        print(json.dumps({k: summary.get(k) for k in ("kind", "status", "reason")}, default=str))
        return summary

    _tick_kw = dict(image=coordinator_image, volumes=VOLUMES, timeout=TICK_TIMEOUT_S,
                    retries=MAX_RETRIES, max_containers=1)

    @app.function(schedule=modal.Cron(SCHEDULE["snapshot_cron"], timezone="UTC"), **_tick_kw)
    def snapshot_tick() -> dict:
        """Daily: refresh the allowlisted inputs into a new immutable snapshot."""
        return _tick("snapshot")

    @app.function(schedule=modal.Cron(SCHEDULE["nightly_cron"], timezone="UTC"), **_tick_kw)
    def nightly_tick() -> dict:
        """Nightly: train and evaluate a candidate, only if a new snapshot exists."""
        return _tick("nightly")

    @app.function(schedule=modal.Cron(SCHEDULE["weekly_cron"], timezone="UTC"), **_tick_kw)
    def weekly_tick() -> dict:
        """Weekly: a longer candidate, only if the latest nightly passed its gates."""
        return _tick("weekly")

    # --- the workers ----------------------------------------------------------------

    @app.function(image=snapshot_image, volumes=VOLUMES, timeout=2 * 3600, retries=MAX_RETRIES)
    def snapshot_worker(payload: dict) -> dict:
        os.chdir("/root")
        from naigos.pipeline import worker

        rec = worker.run_snapshot(_services(), payload)
        return {"snapshot_id": rec["snapshot_id"], "content_sha256": rec["content"]["sha256"]}

    def _gpu_trainer(*, out_dir: Path, meta: dict, profile: dict, seed: int, aoi: str,
                     use_cbf: bool, resume: bool) -> dict:
        """`naigos.rl.train.run` with the manifest bookkeeping of modal_train.py."""
        import jax

        from naigos.env.theatre_bridge import describe, env_from_theatre
        from naigos.pipeline import layout
        from naigos.rl import checkpoint as ckpt
        from naigos.rl.ppo import PPOConfig
        from naigos.rl.train import TrainConfig, run

        rm = runmeta
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prof = rm.RunProfile(**{**profile, "overrides": tuple(profile.get("overrides") or ())})
        if rm.read_manifest(out_dir) is None:
            rm.write_json(out_dir / rm.MANIFEST_FILENAME, rm.build_manifest(
                run_name=out_dir.name, job_id=_call_id(), profile=prof.name, gpu=GPU_KIND,
                timeout_s=TRAIN_TIMEOUT_S, max_retries=MAX_RETRIES,
                checkpoint_every=prof.checkpoint_every, code=meta.get("code"),
                config=meta.get("config"), status=rm.RUNNING))
        # The candidate lease is the real mutex (atomic, Modal-Dict backed); the
        # run directory's own lock file is taken too so the directory reads the
        # same as any other run, and forced because a lock left by a killed
        # attempt of this same candidate is exactly what the lease already ruled on.
        rm.acquire_writer_lock(out_dir, owner=f"modal:{_call_id()}", force=True)
        rm.update_manifest(out_dir, status=rm.RUNNING, termination_reason=None,
                           started_utc=layout.utc_stamp(), job_id=_call_id(),
                           attempt={"kind": "resume" if resume else "start",
                                    "started_utc": layout.utc_stamp(), "task_id": _call_id()})
        _commit()
        status, reason, summary = rm.COMPLETED, rm.REASON_COMPLETED, {}
        try:
            if jax.default_backend() != "gpu":
                raise RuntimeError(f"requested gpu={GPU_KIND} but JAX resolved "
                                   f"{jax.default_backend()!r}; refusing a silent CPU run")
            cfg, hmap, notes = env_from_theatre(aoi=aoi, **prof.theatre_kwargs())
            print(describe(notes))
            rm.write_json(out_dir / rm.THEATRE_FILENAME,
                          {**notes, "snapshot_id": (meta.get("pipeline") or {}).get("snapshot_id")})
            resume_path = ckpt.latest_resumable(out_dir) if resume else None
            if resume and resume_path is None:
                raise RuntimeError("resume requested but no checkpoint carries recovery state")

            def progress(info: dict) -> None:
                rm.update_manifest(out_dir, last_iteration=info.get("last_iteration"),
                                   iterations_declared=info.get("iterations_declared"))

            run(cfg, PPOConfig(**prof.ppo_kwargs()),
                TrainConfig(out_dir=str(out_dir), seed=seed, use_cbf=use_cbf, **prof.train_kwargs()),
                hmap=hmap, meta=meta, on_persist=_commit, on_progress=progress,
                resume_from=resume_path)
            summary = rm.summarize_run(out_dir)
        except BaseException as e:  # noqa: BLE001 - every exit path is recorded
            status, reason = rm.FAILED, rm.REASON_EXCEPTION
            summary = {"error": rm.redact(f"{type(e).__name__}: {e}")}
            raise
        finally:
            rm.update_manifest(out_dir, status=status, termination_reason=reason,
                               finished_utc=layout.utc_stamp(),
                               actual_backend=(summary or {}).get("platform"),
                               last_iteration=(summary or {}).get("iterations_done"),
                               error=(summary or {}).get("error"))
            rm.release_writer_lock(out_dir)
            _commit()
        return summary

    @app.function(image=train_image, gpu=GPU_KIND, volumes=VOLUMES, timeout=TRAIN_TIMEOUT_S,
                  retries=MAX_RETRIES, block_network=BLOCK_NETWORK)
    def train_candidate(payload: dict) -> dict:
        os.chdir("/root")
        from naigos.pipeline import coordinator, layout, worker

        svc = _services()
        cid = layout.validate_candidate_id(payload["candidate_id"])

        def dispatch_eval(c: str):
            # Best effort: if this fails, the next coordinator tick reconciles
            # a completed training with no evaluation and dispatches it.
            with contextlib.suppress(Exception):
                return coordinator.dispatch_evaluation(svc, c)

        with _logged(svc.lay.candidate_dir(cid) / "train.log"):
            return worker.run_training(svc, payload, trainer=_gpu_trainer, dispatch_eval=dispatch_eval)

    def _cpu_measure(*, snapshot_root, plan, env_shape, candidate_ckpt, champion_ckpt) -> dict:
        from naigos.pipeline import evaluation as ev

        env = ev.build_env(aoi=env_shape["aoi"], n_blue=env_shape["n_blue"],
                           n_threat=env_shape["n_threat"], cell_m=env_shape["cell_m"],
                           red_level=env_shape["red_level"])
        champ = None
        if champion_ckpt is not None:
            try:
                champ = ev.load_actor_params(champion_ckpt)
            except Exception as e:  # noqa: BLE001 - recorded as a champion error
                return {**ev.run_evaluation(env, plan, ev.load_actor_params(candidate_ckpt)),
                        "champion_error": f"{type(e).__name__}: {e}"}
        return ev.run_evaluation(env, plan, ev.load_actor_params(candidate_ckpt), champ)

    _eval_kw = dict(image=eval_image, cpu=4.0, memory=8192, volumes=VOLUMES,
                    timeout=EVAL_TIMEOUT_S, retries=MAX_RETRIES, block_network=BLOCK_NETWORK)

    @app.function(**_eval_kw)
    def evaluate_candidate(payload: dict) -> dict:
        """Held-out evaluation, verifier, decision -- and the only automatic promoter."""
        os.chdir("/root")
        from naigos.pipeline import layout, worker

        svc = _services()
        cid = layout.validate_candidate_id(payload["candidate_id"])
        with _logged(svc.lay.root / "status" / "logs" / f"evaluate__{cid}.log"):
            return worker.run_evaluation(svc, payload, measure=_cpu_measure)

    @app.function(**_eval_kw)
    def promote_candidate(candidate_id: str, actor: str, expected_generation: int | None = None,
                          note: str | None = None) -> dict:
        """Operator promotion: re-validates everything and re-runs the evaluation first."""
        os.chdir("/root")
        from naigos.pipeline import promotion, worker

        try:
            return {"ok": True, "pointer": worker.promote(
                _services(), candidate_id, actor=actor, mode="manual", rerun_measure=_cpu_measure,
                expected_generation=expected_generation, note=note)}
        except promotion.PromotionRefused as e:
            return {"ok": False, "error": str(e), "problems": e.problems}
        except Exception as e:  # noqa: BLE001 - reported, pointer untouched
            return {"ok": False, "error": runmeta.redact(f"{type(e).__name__}: {e}"), "problems": []}

    @app.function(**_eval_kw)
    def rollback_champion(generation: int, actor: str, note: str | None = None) -> dict:
        os.chdir("/root")
        from naigos.pipeline import promotion, worker

        try:
            return {"ok": True, "pointer": worker.rollback(_services(), int(generation),
                                                           actor=actor, note=note)}
        except promotion.PromotionRefused as e:
            return {"ok": False, "error": str(e), "problems": e.problems}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": runmeta.redact(f"{type(e).__name__}: {e}"), "problems": []}

    # --- admin ----------------------------------------------------------------------

    @app.function(image=coordinator_image, volumes=VOLUMES, timeout=TICK_TIMEOUT_S,
                  retries=MAX_RETRIES)
    def admin(command: str, args: dict | None = None) -> dict:
        """Everything scripts/pipeline.py asks of the deployment except promotion."""
        os.chdir("/root")
        from naigos.pipeline import admin as padmin

        svc = _services()
        svc.reload()
        if command == "run-now":
            return _tick(str((args or {}).get("kind")), manual=True)
        return padmin.handle(svc, command, args or {}, deployed_schedule=SCHEDULE,
                             remove_tree=shutil.rmtree)
