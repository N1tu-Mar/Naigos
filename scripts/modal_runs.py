#!/usr/bin/env python
"""Submit, inspect, resume, fetch and verify training runs on Modal.

The whole operator flow, in order. Nothing here needs the laptop to stay awake
past step 3.

    modal token new                                     # 1. authenticate
    modal deploy naigos/rl/modal_train.py               # 2. publish the app
    uv run python scripts/modal_runs.py submit --profile smoke   # 3. detached
    uv run python scripts/modal_runs.py status <run-name>        # 4. verify smoke
    uv run python scripts/modal_runs.py submit --profile short   # 5. detached
    uv run python scripts/modal_runs.py status <run-name>        # 6. inspect
    uv run python scripts/modal_runs.py fetch <run-name>         # 7. retrieve + verify
    uv run python scripts/modal_runs.py resume <run-name>        # 8. only if needed

`submit` calls `.spawn()` on a *deployed* Modal function. The call goes into
Modal's queue and is executed by Modal's infrastructure; the returned call id is
the stable job identifier and nothing about the job depends on this process
afterwards. That is the difference from `modal run`, which holds the job open
for as long as the local entrypoint is alive.

Which commands need what:

  * `submit`, `status`, `logs`, `cancel`, `resume`, `list`, `fetch` talk to
    Modal, so they need `modal` installed and authenticated.
  * `verify` and `jobs` do not. `verify` reads a local directory, so a run
    fetched by any means -- including `modal volume get` typed by hand -- can be
    checked, and so the checking logic is unit tested offline.

Verification is not a formality. A run that hit the Modal timeout leaves a
directory shaped exactly like a completed one, and a run whose worker fell back
to CPU leaves timings that look like GPU timings. Both are reported here.

No credential is read, printed or stored by this script. Modal authenticates
itself from `~/.modal.toml` or from `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, and
everything fetched off the shared Volume passes through `runmeta.redact` before
it reaches a terminal.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naigos.rl import runmeta  # noqa: E402
from naigos.rl.modal_train import (  # noqa: E402
    APP_NAME,
    GPU_KIND,
    LOG_FILENAME,
    MAX_RETRIES,
    RUNS_ROOT,
    SMOKE_TIMEOUT_S,
    TIMEOUT_S,
    VOLUME_NAME,
)

REPO = Path(__file__).resolve().parents[1]
#: Local index of submissions: run name -> job id. Purely a convenience, so a
#: run can be named instead of remembered as a call id. The authoritative record
#: is `manifest.json` on the Volume, which is why `status` works from a fresh
#: clone that has never submitted anything.
JOBS_DIR = REPO / "runs" / ".jobs"


# --- modal plumbing ----------------------------------------------------------


def _require_modal():
    try:
        import modal  # noqa: F401
    except ImportError:
        raise SystemExit(
            "the `modal` client is not installed. `uv pip install modal` then "
            "`modal token new`; no credential is stored in this repository."
        ) from None
    import modal

    return modal


def _modal_cli(*args: str) -> int:
    if shutil.which("modal") is None:
        raise SystemExit(
            "the `modal` CLI is not on PATH. `uv pip install modal` then `modal token new`; "
            "no credential is stored in this repository."
        )
    print("$ modal " + " ".join(args))
    return subprocess.run(["modal", *args]).returncode


def _deployed_function(tag: str):
    """Look up a deployed function, tolerating the client's API renames.

    A spawned call on a *deployed* function is what makes a run detached. If the
    app has not been deployed the failure is confusing (`NotFoundError` naming an
    app the user did not know they needed), so it is translated here.
    """
    modal = _require_modal()
    last = None
    for getter in ("from_name", "lookup"):
        fn = getattr(modal.Function, getter, None)
        if fn is None:
            continue
        try:
            return fn(APP_NAME, tag)
        except Exception as e:  # noqa: BLE001 - client API and auth both land here
            last = e
    raise SystemExit(
        f"could not find the deployed function {APP_NAME}/{tag}: {runmeta.redact(str(last))}\n"
        f"Deploy it first -- a detached run must belong to a deployed app, or Modal tears "
        f"it down when the client exits:\n"
        f"  modal deploy naigos/rl/modal_train.py"
    )


def _function_call(job_id: str):
    modal = _require_modal()
    try:
        return modal.FunctionCall.from_id(job_id)
    except Exception as e:  # noqa: BLE001
        print(f"[modal] cannot address job {job_id}: {runmeta.redact(str(e))}")
        return None


def _remote_state(job_id: str | None) -> object:
    """Ask Modal what became of a call, as one of the lifecycle states.

    `get(timeout=0)` is the portable probe: it returns on success and raises
    otherwise. `TimeoutError` from it means "not finished", which is a *live*
    state and must not be confused with the remote job having exceeded its own
    timeout -- that arrives as `FunctionTimeoutError`. Getting those two the
    wrong way round would report every running job as timed out.
    """
    if not job_id:
        return None
    fc = _function_call(job_id)
    if fc is None:
        return None
    try:
        fc.get(timeout=0)
        return runmeta.COMPLETED
    except TimeoutError:
        return runmeta.RUNNING
    except BaseException as e:  # noqa: BLE001 - the exception type IS the state
        name = type(e).__name__
        if name == "TimeoutError":  # some clients raise their own subclass
            return runmeta.RUNNING
        return e


def _volume():
    modal = _require_modal()
    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _read_volume_json(remote_path: str) -> dict | None:
    try:
        blob = b"".join(_volume().read_file(remote_path))
    except Exception:  # noqa: BLE001 - absent file, or no volume yet
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _put_volume_json(remote_path: str, payload: dict) -> None:
    """Write one small JSON file onto the Volume from the launching machine.

    Used for the `queued` manifest, so a submitted run has a status *before* any
    container exists to write one. Without it there is a window -- image pull,
    GPU wait -- in which a legitimately queued job is indistinguishable from a
    submission that silently failed.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "payload.json"
        local.write_text(json.dumps(payload, indent=2, default=str))
        vol = _volume()
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(local), remote_path)


# --- local job index ---------------------------------------------------------


def _job_record_path(run_name: str) -> Path:
    return JOBS_DIR / f"{runmeta.validate_run_name(run_name)}.json"


def _write_job_record(run_name: str, payload: dict) -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _job_record_path(run_name)
    runmeta.write_json(path, payload)
    return path


def _read_job_record(run_name: str) -> dict | None:
    path = _job_record_path(run_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _resolve_job_id(run_name: str, manifest: dict | None) -> str | None:
    return (manifest or {}).get("job_id") or (_read_job_record(run_name) or {}).get("job_id")


# --- submit ------------------------------------------------------------------


def _build_launch(a, *, run_name: str | None = None) -> tuple:
    """Resolve profile, run name and immutable metadata on the launching machine.

    The git commit is only knowable here, which is why `run.json` is built here
    and shipped to the worker rather than derived on it.
    """
    prof = runmeta.resolve_profile(
        a.profile,
        iterations=a.iterations or None,
        n_envs=a.n_envs or None,
        n_steps=a.n_steps or None,
        n_threat=a.n_threat or None,
        cell_m=a.cell_m or None,
    )
    name = runmeta.validate_run_name(run_name or a.run_name) if (run_name or a.run_name) \
        else runmeta.default_run_name(prof.name, a.seed)
    code = runmeta.git_info(REPO)
    meta = runmeta.build_metadata(
        prof,
        run_name=name,
        seed=a.seed,
        synthetic=a.synthetic,
        aoi=a.aoi or None,
        use_cbf=a.cbf,
        code=code,
        launcher="modal",
    )
    return prof, name, code, meta


def _relaunch_from(existing_meta: dict, run_name: str) -> tuple:
    """Rebuild a run's specification out of its own immutable `run.json`.

    Every field comes back from the record; only `code` is re-read from the
    working tree, so a commit that moved since the run started shows up as the
    one difference `resume_compatibility` is there to catch.
    """
    cfg = existing_meta.get("config") or {}
    prof = runmeta.resolve_profile(
        existing_meta.get("profile"),
        **{k: cfg.get(k) for k in
           ("iterations", "n_envs", "n_steps", "n_threat", "n_blue", "cell_m",
            "eval_every", "eval_worlds", "checkpoint_every")},
    )
    code = runmeta.git_info(REPO)
    meta = runmeta.build_metadata(
        prof,
        run_name=run_name,
        seed=cfg.get("seed", 0),
        synthetic=bool(cfg.get("synthetic")),
        aoi=cfg.get("aoi"),
        use_cbf=bool(cfg.get("use_cbf")),
        code=code,
        launcher="modal",
    )
    return prof, code, meta


def _check_not_live(run_name: str) -> None:
    """Refuse a second job for a run that already has one in flight.

    This is the check that actually prevents two writers in one run directory;
    the lock file inside the directory is the backstop for the case where this
    one could not see the truth (a Volume commit that has not propagated).
    """
    manifest = _read_volume_json(f"{run_name}/{runmeta.MANIFEST_FILENAME}")
    if manifest is None:
        return
    verdict = runmeta.classify_status(manifest, modal_state=_remote_state(manifest.get("job_id")))
    if verdict["status"] in runmeta.LIVE_STATES:
        raise SystemExit(
            f"{run_name} is already {verdict['status']} as job "
            f"{manifest.get('job_id')}. Two writers in one run directory interleave their "
            f"checkpoints and overwrite each other's history.json. Submit under a different "
            f"--run-name, or cancel that job first:\n"
            f"  uv run python scripts/modal_runs.py cancel {run_name}"
        )


def _spawn(prof, name: str, meta: dict, spec: dict, *, gpu: str) -> str:
    tag = "smoke_remote" if prof.name == "smoke" else "train_remote"
    timeout = SMOKE_TIMEOUT_S if prof.name == "smoke" else TIMEOUT_S
    fn = _deployed_function(tag)

    manifest = runmeta.build_manifest(
        run_name=name,
        job_id=None,
        profile=prof.name,
        gpu=gpu,
        timeout_s=timeout,
        max_retries=MAX_RETRIES,
        checkpoint_every=prof.checkpoint_every,
        code=meta.get("code"),
        config=meta.get("config"),
        status=runmeta.QUEUED,
    )
    _put_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}", manifest)

    call = fn.spawn(spec, meta)
    job_id = getattr(call, "object_id", None) or str(call)
    manifest["job_id"] = job_id
    _put_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}", manifest)
    _write_job_record(name, {
        "run_name": name,
        "job_id": job_id,
        "function": tag,
        "profile": prof.name,
        "gpu": gpu,
        "timeout_s": timeout,
        "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": (meta.get("code") or {}).get("commit"),
        "resume_from": spec.get("resume_from"),
    })
    return job_id


def cmd_submit(a) -> int:
    prof, name, code, meta = _build_launch(a)

    # The smoke gate, unchanged: an expensive profile requires a smoke run that
    # verified against THIS commit. The failures a smoke run catches -- a
    # dependency missing from the image, an unmounted cache, an unwritable
    # Volume, no GPU actually attached -- are exactly the ones a code change
    # reintroduces, which is why the marker is stamped with a commit.
    marker = _read_volume_json(runmeta.SMOKE_MARKER)
    refusal = runmeta.smoke_gate(prof.name, marker, code.get("commit"))
    if refusal and not a.skip_smoke_gate:
        raise SystemExit(f"refusing to submit --profile {prof.name}: {refusal}")
    if refusal:
        print(f"[gate] OVERRIDDEN: {refusal}")
    if code.get("dirty"):
        print("[gate] working tree is dirty; this run will not be reproducible from its commit")

    _check_not_live(name)
    spec = {
        "profile": prof.as_dict(),
        "run_name": name,
        "seed": a.seed,
        "synthetic": a.synthetic,
        "aoi": a.aoi or None,
        "use_cbf": a.cbf,
        "resume_from": None,
        "resume_overrides": [],
        "force_unlock": a.force_unlock,
        "job_id": None,
    }
    job_id = _spawn(prof, name, meta, spec, gpu=GPU_KIND)
    print(
        f"[submitted] run={name} job={job_id} profile={prof.name} gpu={GPU_KIND} "
        f"{prof.iterations} iterations x {prof.env_steps_per_iteration} env-steps "
        f"= {prof.total_env_steps:,} env-steps"
    )
    print(
        "This job now belongs to Modal, not to this process. You can close the laptop.\n\n"
        f"  uv run python scripts/modal_runs.py status {name}\n"
        f"  uv run python scripts/modal_runs.py logs   {name}\n"
        f"  uv run python scripts/modal_runs.py fetch  {name}"
    )
    return 0


# --- resume ------------------------------------------------------------------


def cmd_resume(a) -> int:
    """Continue an existing run from its most recent valid checkpoint.

    Explicit by design. There is no automatic retry of a long run anywhere in
    this path (`MAX_RETRIES` is 0 and says why): resuming is a decision made
    against a specific checkpoint after looking at why the run stopped.
    """
    name = runmeta.validate_run_name(a.run_name)
    manifest = _read_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}")
    existing_meta = _read_volume_json(f"{name}/{runmeta.META_FILENAME}")
    if existing_meta is None:
        raise SystemExit(
            f"{name} has no {runmeta.META_FILENAME} on the Volume: there is no recorded run "
            "to continue. Submit a new run instead."
        )

    state = _remote_state(_resolve_job_id(name, manifest))
    verdict = runmeta.classify_status(manifest, modal_state=state,
                                      checkpoints=(manifest or {}).get("checkpoints") or [])
    if verdict["status"] in runmeta.LIVE_STATES:
        raise SystemExit(
            f"{name} is {verdict['status']} (job {verdict['job_id']}). Resuming a live run "
            f"would put two writers in one directory. Cancel it first, or wait."
        )

    # The requested configuration is rebuilt from what the run RECORDED about
    # itself, not from this command line's defaults -- a resume is a
    # continuation, so the only thing that can legitimately differ is the code
    # commit, and defaulting `--profile` to `smoke` here would refuse every real
    # resume for the wrong reason. `--override-config` opts into the other
    # behaviour and is then subject to the same per-key refusals.
    if a.override_config:
        if not a.profile:
            raise SystemExit("--override-config requires an explicit --profile")
        prof, _, code, requested_meta = _build_launch(a, run_name=name)
    else:
        prof, code, requested_meta = _relaunch_from(existing_meta, name)
    try:
        compat = runmeta.resume_compatibility(
            existing_meta, requested_meta, overrides=a.override_resume
        )
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if not compat["ok"]:
        lines = "\n".join(f"  - {b}" for b in compat["blocking"])
        raise SystemExit(
            f"refusing to resume {name}: this is not a continuation of the recorded run.\n"
            f"{lines}\n\n"
            f"If the change is intentional, name each key you accept:\n"
            f"  --override-resume "
            + " --override-resume ".join(sorted(compat["changes"])) + "\n"
            f"Otherwise submit a new run; a run directory describes one experiment."
        )
    for note in compat["overridden"]:
        print(f"[resume] OVERRIDDEN: {note}")

    _check_not_live(name)
    rcfg = requested_meta["config"]
    spec = {
        "profile": prof.as_dict(),
        "run_name": name,
        "seed": rcfg["seed"],
        "synthetic": rcfg["synthetic"],
        "aoi": rcfg["aoi"],
        "use_cbf": rcfg["use_cbf"],
        # The worker re-derives the actual file: "most recent valid checkpoint"
        # is a question only the Volume can answer, and the answer may have
        # moved between submission and execution.
        "resume_from": "auto",
        "resume_overrides": list(a.override_resume),
        "force_unlock": a.force_unlock,
        "job_id": None,
    }
    job_id = _spawn(prof, name, requested_meta, spec, gpu=GPU_KIND)
    print(
        f"[resumed] run={name} job={job_id} continuing from the most recent valid checkpoint "
        f"on the Volume (manifest last saw iteration {verdict['last_iteration']})"
    )
    print(f"  uv run python scripts/modal_runs.py status {name}")
    return 0


# --- status / logs / cancel --------------------------------------------------


def cmd_status(a) -> int:
    name = runmeta.validate_run_name(a.run_name)
    manifest = _read_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}")
    job_id = _resolve_job_id(name, manifest)
    state = None if a.no_remote else _remote_state(job_id)
    if manifest is not None and job_id and not manifest.get("job_id"):
        manifest = {**manifest, "job_id": job_id}
    verdict = runmeta.classify_status(
        manifest, modal_state=state, checkpoints=(manifest or {}).get("checkpoints") or []
    )
    if manifest is None:
        print(
            f"no manifest for {name} on the Volume. Either it was never submitted, or the "
            f"submission failed before it could record anything."
        )
    payload = {
        **verdict,
        "profile": (manifest or {}).get("profile"),
        "requested_gpu": (manifest or {}).get("requested_gpu"),
        "actual_backend": (manifest or {}).get("actual_backend"),
        "actual_devices": (manifest or {}).get("actual_devices"),
        "commit": ((manifest or {}).get("code") or {}).get("commit"),
        "submitted_utc": (manifest or {}).get("submitted_utc"),
        "started_utc": (manifest or {}).get("started_utc"),
        "finished_utc": (manifest or {}).get("finished_utc"),
        "timeout_s": (manifest or {}).get("timeout_s"),
        "max_retries": (manifest or {}).get("max_retries"),
        "attempts": len((manifest or {}).get("attempts") or []),
        "error": runmeta.redact((manifest or {}).get("error") or "") or None,
    }
    print(json.dumps(payload, indent=2, default=str))
    if (manifest or {}).get("profile") == "smoke":
        # The smoke run's whole deliverable is which of the six things it proves
        # is broken, so it is printed rather than left inside the manifest.
        report = (manifest or {}).get("preflight") or {}
        print("\npreflight:")
        for name in runmeta.PREFLIGHT_CHECKS:
            entry = report.get(name)
            mark = "ok  " if (entry or {}).get("ok") else "FAIL"
            detail = (entry or {}).get("detail") or ("not run" if entry is None else "")
            print(f"  {mark} {name}: {runmeta.redact(str(detail))}")
        failed = runmeta.preflight_problems(report)
        if failed:
            print(f"\n{len(failed)} preflight check(s) did not pass; this smoke run does not "
                  "arm the gate in front of --profile short/full.")
    if verdict["resumable"]:
        print(
            f"\nresumable from iteration {verdict['resume_from_iteration']}:\n"
            f"  uv run python scripts/modal_runs.py resume {name}"
        )
    return 0 if verdict["status"] == runmeta.COMPLETED else 1


def cmd_jobs(a) -> int:
    """Every submission this machine made. Offline: reads the local index only."""
    if not JOBS_DIR.is_dir():
        print(f"no submissions recorded in {JOBS_DIR}")
        return 0
    rows = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    print(json.dumps(rows, indent=2, default=str))
    return 0


def cmd_logs(a) -> int:
    """The run's own log off the Volume, or the app's live stream.

    `train.log` is the trustworthy one: it is per-run, it is committed alongside
    the checkpoints it describes, and it is still there next week. `--stream`
    shells out to `modal app logs`, which is live but is addressed by app rather
    than by run and does not outlive Modal's retention.
    """
    name = runmeta.validate_run_name(a.run_name)
    if a.stream:
        return _modal_cli("app", "logs", APP_NAME)
    text = None
    try:
        blob = b"".join(_volume().read_file(f"{name}/{LOG_FILENAME}"))
        text = runmeta.redact(blob.decode("utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        print(f"no {LOG_FILENAME} for {name} on the Volume ({runmeta.redact(str(e))}).")
        print(f"The container may not have started yet. Live stream: "
              f"scripts/modal_runs.py logs {name} --stream")
        return 1
    lines = text.splitlines()
    if a.tail and len(lines) > a.tail:
        print(f"... {len(lines) - a.tail} earlier lines omitted ...")
        lines = lines[-a.tail :]
    print("\n".join(lines))
    return 0


def cmd_cancel(a) -> int:
    name = runmeta.validate_run_name(a.run_name)
    manifest = _read_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}")
    job_id = _resolve_job_id(name, manifest)
    if not job_id:
        raise SystemExit(f"no job id recorded for {name}; nothing to cancel")
    fc = _function_call(job_id)
    if fc is None:
        return 1
    fc.cancel()
    # The container cannot record its own cancellation, so the launching machine
    # does it. Everything already committed to the Volume stays exactly as it is.
    if manifest is not None:
        _put_volume_json(f"{name}/{runmeta.MANIFEST_FILENAME}", {
            **manifest,
            "status": runmeta.CANCELLED,
            "termination_reason": runmeta.REASON_CANCELLED,
            "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    print(f"cancelled job {job_id} for {name}. Artifacts committed so far are untouched:")
    print(f"  uv run python scripts/modal_runs.py fetch {name}")
    return 0


# --- list / fetch / verify ---------------------------------------------------


def cmd_list(a) -> int:
    return _modal_cli("volume", "ls", VOLUME_NAME, a.path)


def cmd_fetch(a) -> int:
    name = runmeta.validate_run_name(a.run_name)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    rc = _modal_cli("volume", "get", VOLUME_NAME, f"{RUNS_ROOT}/{name}", str(dest), "--force")
    if rc != 0:
        return rc
    return _report(dest / name)


def cmd_verify(a) -> int:
    return _report(Path(a.path))


def _report(directory: Path) -> int:
    print(json.dumps(runmeta.summarize_run(directory), indent=2, default=str))
    manifest = runmeta.read_manifest(directory)
    if manifest is not None:
        ckpts = [runmeta.checkpoint_iteration(p) for p in runmeta.checkpoint_paths(directory)]
        verdict = runmeta.classify_status(manifest, checkpoints=[c for c in ckpts if c is not None])
        print("\nmanifest:")
        print(json.dumps({k: verdict[k] for k in
                          ("status", "resumable", "termination_reason", "job_id",
                           "last_iteration", "iterations_declared", "notes")},
                         indent=2, default=str))
    problems = runmeta.verify_run_dir(directory)
    if not problems:
        print(f"\nOK: {directory} is a complete, self-describing run.")
        return 0
    print(f"\n{len(problems)} problem(s) with {directory}:")
    for p in problems:
        print(f"  - {p}")
    return 1


# --- argument plumbing -------------------------------------------------------


def _add_launch_args(p, *, with_run_name: bool = True) -> None:
    p.add_argument("--profile", default="smoke", choices=sorted(runmeta.PROFILES))
    if with_run_name:
        p.add_argument("--run-name", default="", help="default: <profile>-s<seed>-<utc timestamp>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true",
                   help="synthetic ridged terrain: no cited data, so not a reportable result")
    p.add_argument("--aoi", default="", help="theatre to train on (default: the packaged one)")
    p.add_argument("--cbf", action="store_true", help="run the HOCBF-QP backstop during evaluation")
    for flag, kind in (("--iterations", int), ("--n-envs", int), ("--n-steps", int),
                       ("--n-threat", int)):
        p.add_argument(flag, type=kind, default=0, help="override the profile's value")
    p.add_argument("--cell-m", type=float, default=0.0, help="override the profile's value")
    p.add_argument("--force-unlock", action="store_true",
                   help="take the run directory's writer lock from a worker that is gone")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"defaults: gpu={GPU_KIND} timeout={TIMEOUT_S}s "
               f"(smoke {SMOKE_TIMEOUT_S}s) retries={MAX_RETRIES}",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="start a detached run and return its job id")
    _add_launch_args(p_submit)
    p_submit.add_argument("--skip-smoke-gate", action="store_true",
                          help="launch an expensive profile with no verified smoke run")
    p_submit.set_defaults(fn=cmd_submit)

    p_resume = sub.add_parser(
        "resume", help="continue a stopped run from its most recent valid checkpoint")
    p_resume.add_argument("run_name")
    p_resume.add_argument(
        "--override-config", action="store_true",
        help="rebuild the run specification from these flags instead of from the recorded "
             "run.json. Almost never what you want: a resume is a continuation.")
    _add_launch_args(p_resume, with_run_name=False)
    p_resume.set_defaults(profile="")
    p_resume.add_argument(
        "--override-resume", action="append", default=[], metavar="KEY",
        help="accept one named incompatibility (e.g. code.commit). Repeatable. "
             f"Keys: {', '.join(runmeta.RESUME_BLOCKING_KEYS)}")
    p_resume.set_defaults(fn=cmd_resume)

    p_status = sub.add_parser("status", help="queued / running / completed / timed out / failed / "
                                             "cancelled, and whether it is resumable")
    p_status.add_argument("run_name")
    p_status.add_argument("--no-remote", action="store_true",
                          help="read the Volume manifest only; do not query Modal")
    p_status.set_defaults(fn=cmd_status)

    p_logs = sub.add_parser("logs", help="the run's log off the Volume")
    p_logs.add_argument("run_name")
    p_logs.add_argument("--tail", type=int, default=200, help="0 for the whole file")
    p_logs.add_argument("--stream", action="store_true",
                        help="follow the live app log instead (not per-run)")
    p_logs.set_defaults(fn=cmd_logs)

    p_cancel = sub.add_parser("cancel", help="stop a running job; artifacts are kept")
    p_cancel.add_argument("run_name")
    p_cancel.set_defaults(fn=cmd_cancel)

    p_jobs = sub.add_parser("jobs", help="submissions made from this machine (offline)")
    p_jobs.set_defaults(fn=cmd_jobs)

    p_list = sub.add_parser("list", help="list runs on the Volume")
    p_list.add_argument("path", nargs="?", default="/", help="path within the Volume")
    p_list.set_defaults(fn=cmd_list)

    p_fetch = sub.add_parser("fetch", help="download one run, then verify it")
    p_fetch.add_argument("run_name")
    p_fetch.add_argument("--dest", default="runs", help="local directory to download into")
    p_fetch.set_defaults(fn=cmd_fetch)

    p_verify = sub.add_parser("verify", help="check an already-downloaded run directory (offline)")
    p_verify.add_argument("path")
    p_verify.set_defaults(fn=cmd_verify)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
