"""Run identity, cost profiles and output verification for training runs.

Deliberately dependency-free (stdlib only, no JAX, no Modal) so the rules that
decide *what* a run is can be unit tested without a GPU, without a network and
without importing the training stack. `train.py` writes what this module
describes; `modal_train.py` and `scripts/modal_runs.py` are thin users of it.

Three problems this exists to solve, all of them learned from the local runs:

  * **Run isolation.** Every run so far wrote into a path the caller typed by
    hand (`runs/theatre`). Two runs with different settings and the same `--out`
    silently interleave their checkpoints and their `history.json`, and the
    second one looks like a continuation of the first. Run names are validated
    and default to something unique.
  * **Immutable metadata.** A `history.json` on its own does not record what
    produced it. `run.json` is written once per run directory and a second write
    with a *different* configuration is refused rather than overwriting, so a
    directory cannot come to describe a run other than the one inside it.
  * **Verification.** A run that hit a timeout leaves a directory that looks
    exactly like a completed one, only shorter. `verify_run_dir` is the check
    that says so out loud, including the one that matters most for a GPU run:
    whether it actually ran on a GPU.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 1

META_FILENAME = "run.json"
PERF_FILENAME = "perf.json"
HISTORY_FILENAME = "history.json"
THEATRE_FILENAME = "theatre.json"
SMOKE_MARKER = ".smoke_ok"  # written at the runs root, not inside a run dir

# A run name becomes a directory name on a shared Modal Volume. Anything that
# can traverse out of the runs root, collide after normalisation, or confuse a
# shell is rejected here rather than at the filesystem.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# The parts of `run.json` that define *which run this is*. Everything outside
# this set (device, versions, wall-clock) may legitimately differ between a run
# and a retry of that same run, so it is not compared.
IDENTITY_KEYS = ("run_name", "profile", "config", "code")


class RunNameError(ValueError):
    """The requested run name cannot be used as a run directory."""


class RunCollision(RuntimeError):
    """A run directory already describes a different run."""


# --- cost profiles ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunProfile:
    """A named training size. Nothing here is a claim about speed."""

    name: str
    purpose: str
    iterations: int
    n_envs: int
    n_steps: int
    n_threat: int
    n_blue: int
    cell_m: float
    eval_every: int
    eval_worlds: int
    checkpoint_every: int
    overrides: tuple[str, ...] = ()

    # --- how much work one iteration is, which is the only cost unit that is
    # comparable across profiles and across devices ---
    @property
    def env_steps_per_iteration(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def agent_steps_per_iteration(self) -> int:
        return self.env_steps_per_iteration * self.n_blue

    @property
    def total_env_steps(self) -> int:
        return self.env_steps_per_iteration * self.iterations

    def ppo_kwargs(self) -> dict:
        """Keyword arguments for `naigos.rl.ppo.PPOConfig`."""
        return {"n_envs": self.n_envs, "n_steps": self.n_steps}

    def train_kwargs(self) -> dict:
        """Keyword arguments for `naigos.rl.train.TrainConfig`."""
        return {
            "iterations": self.iterations,
            "eval_every": self.eval_every,
            "eval_worlds": self.eval_worlds,
            "checkpoint_every": self.checkpoint_every,
        }

    def theatre_kwargs(self) -> dict:
        """Keyword arguments for `naigos.env.theatre_bridge.env_from_theatre`."""
        return {"n_blue": self.n_blue, "n_threat": self.n_threat, "cell_m": self.cell_m}

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# `smoke` exists to be run FIRST, every time, on a new image or a new commit.
# It is sized so the failure modes that only appear remotely -- image missing a
# dependency, cache not mounted, Volume not writable, no GPU actually attached
# -- all surface within a few minutes instead of six hours in.
#
# `cell_m` follows the fidelity a result is reported at: 500 m for anything
# publishable (see DEVLOG on the line-of-sight correction), 1500 m for smoke,
# where the point is that the plumbing works and not what the policy learns.
PROFILES: dict[str, RunProfile] = {
    "smoke": RunProfile(
        name="smoke",
        purpose="prove the image, the GPU, the cache mount and the Volume in minutes",
        iterations=3,
        n_envs=16,
        n_steps=32,
        n_threat=8,
        n_blue=4,
        cell_m=1500.0,
        eval_every=1,
        eval_worlds=8,
        checkpoint_every=3,
    ),
    "short": RunProfile(
        name="short",
        purpose="a real but bounded run: enough to read a learning curve and a throughput number",
        iterations=200,
        n_envs=128,
        n_steps=128,
        n_threat=16,
        n_blue=4,
        cell_m=500.0,
        eval_every=25,
        eval_worlds=32,
        checkpoint_every=50,
    ),
    "full": RunProfile(
        name="full",
        purpose="the run next-steps.md C-2 asks for: full curriculum difficulty",
        iterations=3000,
        n_envs=256,
        n_steps=128,
        n_threat=16,
        n_blue=4,
        cell_m=500.0,
        eval_every=25,
        eval_worlds=64,
        checkpoint_every=200,
    ),
}

_OVERRIDABLE = ("iterations", "n_envs", "n_steps", "n_threat", "n_blue", "cell_m",
                "eval_every", "eval_worlds", "checkpoint_every")


def resolve_profile(name: str, **overrides) -> RunProfile:
    """Look up a profile and apply explicit overrides.

    `None` overrides are dropped, so a CLI can pass every flag unconditionally
    and still get the profile's value for the ones the user did not set.
    """
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    given = {k: v for k, v in overrides.items() if v is not None}
    bad = sorted(set(given) - set(_OVERRIDABLE))
    if bad:
        raise ValueError(f"cannot override {bad}; overridable fields are {list(_OVERRIDABLE)}")
    for k, v in given.items():
        if v <= 0:
            raise ValueError(f"{k} must be positive, got {v!r}")
    base = PROFILES[name]
    return dataclasses.replace(base, overrides=tuple(sorted(given)), **given)


# --- run naming and directories ---------------------------------------------


def validate_run_name(name: str) -> str:
    """Return `name` if it is usable as a run directory, else raise."""
    if not isinstance(name, str) or not RUN_NAME_RE.match(name):
        raise RunNameError(
            f"invalid run name {name!r}: 1-64 chars, must start alphanumeric and "
            "contain only letters, digits, dot, dash or underscore (no path separators)"
        )
    if name in (".", "..") or name.startswith("."):
        raise RunNameError(f"invalid run name {name!r}: must not start with a dot")
    return name


def default_run_name(profile: str, seed: int, now: datetime | None = None) -> str:
    """`<profile>-s<seed>-<utc timestamp>`.

    Unique per second, which is enough: two runs launched in the same second
    with the same profile and seed would be the same run twice.
    """
    now = now or datetime.now(timezone.utc)
    return validate_run_name(f"{profile}-s{int(seed)}-{now.strftime('%Y%m%dT%H%M%SZ')}")


def run_dir(root: str | os.PathLike, run_name: str) -> Path:
    """Join a validated run name onto the runs root. Cannot escape the root."""
    return Path(root) / validate_run_name(run_name)


# --- immutable run metadata --------------------------------------------------


def git_info(repo: str | os.PathLike | None = None) -> dict:
    """Commit and dirty flag, or nulls outside a git checkout.

    Recorded because a run's identity includes the code that produced it; a
    dirty tree is reported rather than hidden, since a result from one is not
    reproducible from the commit it names.
    """
    cwd = str(repo) if repo is not None else None

    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None

    head = _git("rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return {"commit": None, "dirty": None}
    status = _git("status", "--porcelain")
    dirty = None if status is None or status.returncode != 0 else bool(status.stdout.strip())
    return {"commit": head.stdout.strip(), "dirty": dirty}


def build_metadata(
    profile: RunProfile,
    *,
    run_name: str,
    seed: int,
    synthetic: bool,
    aoi: str | None = None,
    use_cbf: bool = False,
    code: dict | None = None,
    runtime: dict | None = None,
    launcher: str | None = None,
    now: datetime | None = None,
) -> dict:
    """The `run.json` payload.

    Carries configuration and provenance only. It never captures the process
    environment: `os.environ` on a machine that has run `modal token set` or
    exported an ion token would put credentials into a file that gets committed
    alongside results.
    """
    validate_run_name(run_name)
    now = now or datetime.now(timezone.utc)
    return {
        "schema": SCHEMA_VERSION,
        "run_name": run_name,
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "launcher": launcher,
        "profile": profile.name,
        "config": {
            **{k: getattr(profile, k) for k in _OVERRIDABLE},
            "overrides": list(profile.overrides),
            "seed": int(seed),
            "synthetic": bool(synthetic),
            "aoi": aoi,
            "use_cbf": bool(use_cbf),
            "env_steps_per_iteration": profile.env_steps_per_iteration,
            "total_env_steps": profile.total_env_steps,
        },
        "code": code if code is not None else {"commit": None, "dirty": None},
        # Mutable half: what the run turned out to execute on. Deliberately
        # OUTSIDE the identity comparison -- the same run re-launched onto a
        # different GPU is still the same run specification.
        "runtime": runtime or {},
    }


def identity(meta: dict) -> dict:
    return {k: meta.get(k) for k in IDENTITY_KEYS}


def read_metadata(directory: str | os.PathLike) -> dict | None:
    path = Path(directory) / META_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_metadata(directory: str | os.PathLike, meta: dict) -> tuple[dict, bool]:
    """Write `run.json` once. Returns `(metadata_on_disk, was_written)`.

    A second write of the *same* run specification is a no-op, so a retry into
    the same directory is allowed. A write that would change the specification
    raises `RunCollision`: the alternative is a directory whose metadata
    describes one run and whose checkpoints came from another.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / META_FILENAME
    existing = read_metadata(directory)
    if existing is not None:
        old, new = identity(existing), identity(meta)
        if old != new:
            differing = sorted(k for k in IDENTITY_KEYS if old.get(k) != new.get(k))
            raise RunCollision(
                f"{path} already describes a different run (differs in {differing}). "
                "Point this run at a new directory (--out locally, --run-name on Modal) "
                "rather than writing into this one."
            )
        return existing, False
    write_json(path, meta)
    return meta, True


def write_json(path: str | os.PathLike, payload) -> None:
    """Atomic-ish write: a killed process leaves the old file, not a half file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


# --- the smoke gate ----------------------------------------------------------


def smoke_marker_payload(run_name: str, code: dict, runtime: dict | None = None,
                         now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {
        "run_name": run_name,
        "commit": (code or {}).get("commit"),
        "dirty": (code or {}).get("dirty"),
        "completed_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime": runtime or {},
    }


def smoke_gate(profile: str, marker: dict | None, commit: str | None) -> str | None:
    """Return a refusal message, or None if an expensive run may proceed.

    The smoke profile is always allowed -- it is the thing being asked for. Any
    larger profile requires a smoke run recorded against *this* commit, because
    the failures a smoke run catches (a missing dependency in the image, an
    unmounted cache, no GPU actually attached) are exactly the ones a code
    change reintroduces.
    """
    if profile == "smoke":
        return None
    if not marker:
        return (
            "no smoke run is recorded on the volume. Run "
            "`modal run naigos/rl/modal_train.py --profile smoke` first, "
            "or pass --skip-smoke-gate to override."
        )
    if commit is None or marker.get("commit") is None:
        return None  # cannot verify; a warning is printed by the caller instead
    if marker["commit"] != commit:
        return (
            f"the last smoke run was against commit {marker['commit'][:12]}, this launch is "
            f"{commit[:12]}. Re-run `--profile smoke`, or pass --skip-smoke-gate to override."
        )
    return None


# --- output verification -----------------------------------------------------


def _checkpoint_iters(directory: Path) -> list[int]:
    out = []
    for p in directory.glob("ckpt_*.pkl"):
        stem = p.stem.split("_")[-1]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def verify_run_dir(directory: str | os.PathLike) -> list[str]:
    """Return the list of problems with a fetched run directory. Empty is good.

    This is the check that a run which hit the timeout, silently fell back to
    CPU, or was interrupted before its first checkpoint does not read as a
    completed GPU run.
    """
    directory = Path(directory)
    problems: list[str] = []
    if not directory.is_dir():
        return [f"{directory} is not a directory"]

    meta = None
    try:
        meta = read_metadata(directory)
    except json.JSONDecodeError as e:
        problems.append(f"{META_FILENAME} does not parse: {e}")
    if meta is None:
        problems.append(f"no {META_FILENAME}: this directory does not record what produced it")
    else:
        if meta.get("schema") != SCHEMA_VERSION:
            problems.append(f"{META_FILENAME} schema {meta.get('schema')!r}, expected {SCHEMA_VERSION}")
        if meta.get("run_name") != directory.name:
            problems.append(
                f"{META_FILENAME} run_name {meta.get('run_name')!r} does not match "
                f"directory name {directory.name!r}"
            )
        if (meta.get("code") or {}).get("dirty"):
            problems.append("produced from a dirty working tree: not reproducible from its commit")

    cfg = (meta or {}).get("config") or {}
    declared = cfg.get("iterations")

    history = None
    hpath = directory / HISTORY_FILENAME
    if not hpath.exists():
        problems.append(f"no {HISTORY_FILENAME}")
    else:
        try:
            history = json.loads(hpath.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{HISTORY_FILENAME} does not parse: {e}")
    if isinstance(history, list) and history:
        iters = [row.get("iter") for row in history]
        if any(i is None for i in iters):
            problems.append(f"{HISTORY_FILENAME} has a row with no `iter`")
        elif any(b < a for a, b in zip(iters, iters[1:])):
            problems.append(f"{HISTORY_FILENAME} iterations are not monotonic: two runs in one directory?")
        elif declared and iters[-1] < declared:
            problems.append(
                f"run stopped at iteration {iters[-1]} of {declared}: truncated, not a completed run"
            )
    elif history is not None:
        problems.append(f"{HISTORY_FILENAME} is empty")

    ckpts = _checkpoint_iters(directory)
    if not ckpts:
        problems.append("no checkpoints: nothing was persisted")
    elif declared and ckpts[-1] < declared:
        problems.append(f"last checkpoint is iteration {ckpts[-1]} of {declared}")

    perf = None
    ppath = directory / PERF_FILENAME
    if not ppath.exists():
        problems.append(f"no {PERF_FILENAME}: no device or throughput record")
    else:
        try:
            perf = json.loads(ppath.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{PERF_FILENAME} does not parse: {e}")
    if isinstance(perf, dict):
        platform = ((perf.get("device") or {}).get("platform") or "").lower()
        if not platform:
            problems.append(f"{PERF_FILENAME} records no device platform")
        elif (meta or {}).get("launcher") == "modal" and platform != "gpu":
            problems.append(
                f"launched on Modal but ran on {platform!r}: any speed number from this run "
                "is a CPU number"
            )
        if perf.get("steady_iteration_s_median") in (None, 0):
            problems.append(f"{PERF_FILENAME} has no steady-state iteration time")

    if cfg and not cfg.get("synthetic") and not (directory / THEATRE_FILENAME).exists():
        problems.append(
            f"a real-theatre run with no {THEATRE_FILENAME}: its data provenance was not recorded"
        )

    # The manifest is how a remote run says why it stopped. A job that timed out
    # or was cancelled leaves a directory shaped exactly like a short completed
    # one, so the reason it stopped is part of verifying it -- and its absence on
    # a Modal-launched run means the status record itself did not survive.
    man = read_manifest(directory)
    if man is None:
        if (meta or {}).get("launcher") == "modal":
            problems.append(
                f"no {MANIFEST_FILENAME}: launched on Modal but nothing recorded which job "
                "ran it or why it stopped"
            )
    else:
        verdict = classify_status(
            man, checkpoints=[i for i in _checkpoint_iters(directory)]
        )
        if verdict["status"] != COMPLETED:
            problems.append(
                f"{MANIFEST_FILENAME} status is {verdict['status']!r}"
                + (f" ({verdict['termination_reason']})" if verdict["termination_reason"] else "")
                + (f"; resumable from iteration {verdict['resume_from_iteration']}"
                   if verdict["resumable"] else "")
            )
        if man.get("run_name") not in (None, directory.name):
            problems.append(
                f"{MANIFEST_FILENAME} run_name {man.get('run_name')!r} does not match "
                f"directory name {directory.name!r}"
            )
        if man.get("error"):
            problems.append(f"the worker recorded an error: {redact(str(man['error']))}")
        if man.get("profile") == "smoke":
            problems.extend(preflight_problems(man.get("preflight")))
    return problems


def summarize_run(directory: str | os.PathLike) -> dict:
    """A short, printable description of a fetched run."""
    directory = Path(directory)
    meta = None
    try:
        meta = read_metadata(directory)
    except (OSError, json.JSONDecodeError):
        pass
    perf: dict = {}
    try:
        perf = json.loads((directory / PERF_FILENAME).read_text())
    except (OSError, json.JSONDecodeError):
        pass
    last: dict = {}
    try:
        rows = json.loads((directory / HISTORY_FILENAME).read_text())
        last = rows[-1] if rows else {}
    except (OSError, json.JSONDecodeError, IndexError):
        pass
    man = read_manifest(directory) or {}
    return {
        "run_name": (meta or {}).get("run_name", directory.name),
        "profile": (meta or {}).get("profile"),
        "status": man.get("status"),
        "termination_reason": man.get("termination_reason"),
        "job_id": man.get("job_id"),
        # A resumed run's `perf.json` covers only the resumed segment, so saying
        # so next to the throughput number is not decoration.
        "resumed": bool(perf.get("resumed")),
        "resumed_from_iteration": perf.get("resumed_from_iteration"),
        "commit": ((meta or {}).get("code") or {}).get("commit"),
        "device": (perf.get("device") or {}).get("devices"),
        "platform": (perf.get("device") or {}).get("platform"),
        "iterations_done": last.get("iter"),
        "iterations_declared": ((meta or {}).get("config") or {}).get("iterations"),
        "steady_iteration_s": perf.get("steady_iteration_s_median"),
        "env_steps_per_s": perf.get("env_steps_per_s"),
        "peak_mem_mb": perf.get("peak_mem_mb"),
        "wall_s": perf.get("wall_s"),
        "checkpoints": _checkpoint_iters(directory),
        "survival_rate": last.get("survival_rate"),
    }


# --- run status: the seven distinctions a remote run has to support ----------
#
# Six of these are LIFECYCLE states and exactly one of them is true at a time.
# "Resumable" is deliberately NOT one of them: a run can be timed out *and*
# resumable, or failed *and* not resumable because it died before its first
# checkpoint. Modelling it as a seventh mutually-exclusive state would force a
# lie in exactly the case that matters. It is a separate boolean, computed from
# whether recovery state actually exists on disk.

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
TIMED_OUT = "timed_out"
FAILED = "failed"
CANCELLED = "cancelled"
UNKNOWN = "unknown"

LIFECYCLE_STATES = (QUEUED, RUNNING, COMPLETED, TIMED_OUT, FAILED, CANCELLED, UNKNOWN)
TERMINAL_STATES = (COMPLETED, TIMED_OUT, FAILED, CANCELLED)
#: A run in one of these has a live or pending writer; a second writer must not
#: be pointed at the same directory.
LIVE_STATES = (QUEUED, RUNNING)

# Reasons a run stopped, recorded by whoever knew: the worker for the first two,
# the launching machine for the rest (a container that is killed cannot write).
REASON_COMPLETED = "completed"
REASON_EXCEPTION = "exception"
REASON_TIMEOUT = "timeout"
REASON_CANCELLED = "cancelled"
REASON_LOST = "lost"  # terminal upstream, no worker record: killed without warning


class RunBusy(RuntimeError):
    """Another writer holds this run directory."""


class ResumeRefused(RuntimeError):
    """The requested resume is not a continuation of the recorded run."""


# Modal reports states through several spellings depending on whether they come
# from the CLI, the gRPC enums or an exception type, and the spellings have
# changed between client versions. Normalising here -- rather than at each call
# site -- means a client upgrade that renames a state degrades to `unknown`
# instead of silently reading as `completed`.
_MODAL_STATE_MAP = {
    "pending": QUEUED,
    "queued": QUEUED,
    "scheduled": QUEUED,
    "enqueued": QUEUED,
    "running": RUNNING,
    "started": RUNNING,
    "in_progress": RUNNING,
    "ephemeral": RUNNING,
    "deployed": RUNNING,
    "success": COMPLETED,
    "succeeded": COMPLETED,
    "complete": COMPLETED,
    "completed": COMPLETED,
    "finished": COMPLETED,
    "done": COMPLETED,
    "failure": FAILED,
    "failed": FAILED,
    "error": FAILED,
    "crashed": FAILED,
    "functionfailederror": FAILED,
    "remoteerror": FAILED,
    "timeout": TIMED_OUT,
    "timed_out": TIMED_OUT,
    "timedout": TIMED_OUT,
    "expired": TIMED_OUT,
    "functiontimeouterror": TIMED_OUT,
    "cancelled": CANCELLED,
    "canceled": CANCELLED,
    "terminated": CANCELLED,
    "stopped": CANCELLED,
    "aborted": CANCELLED,
    "functioncancelled": CANCELLED,
}

# gRPC enum members arrive fully qualified; the prefix carries no information.
_MODAL_STATE_PREFIXES = (
    "input_status_",
    "function_call_status_",
    "app_state_",
    "task_state_",
    "status_",
    "state_",
)


def parse_modal_state(raw: object) -> str:
    """Map anything Modal calls a state onto one lifecycle state.

    Accepts an enum member, an exception class, an exception instance or a
    string in any of the spellings above. Returns `UNKNOWN` for anything not
    recognised, which is the honest answer and is never mistaken for success.
    """
    if raw is None:
        return UNKNOWN
    if isinstance(raw, type):
        text = raw.__name__
    elif isinstance(raw, BaseException):
        text = type(raw).__name__
    elif isinstance(raw, str):
        text = raw
    else:
        text = getattr(raw, "name", None) or str(raw)
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    text = text.rsplit(".", 1)[-1]
    for prefix in _MODAL_STATE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return _MODAL_STATE_MAP.get(text, _MODAL_STATE_MAP.get(text.replace("_", ""), UNKNOWN))


# A worker writes a heartbeat into the manifest at every persist. If the last
# one is older than this and the job is still nominally running, the run is
# reported as `unknown` rather than as `running`: a container that was killed
# between commits leaves a manifest that says "running" forever.
STALE_HEARTBEAT_S = 3600


def _parse_utc(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def classify_status(
    manifest: dict | None,
    *,
    modal_state: object = None,
    checkpoints: list[int] | None = None,
    now: datetime | None = None,
    stale_after_s: int = STALE_HEARTBEAT_S,
) -> dict:
    """Merge what the worker recorded with what Modal reports into one verdict.

    Two independent sources, because neither alone is trustworthy:

      * the **manifest on the Volume** is written by the worker and is the only
        thing that knows how far training actually got -- but a container that
        is killed (timeout, preemption, cancellation) never gets to update it,
        so a stale `running` is its normal failure mode;
      * the **Modal call state** knows the container is gone but not whether
        anything was persisted before it went.

    Precedence: a worker-recorded *terminal* reason wins, because the worker was
    there. Otherwise an upstream terminal state overrides a live manifest. A
    live manifest with a stale heartbeat and no upstream signal is `unknown`.
    """
    manifest = manifest or {}
    now = now or datetime.now(timezone.utc)
    ckpts = list(checkpoints or [])
    upstream = parse_modal_state(modal_state)

    recorded = manifest.get("status")
    if recorded not in LIFECYCLE_STATES:
        recorded = UNKNOWN
    reason = manifest.get("termination_reason")
    notes: list[str] = []

    if recorded in TERMINAL_STATES:
        status = recorded
        if upstream in TERMINAL_STATES and upstream != recorded:
            # e.g. the worker caught its own exception and then the container
            # was also reaped. The worker's account is the specific one.
            notes.append(f"modal reports {upstream!r}, worker recorded {recorded!r}; worker wins")
    elif upstream in TERMINAL_STATES:
        status = upstream
        reason = reason or {
            COMPLETED: REASON_COMPLETED,
            TIMED_OUT: REASON_TIMEOUT,
            CANCELLED: REASON_CANCELLED,
            FAILED: REASON_EXCEPTION,
        }[upstream]
        if recorded in LIVE_STATES:
            notes.append(
                f"worker was still {recorded!r} when the job ended: it was killed "
                "without being able to record why"
            )
            if upstream != COMPLETED:
                reason = REASON_LOST if upstream == FAILED else reason
    elif upstream in LIVE_STATES:
        status = upstream if recorded == UNKNOWN else recorded
    else:
        status = recorded

    beat = _parse_utc(manifest.get("heartbeat_utc") or manifest.get("submitted_utc"))
    age_s = None
    if beat is not None:
        age_s = max(int((now - beat).total_seconds()), 0)
    if status in LIVE_STATES and upstream == UNKNOWN and age_s is not None and age_s > stale_after_s:
        notes.append(
            f"no heartbeat for {age_s}s and Modal reported nothing: the job state cannot be confirmed"
        )
        status = UNKNOWN

    declared = ((manifest.get("config") or {}).get("iterations")) or manifest.get("iterations_declared")
    done = manifest.get("last_iteration")
    truncated = bool(declared and done is not None and done < declared)
    if status == COMPLETED and truncated:
        notes.append(f"marked completed at iteration {done} of {declared}: truncated")

    # Resumable is about recovery state existing, not about how the run ended.
    resumable = bool(ckpts) and (status in (TIMED_OUT, FAILED, CANCELLED, UNKNOWN)
                                 or (status == COMPLETED and truncated))
    if not ckpts and status in (TIMED_OUT, FAILED, CANCELLED):
        notes.append("no checkpoint on the Volume: this run cannot be resumed, only restarted")

    return {
        "run_name": manifest.get("run_name"),
        "job_id": manifest.get("job_id"),
        "status": status,
        "resumable": resumable,
        "termination_reason": reason,
        "modal_state": upstream,
        "worker_status": recorded,
        "last_iteration": done,
        "iterations_declared": declared,
        "checkpoints": ckpts,
        "resume_from_iteration": ckpts[-1] if (resumable and ckpts) else None,
        "heartbeat_age_s": age_s,
        "notes": notes,
    }


# --- the smoke run's preflight ----------------------------------------------

#: What a smoke run exists to prove, named one by one. A smoke run that
#: "completed" without an answer for each of these has not done its job -- the
#: whole point of paying for three iterations on a GPU is to learn which of
#: these six things is broken, in minutes rather than six hours in.
PREFLIGHT_CHECKS = (
    "image",          # the remote image built and imports the training stack
    "gpu_backend",    # JAX resolved a CUDA backend, not a silent CPU fallback
    "data_cache",     # the cited cache travelled with the image and is readable
    "volume_write",   # the Volume is mounted, writable and survives a commit
    "checkpoint",     # a checkpoint was written and reads back
    "artifact_fetch", # the run directory is complete enough to retrieve
)


def preflight_problems(report: dict | None) -> list[str]:
    """Turn a preflight report into the list of things that are not proven.

    A check that is absent is a problem, not a pass. The failure this guards
    against is a preflight that silently stopped reporting a check after a
    refactor, leaving a smoke run that verifies less than it claims to.
    """
    report = report or {}
    problems = []
    for name in PREFLIGHT_CHECKS:
        entry = report.get(name)
        if entry is None:
            problems.append(f"preflight check {name!r} did not run")
        elif not entry.get("ok"):
            problems.append(
                f"preflight check {name!r} failed: {entry.get('detail') or 'no detail recorded'}"
            )
    return problems


# --- the run manifest --------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA = 1


def build_manifest(
    *,
    run_name: str,
    job_id: str | None,
    profile: str,
    gpu: str | None,
    timeout_s: int | None,
    max_retries: int,
    checkpoint_every: int,
    code: dict | None = None,
    config: dict | None = None,
    status: str = QUEUED,
    now: datetime | None = None,
) -> dict:
    """The mutable companion to the immutable `run.json`.

    `run.json` says what run this is and must never change. This says what
    happened to it -- which job carried it, when, on what device it landed, and
    why it stopped -- and is rewritten throughout the run's life. Keeping the
    two apart is what lets the second be honest about a timeout without the
    first ceasing to describe the run.
    """
    validate_run_name(run_name)
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": MANIFEST_SCHEMA,
        "run_name": run_name,
        "job_id": job_id,
        "profile": profile,
        "status": status,
        "termination_reason": None,
        "submitted_utc": stamp,
        "started_utc": None,
        "finished_utc": None,
        "heartbeat_utc": stamp,
        "requested_gpu": gpu,
        "timeout_s": timeout_s,
        "max_retries": int(max_retries),
        "checkpoint_every": int(checkpoint_every),
        "actual_backend": None,
        "actual_devices": None,
        "code": code or {"commit": None, "dirty": None},
        "config": config or {},
        "last_iteration": None,
        # One entry per execution of this run: the first submission and every
        # resume. Appended to, never rewritten, so the history of a run that
        # took three attempts is legible after the fact.
        "attempts": [],
    }


def read_manifest(directory: str | os.PathLike) -> dict | None:
    path = Path(directory) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def update_manifest(directory: str | os.PathLike, **fields) -> dict:
    """Shallow-merge `fields` into the manifest and rewrite it atomically.

    `attempts` is the one key that appends rather than replaces: pass
    `attempt={...}` to add an entry.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    man = read_manifest(directory) or {}
    attempt = fields.pop("attempt", None)
    man.update({k: v for k, v in fields.items() if v is not None or k in man})
    if attempt is not None:
        man.setdefault("attempts", []).append(attempt)
    man["heartbeat_utc"] = fields.get(
        "heartbeat_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    write_json(directory / MANIFEST_FILENAME, man)
    return man


# --- one writer per run directory -------------------------------------------

LOCK_FILENAME = ".writer.json"
#: A held lock older than this with no refresh is assumed to belong to a process
#: that no longer exists. Longer than any checkpoint interval by a wide margin.
LOCK_STALE_S = 2 * 3600


def acquire_writer_lock(
    directory: str | os.PathLike,
    owner: str,
    *,
    force: bool = False,
    now: datetime | None = None,
    stale_after_s: int = LOCK_STALE_S,
) -> dict:
    """Claim exclusive write access to a run directory, or raise `RunBusy`.

    `history.json` is rewritten whole on every eval boundary, so two writers in
    one directory do not merge -- the later write erases the earlier run's
    curve, and the checkpoints from both interleave under one `run.json`. This
    makes that collision an error instead of a silent corruption.

    Best-effort on a Modal Volume, where a commit from another container is not
    instantaneously visible; it is a guard against the mistake (resubmitting a
    run name that is already live), not a distributed mutex. The submission path
    additionally refuses to spawn a second job for a run whose manifest is live,
    which is the check that actually catches it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_FILENAME
    now = now or datetime.now(timezone.utc)
    payload = {
        "owner": owner,
        "acquired_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heartbeat_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = {}
        try:
            held = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        beat = _parse_utc(held.get("heartbeat_utc") or held.get("acquired_utc"))
        age = int((now - beat).total_seconds()) if beat else None
        stale = age is not None and age > stale_after_s
        if not (force or stale):
            raise RunBusy(
                f"{path} is held by {held.get('owner', 'an unknown writer')} "
                f"(last heartbeat {held.get('heartbeat_utc', 'unknown')}"
                f"{f', {age}s ago' if age is not None else ''}). Two writers in one run "
                "directory interleave their checkpoints and erase each other's history.json. "
                "Submit under a new --run-name, or pass --force-unlock if that writer is gone."
            )
        write_json(path, payload)
        return payload
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(payload, indent=2))
    return payload


def refresh_writer_lock(directory: str | os.PathLike, now: datetime | None = None) -> None:
    path = Path(directory) / LOCK_FILENAME
    try:
        held = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    held["heartbeat_utc"] = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(path, held)


def release_writer_lock(directory: str | os.PathLike) -> None:
    try:
        (Path(directory) / LOCK_FILENAME).unlink()
    except OSError:
        pass


# --- checkpoint selection ----------------------------------------------------

CHECKPOINT_RE = re.compile(r"^ckpt_(\d{6,})\.pkl$")


def checkpoint_paths(directory: str | os.PathLike) -> list[Path]:
    """Every checkpoint in a run directory, oldest first.

    Sorted by the iteration in the *filename*, not lexically and not by mtime:
    a Volume download does not preserve modification order, and iteration 1000
    sorts before iteration 200 as a string.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        m = CHECKPOINT_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def checkpoint_iteration(path: str | os.PathLike) -> int | None:
    m = CHECKPOINT_RE.match(Path(path).name)
    return int(m.group(1)) if m else None


def select_checkpoint(
    directory: str | os.PathLike, is_valid: Callable[[Path], bool] | None = None
) -> Path | None:
    """The most recent *valid* checkpoint, or None.

    Walks backwards from the newest rather than trusting it, because the newest
    is exactly the one a killed process was most likely writing when it died:
    the file exists, and it is a truncated pickle. `is_valid` defaults to a
    structural check (present, non-empty); `naigos.rl.checkpoint` passes one
    that actually unpickles the file and checks its schema, which is the real
    test and which needs numpy, so it does not live here.
    """
    def _structural(p: Path) -> bool:
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    check = is_valid or _structural
    for path in reversed(checkpoint_paths(directory)):
        try:
            if check(path):
                return path
        except Exception:  # pragma: no cover - a validator must not be able to
            continue      # hide every later checkpoint by raising on one file
    return None


# --- resume compatibility ----------------------------------------------------

#: Changing any of these changes what is being trained, so a checkpoint from
#: before the change is not a prefix of the run after it. Resuming across one
#: produces a curve that is a splice of two different experiments while
#: `run.json` continues to describe only the first.
RESUME_BLOCKING_KEYS = (
    "code.commit",
    "profile",
    "config.seed",
    "config.synthetic",
    "config.aoi",
    "config.use_cbf",
    "config.n_blue",
    "config.n_threat",
    "config.cell_m",
    "config.n_envs",
    "config.n_steps",
    "config.iterations",
    # Not a `run.json` field: the curricula live in code, so only a code change
    # can move them. Named here so it is a legal `--override-resume` key and so
    # `naigos.rl.checkpoint` and this module share one vocabulary.
    "curriculum",
)


def _dig(meta: dict, dotted: str):
    node = meta
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def resume_compatibility(existing: dict | None, requested: dict, *, overrides=()) -> dict:
    """Decide whether `requested` may continue the run described by `existing`.

    Returns `{"ok": bool, "blocking": [...], "overridden": [...], "changes": {...}}`.
    A blocking difference is refused rather than warned about: the failure it
    prevents is a `history.json` whose first half came from one configuration
    and whose second half came from another, which is indistinguishable from a
    real result once it is plotted.

    `overrides` names the specific keys the operator has decided to accept. It
    is per-key on purpose -- a blanket `--force` would also wave through the
    theatre change nobody meant to make.
    """
    overrides = set(overrides or ())
    unknown = sorted(overrides - set(RESUME_BLOCKING_KEYS))
    if unknown:
        raise ValueError(
            f"cannot override {unknown}; resume-blocking keys are {list(RESUME_BLOCKING_KEYS)}"
        )
    if existing is None:
        return {
            "ok": False,
            "blocking": ["no run.json: this directory does not record what it was running"],
            "overridden": [],
            "changes": {},
        }
    blocking, overridden, changes = [], [], {}
    for key in RESUME_BLOCKING_KEYS:
        old, new = _dig(existing, key), _dig(requested, key)
        if old == new:
            continue
        changes[key] = {"recorded": old, "requested": new}
        message = f"{key}: run was {old!r}, this resume asks for {new!r}"
        if key in overrides:
            overridden.append(message)
        else:
            blocking.append(message)
    return {
        "ok": not blocking,
        "blocking": blocking,
        "overridden": overridden,
        "changes": changes,
    }


# --- credential hygiene ------------------------------------------------------

# Modal token ids start `ak-` and secrets start `as-`; the viewer's map-tile
# credential is a JWT and starts `eyJ`. Anything matching is masked before it
# can reach a log file on a shared Volume, a manifest, or a terminal someone
# screenshots. The environment-variable pattern is written generically
# (`*_TOKEN`, `*_API_KEY`, `*_SECRET`) rather than by naming specific services,
# both because it should catch the next one too and because a test enforces that
# simulation code cannot so much as name a visual provider.
_SECRET_RE = re.compile(
    r"\b(?:ak|as)-[A-Za-z0-9_\-]{8,}\b|\beyJ[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-.]+\b"
)
_SECRET_ENV_RE = re.compile(
    r"((?:[A-Z][A-Z0-9_]*_)?(?:TOKEN(?:_ID|_SECRET)?|API_KEY|SECRET|PASSWORD)"
    r"\s*[=:]\s*)([\"']?)([A-Za-z0-9][A-Za-z0-9_.\-]{7,})"
)
# Documentation and tests are full of `TOKEN=<your token>` and `TOKEN=example`.
# Masking those would make the repository-wide scan noisy enough to be turned
# off, which is the only way this check actually fails.
_PLACEHOLDER = re.compile(
    r"your|example|placeholder|redacted|changeme|replace|dummy|fake|xxxx|token-here|"
    r"^generic$|^from-|^a-token$",
    re.IGNORECASE,
)
REDACTED = "[redacted]"


def _mask_env(match: re.Match) -> str:
    key, quote, value = match.group(1), match.group(2), match.group(3)
    if _PLACEHOLDER.search(value):
        return match.group(0)
    return f"{key}{quote}{REDACTED}"


def redact(text: str) -> str:
    """Mask credential-shaped substrings.

    Applied to everything the remote worker prints into `train.log` on the
    Volume and to every error string copied into the manifest. Nothing in this
    repository holds a credential, but a traceback from a client library can
    quote the environment it was handed, and a Volume is shared storage.

    The same function is the repository-wide scan (`tests/test_run_status.py`):
    a file whose redaction differs from itself contains something token-shaped.
    """
    if not isinstance(text, str):
        return text
    return _SECRET_ENV_RE.sub(_mask_env, _SECRET_RE.sub(REDACTED, text))
