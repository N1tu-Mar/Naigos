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
    return {
        "run_name": (meta or {}).get("run_name", directory.name),
        "profile": (meta or {}).get("profile"),
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
