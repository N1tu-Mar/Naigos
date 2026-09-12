"""Offline calibration sweep for the CBF safety backstop.

WHAT THIS IS FOR. `naigos/rl/cbf.py` ships a margin (1500 m) that nothing in the
repository measured: the documented trained-policy infeasibility rate is 0.227,
and a filter that cannot solve its QP a fifth of the time is not obviously
helping. The question "which margin buys survival without spending completion"
has three inputs -- the margin itself, how many threats are active, and how
capable red is -- so the answer is a surface, not a number, and it has to be
measured on the same scenarios at every point or the comparison is noise.

WHAT THIS IS NOT. Evaluation infrastructure only. Nothing here tunes a CBF
constant, touches the learned policy, or writes a recommendation into the code.
`cbf.py`, `train.py`, `red_team.py` and `naigos/env/**` are read and reused
verbatim -- the simulation, the detection model and the CBF mathematics live
there and are not reimplemented here.

WHY THE SEEDS ARE THE CENTRAL DESIGN POINT. Every cell of the grid is evaluated
on one explicitly recorded seed list, and the pairing that buys is strong
because of how the env happens to be built:

  * `threats.spawn` draws all `n_threat` positions and then masks with
    `active = arange(T) < n_threat_active`. Raising the density therefore ADDS
    threats to a scenario; it does not move the ones already there.
  * `RedCurriculum.apply` scales detection, lethality, latency and speed. It
    changes what the threats can do, not where they are.
  * The margin is a filter parameter. It does not enter `reset` at all, so two
    cells that differ only in margin are flown on byte-identical worlds.

So a margin column is an exactly paired comparison, and a density or level
column is the same terrain, the same start, the same objective and the same
threat placement under a different threat set or a different red. Each cell is
also flown twice -- with and without the filter -- on those same seeds, so the
"did the backstop help" question is answered against the right control instead
of against a remembered number from another run.

The result is a versioned JSON payload (`SCHEMA_VERSION`) carrying the grid, the
seed list, the complete CBF and environment configuration, checkpoint identity
(path, size, sha256, iteration, the env_cfg it was trained under), git commit
and dirty flag, and a timestamp. `serialize` is stable under re-running the same
sweep; `canonical_json` drops the two deliberately volatile keys so that claim
can be asserted in a test.

    from naigos.rl.cbf_sweep import SweepGrid, run_sweep, markdown_summary

    payload = run_sweep(
        grid=SweepGrid(margins=(0.0, 1500.0, 3000.0)),
        checkpoint="checkpoints/theatre_1000.pkl",
        n_seeds=16,
    )
    print(markdown_summary(payload))

See `docs/cbf-sweep.md` and `scripts/sweep_cbf.py`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..env.config import EnvConfig
from ..env.flight_env import NaigosEnv
from . import checkpoint as ckpt_mod
from . import runmeta
from .cbf import CBFConfig, make_policy_filter
from .ppo import greedy_policy
from .red_team import RedCurriculum
from .train import rollout_metrics

#: Bumped when the shape of the payload changes in a way a reader must notice.
SCHEMA_VERSION = 1

#: Keys whose values legitimately differ between two runs of the same sweep.
#: `canonical_json` removes them; everything else must be reproducible.
VOLATILE_KEYS = ("created_utc", "runtime")

#: Every per-cell metric, in report order. Named here rather than inferred from
#: whatever `rollout_metrics` happened to return, so a column silently
#: disappearing from the upstream dict is a test failure and not a quiet hole.
METRIC_KEYS: tuple[str, ...] = (
    "survival_rate",
    "objective_rate",
    "shootdown_rate",
    "terrain_rate",
    "bounds_rate",
    "timeout_rate",
    "fuel_loss_rate",
    "ceiling_rate",
    "stall_rate",
    "mean_exposure",
    "exposure_early",
    "exposure_successful",
    "cumulative_exposure_per_sortie",
    "mean_lock",
    "mean_min_agl",
    "mean_agl_live",
    "cbf_infeasible_rate",
)

#: The arms flown in every cell. `cbf` is the filtered policy; `no_cbf` is the
#: same policy on the same seeds with `action_filter=None`.
ARMS: tuple[str, ...] = ("cbf", "no_cbf")

#: Deltas reported per cell: cbf minus no_cbf. Positive is better for the first
#: two and worse for the rest -- `markdown_summary` says so in the legend.
DELTA_KEYS: tuple[str, ...] = (
    "survival_rate",
    "objective_rate",
    "shootdown_rate",
    "terrain_rate",
    "bounds_rate",
    "cumulative_exposure_per_sortie",
)


class SweepConfigError(ValueError):
    """A grid, seed plan or environment configuration that cannot be swept.

    Raised eagerly, before anything is compiled or flown, so an eight-hour sweep
    does not fail on its last cell.
    """


# --- the grid ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class GridCell:
    """One point of the sweep. Ordered by margin first -- reports sort by it."""

    margin: float
    n_threat_active: int
    red_level: float

    @property
    def key(self) -> str:
        return f"margin={self.margin:.1f}|threats={self.n_threat_active:d}|red={self.red_level:.3f}"

    def as_dict(self) -> dict:
        return {
            "margin": float(self.margin),
            "n_threat_active": int(self.n_threat_active),
            "red_level": float(self.red_level),
            "key": self.key,
        }


@dataclasses.dataclass(frozen=True)
class SweepGrid:
    """The cartesian product of the three axes, deduplicated and sorted.

    Defaults are deliberately small and CPU-friendly: 2x2x2 is eight cells and
    sixteen rollout batches, which is a sweep you can run on a laptop while
    deciding whether the big one is worth it.
    """

    margins: tuple[float, ...] = (0.0, 1_500.0)
    threat_counts: tuple[int, ...] = (6, 12)
    red_levels: tuple[float, ...] = (0.2, 0.6)

    def __post_init__(self):
        # tolerate lists from argparse/JSON without making the dataclass mutable
        object.__setattr__(self, "margins", tuple(float(m) for m in self.margins))
        object.__setattr__(self, "threat_counts", tuple(int(n) for n in self.threat_counts))
        object.__setattr__(self, "red_levels", tuple(float(r) for r in self.red_levels))

    def validate(self, cfg: EnvConfig | None = None) -> None:
        """Every reason this grid cannot be flown, as one explicit error.

        `cfg` is the base environment config; when given, threat counts are
        checked against its padding capacity, because `n_threat_active` above
        `n_threat` silently clamps inside `RedCurriculum.apply` and a sweep that
        reports two different densities that were in fact the same density is
        worse than one that refuses to start.
        """
        problems: list[str] = []
        if not self.margins:
            problems.append("margins is empty")
        if not self.threat_counts:
            problems.append("threat_counts is empty")
        if not self.red_levels:
            problems.append("red_levels is empty")

        for m in self.margins:
            if not math.isfinite(m):
                problems.append(f"margin {m!r} is not finite")
            elif m < 0.0:
                problems.append(f"margin {m!r} is negative; an envelope cannot be deflated")
        for n in self.threat_counts:
            if n < 0:
                problems.append(f"threat count {n!r} is negative")
            elif cfg is not None and n > cfg.n_threat:
                problems.append(
                    f"threat count {n} exceeds the env's padding capacity n_threat={cfg.n_threat}; "
                    "raise n_threat or lower the count"
                )
        for r in self.red_levels:
            if not math.isfinite(r):
                problems.append(f"red level {r!r} is not finite")
            elif not (0.0 <= r <= 1.0):
                problems.append(f"red level {r!r} is outside [0, 1]")

        for axis, values in (
            ("margins", self.margins),
            ("threat_counts", self.threat_counts),
            ("red_levels", self.red_levels),
        ):
            if len(set(values)) != len(values):
                problems.append(f"{axis} contains duplicates: {values!r}")

        if problems:
            raise SweepConfigError("invalid sweep grid: " + "; ".join(problems))

    def cells(self) -> tuple[GridCell, ...]:
        """Expanded product, sorted by (margin, threats, level).

        Sorted rather than nested-loop order so the JSON, the table and the
        Markdown all agree without any of them re-sorting, and so two sweeps
        that named the same axes in a different order serialize identically.
        """
        self.validate()
        cells = [
            GridCell(margin=m, n_threat_active=n, red_level=r)
            for m in self.margins
            for n in self.threat_counts
            for r in self.red_levels
        ]
        return tuple(sorted(cells))

    def as_dict(self) -> dict:
        return {
            "margins": list(self.margins),
            "threat_counts": list(self.threat_counts),
            "red_levels": list(self.red_levels),
            "n_cells": len(self.cells()),
        }


# --- the seed plan -----------------------------------------------------------


def seed_plan(base_seed: int = 20_000, n_seeds: int = 16) -> tuple[int, ...]:
    """The held-out scenario seeds, as a contiguous block from `base_seed`.

    Contiguous and explicit on purpose. The alternative -- splitting one PRNG key
    n ways inside the rollout -- produces seeds that cannot be written down,
    cannot be quoted in a bug report and cannot be re-run one at a time. Every
    cell and both arms consume this exact list, so the whole sweep is paired.

    `base_seed` defaults well away from the training seeds so "held out" means
    something; the caller is responsible for actually keeping it that way and the
    value is recorded in the payload either way.
    """
    if n_seeds < 1:
        raise SweepConfigError(f"n_seeds must be >= 1, got {n_seeds}")
    if base_seed < 0:
        raise SweepConfigError(f"base_seed must be >= 0, got {base_seed}")
    return tuple(int(base_seed) + i for i in range(int(n_seeds)))


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Coerce and check an explicitly supplied seed list."""
    if not seeds:
        raise SweepConfigError("the seed plan is empty; a sweep with no scenarios measures nothing")
    out = []
    for s in seeds:
        si = int(s)
        if si < 0:
            raise SweepConfigError(f"seed {s!r} is negative")
        out.append(si)
    if len(set(out)) != len(out):
        raise SweepConfigError(f"the seed plan contains duplicates: {out!r}")
    return tuple(out)


def seed_keys(seeds: Sequence[int]) -> jax.Array:
    """(n_seeds, 2) stacked PRNG keys -- one world per seed, vmapped over."""
    return jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])


# --- configuration per cell --------------------------------------------------


def cell_env_config(base: EnvConfig, cell: GridCell, curriculum: RedCurriculum) -> EnvConfig:
    """The env this cell is flown on.

    ORDER MATTERS: the curriculum is applied first and the density is written
    afterwards. `RedCurriculum.apply` sets `n_threat_active` from the level, so
    applying it second would silently overwrite the density axis with a function
    of the red axis and collapse the grid to a diagonal.
    """
    if cell.n_threat_active > base.n_threat:
        raise SweepConfigError(
            f"cell {cell.key} wants {cell.n_threat_active} active threats but the env pads to "
            f"n_threat={base.n_threat}"
        )
    return curriculum.apply(base, cell.red_level).replace(n_threat_active=int(cell.n_threat_active))


def cell_cbf_config(base: CBFConfig, cell: GridCell) -> CBFConfig:
    """The filter this cell is flown with. Only the margin moves."""
    return dataclasses.replace(base, margin=float(cell.margin))


# --- measurement -------------------------------------------------------------


def _extra_metrics(final, traj, n_worlds: int, n_blue: int) -> dict:
    """The death causes `rollout_metrics` does not carry.

    `out_of_fuel` is NOT a one-shot event: running dry does not kill the
    aircraft (see `flight_env.step`, where `alive` excludes only shot, terrain
    and bounds), so the flag stays raised on every subsequent unfrozen step.
    Summing it over time would report a "rate" of several hundred percent. The
    honest statistic is the fraction of sorties that ran dry at any point.
    """
    denom = max(n_worlds * n_blue, 1)

    def any_step(name: str) -> float:
        flag = np.asarray(getattr(traj["terms"], name))  # (W, S, B)
        return float((flag.max(axis=1) > 0).sum() / denom)

    return {
        "fuel_loss_rate": any_step("out_of_fuel"),
        "ceiling_rate": any_step("ceiling_violation"),
        "stall_rate": any_step("stall_violation"),
    }


def evaluate_arm(
    env: NaigosEnv,
    policy,
    keys: jax.Array,
    *,
    action_filter=None,
    n_steps: int | None = None,
) -> dict:
    """Fly one arm over the seed batch and reduce it to the reported metrics.

    `rollout_metrics` from `naigos.rl.train` does the arithmetic, verbatim, for
    exactly the reason that function documents: a sweep that computed survival
    with its own formula would be comparing formulas with the training history
    rather than comparing policies.
    """
    n_worlds = int(keys.shape[0])
    rolled = jax.jit(
        jax.vmap(lambda k: env.rollout(k, policy, n_steps=n_steps, action_filter=action_filter))
    )(keys)
    final, traj = rolled
    metrics = dict(rollout_metrics(final, traj, n_worlds, env.cfg.n_blue))
    metrics.update(_extra_metrics(final, traj, n_worlds, env.cfg.n_blue))
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:  # pragma: no cover - guards an upstream rename, not a branch
        raise SweepConfigError(f"rollout metrics are missing {missing!r}; METRIC_KEYS is stale")
    return {k: float(metrics[k]) for k in METRIC_KEYS}


def evaluate_cell(
    cell: GridCell,
    *,
    base_env_cfg: EnvConfig,
    base_cbf_cfg: CBFConfig,
    curriculum: RedCurriculum,
    make_policy: Callable[[EnvConfig], Callable],
    keys: jax.Array,
    hmap=None,
    n_steps: int | None = None,
    include_no_cbf: bool = True,
) -> dict:
    """One grid cell: the filtered arm, the unfiltered control, and the delta.

    Both arms get the same `keys`, the same env config and the same policy, so
    the only difference between them is whether `make_policy_filter` sits in the
    loop. `no_cbf` is the baseline the margin question actually needs; comparing
    a filtered run against a number remembered from a different run is how a
    filter gets credit for a lucky seed.
    """
    env_cfg = cell_env_config(base_env_cfg, cell, curriculum)
    cbf_cfg = cell_cbf_config(base_cbf_cfg, cell)
    env = NaigosEnv(env_cfg, hmap=hmap)
    policy = make_policy(env_cfg)

    arms = {
        "cbf": evaluate_arm(
            env, policy, keys, action_filter=make_policy_filter(cbf_cfg, env_cfg), n_steps=n_steps
        )
    }
    if include_no_cbf:
        # The unfiltered arm reports cbf_infeasible_rate 0.0 by construction:
        # `rollout` fills `cbf_feasible` with True when there is no filter.
        # Kept in the row rather than blanked so the two arms have one schema.
        arms["no_cbf"] = evaluate_arm(env, policy, keys, action_filter=None, n_steps=n_steps)

    row = {
        "cell": cell.as_dict(),
        "env_config": _config_dict(env_cfg),
        "cbf_config": _config_dict(cbf_cfg),
        "arms": arms,
    }
    if include_no_cbf:
        row["delta"] = {k: arms["cbf"][k] - arms["no_cbf"][k] for k in DELTA_KEYS}
    return row


# --- provenance --------------------------------------------------------------


def _config_dict(obj) -> dict:
    """A frozen config as plain JSON. Tuples become lists; nothing is dropped."""
    return json.loads(json.dumps(dataclasses.asdict(obj), default=str, sort_keys=True))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_config_from_blob(blob: dict, **overrides) -> EnvConfig:
    """Rebuild the `EnvConfig` a checkpoint's actor was trained under.

    The actor's input widths are functions of the config -- the ego vector is 10
    or 14 wide depending on `obs_edge_features`, and each threat slot is
    `9 + len(threat_kinds)` wide. Evaluating a theatre checkpoint against a
    default synthetic `EnvConfig` is therefore not a worse measurement, it is a
    shape error at best and a meaningless one at worst. Rebuilding from the
    blob's own `env_cfg` is the only way a sweep can honestly claim to have
    evaluated THAT checkpoint.

    Fields the stored dict does not carry keep their current defaults, and
    `overrides` (n_blue, max_steps, n_threat, ...) are applied last.
    """
    from ..env.config import (
        AirframeConfig,
        DetectionConfig,
        SpatialHashConfig,
        TerrainConfig,
        ThreatKindConfig,
    )

    stored = dict(blob.get("env_cfg") or {})
    nested = {
        "airframe": AirframeConfig,
        "detection": DetectionConfig,
        "terrain": TerrainConfig,
        "hash": SpatialHashConfig,
    }
    kw: dict = {}
    for name, klass in nested.items():
        sub = stored.pop(name, None)
        if isinstance(sub, dict):
            fields = {f.name for f in dataclasses.fields(klass)}
            kw[name] = klass(**{k: v for k, v in sub.items() if k in fields})

    kinds = stored.pop("threat_kinds", None)
    if kinds:
        fields = {f.name for f in dataclasses.fields(ThreatKindConfig)}
        kw["threat_kinds"] = tuple(
            ThreatKindConfig(**{k: v for k, v in spec.items() if k in fields}) for spec in kinds
        )

    top = {f.name for f in dataclasses.fields(EnvConfig)}
    kw.update({k: v for k, v in stored.items() if k in top})
    kw.update(overrides)
    return EnvConfig(**kw)


def require_actor_compatible(actor_params, cfg: EnvConfig, source: str) -> None:
    """Refuse an actor whose observation widths are not the ones `cfg` builds.

    Two checks, because there are two ways to get this wrong and only one of
    them is already covered upstream. The ego width is
    `checkpoint.require_actor_ego_dim`'s job. The threat-slot width is not
    checked anywhere, and it is the one that bites when a theatre checkpoint
    (five threat classes) meets a default config (three).
    """
    try:
        ckpt_mod.require_actor_ego_dim(actor_params, cfg, source)
    except SystemExit as e:
        raise SweepConfigError(str(e)) from e

    enc = actor_params["params"]["threat_encoder"]["Dense_0"]["kernel"]
    got = int(np.shape(enc)[0])
    if got != cfg.threat_feat_dim:
        raise SweepConfigError(
            f"{source}: the actor takes a {got}-wide threat slot but this env builds "
            f"{cfg.threat_feat_dim} (= 9 + {cfg.n_threat_kinds} threat kinds). The sweep would be "
            "measuring a shape mismatch, not a margin. Rebuild the env config from the "
            "checkpoint (see env_config_from_blob) or point at a matching checkpoint."
        )


def checkpoint_provenance(path: str | os.PathLike, blob: dict) -> dict:
    """Everything needed to say which artefact produced these numbers."""
    p = Path(path)
    return {
        "path": str(p),
        "resolved_path": str(p.resolve()),
        "sha256": _sha256(p),
        "bytes": p.stat().st_size,
        "iteration": blob.get("iter"),
        "schema": blob.get("schema"),
        "red_level_at_checkpoint": blob.get("red_level"),
        "trained_env_cfg": json.loads(json.dumps(blob.get("env_cfg"), default=str, sort_keys=True)),
        "obs_config": ckpt_mod.obs_config_from_blob(blob),
    }


# --- the sweep ---------------------------------------------------------------


def run_sweep(
    *,
    grid: SweepGrid | None = None,
    checkpoint: str | os.PathLike | None = None,
    make_policy: Callable[[EnvConfig], Callable] | None = None,
    policy_provenance: dict | None = None,
    base_env_cfg: EnvConfig | None = None,
    base_cbf_cfg: CBFConfig | None = None,
    curriculum: RedCurriculum | None = None,
    seeds: Sequence[int] | None = None,
    base_seed: int = 20_000,
    n_seeds: int = 16,
    n_steps: int | None = None,
    hmap=None,
    include_no_cbf: bool = True,
    env_overrides: dict | None = None,
    notes: str | None = None,
    repo: str | os.PathLike | None = None,
    now: datetime | None = None,
    on_cell: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Fly the whole grid and return the versioned payload.

    Exactly one policy source: either `checkpoint` (the real thing -- the env
    config is rebuilt from the checkpoint unless `base_env_cfg` is given, and
    the actor's input widths are checked against it) or `make_policy` (a
    `(EnvConfig) -> policy` factory, for tests and for hand-written reference
    controllers). `policy_provenance` is required with `make_policy`, because a
    result whose policy cannot be identified is not a result.

    `on_cell(index, total, row)` is called after each cell so a CLI can print
    progress without this function knowing what a terminal is.
    """
    grid = grid if grid is not None else SweepGrid()
    base_cbf_cfg = base_cbf_cfg if base_cbf_cfg is not None else CBFConfig()
    curriculum = curriculum if curriculum is not None else RedCurriculum()
    env_overrides = dict(env_overrides or {})

    if (checkpoint is None) == (make_policy is None):
        raise SweepConfigError(
            "pass exactly one of `checkpoint` (evaluate a trained actor) or `make_policy` "
            "(a (EnvConfig) -> policy factory)"
        )

    seeds = validate_seeds(seeds) if seeds is not None else seed_plan(base_seed, n_seeds)

    blob = None
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise SweepConfigError(f"checkpoint not found: {path}")
        blob = ckpt_mod.load(path)
        if base_env_cfg is None:
            base_env_cfg = env_config_from_blob(blob, **env_overrides)
        elif env_overrides:
            base_env_cfg = base_env_cfg.replace(**env_overrides)
        require_actor_compatible(blob["actor"], base_env_cfg, str(path))
        actor = blob["actor"]
        make_policy = lambda cfg: greedy_policy(actor, cfg)  # noqa: E731
        policy_provenance = {"kind": "checkpoint", **checkpoint_provenance(path, blob)}
    else:
        if policy_provenance is None:
            raise SweepConfigError(
                "make_policy requires policy_provenance: a sweep whose policy cannot be "
                "identified cannot be reproduced"
            )
        base_env_cfg = base_env_cfg if base_env_cfg is not None else EnvConfig()
        if env_overrides:
            base_env_cfg = base_env_cfg.replace(**env_overrides)
        policy_provenance = {"kind": "callable", **policy_provenance}

    grid.validate(base_env_cfg)
    cells = grid.cells()
    keys = seed_keys(seeds)
    started = time.time()

    rows = []
    for i, cell in enumerate(cells):
        row = evaluate_cell(
            cell,
            base_env_cfg=base_env_cfg,
            base_cbf_cfg=base_cbf_cfg,
            curriculum=curriculum,
            make_policy=make_policy,
            keys=keys,
            hmap=hmap,
            n_steps=n_steps,
            include_no_cbf=include_no_cbf,
        )
        rows.append(row)
        if on_cell is not None:
            on_cell(i + 1, len(cells), row)

    now = now or datetime.now(timezone.utc)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "cbf_calibration_sweep",
        "created_utc": now.replace(microsecond=0).isoformat(),
        "notes": notes,
        "grid": grid.as_dict(),
        "seeds": {
            "values": list(seeds),
            "n_seeds": len(seeds),
            "base_seed": int(seeds[0]),
            "contiguous": list(seeds) == list(range(seeds[0], seeds[0] + len(seeds))),
            "shared_across_cells": True,
            "shared_across_arms": True,
        },
        "policy": policy_provenance,
        "base_env_config": _config_dict(base_env_cfg),
        "base_cbf_config": _config_dict(base_cbf_cfg),
        "red_curriculum": _config_dict(curriculum),
        "rollout": {
            "n_steps": int(n_steps) if n_steps is not None else int(base_env_cfg.max_steps),
            "n_worlds": len(seeds),
            "n_blue": int(base_env_cfg.n_blue),
            "sorties_per_arm": len(seeds) * int(base_env_cfg.n_blue),
            "fixed_hmap": hmap is not None,
            "arms": list(ARMS) if include_no_cbf else ["cbf"],
        },
        "metric_keys": list(METRIC_KEYS),
        "code": runmeta.git_info(repo if repo is not None else _repo_root()),
        "cells": rows,
        "runtime": {"wall_s": round(time.time() - started, 3)},
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --- serialization -----------------------------------------------------------


def serialize(payload: dict) -> str:
    """The on-disk form: sorted keys, two-space indent, trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def canonical_json(payload: dict) -> str:
    """`serialize` minus the keys that are allowed to differ between runs.

    Two sweeps of the same grid, seeds, checkpoint and commit must produce the
    same string here. That is the property a reader relies on when they diff two
    result files to see what actually changed, and it is asserted in the tests.
    """
    return serialize({k: v for k, v in payload.items() if k not in VOLATILE_KEYS})


def write_result(path: str | os.PathLike, payload: dict) -> Path:
    """Atomic-ish write, same discipline as `runmeta.write_json`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(serialize(payload))
    os.replace(tmp, p)
    return p


def load_result(path: str | os.PathLike) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA_VERSION:
        raise SweepConfigError(
            f"{path}: schema {payload.get('schema')!r}, this code reads {SCHEMA_VERSION}"
        )
    return payload


# --- reporting ---------------------------------------------------------------

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("margin", "margin m"),
    ("n_threat_active", "threats"),
    ("red_level", "red"),
    ("survival_rate", "surv"),
    ("objective_rate", "obj"),
    ("shootdown_rate", "shot"),
    ("terrain_rate", "terr"),
    ("bounds_rate", "oob"),
    ("timeout_rate", "t/o"),
    ("fuel_loss_rate", "fuel"),
    ("cumulative_exposure_per_sortie", "expo"),
    ("cbf_infeasible_rate", "infeas"),
)


def _cell_values(row: dict, arm: str) -> dict:
    c = row["cell"]
    m = row["arms"][arm]
    return {
        "margin": f"{c['margin']:.0f}",
        "n_threat_active": f"{c['n_threat_active']:d}",
        "red_level": f"{c['red_level']:.2f}",
        **{k: f"{v:.3f}" for k, v in m.items()},
    }


def _rows_for_report(payload: dict, arm: str) -> list[dict]:
    """Report rows, sorted by margin then density then red level.

    `SweepGrid.cells` already emits that order and `run_sweep` preserves it, but
    a payload can be hand-edited or merged, so sort again rather than trust it.
    """
    rows = [r for r in payload.get("cells", []) if arm in r.get("arms", {})]
    rows.sort(key=lambda r: (r["cell"]["margin"], r["cell"]["n_threat_active"], r["cell"]["red_level"]))
    return [_cell_values(r, arm) for r in rows]


def format_table(payload: dict, arm: str = "cbf") -> str:
    """Fixed-width table for a terminal, sorted by margin."""
    rows = _rows_for_report(payload, arm)
    heads = [h for _, h in _COLUMNS]
    widths = [
        max(len(h), *(len(r.get(k, "-")) for r in rows)) if rows else len(h)
        for (k, _), h in zip(_COLUMNS, heads)
    ]
    line = lambda vals: "  ".join(v.rjust(w) for v, w in zip(vals, widths))  # noqa: E731
    out = [line(heads), line(["-" * w for w in widths])]
    out += [line([r.get(k, "-") for k, _ in _COLUMNS]) for r in rows]
    return "\n".join(out)


def markdown_summary(payload: dict, arm: str = "cbf") -> str:
    """Markdown table plus the provenance a reader needs to trust the numbers.

    The legend is not decoration. `infeas` is the statistic this whole exercise
    exists to move, and a table that prints it without saying that a high value
    means the backstop was not solving its QP invites exactly the misreading
    `cbf.py`'s honest caveat warns about.
    """
    rows = _rows_for_report(payload, arm)
    heads = [h for _, h in _COLUMNS]
    lines = [
        f"# CBF calibration sweep (`{arm}` arm)",
        "",
        f"- schema `{payload.get('schema')}`, generated `{payload.get('created_utc')}`",
        f"- policy: `{(payload.get('policy') or {}).get('kind')}` "
        f"`{(payload.get('policy') or {}).get('path', (payload.get('policy') or {}).get('label', '-'))}`",
        f"- commit: `{(payload.get('code') or {}).get('commit')}` "
        f"(dirty: `{(payload.get('code') or {}).get('dirty')}`)",
        f"- seeds: `{(payload.get('seeds') or {}).get('n_seeds')}` held-out, identical in every cell "
        f"and both arms, base `{(payload.get('seeds') or {}).get('base_seed')}`",
        f"- rollout: `{(payload.get('rollout') or {}).get('sorties_per_arm')}` sorties per arm, "
        f"`{(payload.get('rollout') or {}).get('n_steps')}` steps",
        "",
        "| " + " | ".join(heads) + " |",
        "|" + "|".join("---:" for _ in heads) + "|",
    ]
    lines += ["| " + " | ".join(r.get(k, "-") for k, _ in _COLUMNS) + " |" for r in rows]
    lines += [
        "",
        "Sorted by margin. `surv`/`obj` higher is better; `shot`/`terr`/`oob`/`t/o`/`fuel`/`expo` "
        "lower is better. `infeas` is the fraction of live aircraft-steps where the CBF-QP had no "
        "solution -- the filter guarantees nothing on those steps, so a margin that buys survival "
        "while driving `infeas` up has bought it somewhere other than the barrier.",
    ]

    if any("delta" in r for r in payload.get("cells", [])):
        lines += ["", "## CBF minus no-CBF, same seeds", ""]
        dkeys = list(DELTA_KEYS)
        lines += [
            "| margin m | threats | red | " + " | ".join(dkeys) + " |",
            "|---:" * (3 + len(dkeys)) + "|",
        ]
        for r in sorted(
            (r for r in payload.get("cells", []) if "delta" in r),
            key=lambda r: (r["cell"]["margin"], r["cell"]["n_threat_active"], r["cell"]["red_level"]),
        ):
            c = r["cell"]
            vals = [f"{r['delta'][k]:+.3f}" for k in dkeys]
            lines.append(
                f"| {c['margin']:.0f} | {c['n_threat_active']:d} | {c['red_level']:.2f} | "
                + " | ".join(vals)
                + " |"
            )
    return "\n".join(lines) + "\n"
