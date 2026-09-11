"""Pipeline configuration: a checked-in default plus a versioned copy on the Volume.

A scheduled Modal Function has no command line and no laptop to read a file
from, so its configuration comes from exactly two places:

  * ``naigos/pipeline/default_config.json`` -- checked in, baked into the image,
    immutable for the life of a deployment. It is the only place the **cadence**
    lives, because Modal fixes a ``modal.Cron`` at deploy time: changing the
    schedule means editing this file and running ``modal deploy`` again.
  * ``/pipeline/config/current.json`` on the Volume -- everything else (pause,
    gates, seeds, profiles, ``auto_promote``), versioned. Every published version
    is also kept write-once under ``config/history/``, and a new version must name
    the version it replaces, so two operators cannot silently overwrite each
    other. A Volume config that mentions ``schedule`` is refused: a schedule
    written there would read as in force while the deployed crons ignored it.

Parsing is strict. Unknown keys, wrong types, out-of-range numbers, an unknown
profile or AOI, a non-allowlisted refresh source and held-out evaluation seeds
that overlap a training seed are all errors, and a coordinator that cannot parse
its configuration does no costly work (``load_effective`` raises; the Modal
wrapper records the failure and stops).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..research.allowlist import ALLOWLIST
from ..research.aoi import AOIS
from ..rl import runmeta
from . import layout
from .schedule import CronError, parse_cron

CONFIG_SCHEMA = 1
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")

#: Fields that describe the *operation* of a config version rather than what it
#: asks the pipeline to compute. Excluded from digests so pausing, resuming or
#: annotating a config does not change any idempotency key.
OPERATIONAL_KEYS = ("version", "paused", "pause_reason", "updated_utc", "updated_by", "note",
                    "schedule")

_TOP = {
    "schema", "version", "paused", "pause_reason", "auto_promote", "aoi", "schedule",
    "snapshot", "training", "evaluation", "gates", "retention",
    "updated_utc", "updated_by", "note",
}
_TOP_REQUIRED = _TOP - {"schedule", "updated_utc", "updated_by", "note", "pause_reason"}
CHAMPION_TOLERANCE_KEYS = ("exposure_early", "shootdown_rate", "terrain_rate", "bounds_rate",
                           "survival_rate", "objective_rate")
BASELINE_TOLERANCE_KEYS = ("survival_rate", "objective_rate")
#: Sources a snapshot may re-pull. Derived from the allowlist, so a source that
#: is not fetchable (no hosts) can never be asked for.
REFRESHABLE_SOURCES = tuple(sorted(k for k, s in ALLOWLIST.items() if s.hosts))


class ConfigError(ValueError):
    """A configuration the pipeline refuses to act on."""


class ConfigConflict(RuntimeError):
    """A new config version was based on a version that is no longer current."""


# --- strict field checks -----------------------------------------------------


def _keys(where: str, obj, allowed: set, required: set | None = None) -> dict:
    if not isinstance(obj, dict):
        raise ConfigError(f"{where}: expected an object, got {type(obj).__name__}")
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    missing = sorted((required if required is not None else allowed) - set(obj))
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")
    return obj


def _bool(where: str, v) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{where}: expected true/false, got {v!r}")
    return v


def _int(where: str, v, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ConfigError(f"{where}: expected an integer in [{lo}, {hi}], got {v!r}")
    return v


def _num(where: str, v, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ConfigError(f"{where}: expected a number in [{lo}, {hi}], got {v!r}")
    return float(v)


def _str_or_none(where: str, v, max_len: int = 500) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or len(v) > max_len:
        raise ConfigError(f"{where}: expected a string of at most {max_len} chars")
    return runmeta.redact(v)


def _training_kind(where: str, obj, *, weekly: bool) -> dict:
    allowed = {"profile", "overrides", "seed"} | ({"enabled"} if weekly else set())
    _keys(where, obj, allowed)
    if obj["profile"] not in runmeta.PROFILES:
        raise ConfigError(f"{where}.profile: {obj['profile']!r} is not one of {sorted(runmeta.PROFILES)}")
    if not isinstance(obj["overrides"], dict):
        raise ConfigError(f"{where}.overrides: expected an object")
    try:
        runmeta.resolve_profile(obj["profile"], **obj["overrides"])
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}.overrides: {e}") from None
    _int(f"{where}.seed", obj["seed"], 0, 2**31 - 1)
    if weekly:
        _bool(f"{where}.enabled", obj["enabled"])
    return obj


def validate(doc: dict, *, source: str) -> dict:
    """Return a validated deep copy of ``doc``, or raise ``ConfigError``.

    ``source`` is ``"default"`` (the checked-in file, which must carry the
    schedule) or ``"volume"`` (which must not).
    """
    if source not in ("default", "volume"):
        raise ValueError(f"unknown config source {source!r}")
    doc = copy.deepcopy(doc)
    required = set(_TOP_REQUIRED) | ({"schedule"} if source == "default" else set())
    _keys("config", doc, _TOP, required)
    if source == "volume" and "schedule" in doc:
        raise ConfigError(
            "config.schedule: the schedule is deployment-defined. Edit "
            "naigos/pipeline/default_config.json and run `modal deploy "
            "naigos/rl/modal_pipeline.py`; a schedule in the Volume config would "
            "read as in force while the deployed crons ignored it."
        )
    if doc["schema"] != CONFIG_SCHEMA:
        raise ConfigError(f"config.schema: expected {CONFIG_SCHEMA}, got {doc['schema']!r}")
    _int("config.version", doc["version"], 1, 10**9)
    _bool("config.paused", doc["paused"])
    doc["pause_reason"] = _str_or_none("config.pause_reason", doc.get("pause_reason"))
    _bool("config.auto_promote", doc["auto_promote"])
    if doc["aoi"] not in AOIS:
        raise ConfigError(f"config.aoi: {doc['aoi']!r} is not one of {sorted(AOIS)}")
    for k in ("updated_utc", "updated_by", "note"):
        if k in doc:
            doc[k] = _str_or_none(f"config.{k}", doc[k])

    if "schedule" in doc:
        sch = _keys("config.schedule", doc["schedule"],
                    {"snapshot_cron", "nightly_cron", "weekly_cron", "timezone"})
        if sch["timezone"] != "UTC":
            raise ConfigError("config.schedule.timezone: only UTC is supported; times are documented in UTC")
        for k in ("snapshot_cron", "nightly_cron", "weekly_cron"):
            try:
                parse_cron(sch[k])
            except CronError as e:
                raise ConfigError(f"config.schedule.{k}: {e}") from None

    snap = _keys("config.snapshot", doc["snapshot"],
                 {"refresh_sources", "skip_flights", "flight_snapshots", "timeout_s"})
    if not isinstance(snap["refresh_sources"], list) or len(set(snap["refresh_sources"])) != len(snap["refresh_sources"]):
        raise ConfigError("config.snapshot.refresh_sources: expected a list without duplicates")
    bad = sorted(set(snap["refresh_sources"]) - set(REFRESHABLE_SOURCES))
    if bad:
        raise ConfigError(
            f"config.snapshot.refresh_sources: {bad} are not fetchable allowlisted sources "
            f"({list(REFRESHABLE_SOURCES)}). The allowlist is fixed; it is not extended by config."
        )
    _bool("config.snapshot.skip_flights", snap["skip_flights"])
    _int("config.snapshot.flight_snapshots", snap["flight_snapshots"], 1, 200)
    _int("config.snapshot.timeout_s", snap["timeout_s"], 60, 6 * 3600)

    tr = _keys("config.training", doc["training"],
               {"nightly", "weekly", "use_cbf", "require_smoke", "max_candidates_per_day"})
    _training_kind("config.training.nightly", tr["nightly"], weekly=False)
    _training_kind("config.training.weekly", tr["weekly"], weekly=True)
    _bool("config.training.use_cbf", tr["use_cbf"])
    _bool("config.training.require_smoke", tr["require_smoke"])
    _int("config.training.max_candidates_per_day", tr["max_candidates_per_day"], 0, 4)

    ev = _keys("config.evaluation", doc["evaluation"],
               {"heldout_seeds", "worlds_per_seed", "weekly_worlds_per_seed", "red_level",
                "verifier_episodes", "use_cbf"})
    seeds = ev["heldout_seeds"]
    if not isinstance(seeds, list) or not seeds or len(seeds) > 256:
        raise ConfigError("config.evaluation.heldout_seeds: expected 1-256 integers")
    for i, s in enumerate(seeds):
        _int(f"config.evaluation.heldout_seeds[{i}]", s, 0, 2**31 - 1)
    if len(set(seeds)) != len(seeds):
        raise ConfigError("config.evaluation.heldout_seeds: duplicate seeds")
    _int("config.evaluation.worlds_per_seed", ev["worlds_per_seed"], 1, 1024)
    _int("config.evaluation.weekly_worlds_per_seed", ev["weekly_worlds_per_seed"], 1, 1024)
    _num("config.evaluation.red_level", ev["red_level"], 0.0, 1.0)
    # At least one: the verifier is a gate, and a gate with no sample is a waiver.
    _int("config.evaluation.verifier_episodes", ev["verifier_episodes"], 1, 64)
    _bool("config.evaluation.use_cbf", ev["use_cbf"])
    overlap = sorted(set(seeds) & set(training_seeds(doc)))
    if overlap:
        raise ConfigError(
            f"config.evaluation.heldout_seeds overlap training seeds {overlap}: a held-out "
            "evaluation that shares a seed with training is not held out"
        )

    g = _keys("config.gates", doc["gates"],
              {"min_survival_rate", "min_objective_rate", "champion_tolerance",
               "baseline_tolerance", "max_reproduction_error"})
    _num("config.gates.min_survival_rate", g["min_survival_rate"], 0.0, 1.0)
    _num("config.gates.min_objective_rate", g["min_objective_rate"], 0.0, 1.0)
    _keys("config.gates.champion_tolerance", g["champion_tolerance"], set(CHAMPION_TOLERANCE_KEYS))
    for k in CHAMPION_TOLERANCE_KEYS:
        _num(f"config.gates.champion_tolerance.{k}", g["champion_tolerance"][k], 0.0, 1.0)
    _keys("config.gates.baseline_tolerance", g["baseline_tolerance"], set(BASELINE_TOLERANCE_KEYS))
    for k in BASELINE_TOLERANCE_KEYS:
        _num(f"config.gates.baseline_tolerance.{k}", g["baseline_tolerance"][k], 0.0, 1.0)
    _num("config.gates.max_reproduction_error", g["max_reproduction_error"], 0.0, 1e-2)

    ret = _keys("config.retention", doc["retention"], {"keep_snapshots", "keep_candidates"})
    _int("config.retention.keep_snapshots", ret["keep_snapshots"], 2, 10_000)
    _int("config.retention.keep_candidates", ret["keep_candidates"], 2, 10_000)
    return doc


def training_seeds(doc: dict) -> list[int]:
    tr = doc.get("training") or {}
    return [int(tr[k]["seed"]) for k in ("nightly", "weekly")
            if isinstance(tr.get(k), dict) and isinstance(tr[k].get("seed"), int)]


# --- loading -----------------------------------------------------------------


def load_default() -> dict:
    """The checked-in default, validated. Raises if the shipped file is broken."""
    return validate(json.loads(DEFAULT_CONFIG_PATH.read_text()), source="default")


def deployed_schedule() -> dict:
    return dict(load_default()["schedule"])


def load_effective(lay: layout.Layout) -> tuple[dict, str]:
    """``(config, source)``: the Volume config with the deployed schedule, or the default.

    Raises ``ConfigError`` if a Volume config exists but is invalid. It does NOT
    fall back to the default in that case: an operator who paused the pipeline
    and then broke the file must not have the pause silently lifted.
    """
    default = load_default()
    try:
        raw = layout.read_json(lay.config_current)
    except (json.JSONDecodeError, ValueError) as e:
        raise ConfigError(f"{lay.config_current} does not parse: {e}") from None
    if raw is None:
        return default, "default"
    cfg = validate(raw, source="volume")
    cfg["schedule"] = default["schedule"]
    return cfg, "volume"


# --- digests -----------------------------------------------------------------


def config_digest(cfg: dict) -> str:
    """Digest of what a config asks for, ignoring operational fields."""
    return layout.digest({k: v for k, v in cfg.items() if k not in OPERATIONAL_KEYS})


def stage_digest(cfg: dict, stage: str) -> str:
    """Digest of only the sections a stage's output depends on.

    A gate threshold change must not re-train anything, and a training change
    must not re-snapshot anything, so each stage keys on its own inputs.
    """
    if stage == "snapshot":
        part = {"aoi": cfg["aoi"], "snapshot": cfg["snapshot"]}
    elif stage in ("nightly", "weekly", "manual"):
        kind = "weekly" if stage == "weekly" else "nightly"
        part = {
            "aoi": cfg["aoi"],
            "training": {kind: cfg["training"][kind], "use_cbf": cfg["training"]["use_cbf"]},
            "evaluation": cfg["evaluation"],
        }
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return layout.digest(part)


# --- versioning --------------------------------------------------------------


def next_version(current: dict, changes: dict, *, updated_by: str, now=None) -> dict:
    """A new config version: ``current`` with ``changes`` applied, version + 1.

    ``changes`` replaces top-level sections wholesale (no deep merge -- a merge
    that silently keeps a stale nested key is how a threshold survives the edit
    that was meant to remove it). The result is validated before it is returned.
    """
    forbidden = sorted(set(changes) & {"schema", "version", "updated_utc", "updated_by", "schedule"})
    if forbidden:
        raise ConfigError(f"cannot set {forbidden} directly")
    doc = {k: v for k, v in copy.deepcopy(current).items() if k != "schedule"}
    doc.update(copy.deepcopy(changes))
    doc["version"] = int(current["version"]) + 1
    doc["updated_utc"] = layout.utc_stamp(now)
    doc["updated_by"] = updated_by
    return validate(doc, source="volume")


def publish(lay: layout.Layout, doc: dict, *, expected_version: int | None) -> dict:
    """Write ``doc`` as the current config, refusing a stale base version.

    ``expected_version`` is the version the edit was based on (``None`` when no
    Volume config exists yet). The history entry is written first and is
    write-once, so two publishers racing to the same version number cannot
    both succeed; ``current.json`` is then replaced atomically.
    """
    doc = validate(doc, source="volume")
    on_disk = layout.read_json(lay.config_current)
    base = None if on_disk is None else on_disk.get("version")
    if base != expected_version:
        raise ConfigConflict(
            f"config is at version {base}, this edit was based on {expected_version}; "
            "re-read and re-apply"
        )
    if expected_version is not None and doc["version"] != expected_version + 1:
        raise ConfigConflict(f"new version must be {expected_version + 1}, got {doc['version']}")
    try:
        layout.write_once_json(lay.config_version(doc["version"]), doc)
    except layout.ImmutableRecordError:
        raise ConfigConflict(f"config version {doc['version']} was already published") from None
    layout.atomic_write_json(lay.config_current, doc)
    return doc
