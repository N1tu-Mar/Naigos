"""The persistent layout on the ``naigos-runs`` Volume, and the rules for writing it.

The Volume is an artifact store, not a working directory. Everything the
pipeline writes lives under one namespace, ``/pipeline``, next to the existing
detached-run directories (which are named runs at the Volume root; the name
``pipeline`` is reserved so the two can never collide):

    /pipeline/config/current.json            versioned runtime config (small, replaced atomically)
    /pipeline/config/history/v<NNNNNN>.json  every config version ever published (write-once)
    /pipeline/snapshots/<snapshot-id>/       cache/, components/, DATA.md, snapshot.json (immutable)
    /pipeline/snapshots/.staging/<id>.<n>/   a snapshot being built; renamed into place when valid
    /pipeline/candidates/<candidate-id>/     candidate.json, run.json, checkpoints, history,
                                             evaluation.json, decision.json (immutable once decided)
    /pipeline/champions/current.json         the champion pointer (small, replaced atomically)
    /pipeline/champions/history/g<NNNNNN>.json  every pointer generation (write-once)
    /pipeline/locks/<lease-key>.json         lease records (mirrors of the atomic lease store)
    /pipeline/status/jobs/<job-key>.json     one record per scheduled stage attempt
    /pipeline/status/coordinator.json        coordinator heartbeat and last decisions

Three write disciplines, and every writer in the package uses one of them:

  * **write-once** (``write_once_json``) for records that describe something that
    has happened -- ``snapshot.json``, ``candidate.json``, ``evaluation.json``,
    ``decision.json``, config and champion history. A second write raises
    instead of overwriting, so a completed record cannot come to describe a
    different event.
  * **atomic replace** (``atomic_write_json``) for the two small pointers that
    are *meant* to move -- ``config/current.json`` and ``champions/current.json``
    -- and for status records. A reader sees the old file or the new file, never
    half of one.
  * **staged rename** for directories: a snapshot is built under ``.staging`` and
    renamed into ``snapshots/`` only after it validates, so a snapshot directory
    that exists is a complete one.

IDs are validated before they become paths, and every path is checked to stay
inside the pipeline root, so no ID -- scheduled, typed at a CLI or read back
out of a record -- can traverse out of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

#: Directory name of the pipeline namespace at the Volume root.
PIPELINE_DIRNAME = "pipeline"

SNAPSHOT_RECORD = "snapshot.json"
CANDIDATE_RECORD = "candidate.json"
EVALUATION_RECORD = "evaluation.json"
DECISION_RECORD = "decision.json"

# `s-<window>-<key>`: the window is the UTC date (or full stamp for a manual
# run) the snapshot belongs to, the key is the first 12 hex digits of its
# idempotency key. Same window + same inputs = same ID, which is what makes a
# double invocation of the cron land on one directory instead of two.
SNAPSHOT_ID_RE = re.compile(r"^s-[0-9]{8}(?:T[0-9]{6}Z)?-[0-9a-f]{12}$")
CANDIDATE_KINDS = ("nightly", "weekly", "manual")
CANDIDATE_ID_RE = re.compile(
    r"^c-[0-9]{8}(?:T[0-9]{6}Z)?-(?:" + "|".join(CANDIDATE_KINDS) + r")-[0-9a-f]{12}$"
)
# Lease and job keys: `<stage>` or `<stage>:<id>`.
KEY_RE = re.compile(r"^[a-z][a-z_]{0,31}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,80})?$")


class LayoutError(ValueError):
    """An ID or path that must not become a location on the Volume."""


class ImmutableRecordError(RuntimeError):
    """A write-once record already exists; it is never overwritten."""


# --- time --------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(when: datetime | None = None) -> str:
    return (when or utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# --- ids ---------------------------------------------------------------------


def validate_snapshot_id(value: object) -> str:
    if not isinstance(value, str) or not SNAPSHOT_ID_RE.match(value):
        raise LayoutError(f"invalid snapshot id {value!r}; expected s-<YYYYMMDD>-<12 hex>")
    return value


def validate_candidate_id(value: object) -> str:
    if not isinstance(value, str) or not CANDIDATE_ID_RE.match(value):
        raise LayoutError(
            f"invalid candidate id {value!r}; expected c-<YYYYMMDD>-<kind>-<12 hex> "
            f"with kind in {CANDIDATE_KINDS}"
        )
    return value


def validate_key(value: object) -> str:
    if not isinstance(value, str) or not KEY_RE.match(value) or ".." in value:
        raise LayoutError(f"invalid lease/job key {value!r}")
    return value


def snapshot_id(window: str, idem_key: str) -> str:
    return validate_snapshot_id(f"s-{window}-{idem_key[:12]}")


def candidate_id(window: str, kind: str, idem_key: str) -> str:
    return validate_candidate_id(f"c-{window}-{kind}-{idem_key[:12]}")


def kind_of_candidate(cid: str) -> str:
    return validate_candidate_id(cid).split("-")[2]


# --- hashing -----------------------------------------------------------------


def canonical_json(payload) -> str:
    """One byte sequence per value: the input to every digest in the pipeline."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def digest(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tree_files(root: str | os.PathLike) -> list[str]:
    """Every regular file under ``root``, as sorted POSIX relative paths.

    Temporary files from an interrupted atomic write are excluded; they are not
    part of anything.
    """
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.endswith(".tmp") and ".tmp-" not in p.name:
            out.append(p.relative_to(root).as_posix())
    return sorted(out)


def tree_digest(root: str | os.PathLike, files: Iterable[str] | None = None) -> dict:
    """Content hash of a directory: sha256 over sorted ``(path, sha256)`` pairs.

    Returns the per-file table as well, so a record carries exactly which bytes
    it vouches for and a later check can say which one changed.
    """
    root = Path(root)
    table = {rel: sha256_file(root / rel) for rel in (files if files is not None else tree_files(root))}
    return {"sha256": digest(sorted(table.items())), "files": table}


# --- writes ------------------------------------------------------------------


def _tmp_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")


def _write_tmp(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    with open(tmp, "w") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return tmp


def atomic_write_json(path: str | os.PathLike, payload) -> None:
    """Write to a unique temporary file in the same directory, then rename over.

    ``os.replace`` within a directory is atomic, so a reader sees the previous
    file or this one. The temporary name is unique per writer, so two writers
    racing cannot interleave their bytes into one temporary file.
    """
    path = Path(path)
    tmp = _write_tmp(path, payload)
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_once_json(path: str | os.PathLike, payload) -> None:
    """Create ``path`` with ``payload``; raise ``ImmutableRecordError`` if it exists.

    ``os.link`` fails if the destination exists, which makes create-if-absent a
    single filesystem operation where hard links are supported. Where they are
    not, the fallback checks then renames -- safe under the pipeline's leases,
    which are what guarantee one writer per record in the first place.
    """
    path = Path(path)
    if path.exists():
        raise ImmutableRecordError(f"{path} already exists and is write-once")
    tmp = _write_tmp(path, payload)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise ImmutableRecordError(f"{path} already exists and is write-once") from None
        except OSError:
            if path.exists():
                raise ImmutableRecordError(f"{path} already exists and is write-once") from None
            os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: str | os.PathLike) -> dict | None:
    """A JSON object, or None if absent. A file that exists but does not parse raises."""
    path = Path(path)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return doc


# --- the layout --------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Paths under one pipeline root. Construct with the Volume mount point."""

    volume_root: Path

    @property
    def root(self) -> Path:
        return Path(self.volume_root) / PIPELINE_DIRNAME

    def _inside(self, path: Path) -> Path:
        # Resolve lexically (no symlink following needed for validated IDs) and
        # refuse anything that would land outside the namespace.
        root = os.path.normpath(self.root)
        full = os.path.normpath(path)
        if full != root and not full.startswith(root + os.sep):
            raise LayoutError(f"{path} escapes the pipeline root {self.root}")
        return Path(full)

    # config
    @property
    def config_current(self) -> Path:
        return self.root / "config" / "current.json"

    def config_version(self, version: int) -> Path:
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise LayoutError(f"invalid config version {version!r}")
        return self.root / "config" / "history" / f"v{version:06d}.json"

    # snapshots
    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    def snapshot_dir(self, sid: str) -> Path:
        return self._inside(self.snapshots / validate_snapshot_id(sid))

    def snapshot_record(self, sid: str) -> Path:
        return self.snapshot_dir(sid) / SNAPSHOT_RECORD

    def staging_dir(self, sid: str, attempt: int) -> Path:
        if not isinstance(attempt, int) or attempt < 1:
            raise LayoutError(f"invalid staging attempt {attempt!r}")
        return self._inside(self.snapshots / ".staging" / f"{validate_snapshot_id(sid)}.{attempt}")

    def snapshot_ids(self) -> list[str]:
        if not self.snapshots.is_dir():
            return []
        return sorted(p.name for p in self.snapshots.iterdir()
                      if p.is_dir() and SNAPSHOT_ID_RE.match(p.name))

    # candidates
    @property
    def candidates(self) -> Path:
        return self.root / "candidates"

    def candidate_dir(self, cid: str) -> Path:
        return self._inside(self.candidates / validate_candidate_id(cid))

    def candidate_ids(self) -> list[str]:
        if not self.candidates.is_dir():
            return []
        return sorted(p.name for p in self.candidates.iterdir()
                      if p.is_dir() and CANDIDATE_ID_RE.match(p.name))

    # champions
    @property
    def champion_pointer(self) -> Path:
        return self.root / "champions" / "current.json"

    def champion_generation(self, generation: int) -> Path:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            raise LayoutError(f"invalid champion generation {generation!r}")
        return self.root / "champions" / "history" / f"g{generation:06d}.json"

    # locks and status
    def lock_path(self, key: str) -> Path:
        return self._inside(self.root / "locks" / f"{validate_key(key).replace(':', '__')}.json")

    def job_record(self, key: str) -> Path:
        return self._inside(self.root / "status" / "jobs" / f"{validate_key(key).replace(':', '__')}.json")

    @property
    def jobs_dir(self) -> Path:
        return self.root / "status" / "jobs"

    @property
    def coordinator_record(self) -> Path:
        return self.root / "status" / "coordinator.json"
