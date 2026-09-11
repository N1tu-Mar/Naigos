"""Narrow leases: at most one writer per snapshot, per candidate, per champion pointer.

Why not a lock file on the Volume alone: Modal Volumes are last-write-wins with
no compare-and-swap ("any data the last writer didn't have when committing
changes will be lost"), and a commit from one container is invisible to another
until it reloads. Two containers can each check for a lock file, see none, and
both write one. So the *atomic* part of a lease lives in a store that has a
real create-if-absent -- a named ``modal.Dict``, whose ``put(key, value,
skip_if_exists=True)`` returns whether it inserted -- and the Volume gets a
mirror under ``/pipeline/locks/`` purely so the lease is visible next to the
artifacts it guards. Tests use ``FileLeaseStore``, whose ``O_EXCL`` create has
the same semantics on a local filesystem.

The decision about a lease that already exists is where the failure modes are,
and it is made by one pure function, ``assess``:

  * the call that holds it is **terminal** by Modal's own account  -> ``dead``:
    no writer can exist, reclaiming is safe;
  * the call is **queued or running**                              -> ``held``;
  * Modal says nothing, heartbeat **recent**                        -> ``held``;
  * Modal says nothing, heartbeat **old** (or the lease names no
    call, e.g. the coordinator died between acquiring and spawning) -> ``stale``.

``stale`` is ambiguous, and ambiguity fails closed: the stage does not run and
the job records why. An operator who has checked that the writer is gone
clears it with ``scripts/pipeline.py unlock <key>``. Running a second writer
on a guess is the one outcome this module exists to prevent.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from ..rl import runmeta
from . import layout

#: A lease with no heartbeat for this long, whose call Modal cannot vouch for,
#: is ambiguous. Workers heartbeat every ``HEARTBEAT_S``.
STALE_AFTER_S = 15 * 60
HEARTBEAT_S = 60

FREE, HELD, STALE, DEAD = "free", "held", "stale", "dead"


class LeaseError(RuntimeError):
    """Base class; carries the lease record that blocked the caller."""

    def __init__(self, message: str, record: dict | None = None):
        super().__init__(message)
        self.record = record


class LeaseHeld(LeaseError):
    """A live writer holds the lease. Skip; this is the normal duplicate case."""


class LeaseAmbiguous(LeaseError):
    """The holder may or may not be alive. Fail closed; an operator decides."""


class LeaseLost(LeaseError):
    """The caller's token no longer matches: someone cleared or took the lease."""


class LeaseStore(Protocol):
    def create(self, key: str, record: dict) -> bool: ...
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, record: dict) -> None: ...
    def delete(self, key: str) -> None: ...


class FileLeaseStore:
    """``O_EXCL`` create-if-absent on a local filesystem. Atomic where the FS is."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{layout.validate_key(key).replace(':', '__')}.json"

    def create(self, key: str, record: dict) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(record, indent=2, sort_keys=True))
        return True

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            # A lease file that exists but cannot be read is not "free".
            return {"key": key, "token": None, "unreadable": True}

    def put(self, key: str, record: dict) -> None:
        layout.atomic_write_json(self._path(key), record)

    def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path(key).unlink()


class DictLeaseStore:
    """A named ``modal.Dict`` (or anything with its ``put(..., skip_if_exists=)``).

    ``put(key, value, skip_if_exists=True)`` is documented to return False when
    the key already existed, which is the whole of what a lease needs. Entries
    expire after 7 days without reads or writes; a held lease is heartbeated
    every minute, so that only ever reaps leases nobody is looking at.
    """

    def __init__(self, modal_dict, mirror_dir: str | os.PathLike | None = None):
        self._d = modal_dict
        self._mirror = FileLeaseStore(mirror_dir) if mirror_dir is not None else None

    def _mirror_put(self, key: str, record: dict | None) -> None:
        if self._mirror is None:
            return
        # The mirror is for humans reading the Volume. Never let it fail a lease.
        with contextlib.suppress(Exception):
            if record is None:
                self._mirror.delete(key)
            else:
                self._mirror.put(key, record)

    def create(self, key: str, record: dict) -> bool:
        ok = bool(self._d.put(key, record, skip_if_exists=True))
        if ok:
            self._mirror_put(key, record)
        return ok

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def put(self, key: str, record: dict) -> None:
        self._d.put(key, record)
        self._mirror_put(key, record)

    def delete(self, key: str) -> None:
        with contextlib.suppress(KeyError):
            self._d.pop(key)
        self._mirror_put(key, None)


@dataclass
class Lease:
    key: str
    token: str
    owner: str
    acquired_utc: str
    heartbeat_utc: str
    call_id: str | None = None
    note: str | None = None

    def as_record(self) -> dict:
        return asdict(self)


def assess(record: dict | None, *, now: datetime, remote_state: object = None,
           stale_after_s: int = STALE_AFTER_S) -> tuple[str, str]:
    """``(verdict, why)`` for an existing lease record. Pure; see module docstring."""
    if record is None:
        return FREE, "no lease"
    if record.get("unreadable"):
        return STALE, "the lease record exists but cannot be read"
    state = runmeta.parse_modal_state(remote_state)
    if state in runmeta.TERMINAL_STATES:
        return DEAD, f"call {record.get('call_id')} is {state} by Modal's account"
    if state in runmeta.LIVE_STATES:
        return HELD, f"call {record.get('call_id')} is {state}"
    beat = layout.parse_utc(record.get("heartbeat_utc") or record.get("acquired_utc"))
    if beat is None:
        return STALE, "the lease has no parseable heartbeat"
    age = int((now - beat).total_seconds())
    if age <= stale_after_s:
        return HELD, f"heartbeat {age}s ago"
    who = f"call {record['call_id']}" if record.get("call_id") else "a holder that never recorded a call"
    return STALE, (f"no heartbeat for {age}s from {who}, and Modal cannot confirm it is gone. "
                   "Refusing to guess; clear it with `scripts/pipeline.py unlock` once checked.")


class LeaseManager:
    """Acquire, heartbeat, hand off and release leases against one store."""

    def __init__(self, store: LeaseStore, *, now: Callable[[], datetime] = layout.utc_now,
                 remote_state: Callable[[str | None], object] | None = None,
                 stale_after_s: int = STALE_AFTER_S):
        self.store = store
        self._now = now
        self._remote_state = remote_state or (lambda call_id: None)
        self.stale_after_s = stale_after_s

    def inspect(self, key: str) -> tuple[str, str, dict | None]:
        rec = self.store.get(layout.validate_key(key))
        remote = self._remote_state(rec.get("call_id")) if rec and rec.get("call_id") else None
        verdict, why = assess(rec, now=self._now(), remote_state=remote,
                              stale_after_s=self.stale_after_s)
        return verdict, why, rec

    def acquire(self, key: str, owner: str, *, note: str | None = None) -> Lease:
        """Take the lease or raise ``LeaseHeld`` / ``LeaseAmbiguous``.

        A ``dead`` holder is reclaimed; that is the only case where one writer's
        lease is replaced by another's without an operator.
        """
        key = layout.validate_key(key)
        stamp = layout.utc_stamp(self._now())
        lease = Lease(key=key, token=secrets.token_hex(16), owner=runmeta.redact(owner),
                      acquired_utc=stamp, heartbeat_utc=stamp, note=note)
        if self.store.create(key, lease.as_record()):
            return lease
        verdict, why, rec = self.inspect(key)
        if verdict == FREE:  # released between our create and our read
            if self.store.create(key, lease.as_record()):
                return lease
            raise LeaseHeld(f"{key}: taken concurrently", self.store.get(key))
        if verdict == HELD:
            raise LeaseHeld(f"{key} is held: {why}", rec)
        if verdict == STALE:
            raise LeaseAmbiguous(f"{key}: {why}", rec)
        # DEAD: reclaim. Delete only if it is still the dead holder's record.
        current = self.store.get(key)
        if current and current.get("token") == (rec or {}).get("token"):
            self.store.delete(key)
        if self.store.create(key, {**lease.as_record(), "reclaimed_from": rec}):
            return lease
        raise LeaseHeld(f"{key}: reclaimed concurrently by another writer", self.store.get(key))

    def _owned(self, lease: Lease) -> dict:
        rec = self.store.get(lease.key)
        if not rec or rec.get("token") != lease.token:
            raise LeaseLost(f"{lease.key}: this writer no longer holds the lease", rec)
        return rec

    def heartbeat(self, lease: Lease, **fields) -> Lease:
        rec = self._owned(lease)
        lease.heartbeat_utc = layout.utc_stamp(self._now())
        for k, v in fields.items():
            if k in ("call_id", "note", "owner"):
                setattr(lease, k, v)
        self.store.put(lease.key, {**rec, **lease.as_record()})
        return lease

    def release(self, lease: Lease) -> None:
        """Release if still ours. Releasing a lease someone else holds is a no-op."""
        rec = self.store.get(lease.key)
        if rec and rec.get("token") == lease.token:
            self.store.delete(lease.key)

    def force_clear(self, key: str) -> dict | None:
        """Operator action: remove whatever lease exists. Returns what was removed."""
        key = layout.validate_key(key)
        rec = self.store.get(key)
        self.store.delete(key)
        return rec

    @contextlib.contextmanager
    def keep_alive(self, lease: Lease, interval_s: float = HEARTBEAT_S):
        """Heartbeat from a daemon thread for the duration of the block.

        Independent of the work's own cadence: a training run that evaluates
        every twenty minutes must not look stale between evaluations. A lost
        lease (someone cleared it) is recorded on ``lost`` so the worker can
        refuse to publish anything once the block ends.
        """
        stop = threading.Event()
        state = {"lost": None}

        def beat():
            while not stop.wait(interval_s):
                try:
                    self.heartbeat(lease)
                except LeaseLost as e:
                    state["lost"] = str(e)
                    return
                except Exception:  # noqa: BLE001 - transient store errors are retried next beat
                    continue

        t = threading.Thread(target=beat, name=f"lease-{lease.key}", daemon=True)
        t.start()
        try:
            yield state
        finally:
            stop.set()
            t.join(timeout=5)
