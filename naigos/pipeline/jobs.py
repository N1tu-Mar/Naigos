"""Job records and the pipeline's status vocabulary.

A *job* is one attempt at one stage for one target: snapshot ``s-...``, train
``c-...``, evaluate ``c-...``. Its record lives at
``/pipeline/status/jobs/<stage>__<target>.json`` and is the authoritative
remote record: ``scripts/pipeline.py status`` reads it from a fresh clone that
has never submitted anything, the same way ``modal_runs.py status`` reads a
run's manifest.

The vocabulary, and what each one promises:

  scheduled     not yet due; a cron will fire at ``next_fire`` (reported, never stored)
  queued        a Modal call was spawned; no container has recorded anything yet
  running       a worker is writing, heartbeat recent or Modal confirms it is live
  completed     the stage finished and its output validated
  failed        the stage raised, was killed, or was refused (e.g. an ambiguous lease)
  skipped       the stage deliberately did nothing -- paused, not due, no new input
  rejected      a candidate that was evaluated and failed a promotion gate
  inconclusive  a candidate whose evaluation could not decide a gate; never promoted
  promoted      a candidate the champion pointer names, now or in its history
  unknown       the record says live, the heartbeat is old and Modal says nothing

Merging a job record with Modal's call state follows ``runmeta.classify_status``:
a worker-recorded terminal state wins; otherwise an upstream terminal state
overrides a live record (the worker was killed before it could say so).
"""

from __future__ import annotations

from datetime import datetime

from ..rl import runmeta
from . import layout

SCHEDULED = "scheduled"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"
REJECTED = "rejected"
INCONCLUSIVE = "inconclusive"
PROMOTED = "promoted"
UNKNOWN = "unknown"

STATUSES = (SCHEDULED, QUEUED, RUNNING, COMPLETED, FAILED, SKIPPED, REJECTED, INCONCLUSIVE,
            PROMOTED, UNKNOWN)
LIVE = (QUEUED, RUNNING)
TERMINAL = (COMPLETED, FAILED, SKIPPED, REJECTED, INCONCLUSIVE, PROMOTED)

STAGES = ("snapshot", "train", "evaluate", "promote")
STALE_AFTER_S = 3600


def job_key(stage: str, target: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return layout.validate_key(f"{stage}:{target}")


def new_record(*, stage: str, target: str, idempotency_key: str, window: str,
               config_digest: str, code: dict, parents: dict, now: datetime | None = None) -> dict:
    stamp = layout.utc_stamp(now)
    return {
        "schema": 1,
        "key": job_key(stage, target),
        "stage": stage,
        "target": target,
        "status": QUEUED,
        "idempotency_key": idempotency_key,
        "window": window,
        "config_digest": config_digest,
        "code": code,
        "parents": parents,
        "call_id": None,
        "created_utc": stamp,
        "updated_utc": stamp,
        "heartbeat_utc": stamp,
        "failure_reason": None,
        "events": [{"at_utc": stamp, "status": QUEUED}],
    }


def update(record: dict, *, status: str | None = None, now: datetime | None = None,
           **fields) -> dict:
    """A new record with ``fields`` applied and one event appended. Pure."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    stamp = layout.utc_stamp(now)
    out = {**record, **{k: v for k, v in fields.items()}}
    if "failure_reason" in fields and fields["failure_reason"] is not None:
        out["failure_reason"] = runmeta.redact(str(fields["failure_reason"]))[:2000]
    out["updated_utc"] = stamp
    out["heartbeat_utc"] = stamp
    if status is not None:
        out["status"] = status
        event = {"at_utc": stamp, "status": status}
        if status == FAILED and out.get("failure_reason"):
            event["reason"] = out["failure_reason"]
        out["events"] = (list(record.get("events") or []) + [event])[-50:]
    return out


def read(lay: layout.Layout, key: str) -> dict | None:
    return layout.read_json(lay.job_record(key))


def write(lay: layout.Layout, record: dict) -> None:
    layout.atomic_write_json(lay.job_record(record["key"]), record)


def all_records(lay: layout.Layout) -> list[dict]:
    out = []
    if not lay.jobs_dir.is_dir():
        return out
    for p in sorted(lay.jobs_dir.glob("*.json")):
        try:
            doc = layout.read_json(p)
        except (ValueError, OSError):
            continue
        if doc:
            out.append(doc)
    return out


def classify(record: dict | None, *, remote_state: object = None, now: datetime | None = None,
             stale_after_s: int = STALE_AFTER_S) -> dict:
    """Merge a job record with Modal's account of its call into one verdict."""
    now = now or layout.utc_now()
    if record is None:
        return {"status": UNKNOWN, "notes": ["no job record"], "heartbeat_age_s": None}
    recorded = record.get("status")
    if recorded not in STATUSES:
        recorded = UNKNOWN
    upstream = runmeta.parse_modal_state(remote_state)
    notes: list[str] = []
    reason = record.get("failure_reason")
    if recorded in TERMINAL:
        status = recorded
    elif upstream in runmeta.TERMINAL_STATES:
        # A live record whose call has ended: the worker did not get to say how.
        status = FAILED
        reason = reason or (
            f"lost: Modal reports the call {upstream} while the job record still said "
            f"{recorded}; the worker was killed before it could record why"
        )
        if upstream == runmeta.COMPLETED:
            reason = ("lost: the call returned but the worker never recorded completion; "
                      "its output is not trusted")
        notes.append(f"modal={upstream}, record={recorded}")
    elif upstream in runmeta.LIVE_STATES:
        status = RUNNING if upstream == runmeta.RUNNING else (recorded if recorded in LIVE else QUEUED)
    else:
        status = recorded
    beat = layout.parse_utc(record.get("heartbeat_utc"))
    age = None if beat is None else max(int((now - beat).total_seconds()), 0)
    if status in LIVE and upstream == runmeta.UNKNOWN and age is not None and age > stale_after_s:
        notes.append(f"no heartbeat for {age}s and Modal reported nothing")
        status = UNKNOWN
    return {"status": status, "failure_reason": reason, "heartbeat_age_s": age,
            "modal_state": upstream, "recorded": recorded, "notes": notes}
