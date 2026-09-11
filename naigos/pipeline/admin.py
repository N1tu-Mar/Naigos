"""Operator commands, executed where the data is.

``scripts/pipeline.py`` sends each command to the deployed ``admin`` Function,
which calls ``handle`` against the mounted Volume; the same function runs
offline against a local directory (``--volume-dir``) for inspection and tests.
Nothing here trains, evaluates or promotes -- promotion is its own Function
because it must re-run the evaluation.

Pause is a config change, not a local flag. Modal schedules are
deployment-defined and cannot be paused; what the pipeline can do, and does, is
publish a new config version with ``paused: true`` that every tick checks
before any costly work. The crons keep firing (a few seconds of CPU each) and
record ``skipped: paused``, so the state is visible in ``status`` rather than
inferred from silence.
"""

from __future__ import annotations

import getpass
from typing import Callable

from ..rl import runmeta
from . import config as pcfg
from . import coordinator, jobs, layout, promotion
from .worker import Services

READ_ONLY = ("status", "list-snapshots", "list-candidates", "inspect", "config-show", "prune-plan")


class AdminError(ValueError):
    """A command the pipeline refuses, with the reason."""


def default_actor() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "operator"


def _publish_change(svc: Services, changes: dict, *, actor: str) -> dict:
    cur = layout.read_json(svc.lay.config_current)
    base = cur or pcfg.load_default()
    doc = pcfg.next_version(base, changes, updated_by=actor, now=svc.now())
    pcfg.publish(svc.lay, doc, expected_version=(cur or {}).get("version"))
    svc.commit()
    return doc


def _inspect(svc: Services, target: str) -> dict:
    lay = svc.lay
    if target.startswith("s-"):
        sid = layout.validate_snapshot_id(target)
        rec = layout.read_json(lay.snapshot_record(sid)) if lay.snapshot_record(sid).exists() else None
        from . import snapshot as psnap

        return {"snapshot": rec and {k: v for k, v in rec.items() if k != "content"},
                "content_sha256": ((rec or {}).get("content") or {}).get("sha256"),
                "verify_problems": psnap.verify_completed(lay, sid) if rec else ["not completed"],
                "job": jobs.read(lay, jobs.job_key("snapshot", sid))}
    if target.startswith("c-"):
        cid = layout.validate_candidate_id(target)
        d = lay.candidate_dir(cid)

        def rd(name):
            return layout.read_json(d / name) if (d / name).exists() else None

        rec = rd(layout.CANDIDATE_RECORD)
        return {"candidate": rec and {k: v for k, v in rec.items() if k != "run_meta"},
                "manifest": runmeta.read_manifest(d),
                "evaluation": rd(layout.EVALUATION_RECORD),
                "decision": rd(layout.DECISION_RECORD),
                "jobs": {s: jobs.read(lay, jobs.job_key(s, cid)) for s in ("train", "evaluate")},
                "checkpoints": [p.name for p in runmeta.checkpoint_paths(d)]}
    if target.startswith("g") and target[1:].isdigit():
        return {"generation": layout.read_json(lay.champion_generation(int(target[1:])))}
    if target == "champion":
        return {"champion": promotion.read_champion(lay), "history": promotion.champion_history(lay)}
    raise AdminError(f"cannot inspect {target!r}: give a snapshot id (s-...), candidate id (c-...), "
                     "a champion generation (g<N>) or 'champion'")


def handle(svc: Services, command: str, args: dict, *, deployed_schedule: dict | None = None,
           remove_tree: Callable | None = None) -> dict:
    lay = svc.lay
    actor = runmeta.redact(str(args.get("actor") or default_actor()))[:120]
    if command == "status":
        report = coordinator.status_report(lay, remote_state=svc.remote_state, now=svc.now())
        if deployed_schedule is not None:
            report["deployed_schedule"] = deployed_schedule
        return report
    if command == "list-snapshots":
        return {"snapshots": coordinator.status_report(lay, remote_state=svc.remote_state,
                                                       now=svc.now())["snapshots"]}
    if command == "list-candidates":
        return {"candidates": coordinator.status_report(lay, remote_state=svc.remote_state,
                                                        now=svc.now())["candidates"]}
    if command == "inspect":
        return _inspect(svc, str(args.get("id")))
    if command == "config-show":
        cfg, source = pcfg.load_effective(lay)
        return {"source": source, "config": cfg}
    if command == "config-set":
        changes = args.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise AdminError("config-set needs a JSON object of top-level sections to replace")
        return {"config": _publish_change(svc, changes, actor=actor)}
    if command == "pause":
        doc = _publish_change(svc, {"paused": True, "pause_reason": args.get("reason") or "paused by operator"},
                              actor=actor)
        return {"paused": True, "version": doc["version"],
                "note": "the crons still fire; each tick records 'skipped: paused' and does no costly "
                        "work. Jobs already running are not interrupted -- cancel them in the Modal "
                        "dashboard if needed."}
    if command == "resume":
        doc = _publish_change(svc, {"paused": False, "pause_reason": None}, actor=actor)
        return {"paused": False, "version": doc["version"]}
    if command == "retry":
        return coordinator.retry(svc, str(args.get("id")))
    if command == "unlock":
        key = layout.validate_key(str(args.get("key")))
        verdict, why, rec = svc.leases.inspect(key)
        if verdict == "held" and not args.get("force_live"):
            raise AdminError(f"{key} is held by a live writer ({why}); refusing to clear it")
        removed = svc.leases.force_clear(key)
        return {"cleared": key, "was": removed, "verdict": verdict, "why": why}
    if command in ("prune-plan", "prune"):
        cfg, _ = pcfg.load_effective(lay)
        live = {r["target"] for r in jobs.all_records(lay)
                if jobs.classify(r, remote_state=svc.remote_state(r.get("call_id")),
                                 now=svc.now())["status"] in jobs.LIVE + (jobs.UNKNOWN,)}
        plan = coordinator.retention_plan(lay, cfg, live_targets=live)
        if command == "prune":
            if remove_tree is None:
                raise AdminError("prune can only run where the Volume is mounted")
            for cid in plan["delete_candidates"]:
                remove_tree(lay.candidate_dir(cid))
            for sid in plan["delete_snapshots"]:
                remove_tree(lay.snapshot_dir(sid))
            svc.commit()
            plan["applied"] = True
        return plan
    raise AdminError(f"unknown command {command!r}")
