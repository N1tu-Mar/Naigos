"""The scheduled coordinator: decide what is due, and dispatch it at most once.

Three deployed crons call ``tick`` -- ``snapshot`` daily, ``nightly`` and
``weekly`` -- and an operator can call it by hand. A tick is cheap (a small CPU
container reading a few JSON files) and does no costly work itself; it spawns
the snapshot, training and evaluation Functions and records what it did.

Safe under duplicate delivery, overlap and partial failure, because every
dispatch goes through the same four checks, in order:

  1. **Deterministic identity.** The target's ID comes from an idempotency key
     over the stage's config digest, the input snapshot and the code commit
     (plus the cron window for snapshots). Two ticks for the same work compute
     the same ID.
  2. **Already done?** A completed snapshot directory, or a candidate with a
     decision, is skipped.
  3. **Already in flight or failed?** The job record is merged with Modal's
     call state. Live -> skip. Failed -> skip and surface the reason: nothing is
     re-run without an operator (``scripts/pipeline.py retry``), which is the
     answer to "never use blind retries to duplicate expensive training".
  4. **Lease.** An atomic create-if-absent. Held -> skip; ambiguous (stale, no
     remote answer) -> fail closed and say so.

Before any of that, a tick reconciles job records whose calls Modal reports as
finished but whose workers never recorded it, and it refuses all costly work
when the Volume config says ``paused`` or the deployed code is not a clean
commit (a candidate trained from it could never pass provenance anyway).
"""

from __future__ import annotations

from datetime import datetime

from ..rl import runmeta
from . import candidate as pcand
from . import config as pcfg
from . import jobs, layout, leases, promotion, schedule
from . import snapshot as psnap
from .worker import Services

SNAPSHOT_FN = "snapshot_worker"
TRAIN_FN = "train_candidate"
EVALUATE_FN = "evaluate_candidate"
TICK_KINDS = ("snapshot", "nightly", "weekly")


def _action(kind: str, status: str, reason: str, **extra) -> dict:
    return {"stage": kind, "status": status, "reason": runmeta.redact(reason), **extra}


def _write_coordinator(svc: Services, summary: dict) -> None:
    prev = layout.read_json(svc.lay.coordinator_record) or {}
    history = (list(prev.get("recent") or []) + [summary])[-30:]
    layout.atomic_write_json(svc.lay.coordinator_record, {
        "schema": 1, "heartbeat_utc": summary["at_utc"], "last": summary, "recent": history})
    svc.commit()


# --- dispatch ----------------------------------------------------------------------


def _dispatch(svc: Services, *, stage: str, target: str, lease_key: str, fn: str, payload: dict,
              idem: str, window: str, cfg: dict, parents: dict, retry: bool = False,
              before_spawn=None) -> dict:
    key = jobs.job_key(stage, target)
    existing = jobs.read(svc.lay, key)
    attempt = 1
    if existing is not None:
        v = jobs.classify(existing, remote_state=svc.remote_state(existing.get("call_id")),
                          now=svc.now())
        attempt = int(existing.get("attempt") or 1) + 1
        if v["status"] in jobs.LIVE:
            return _action(stage, jobs.SKIPPED, f"{target} is already {v['status']}", target=target)
        if v["status"] == jobs.UNKNOWN:
            return _action(stage, jobs.SKIPPED,
                           f"{target} is unconfirmed ({'; '.join(v['notes'])}); not starting a second writer",
                           target=target)
        if v["status"] == jobs.FAILED and not retry:
            return _action(stage, jobs.SKIPPED,
                           f"{target} failed earlier ({v.get('failure_reason')}); it is not retried "
                           f"automatically -- `scripts/pipeline.py retry {target}`", target=target)
        if v["status"] in (jobs.COMPLETED, jobs.REJECTED, jobs.INCONCLUSIVE, jobs.PROMOTED, jobs.SKIPPED) \
                and not retry:
            return _action(stage, jobs.SKIPPED, f"{target} already {v['status']}", target=target)
    try:
        lease = svc.leases.acquire(lease_key, f"coordinator:{stage}:{target}")
    except leases.LeaseHeld as e:
        return _action(stage, jobs.SKIPPED, str(e), target=target)
    except leases.LeaseAmbiguous as e:
        rec = existing or jobs.new_record(stage=stage, target=target, idempotency_key=idem,
                                          window=window, config_digest=pcfg.config_digest(cfg),
                                          code=svc.code, parents=parents, now=svc.now())
        jobs.write(svc.lay, jobs.update(rec, status=jobs.FAILED, now=svc.now(),
                                        failure_reason=f"ambiguous lease: {e}"))
        svc.commit()
        return _action(stage, jobs.FAILED, f"ambiguous lease, failing closed: {e}", target=target)
    try:
        if before_spawn is not None:
            before_spawn()
        rec = jobs.new_record(stage=stage, target=target, idempotency_key=idem, window=window,
                              config_digest=pcfg.config_digest(cfg), code=svc.code,
                              parents=parents, now=svc.now())
        if existing is not None:
            rec["events"] = (list(existing.get("events") or []) + rec["events"])[-50:]
        rec["attempt"] = attempt
        rec["config_version"] = cfg.get("version")
        jobs.write(svc.lay, rec)
        svc.commit()
        call_id = svc.spawn(fn, {**payload, "attempt": attempt, "lease": lease.as_record()})
    except Exception as e:  # noqa: BLE001 - a failed spawn must release and record
        svc.leases.release(lease)
        rec = jobs.read(svc.lay, key) or jobs.new_record(
            stage=stage, target=target, idempotency_key=idem, window=window,
            config_digest=pcfg.config_digest(cfg), code=svc.code, parents=parents, now=svc.now())
        jobs.write(svc.lay, jobs.update(rec, status=jobs.FAILED, now=svc.now(),
                                        failure_reason=f"dispatch failed: {type(e).__name__}: {e}"))
        svc.commit()
        return _action(stage, jobs.FAILED, f"dispatch failed: {e}", target=target)
    try:
        svc.leases.heartbeat(lease, call_id=call_id)
    except leases.LeaseLost:
        pass  # the worker already finished and released it
    cur = jobs.read(svc.lay, key)
    if cur is not None and cur.get("status") == jobs.QUEUED and not cur.get("call_id"):
        jobs.write(svc.lay, jobs.update(cur, now=svc.now(), call_id=call_id))
        svc.commit()
    return _action(stage, jobs.QUEUED, f"spawned {fn}", target=target, call_id=call_id)


def dispatch_snapshot(svc: Services, cfg: dict, *, window: str, retry: bool = False) -> dict:
    sdig = pcfg.stage_digest(cfg, "snapshot")
    idem = layout.digest({"stage": "snapshot", "stage_digest": sdig,
                          "commit": svc.code.get("commit"), "window": window})
    sid = layout.snapshot_id(window, idem)
    if svc.lay.snapshot_dir(sid).exists() and not psnap.verify_completed(svc.lay, sid):
        return _action("snapshot", jobs.SKIPPED, f"{sid} already completed", target=sid)
    return _dispatch(svc, stage="snapshot", target=sid, lease_key=f"snapshot:{sid}", fn=SNAPSHOT_FN,
                     payload={"snapshot_id": sid, "window": window, "idempotency_key": idem,
                              "config_digest": pcfg.config_digest(cfg), "stage_digest": sdig},
                     idem=idem, window=window, cfg=cfg, parents={}, retry=retry)


def _candidates_created_on(lay: layout.Layout, day: str, kinds: tuple) -> int:
    n = 0
    for cid in lay.candidate_ids():
        if layout.kind_of_candidate(cid) not in kinds:
            continue
        rec = pcand.read_record(lay, cid) or {}
        if str(rec.get("created_utc", "")).startswith(day):
            n += 1
    return n


def _latest_decision(lay: layout.Layout, kind: str) -> tuple[str | None, dict | None]:
    for cid in sorted(lay.candidate_ids(), reverse=True):
        if layout.kind_of_candidate(cid) != kind:
            continue
        return cid, layout.read_json(lay.candidate_dir(cid) / layout.DECISION_RECORD)
    return None, None


def dispatch_candidate(svc: Services, cfg: dict, kind: str, *, retry: bool = False) -> dict:
    """Train a candidate of ``kind`` on the newest verified snapshot, if due."""
    lay = svc.lay
    sid = psnap.latest_completed(lay, aoi_name=cfg["aoi"])
    if sid is None:
        return _action(kind, jobs.SKIPPED, "no completed snapshot to train on")
    snap_rec = layout.read_json(lay.snapshot_record(sid))
    champion = promotion.read_champion(lay)
    p = pcand.plan(cfg, kind=kind, snapshot_rec=snap_rec, code=svc.code,
                   config_version=int(cfg["version"]), champion=champion, now=svc.now())
    cid, cand_dir = p["candidate_id"], lay.candidate_dir(p["candidate_id"])
    if (cand_dir / layout.DECISION_RECORD).exists():
        return _action(kind, jobs.SKIPPED,
                       f"{cid} already decided for snapshot {sid}; waiting for a new snapshot",
                       target=cid)
    if not retry and not cand_dir.exists():
        if kind == "weekly":
            if not cfg["training"]["weekly"]["enabled"]:
                return _action(kind, jobs.SKIPPED, "weekly training is disabled in config")
            prior, decision = _latest_decision(lay, "nightly")
            promoted = promotion.promoted_candidates(lay)
            if prior is None or decision is None or not (
                    decision.get("outcome") == promotion.ELIGIBLE or prior in promoted):
                return _action(kind, jobs.SKIPPED,
                               f"prior jobs did not pass: latest nightly {prior} decision "
                               f"{(decision or {}).get('outcome', 'none')}")
        else:
            day = layout.utc_stamp(svc.now())[:10]
            used = _candidates_created_on(lay, day, ("nightly", "manual"))
            if used >= cfg["training"]["max_candidates_per_day"]:
                return _action(kind, jobs.SKIPPED,
                               f"daily candidate budget used ({used}/"
                               f"{cfg['training']['max_candidates_per_day']})")
    if cfg["training"]["require_smoke"]:
        refusal = runmeta.smoke_gate(p["profile"].name, svc.smoke_marker(), svc.code.get("commit"))
        if refusal:
            return _action(kind, jobs.SKIPPED, f"smoke gate: {refusal}", target=cid)

    def create_candidate():
        path = cand_dir / layout.CANDIDATE_RECORD
        if path.exists():
            on_disk = layout.read_json(path)
            if on_disk.get("idempotency_key") != p["idempotency_key"]:
                raise layout.ImmutableRecordError(f"{path} describes a different candidate")
            return
        layout.write_once_json(path, p["record"])

    resume = bool(retry and runmeta.checkpoint_paths(cand_dir))
    return _dispatch(svc, stage="train", target=cid, lease_key=f"candidate:{cid}", fn=TRAIN_FN,
                     payload={"candidate_id": cid, "resume": resume},
                     idem=p["idempotency_key"], window=snap_rec["window"], cfg=cfg,
                     parents={"snapshot_id": sid}, retry=retry, before_spawn=create_candidate)


def dispatch_evaluation(svc: Services, cid: str, *, cfg: dict | None = None,
                        retry: bool = False) -> dict:
    cfg = cfg or pcfg.load_effective(svc.lay)[0]
    rec = pcand.read_record(svc.lay, cid) or {}
    return _dispatch(svc, stage="evaluate", target=cid, lease_key=f"candidate:{cid}",
                     fn=EVALUATE_FN, payload={"candidate_id": cid},
                     idem=rec.get("idempotency_key", ""), window=rec.get("window", ""), cfg=cfg,
                     parents={"snapshot_id": (rec.get("parents") or {}).get("snapshot_id")},
                     retry=retry)


# --- reconcile and tick ------------------------------------------------------------


def reconcile(svc: Services, cfg: dict, *, paused: bool) -> list[dict]:
    """Record lost workers as failed; queue evaluations that training left undone."""
    actions = []
    for rec in jobs.all_records(svc.lay):
        v = jobs.classify(rec, remote_state=svc.remote_state(rec.get("call_id")), now=svc.now())
        if rec.get("status") in jobs.LIVE and v["status"] == jobs.FAILED:
            jobs.write(svc.lay, jobs.update(rec, status=jobs.FAILED, now=svc.now(),
                                            failure_reason=v["failure_reason"]))
            svc.commit()
            actions.append(_action(rec["stage"], jobs.FAILED, v["failure_reason"], target=rec["target"]))
        if paused or rec.get("stage") != "train" or rec.get("status") != jobs.COMPLETED:
            continue
        cid = rec["target"]
        if (svc.lay.candidate_dir(cid) / layout.DECISION_RECORD).exists():
            continue
        if jobs.read(svc.lay, jobs.job_key("evaluate", cid)) is None:
            actions.append(dispatch_evaluation(svc, cid, cfg=cfg))
    return actions


def tick(svc: Services, kind: str, *, manual: bool = False) -> dict:
    """One coordinator invocation. Always records a summary; never raises for policy reasons."""
    if kind not in TICK_KINDS:
        raise ValueError(f"unknown tick kind {kind!r}")
    now: datetime = svc.now()
    svc.reload()
    summary = {"kind": kind, "at_utc": layout.utc_stamp(now), "manual": manual,
               "code": svc.code, "actions": []}
    try:
        cfg, source = pcfg.load_effective(svc.lay)
    except pcfg.ConfigError as e:
        summary.update(status=jobs.FAILED, reason=f"config invalid, doing nothing: {e}")
        _write_coordinator(svc, summary)
        return summary
    summary.update(config_version=cfg["version"], config_source=source)
    summary["actions"].extend(reconcile(svc, cfg, paused=cfg["paused"]))
    if cfg["paused"]:
        summary.update(status=jobs.SKIPPED,
                       reason=f"paused by config v{cfg['version']}: {cfg.get('pause_reason') or 'no reason given'}")
        _write_coordinator(svc, summary)
        return summary
    if svc.code.get("commit") is None or svc.code.get("dirty") is not False:
        summary.update(status=jobs.SKIPPED,
                       reason="deployed code is not a clean commit; redeploy from a clean tree. "
                              "Candidates from it could never pass provenance.")
        _write_coordinator(svc, summary)
        return summary
    sch = cfg["schedule"]
    if kind == "snapshot":
        window = schedule.manual_window(now) if manual else \
            schedule.window_label(schedule.previous_fire(sch["snapshot_cron"], now))
        action = dispatch_snapshot(svc, cfg, window=window)
    else:
        action = dispatch_candidate(svc, cfg, kind)
    summary["actions"].append(action)
    summary.update(status=action["status"], reason=action["reason"])
    _write_coordinator(svc, summary)
    return summary


def retry(svc: Services, target: str) -> dict:
    """Operator action: re-dispatch one failed stage. Training resumes from its last checkpoint."""
    cfg, _ = pcfg.load_effective(svc.lay)
    if cfg["paused"]:
        return _action("retry", jobs.SKIPPED, "pipeline is paused; resume it first")
    if target.startswith("s-"):
        sid = layout.validate_snapshot_id(target)
        rec = jobs.read(svc.lay, jobs.job_key("snapshot", sid))
        if rec is None:
            return _action("snapshot", jobs.SKIPPED, f"no job record for {sid}")
        return _dispatch(svc, stage="snapshot", target=sid, lease_key=f"snapshot:{sid}",
                         fn=SNAPSHOT_FN, payload={
                             "snapshot_id": sid, "window": rec["window"],
                             "idempotency_key": rec["idempotency_key"],
                             "config_digest": pcfg.config_digest(cfg),
                             "stage_digest": pcfg.stage_digest(cfg, "snapshot")},
                         idem=rec["idempotency_key"], window=rec["window"], cfg=cfg, parents={},
                         retry=True)
    cid = layout.validate_candidate_id(target)
    if (svc.lay.candidate_dir(cid) / layout.DECISION_RECORD).exists():
        return _action("retry", jobs.SKIPPED, f"{cid} already has a decision; it is immutable")
    train = jobs.read(svc.lay, jobs.job_key("train", cid))
    if train is not None and train.get("status") == jobs.COMPLETED:
        return dispatch_evaluation(svc, cid, cfg=cfg, retry=True)
    rec = pcand.read_record(svc.lay, cid)
    if rec is None:
        return _action("retry", jobs.SKIPPED, f"{cid} has no candidate.json")
    return _dispatch(svc, stage="train", target=cid, lease_key=f"candidate:{cid}", fn=TRAIN_FN,
                     payload={"candidate_id": cid,
                              "resume": bool(runmeta.checkpoint_paths(svc.lay.candidate_dir(cid)))},
                     idem=rec["idempotency_key"], window=rec["window"], cfg=cfg,
                     parents={"snapshot_id": rec["parents"]["snapshot_id"]}, retry=True)


# --- status ------------------------------------------------------------------------


def candidate_status(lay: layout.Layout, cid: str, verdicts: dict, promoted: set,
                     champion_cid: str | None) -> dict:
    d = lay.candidate_dir(cid)
    rec = pcand.read_record(lay, cid) or {}
    decision = layout.read_json(d / layout.DECISION_RECORD) if (d / layout.DECISION_RECORD).exists() else None
    train = verdicts.get(jobs.job_key("train", cid))
    ev = verdicts.get(jobs.job_key("evaluate", cid))
    if decision is not None:
        outcome = decision["outcome"]
        status = (jobs.PROMOTED if cid in promoted else
                  {promotion.ELIGIBLE: jobs.COMPLETED, promotion.REJECTED: jobs.REJECTED,
                   promotion.INCONCLUSIVE: jobs.INCONCLUSIVE}[outcome])
    else:
        outcome = None
        live = ev or train
        status = live["status"] if live else jobs.UNKNOWN
    failing = [c for c in (decision or {}).get("checks", []) if c["status"] != promotion.PASS]
    return {
        "candidate_id": cid, "kind": layout.kind_of_candidate(cid), "status": status,
        "decision": outcome, "action": (decision or {}).get("action"),
        "is_champion": cid == champion_cid,
        "snapshot_id": (rec.get("parents") or {}).get("snapshot_id"),
        "created_utc": rec.get("created_utc"),
        "failing_gates": [f"{c['gate']}: {c['status']} ({c['detail']})" for c in failing][:10],
        "failure_reason": ((ev or train) or {}).get("failure_reason"),
    }


def status_report(lay: layout.Layout, *, remote_state=lambda call_id: None, now=None) -> dict:
    """The authoritative remote picture, assembled from the Volume alone (plus Modal call state)."""
    now = now or layout.utc_now()
    report: dict = {"at_utc": layout.utc_stamp(now)}
    try:
        cfg, source = pcfg.load_effective(lay)
        report["config"] = {"version": cfg["version"], "source": source, "paused": cfg["paused"],
                            "pause_reason": cfg.get("pause_reason"),
                            "auto_promote": cfg["auto_promote"], "aoi": cfg["aoi"]}
    except pcfg.ConfigError as e:
        cfg = pcfg.load_default()
        report["config"] = {"error": str(e), "note": "the coordinator does nothing until this is fixed"}
    sch = cfg["schedule"]
    report["schedule"] = {
        name: {"cron": sch[f"{name}_cron"], "timezone": "UTC",
               "status": jobs.SKIPPED if cfg.get("paused") else jobs.SCHEDULED,
               "next_utc": layout.utc_stamp(schedule.next_fire(sch[f"{name}_cron"], now)),
               **({"note": "paused: the cron still fires but does no costly work"}
                  if cfg.get("paused") else {})}
        for name in TICK_KINDS
    }
    report["coordinator"] = (layout.read_json(lay.coordinator_record) or {}).get("last")
    verdicts = {}
    report["jobs"] = []
    for rec in jobs.all_records(lay):
        v = jobs.classify(rec, remote_state=remote_state(rec.get("call_id")), now=now)
        verdicts[rec["key"]] = {**v, "failure_reason": v.get("failure_reason")}
        report["jobs"].append({"key": rec["key"], "status": v["status"], "call_id": rec.get("call_id"),
                               "attempt": rec.get("attempt"), "updated_utc": rec.get("updated_utc"),
                               "heartbeat_age_s": v["heartbeat_age_s"],
                               "failure_reason": v.get("failure_reason"), "note": rec.get("note"),
                               "notes": v["notes"]})
    champion = promotion.read_champion(lay)
    report["champion"] = champion and {k: champion.get(k) for k in (
        "generation", "candidate_id", "snapshot_id", "mode", "promoted_utc", "promoted_by", "previous")}
    promoted = promotion.promoted_candidates(lay)
    report["snapshots"] = []
    for sid in lay.snapshot_ids():
        rec = layout.read_json(lay.snapshot_record(sid)) if lay.snapshot_record(sid).exists() else None
        report["snapshots"].append({
            "snapshot_id": sid, "status": jobs.COMPLETED if rec else jobs.UNKNOWN,
            "created_utc": (rec or {}).get("created_utc"),
            "parent": ((rec or {}).get("parents") or {}).get("snapshot_id"),
            "content_sha256": ((rec or {}).get("content") or {}).get("sha256")})
    for key, v in verdicts.items():
        if key.startswith("snapshot:") and not lay.snapshot_dir(key.split(":", 1)[1]).exists():
            report["snapshots"].append({"snapshot_id": key.split(":", 1)[1], "status": v["status"],
                                        "failure_reason": v.get("failure_reason")})
    report["candidates"] = [
        candidate_status(lay, cid, verdicts, promoted, (champion or {}).get("candidate_id"))
        for cid in lay.candidate_ids()]
    return report


# --- retention ---------------------------------------------------------------------


def retention_plan(lay: layout.Layout, cfg: dict, *, live_targets: set[str] | None = None) -> dict:
    """What ``prune`` may delete. Never a champion (current or historical), its
    snapshot, anything with a live job, or anything inside the keep windows."""
    live_targets = set(live_targets or ())
    keep_c = set(sorted(lay.candidate_ids())[-cfg["retention"]["keep_candidates"]:])
    history = promotion.champion_history(lay)
    keep_c |= {h.get("candidate_id") for h in history if h}
    keep_c |= {t for t in live_targets if t.startswith("c-")}
    keep_s = set(sorted(lay.snapshot_ids())[-cfg["retention"]["keep_snapshots"]:])
    keep_s |= {h.get("snapshot_id") for h in history if h}
    keep_s |= {t for t in live_targets if t.startswith("s-")}
    for cid in keep_c:
        rec = pcand.read_record(lay, cid) if lay.candidate_dir(cid).exists() else None
        if rec:
            keep_s.add((rec.get("parents") or {}).get("snapshot_id"))
    return {"delete_candidates": sorted(set(lay.candidate_ids()) - keep_c),
            "delete_snapshots": sorted(set(lay.snapshot_ids()) - keep_s),
            "keep_candidates": sorted(keep_c & set(lay.candidate_ids())),
            "keep_snapshots": sorted(keep_s & set(lay.snapshot_ids()))}
