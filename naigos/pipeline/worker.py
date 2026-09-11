"""The work each scheduled stage does, independent of Modal.

Every function here takes a ``Services`` bundle -- the layout, the lease
manager, the deployed code identity, a way to spawn a deployed Function, a way
to ask Modal about a call, and the Volume's commit/reload -- so the whole
lifecycle is exercised in tests with fakes, and ``naigos/rl/modal_pipeline.py``
is left binding real implementations to them.

The rules every runner follows:

  * **Adopt, don't assume, the lease.** The coordinator acquires a stage's lease
    before spawning and passes its token. The worker verifies the token is
    still current before writing anything; a lease that was cleared or
    reclaimed in the meantime means another writer may exist, and the worker
    stops.
  * **Heartbeat from a thread** for the whole stage (``LeaseManager.keep_alive``).
  * **Record every exit.** The job record ends ``completed``, ``failed`` (with a
    redacted reason), ``rejected`` or ``inconclusive``; nothing ends silently.
  * **Never retry expensive work on its own.** A failed training job stays
    failed until an operator resumes it (``scripts/pipeline.py retry``), which
    continues from the run's last valid checkpoint.
  * **The champion pointer is moved only by ``promote``/``rollback`` here**, both
    under the ``champion`` lease, both re-validating before the pointer moves.
"""

from __future__ import annotations

import contextlib
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..research import roots
from ..rl import runmeta
from . import candidate as pcand
from . import config as pcfg
from . import evaluation as peval
from . import jobs, layout, leases, promotion
from . import snapshot as psnap

CHAMPION_LEASE = "champion"


@dataclass
class Services:
    lay: layout.Layout
    leases: leases.LeaseManager
    code: dict
    spawn: Callable[[str, dict], str] = lambda fn, payload: (_ for _ in ()).throw(
        RuntimeError("no spawner configured"))
    remote_state: Callable[[str | None], object] = lambda call_id: None
    smoke_marker: Callable[[], dict | None] = lambda: None
    now: Callable[[], datetime] = layout.utc_now
    commit: Callable[[], None] = lambda: None
    reload: Callable[[], None] = lambda: None
    call_id: Callable[[], str | None] = lambda: None
    log: Callable[[str], None] = field(default=lambda msg: print(runmeta.redact(msg), flush=True))


def _set_job(svc: Services, key: str, status: str | None = None, **fields) -> dict:
    rec = jobs.read(svc.lay, key)
    if rec is None:
        raise RuntimeError(f"no job record {key}")
    rec = jobs.update(rec, status=status, now=svc.now(), **fields)
    jobs.write(svc.lay, rec)
    svc.commit()
    return rec


def _adopt(svc: Services, payload: dict) -> leases.Lease:
    lease = leases.Lease(**payload["lease"])
    svc.leases.heartbeat(lease, call_id=svc.call_id() or lease.call_id)  # raises LeaseLost
    return lease


@contextlib.contextmanager
def _stage(svc: Services, key: str, payload: dict):
    """Adopt the lease, mark running, heartbeat, and record any exit as failed."""
    svc.reload()
    try:
        lease = _adopt(svc, payload)
    except leases.LeaseError as e:
        with contextlib.suppress(Exception):
            _set_job(svc, key, jobs.FAILED, failure_reason=f"lease not held at start: {e}")
        raise
    try:
        _set_job(svc, key, jobs.RUNNING, call_id=svc.call_id(),
                 started_utc=layout.utc_stamp(svc.now()))
        with svc.leases.keep_alive(lease) as ka:
            yield lease
        if ka.get("lost"):
            raise leases.LeaseLost(ka["lost"])
    except BaseException as e:  # noqa: BLE001 - every exit path is recorded
        svc.log(runmeta.redact(traceback.format_exc()))
        with contextlib.suppress(Exception):
            _set_job(svc, key, jobs.FAILED, failure_reason=f"{type(e).__name__}: {e}",
                     finished_utc=layout.utc_stamp(svc.now()))
        raise
    finally:
        svc.leases.release(lease)
        svc.commit()


# --- snapshot ----------------------------------------------------------------------


def run_snapshot(svc: Services, payload: dict, *, research_runner=psnap.run_research_subprocess) -> dict:
    sid = layout.validate_snapshot_id(payload["snapshot_id"])
    key = jobs.job_key("snapshot", sid)
    cfg, _ = pcfg.load_effective(svc.lay)
    with _stage(svc, key, payload):
        rec = psnap.build(
            svc.lay, sid=sid, attempt=int(payload.get("attempt") or 1), cfg=cfg,
            window=payload["window"], idempotency_key=payload["idempotency_key"],
            config_digest=payload["config_digest"], stage_digest=payload["stage_digest"],
            code=svc.code, research_runner=research_runner,
            on_progress=lambda m: svc.log(f"[snapshot {sid}] {m}"))
        _set_job(svc, key, jobs.COMPLETED, finished_utc=layout.utc_stamp(svc.now()),
                 content_sha256=rec["content"]["sha256"])
    return rec


# --- training ----------------------------------------------------------------------


def run_training(svc: Services, payload: dict, *, trainer: Callable[..., dict],
                 dispatch_eval: Callable[[str], object] | None = None) -> dict:
    """Train one candidate on a local, re-hashed copy of its snapshot.

    ``trainer(out_dir=..., meta=..., profile=..., seed=..., aoi=..., use_cbf=...,
    resume=bool)`` runs ``naigos.rl.train.run`` (the Modal wrapper supplies it,
    with the manifest bookkeeping and Volume commits of ``modal_train.py``).
    """
    cid = layout.validate_candidate_id(payload["candidate_id"])
    key = jobs.job_key("train", cid)
    cand_dir = svc.lay.candidate_dir(cid)
    summary: dict = {}
    with _stage(svc, key, payload):
        rec = pcand.read_record(svc.lay, cid)
        if rec is None:
            raise RuntimeError(f"{cid} has no candidate.json")
        if (cand_dir / layout.DECISION_RECORD).exists():
            raise layout.ImmutableRecordError(f"{cid} already has a decision; it is never retrained")
        resume = bool(payload.get("resume"))
        on_disk = runmeta.read_metadata(cand_dir) if (cand_dir / runmeta.META_FILENAME).exists() else None
        if on_disk is not None:
            # A resume continues one experiment. A different commit is a
            # different candidate (it has a different idempotency key), so it
            # is refused here rather than spliced into this run's history.
            compat = runmeta.resume_compatibility(on_disk, rec["run_meta"])
            if not compat["ok"]:
                raise runmeta.ResumeRefused("; ".join(compat["blocking"]))
            if not resume and runmeta.checkpoint_paths(cand_dir):
                raise RuntimeError(f"{cid} already has checkpoints; only a resume may continue it")
        sid = rec["parents"]["snapshot_id"]
        with tempfile.TemporaryDirectory(prefix="naigos-snap-") as td:
            local = psnap.stage_local_copy(svc.lay, sid, Path(td) / sid)
            with roots.research_roots(cache_dir=local / psnap.CACHE_DIRNAME,
                                      components_dir=local / psnap.COMPONENTS_DIRNAME):
                summary = trainer(out_dir=cand_dir, meta=rec["run_meta"],
                                  profile=rec["training"]["profile"], seed=rec["training"]["seed"],
                                  aoi=rec["aoi"], use_cbf=rec["training"]["use_cbf"],
                                  resume=resume)
        problems = runmeta.verify_run_dir(cand_dir)
        if problems:
            raise RuntimeError("trained run does not verify: " + "; ".join(problems[:10]))
        _set_job(svc, key, jobs.COMPLETED, finished_utc=layout.utc_stamp(svc.now()))
    if dispatch_eval is not None:
        dispatch_eval(cid)
    return summary


# --- evaluation and decision -------------------------------------------------------


def _read_champion_for_eval(svc: Services) -> tuple[dict | None, Path | None, str | None]:
    pointer = promotion.read_champion(svc.lay)
    if pointer is None:
        return None, None, None
    try:
        path = promotion.champion_checkpoint_path(svc.lay, pointer)
    except layout.LayoutError as e:
        return pointer, None, str(e)
    if not path.is_file():
        return pointer, None, f"champion checkpoint {path.name} is missing"
    if layout.sha256_file(path) != (pointer.get("checkpoint") or {}).get("sha256"):
        return pointer, None, "champion checkpoint does not hash to the pointer's record"
    return pointer, path, None


def evaluate_candidate(svc: Services, cid: str, cfg: dict, *,
                       measure: Callable[..., dict]) -> dict:
    """Produce an evaluation record for ``cid``. Does not write it.

    ``measure(snapshot_root=..., plan=..., env_shape=..., candidate_ckpt=...,
    champion_ckpt=...)`` returns what ``evaluation.run_evaluation`` returns; the
    Modal wrapper binds the JAX implementation, tests bind a fake.
    """
    rec = pcand.read_record(svc.lay, cid)
    cand_dir = svc.lay.candidate_dir(cid)
    prof = rec["training"]["profile"]
    plan = peval.assert_heldout(rec["evaluation"], rec["training"]["seed"])
    ckpt = pcand.final_checkpoint(cand_dir, prof["iterations"])
    if ckpt is None:
        raise RuntimeError(f"{cid} has no final checkpoint")
    sid = rec["parents"]["snapshot_id"]
    snap_rec = layout.read_json(svc.lay.snapshot_record(sid))
    pointer, champ_path, champ_err = _read_champion_for_eval(svc)
    env_shape = {"n_blue": prof["n_blue"], "n_threat": prof["n_threat"], "cell_m": prof["cell_m"],
                 "red_level": plan["red_level"], "aoi": rec["aoi"]}
    with tempfile.TemporaryDirectory(prefix="naigos-eval-") as td:
        local = psnap.stage_local_copy(svc.lay, sid, Path(td) / sid)
        with roots.research_roots(cache_dir=local / psnap.CACHE_DIRNAME,
                                  components_dir=local / psnap.COMPONENTS_DIRNAME):
            measured = measure(snapshot_root=local, plan=plan, env_shape=env_shape,
                               candidate_ckpt=ckpt, champion_ckpt=champ_path)
    champion = None
    if pointer is not None:
        champion = {**pointer, "error": champ_err}
    cfg_version = int(cfg.get("version") or 0)
    return peval.build_record(candidate_id=cid, snapshot_rec=snap_rec, plan=plan,
                              env_shape=env_shape, checkpoint=peval.checkpoint_entry(ckpt),
                              measured=measured, champion=champion, code=svc.code,
                              config_digest=pcfg.config_digest(cfg), config_version=cfg_version)


def run_evaluation(svc: Services, payload: dict, *, measure: Callable[..., dict]) -> dict:
    cid = layout.validate_candidate_id(payload["candidate_id"])
    key = jobs.job_key("evaluate", cid)
    cand_dir = svc.lay.candidate_dir(cid)
    cfg, _ = pcfg.load_effective(svc.lay)
    with _stage(svc, key, payload):
        existing = layout.read_json(cand_dir / layout.DECISION_RECORD)
        if existing is not None:
            _set_job(svc, key, jobs.SKIPPED, note="decision already recorded")
            return existing
        problems = pcand.verify(svc.lay, cid)
        if problems:
            # Deterministic: a candidate whose provenance does not verify is
            # rejected on that ground alone, and the record says why.
            ev = {"schema": peval.EVALUATION_SCHEMA, "candidate_id": cid,
                  "created_utc": layout.utc_stamp(svc.now()), "code": svc.code,
                  "error": "provenance did not verify; not evaluated", "problems": problems}
        else:
            ev = evaluate_candidate(svc, cid, cfg, measure=measure)
        ev_path = cand_dir / layout.EVALUATION_RECORD
        if not ev_path.exists():
            layout.write_once_json(ev_path, ev)
        ev = layout.read_json(ev_path)
        decision = promotion.build_decision(
            candidate_id=cid, evaluation=ev, evaluation_sha256=layout.sha256_file(ev_path),
            gates=cfg["gates"], config_version=int(cfg["version"]),
            config_digest=pcfg.config_digest(cfg), auto_promote=bool(cfg["auto_promote"]),
            code=svc.code, provenance_problems=problems)
        layout.write_once_json(cand_dir / layout.DECISION_RECORD, decision)
        svc.commit()
        status = {promotion.ELIGIBLE: jobs.COMPLETED, promotion.REJECTED: jobs.REJECTED,
                  promotion.INCONCLUSIVE: jobs.INCONCLUSIVE}[decision["outcome"]]
        note = None
        if decision["action"] == "auto_promote":
            try:
                pointer = promote(svc, cid, actor="auto-promote", mode="auto", rerun_measure=None)
                status, note = jobs.PROMOTED, f"auto-promoted to generation {pointer['generation']}"
            except (promotion.PromotionRefused, leases.LeaseError) as e:
                note = f"auto-promotion refused: {e} {getattr(e, 'problems', '')}"
        elif decision["action"] == "shadow":
            note = "eligible; shadow mode -- promote explicitly with scripts/pipeline.py promote"
        _set_job(svc, key, status, finished_utc=layout.utc_stamp(svc.now()),
                 outcome=decision["outcome"], note=note)
    return decision


# --- promotion ---------------------------------------------------------------------


def revalidate(svc: Services, cid: str, cfg: dict, *,
               rerun_measure: Callable[..., dict] | None) -> dict:
    """Everything that must still be true for ``cid`` to become champion now."""
    cand_dir = svc.lay.candidate_dir(cid)
    problems = list(pcand.verify(svc.lay, cid))
    decision = layout.read_json(cand_dir / layout.DECISION_RECORD)
    ev_path = cand_dir / layout.EVALUATION_RECORD
    ev = layout.read_json(ev_path)
    if decision is None:
        problems.append("no decision.json: the candidate was never evaluated")
    elif decision.get("outcome") != promotion.ELIGIBLE:
        problems.append(f"decision outcome is {decision.get('outcome')!r}, not eligible")
    if ev is None:
        problems.append("no evaluation.json")
    elif decision is not None and layout.sha256_file(ev_path) != decision.get("evaluation_sha256"):
        problems.append("evaluation.json does not hash to the value its decision recorded")
    rec = pcand.read_record(svc.lay, cid) or {}
    iters = (((rec.get("training") or {}).get("profile")) or {}).get("iterations")
    ckpt = pcand.final_checkpoint(cand_dir, iters) if iters else None
    entry = peval.checkpoint_entry(ckpt) if ckpt else None
    if entry is None:
        problems.append("final checkpoint missing")
    elif ev is not None and entry["sha256"] != (ev.get("checkpoint") or {}).get("sha256"):
        problems.append("final checkpoint does not hash to the evaluated checkpoint")
    if rerun_measure is not None and not problems:
        fresh = evaluate_candidate(svc, cid, cfg, measure=rerun_measure)
        problems.extend(f"reproduction: {p}" for p in peval.compare_reproduction(
            ev, fresh, cfg["gates"]["max_reproduction_error"]))
        verdict = promotion.decide(fresh, cfg["gates"], provenance_problems=[])
        if verdict["outcome"] != promotion.ELIGIBLE:
            problems.extend(f"gate {c['gate']}: {c['status']} ({c['detail']})"
                            for c in verdict["checks"] if c["status"] != promotion.PASS)
    return {"problems": problems, "snapshot_id": rec.get("parents", {}).get("snapshot_id"),
            "checkpoint": entry, "decision_sha256": layout.sha256_file(cand_dir / layout.DECISION_RECORD)
            if decision is not None else None,
            "evaluation_sha256": layout.sha256_file(ev_path) if ev is not None else None,
            "code": rec.get("code")}


def _provenance_only(svc: Services, cid: str) -> dict:
    cand_dir = svc.lay.candidate_dir(cid)
    rec = pcand.read_record(svc.lay, cid) or {}
    problems = list(pcand.verify(svc.lay, cid))
    iters = (((rec.get("training") or {}).get("profile")) or {}).get("iterations")
    ckpt = pcand.final_checkpoint(cand_dir, iters) if iters else None
    if ckpt is None:
        problems.append("final checkpoint missing")
    return {"problems": problems, "checkpoint": peval.checkpoint_entry(ckpt) if ckpt else {},
            "code": rec.get("code")}


@contextlib.contextmanager
def _champion_lease(svc: Services, owner: str):
    lease = svc.leases.acquire(CHAMPION_LEASE, owner)  # LeaseHeld / LeaseAmbiguous propagate
    try:
        yield lease
    finally:
        svc.leases.release(lease)
        svc.commit()


def promote(svc: Services, cid: str, *, actor: str, mode: str = "manual",
            rerun_measure: Callable[..., dict] | None, expected_generation: int | None = None,
            note: str | None = None) -> dict:
    """Re-validate and publish, under the champion lease. Raises without moving the pointer."""
    layout.validate_candidate_id(cid)
    svc.reload()
    cfg, _ = pcfg.load_effective(svc.lay)
    with _champion_lease(svc, f"promote:{cid}:{actor}"):
        pointer = promotion.promote(
            svc.lay, candidate_id=cid, actor=actor, mode=mode,
            expected_generation=expected_generation, note=note,
            revalidate=lambda c: revalidate(svc, c, cfg, rerun_measure=rerun_measure))
    svc.log(f"[champion] generation {pointer['generation']} -> {cid} ({mode} by {actor})")
    return pointer


def rollback(svc: Services, generation: int, *, actor: str, note: str | None = None) -> dict:
    svc.reload()
    with _champion_lease(svc, f"rollback:g{generation}:{actor}"):
        pointer = promotion.rollback(svc.lay, to_generation=generation, actor=actor, note=note,
                                     revalidate_provenance=lambda c: _provenance_only(svc, c))
    svc.log(f"[champion] generation {pointer['generation']} -> {pointer['candidate_id']} "
            f"(rollback to g{generation} by {actor})")
    return pointer
