"""Candidate identity, records and provenance.

A candidate is one training run on one snapshot, plus its evaluation and its
decision. Its directory *is* a run directory in the sense ``naigos.rl.runmeta``
already understands (``run.json``, ``manifest.json``, ``history.json``,
``perf.json``, ``theatre.json``, ``ckpt_NNNNNN.pkl``), so the existing
verification, status and resume logic applies unchanged, and three pipeline
records sit beside them:

    candidate.json    write-once, before training: what is being attempted and from what
    evaluation.json   write-once, after held-out evaluation
    decision.json     write-once, last. Once it exists nothing writes here again.

Idempotency. The candidate ID is derived from a key over exactly the inputs
that determine the training -- the stage's config digest, the snapshot ID and
its content hash, and the code commit -- plus the snapshot's window. The same
inputs always name the same directory, so the coordinator's "has this been
done?" is "does this directory exist?", and a new snapshot is what makes a new
candidate possible at all.
"""

from __future__ import annotations

from pathlib import Path

from ..rl import runmeta
from . import config as pcfg
from . import evaluation, layout
from . import snapshot as psnap

CANDIDATE_SCHEMA = 1


def idempotency_key(*, kind: str, stage_digest: str, snapshot_rec: dict, code: dict) -> str:
    return layout.digest({
        "stage": kind,
        "stage_digest": stage_digest,
        "snapshot_id": snapshot_rec.get("snapshot_id"),
        "snapshot_content": (snapshot_rec.get("content") or {}).get("sha256"),
        "commit": (code or {}).get("commit"),
    })


def plan(cfg: dict, *, kind: str, snapshot_rec: dict, code: dict, config_version: int,
         champion: dict | None, now=None) -> dict:
    """Everything needed to create a candidate: its ID, record and run metadata. Pure."""
    if kind not in layout.CANDIDATE_KINDS:
        raise ValueError(f"unknown candidate kind {kind!r}")
    tkind = "weekly" if kind == "weekly" else "nightly"
    tcfg = cfg["training"][tkind]
    sdig = pcfg.stage_digest(cfg, kind)
    key = idempotency_key(kind=kind, stage_digest=sdig, snapshot_rec=snapshot_rec, code=code)
    cid = layout.candidate_id(snapshot_rec["window"], kind, key)
    prof = runmeta.resolve_profile(tcfg["profile"], **tcfg["overrides"])
    heldout = evaluation.assert_heldout(evaluation.heldout_plan(cfg, kind), tcfg["seed"])
    meta = runmeta.build_metadata(prof, run_name=cid, seed=tcfg["seed"], synthetic=False,
                                  aoi=cfg["aoi"], use_cbf=cfg["training"]["use_cbf"], code=code,
                                  launcher="modal", now=now)
    # Outside runmeta's identity keys on purpose: the pipeline's provenance rides
    # along in run.json without changing what makes two runs "the same run".
    meta["pipeline"] = {"candidate_id": cid, "snapshot_id": snapshot_rec["snapshot_id"],
                        "config_digest": pcfg.config_digest(cfg), "stage_digest": sdig,
                        "heldout_seeds": heldout["seeds"]}
    record = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": cid,
        "kind": kind,
        "created_utc": layout.utc_stamp(now),
        "idempotency_key": key,
        "window": snapshot_rec["window"],
        "code": code,
        "config_version": config_version,
        "config_digest": pcfg.config_digest(cfg),
        "stage_digest": sdig,
        "aoi": cfg["aoi"],
        "parents": {
            "snapshot_id": snapshot_rec["snapshot_id"],
            "snapshot_content_sha256": (snapshot_rec.get("content") or {}).get("sha256"),
            "champion_generation": (champion or {}).get("generation"),
            "champion_candidate_id": (champion or {}).get("candidate_id"),
        },
        "training": {"profile": prof.as_dict(), "seed": tcfg["seed"],
                     "use_cbf": cfg["training"]["use_cbf"]},
        "evaluation": heldout,
        # The exact run.json the trainer writes, fixed at creation, so a retry
        # or resume re-derives nothing from whatever config is current then.
        "run_meta": meta,
    }
    return {"candidate_id": cid, "idempotency_key": key, "record": record, "meta": meta,
            "profile": prof}


def read_record(lay: layout.Layout, cid: str) -> dict | None:
    return layout.read_json(lay.candidate_dir(cid) / layout.CANDIDATE_RECORD)


def final_checkpoint(cand_dir: Path, iterations: int) -> Path | None:
    p = cand_dir / f"ckpt_{int(iterations):06d}.pkl"
    return p if p.is_file() else None


def verify(lay: layout.Layout, cid: str) -> list[str]:
    """Provenance problems with a trained candidate. Empty means it may be evaluated."""
    try:
        d = lay.candidate_dir(cid)
    except layout.LayoutError as e:
        return [str(e)]
    problems: list[str] = []
    try:
        rec = read_record(lay, cid)
    except (ValueError, OSError) as e:
        return [f"candidate.json does not parse: {e}"]
    if rec is None:
        return [f"{cid}: no candidate.json"]
    if rec.get("candidate_id") != cid:
        problems.append(f"candidate.json names {rec.get('candidate_id')!r}, directory is {cid!r}")
    if (rec.get("code") or {}).get("dirty") is not False:
        problems.append("candidate code is dirty or unknown: not reproducible from its commit")
    # The run itself: complete, not truncated, on the device it claims, with its
    # theatre provenance -- the checks `modal_runs.py verify` applies.
    problems.extend(f"run: {p}" for p in runmeta.verify_run_dir(d))
    meta = runmeta.read_metadata(d) if (d / runmeta.META_FILENAME).exists() else None
    sid = (rec.get("parents") or {}).get("snapshot_id")
    if meta is not None and (meta.get("pipeline") or {}).get("snapshot_id") != sid:
        problems.append("run.json and candidate.json disagree about the snapshot")
    if meta is not None and (meta.get("code") or {}).get("commit") != (rec.get("code") or {}).get("commit"):
        problems.append("run.json and candidate.json disagree about the code commit")
    snap_problems = psnap.verify_completed(lay, sid, aoi_name=rec.get("aoi")) if sid else ["no parent snapshot"]
    problems.extend(f"snapshot {sid}: {p}" for p in snap_problems)
    snap_rec = layout.read_json(lay.snapshot_record(sid)) if sid and not snap_problems else None
    if snap_rec and (snap_rec.get("content") or {}).get("sha256") != \
            (rec.get("parents") or {}).get("snapshot_content_sha256"):
        problems.append("the snapshot's content hash is not the one this candidate trained on")
    iters = (((rec.get("training") or {}).get("profile")) or {}).get("iterations")
    if iters and final_checkpoint(d, iters) is None:
        problems.append(f"no final checkpoint ckpt_{int(iters):06d}.pkl")
    return problems
