"""The scheduled pipeline end to end, offline: cron tick -> snapshot -> train -> eval -> decision.

Modal is replaced by a queue (`FakeModal`), training by a writer of a verifiable
run directory, and evaluation by deterministic metrics -- everything else,
including leases, idempotency, gates and the champion pointer, is the real code.
"""

from __future__ import annotations

import json
import pickle

import pytest

from _pipeline_fixtures import Harness, fake_measure, fake_trainer
from naigos.pipeline import coordinator, evaluation, jobs, layout, leases, promotion, worker
from naigos.pipeline import snapshot as psnap


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def _first_candidate(h: Harness) -> str:
    h.tick("snapshot")
    h.drain()
    h.clock.advance(hours=2)
    h.tick("nightly")
    h.drain()
    (cid,) = h.lay.candidate_ids()
    return cid


def _new_day_candidate(h: Harness, quality: float) -> str:
    before = set(h.lay.candidate_ids())
    h.clock.advance(days=1)
    h.tick("snapshot")
    h.drain()
    h.qualities.append(quality)
    h.tick("nightly")
    h.drain()
    (cid,) = set(h.lay.candidate_ids()) - before
    return cid


def _decision(h, cid):
    return json.loads((h.lay.candidate_dir(cid) / "decision.json").read_text())


# --- the happy path, in shadow mode -----------------------------------------------


def test_scheduled_run_produces_a_shadow_decision_and_leaves_no_champion(h):
    cid = _first_candidate(h)
    d = _decision(h, cid)
    assert d["outcome"] == promotion.ELIGIBLE
    assert d["action"] == "shadow"
    assert h.champion_bytes() is None  # auto_promote defaults to false
    rec = json.loads((h.lay.candidate_dir(cid) / "candidate.json").read_text())
    (sid,) = h.lay.snapshot_ids()
    assert rec["parents"]["snapshot_id"] == sid
    assert rec["code"] == h.svc.code and rec["config_digest"] and rec["created_utc"].endswith("Z")
    run = json.loads((h.lay.candidate_dir(cid) / "run.json").read_text())
    assert run["pipeline"]["snapshot_id"] == sid and run["run_name"] == cid
    ev = json.loads((h.lay.candidate_dir(cid) / "evaluation.json").read_text())
    assert ev["snapshot_id"] == sid and ev["checkpoint"]["sha256"]
    assert ev["heldout"]["seeds"] == h.set_config({})["evaluation"]["heldout_seeds"]
    assert not set(ev["heldout"]["seeds"]) & set(ev["heldout"]["training_seeds"])
    status = {c["candidate_id"]: c for c in coordinator.status_report(h.lay)["candidates"]}
    assert status[cid]["status"] == jobs.COMPLETED and status[cid]["action"] == "shadow"
    assert [c["fn"] for c in h.modal.spawned()] == [
        coordinator.SNAPSHOT_FN, coordinator.TRAIN_FN, coordinator.EVALUATE_FN]


def test_training_reads_a_local_copy_of_the_snapshot_never_the_volume(h):
    seen: list = []
    h.handlers_train = None
    h.tick("snapshot")
    h.drain()
    h.tick("nightly")
    call = h.modal.calls[h.modal.queue.pop(0)]
    worker.run_training(h.svc, call["payload"], trainer=fake_trainer([0.8], seen_roots=seen))
    assert "naigos-snap-" in seen[0] and str(h.lay.root) not in seen[0]


def test_evaluation_uses_the_recorded_heldout_seeds(h):
    seen: list = []
    h.measure = fake_measure(seen=seen)
    _first_candidate(h)
    plan = seen[0]["plan"]
    cfg = json.loads(h.lay.config_current.read_text())
    assert plan["seeds"] == cfg["evaluation"]["heldout_seeds"]
    assert plan["worlds_per_seed"] == cfg["evaluation"]["worlds_per_seed"]
    assert "naigos-eval-" in seen[0]["snapshot_root"]


def test_evaluation_refuses_heldout_seeds_that_overlap_training():
    plan = {"seeds": [3, 4], "training_seeds": [0, 1]}
    assert evaluation.assert_heldout(plan, 2)["training_seeds"] == [0, 1, 2]
    with pytest.raises(evaluation.HeldOutOverlap):
        evaluation.assert_heldout(plan, 4)
    with pytest.raises(evaluation.HeldOutOverlap):
        evaluation.assert_heldout({"seeds": [], "training_seeds": []}, 0)


def test_a_candidate_whose_recorded_seeds_overlap_is_never_decided(h):
    h.tick("snapshot")
    h.drain()
    h.tick("nightly")
    (cid,) = h.lay.candidate_ids()
    # Tamper with the candidate's own record so its evaluation plan overlaps training.
    path = h.lay.candidate_dir(cid) / "candidate.json"
    rec = json.loads(path.read_text())
    rec["evaluation"]["seeds"] = [rec["training"]["seed"]]
    path.write_text(json.dumps(rec))
    h.drain()
    assert not (h.lay.candidate_dir(cid) / "decision.json").exists()
    v = jobs.read(h.lay, jobs.job_key("evaluate", cid))
    assert v["status"] == jobs.FAILED and "HeldOutOverlap" in v["failure_reason"]
    assert h.champion_bytes() is None


# --- idempotency -----------------------------------------------------------------


def test_a_duplicate_cron_delivery_spawns_nothing_twice(h):
    a = h.tick("snapshot")
    b = h.tick("snapshot")  # Modal delivered the same schedule twice
    assert a["status"] == jobs.QUEUED and b["status"] == jobs.SKIPPED
    assert len(h.modal.spawned(coordinator.SNAPSHOT_FN)) == 1
    h.drain()
    h.clock.advance(hours=5)  # an old invocation still thinks it is 06:00's window
    assert h.tick("snapshot")["status"] == jobs.SKIPPED
    assert len(h.lay.snapshot_ids()) == 1


def test_overlapping_nightly_ticks_create_one_candidate(h):
    h.tick("snapshot")
    h.drain()
    first = h.tick("nightly")
    second = h.tick("nightly")
    assert first["status"] == jobs.QUEUED and second["status"] == jobs.SKIPPED
    assert len(h.modal.spawned(coordinator.TRAIN_FN)) == 1


def test_nightly_waits_for_a_new_snapshot(h):
    cid = _first_candidate(h)
    h.clock.advance(days=1)
    out = h.tick("nightly")  # no new snapshot today
    assert out["status"] == jobs.SKIPPED and "waiting for a new snapshot" in out["reason"]
    assert h.lay.candidate_ids() == [cid]


def test_candidate_ids_are_deterministic_in_their_inputs(h):
    cid = _first_candidate(h)
    rec = json.loads((h.lay.candidate_dir(cid) / "candidate.json").read_text())
    assert cid.endswith(rec["idempotency_key"][:12])
    assert layout.kind_of_candidate(cid) == "nightly"


def test_the_daily_budget_is_enforced(tmp_path):
    h = Harness(tmp_path)
    cfg = json.loads(h.lay.config_current.read_text())
    h.set_config({"training": {**cfg["training"], "max_candidates_per_day": 0}})
    h.tick("snapshot")
    h.drain()
    out = h.tick("nightly")
    assert out["status"] == jobs.SKIPPED and "budget" in out["reason"]


def test_the_smoke_gate_is_respected(tmp_path):
    h = Harness(tmp_path)
    cfg = json.loads(h.lay.config_current.read_text())
    h.set_config({"training": {**cfg["training"], "require_smoke": True}})
    h.tick("snapshot")
    h.drain()
    out = h.tick("nightly")
    assert out["status"] == jobs.SKIPPED and "smoke gate" in out["reason"]
    h.svc.smoke_marker = lambda: {"commit": h.svc.code["commit"]}
    assert h.tick("nightly")["status"] == jobs.QUEUED


def test_weekly_runs_only_if_the_prior_jobs_passed(tmp_path):
    h = Harness(tmp_path)
    h.qualities.append(0.2)  # the nightly candidate will fail the survival gate
    cid = _first_candidate(h)
    assert _decision(h, cid)["outcome"] == promotion.REJECTED
    out = h.tick("weekly")
    assert out["status"] == jobs.SKIPPED and "prior jobs did not pass" in out["reason"]

    h2 = Harness(tmp_path / "ok")
    _first_candidate(h2)
    assert h2.tick("weekly")["status"] == jobs.QUEUED
    h2.drain()
    kinds = sorted(layout.kind_of_candidate(c) for c in h2.lay.candidate_ids())
    assert kinds == ["nightly", "weekly"]


# --- pause, config, code identity --------------------------------------------------


def test_pause_is_a_remote_gate_checked_before_costly_work(h):
    h.set_config({"paused": True, "pause_reason": "budget review"})
    for kind in coordinator.TICK_KINDS:
        out = h.tick(kind)
        assert out["status"] == jobs.SKIPPED and "budget review" in out["reason"]
    assert h.modal.spawned() == []
    report = coordinator.status_report(h.lay)
    assert report["config"]["paused"] is True
    assert all(s["status"] == jobs.SKIPPED and "paused" in s["note"] for s in report["schedule"].values())
    h.set_config({"paused": False, "pause_reason": None})
    assert h.tick("snapshot")["status"] == jobs.QUEUED


def test_an_invalid_volume_config_fails_closed(h):
    h.lay.config_current.write_text('{"schema": 1, "paused": "no"}')
    out = h.tick("snapshot")
    assert out["status"] == jobs.FAILED and "config invalid" in out["reason"]
    assert h.modal.spawned() == []


def test_a_dirty_deployment_does_no_costly_work(tmp_path):
    h = Harness(tmp_path, code={"commit": "abc", "dirty": True})
    out = h.tick("snapshot")
    assert out["status"] == jobs.SKIPPED and "clean commit" in out["reason"]
    assert h.modal.spawned() == []


def test_every_tick_leaves_a_coordinator_heartbeat(h):
    h.tick("nightly")
    rec = json.loads(h.lay.coordinator_record.read_text())
    assert rec["last"]["kind"] == "nightly" and rec["last"]["reason"]
    assert rec["heartbeat_utc"] == layout.utc_stamp(h.clock())


# --- failure and recovery ------------------------------------------------------------


def test_a_failed_training_is_not_retried_automatically(h):
    h.tick("snapshot")
    h.drain()
    h.trainer_fail = True
    h.tick("nightly")
    h.drain()
    (cid,) = h.lay.candidate_ids()
    rec = jobs.read(h.lay, jobs.job_key("train", cid))
    assert rec["status"] == jobs.FAILED and "out of memory" in rec["failure_reason"]
    out = h.tick("nightly")
    assert out["status"] == jobs.SKIPPED and "not retried automatically" in out["reason"]
    assert len(h.modal.spawned(coordinator.TRAIN_FN)) == 1
    # The operator retries explicitly; this time it trains.
    h.trainer_fail = False
    assert coordinator.retry(h.svc, cid)["status"] == jobs.QUEUED
    h.drain()
    assert _decision(h, cid)["outcome"] == promotion.ELIGIBLE
    assert jobs.read(h.lay, jobs.job_key("train", cid))["attempt"] == 2


def test_a_worker_killed_without_recording_is_reconciled_as_lost(h):
    h.tick("snapshot")
    call_id = h.modal.queue.pop(0)
    h.modal.calls[call_id]["state"] = "FunctionTimeoutError"  # never ran a line
    out = h.tick("nightly")
    (sid,) = [r["target"] for r in jobs.all_records(h.lay) if r["stage"] == "snapshot"]
    rec = jobs.read(h.lay, jobs.job_key("snapshot", sid))
    assert rec["status"] == jobs.FAILED and "lost" in rec["failure_reason"]
    assert any(a["status"] == jobs.FAILED for a in out["actions"])


def test_an_ambiguous_lease_fails_closed(h):
    h.tick("snapshot")
    h.drain()
    cfg = json.loads(h.lay.config_current.read_text())
    from naigos.pipeline import candidate as pcand
    sid = h.lay.snapshot_ids()[0]
    p = pcand.plan(cfg | {"schedule": {}}, kind="nightly",
                   snapshot_rec=json.loads(h.lay.snapshot_record(sid).read_text()),
                   code=h.svc.code, config_version=cfg["version"], champion=None)
    h.leases.acquire(f"candidate:{p['candidate_id']}", "a coordinator that died")
    h.clock.advance(hours=1)
    out = h.tick("nightly")
    assert out["status"] == jobs.FAILED and "ambiguous lease" in out["reason"]
    assert h.modal.spawned(coordinator.TRAIN_FN) == []


def test_no_concurrent_writers_for_a_candidate(h):
    h.tick("snapshot")
    h.drain()
    h.tick("nightly")
    call = h.modal.calls[h.modal.queue.pop(0)]
    # A second worker handed a stale copy of the lease cannot start.
    stolen = dict(call["payload"], lease={**call["payload"]["lease"], "token": "not-the-token"})
    with pytest.raises(leases.LeaseLost):
        worker.run_training(h.svc, stolen, trainer=fake_trainer([0.8]))
    # While the real worker holds it, nobody else can acquire it.
    cid = call["payload"]["candidate_id"]
    lease = leases.Lease(**call["payload"]["lease"])
    h.leases.heartbeat(lease, call_id="fc-live")
    with pytest.raises(leases.LeaseHeld):
        h.leases.acquire(f"candidate:{cid}", "intruder")


def test_a_decided_candidate_is_never_retrained_or_rewritten(h):
    cid = _first_candidate(h)
    before = {p.name: p.read_bytes() for p in h.lay.candidate_dir(cid).iterdir() if p.is_file()}
    out = coordinator.retry(h.svc, cid)
    assert out["status"] == jobs.SKIPPED and "immutable" in out["reason"]
    after = {p.name: p.read_bytes() for p in h.lay.candidate_dir(cid).iterdir() if p.is_file()}
    assert before == after


def test_provenance_failure_is_a_rejection_not_a_crash(h):
    h.tick("snapshot")
    h.drain()
    h.tick("nightly")
    h.drain(limit=1)  # train only
    (cid,) = h.lay.candidate_ids()
    ckpt = next(h.lay.candidate_dir(cid).glob("ckpt_*.pkl"))
    (h.lay.candidate_dir(cid) / "perf.json").write_text(json.dumps({"device": {"platform": "cpu"}}))
    h.drain()
    d = _decision(h, cid)
    assert d["outcome"] == promotion.REJECTED
    assert any(c["gate"] == "provenance" and c["status"] == "fail" for c in d["checks"])
    assert ckpt.exists()


# --- retention -------------------------------------------------------------------


def test_retention_never_deletes_a_champion_or_its_snapshot(tmp_path):
    h = Harness(tmp_path, config_changes={"auto_promote": True})
    first = _first_candidate(h)
    for q in (0.3, 0.3, 0.3):
        _new_day_candidate(h, q)
    cfg = json.loads(h.lay.config_current.read_text()) | {"retention": {"keep_snapshots": 2,
                                                                         "keep_candidates": 2}}
    plan = coordinator.retention_plan(h.lay, cfg)
    assert first in plan["keep_candidates"] and first not in plan["delete_candidates"]
    first_snap = json.loads((h.lay.candidate_dir(first) / "candidate.json").read_text())["parents"]["snapshot_id"]
    assert first_snap in plan["keep_snapshots"]
    assert plan["delete_candidates"] and plan["delete_snapshots"]
