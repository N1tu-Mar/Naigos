"""Operator surface: admin commands, remote-state classification, the CLI, the Modal binding.

Offline. The CLI is driven with `--volume-dir`, which runs the same `admin.handle`
the deployed app runs, against a directory built by the pipeline harness.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from _pipeline_fixtures import Harness, fake_measure
from naigos.pipeline import admin, coordinator, jobs, leases, promotion, worker

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts" / "pipeline.py"


def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), *args], cwd=REPO, capture_output=True,
                          text=True, timeout=120)


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def _run_to_decision(h, quality=None):
    if quality is not None:
        h.qualities.append(quality)
    h.tick("snapshot")
    h.drain()
    h.tick("nightly")
    h.drain()


# --- status distinguishes every state, from the remote record alone -------------------


def test_status_distinguishes_every_documented_state(tmp_path):
    h = Harness(tmp_path, config_changes={"auto_promote": True})
    seen = set()

    # promoted, completed (eligible shadow), rejected, inconclusive, failed, queued,
    # running, skipped and scheduled -- each produced by real pipeline behaviour.
    _run_to_decision(h)                                    # promoted (auto)
    h.set_config({"auto_promote": False})
    h.clock.advance(days=1)
    _run_to_decision(h, 0.95)                              # completed, shadow
    h.clock.advance(days=1)
    _run_to_decision(h, 0.3)                               # rejected
    h.measure = fake_measure(champion_error="shape mismatch")
    h.clock.advance(days=1)
    _run_to_decision(h, 0.9)                               # inconclusive
    h.measure = fake_measure()
    h.clock.advance(days=1)
    h.tick("snapshot")
    h.drain()
    h.trainer_fail = True
    h.tick("nightly")
    h.drain()                                              # failed
    h.trainer_fail = False
    h.clock.advance(days=1)
    h.tick("snapshot")                                     # queued (spawned, no container yet)
    call = h.modal.queue[0]
    report = coordinator.status_report(h.lay, remote_state=h.modal.remote_state, now=h.clock())
    for c in report["candidates"]:
        seen.add(c["status"])
    for j in report["jobs"]:
        seen.add(j["status"])
    for s in report["schedule"].values():
        seen.add(s["status"])
    # The worker's first act is to mark its record running; Modal says "not finished".
    key = jobs.job_key("snapshot", h.modal.calls[call]["payload"]["snapshot_id"])
    jobs.write(h.lay, jobs.update(jobs.read(h.lay, key), status=jobs.RUNNING, now=h.clock()))
    h.modal.calls[call]["state"] = "running"
    report = coordinator.status_report(h.lay, remote_state=h.modal.remote_state, now=h.clock())
    seen |= {j["status"] for j in report["jobs"]}
    h.set_config({"paused": True, "pause_reason": "x"})
    seen.add(h.tick("nightly")["status"])                  # skipped
    assert {"scheduled", "queued", "running", "completed", "failed", "skipped", "rejected",
            "inconclusive", "promoted"} <= seen, seen


# --- admin -------------------------------------------------------------------------------


def test_pause_and_resume_are_versioned_config_changes(h):
    out = admin.handle(h.svc, "pause", {"reason": "budget review", "actor": "op"})
    assert out["paused"] is True and "still fire" in out["note"]
    cfg = admin.handle(h.svc, "config-show", {})["config"]
    assert cfg["paused"] is True and cfg["pause_reason"] == "budget review"
    assert cfg["updated_by"] == "op"
    v = cfg["version"]
    assert admin.handle(h.svc, "resume", {"actor": "op"})["version"] == v + 1
    assert h.lay.config_version(v).exists() and h.lay.config_version(v + 1).exists()


def test_config_set_validates_before_publishing(h):
    before = h.lay.config_current.read_bytes()
    with pytest.raises(Exception):
        admin.handle(h.svc, "config-set", {"changes": {"gates": {"min_survival_rate": 2}}})
    with pytest.raises(Exception):
        admin.handle(h.svc, "config-set", {"changes": {"schedule": {}}})
    assert h.lay.config_current.read_bytes() == before
    cfg = admin.handle(h.svc, "config-show", {})["config"]
    gates = {**cfg["gates"], "min_survival_rate": 0.7}
    out = admin.handle(h.svc, "config-set", {"changes": {"gates": gates}, "actor": "op"})
    assert out["config"]["gates"]["min_survival_rate"] == 0.7


def test_unlock_refuses_a_live_holder_and_clears_a_dead_one(h):
    lease = h.leases.acquire("candidate:c-20260911-nightly-0123456789ab", "worker")
    h.leases.heartbeat(lease, call_id="fc-live")
    h.modal.calls["fc-live"] = {"state": "running"}
    with pytest.raises(admin.AdminError, match="live writer"):
        admin.handle(h.svc, "unlock", {"key": lease.key})
    h.modal.calls["fc-live"]["state"] = "failed"
    out = admin.handle(h.svc, "unlock", {"key": lease.key})
    assert out["cleared"] == lease.key and out["verdict"] == "dead"
    assert h.store.get(lease.key) is None


def test_inspect_every_kind_of_record(tmp_path):
    h = Harness(tmp_path, config_changes={"auto_promote": True})
    _run_to_decision(h)
    (sid,) = h.lay.snapshot_ids()
    (cid,) = h.lay.candidate_ids()
    s = admin.handle(h.svc, "inspect", {"id": sid})
    assert s["verify_problems"] == [] and s["snapshot"]["snapshot_id"] == sid
    c = admin.handle(h.svc, "inspect", {"id": cid})
    assert c["decision"]["outcome"] == "eligible" and c["checkpoints"]
    assert "run_meta" not in c["candidate"]
    assert admin.handle(h.svc, "inspect", {"id": "g1"})["generation"]["candidate_id"] == cid
    assert admin.handle(h.svc, "inspect", {"id": "champion"})["champion"]["generation"] == 1
    for bad in ("../../x", "c-../../x", "nonsense"):
        with pytest.raises(Exception):
            admin.handle(h.svc, "inspect", {"id": bad})


def test_prune_applies_the_retention_plan_only(tmp_path):
    h = Harness(tmp_path, config_changes={"auto_promote": True,
                                          "retention": {"keep_snapshots": 2, "keep_candidates": 2}})
    _run_to_decision(h)
    champion = promotion.read_champion(h.lay)["candidate_id"]
    for _ in range(3):
        h.clock.advance(days=1)
        _run_to_decision(h, 0.3)
    plan = admin.handle(h.svc, "prune-plan", {})
    removed = []
    import shutil

    out = admin.handle(h.svc, "prune", {}, remove_tree=lambda p: (removed.append(p), shutil.rmtree(p)))
    assert out["applied"] and len(removed) == len(plan["delete_candidates"]) + len(plan["delete_snapshots"])
    assert champion in h.lay.candidate_ids()
    assert worker.revalidate(h.svc, champion, json.loads(h.lay.config_current.read_text()) |
                             {"schedule": {}}, rerun_measure=None)["problems"] == []


def test_unknown_admin_command_is_refused(h):
    with pytest.raises(admin.AdminError):
        admin.handle(h.svc, "format-volume", {})


# --- the CLI, offline ----------------------------------------------------------------------


def test_cli_status_reads_the_remote_record_with_no_local_job_index(tmp_path):
    h = Harness(tmp_path)
    _run_to_decision(h)
    out = _cli("--volume-dir", str(tmp_path), "status")
    assert out.returncode == 0, out.stderr
    assert "pipeline: active" in out.stdout and "scheduled" in out.stdout
    assert "completed" in out.stdout and "shadow" in out.stdout
    js = json.loads(_cli("--json", "--volume-dir", str(tmp_path), "status").stdout)
    assert js["candidates"][0]["decision"] == "eligible"
    assert not (REPO / "runs" / ".pipeline").exists()


def test_cli_lists_and_inspects(tmp_path):
    h = Harness(tmp_path)
    _run_to_decision(h)
    (cid,) = h.lay.candidate_ids()
    assert cid in _cli("--volume-dir", str(tmp_path), "list-candidates").stdout
    assert h.lay.snapshot_ids()[0] in _cli("--volume-dir", str(tmp_path), "list-snapshots").stdout
    doc = json.loads(_cli("--volume-dir", str(tmp_path), "inspect", cid).stdout)
    assert doc["decision"]["action"] == "shadow"


def test_cli_refuses_writes_against_a_local_copy(tmp_path):
    Harness(tmp_path)
    for args in (["pause"], ["resume"], ["retry", "s-20260911-0123456789ab"], ["prune", "--apply"]):
        out = _cli("--volume-dir", str(tmp_path), *args)
        assert out.returncode != 0 and "read-only" in (out.stderr + out.stdout), args


def test_cli_unlock_requires_explicit_confirmation(tmp_path):
    Harness(tmp_path)
    out = _cli("--volume-dir", str(tmp_path), "unlock", "champion")
    assert out.returncode != 0 and "--yes" in out.stderr


def test_cli_without_modal_explains_how_to_deploy(tmp_path):
    out = _cli("status")
    if out.returncode == 0:
        pytest.skip("a deployed app answered; this environment has Modal credentials")
    assert "modal" in (out.stderr + out.stdout).lower()


# --- the Modal binding, statically (no Modal account) -------------------------------------


def _decorated_functions(src: str) -> dict:
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "function":
                    out[node.name] = ast.unparse(dec)
    return out


SRC = (REPO / "naigos" / "rl" / "modal_pipeline.py").read_text()


def test_the_binding_imports_without_modal_and_names_its_app():
    from naigos.rl import modal_pipeline as mp

    assert mp.APP_NAME == "naigos-pipeline" and mp.VOLUME_NAME == "naigos-runs"
    assert mp.MAX_RETRIES == 0


def test_every_function_has_zero_retries_and_the_workers_are_separate():
    fns = _decorated_functions(SRC)
    assert {"snapshot_tick", "nightly_tick", "weekly_tick", "snapshot_worker", "train_candidate",
            "evaluate_candidate", "promote_candidate", "rollback_champion", "admin"} <= set(fns)
    for name, dec in fns.items():
        assert "retries=MAX_RETRIES" in dec or "**_tick_kw" in dec or "**_eval_kw" in dec, name
    assert "retries=MAX_RETRIES" in SRC.split("_tick_kw = dict(")[1].split(")")[0]
    assert "retries=MAX_RETRIES" in SRC.split("_eval_kw = dict(")[1].split(")")[0]
    assert "image=snapshot_image" in fns["snapshot_worker"]
    assert "image=train_image" in fns["train_candidate"] and "gpu=GPU_KIND" in fns["train_candidate"]
    assert "block_network=BLOCK_NETWORK" in fns["train_candidate"]
    assert "block_network=BLOCK_NETWORK" in SRC.split("_eval_kw = dict(")[1].split(")")[0]


def test_the_crons_come_from_the_checked_in_schedule():
    fns = _decorated_functions(SRC)
    for name, key in (("snapshot_tick", "snapshot_cron"), ("nightly_tick", "nightly_cron"),
                      ("weekly_tick", "weekly_cron")):
        assert f"modal.Cron(SCHEDULE['{key}'], timezone='UTC')" in fns[name]
    assert "max_containers=1" in SRC.split("_tick_kw = dict(")[1].split(")")[0]


def test_the_training_image_bakes_no_data_and_the_snapshot_image_no_training_stack():
    images = SRC.split("app = modal.App(APP_NAME)")[0]
    calls = [ast.unparse(n) for n in ast.walk(ast.parse(SRC))
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("add_local_dir", "add_local_file")]
    assert len(calls) == 1 and "REPO / 'naigos'" in calls[0], calls
    snapshot_block = images.split("snapshot_image = ")[1].split("train_image")[0]
    assert "jax" not in snapshot_block.split("#")[0]


def test_only_the_evaluator_and_promoter_functions_reach_the_champion():
    """The champion pointer moves through worker.promote/rollback, called only from
    evaluate_candidate (auto), promote_candidate and rollback_champion."""
    tree = ast.parse(SRC)
    callers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = ast.unparse(node)
            if "worker.promote(" in body or "worker.rollback(" in body or "run_evaluation(svc" in body:
                callers.add(node.name)
    assert callers <= {"evaluate_candidate", "promote_candidate", "rollback_champion"}, callers
    for mod in ("coordinator", "admin", "snapshot", "leases", "jobs", "config"):
        text = (REPO / "naigos" / "pipeline" / f"{mod}.py").read_text()
        assert "publish_champion(" not in text and "champion_pointer" not in text.replace(
            "lay.champion_pointer.parent", ""), mod


def test_no_credential_is_passed_as_an_argument():
    for path in (CLI, REPO / "naigos" / "rl" / "modal_pipeline.py"):
        text = path.read_text()
        assert "MODAL_TOKEN_SECRET=" not in text and "add_argument(\"--token" not in text
        assert "Secret.from_dict" not in text
