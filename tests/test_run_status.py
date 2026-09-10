"""Remote-run status, manifests, locking and resume rules -- `naigos.rl.runmeta`.

Everything here runs offline: no Modal, no GPU, no network, no JAX. That is the
point. The rules that decide whether a remote run succeeded are exactly the
rules that must not be trusted to a live service to demonstrate, because the
failure they exist to catch -- a job that died without saying so -- is the one
that never reproduces on demand.

The interesting cases are all the ones where the two sources of truth disagree:
the worker's manifest says `running` because the container was killed before it
could say anything else, and Modal says the call is over.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from naigos.rl import runmeta as rm

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest(**over) -> dict:
    man = rm.build_manifest(
        run_name="short-s0-20260909T101500Z",
        job_id="fc-abc123",
        profile="short",
        gpu="A10G",
        timeout_s=21600,
        max_retries=0,
        checkpoint_every=50,
        code={"commit": "deadbeef", "dirty": False},
        config={"iterations": 200, "seed": 0, "synthetic": False},
        now=NOW - timedelta(minutes=5),
    )
    man.update(over)
    return man


# --- parsing what Modal says -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PENDING", rm.QUEUED),
        ("INPUT_STATUS_PENDING", rm.QUEUED),
        ("running", rm.RUNNING),
        ("FUNCTION_CALL_STATUS_SUCCESS", rm.COMPLETED),
        ("Succeeded", rm.COMPLETED),
        ("INPUT_STATUS_TIMEOUT", rm.TIMED_OUT),
        ("timed-out", rm.TIMED_OUT),
        ("TERMINATED", rm.CANCELLED),
        ("app_state_stopped", rm.CANCELLED),
        ("FAILURE", rm.FAILED),
        (None, rm.UNKNOWN),
        ("some_state_modal_invented_last_tuesday", rm.UNKNOWN),
    ],
)
def test_every_spelling_modal_uses_maps_to_one_state(raw, expected):
    assert rm.parse_modal_state(raw) == expected


def test_an_unrecognised_state_is_unknown_and_never_success():
    """A client upgrade that renames a state must degrade to `unknown`.

    The alternative -- a default of `completed` or a silent pass-through -- turns
    a Modal API change into a repository that reports unfinished runs as done.
    """
    for raw in ("weird", "", "STATE_", object()):
        assert rm.parse_modal_state(raw) == rm.UNKNOWN


def test_an_exception_type_is_a_state():
    """`FunctionCall.get(timeout=0)` reports failure by raising, so the exception
    type is the only signal there is."""

    class FunctionTimeoutError(Exception):
        pass

    assert rm.parse_modal_state(FunctionTimeoutError) == rm.TIMED_OUT
    assert rm.parse_modal_state(FunctionTimeoutError("6h")) == rm.TIMED_OUT


# --- merging the manifest with the call state --------------------------------


def test_a_queued_job_is_queued_not_missing():
    v = rm.classify_status(_manifest(status=rm.QUEUED), modal_state="PENDING", now=NOW)
    assert v["status"] == rm.QUEUED
    assert v["resumable"] is False


def test_a_running_job_with_a_fresh_heartbeat_is_running():
    v = rm.classify_status(
        _manifest(status=rm.RUNNING, heartbeat_utc=_stamp(NOW - timedelta(seconds=30))),
        modal_state="running", now=NOW,
    )
    assert v["status"] == rm.RUNNING


def test_a_killed_container_leaves_a_running_manifest_and_is_reported_as_timed_out():
    """The core case. A container that hits its timeout never gets to update its
    own manifest, so the manifest says `running` forever. Trusting it alone
    reports a dead job as live; trusting Modal alone loses the iteration count."""
    v = rm.classify_status(
        _manifest(status=rm.RUNNING, last_iteration=137,
                  heartbeat_utc=_stamp(NOW - timedelta(minutes=2))),
        modal_state="INPUT_STATUS_TIMEOUT", checkpoints=[50, 100], now=NOW,
    )
    assert v["status"] == rm.TIMED_OUT
    assert v["termination_reason"] == rm.REASON_TIMEOUT
    assert v["resumable"] is True
    assert v["resume_from_iteration"] == 100
    assert v["last_iteration"] == 137
    assert any("killed without being able to record why" in n for n in v["notes"])


def test_a_worker_recorded_failure_beats_the_upstream_state():
    """The worker was there. If it caught its own exception and wrote it down,
    that is the specific account and the container reaper's is the vague one."""
    v = rm.classify_status(
        _manifest(status=rm.FAILED, termination_reason=rm.REASON_EXCEPTION, last_iteration=12),
        modal_state="TERMINATED", checkpoints=[10], now=NOW,
    )
    assert v["status"] == rm.FAILED
    assert v["termination_reason"] == rm.REASON_EXCEPTION
    assert any("worker wins" in n for n in v["notes"])


def test_a_cancelled_job_is_cancelled_and_not_failed():
    v = rm.classify_status(
        _manifest(status=rm.RUNNING), modal_state="cancelled", checkpoints=[50], now=NOW
    )
    assert v["status"] == rm.CANCELLED
    assert v["resumable"] is True


def test_a_stale_heartbeat_with_no_upstream_signal_is_unknown_not_running():
    """Reporting `running` for a job nobody has heard from in hours is the same
    mistake as reporting it complete: it is a claim the evidence does not support."""
    v = rm.classify_status(
        _manifest(status=rm.RUNNING, heartbeat_utc=_stamp(NOW - timedelta(hours=9))),
        modal_state=None, checkpoints=[50], now=NOW,
    )
    assert v["status"] == rm.UNKNOWN
    assert v["resumable"] is True
    assert any("cannot be confirmed" in n for n in v["notes"])


def test_a_completed_run_is_not_resumable():
    v = rm.classify_status(
        _manifest(status=rm.COMPLETED, termination_reason=rm.REASON_COMPLETED,
                  last_iteration=200),
        modal_state="SUCCESS", checkpoints=[50, 100, 150, 200], now=NOW,
    )
    assert v["status"] == rm.COMPLETED
    assert v["resumable"] is False


def test_a_run_marked_completed_short_of_its_declared_iterations_is_flagged():
    v = rm.classify_status(
        _manifest(status=rm.COMPLETED, last_iteration=100), checkpoints=[100], now=NOW
    )
    assert v["status"] == rm.COMPLETED
    assert v["resumable"] is True
    assert any("truncated" in n for n in v["notes"])


def test_a_failure_before_the_first_checkpoint_is_not_resumable():
    """Resumable means recovery state exists, not that the run ended badly.
    Saying otherwise sends the operator to a `resume` that cannot work."""
    v = rm.classify_status(_manifest(status=rm.RUNNING), modal_state="FAILURE",
                           checkpoints=[], now=NOW)
    assert v["status"] == rm.FAILED
    assert v["resumable"] is False
    assert any("only restarted" in n for n in v["notes"])


def test_nothing_at_all_is_unknown():
    v = rm.classify_status(None, now=NOW)
    assert v["status"] == rm.UNKNOWN
    assert v["resumable"] is False


def test_every_lifecycle_state_the_spec_asks_for_is_reachable():
    """queued, running, completed, timed out, failed, cancelled -- plus the
    orthogonal `resumable` flag, which is a flag and not a seventh state because
    a run can be timed out AND resumable at once."""
    assert set(rm.LIFECYCLE_STATES) == {
        rm.QUEUED, rm.RUNNING, rm.COMPLETED, rm.TIMED_OUT, rm.FAILED, rm.CANCELLED, rm.UNKNOWN
    }
    assert rm.classify_status(_manifest(status=rm.TIMED_OUT), checkpoints=[7])["resumable"]


# --- the manifest ------------------------------------------------------------


def test_the_manifest_carries_everything_needed_to_reconstruct_the_run(tmp_path):
    man = _manifest()
    for key in ("job_id", "run_name", "submitted_utc", "requested_gpu", "timeout_s",
                "max_retries", "checkpoint_every", "actual_backend", "actual_devices",
                "code", "termination_reason", "status"):
        assert key in man, key
    assert man["code"]["commit"] == "deadbeef"


def test_manifest_updates_merge_and_attempts_append(tmp_path):
    rm.write_json(tmp_path / rm.MANIFEST_FILENAME, _manifest())
    rm.update_manifest(tmp_path, status=rm.RUNNING, attempt={"kind": "start"})
    rm.update_manifest(tmp_path, last_iteration=50, attempt={"kind": "resume"})
    man = rm.read_manifest(tmp_path)
    assert man["status"] == rm.RUNNING
    assert man["last_iteration"] == 50
    assert man["job_id"] == "fc-abc123"  # untouched by either update
    assert [a["kind"] for a in man["attempts"]] == ["start", "resume"]


def test_the_manifest_is_separate_from_the_immutable_metadata(tmp_path):
    """`run.json` must keep describing the run even after the run times out.
    Recording the termination reason in it would make the identity record
    mutable, which is the thing `write_metadata` exists to prevent."""
    profile = rm.resolve_profile("smoke")
    meta = rm.build_metadata(profile, run_name="r1", seed=0, synthetic=True,
                             code={"commit": "abc", "dirty": False}, launcher="modal")
    rm.write_metadata(tmp_path, meta)
    rm.write_json(tmp_path / rm.MANIFEST_FILENAME, _manifest())
    rm.update_manifest(tmp_path, status=rm.TIMED_OUT, termination_reason=rm.REASON_TIMEOUT)
    assert rm.read_metadata(tmp_path) == meta
    assert rm.read_manifest(tmp_path)["status"] == rm.TIMED_OUT


def test_a_second_run_still_cannot_overwrite_the_first_ones_metadata(tmp_path):
    profile = rm.resolve_profile("smoke")
    base = dict(run_name="r1", seed=0, synthetic=True,
                code={"commit": "abc", "dirty": False}, launcher="modal")
    rm.write_metadata(tmp_path, rm.build_metadata(profile, **base))
    rm.write_metadata(tmp_path, rm.build_metadata(profile, **base))  # same run: allowed
    with pytest.raises(rm.RunCollision):
        rm.write_metadata(tmp_path, rm.build_metadata(profile, **{**base, "seed": 1}))


def test_a_failed_run_keeps_its_artifacts_and_says_why(tmp_path):
    """The requirement that matters after a six-hour job dies: perf.json,
    history.json, the checkpoints and the reason all survive together."""
    rm.write_json(tmp_path / rm.HISTORY_FILENAME, [{"iter": 0}, {"iter": 50}])
    rm.write_json(tmp_path / rm.PERF_FILENAME, {"device": {"platform": "gpu"},
                                                "steady_iteration_s_median": 0.2})
    (tmp_path / "ckpt_000050.pkl").write_bytes(b"x")
    rm.write_json(tmp_path / rm.MANIFEST_FILENAME,
                  _manifest(status=rm.FAILED, termination_reason=rm.REASON_EXCEPTION,
                            last_iteration=50, error="RuntimeError: boom"))
    summary = rm.summarize_run(tmp_path)
    assert summary["status"] == rm.FAILED
    assert summary["termination_reason"] == rm.REASON_EXCEPTION
    assert summary["checkpoints"] == [50]
    problems = rm.verify_run_dir(tmp_path)
    assert any("failed" in p for p in problems)
    assert any("boom" in p for p in problems)


def test_verification_reports_a_modal_run_with_no_manifest(tmp_path):
    profile = rm.resolve_profile("smoke")
    rm.write_metadata(tmp_path, rm.build_metadata(
        profile, run_name=tmp_path.name, seed=0, synthetic=True,
        code={"commit": "abc", "dirty": False}, launcher="modal"))
    assert any("no manifest.json" in p for p in rm.verify_run_dir(tmp_path))


# --- one writer per run directory --------------------------------------------


def test_a_second_writer_is_refused(tmp_path):
    rm.acquire_writer_lock(tmp_path, "worker-1", now=NOW)
    with pytest.raises(rm.RunBusy) as e:
        rm.acquire_writer_lock(tmp_path, "worker-2", now=NOW)
    assert "worker-1" in str(e.value)


def test_a_stale_lock_can_be_taken_over(tmp_path):
    rm.acquire_writer_lock(tmp_path, "worker-1", now=NOW - timedelta(hours=5))
    held = rm.acquire_writer_lock(tmp_path, "worker-2", now=NOW)
    assert held["owner"] == "worker-2"


def test_a_refreshed_lock_does_not_go_stale(tmp_path):
    rm.acquire_writer_lock(tmp_path, "worker-1", now=NOW - timedelta(hours=5))
    rm.refresh_writer_lock(tmp_path, now=NOW)
    with pytest.raises(rm.RunBusy):
        rm.acquire_writer_lock(tmp_path, "worker-2", now=NOW)


def test_a_released_lock_frees_the_directory(tmp_path):
    rm.acquire_writer_lock(tmp_path, "worker-1", now=NOW)
    rm.release_writer_lock(tmp_path)
    assert rm.acquire_writer_lock(tmp_path, "worker-2", now=NOW)["owner"] == "worker-2"


def test_force_takes_a_live_lock_because_sometimes_the_worker_really_is_gone(tmp_path):
    rm.acquire_writer_lock(tmp_path, "worker-1", now=NOW)
    assert rm.acquire_writer_lock(tmp_path, "worker-2", force=True, now=NOW)["owner"] == "worker-2"


# --- checkpoint selection ----------------------------------------------------


def _ckpt(directory: Path, it: int, body: bytes = b"payload") -> Path:
    p = directory / f"ckpt_{it:06d}.pkl"
    p.write_bytes(body)
    return p


def test_checkpoints_sort_by_iteration_not_lexically(tmp_path):
    for it in (200, 1000, 50):
        _ckpt(tmp_path, it)
    assert [rm.checkpoint_iteration(p) for p in rm.checkpoint_paths(tmp_path)] == [50, 200, 1000]


def test_the_newest_checkpoint_is_selected(tmp_path):
    for it in (50, 100, 150):
        _ckpt(tmp_path, it)
    assert rm.select_checkpoint(tmp_path).name == "ckpt_000150.pkl"


def test_a_truncated_newest_checkpoint_is_skipped(tmp_path):
    """The newest file is exactly the one a killed process was most likely
    mid-write on, so `select_checkpoint` walks backwards rather than trusting it."""
    _ckpt(tmp_path, 100)
    _ckpt(tmp_path, 150, body=b"")
    assert rm.select_checkpoint(tmp_path).name == "ckpt_000100.pkl"


def test_an_empty_directory_selects_nothing(tmp_path):
    assert rm.select_checkpoint(tmp_path) is None
    assert rm.checkpoint_paths(tmp_path / "does-not-exist") == []


def test_a_validator_that_raises_does_not_hide_earlier_checkpoints(tmp_path):
    _ckpt(tmp_path, 10)
    _ckpt(tmp_path, 20)

    def explodes(path: Path) -> bool:
        if path.name.endswith("000020.pkl"):
            raise RuntimeError("unpickling blew up")
        return True

    assert rm.select_checkpoint(tmp_path, is_valid=explodes).name == "ckpt_000010.pkl"


def test_files_that_are_not_checkpoints_are_ignored(tmp_path):
    (tmp_path / "ckpt_notanumber.pkl").write_bytes(b"x")
    (tmp_path / "history.json").write_text("[]")
    _ckpt(tmp_path, 5)
    assert [p.name for p in rm.checkpoint_paths(tmp_path)] == ["ckpt_000005.pkl"]


# --- resume compatibility ----------------------------------------------------


def _meta(**over) -> dict:
    cfg = dict(seed=0, synthetic=False, aoi="owens_valley", use_cbf=False)
    cfg.update(over.pop("config", {}))
    profile = rm.resolve_profile(over.pop("profile", "short"))
    meta = rm.build_metadata(
        profile, run_name="short-s0-20260909T101500Z",
        code=over.pop("code", {"commit": "deadbeef", "dirty": False}),
        launcher="modal", **cfg,
    )
    meta.update(over)
    return meta


def test_an_identical_relaunch_may_resume():
    assert rm.resume_compatibility(_meta(), _meta())["ok"] is True


def test_a_different_commit_blocks_resume():
    v = rm.resume_compatibility(_meta(), _meta(code={"commit": "cafe", "dirty": False}))
    assert v["ok"] is False
    assert any("code.commit" in b for b in v["blocking"])


def test_a_different_theatre_blocks_resume():
    v = rm.resume_compatibility(_meta(), _meta(config={"aoi": "tehran_basin"}))
    assert v["ok"] is False
    assert "config.aoi" in v["changes"]


def test_a_different_seed_blocks_resume():
    assert rm.resume_compatibility(_meta(), _meta(config={"seed": 7}))["ok"] is False


def test_a_different_length_blocks_resume():
    """`run.json` declares how long the run is and is immutable, so extending a
    run at resume would leave the record describing a different experiment --
    and would shift every RNG split that depends on the final iteration."""
    assert rm.resume_compatibility(_meta(), _meta(profile="full"))["ok"] is False


def test_an_override_must_name_the_key_it_accepts():
    v = rm.resume_compatibility(
        _meta(), _meta(code={"commit": "cafe", "dirty": False}), overrides=["code.commit"]
    )
    assert v["ok"] is True
    assert v["overridden"] and "code.commit" in v["overridden"][0]


def test_an_override_of_one_key_does_not_wave_through_another():
    """A blanket `--force` would also accept the theatre change nobody meant to
    make, which is why the override is per key."""
    v = rm.resume_compatibility(
        _meta(),
        _meta(code={"commit": "cafe", "dirty": False}, config={"aoi": "tehran_basin"}),
        overrides=["code.commit"],
    )
    assert v["ok"] is False
    assert v["blocking"] == [b for b in v["blocking"] if "config.aoi" in b]


def test_an_unknown_override_key_is_an_error():
    with pytest.raises(ValueError):
        rm.resume_compatibility(_meta(), _meta(), overrides=["config.nonsense"])


def test_resume_into_a_directory_with_no_metadata_is_refused():
    v = rm.resume_compatibility(None, _meta())
    assert v["ok"] is False


# --- the smoke gate still gates ----------------------------------------------


def test_an_expensive_profile_still_needs_a_smoke_run_for_this_commit():
    assert rm.smoke_gate("smoke", None, "abc") is None
    assert "no smoke run is recorded" in rm.smoke_gate("full", None, "abc")
    marker = rm.smoke_marker_payload("smoke-s0", {"commit": "abc", "dirty": False})
    assert rm.smoke_gate("full", marker, "abc") is None
    assert "Re-run" in rm.smoke_gate("full", marker, "def456789012")


# --- credential hygiene ------------------------------------------------------


#: Token-shaped example strings are ASSEMBLED AT RUNTIME rather than written as
#: literals. A test that hard-codes `ak-...` puts a credential-shaped string into
#: a tracked file, which is exactly what the repository-wide scans below and in
#: `test_run_metadata.py` exist to fail on -- and the only way out of that would
#: be to teach both scans to skip this file, which is how a scan stops working.
_ID = "ak-" + "AbCdEf" + "0123456789"
_SECRET = "as-" + "ZyXwVu" + "9876543210"


def test_modal_tokens_are_masked_before_anything_is_written():
    out = rm.redact(f"connecting with {_ID} / {_SECRET}")
    assert _ID not in out and _SECRET not in out
    assert rm.REDACTED in out


def test_environment_style_assignments_are_masked():
    for line in ("MODAL_TOKEN" + "_SECRET=" + "hunter2hunter2",
                 "MODAL_TOKEN" + "_ID: " + "something-opaque",
                 "NAIGOS_MAP_TOKEN=" + "abcdefgh.ijklmnop.qrstuvwx"):
        assert rm.REDACTED in rm.redact(line)


def test_ordinary_output_is_left_alone():
    line = "[  25] R  1234.5 cost 0.031 lam 1.02 | surv 0.812"
    assert rm.redact(line) == line


def test_no_credential_shaped_string_is_committed_to_the_repository():
    """Broader than `test_run_metadata.py`'s Modal-token scan, and deliberately
    kept beside it: that one looks for `ak-`/`as-` literals, this one runs the
    same `redact` the worker applies to its log, so it also catches `*_TOKEN=`,
    `*_API_KEY=` and JWT-shaped values.

    Neither scan has a file-level exemption. A scan that grows exceptions stops
    being a scan."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = []
    for name in tracked:
        path = REPO / name
        if not path.is_file() or path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".pkl", ".tif", ".npz", ".ico", ".woff2"
        }:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if rm.redact(text) != text:
            offenders.append(name)
    assert offenders == [], f"credential-shaped strings in tracked files: {offenders}"


def test_metadata_never_captures_the_process_environment():
    """`run.json` travels with results. Snapshotting `os.environ` into it on a
    machine that has run `modal token set` would commit a credential."""
    meta = _meta()
    blob = json.dumps(meta)
    assert "environ" not in blob and "MODAL_TOKEN" not in blob
    man = _manifest()
    assert "MODAL_TOKEN" not in json.dumps(man)


# --- the smoke run's preflight -----------------------------------------------


def test_a_complete_preflight_reports_nothing():
    report = {c: {"ok": True, "detail": "fine"} for c in rm.PREFLIGHT_CHECKS}
    assert rm.preflight_problems(report) == []


def test_a_missing_check_is_a_problem_not_a_pass():
    """A preflight that stopped reporting a check after a refactor would leave a
    smoke run verifying less than it claims to, and the smoke run is the gate in
    front of every expensive run."""
    report = {c: {"ok": True} for c in rm.PREFLIGHT_CHECKS if c != "gpu_backend"}
    assert rm.preflight_problems(report) == ["preflight check 'gpu_backend' did not run"]


def test_a_failed_check_names_itself_and_its_reason():
    report = {c: {"ok": True} for c in rm.PREFLIGHT_CHECKS}
    report["data_cache"] = {"ok": False, "detail": "/root/data_cache is present but empty"}
    problems = rm.preflight_problems(report)
    assert len(problems) == 1
    assert "data_cache" in problems[0] and "empty" in problems[0]


def test_an_absent_preflight_fails_every_check():
    assert len(rm.preflight_problems(None)) == len(rm.PREFLIGHT_CHECKS)


def test_the_preflight_covers_everything_the_smoke_run_is_for():
    """image build, GPU backend, data cache, Volume writes, checkpoint
    persistence, artifact retrieval -- one named check each."""
    assert set(rm.PREFLIGHT_CHECKS) == {
        "image", "gpu_backend", "data_cache", "volume_write", "checkpoint", "artifact_fetch"
    }


def test_a_smoke_run_with_a_failed_preflight_does_not_verify(tmp_path):
    rm.write_json(tmp_path / rm.HISTORY_FILENAME, [{"iter": 0}, {"iter": 3}])
    rm.write_json(tmp_path / rm.PERF_FILENAME,
                  {"device": {"platform": "gpu"}, "steady_iteration_s_median": 0.2})
    (tmp_path / "ckpt_000003.pkl").write_bytes(b"x")
    man = _manifest(profile="smoke", status=rm.COMPLETED,
                    termination_reason=rm.REASON_COMPLETED, last_iteration=3)
    man["preflight"] = {c: {"ok": True} for c in rm.PREFLIGHT_CHECKS}
    man["preflight"]["gpu_backend"] = {"ok": False, "detail": "JAX resolved backend 'cpu'"}
    rm.write_json(tmp_path / rm.MANIFEST_FILENAME, man)
    assert any("gpu_backend" in p for p in rm.verify_run_dir(tmp_path))
