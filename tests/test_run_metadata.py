"""Run identity, cost profiles and output verification -- `naigos.rl.runmeta`.

These are the rules that decide what a training run *is*, so they are tested
without a GPU, without Modal and without importing the training stack. The
interesting cases are all failure cases: a run name that escapes its directory,
a second run overwriting the first one's metadata, and a truncated or
silently-CPU run reading as a completed GPU run.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from naigos.rl import runmeta as rm

REPO = Path(__file__).resolve().parents[1]


# --- cost profiles ----------------------------------------------------------


def test_every_profile_is_positively_sized():
    for name, p in rm.PROFILES.items():
        assert p.name == name
        assert p.purpose
        for field in ("iterations", "n_envs", "n_steps", "n_threat", "n_blue",
                      "eval_every", "eval_worlds", "checkpoint_every"):
            assert getattr(p, field) > 0, f"{name}.{field}"
        assert p.cell_m > 0


def test_the_smoke_profile_is_strictly_the_cheapest():
    """The gate before an expensive run has to be cheap or it is not a gate."""
    smoke = rm.PROFILES["smoke"]
    for name in ("short", "full"):
        assert smoke.total_env_steps < rm.PROFILES[name].total_env_steps
    assert rm.PROFILES["short"].total_env_steps < rm.PROFILES["full"].total_env_steps
    # a couple of minutes of plumbing check, not a training run
    assert smoke.iterations <= 5


def test_publishable_profiles_use_the_corrected_terrain_fidelity():
    """1500 m cells over-report visibility by ~50% relative at low altitude
    (DEVLOG). Anything whose numbers could be reported must run at 500 m."""
    for name in ("short", "full"):
        assert rm.PROFILES[name].cell_m == 500.0


def test_work_per_iteration_counts_agents_and_worlds():
    p = rm.resolve_profile("smoke", n_envs=16, n_steps=32)
    assert p.env_steps_per_iteration == 512
    assert p.agent_steps_per_iteration == 512 * p.n_blue
    assert p.total_env_steps == 512 * p.iterations


def test_resolve_profile_applies_overrides_and_records_them():
    p = rm.resolve_profile("full", iterations=10, n_envs=None)
    assert p.iterations == 10
    assert p.n_envs == rm.PROFILES["full"].n_envs  # None means "leave it alone"
    assert p.overrides == ("iterations",)
    assert rm.PROFILES["full"].iterations != 10, "the profile table was mutated"


def test_resolve_profile_rejects_unknown_names_and_fields_and_nonsense_sizes():
    with pytest.raises(ValueError, match="unknown profile"):
        rm.resolve_profile("enormous")
    with pytest.raises(ValueError, match="cannot override"):
        rm.resolve_profile("smoke", learning_rate=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        rm.resolve_profile("smoke", iterations=0)


def test_profile_kwargs_match_the_dataclasses_they_are_passed_to():
    """The profile is only useful if it can actually be splatted into the
    configs; a renamed field would otherwise fail at launch on a GPU worker."""
    import dataclasses

    from naigos.env.theatre_bridge import env_from_theatre  # noqa: F401  (import check)
    from naigos.rl.ppo import PPOConfig
    from naigos.rl.train import TrainConfig

    p = rm.PROFILES["full"]
    assert set(p.ppo_kwargs()) <= {f.name for f in dataclasses.fields(PPOConfig)}
    assert set(p.train_kwargs()) <= {f.name for f in dataclasses.fields(TrainConfig)}
    PPOConfig(**p.ppo_kwargs())
    TrainConfig(**p.train_kwargs())
    assert set(p.theatre_kwargs()) == {"n_blue", "n_threat", "cell_m"}


# --- run names and isolation -------------------------------------------------


@pytest.mark.parametrize("name", ["a", "smoke-s0-20260101T000000Z", "full.2", "A_b-1"])
def test_valid_run_names_are_accepted(name):
    assert rm.validate_run_name(name) == name


@pytest.mark.parametrize("name", [
    "", "/abs", "a/b", "..", ".", "../escape", ".hidden", "-leading", "x" * 65,
    "has space", "semi;colon", "quote'", "star*",
])
def test_run_names_that_could_escape_or_collide_are_refused(name):
    """A run name becomes a directory on a shared Volume. Path traversal and
    shell metacharacters are rejected here, not at the filesystem."""
    with pytest.raises(rm.RunNameError):
        rm.validate_run_name(name)


def test_run_dir_cannot_escape_the_runs_root():
    assert rm.run_dir("/runs", "abc") == Path("/runs/abc")
    with pytest.raises(rm.RunNameError):
        rm.run_dir("/runs", "../../etc")


def test_default_run_name_is_unique_per_profile_seed_and_second():
    t1 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert rm.default_run_name("full", 7, t1) == "full-s7-20260102T030405Z"
    assert rm.default_run_name("full", 8, t1) != rm.default_run_name("full", 7, t1)
    t2 = datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
    assert rm.default_run_name("full", 7, t2) != rm.default_run_name("full", 7, t1)
    rm.validate_run_name(rm.default_run_name("smoke", 0))


# --- immutable metadata ------------------------------------------------------


def _meta(**kw):
    defaults = dict(run_name="r1", seed=0, synthetic=True, code={"commit": "abc", "dirty": False})
    defaults.update(kw)
    profile = defaults.pop("profile", rm.resolve_profile("smoke"))
    return rm.build_metadata(profile, **defaults)


def test_metadata_records_the_configuration_and_the_commit(tmp_path):
    meta = _meta(aoi="owens_valley", synthetic=False)
    assert meta["schema"] == rm.SCHEMA_VERSION
    assert meta["profile"] == "smoke"
    assert meta["config"]["aoi"] == "owens_valley"
    assert meta["config"]["synthetic"] is False
    assert meta["config"]["total_env_steps"] > 0
    assert meta["code"]["commit"] == "abc"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_utc"])


def test_metadata_never_captures_the_process_environment():
    """A machine that has run `modal token set` or exported an ion token would
    otherwise write credentials into a file that sits next to results."""
    blob = json.dumps(_meta(), default=str).lower()
    for leak in ("modal_token", "aws_", "cesium_ion", "api_key", "secret", "password"):
        assert leak not in blob


def test_metadata_is_written_once_and_a_rerun_of_the_same_run_is_a_no_op(tmp_path):
    meta = _meta()
    written, fresh = rm.write_metadata(tmp_path, meta)
    assert fresh and written == meta
    again, fresh2 = rm.write_metadata(tmp_path, _meta())
    assert not fresh2 and again["created_utc"] == meta["created_utc"]


def test_a_different_run_cannot_overwrite_an_existing_run_directory(tmp_path):
    """Two runs sharing a directory interleave their checkpoints and their
    history, and the second reads as a continuation of the first."""
    rm.write_metadata(tmp_path, _meta())
    with pytest.raises(rm.RunCollision, match="config"):
        rm.write_metadata(tmp_path, _meta(profile=rm.resolve_profile("smoke", iterations=99)))


def test_a_different_commit_is_also_a_collision(tmp_path):
    rm.write_metadata(tmp_path, _meta())
    with pytest.raises(rm.RunCollision, match="code"):
        rm.write_metadata(tmp_path, _meta(code={"commit": "def", "dirty": False}))


def test_a_different_device_is_not_a_collision(tmp_path):
    """The same run relaunched onto another GPU is the same run specification."""
    rm.write_metadata(tmp_path, {**_meta(), "runtime": {"platform": "cpu"}})
    _, fresh = rm.write_metadata(tmp_path, {**_meta(), "runtime": {"platform": "gpu"}})
    assert not fresh


def test_write_json_leaves_no_half_written_file(tmp_path):
    p = tmp_path / "x.json"
    rm.write_json(p, {"a": 1})
    rm.write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
    assert not list(tmp_path.glob("*.tmp"))


# --- the smoke gate ----------------------------------------------------------


def test_the_smoke_profile_is_never_gated():
    assert rm.smoke_gate("smoke", None, "abc") is None


def test_an_expensive_profile_needs_a_smoke_run_first():
    msg = rm.smoke_gate("full", None, "abc")
    assert msg and "--profile smoke" in msg


def test_a_smoke_run_against_another_commit_does_not_count():
    marker = rm.smoke_marker_payload("smoke-s0", {"commit": "aaaa1111", "dirty": False})
    assert rm.smoke_gate("full", marker, "aaaa1111") is None
    msg = rm.smoke_gate("full", marker, "bbbb2222")
    assert msg and "aaaa1111"[:12] in msg


def test_the_gate_does_not_block_when_the_commit_is_unknowable():
    marker = rm.smoke_marker_payload("smoke-s0", {"commit": None, "dirty": None})
    assert rm.smoke_gate("full", marker, None) is None


# --- output verification -----------------------------------------------------


def _good_run(tmp_path: Path, *, iterations=4, launcher="modal", platform="gpu",
              synthetic=True, dirty=False, manifest=True) -> Path:
    d = tmp_path / "r1"
    d.mkdir()
    profile = rm.resolve_profile("smoke", iterations=iterations)
    rm.write_json(d / rm.META_FILENAME, rm.build_metadata(
        profile, run_name="r1", seed=0, synthetic=synthetic,
        code={"commit": "abc", "dirty": dirty}, launcher=launcher))
    rm.write_json(d / rm.HISTORY_FILENAME, [
        {"iter": 0, "phase": "baseline"}, {"iter": 1}, {"iter": iterations},
    ])
    rm.write_json(d / rm.PERF_FILENAME, {
        "device": {"platform": platform}, "steady_iteration_s_median": 0.5,
        "env_steps_per_s": 1000.0,
    })
    (d / f"ckpt_{iterations:06d}.pkl").write_bytes(b"x")
    if manifest:
        man = rm.build_manifest(
            run_name="r1", job_id="fc-1", profile="smoke", gpu="A10G", timeout_s=1800,
            max_retries=0, checkpoint_every=iterations, code={"commit": "abc", "dirty": dirty},
            config={"iterations": iterations},
        )
        man.update(status=rm.COMPLETED, termination_reason=rm.REASON_COMPLETED,
                   last_iteration=iterations, actual_backend=platform,
                   preflight={c: {"ok": True, "detail": "fixture"}
                              for c in rm.PREFLIGHT_CHECKS})
        rm.write_json(d / rm.MANIFEST_FILENAME, man)
    if not synthetic:
        rm.write_json(d / rm.THEATRE_FILENAME, {"aoi": "owens_valley"})
    return d


def test_a_complete_run_verifies_clean(tmp_path):
    assert rm.verify_run_dir(_good_run(tmp_path)) == []


def test_a_truncated_run_does_not_read_as_a_completed_one(tmp_path):
    d = _good_run(tmp_path, iterations=1000)
    rm.write_json(d / rm.HISTORY_FILENAME, [{"iter": 0}, {"iter": 25}])
    (d / "ckpt_001000.pkl").rename(d / "ckpt_000025.pkl")
    problems = rm.verify_run_dir(d)
    assert any("truncated" in p for p in problems)
    assert any("last checkpoint" in p for p in problems)


def test_a_modal_run_that_actually_ran_on_cpu_is_reported(tmp_path):
    """JAX falls back to CPU silently. The loop still finishes and the timings
    still look like timings, which is exactly how a CPU number gets published
    as a GPU number."""
    problems = rm.verify_run_dir(_good_run(tmp_path, platform="cpu"))
    assert any("ran on 'cpu'" in p for p in problems)


def test_a_local_cpu_run_is_not_accused_of_anything(tmp_path):
    assert rm.verify_run_dir(_good_run(tmp_path, launcher="local", platform="cpu")) == []


def test_a_directory_with_no_metadata_cannot_describe_itself(tmp_path):
    d = _good_run(tmp_path)
    (d / rm.META_FILENAME).unlink()
    assert any("does not record what produced it" in p for p in rm.verify_run_dir(d))


def test_metadata_belonging_to_a_different_run_is_caught(tmp_path):
    d = _good_run(tmp_path)
    meta = json.loads((d / rm.META_FILENAME).read_text())
    meta["run_name"] = "somebody-elses-run"
    rm.write_json(d / rm.META_FILENAME, meta)
    assert any("does not match" in p for p in rm.verify_run_dir(d))


def test_two_runs_interleaved_into_one_directory_show_up_as_non_monotonic(tmp_path):
    d = _good_run(tmp_path)
    rm.write_json(d / rm.HISTORY_FILENAME, [{"iter": 0}, {"iter": 100}, {"iter": 4}])
    assert any("not monotonic" in p for p in rm.verify_run_dir(d))


def test_missing_checkpoints_history_and_perf_are_each_reported(tmp_path):
    d = _good_run(tmp_path)
    (d / "ckpt_000004.pkl").unlink()
    (d / rm.HISTORY_FILENAME).unlink()
    (d / rm.PERF_FILENAME).unlink()
    problems = " | ".join(rm.verify_run_dir(d))
    assert "no checkpoints" in problems
    assert rm.HISTORY_FILENAME in problems
    assert rm.PERF_FILENAME in problems


def test_a_real_theatre_run_must_carry_its_data_provenance(tmp_path):
    d = _good_run(tmp_path, synthetic=False)
    (d / rm.THEATRE_FILENAME).unlink()
    assert any("provenance" in p for p in rm.verify_run_dir(d))


def test_a_dirty_tree_is_reported_as_not_reproducible(tmp_path):
    assert any("dirty" in p for p in rm.verify_run_dir(_good_run(tmp_path, dirty=True)))


def test_corrupt_json_is_reported_rather_than_raised(tmp_path):
    d = _good_run(tmp_path)
    (d / rm.HISTORY_FILENAME).write_text("{not json")
    assert any("does not parse" in p for p in rm.verify_run_dir(d))


def test_summarize_run_survives_a_missing_or_partial_directory(tmp_path):
    assert rm.summarize_run(tmp_path / "nope")["run_name"] == "nope"
    s = rm.summarize_run(_good_run(tmp_path))
    assert s["platform"] == "gpu" and s["checkpoints"] == [4]


# --- credentials -------------------------------------------------------------


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")
    return [REPO / line for line in out.stdout.split("\n") if line]


def test_no_modal_credential_is_committed():
    """Modal tokens are `ak-...` / `as-...` and `~/.modal.toml` holds them. The
    documented setup keeps both outside the repository; this fails if one ever
    lands in it."""
    token = re.compile(
        r"\b(ak|as)-[A-Za-z0-9]{16,}\b"                       # a literal Modal token
        r"|token_secret\s*[=:]\s*[\"']?[A-Za-z0-9]{8,}"        # one assigned inline
    )
    offenders = []
    for path in _tracked_files():
        if path.name == Path(__file__).name or not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".pkl", ".npz", ".ico"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover
            continue
        if token.search(text.lower()):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"Modal-shaped credential committed in: {offenders}"


def test_modal_toml_is_ignored():
    out = subprocess.run(["git", "check-ignore", "-q", ".modal.toml"], cwd=REPO)
    assert out.returncode == 0, ".modal.toml must be gitignored"
