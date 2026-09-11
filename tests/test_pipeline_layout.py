"""Pipeline layout, write disciplines, cron windows and strict configuration.

Offline: no Modal, no network, no cache.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from naigos.pipeline import config as pcfg
from naigos.pipeline import layout, schedule
from naigos.rl import runmeta


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


# --- ids and paths -----------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "../etc", "s-20260911-../../x", "s-20260911-ABCDEF123456", "s-2026091-abcdef123456",
    "s-20260911-abcdef12345", "/abs", "", None, 7, "s-20260911-abcdef123456/..",
])
def test_snapshot_ids_are_strict(bad):
    with pytest.raises(layout.LayoutError):
        layout.validate_snapshot_id(bad)


@pytest.mark.parametrize("bad", [
    "c-20260911-nightly-../../../x", "c-20260911-evil-abcdef123456", "pipeline",
    "c-20260911-nightly-abcdef123456/../../s", "c-20260911-nightly-abcdef12345g",
])
def test_candidate_ids_are_strict(bad):
    with pytest.raises(layout.LayoutError):
        layout.validate_candidate_id(bad)


def test_generated_ids_are_valid_run_names():
    """A candidate directory is a run directory, so its ID must satisfy both rules."""
    key = "0123456789abcdef" * 4
    cid = layout.candidate_id("20260911", "nightly", key)
    assert cid == "c-20260911-nightly-0123456789ab"
    assert runmeta.validate_run_name(cid) == cid
    assert layout.kind_of_candidate(cid) == "nightly"
    assert layout.snapshot_id("20260911T061500Z", key) == "s-20260911T061500Z-0123456789ab"


def test_paths_cannot_escape_the_pipeline_root(tmp_path):
    lay = layout.Layout(tmp_path)
    assert lay.snapshot_dir("s-20260911-0123456789ab").parent == tmp_path / "pipeline" / "snapshots"
    with pytest.raises(layout.LayoutError):
        lay.candidate_dir("../../outside")
    with pytest.raises(layout.LayoutError):
        lay.lock_path("candidate:../../x")
    with pytest.raises(layout.LayoutError):
        lay.lock_path("Candidate:x")
    with pytest.raises(layout.LayoutError):
        lay._inside(tmp_path / "pipeline" / ".." / "elsewhere")
    assert lay.lock_path("candidate:c-20260911-nightly-0123456789ab").name == \
        "candidate__c-20260911-nightly-0123456789ab.json"


def test_pipeline_is_a_reserved_run_name():
    with pytest.raises(runmeta.RunNameError):
        runmeta.validate_run_name("pipeline")
    with pytest.raises(runmeta.RunNameError):
        runmeta.validate_run_name("Pipeline")
    assert runmeta.validate_run_name("pipeline-2") == "pipeline-2"


# --- write disciplines -------------------------------------------------------


def test_write_once_refuses_a_second_write_and_keeps_the_bytes(tmp_path):
    path = tmp_path / "rec" / "decision.json"
    layout.write_once_json(path, {"a": 1})
    before = path.read_bytes()
    with pytest.raises(layout.ImmutableRecordError):
        layout.write_once_json(path, {"a": 2})
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["decision.json"]  # no temp left behind


def test_atomic_write_replaces_and_leaves_no_temporaries(tmp_path):
    path = tmp_path / "current.json"
    layout.atomic_write_json(path, {"generation": 1})
    layout.atomic_write_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["current.json"]


def test_tree_digest_names_the_bytes_it_vouches_for(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"x")
    (tmp_path / "y.json").write_text("{}")
    (tmp_path / "y.json.tmp-1-ab").write_text("half")  # interrupted write, not content
    d1 = layout.tree_digest(tmp_path)
    assert set(d1["files"]) == {"a/x.bin", "y.json"}
    (tmp_path / "a" / "x.bin").write_bytes(b"z")
    d2 = layout.tree_digest(tmp_path)
    assert d1["sha256"] != d2["sha256"]
    assert d1["files"]["y.json"] == d2["files"]["y.json"]


def test_digest_is_key_order_independent():
    assert layout.digest({"a": 1, "b": [1, 2]}) == layout.digest({"b": [1, 2], "a": 1})


# --- cron --------------------------------------------------------------------


def test_cron_windows_are_stable_across_duplicate_invocations():
    daily = "0 6 * * *"
    first = schedule.previous_fire(daily, _utc(2026, 9, 11, 6, 0, 20))
    late = schedule.previous_fire(daily, _utc(2026, 9, 11, 17, 45))
    assert first == late == _utc(2026, 9, 11, 6, 0)
    assert schedule.window_label(first) == "20260911"
    assert schedule.previous_fire(daily, _utc(2026, 9, 11, 5, 59)) == _utc(2026, 9, 10, 6, 0)
    assert schedule.next_fire(daily, _utc(2026, 9, 11, 6, 0)) == _utc(2026, 9, 12, 6, 0)


def test_weekly_cron_on_sunday():
    weekly = "0 10 * * 0"
    assert schedule.next_fire(weekly, _utc(2026, 9, 11, 12, 0)) == _utc(2026, 9, 13, 10, 0)
    assert schedule.previous_fire(weekly, _utc(2026, 9, 11, 12, 0)) == _utc(2026, 9, 6, 10, 0)
    assert schedule.parse_cron("0 10 * * 7").weekdays == frozenset({0})


def test_cron_steps_lists_and_ranges():
    c = schedule.parse_cron("*/15 1-3,22 * * 1-5")
    assert c.minutes == frozenset({0, 15, 30, 45})
    assert c.hours == frozenset({1, 2, 3, 22})
    assert c.weekdays == frozenset({1, 2, 3, 4, 5})


@pytest.mark.parametrize("bad", ["", "* * * *", "61 * * * *", "* 24 * * *", "a b c d e",
                                 "*/0 * * * *", "5-1 * * * *", "0 0 31 2 *"])
def test_bad_cron_is_refused(bad):
    with pytest.raises(schedule.CronError):
        c = schedule.parse_cron(bad)
        schedule.next_fire(c, _utc(2026, 1, 1))


# --- configuration -----------------------------------------------------------


def _volume_doc(**changes):
    doc = {k: v for k, v in pcfg.load_default().items() if k != "schedule"}
    doc.update(changes)
    return doc


def test_the_shipped_default_is_valid_and_conservative():
    cfg = pcfg.load_default()
    assert cfg["auto_promote"] is False
    assert cfg["paused"] is False
    assert cfg["schedule"]["timezone"] == "UTC"
    assert not set(cfg["evaluation"]["heldout_seeds"]) & set(pcfg.training_seeds(cfg))
    assert cfg["evaluation"]["verifier_episodes"] >= 1


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(surprise=1), "unknown key"),
    (lambda d: d.pop("gates"), "missing key"),
    (lambda d: d.update(auto_promote="yes"), "true/false"),
    (lambda d: d.update(aoi="somewhere_else"), "config.aoi"),
    (lambda d: d["snapshot"].update(refresh_sources=["some_random_blog"]), "not fetchable"),
    (lambda d: d["snapshot"].update(refresh_sources=["radar_theory"]), "not fetchable"),
    (lambda d: d["training"]["nightly"].update(profile="huge"), "profile"),
    (lambda d: d["training"]["nightly"].update(overrides={"iterations": -1}), "positive"),
    (lambda d: d["training"]["nightly"].update(overrides={"learning_rate": 1}), "cannot override"),
    (lambda d: d["evaluation"].update(heldout_seeds=[0, 5]), "overlap training seeds"),
    (lambda d: d["evaluation"].update(heldout_seeds=[5, 5]), "duplicate"),
    (lambda d: d["evaluation"].update(heldout_seeds=[]), "1-256"),
    (lambda d: d["evaluation"].update(verifier_episodes=0), "verifier_episodes"),
    (lambda d: d["gates"].update(min_survival_rate=1.5), "min_survival_rate"),
    (lambda d: d["gates"]["champion_tolerance"].pop("shootdown_rate"), "missing key"),
    (lambda d: d.update(version=True), "config.version"),
    (lambda d: d.update(schedule={"snapshot_cron": "0 6 * * *"}), "deployment-defined"),
])
def test_invalid_volume_configs_are_refused(mutate, message):
    doc = _volume_doc()
    mutate(doc)
    with pytest.raises(pcfg.ConfigError, match=message):
        pcfg.validate(doc, source="volume")


def test_the_default_must_carry_a_valid_schedule():
    doc = pcfg.load_default()
    doc["schedule"]["nightly_cron"] = "every night"
    with pytest.raises(pcfg.ConfigError, match="nightly_cron"):
        pcfg.validate(doc, source="default")
    doc = pcfg.load_default()
    doc["schedule"]["timezone"] = "America/New_York"
    with pytest.raises(pcfg.ConfigError, match="UTC"):
        pcfg.validate(doc, source="default")


def test_effective_config_falls_back_to_default_only_when_absent(tmp_path):
    lay = layout.Layout(tmp_path)
    cfg, source = pcfg.load_effective(lay)
    assert source == "default"

    lay.config_current.parent.mkdir(parents=True)
    lay.config_current.write_text(json.dumps(_volume_doc(paused=True)))
    cfg, source = pcfg.load_effective(lay)
    assert source == "volume" and cfg["paused"] is True
    assert cfg["schedule"] == pcfg.deployed_schedule()

    # A broken file must not silently lift a pause by falling back to the default.
    lay.config_current.write_text("{not json")
    with pytest.raises(pcfg.ConfigError):
        pcfg.load_effective(lay)
    lay.config_current.write_text(json.dumps(_volume_doc(paused="maybe")))
    with pytest.raises(pcfg.ConfigError):
        pcfg.load_effective(lay)


def test_pause_does_not_change_any_idempotency_digest():
    cfg = pcfg.load_default()
    paused = copy.deepcopy(cfg)
    paused.update(paused=True, pause_reason="budget", version=9)
    for stage in ("snapshot", "nightly", "weekly"):
        assert pcfg.stage_digest(cfg, stage) == pcfg.stage_digest(paused, stage)
    assert pcfg.config_digest(cfg) == pcfg.config_digest(paused)


def test_stage_digests_follow_only_their_own_inputs():
    cfg = pcfg.load_default()
    gates = copy.deepcopy(cfg)
    gates["gates"]["min_survival_rate"] = 0.9
    assert pcfg.stage_digest(cfg, "nightly") == pcfg.stage_digest(gates, "nightly")
    assert pcfg.config_digest(cfg) != pcfg.config_digest(gates)
    training = copy.deepcopy(cfg)
    training["training"]["nightly"]["seed"] = 7
    assert pcfg.stage_digest(cfg, "snapshot") == pcfg.stage_digest(training, "snapshot")
    assert pcfg.stage_digest(cfg, "nightly") != pcfg.stage_digest(training, "nightly")


def test_versions_are_published_with_compare_and_swap(tmp_path):
    lay = layout.Layout(tmp_path)
    base = pcfg.load_default()
    v2 = pcfg.next_version(base, {"paused": True, "pause_reason": "holiday"}, updated_by="op")
    assert v2["version"] == 2 and "schedule" not in v2
    pcfg.publish(lay, v2, expected_version=None)
    assert json.loads(lay.config_current.read_text())["paused"] is True
    assert lay.config_version(2).exists()

    v3 = pcfg.next_version(v2, {"paused": False, "pause_reason": None}, updated_by="op")
    stale = pcfg.next_version(v2, {"auto_promote": True}, updated_by="other")
    pcfg.publish(lay, v3, expected_version=2)
    with pytest.raises(pcfg.ConfigConflict):
        pcfg.publish(lay, stale, expected_version=2)
    assert json.loads(lay.config_current.read_text())["auto_promote"] is False
    with pytest.raises(pcfg.ConfigError):
        pcfg.next_version(v3, {"schedule": {}}, updated_by="op")


def test_free_text_config_fields_are_redacted():
    # Assembled at runtime so this file itself stays clean under the
    # repository-wide credential scan in tests/test_run_status.py.
    value = "q7" + "Lm2" + "Vx9" + "Rt4" + "Kp"
    doc = _volume_doc(note="MODAL_TOKEN_" + "SECRET=" + value)
    assert value not in pcfg.validate(doc, source="volume")["note"]
