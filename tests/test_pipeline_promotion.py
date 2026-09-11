"""Promotion: strict gates, shadow by default, and a champion pointer that only moves cleanly.

The central property, tested path by path: every way a promotion can be refused
leaves `/pipeline/champions/current.json` byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from _pipeline_fixtures import Harness, fake_measure, metrics_for
from naigos.pipeline import config as pcfg
from naigos.pipeline import coordinator, jobs, layout, leases, promotion, worker

GATES = pcfg.load_default()["gates"]


def _evaluation(**over):
    ev = {
        "snapshot_id": "s-20260911-0123456789ab",
        "heldout": {"seeds": [900001, 900002], "worlds_per_seed": 8, "training_seeds": [0, 1]},
        "verifier": {"episodes": 4, "ok": True, "mismatches": []},
        "candidate": metrics_for(0.8),
        "baselines": {"avoid_nap": metrics_for(0.55), "direct": metrics_for(0.3)},
        "champion": None,
    }
    ev.update(over)
    return ev


def _outcome(ev, problems=()):
    return promotion.decide(ev, GATES, provenance_problems=list(problems))


# --- the decision, gate by gate -------------------------------------------------------


def test_a_candidate_passing_every_gate_is_eligible():
    out = _outcome(_evaluation())
    assert out["outcome"] == promotion.ELIGIBLE
    gates = {c["gate"] for c in out["checks"]}
    assert {"provenance", "heldout_seeds", "verifier", "min_survival", "min_objective",
            "vs_avoid_nap.survival_rate", "vs_champion"} <= gates


@pytest.mark.parametrize("ev, gate", [
    (_evaluation(candidate=metrics_for(0.5)), "min_survival"),
    (_evaluation(candidate={**metrics_for(0.8), "objective_rate": 0.1}), "min_objective"),
    (_evaluation(verifier={"episodes": 4, "ok": False, "mismatches": ["phantom kill"]}), "verifier"),
    (_evaluation(heldout={"seeds": [0, 5], "worlds_per_seed": 8, "training_seeds": [0, 1]}), "heldout_seeds"),
    (_evaluation(baselines={"avoid_nap": metrics_for(0.9)}), "vs_avoid_nap.survival_rate"),
    (_evaluation(champion={"candidate_id": "c-20260910-nightly-0123456789ab",
                           "metrics": metrics_for(0.95)}), "vs_champion.survival_rate"),
    (_evaluation(champion={"candidate_id": "c-20260910-nightly-0123456789ab",
                           "metrics": {**metrics_for(0.8), "bounds_rate": 0.0}}), "vs_champion.bounds_rate"),
    (_evaluation(champion={"candidate_id": "c-20260910-nightly-0123456789ab",
                           "metrics": {**metrics_for(0.8), "exposure_early": 0.1}}), "vs_champion.exposure_early"),
])
def test_each_failed_gate_rejects(ev, gate):
    out = _outcome(ev)
    assert out["outcome"] == promotion.REJECTED
    assert any(c["gate"] == gate and c["status"] == promotion.FAIL for c in out["checks"])


def test_provenance_problems_reject():
    out = _outcome(_evaluation(), problems=["run: produced from a dirty working tree"])
    assert out["outcome"] == promotion.REJECTED


@pytest.mark.parametrize("ev", [
    _evaluation(candidate={k: v for k, v in metrics_for(0.8).items() if k != "shootdown_rate"}
                | {"survival_rate": 0.8},
                champion={"candidate_id": "c-x", "metrics": metrics_for(0.7)}),
    _evaluation(candidate={**metrics_for(0.8), "survival_rate": math.nan}),
    _evaluation(candidate={**metrics_for(0.8), "objective_rate": 1.7}),
    _evaluation(verifier={}),
    _evaluation(heldout={}),
    _evaluation(baselines={}),
    _evaluation(champion={"candidate_id": "c-x", "error": "shape mismatch", "metrics": None}),
])
def test_an_undecidable_gate_is_inconclusive_never_waived(ev):
    assert _outcome(ev)["outcome"] == promotion.INCONCLUSIVE


def test_a_fail_outranks_an_unknown():
    ev = _evaluation(candidate={**metrics_for(0.4), "objective_rate": math.inf})
    assert _outcome(ev)["outcome"] == promotion.REJECTED


def test_shadow_is_the_default_action():
    kw = dict(candidate_id="c-20260911-nightly-0123456789ab", evaluation=_evaluation(),
              evaluation_sha256="e", gates=GATES, config_version=2, config_digest="d", code={},
              provenance_problems=[])
    assert promotion.build_decision(auto_promote=False, **kw)["action"] == "shadow"
    assert promotion.build_decision(auto_promote=True, **kw)["action"] == "auto_promote"
    rejected = dict(kw, evaluation=_evaluation(candidate=metrics_for(0.1)))
    assert promotion.build_decision(auto_promote=True, **rejected)["action"] == "none"


# --- the pointer: atomic, compare-and-swap, append-only history -------------------------


def _ckpt():
    return {"file": "ckpt_000002.pkl", "sha256": "a" * 64, "iteration": 2}


def _publish(lay, cid, expected, **kw):
    return promotion.publish_champion(
        lay, candidate_id=cid, snapshot_id="s-20260911-0123456789ab", checkpoint=_ckpt(),
        decision_sha256="d", evaluation_sha256="e", mode="manual", actor="op",
        expected_generation=expected, code={}, **kw)


C1, C2, C3 = ("c-20260911-nightly-000000000001", "c-20260912-nightly-000000000002",
              "c-20260913-nightly-000000000003")


def test_publish_increments_generation_and_links_the_previous(tmp_path):
    lay = layout.Layout(tmp_path)
    p1 = _publish(lay, C1, 0)
    p2 = _publish(lay, C2, 1)
    assert (p1["generation"], p2["generation"]) == (1, 2)
    assert p2["previous"] == {"generation": 1, "candidate_id": C1}
    assert promotion.read_champion(lay)["candidate_id"] == C2
    assert [h["candidate_id"] for h in promotion.champion_history(lay)] == [C1, C2]
    assert not [p for p in lay.champion_pointer.parent.iterdir() if ".tmp" in p.name]


def test_a_stale_generation_is_refused_and_changes_nothing(tmp_path):
    lay = layout.Layout(tmp_path)
    _publish(lay, C1, 0)
    before = lay.champion_pointer.read_bytes()
    with pytest.raises(promotion.ChampionConflict):
        _publish(lay, C2, 0)  # based on a read from before C1 landed
    assert lay.champion_pointer.read_bytes() == before
    assert len(promotion.champion_history(lay)) == 1


def test_an_interrupted_publish_does_not_wedge_the_pointer(tmp_path):
    """History written, pointer not replaced (container died between the two)."""
    lay = layout.Layout(tmp_path)
    _publish(lay, C1, 0)
    layout.write_once_json(lay.champion_generation(2), {"generation": 2, "candidate_id": C2})
    p = _publish(lay, C3, 1)
    assert p["generation"] == 3 and promotion.read_champion(lay)["candidate_id"] == C3


def test_atomic_replace_means_readers_never_see_a_partial_pointer(tmp_path, monkeypatch):
    lay = layout.Layout(tmp_path)
    _publish(lay, C1, 0)
    before = lay.champion_pointer.read_bytes()

    def crash(*a, **k):
        raise OSError("container killed mid-rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        _publish(lay, C2, 1)
    monkeypatch.undo()
    assert lay.champion_pointer.read_bytes() == before  # the old champion is still whole


def test_the_pointer_cannot_name_a_path_outside_its_candidate(tmp_path):
    lay = layout.Layout(tmp_path)
    for bad in ("../../etc/passwd", "ckpt_1.pkl", None):
        with pytest.raises(layout.LayoutError):
            promotion.champion_checkpoint_path(lay, {"candidate_id": C1, "checkpoint": {"file": bad}})


# --- end to end: every refusal path leaves the champion byte-for-byte unchanged ---------


@pytest.fixture
def two(tmp_path):
    """A champion (generation 1) and a second, eligible shadow candidate."""
    h = Harness(tmp_path)
    from test_pipeline_flow import _first_candidate, _new_day_candidate

    first = _first_candidate(h)
    worker.promote(h.svc, first, actor="op", rerun_measure=h.measure)
    second = _new_day_candidate(h, 0.9)
    assert json.loads((h.lay.candidate_dir(second) / "decision.json").read_text())["outcome"] == "eligible"
    return h, first, second


def _refused(h, fn):
    before = h.champion_bytes()
    with pytest.raises((promotion.PromotionRefused, leases.LeaseError)) as e:
        fn()
    assert h.champion_bytes() == before
    return e.value


def test_manual_promotion_revalidates_and_publishes(two):
    h, first, second = two
    p = worker.promote(h.svc, second, actor="op", rerun_measure=h.measure, note="reviewed")
    assert p["generation"] == 2 and p["candidate_id"] == second and p["mode"] == "manual"
    assert p["previous"]["candidate_id"] == first
    status = {c["candidate_id"]: c for c in coordinator.status_report(h.lay)["candidates"]}
    assert status[second]["status"] == jobs.PROMOTED and status[second]["is_champion"]
    assert status[first]["status"] == jobs.PROMOTED and not status[first]["is_champion"]


def test_refused_when_the_re_run_does_not_reproduce(two):
    h, _, second = two
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op",
                                           rerun_measure=fake_measure(jitter=1e-3)))
    assert any("reproduction" in p for p in e.problems)


def test_refused_when_the_checkpoint_changed(two):
    h, _, second = two
    ckpt = next(h.lay.candidate_dir(second).glob("ckpt_*.pkl"))
    ckpt.write_bytes(ckpt.read_bytes() + b"\0")
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))
    assert any("checkpoint" in p for p in e.problems)


def test_refused_when_the_evaluation_record_changed(two):
    h, _, second = two
    p = h.lay.candidate_dir(second) / "evaluation.json"
    doc = json.loads(p.read_text())
    doc["candidate"]["survival_rate"] = 0.99
    os.chmod(p, 0o644)
    p.write_text(json.dumps(doc))
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))
    assert any("does not hash" in p for p in e.problems)


def test_refused_when_the_snapshot_changed(two):
    h, _, second = two
    sid = json.loads((h.lay.candidate_dir(second) / "candidate.json").read_text())["parents"]["snapshot_id"]
    (h.lay.snapshot_dir(sid) / "DATA.md").write_text("edited after publication")
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))
    assert any("content hash differs" in p for p in e.problems)


def test_refused_when_the_candidate_was_rejected(tmp_path):
    h = Harness(tmp_path)
    from test_pipeline_flow import _first_candidate

    h.qualities.append(0.2)
    cid = _first_candidate(h)
    e = _refused(h, lambda: worker.promote(h.svc, cid, actor="op", rerun_measure=h.measure))
    assert any("rejected" in p for p in e.problems)


def test_refused_when_the_candidate_was_inconclusive(tmp_path):
    h = Harness(tmp_path)
    from test_pipeline_flow import _first_candidate

    h.measure = fake_measure(drop_metric="exposure_early")
    first = _first_candidate(h)
    assert json.loads((h.lay.candidate_dir(first) / "decision.json").read_text())["outcome"] == "eligible"
    # exposure_early only matters against a champion; make one, then evaluate a second.
    worker.promote(h.svc, first, actor="op", rerun_measure=fake_measure(drop_metric="exposure_early"))
    from test_pipeline_flow import _new_day_candidate

    second = _new_day_candidate(h, 0.9)
    assert json.loads((h.lay.candidate_dir(second) / "decision.json").read_text())["outcome"] == "inconclusive"
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))
    assert any("inconclusive" in p for p in e.problems)


def test_refused_when_the_champion_has_improved_since(two):
    """Re-validation re-runs the gates against the champion as it is *now*."""
    h, first, second = two
    third = None
    from test_pipeline_flow import _new_day_candidate

    third = _new_day_candidate(h, 0.97)
    worker.promote(h.svc, third, actor="op", rerun_measure=h.measure)
    e = _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))
    assert any("vs_champion" in p for p in e.problems)


def test_refused_while_another_promoter_holds_the_champion_lease(two):
    h, _, second = two
    h.leases.acquire("champion", "another operator")
    _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure))


def test_refused_on_a_stale_expected_generation(two):
    h, _, second = two
    _refused(h, lambda: worker.promote(h.svc, second, actor="op", rerun_measure=h.measure,
                                       expected_generation=0))


def test_refused_when_already_champion(two):
    h, first, _ = two
    _refused(h, lambda: worker.promote(h.svc, first, actor="op", rerun_measure=h.measure))


def test_refused_for_an_unknown_or_malformed_candidate(two):
    h, _, _ = two
    before = h.champion_bytes()
    with pytest.raises(layout.LayoutError):
        worker.promote(h.svc, "../../champions", actor="op", rerun_measure=h.measure)
    _refused(h, lambda: worker.promote(h.svc, "c-20260101-nightly-ffffffffffff", actor="op",
                                       rerun_measure=h.measure))
    assert h.champion_bytes() == before


def test_auto_promote_false_never_changes_the_champion(two):
    h, first, second = two
    assert json.loads(h.lay.config_current.read_text())["auto_promote"] is False
    before = h.champion_bytes()
    for q in (0.95, 0.99):
        cid = __import__("test_pipeline_flow")._new_day_candidate(h, q)
        assert json.loads((h.lay.candidate_dir(cid) / "decision.json").read_text())["action"] == "shadow"
    assert h.champion_bytes() == before


def test_auto_promote_true_promotes_only_eligible_candidates(tmp_path):
    h = Harness(tmp_path, config_changes={"auto_promote": True})
    from test_pipeline_flow import _first_candidate, _new_day_candidate

    first = _first_candidate(h)
    assert promotion.read_champion(h.lay)["candidate_id"] == first
    assert promotion.read_champion(h.lay)["mode"] == "auto"
    before = h.champion_bytes()
    worse = _new_day_candidate(h, 0.5)
    assert json.loads((h.lay.candidate_dir(worse) / "decision.json").read_text())["outcome"] == "rejected"
    assert h.champion_bytes() == before
    better = _new_day_candidate(h, 0.9)
    assert promotion.read_champion(h.lay)["candidate_id"] == better
    assert jobs.read(h.lay, jobs.job_key("evaluate", better))["status"] == jobs.PROMOTED


# --- rollback --------------------------------------------------------------------------


def test_rollback_repoints_to_an_earlier_generation_as_a_new_one(two):
    h, first, second = two
    worker.promote(h.svc, second, actor="op", rerun_measure=h.measure)
    p = worker.rollback(h.svc, 1, actor="op", note="second misbehaved in review")
    assert p["generation"] == 3 and p["candidate_id"] == first
    assert p["mode"] == "rollback" and p["rollback_of"] == 1
    assert [g["candidate_id"] for g in promotion.champion_history(h.lay)] == [first, second, first]


def test_rollback_refuses_a_target_whose_artifacts_changed(two):
    h, first, second = two
    worker.promote(h.svc, second, actor="op", rerun_measure=h.measure)
    ckpt = next(h.lay.candidate_dir(first).glob("ckpt_*.pkl"))
    ckpt.write_bytes(b"swapped")
    e = _refused(h, lambda: worker.rollback(h.svc, 1, actor="op"))
    assert any("checkpoint" in p for p in e.problems)


def test_rollback_refuses_an_unknown_generation(two):
    h, _, _ = two
    _refused(h, lambda: worker.rollback(h.svc, 42, actor="op"))
