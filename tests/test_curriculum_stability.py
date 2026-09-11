"""The red curriculum moves on evidence, and says what evidence it moved on.

The failure this file guards against is not hypothetical arithmetic. Survival is
measured over `eval_worlds` worlds of `n_blue` agents -- 64 x 4 = 256 Bernoulli
sorties by default -- so one evaluation of a policy whose true survival is 0.72
has a standard error near 0.03 and lands above `promote_survival` (0.75) often
enough to matter. Under the single-evaluation rule that draw was a promotion,
red got harder, and the next evaluation measured a policy that had been moved
into a task it had not earned. The reverse draw demoted a policy that was fine.

So three claims are tested here:

  * one outlier does not move the level, in either direction;
  * repeated evidence does;
  * every row says truthfully what was measured, what was smoothed, how many
    evaluations it rests on, and why the level did or did not move.

Plus the two things that make it auditable rather than merely stable: the state
survives a JSON round trip exactly, and a resumed run makes the same decisions
an uninterrupted one would.

Everything here is host-side bookkeeping, so it runs in milliseconds and never
touches JAX beyond the import. The two end-to-end tests at the bottom are the
exception and are kept deliberately tiny.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from naigos.rl.red_team import (
    CURRICULUM_REASONS,
    CurriculumError,
    CurriculumState,
    RedCurriculum,
)
from naigos.rl.train import curriculum_state_from_history

# Promote above 0.75, demote below 0.35, 0.05 per step -- the shipped defaults.
CUR = RedCurriculum()


def drive(cur: RedCurriculum, samples, level: float = 0.5, state=None):
    """Feed `samples` in one at a time, ticking the curriculum after each.

    Mirrors the tightest schedule `train.run` can be configured with
    (`curriculum_every == eval_every`): observe, then decide. Returns the
    decision list so a test can assert on the whole trajectory.
    """
    state = state if state is not None else cur.initial_state(level)
    out = []
    for i, x in enumerate(samples, 1):
        state = cur.observe(state, x, iteration=i)
        d = cur.decide(state, iteration=i)
        state = d.state
        out.append(d)
    return out


def levels(decisions):
    return [d.level_after for d in decisions]


def reasons(decisions):
    return [d.reason for d in decisions]


# --- one evaluation is not evidence -----------------------------------------


#: A policy that has been evaluating at 0.50 survival -- comfortably inside the
#: dead band -- for long enough to fill the window. This is the state the outlier
#: tests interrogate, because it is the state a real run spends most of its time
#: in: the gate has long since been met, so nothing but the statistic itself is
#: standing between one bad draw and a level change.
STEADY = [0.50] * CUR.window_size


def test_a_single_lucky_evaluation_does_not_promote():
    """The headline claim. A settled policy draws one 0.95 and the level does
    not move, because the statistic the threshold sees is the window mean."""
    d = drive(CUR, [*STEADY, 0.95])
    assert set(levels(d)) == {0.5}, "an outlier promoted the level"
    assert d[-1].raw_survival == 0.95, "the outlier must still be recorded truthfully"
    assert d[-1].smoothed_survival == pytest.approx(0.59)
    assert d[-1].reason == "held"


def test_a_single_unlucky_evaluation_does_not_demote():
    """The same bug in the other direction, which is the more expensive one:
    a spurious demotion throws away difficulty the policy had earned."""
    d = drive(CUR, [*STEADY, 0.02])
    assert set(levels(d)) == {0.5}
    assert d[-1].raw_survival == 0.02
    assert d[-1].smoothed_survival == pytest.approx(0.404)
    assert d[-1].reason == "held"


def test_the_legacy_rule_is_the_one_the_outlier_moves():
    """Proof that the two tests above are testing something. Under the rule this
    replaces, each of those identical sample sequences moves the level on the
    final draw."""
    legacy = RedCurriculum.legacy()
    assert legacy.legacy_single_evaluation is True
    up = drive(legacy, [*STEADY, 0.95])
    assert up[-1].reason == "promoted" and up[-1].level_after == pytest.approx(0.55)
    down = drive(legacy, [*STEADY, 0.02])
    assert down[-1].reason == "demoted" and down[-1].level_after == pytest.approx(0.45)
    # and the stateless entry point behaves as it always did
    assert legacy.update(0.5, 0.95) == pytest.approx(0.55)
    assert legacy.update(0.5, 0.10) == pytest.approx(0.45)


def test_an_outlier_is_still_only_damped_not_ignored():
    """Honest about the limit. A window mean is not a robust statistic: the
    outlier moves it by 1/window_size, so an outlier arriving when the policy is
    already close to a threshold can still tip it. What the window buys is that
    ONE draw from a policy in the middle of the dead band cannot, which is the
    case that was actually breaking runs."""
    d = drive(CUR, [*[0.40] * CUR.window_size, 0.02])
    assert d[-1].smoothed_survival == pytest.approx(0.324)
    assert d[-1].reason == "demoted"


# --- repeated evidence does move it ------------------------------------------


def test_repeated_evidence_promotes():
    d = drive(CUR, [0.90, 0.90, 0.90])
    assert reasons(d) == ["insufficient_evidence", "insufficient_evidence", "promoted"]
    assert d[-1].level_before == 0.5 and d[-1].level_after == pytest.approx(0.55)


def test_repeated_evidence_demotes():
    d = drive(CUR, [0.10, 0.10, 0.10])
    assert reasons(d) == ["insufficient_evidence", "insufficient_evidence", "demoted"]
    assert d[-1].level_after == pytest.approx(0.45)


def test_a_level_change_clears_the_window():
    """Samples taken before a change measured a different task: at a new level
    red's detection reach, lethal radius, latency, speed and active count are all
    different. Carrying them across the boundary would smooth two difficulties
    together, and the promotion after next would rest on evidence about the
    theatre the policy has already left."""
    d = drive(CUR, [0.90, 0.90, 0.90, 0.90])
    promoted = d[2]
    assert promoted.reason == "promoted"
    assert promoted.state.window == ()
    assert promoted.state.ewma is None
    assert promoted.state.n_since_change == 0
    # the lifetime count is NOT reset: it is the run's evaluation ledger
    assert promoted.state.n_observations == 3
    # so the very next evaluation cannot promote again
    assert d[3].reason == "insufficient_evidence"
    assert d[3].level_after == pytest.approx(0.55)


def test_sustained_evidence_promotes_repeatedly_at_the_gated_rate():
    """Nine strong evaluations, a gate of three: three promotions, not nine."""
    d = drive(CUR, [0.95] * 9, level=0.0)
    assert reasons(d).count("promoted") == 3
    assert d[-1].level_after == pytest.approx(0.15)


# --- the gates ---------------------------------------------------------------


def test_the_minimum_evaluation_gate_is_configurable_and_binding():
    strict = dataclasses.replace(CUR, min_evaluations=5, window_size=5)
    d = drive(strict, [0.95] * 5)
    assert reasons(d) == ["insufficient_evidence"] * 4 + ["promoted"]
    loose = dataclasses.replace(CUR, min_evaluations=1, window_size=1)
    assert drive(loose, [0.95])[0].reason == "promoted"


def test_insufficient_evidence_is_reported_as_itself_not_as_a_hold():
    """"We did not look" and "we looked and it was inside the dead band" are
    different facts about a run, and only one of them means the curriculum is
    working as designed."""
    d = drive(CUR, [0.95, 0.95])
    assert reasons(d) == ["insufficient_evidence", "insufficient_evidence"]
    # the thresholds were not consulted, but the statistic is still on the row
    assert d[-1].smoothed_survival == pytest.approx(0.95)
    assert d[-1].window_n == 2


def test_a_statistic_inside_the_dead_band_holds():
    d = drive(CUR, [0.50, 0.55, 0.60])
    assert d[-1].reason == "held"
    assert d[-1].level_after == 0.5


def test_thresholds_are_configurable():
    tight = dataclasses.replace(CUR, promote_survival=0.55, demote_survival=0.45)
    assert drive(tight, [0.60] * 3)[-1].reason == "promoted"
    assert drive(tight, [0.40] * 3)[-1].reason == "demoted"
    assert drive(tight, [0.50] * 3)[-1].reason == "held"


def test_level_bounds_clamp_and_report_a_hold_not_a_promotion():
    """A level pinned at a bound did not move, so the row must not claim it did
    -- but the statistic that wanted to move it is still recorded."""
    top = drive(CUR, [0.95] * 3, level=1.0)[-1]
    assert top.level_after == 1.0 and top.reason == "held"
    assert top.smoothed_survival == pytest.approx(0.95)
    bottom = drive(CUR, [0.05] * 3, level=0.0)[-1]
    assert bottom.level_after == 0.0 and bottom.reason == "held"
    assert bottom.smoothed_survival == pytest.approx(0.05)


def test_a_level_near_a_bound_steps_only_as_far_as_the_bound():
    d = drive(CUR, [0.95] * 3, level=0.98)[-1]
    assert d.level_after == 1.0 and d.reason == "promoted"


# --- the statistic itself ----------------------------------------------------


def test_the_rolling_window_is_a_mean_of_the_last_window_size_samples():
    cur = dataclasses.replace(CUR, window_size=3, min_evaluations=99)
    st = cur.initial_state()
    for x in (0.1, 0.2, 0.3, 0.4):
        st = cur.observe(st, x)
    assert st.window == (0.2, 0.3, 0.4), "the window must forget, not accumulate"
    assert cur.smoothed(st) == pytest.approx(0.3)
    assert cur.evidence_n(st) == 3


def test_the_ewma_mode_weights_recent_evaluations_more():
    cur = dataclasses.replace(CUR, smoothing="ewma", ewma_alpha=0.5, min_evaluations=99)
    st = cur.initial_state()
    st = cur.observe(st, 0.0)
    assert cur.smoothed(st) == pytest.approx(0.0), "the first sample seeds the accumulator"
    st = cur.observe(st, 1.0)
    assert cur.smoothed(st) == pytest.approx(0.5)
    st = cur.observe(st, 1.0)
    assert cur.smoothed(st) == pytest.approx(0.75)
    # for an EWMA the window is only a viewport; the evidence is the count
    assert cur.evidence_n(st) == 3


def test_an_unmeasured_statistic_is_none_and_never_zero():
    """`demote_survival` is 0.35. A statistic that read 0.0 before the first
    evaluation would demote a run for not having been measured yet."""
    st = CUR.initial_state(0.5)
    assert CUR.smoothed(st) is None and CUR.raw(st) is None
    d = CUR.decide(st)
    assert d.reason == "insufficient_evidence"
    assert d.level_after == 0.5
    assert d.smoothed_survival is None and d.raw_survival is None


def test_smoothing_none_thresholds_the_newest_measurement():
    cur = dataclasses.replace(CUR, smoothing="none", min_evaluations=99)
    st = cur.observe(cur.observe(cur.initial_state(), 0.1), 0.9)
    assert cur.smoothed(st) == pytest.approx(0.9)


# --- telemetry ---------------------------------------------------------------


def test_telemetry_records_raw_and_smoothed_truthfully_and_separately():
    d = drive(CUR, [0.20, 0.40, 0.90])[-1]
    t = d.telemetry()
    assert t["curriculum_raw_survival"] == pytest.approx(0.90)
    assert t["curriculum_smoothed_survival"] == pytest.approx(0.5)
    assert t["curriculum_raw_survival"] != t["curriculum_smoothed_survival"]
    assert t["curriculum_window_n"] == 3
    assert t["curriculum_evals_total"] == 3
    assert t["curriculum_evals_since_change"] == 3
    assert t["curriculum_min_evaluations"] == CUR.min_evaluations
    assert t["curriculum_smoothing"] == "window"
    assert t["red_level_before"] == 0.5 and t["red_level_after"] == 0.5
    assert t["curriculum_reason"] in CURRICULUM_REASONS


def test_telemetry_does_not_overwrite_the_level_a_row_was_measured_at():
    """`red_level` on an evaluation row means "the level these metrics were
    measured at". A tick merged onto that row must not restamp it with the level
    it is about to move to, or every promotion row would attribute the old
    theatre's metrics to the new theatre."""
    t = drive(CUR, [0.95] * 3)[-1].telemetry()
    assert "red_level" not in t
    assert t["red_level_before"] == 0.5 and t["red_level_after"] == pytest.approx(0.55)


def test_every_telemetry_value_is_json_serializable_and_round_trips():
    for d in drive(CUR, [0.9, 0.2, 0.9, 0.95, 0.95, 0.95]):
        t = d.telemetry()
        again = json.loads(json.dumps(t))
        assert again == t
        assert CurriculumState.from_dict(again["curriculum_state"]) == d.state


# --- state, serialization and resume -----------------------------------------


def test_state_round_trips_through_json_exactly():
    """Not approximately: a resumed run whose accumulator differs in the last
    bit can cross a threshold on a different evaluation to the run it claims to
    be continuing."""
    cur = dataclasses.replace(CUR, smoothing="ewma")
    st = cur.initial_state(0.35)
    for x in (0.123456789012345, 0.98765432109876, 0.5):
        st = cur.observe(st, x, iteration=7)
    back = CurriculumState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert back == st
    assert cur.smoothed(back) == cur.smoothed(st)


def test_a_malformed_state_is_refused_rather_than_defaulted():
    with pytest.raises(CurriculumError, match="missing"):
        CurriculumState.from_dict({"level": 0.5})
    with pytest.raises(CurriculumError, match="mapping"):
        CurriculumState.from_dict([0.5])


def test_the_state_is_read_back_from_the_newest_history_row_that_has_one():
    early = CUR.observe(CUR.initial_state(0.2), 0.4)
    late = CUR.observe(early, 0.6)
    history = [
        {"iter": 0, "phase": "baseline"},
        {"iter": 1, "curriculum_state": early.to_dict()},
        {"iter": 2, "curriculum_state": late.to_dict()},
        {"iter": 3},  # a row with no curriculum event must not hide the state
    ]
    assert curriculum_state_from_history(json.loads(json.dumps(history))) == late
    assert curriculum_state_from_history([{"iter": 1}]) is None
    assert curriculum_state_from_history([]) is None


def test_interrupting_and_resuming_the_state_makes_the_same_decisions():
    """The whole point of carrying the state: the decision sequence must not
    depend on where the run was cut in half."""
    samples = [0.80, 0.40, 0.90, 0.95, 0.20, 0.10, 0.05, 0.99, 0.99, 0.99]
    straight = drive(CUR, samples, level=0.5)

    first = drive(CUR, samples[:4], level=0.5)
    # what a checkpoint carries: the last history row's curriculum_state, having
    # been through JSON on its way to `history.json`
    carried = CurriculumState.from_dict(
        json.loads(json.dumps(first[-1].state.to_dict()))
    )
    second = drive(CUR, samples[4:], state=carried)

    resumed = first + second
    assert reasons(resumed) == reasons(straight)
    assert levels(resumed) == levels(straight)
    assert [d.smoothed_survival for d in resumed] == [d.smoothed_survival for d in straight]


# --- configuration validation ------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"smoothing": "median"}, "smoothing must be one of"),
        ({"promote_survival": 1.5}, r"promote_survival must be a survival rate"),
        ({"demote_survival": -0.1}, r"demote_survival must be a survival rate"),
        ({"promote_survival": 0.3, "demote_survival": 0.4}, "must be below promote_survival"),
        ({"promote_survival": 0.4, "demote_survival": 0.4}, "must be below promote_survival"),
        ({"step_size": 0.0}, "step_size must be positive"),
        ({"step_size": -0.1}, "step_size must be positive"),
        ({"window_size": 0}, "window_size must be at least 1"),
        ({"ewma_alpha": 0.0}, r"ewma_alpha must be in \(0, 1\]"),
        ({"ewma_alpha": 1.5}, r"ewma_alpha must be in \(0, 1\]"),
        ({"min_evaluations": 0}, "min_evaluations must be at least 1"),
        ({"n_lo": 20, "n_hi": 4}, "must not exceed"),
    ],
)
def test_an_unusable_curriculum_is_refused_at_construction(kwargs, match):
    """Every one of these has a silent failure mode: an inverted dead band
    promotes and demotes on the same measurement, a zero step never moves, a
    window of zero has no statistic to threshold against."""
    with pytest.raises(CurriculumError, match=match):
        RedCurriculum(**kwargs)


def test_a_survival_rate_outside_the_unit_interval_is_refused():
    with pytest.raises(CurriculumError, match=r"survival_rate must be in \[0, 1\]"):
        CUR.observe(CUR.initial_state(), 1.4)


def test_the_shipped_defaults_are_the_smoothed_rule_and_the_old_dead_band():
    """The thresholds are deliberately UNCHANGED, so a checkpoint resumes
    against the dead band it was trained under. Only the evidence rule moved."""
    assert CUR.smoothing == "window" and CUR.window_size >= 2
    assert CUR.min_evaluations >= 2
    assert CUR.legacy_single_evaluation is False
    assert (CUR.promote_survival, CUR.demote_survival, CUR.step_size) == (0.75, 0.35, 0.05)
    assert RedCurriculum.legacy().legacy_single_evaluation is True


def test_the_difficulty_mapping_is_untouched():
    """Smoothing changes WHEN the level moves, never what a level means. A
    changed mapping would silently re-scale every checkpoint's `red_level`."""
    from naigos.env.config import EnvConfig

    base = EnvConfig()
    lo, hi = CUR.apply(base, 0.0), CUR.apply(base, 1.0)
    assert (lo.red_detect_scale, hi.red_detect_scale) == (CUR.detect_lo, CUR.detect_hi)
    assert (lo.red_lethal_scale, hi.red_lethal_scale) == (CUR.lethal_lo, CUR.lethal_hi)
    assert (lo.red_latency_scale, hi.red_latency_scale) == (CUR.latency_lo, CUR.latency_hi)
    assert (lo.red_speed_scale, hi.red_speed_scale) == (CUR.speed_lo, CUR.speed_hi)
    # and it still clamps rather than extrapolating
    assert CUR.apply(base, 5.0).red_detect_scale == CUR.apply(base, 1.0).red_detect_scale
