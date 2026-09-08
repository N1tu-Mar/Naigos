"""The detection model's physics, checked against closed-form expectations.

Spec section 5's reward-hacking watchlist says to calibrate the detection model *before* tuning
reward weights. These are that calibration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from naigos.research.sources.radar import (
    RADAR_CLASSES, calibrate_power, detection_probability, radar_horizon_km,
    reference_range_m, required_snr_db_albersheim, snr_at_range,
)

CLASS = "medium_sam_acquisition"


def calibrated(name: str = CLASS, rcs: float = 1.0):
    return calibrate_power(RADAR_CLASSES[name], rcs)


# --- range equation -------------------------------------------------------------------


def test_snr_follows_the_inverse_fourth_power_law():
    p = calibrated()
    near = float(snr_at_range(p, np.array(10_000.0), 1.0))
    far = float(snr_at_range(p, np.array(20_000.0), 1.0))
    assert near / far == pytest.approx(16.0, rel=1e-6)


def test_snr_is_linear_in_radar_cross_section():
    p = calibrated()
    a = float(snr_at_range(p, np.array(50_000.0), 1.0))
    b = float(snr_at_range(p, np.array(50_000.0), 4.0))
    assert b / a == pytest.approx(4.0, rel=1e-9)


def test_detection_range_scales_as_the_fourth_root_of_rcs():
    """The single most important scaling law for an evasion problem: signature is expensive."""
    p = calibrated()
    r1 = reference_range_m(p, 1.0)
    r16 = reference_range_m(p, 16.0)
    assert r16 / r1 == pytest.approx(2.0, rel=1e-6)


def test_calibrate_power_puts_pd_half_at_the_declared_design_range():
    for name, raw in RADAR_CLASSES.items():
        p = calibrate_power(raw, 1.0, target_pd=0.5)
        pd = float(detection_probability(
            snr_at_range(p, np.array(raw.design_range_km * 1000.0), 1.0), p.prob_false_alarm
        ))
        assert pd == pytest.approx(0.5, abs=1e-6), name


# --- detection statistics -------------------------------------------------------------


def test_pd_decreases_monotonically_with_range():
    p = calibrated()
    r = np.linspace(5_000.0, 250_000.0, 60)
    pd = detection_probability(snr_at_range(p, r, 1.0), p.prob_false_alarm)
    assert np.all(np.diff(pd) < 0.0)


def test_pd_stays_a_probability():
    p = calibrated()
    r = np.logspace(2, 6.5, 200)
    pd = detection_probability(snr_at_range(p, r, 1.0), p.prob_false_alarm)
    assert np.all((pd >= 0.0) & (pd <= 1.0))


def test_pd_is_a_smooth_ramp_not_a_step():
    """A step function gives the policy no exposure gradient to descend along."""
    p = calibrated()
    r = np.linspace(40_000.0, 200_000.0, 200)
    pd = detection_probability(snr_at_range(p, r, 1.0), p.prob_false_alarm)
    # The 0.9 -> 0.1 transition must span a wide band of range, not a knife edge.
    band = r[(pd < 0.9) & (pd > 0.1)]
    assert (band.max() - band.min()) > 60_000.0


def test_lower_false_alarm_rate_costs_detection():
    p_loose = calibrate_power(RADAR_CLASSES[CLASS], 1.0)
    snr = snr_at_range(p_loose, np.array(80_000.0), 1.0)
    assert float(detection_probability(snr, 1e-4)) > float(detection_probability(snr, 1e-8))


def test_swerling1_detects_less_sharply_than_a_nonfluctuating_target():
    """The reason Swerling 1 is the default.

    The two models are compared by how *wide* their roll-off is relative to their own detection
    range, not on an absolute range axis: a non-fluctuating target needs far less SNR for the
    same Pd, so it detects much further out and an absolute comparison would measure that
    offset instead of the sharpness. A fluctuating target blurs the detection edge, which is
    what gives the policy an exposure gradient to descend along rather than a cliff to fall off.
    """
    p = calibrated()
    r = np.linspace(20_000.0, 400_000.0, 4000)
    snr = snr_at_range(p, r, 1.0)

    def rolloff_ratio(pd: np.ndarray) -> float:
        # pd is monotonically decreasing in r, so interpolate on the reversed arrays.
        r_at_10 = float(np.interp(0.1, pd[::-1], r[::-1]))
        r_at_90 = float(np.interp(0.9, pd[::-1], r[::-1]))
        return r_at_10 / r_at_90

    sw = rolloff_ratio(detection_probability(snr, p.prob_false_alarm, "swerling1"))
    nf = rolloff_ratio(detection_probability(
        snr, p.prob_false_alarm, "nonfluctuating", n_pulses=p.pulses_integrated
    ))
    assert sw > nf
    assert sw > 1.5  # Swerling 1 rolls off over a wide band of range...
    assert nf < 1.3  # ...while a non-fluctuating target switches off comparatively abruptly.


def test_nonfluctuating_pd_saturates_at_the_validity_band_edge():
    """Documented consequence: Albersheim is inverted only over 0.02 <= Pd <= 0.98."""
    p = calibrated()
    pd = detection_probability(
        snr_at_range(p, np.array(1_000.0), 1.0), p.prob_false_alarm,
        "nonfluctuating", n_pulses=p.pulses_integrated,
    )
    assert 0.97 <= float(pd) <= 0.98


def test_albersheim_refuses_to_extrapolate_outside_its_validity_band():
    with pytest.raises(ValueError, match="not valid"):
        required_snr_db_albersheim(0.001, 1e-6)


def test_albersheim_required_snr_moves_the_right_way():
    assert required_snr_db_albersheim(0.9, 1e-6) > required_snr_db_albersheim(0.5, 1e-6)
    assert required_snr_db_albersheim(0.5, 1e-8) > required_snr_db_albersheim(0.5, 1e-4)
    assert required_snr_db_albersheim(0.5, 1e-6, 10) < required_snr_db_albersheim(0.5, 1e-6, 1)


def test_unknown_detection_model_is_rejected():
    with pytest.raises(ValueError):
        detection_probability(np.array(1.0), 1e-6, "wishful")


# --- horizon --------------------------------------------------------------------------


def test_radar_horizon_grows_with_altitude_and_with_refraction():
    assert radar_horizon_km(50.0, 5000.0) > radar_horizon_km(50.0, 300.0)
    assert radar_horizon_km(50.0, 3000.0, k=4 / 3) > radar_horizon_km(50.0, 3000.0, k=1.0)


def test_radar_horizon_matches_the_classical_rule_of_thumb():
    """d_km ~ 4.12*sqrt(h_m) for k=4/3 with the sensor at ground level."""
    assert radar_horizon_km(0.0, 1000.0) == pytest.approx(4.12 * math.sqrt(1000.0), rel=0.02)


# --- class structure ------------------------------------------------------------------


def test_every_class_can_see_further_than_it_can_shoot():
    """'Detected but not yet engaged' must be a state the policy can act from."""
    from naigos.research.sources.radar import ENGAGEMENT

    for name in RADAR_CLASSES:
        assert ENGAGEMENT[name]["lethal_fraction_of_detection_range"] < 1.0, name


def test_design_ranges_span_the_theatre_usefully():
    ranges = sorted(p.design_range_km for p in RADAR_CLASSES.values())
    assert ranges[0] < 10.0 and ranges[-1] > 200.0
