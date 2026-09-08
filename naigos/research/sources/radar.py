"""Detection physics, derived from the open radar range equation. Nothing is fetched.

This module is the guardrail in code (see ``allowlist.GUARDRAIL``). Threats are described by
ROLE -- what a sensor of that class can physically do -- and every number is either a physical
constant, a quantity derived from the range equation, or a notional design parameter chosen to
produce a usable exposure-vs-survival gradient. There is no system-specific capability data
here, and the RL problem does not need any: what the policy must learn is the *shape* of the
tradeoff between exposure and survival.

References (all open literature):
  Skolnik, Radar Handbook 3rd ed. (2008), ch. 1-2  -- range equation, detection thresholds
  Barton, Radar Equations for Modern Radar (2013)   -- loss budgets, integration
  Blake, Radar Range-Performance Analysis (1986)    -- pattern-propagation factor, refraction
  Swerling, IRE Trans. IT-6 (1960)                  -- fluctuating-target detection statistics
  ITU-R P.525/P.526/P.676/P.834                     -- free space, diffraction, gases, refraction
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

C_LIGHT = 299_792_458.0
K_BOLTZMANN = 1.380649e-23
T0 = 290.0  # standard noise reference temperature, K


# --- the range equation ----------------------------------------------------------------


@dataclass(frozen=True)
class RadarParams:
    """Notional radar design parameters. Role-generic by construction."""

    name: str
    role: str
    frequency_hz: float
    design_range_km: float
    antenna_gain_dbi: float
    pulse_width_s: float
    pulses_integrated: int
    noise_figure_db: float
    system_loss_db: float
    prob_false_alarm: float
    scan_period_s: float

    #: Solved from ``design_range_km`` by :func:`calibrate_power`; never hand-set.
    peak_power_w: float = 0.0

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.frequency_hz

    @property
    def bandwidth_hz(self) -> float:
        """Matched-filter bandwidth, B ~ 1/tau."""
        return 1.0 / self.pulse_width_s


def snr_at_range(p: RadarParams, range_m: np.ndarray, rcs_m2: float) -> np.ndarray:
    """Single-target SNR after pulse integration, from the monostatic range equation.

        SNR = Pt * G^2 * lambda^2 * sigma * n / ((4*pi)^3 * R^4 * k * T0 * B * F * L)

    Free-space; propagation factor and atmospheric absorption are applied separately so each
    effect stays individually inspectable.
    """
    g = 10.0 ** (p.antenna_gain_dbi / 10.0)
    f = 10.0 ** (p.noise_figure_db / 10.0)
    loss = 10.0 ** (p.system_loss_db / 10.0)
    num = p.peak_power_w * g * g * p.wavelength_m**2 * rcs_m2 * p.pulses_integrated
    den = (4.0 * math.pi) ** 3 * np.power(np.asarray(range_m, dtype=float), 4.0)
    den = den * K_BOLTZMANN * T0 * p.bandwidth_hz * f * loss
    return num / den


def gaseous_attenuation_db_per_km(frequency_hz: float, density_ratio: float = 1.0) -> float:
    """Approximate one-way specific attenuation by atmospheric gases (ITU-R P.676 regime).

    A piecewise fit adequate below 15 GHz, where oxygen dominates and water-vapour lines are
    far off. Scaled by air density ratio because attenuation tracks absorber column density.
    Over a 100 km theatre at X band this is a couple of dB two-way -- small, but it is the
    difference between a 1% and a 2% detection probability at the envelope edge.
    """
    f_ghz = frequency_hz / 1e9
    gamma = 0.0067 + 0.0011 * f_ghz**1.6  # dB/km, sea level, temperate
    return gamma * density_ratio


def detection_probability(
    snr_linear: np.ndarray, p_fa: float, model: str = "swerling1", n_pulses: int = 1
) -> np.ndarray:
    """Probability of detection for one look.

    ``swerling1`` (slowly fluctuating target, Rayleigh RCS) is the default: an aircraft's radar
    cross-section swings by an order of magnitude with tiny aspect changes, so a non-fluctuating
    model badly overstates how sharply detection turns on with range.

        Swerling 1, integrated SNR S:  Pd = Pfa^(1 / (1 + S))
        Non-fluctuating, single look:  Pd = Q_M(sqrt(2S), sqrt(-2 ln Pfa))

    The Swerling-1 form is exact for a single pulse and is used here with the post-integration
    SNR, which is the standard coherent-integration approximation.
    """
    s = np.clip(np.asarray(snr_linear, dtype=float), 1e-12, None)
    if model == "swerling1":
        return np.power(p_fa, 1.0 / (1.0 + s))
    if model == "nonfluctuating":
        return _pd_nonfluctuating(s, p_fa, n_pulses)
    raise ValueError(f"unknown detection model {model!r}")


def required_snr_db_albersheim(p_d: float, p_fa: float, n_pulses: int = 1) -> float:
    """SNR needed for ``p_d`` against a non-fluctuating target (Albersheim's equation).

    Albersheim, "A closed-form approximation to Robertson's detection characteristics" (Proc.
    IEEE, 1981). Accurate to ~0.2 dB for 1e-7 <= Pfa <= 1e-3 and 0.1 <= Pd <= 0.9.
    """
    a = math.log(0.62 / p_fa)
    b = math.log(p_d / (1.0 - p_d))
    return -5.0 * math.log10(n_pulses) + (
        6.2 + 4.54 / math.sqrt(n_pulses + 0.44)
    ) * math.log10(a + 0.12 * a * b + 1.7 * b)


def _pd_nonfluctuating(snr_linear: np.ndarray, p_fa: float, n_pulses: int) -> np.ndarray:
    """Invert Albersheim's equation numerically to get Pd from SNR."""
    grid_pd = np.linspace(0.005, 0.995, 400)
    grid_snr = np.array([required_snr_db_albersheim(pd, p_fa, n_pulses) for pd in grid_pd])
    snr_db = 10.0 * np.log10(np.asarray(snr_linear, dtype=float))
    return np.clip(np.interp(snr_db, grid_snr, grid_pd), 0.0, 1.0)


def reference_range_m(p: RadarParams, rcs_m2: float, target_pd: float = 0.5) -> float:
    """Free-space range at which a target of ``rcs_m2`` is detected with probability ``target_pd``.

    Inverting Swerling 1 gives the SNR needed, and the range equation's R^-4 gives the range.
    """
    required_snr = math.log(p.prob_false_alarm) / math.log(target_pd) - 1.0
    snr_1km = float(snr_at_range(p, np.array(1000.0), rcs_m2))
    return 1000.0 * (snr_1km / required_snr) ** 0.25


# --- generic threat classes ------------------------------------------------------------

# Role-labelled, notional sensor designs. Parameters were chosen so the resulting detection
# ranges span the theatre usefully -- a long-range cue that sees most of the AOI, down to a
# short-range sensor an aircraft can fly around -- not to match any fielded system.
RADAR_CLASSES: dict[str, RadarParams] = {
    "long_range_surveillance": RadarParams(
        name="long_range_surveillance", role="Wide-area early warning; cues other threats, no lethality.",
        frequency_hz=1.3e9, design_range_km=250.0, antenna_gain_dbi=36.0, pulse_width_s=100e-6,
        pulses_integrated=20, noise_figure_db=3.0, system_loss_db=8.0,
        prob_false_alarm=1e-6, scan_period_s=10.0,
    ),
    "medium_sam_acquisition": RadarParams(
        name="medium_sam_acquisition", role="Area air defence acquisition and engagement radar.",
        frequency_hz=3.0e9, design_range_km=110.0, antenna_gain_dbi=38.0, pulse_width_s=20e-6,
        pulses_integrated=16, noise_figure_db=3.5, system_loss_db=8.0,
        prob_false_alarm=1e-6, scan_period_s=5.0,
    ),
    "short_range_point_defense": RadarParams(
        name="short_range_point_defense", role="Point defence of a fixed asset; fast reaction, small envelope.",
        frequency_hz=9.5e9, design_range_km=30.0, antenna_gain_dbi=34.0, pulse_width_s=1.0e-6,
        pulses_integrated=12, noise_figure_db=4.0, system_loss_db=8.0,
        prob_false_alarm=1e-6, scan_period_s=2.0,
    ),
    "mobile_short_range": RadarParams(
        name="mobile_short_range", role="Vehicle-mounted mobile air defence; relocates between episodes.",
        frequency_hz=9.5e9, design_range_km=15.0, antenna_gain_dbi=30.0, pulse_width_s=0.8e-6,
        pulses_integrated=8, noise_figure_db=4.5, system_loss_db=8.0,
        prob_false_alarm=1e-6, scan_period_s=2.0,
    ),
    "interceptor_seeker": RadarParams(
        name="interceptor_seeker", role="Seeker carried by a mobile interceptor; narrow field, short range.",
        frequency_hz=3.5e10, design_range_km=8.0, antenna_gain_dbi=30.0, pulse_width_s=0.5e-6,
        pulses_integrated=8, noise_figure_db=6.0, system_loss_db=6.0,
        prob_false_alarm=1e-5, scan_period_s=1.0,
    ),
}


def calibrate_power(p: RadarParams, rcs_m2: float, target_pd: float = 0.5) -> RadarParams:
    """Solve the transmit power that puts Pd=``target_pd`` at the class's declared design range.

    The design range is the notional, role-based parameter; the power is then a *consequence* of
    the range equation rather than another number to invent. This keeps the physics honest in
    both directions -- RCS still scales range as sigma^(1/4), attenuation still bites at high
    frequency -- while the only free choice is a deliberately round, generic engagement distance.
    """
    required_snr = math.log(p.prob_false_alarm) / math.log(target_pd) - 1.0
    probe = RadarParams(**{**asdict(p), "peak_power_w": 1.0})
    snr_1w = float(snr_at_range(probe, np.array(p.design_range_km * 1000.0), rcs_m2))
    return RadarParams(**{**asdict(p), "peak_power_w": required_snr / snr_1w})


def radar_horizon_km(sensor_alt_m: float, target_alt_m: float, k: float = 4.0 / 3.0) -> float:
    """Geometric horizon over a smooth effective Earth, the hard cap on any detection range.

    d = sqrt(2*k*Re*h_s) + sqrt(2*k*Re*h_t). Real terrain masks far more than this, but no
    sensor sees past it, so a design range beyond the horizon buys nothing.
    """
    re = 6_371_000.0
    return (math.sqrt(2 * k * re * max(sensor_alt_m, 0.0)) + math.sqrt(2 * k * re * max(target_alt_m, 0.0))) / 1000.0


# Reference target signature. A 1 m^2 head-on cross-section is the conventional yardstick for
# a small combat aircraft in the open literature; the env varies it by aspect.
REFERENCE_RCS_M2 = 1.0

# Engagement behaviour per class. Lethal radius is set as a fraction of the class's own
# detection range, so a threat can always see further than it can reach -- which is what makes
# "detected but not yet engaged" a state the policy can act from.
ENGAGEMENT: dict[str, dict[str, Any]] = {
    "long_range_surveillance": {
        "lethal_fraction_of_detection_range": 0.0, "reaction_latency_s": 0.0,
        "min_engagement_alt_agl_m": None, "max_engagement_alt_m": None, "mobile": False,
    },
    "medium_sam_acquisition": {
        "lethal_fraction_of_detection_range": 0.45, "reaction_latency_s": 12.0,
        "min_engagement_alt_agl_m": 60.0, "max_engagement_alt_m": 20000.0, "mobile": False,
    },
    "short_range_point_defense": {
        "lethal_fraction_of_detection_range": 0.55, "reaction_latency_s": 5.0,
        "min_engagement_alt_agl_m": 20.0, "max_engagement_alt_m": 6000.0, "mobile": False,
    },
    "mobile_short_range": {
        "lethal_fraction_of_detection_range": 0.55, "reaction_latency_s": 6.0,
        "min_engagement_alt_agl_m": 20.0, "max_engagement_alt_m": 4500.0, "mobile": True,
    },
    "interceptor_seeker": {
        "lethal_fraction_of_detection_range": 0.25, "reaction_latency_s": 2.0,
        "min_engagement_alt_agl_m": 0.0, "max_engagement_alt_m": 12000.0, "mobile": True,
    },
}


def derive_threat_classes(
    density_ratio: float = 1.0,
    target_pd: float = 0.5,
    rcs_m2: float = REFERENCE_RCS_M2,
    k_earth: float = 4.0 / 3.0,
) -> dict[str, Any]:
    """Derive every threat class's envelope from the range equation. This is the env's input."""
    out: dict[str, Any] = {}
    for key, raw in RADAR_CLASSES.items():
        p = calibrate_power(raw, rcs_m2, target_pd=target_pd)
        r50 = reference_range_m(p, rcs_m2, target_pd=target_pd)
        eng = ENGAGEMENT[key]
        out[key] = {
            "role": p.role,
            "radar": asdict(p) | {
                "wavelength_m": round(p.wavelength_m, 4),
                "bandwidth_hz": round(p.bandwidth_hz, 1),
            },
            "derived": {
                "detection_range_km_pd50": round(r50 / 1000.0, 2),
                "detection_range_km_pd90": round(
                    reference_range_m(p, rcs_m2, target_pd=0.9) / 1000.0, 2
                ),
                "reference_rcs_m2": rcs_m2,
                "rcs_scaling": "range scales as RCS^(1/4); halving RCS costs 16% of range",
                "gaseous_attenuation_db_per_km_one_way": round(
                    gaseous_attenuation_db_per_km(p.frequency_hz, density_ratio), 4
                ),
                "lethal_range_km": round(
                    r50 / 1000.0 * eng["lethal_fraction_of_detection_range"], 2
                ),
                "solved_peak_power_w": float(f"{p.peak_power_w:.4g}"),
                "horizon_km_sensor50m_target300m": round(
                    radar_horizon_km(50.0, 300.0, k=k_earth), 1
                ),
                "horizon_km_sensor50m_target5000m": round(
                    radar_horizon_km(50.0, 5000.0, k=k_earth), 1
                ),
            },
            "engagement": eng,
        }
    return out
