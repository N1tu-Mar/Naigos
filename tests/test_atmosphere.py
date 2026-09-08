"""Atmospheric derivations: density, refractivity, and the effective-Earth factor."""

from __future__ import annotations

import numpy as np
import pytest

from naigos.research.sources.atmosphere import (
    effective_earth_factor, moist_air_density, refractivity_n_units,
    saturation_vapour_pressure_pa,
)


def test_standard_refractivity_gradient_reproduces_four_thirds_earth():
    """The textbook -39 N-units/km must return the textbook k. This anchors the whole LOS model."""
    assert effective_earth_factor(-39.0) == pytest.approx(4.0 / 3.0, rel=0.02)


def test_a_weaker_gradient_gives_a_smaller_k():
    """Dry air refracts less, so the radar horizon comes closer. This is the AOI's measured case."""
    assert effective_earth_factor(-32.0) < effective_earth_factor(-39.0)


def test_zero_gradient_gives_a_true_sphere():
    assert effective_earth_factor(0.0) == pytest.approx(1.0)


def test_dry_air_density_matches_the_isa_sea_level_value():
    rho = float(moist_air_density(np.array(101325.0), np.array(288.15), np.array(0.0)))
    assert rho == pytest.approx(1.225, rel=0.005)


def test_humid_air_is_lighter_than_dry_air():
    """Counter-intuitive but load-bearing for density altitude: water vapour displaces N2 and O2."""
    dry = float(moist_air_density(np.array(101325.0), np.array(303.15), np.array(0.0)))
    wet = float(moist_air_density(np.array(101325.0), np.array(303.15), np.array(100.0)))
    assert wet < dry


def test_density_falls_with_altitude():
    p = np.array([101325.0, 79500.0, 54000.0, 26500.0])
    t = np.array([288.15, 278.0, 262.0, 236.0])
    rho = moist_air_density(p, t, np.zeros(4))
    assert np.all(np.diff(rho) < 0.0)


def test_saturation_vapour_pressure_matches_known_values():
    assert float(saturation_vapour_pressure_pa(np.array(273.15))) == pytest.approx(611.2, rel=0.01)
    assert float(saturation_vapour_pressure_pa(np.array(293.15))) == pytest.approx(2339.0, rel=0.02)


def test_refractivity_is_in_the_expected_surface_range():
    """Surface N is ~250-400 N-units almost everywhere on Earth (ITU-R P.453)."""
    n = float(refractivity_n_units(np.array(101325.0), np.array(288.15), np.array(60.0)))
    assert 250.0 < n < 400.0


def test_refractivity_falls_with_altitude():
    n_low = float(refractivity_n_units(np.array(101325.0), np.array(288.15), np.array(50.0)))
    n_high = float(refractivity_n_units(np.array(50000.0), np.array(255.0), np.array(50.0)))
    assert n_high < n_low
