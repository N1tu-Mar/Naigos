"""Terrain sampling and line-of-sight, on synthetic grids with known answers.

Geometry is tested before anything trusts it (spec section 9): a detection model built on a
wrong LOS test would produce a plausible-looking learning curve that means nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from naigos.data.terrain import (
    EARTH_RADIUS_M, TerrainGrid, earth_bulge_m, line_of_sight, los_margin, masked_fraction,
)


def flat_grid(height: float = 0.0, n: int = 200, px: float = 30.0) -> TerrainGrid:
    return TerrainGrid(
        z=np.full((n, n), height, dtype=np.float32),
        origin_x=0.0, origin_y=n * px, pixel_m=px, crs="EPSG:32611",
    )


def ridge_grid(ridge_height: float = 1000.0, n: int = 200, px: float = 30.0) -> TerrainGrid:
    """Flat ground at 0 m with a single north-south ridge down the middle column band."""
    z = np.zeros((n, n), dtype=np.float32)
    z[:, n // 2 - 1 : n // 2 + 2] = ridge_height
    return TerrainGrid(z=z, origin_x=0.0, origin_y=n * px, pixel_m=px, crs="EPSG:32611")


# --- sampling -------------------------------------------------------------------------


def test_elevation_is_exact_on_a_constant_grid():
    g = flat_grid(1234.0)
    assert float(g.elevation(np.array(1500.0), np.array(1500.0))) == pytest.approx(1234.0)


def test_elevation_interpolates_linearly_along_a_ramp():
    n, px = 10, 30.0
    z = np.tile(np.arange(n, dtype=np.float32) * 100.0, (n, 1))  # rises to the east
    g = TerrainGrid(z=z, origin_x=0.0, origin_y=n * px, pixel_m=px, crs="EPSG:32611")
    # Cell centres sit at x = (i+0.5)*px, so halfway between centre 2 and 3 must be 250.
    x = 2.5 * px + 0.5 * px
    assert float(g.elevation(np.array(x), np.array(150.0))) == pytest.approx(250.0, abs=1e-3)


def test_elevation_outside_the_grid_is_nan():
    g = flat_grid(100.0)
    assert math.isnan(float(g.elevation(np.array(-500.0), np.array(1000.0))))


def test_bounds_match_the_grid_extent():
    g = flat_grid(0.0, n=100, px=30.0)
    assert g.bounds == (0.0, 0.0, 3000.0, 3000.0)


# --- earth curvature ------------------------------------------------------------------


def test_earth_bulge_is_zero_at_the_endpoints_and_peaks_at_the_midpoint():
    d = 50_000.0
    assert earth_bulge_m(np.array(0.0), np.array(d)) == pytest.approx(0.0)
    mid = float(earth_bulge_m(np.array(d / 2), np.array(d / 2)))
    quarter = float(earth_bulge_m(np.array(d / 4), np.array(3 * d / 4)))
    assert mid > quarter > 0.0


def test_earth_bulge_matches_the_closed_form():
    d1 = d2 = 40_000.0
    expected = d1 * d2 / (2.0 * (4.0 / 3.0) * EARTH_RADIUS_M)
    assert float(earth_bulge_m(np.array(d1), np.array(d2))) == pytest.approx(expected)


# --- line of sight --------------------------------------------------------------------


def test_flat_terrain_gives_clear_line_of_sight():
    g = flat_grid(0.0)
    assert line_of_sight(g, (100.0, 3000.0, 500.0), (5000.0, 3000.0, 500.0))


def test_a_ridge_blocks_a_low_sightline():
    g = ridge_grid(1000.0)
    sensor = (300.0, 3000.0, 50.0)
    target = (5700.0, 3000.0, 50.0)
    assert not line_of_sight(g, sensor, target)
    assert los_margin(g, sensor, target)[0] < 0.0


def test_climbing_above_the_ridge_restores_line_of_sight():
    """The mechanic in one assertion: altitude buys visibility, low flight buys concealment."""
    g = ridge_grid(1000.0)
    sensor = (300.0, 3000.0, 50.0)
    blocked = los_margin(g, sensor, (5700.0, 3000.0, 50.0))[0]
    clear = los_margin(g, sensor, (5700.0, 3000.0, 4000.0))[0]
    assert blocked < 0.0 < clear


def test_los_margin_increases_monotonically_with_target_altitude():
    g = ridge_grid(1000.0)
    sensor = (300.0, 3000.0, 50.0)
    margins = [
        los_margin(g, sensor, (5700.0, 3000.0, alt))[0] for alt in (50, 500, 1000, 2000, 4000)
    ]
    assert all(b > a for a, b in zip(margins, margins[1:]))


def test_the_blocking_point_is_located_at_the_ridge():
    g = ridge_grid(1000.0)
    _, t = los_margin(g, (300.0, 3000.0, 50.0), (5700.0, 3000.0, 50.0))
    assert 0.4 < t < 0.6  # the ridge is at the midpoint of this segment


def test_curvature_blocks_a_long_flat_shot_that_geometry_alone_would_clear():
    """Over 100 km of flat ground, two 100 m masts cannot see each other. That is the bulge."""
    g = TerrainGrid(
        z=np.zeros((60, 4000), dtype=np.float32),
        origin_x=0.0, origin_y=1800.0, pixel_m=30.0, crs="EPSG:32611",
    )
    assert not line_of_sight(g, (100.0, 900.0, 100.0), (119_000.0, 900.0, 100.0))
    assert line_of_sight(g, (100.0, 900.0, 100.0), (119_000.0, 900.0, 2000.0))


def test_a_smaller_k_shortens_the_horizon():
    """Less refraction means less bending means the horizon comes closer."""
    g = TerrainGrid(
        z=np.zeros((60, 3000), dtype=np.float32),
        origin_x=0.0, origin_y=1800.0, pixel_m=30.0, crs="EPSG:32611",
    )
    p0, p1 = (100.0, 900.0, 100.0), (80_000.0, 900.0, 100.0)
    assert los_margin(g, p0, p1, k=4.0 / 3.0)[0] > los_margin(g, p0, p1, k=1.0)[0]


def test_missing_terrain_data_never_fabricates_cover():
    g = flat_grid(0.0)
    g.z[:, 100:] = np.nan
    assert line_of_sight(g, (100.0, 3000.0, 100.0), (5000.0, 3000.0, 100.0))


# --- viewshed -------------------------------------------------------------------------


def test_flat_terrain_masks_nothing():
    g = flat_grid(0.0, n=120)
    assert masked_fraction(g, (1800.0, 1800.0, 500.0), target_agl_m=300.0, n_samples=150) == 0.0


def test_masking_decreases_as_the_target_climbs():
    g = ridge_grid(1500.0, n=120)
    sensor = (200.0, 1800.0, 60.0)
    low = masked_fraction(g, sensor, target_agl_m=50.0, n_samples=200)
    high = masked_fraction(g, sensor, target_agl_m=4000.0, n_samples=200)
    assert low > high
