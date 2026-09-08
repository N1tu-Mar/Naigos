"""Resampling the cached DEM onto the env's ENU grid.

The frame conversion here is the highest-risk quiet bug in the data layer: a vertical flip
mirrors the terrain, every LOS ray is then computed against a ridge that is not there, and the
policy still learns *something*, so nothing visibly breaks. These tests make orientation
falsifiable rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest

from naigos.data.enu import ENUGrid, TheatreTooSmall, natural_grid_shape, real_terrain
from naigos.data.terrain import TerrainGrid
from naigos.research.spec import COMPONENTS_DIR

needs_cache = pytest.mark.skipif(
    not (COMPONENTS_DIR / "data.terrain_dem.json").exists(),
    reason="research cache not built; run `naigos-research`",
)


def north_ramp_grid(n: int = 100, px: float = 100.0) -> TerrainGrid:
    """Terrain rising towards the north. Stored north-up, so row 0 is the HIGHEST row."""
    z = np.repeat(np.linspace(1000.0, 0.0, n, dtype=np.float32)[:, None], n, axis=1)
    return TerrainGrid(z=z, origin_x=0.0, origin_y=n * px, pixel_m=px, crs="EPSG:32611")


# --- orientation ----------------------------------------------------------------------


def test_enu_row_zero_is_the_south_edge():
    """The decisive orientation test: on terrain that rises northward, ENU row 0 is lowest."""
    g = real_terrain(nx=20, ny=20, cell_m=300.0, grid=north_ramp_grid(), center=False)
    assert g.heightmap[0].mean() < g.heightmap[-1].mean()
    assert np.all(np.diff(g.heightmap.mean(axis=1)) > 0.0)


def test_enu_sampling_agrees_with_the_source_grid_at_the_same_utm_point():
    """Both paths must return the same elevation for the same absolute position."""
    src = north_ramp_grid()
    g = real_terrain(nx=15, ny=15, cell_m=400.0, grid=src, center=True)
    for iy in (0, 7, 14):
        for ix in (0, 7, 14):
            east, north = g.to_utm(ix * g.cell_m, iy * g.cell_m)
            direct = float(src.elevation(np.array(east), np.array(north)))
            assert float(g.heightmap[iy, ix]) == pytest.approx(direct, abs=1e-3)


def test_utm_round_trip_is_exact():
    g = real_terrain(nx=10, ny=10, cell_m=500.0, grid=north_ramp_grid(), center=True)
    x, y = np.array([0.0, 1500.0, 4500.0]), np.array([0.0, 2000.0, 4500.0])
    bx, by = g.from_utm(*g.to_utm(x, y))
    assert np.allclose(bx, x) and np.allclose(by, y)


# --- shape and bounds ------------------------------------------------------------------


def test_requested_shape_and_dtype_are_honoured():
    g = real_terrain(nx=13, ny=17, cell_m=250.0, grid=north_ramp_grid())
    assert g.heightmap.shape == (17, 13)
    assert g.heightmap.dtype == np.float32
    assert g.extent_m == (12 * 250.0, 16 * 250.0)


def test_a_grid_larger_than_the_dem_is_refused_not_padded():
    """Padding would invent flat ground, and flat ground is perfect radar visibility."""
    with pytest.raises(TheatreTooSmall, match="widen the AOI"):
        real_terrain(nx=500, ny=500, cell_m=1000.0, grid=north_ramp_grid())


def test_natural_grid_shape_fits_inside_the_dem():
    src = north_ramp_grid()
    nx, ny = natural_grid_shape(500.0, src)
    g = real_terrain(nx=nx, ny=ny, cell_m=500.0, grid=src)  # must not raise
    min_x, min_y, max_x, max_y = src.bounds
    assert g.extent_m[0] <= max_x - min_x and g.extent_m[1] <= max_y - min_y


def test_the_heightmap_is_always_finite():
    g = real_terrain(nx=30, ny=30, cell_m=300.0, grid=north_ramp_grid())
    assert np.isfinite(g.heightmap).all()


def test_edge_gaps_are_filled_from_neighbours_not_with_a_constant():
    src = north_ramp_grid()
    src.z[:, -3:] = np.nan  # ragged eastern edge
    g = real_terrain(nx=25, ny=25, cell_m=380.0, grid=src, center=False)
    assert np.isfinite(g.heightmap).all()
    assert np.ptp(g.heightmap) > 100.0, "fill flattened the terrain"


# --- against the real cached DEM -------------------------------------------------------


@needs_cache
def test_the_real_theatre_resamples_with_its_relief_intact():
    nx, ny = natural_grid_shape(1500.0)
    g = real_terrain(nx=nx, ny=ny, cell_m=1500.0)
    assert isinstance(g, ENUGrid)
    assert np.isfinite(g.heightmap).all()
    assert np.ptp(g.heightmap) > 3000.0, "resampling lost the relief the AOI was chosen for"


@needs_cache
def test_the_real_theatre_keeps_its_orientation_against_the_source():
    """Same check as the synthetic case, but on the actual DEM the env will fly over."""
    src = TerrainGrid.from_npz(__import__(
        "naigos.data.theatre", fromlist=["_dem_path"]
    )._dem_path())
    g = real_terrain(nx=40, ny=40, cell_m=1500.0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        ix, iy = int(rng.integers(0, 40)), int(rng.integers(0, 40))
        east, north = g.to_utm(ix * g.cell_m, iy * g.cell_m)
        direct = float(src.elevation(np.array(east), np.array(north)))
        assert float(g.heightmap[iy, ix]) == pytest.approx(direct, abs=0.5)
