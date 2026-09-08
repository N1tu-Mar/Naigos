"""Real terrain resampled onto the uniform ENU heightmap the env flies over.

``naigos.env.terrain`` consumes a single ``(ny, nx)`` float32 heightmap in metres AMSL on a
uniform local ENU grid: origin at the (0, 0) corner, rows increasing with +y (north). This
module produces that array from the *cited* research cache, so the env can run on real terrain
without re-fetching and without bypassing the allowlist.

Two frame conversions happen here and nowhere else:
  * north-up raster storage (row 0 = north) -> ENU row order (row 0 = south);
  * absolute UTM easting/northing -> local metres from the window's south-west corner.

Keeping them in one place is deliberate, and the orientation is asserted in
``tests/test_dem_enu.py`` rather than eyeballed. A silent vertical flip mirrors the terrain:
every line-of-sight ray is then computed against a ridge that is not there, the policy still
learns *something*, and nothing visibly breaks until the demo.

Note this module lives beside ``naigos/data/dem.py``, which fetches and resamples a DEM by its
own path. See docs/DEVLOG.md -- the two overlap and should be reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import TerrainGrid
from .theatre import _dem_path


@dataclass(frozen=True)
class ENUGrid:
    """A real heightmap plus the georeference needed to map back to UTM."""

    heightmap: np.ndarray  # (ny, nx) float32, metres AMSL, row 0 = south
    nx: int
    ny: int
    cell_m: float
    origin_easting_m: float  # UTM easting of local x = 0
    origin_northing_m: float  # UTM northing of local y = 0
    crs: str

    @property
    def extent_m(self) -> tuple[float, float]:
        return ((self.nx - 1) * self.cell_m, (self.ny - 1) * self.cell_m)

    def to_utm(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local ENU metres -> absolute UTM, for placing surveyed airfields and threats."""
        return (self.origin_easting_m + np.asarray(x), self.origin_northing_m + np.asarray(y))

    def from_utm(self, easting: np.ndarray, northing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Absolute UTM -> local ENU metres."""
        return (
            np.asarray(easting) - self.origin_easting_m,
            np.asarray(northing) - self.origin_northing_m,
        )


class TheatreTooSmall(ValueError):
    """Raised when the requested grid extends past the cached DEM.

    Padding would fabricate flat ground outside the DEM, and flat ground is perfect radar
    visibility -- an artificial region the policy could learn to exploit.
    """


def natural_grid_shape(cell_m: float, grid: TerrainGrid | None = None) -> tuple[int, int]:
    """The largest (nx, ny) at ``cell_m`` that fits inside the cached DEM."""
    grid = grid if grid is not None else TerrainGrid.from_npz(_dem_path())
    min_x, min_y, max_x, max_y = grid.bounds
    return (int((max_x - min_x) // cell_m), int((max_y - min_y) // cell_m))


def real_terrain(
    nx: int = 64,
    ny: int = 88,
    cell_m: float = 1500.0,
    grid: TerrainGrid | None = None,
    center: bool = True,
) -> ENUGrid:
    """Resample the cached DEM onto a uniform ENU grid of ``(ny, nx)`` cells.

    ``center`` places the requested window in the middle of the theatre, which keeps the most
    relief in frame; set it False to anchor at the DEM's south-west corner.
    """
    grid = grid if grid is not None else TerrainGrid.from_npz(_dem_path())
    min_x, min_y, max_x, max_y = grid.bounds
    want_x, want_y = (nx - 1) * cell_m, (ny - 1) * cell_m
    have_x, have_y = max_x - min_x, max_y - min_y

    if want_x > have_x or want_y > have_y:
        fit_nx, fit_ny = natural_grid_shape(cell_m, grid)
        raise TheatreTooSmall(
            f"requested {nx}x{ny} at {cell_m} m = {want_x / 1000:.0f}x{want_y / 1000:.0f} km, "
            f"but the cached DEM spans only {have_x / 1000:.0f}x{have_y / 1000:.0f} km. "
            f"Use at most {fit_nx}x{fit_ny} at this cell size, or widen the AOI and re-run "
            "naigos-research."
        )

    ox = min_x + (have_x - want_x) / 2.0 if center else min_x
    oy = min_y + (have_y - want_y) / 2.0 if center else min_y

    xs = ox + np.arange(nx, dtype=np.float64) * cell_m
    ys = oy + np.arange(ny, dtype=np.float64) * cell_m  # ascending north = ENU row order
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    h = grid.elevation(gx, gy)

    if not np.isfinite(h).all():
        # Only reachable at the DEM's ragged edge; fill from the nearest valid row/column
        # rather than with a constant, so no artificial plateau is introduced.
        h = _fill_edges(h)

    return ENUGrid(
        heightmap=h.astype(np.float32), nx=nx, ny=ny, cell_m=cell_m,
        origin_easting_m=float(ox), origin_northing_m=float(oy), crs=grid.crs,
    )


def _fill_edges(h: np.ndarray) -> np.ndarray:
    """Nearest-valid fill along rows then columns."""
    out = h.copy()
    for _ in range(4):
        if np.isfinite(out).all():
            break
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            gap = ~np.isfinite(out)
            if not gap.any():
                break
            out[gap] = np.roll(out, shift, axis=axis)[gap]
    return np.where(np.isfinite(out), out, np.nanmin(h[np.isfinite(h)]))
