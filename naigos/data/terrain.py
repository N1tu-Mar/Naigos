"""Terrain grid + line-of-sight, the reference implementation the JAX env mirrors.

Pure numpy on purpose: the CMDP verifier and the geometry tests must run without JAX (spec
section 9), and this is the ground truth the vectorized env is checked against.

Geometry conventions:
  * Coordinates are metres in the AOI's UTM zone. ``x`` is easting, ``y`` is northing,
    ``z`` is elevation above the WGS84-derived DEM datum (treated as MSL).
  * The grid is stored north-up: ``z[0, 0]`` is the north-west corner.
  * Line of sight applies the standard 4/3-Earth effective-radius correction, which folds
    normal atmospheric refraction into a straight-ray geometry (Blake, Radar Range-Performance
    Analysis, 1986; ITU-R P.834). Over an 80 km leg this is a ~130 m bulge -- the difference
    between "masked" and "tracked" for a low-flying aircraft, so it is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
K_EFFECTIVE_EARTH = 4.0 / 3.0


@dataclass
class TerrainGrid:
    """A north-up metric elevation grid."""

    z: np.ndarray  # (rows, cols) float32 elevation in metres, NaN where no data
    origin_x: float  # easting of the west edge of column 0
    origin_y: float  # northing of the north edge of row 0
    pixel_m: float
    crs: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in metres."""
        rows, cols = self.z.shape
        return (
            self.origin_x,
            self.origin_y - rows * self.pixel_m,
            self.origin_x + cols * self.pixel_m,
            self.origin_y,
        )

    # --- sampling ---------------------------------------------------------------------

    def elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear terrain elevation at metric coordinates. Out-of-bounds -> NaN."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        col = (x - self.origin_x) / self.pixel_m - 0.5
        row = (self.origin_y - y) / self.pixel_m - 0.5
        rows, cols = self.z.shape

        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc = col - c0
        fr = row - r0
        inside = (c0 >= 0) & (r0 >= 0) & (c0 < cols - 1) & (r0 < rows - 1)

        c0c = np.clip(c0, 0, cols - 2)
        r0c = np.clip(r0, 0, rows - 2)
        z00 = self.z[r0c, c0c]
        z01 = self.z[r0c, c0c + 1]
        z10 = self.z[r0c + 1, c0c]
        z11 = self.z[r0c + 1, c0c + 1]
        top = z00 * (1 - fc) + z01 * fc
        bot = z10 * (1 - fc) + z11 * fc
        out = top * (1 - fr) + bot * fr
        return np.where(inside, out, np.nan)

    # --- serialization ----------------------------------------------------------------

    @classmethod
    def from_geotiff(cls, path: str | Path) -> "TerrainGrid":
        import rasterio

        with rasterio.open(path) as ds:
            z = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
            t = ds.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                raise ValueError("rotated rasters are not supported; reproject to a north-up grid")
            return cls(z=z, origin_x=t.c, origin_y=t.f, pixel_m=abs(t.a), crs=str(ds.crs))

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z=self.z,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            pixel_m=self.pixel_m,
            crs=self.crs,
        )
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> "TerrainGrid":
        d = np.load(path, allow_pickle=False)
        return cls(
            z=d["z"],
            origin_x=float(d["origin_x"]),
            origin_y=float(d["origin_y"]),
            pixel_m=float(d["pixel_m"]),
            crs=str(d["crs"]),
        )


# --- line of sight --------------------------------------------------------------------


def earth_bulge_m(d1: np.ndarray, d2: np.ndarray, k: float = K_EFFECTIVE_EARTH) -> np.ndarray:
    """Apparent terrain rise at a point ``d1`` from one end and ``d2`` from the other.

    Standard effective-Earth formulation: h = d1*d2 / (2*k*Re). Adding this to the terrain
    height along the ray is equivalent to bending the ray, and keeps the LOS test a straight
    line comparison.
    """
    return d1 * d2 / (2.0 * k * EARTH_RADIUS_M)


def _sample_count(grid: TerrainGrid, dist_m: float) -> int:
    """One sample per half pixel along the ray, clamped so short rays stay cheap."""
    return int(max(2, min(4096, np.ceil(2.0 * dist_m / grid.pixel_m) + 1)))


def line_of_sight(
    grid: TerrainGrid,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    *,
    k: float = K_EFFECTIVE_EARTH,
    clearance_m: float = 0.0,
) -> bool:
    """True if the straight segment ``p0``->``p1`` clears the terrain.

    ``clearance_m`` requires the ray to pass that far above the terrain, which models the
    first-Fresnel-zone grazing loss crudely; set it to 0 for pure geometric LOS.
    """
    return bool(los_margin(grid, p0, p1, k=k)[0] > clearance_m)


def los_margin(
    grid: TerrainGrid,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    *,
    k: float = K_EFFECTIVE_EARTH,
) -> tuple[float, float]:
    """Minimum ray-above-terrain clearance along the segment, and where it occurs.

    Returns ``(min_clearance_m, fraction_along_ray)``. Negative clearance means the terrain
    blocks the sightline -- the aircraft is masked. This single scalar is what the detection
    model consumes: it degrades smoothly through grazing geometry instead of flipping a
    boolean, which matters because a hard 0/1 mask gives the policy no gradient to climb.
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    total = float(np.hypot(x1 - x0, y1 - y0))
    if total <= 0.0:
        return (float(z0 - np.nan_to_num(grid.elevation(np.array(x0), np.array(y0)), nan=-1e4)), 0.0)

    n = _sample_count(grid, total)
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    ray_z = z0 + (z1 - z0) * t

    ground = grid.elevation(xs, ys)
    # Outside the DEM there is no terrain evidence; treat it as non-blocking rather than as
    # a wall, so an AOI edge never fabricates cover.
    ground = np.where(np.isfinite(ground), ground, -np.inf)

    d1 = t * total
    d2 = (1.0 - t) * total
    apparent = ground + earth_bulge_m(d1, d2, k=k)

    clearance = ray_z - apparent
    interior = slice(1, -1) if n > 2 else slice(None)
    idx = int(np.argmin(clearance[interior])) + (1 if n > 2 else 0)
    return (float(clearance[idx]), float(t[idx]))


def masked_fraction(
    grid: TerrainGrid,
    sensor: tuple[float, float, float],
    *,
    target_agl_m: float = 100.0,
    n_samples: int = 4000,
    seed: int = 0,
) -> float:
    """Fraction of the AOI at ``target_agl_m`` above ground that is hidden from ``sensor``.

    This is the honest masking statistic: it runs the real LOS test from a real sensor
    position over random points in the AOI, and answers "how much of this terrain can actually
    hide an aircraft flying this low". A value near 0 means the AOI is too flat for masking to
    be a learnable mechanic; a value near 1 means the sensor is nearly useless there.
    """
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = grid.bounds
    xs = rng.uniform(min_x, max_x, n_samples)
    ys = rng.uniform(min_y, max_y, n_samples)
    ground = grid.elevation(xs, ys)
    ok = np.isfinite(ground)
    xs, ys, ground = xs[ok], ys[ok], ground[ok]

    blocked = 0
    for x, y, g in zip(xs, ys, ground):
        if los_margin(grid, sensor, (float(x), float(y), float(g) + target_agl_m))[0] <= 0.0:
            blocked += 1
    return blocked / max(len(xs), 1)
