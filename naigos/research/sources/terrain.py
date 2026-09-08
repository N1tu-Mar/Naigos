"""Terrain elevation (USGS 3DEP) -- the dataset radar line-of-sight masking is built on.

3DEP returns its data in EPSG:5070 (Albers equal-area CONUS), which is fine for area but not
for the ranges and bearings the detection model computes. We reproject once, here, to the local
UTM zone: within a single AOI, UTM is conformal and metre-accurate, so slant range in the env is
a plain Euclidean distance in grid coordinates and no per-step geodesy is needed.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np

from .. import cache
from ..aoi import AOI

RESOLUTION_M = 30  # 1 arc-second class; matches 3DEP's seamless national coverage.


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the UTM zone containing (lat, lon)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _build_dem_geotiff(aoi: AOI, resolution: int) -> bytes:
    import py3dep
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    dem = py3dep.get_dem(aoi.bbox, resolution)
    lat, lon = aoi.center
    dem = dem.rio.reproject(f"EPSG:{utm_epsg(lat, lon)}", resolution=resolution)
    dem = dem.rio.write_nodata(np.float32("nan"), encoded=False)
    buf = io.BytesIO()
    dem.astype("float32").rio.to_raster(buf, driver="GTiff", compress="deflate")
    return buf.getvalue()


def fetch_dem(aoi: AOI, resolution: int = RESOLUTION_M, force: bool = False) -> cache.Artifact:
    """Pull (or reuse) the DEM for ``aoi`` as a UTM GeoTIFF in the cache."""
    key = f"dem/{aoi.name}/{resolution}m"
    return cache.produce(
        key=key,
        source_key=aoi.dem_source,
        url="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
        rel_path=f"terrain/{aoi.name}_dem_{resolution}m_utm.tif",
        builder=lambda: _build_dem_geotiff(aoi, resolution),
        force=force,
        note=(
            f"USGS 3DEP DEM for AOI {aoi.name} at {resolution} m, reprojected to local UTM "
            "for metric line-of-sight geometry."
        ),
    )


def summarize(art: cache.Artifact) -> dict[str, Any]:
    """Terrain statistics that justify the AOI choice and parameterize the LOS model."""
    import rasterio

    with rasterio.open(art.abs_path) as ds:
        band = ds.read(1, masked=True).astype("float64")
        transform, crs, shape = ds.transform, ds.crs, (ds.height, ds.width)

    z = band.compressed()
    px = abs(transform.a)
    gy, gx = np.gradient(np.ma.filled(band, np.nan), px, px)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope = slope_deg[np.isfinite(slope_deg)]

    return {
        "crs": str(crs),
        "grid_shape": [int(shape[0]), int(shape[1])],
        "pixel_size_m": round(px, 3),
        "extent_km": [
            round(shape[1] * px / 1000.0, 2),
            round(shape[0] * px / 1000.0, 2),
        ],
        "origin_easting_m": round(transform.c, 2),
        "origin_northing_m": round(transform.f, 2),
        "elevation_m": {
            "min": round(float(z.min()), 1),
            "max": round(float(z.max()), 1),
            "mean": round(float(z.mean()), 1),
            "p05": round(float(np.percentile(z, 5)), 1),
            "p95": round(float(np.percentile(z, 95)), 1),
            "relief": round(float(z.max() - z.min()), 1),
        },
        "slope_deg": {
            "mean": round(float(slope.mean()), 2),
            "p95": round(float(np.percentile(slope, 95)), 2),
            "max": round(float(slope.max()), 2),
        },
        "masking_potential": _masking_potential(band, px),
    }


def _masking_potential(band: "np.ma.MaskedArray", px: float) -> dict[str, Any]:
    """How much of the AOI can actually hide an aircraft from a ridge-top sensor.

    A cheap proxy for the real LOS test: for a sensor placed on the highest terrain in the AOI,
    what fraction of the grid lies below the straight-line sightline to that peak, i.e. how much
    terrain has to be climbed over to be seen. Reported as a sanity check that the AOI has
    enough relief for masking to be a learnable mechanic, not as the detection model itself.
    """
    z = np.ma.filled(band, np.nan)
    finite = np.isfinite(z)
    peak_idx = np.unravel_index(np.nanargmax(z), z.shape)
    peak_z = float(z[peak_idx])
    rows, cols = np.indices(z.shape)
    dist_m = np.hypot(rows - peak_idx[0], cols - peak_idx[1]) * px
    # Straight sightline from the peak, ignoring Earth curvature at these ranges (<80 km,
    # where the 4/3-Earth bulge is ~130 m and is applied properly in the env's LOS model).
    with np.errstate(invalid="ignore", divide="ignore"):
        # Terrain that rises above the direct line from the peak to a target 100 m AGL.
        shadowed = finite & (z + 100.0 < peak_z - dist_m * 0.0)  # below peak altitude at all
    return {
        "peak_elevation_m": round(peak_z, 1),
        "fraction_below_peak_minus_100m": round(float(shadowed.sum() / max(finite.sum(), 1)), 3),
        "note": (
            "Proxy statistic only. Confirms the AOI has terrain high enough to occlude "
            "low-altitude flight; the env computes true per-ray LOS with Earth curvature."
        ),
    }
