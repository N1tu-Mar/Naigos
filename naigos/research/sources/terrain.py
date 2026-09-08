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
    from pyproj import Transformer

    epsg = utm_epsg(*aoi.center)
    dem = py3dep.get_dem(aoi.bbox, resolution)
    dem = dem.rio.reproject(f"EPSG:{epsg}", resolution=resolution)
    # Reprojecting a lat/lon box rotates it, so the raster comes back padded with NaN corners.
    # Clip back to the AOI's own UTM envelope: a clean rectangle with no fabricated no-data,
    # which matters because the LOS model treats no-data as "no terrain evidence".
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    corners = [
        tf.transform(lon, lat)
        for lon in (aoi.west, aoi.east)
        for lat in (aoi.south, aoi.north)
    ]
    xs, ys = zip(*corners)
    dem = dem.rio.clip_box(min(xs), min(ys), max(xs), max(ys))
    dem = dem.rio.write_nodata(np.float32("nan"), encoded=False)
    buf = io.BytesIO()
    dem.astype("float32").rio.to_raster(buf, driver="GTiff", compress="deflate")
    return buf.getvalue()


def fetch_dem(aoi: AOI, resolution: int = RESOLUTION_M, force: bool = False) -> cache.Artifact:
    """Pull (or reuse) the DEM for ``aoi`` as a UTM GeoTIFF in the cache."""
    key = f"dem/{aoi.name}/{resolution}m/{aoi.fingerprint}"
    return cache.produce(
        key=key,
        source_key=aoi.dem_source,
        url="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
        rel_path=f"terrain/{aoi.name}_{aoi.fingerprint}_dem_{resolution}m_utm.tif",
        builder=lambda: _build_dem_geotiff(aoi, resolution),
        force=force,
        note=(
            f"USGS 3DEP DEM for AOI {aoi.name} bbox={aoi.bbox} at {resolution} m, reprojected to UTM "
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
        "masking_potential": _masking_potential(art),
    }


def _masking_potential(art: cache.Artifact) -> dict[str, Any]:
    """Run the real LOS test, so the AOI's relief is justified by the mechanic it has to feed.

    A sensor is placed on the highest terrain in the AOI -- the best case for a ground radar --
    and the true per-ray line-of-sight test (with 4/3-Earth refraction) is run against random
    points across the AOI at several heights above ground. If a low-flying aircraft is not
    substantially harder to see than a high one, terrain masking is not a learnable mechanic
    here and the AOI is the wrong choice.
    """
    from naigos.data.terrain import TerrainGrid, masked_fraction

    grid = TerrainGrid.from_geotiff(art.abs_path)
    z = grid.z
    r, c = np.unravel_index(int(np.nanargmax(z)), z.shape)
    sensor = (
        grid.origin_x + (c + 0.5) * grid.pixel_m,
        grid.origin_y - (r + 0.5) * grid.pixel_m,
        float(z[r, c]) + 10.0,  # 10 m mast on the summit
    )
    by_agl = {
        f"{agl}m_agl": round(masked_fraction(grid, sensor, target_agl_m=agl, n_samples=1500), 3)
        for agl in (100, 300, 1000, 3000)
    }
    return {
        "sensor_easting_m": round(sensor[0], 1),
        "sensor_northing_m": round(sensor[1], 1),
        "sensor_elevation_m": round(sensor[2], 1),
        "masked_fraction_by_height_above_ground": by_agl,
        "method": (
            "True per-ray LOS from the AOI high point over 1500 random points, 4/3-Earth "
            "effective radius, 15 m ray sampling. See naigos.data.terrain.masked_fraction."
        ),
    }
