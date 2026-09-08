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


COPERNICUS_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"


def _copernicus_tile_url(lat: int, lon: int) -> str:
    """GLO-30 tiles are 1x1 degree, named by their south-west corner."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    stem = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"{COPERNICUS_BUCKET}/{stem}/{stem}.tif"


def _copernicus_tiles(aoi: AOI) -> list[str]:
    lats = range(math.floor(aoi.south), math.floor(aoi.north) + 1)
    lons = range(math.floor(aoi.west), math.floor(aoi.east) + 1)
    return [_copernicus_tile_url(la, lo) for la in lats for lo in lons]


def _build_copernicus_geotiff(aoi: AOI, resolution: int) -> bytes:
    """Mosaic the GLO-30 tiles covering the AOI and reproject to local UTM.

    Same output contract as the 3DEP path -- a float32 UTM GeoTIFF clipped to the
    AOI envelope with NaN nodata -- so `naigos/data/terrain.py` cannot tell which
    source produced it. That matters: the env must behave identically whichever
    DEM the AOI happens to need.

    Reads through GDAL's /vsicurl/ against Cloud-Optimized GeoTIFFs, so only the
    blocks overlapping the AOI cross the network rather than whole 1-degree tiles.
    """
    import rasterio
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr
    from rasterio.merge import merge
    from rasterio.session import AWSSession  # noqa: F401  (import guard only)

    from ..allowlist import check_url

    urls = _copernicus_tiles(aoi)
    for u in urls:
        check_url(u, aoi.dem_source)  # the allowlist is the only network gate

    env = rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    )
    with env:
        srcs = []
        try:
            for u in urls:
                try:
                    srcs.append(rasterio.open(f"/vsicurl/{u}"))
                except rasterio.RasterioIOError:
                    # GLO-30 has no tiles over open ocean; a missing tile is data,
                    # not an error. Fail only if NOTHING covers the AOI.
                    continue
            if not srcs:
                raise RuntimeError(
                    f"no Copernicus GLO-30 tiles found for {aoi.name} bbox={aoi.bbox}; "
                    f"tried {len(urls)} tiles"
                )
            mosaic, transform = merge(srcs, bounds=aoi.bbox)
        finally:
            for s_ in srcs:
                s_.close()

    band = mosaic[0].astype("float32")
    nodata = srcs[0].nodata if srcs and srcs[0].nodata is not None else -32767.0
    band = np.where(band <= nodata + 1e-6, np.nan, band)

    ny, nx = band.shape
    lons = transform.c + transform.a * (np.arange(nx) + 0.5)
    lats = transform.f + transform.e * (np.arange(ny) + 0.5)
    da = xr.DataArray(band, coords={"y": lats, "x": lons}, dims=("y", "x"))
    da = da.rio.write_crs("EPSG:4326").rio.write_nodata(np.float32("nan"), encoded=False)

    epsg = utm_epsg(*aoi.center)
    dem = da.rio.reproject(f"EPSG:{epsg}", resolution=resolution)
    return _clip_to_aoi_envelope(dem, aoi, epsg)


def _clip_to_aoi_envelope(dem, aoi: AOI, epsg: int) -> bytes:
    """Clip a reprojected DEM back to the AOI's UTM envelope and serialise it.

    Reprojecting a lat/lon box rotates it, so the raster comes back padded with
    NaN corners. Clipping gives a clean rectangle with no fabricated no-data,
    which matters because the LOS model treats no-data as "no terrain evidence".
    """
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    corners = [tf.transform(lon, lat) for lon in (aoi.west, aoi.east) for lat in (aoi.south, aoi.north)]
    xs, ys = zip(*corners)
    dem = dem.rio.clip_box(min(xs), min(ys), max(xs), max(ys))
    dem = dem.rio.write_nodata(np.float32("nan"), encoded=False)
    buf = io.BytesIO()
    dem.astype("float32").rio.to_raster(buf, driver="GTiff", compress="deflate")
    return buf.getvalue()


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


BUILDERS = {
    "usgs_3dep": (
        _build_dem_geotiff,
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
        "USGS 3DEP",
    ),
    "copernicus_dem": (
        _build_copernicus_geotiff,
        COPERNICUS_BUCKET,
        "Copernicus GLO-30 (ESA/Airbus, via the AWS open-data bucket)",
    ),
}


def fetch_dem(aoi: AOI, resolution: int = RESOLUTION_M, force: bool = False) -> cache.Artifact:
    """Pull (or reuse) the DEM for ``aoi`` as a UTM GeoTIFF in the cache.

    Dispatches on ``aoi.dem_source``: 3DEP inside CONUS, Copernicus GLO-30
    everywhere else. Both builders emit the same float32 UTM GeoTIFF, so nothing
    downstream needs to know which one ran.
    """
    if aoi.dem_source not in BUILDERS:
        raise KeyError(f"no DEM builder for source {aoi.dem_source!r}; known: {sorted(BUILDERS)}")
    builder, url, label = BUILDERS[aoi.dem_source]
    key = f"dem/{aoi.name}/{resolution}m/{aoi.fingerprint}"
    return cache.produce(
        key=key,
        source_key=aoi.dem_source,
        url=url,
        rel_path=f"terrain/{aoi.name}_{aoi.fingerprint}_dem_{resolution}m_utm.tif",
        builder=lambda: builder(aoi, resolution),
        force=force,
        note=(
            f"{label} DEM for AOI {aoi.name} bbox={aoi.bbox} at {resolution} m, reprojected to UTM "
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
