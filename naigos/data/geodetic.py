"""Local ENU metres <-> WGS84, so the env's flat world can be drawn on a globe.

The env works entirely in a local ENU frame because slant range, line of sight
and the CBF barriers all want plain Euclidean metres -- doing geodesy in the
inner loop would be both slower and pointless over a 100 km theatre. Cesium, on
the other hand, wants geodetic degrees on an ellipsoid.

This module is the only place the two meet. It reads the georeference the
research layer already attached to the DEM (a UTM EPSG and the grid origin), so
the transform is exact rather than a small-angle approximation, and the aircraft
appear over the actual ground they were flying over.

Nothing in `naigos/env` or `naigos/rl` imports this: it is a presentation
concern, and keeping it out of the simulation means a bug here can misplace a
marker but can never change a rollout.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


@functools.lru_cache(maxsize=16)
def _transformer(src: str, dst: str):
    """Cached pyproj transformer.

    Building one costs a PROJ database lookup. The terrain endpoint transforms a
    512x512 grid and the live loop transforms every aircraft every tick, so
    rebuilding per call showed up immediately.
    """
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


@dataclass(frozen=True)
class GeoRef:
    """Everything needed to place a local ENU grid on the globe."""

    utm_epsg: int
    origin_easting_m: float
    origin_northing_m: float

    def to_wgs84(self, x_m, y_m):
        """Local ENU metres -> (lon_deg, lat_deg). Accepts scalars or arrays."""
        tf = _transformer(f"EPSG:{self.utm_epsg}", "EPSG:4326")
        e = np.asarray(x_m, dtype=np.float64) + self.origin_easting_m
        n = np.asarray(y_m, dtype=np.float64) + self.origin_northing_m
        lon, lat = tf.transform(e, n)
        return np.asarray(lon), np.asarray(lat)

    def from_wgs84(self, lon_deg, lat_deg):
        """(lon_deg, lat_deg) -> local ENU metres."""
        tf = _transformer("EPSG:4326", f"EPSG:{self.utm_epsg}")
        e, n = tf.transform(np.asarray(lon_deg, dtype=np.float64), np.asarray(lat_deg, dtype=np.float64))
        return np.asarray(e) - self.origin_easting_m, np.asarray(n) - self.origin_northing_m

    def corners(self, extent_x_m: float, extent_y_m: float) -> dict[str, float]:
        """Geodetic bounding box of the local grid, for camera framing.

        Uses all four corners rather than two: a UTM rectangle is not a
        lat/lon rectangle, so taking only SW and NE would clip the theatre.
        """
        xs = np.array([0.0, extent_x_m, 0.0, extent_x_m])
        ys = np.array([0.0, 0.0, extent_y_m, extent_y_m])
        lon, lat = self.to_wgs84(xs, ys)
        return {
            "west": float(lon.min()), "east": float(lon.max()),
            "south": float(lat.min()), "north": float(lat.max()),
        }

    def as_dict(self) -> dict:
        return {
            "utm_epsg": self.utm_epsg,
            "origin_easting_m": self.origin_easting_m,
            "origin_northing_m": self.origin_northing_m,
        }


def georef_from_enu(grid) -> GeoRef:
    """Build a GeoRef from a `naigos.data.enu.ENUGrid`."""
    return GeoRef(
        utm_epsg=int(str(grid.crs).split(":")[-1]),
        origin_easting_m=float(grid.origin_easting_m),
        origin_northing_m=float(grid.origin_northing_m),
    )
