"""Airfield and runway geometry (OurAirports) -> sortie start points and objectives.

Why real airfields instead of random spawn points: a start point on a real runway at its real
field elevation, with a real runway heading, means the aircraft's initial state is physically
consistent with the terrain it is flying over. Random spawns inside a mountain range put the
aircraft underground.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from .. import cache
from ..aoi import AOI

BASE = "https://davidmegginson.github.io/ourairports-data"

# Fields with a runway an aircraft could realistically depart from or route to.
FIXED_WING_TYPES = {"small_airport", "medium_airport", "large_airport"}

#: A published name that marks a field as military. OurAirports has no
#: military column, so the name is the only signal it carries. Applied only to
#: AOIs that opt in (``AOI.exclude_military_airfields``): a notional civil
#: theatre must not turn a military field into a start point or objective.
MILITARY_NAME = re.compile(
    r"\b(air\s*base|airbase|air\s*force|military|naval|navy|army|afb|air\s*station)\b",
    re.IGNORECASE,
)


def excluded_reason(aoi: AOI, name: str, lon: float, lat: float) -> str | None:
    """Why an airfield is left out of ``aoi``'s start points, or None to keep it."""
    if aoi.exclude_military_airfields and MILITARY_NAME.search(name or ""):
        return "military_name"
    for z in aoi.protected_zones:
        if "airfield" in z.policies and z.contains(lon, lat):
            return f"protected_zone:{z.name}"
    return None


def fetch_tables(force: bool = False) -> dict[str, cache.Artifact]:
    """Cache the two OurAirports tables we consume."""
    return {
        name: cache.fetch(
            key=f"ourairports/{name}",
            source_key="ourairports",
            url=f"{BASE}/{name}.csv",
            rel_path=f"airspace/ourairports_{name}.csv",
            force=force,
            note=f"OurAirports {name}.csv, public domain, global coverage.",
        )
        for name in ("airports", "runways")
    }


def _rows(art: cache.Artifact) -> list[dict[str, str]]:
    text = art.abs_path.read_text(encoding="utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _f(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


FT_TO_M = 0.3048


def extract_airfields(aoi: AOI, arts: dict[str, cache.Artifact]) -> list[dict[str, Any]]:
    """Airfields inside the AOI, in both WGS84 and the AOI's UTM frame, with runway geometry."""
    from pyproj import Transformer

    from .terrain import utm_epsg

    epsg = utm_epsg(*aoi.center)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    runways: dict[str, list[dict[str, str]]] = {}
    for row in _rows(arts["runways"]):
        if row.get("closed") == "1":
            continue
        runways.setdefault(row["airport_ident"], []).append(row)

    out: list[dict[str, Any]] = []
    for row in _rows(arts["airports"]):
        lat, lon = _f(row["latitude_deg"]), _f(row["longitude_deg"])
        if lat is None or lon is None:
            continue
        if not (aoi.west <= lon <= aoi.east and aoi.south <= lat <= aoi.north):
            continue
        if row["type"] not in FIXED_WING_TYPES:
            continue
        if excluded_reason(aoi, row["name"], lon, lat):
            continue

        east, north = tf.transform(lon, lat)
        elev_ft = _f(row["elevation_ft"])
        rws = []
        for r in runways.get(row["ident"], []):
            length_ft = _f(r["length_ft"])
            heading = _f(r["le_heading_degT"])
            if heading is None and r.get("le_ident", "").rstrip("LRC").isdigit():
                heading = float(r["le_ident"].rstrip("LRC")) * 10.0  # magnetic-ish fallback
            rws.append(
                {
                    "ident": f"{r['le_ident']}/{r['he_ident']}",
                    "length_m": round(length_ft * FT_TO_M, 1) if length_ft else None,
                    "surface": r.get("surface") or None,
                    "heading_deg_true": round(heading, 1) if heading is not None else None,
                    "lighted": r.get("lighted") == "1",
                }
            )

        out.append(
            {
                "ident": row["ident"],
                "name": row["name"],
                "type": row["type"],
                "lat": lat,
                "lon": lon,
                "elevation_m": round(elev_ft * FT_TO_M, 1) if elev_ft is not None else None,
                "utm_easting_m": round(east, 1),
                "utm_northing_m": round(north, 1),
                "utm_epsg": epsg,
                "municipality": row.get("municipality") or None,
                "runways": rws,
                "longest_runway_m": max(
                    (r["length_m"] for r in rws if r["length_m"]), default=None
                ),
            }
        )
    out.sort(key=lambda a: (a["longest_runway_m"] or 0), reverse=True)
    return out


def excluded_counts(aoi: AOI, arts: dict[str, cache.Artifact]) -> dict[str, int]:
    """How many fixed-wing fields inside the box each exclusion rule dropped.

    Counts only, never names: the record that a rule fired is the provenance;
    naming what it fired on would put exactly that list into a committed file.
    """
    counts: dict[str, int] = {}
    for row in _rows(arts["airports"]):
        lat, lon = _f(row["latitude_deg"]), _f(row["longitude_deg"])
        if lat is None or lon is None or row["type"] not in FIXED_WING_TYPES:
            continue
        if not (aoi.west <= lon <= aoi.east and aoi.south <= lat <= aoi.north):
            continue
        why = excluded_reason(aoi, row["name"], lon, lat)
        if why:
            counts[why] = counts.get(why, 0) + 1
    return counts


def reconcile_with_dem(airfields: list[dict[str, Any]], dem_art: cache.Artifact) -> dict[str, Any]:
    """Cross-check published field elevation against the DEM.

    Two independent sources disagreeing by more than a runway's worth of height means one of
    the two frames is wrong -- a datum mismatch or a bad reprojection -- and every spawn point
    and LOS ray downstream would inherit the error. Cheap check, expensive bug.
    """
    import numpy as np

    from naigos.data.terrain import TerrainGrid

    grid = TerrainGrid.from_geotiff(dem_art.abs_path)
    deltas = []
    for a in airfields:
        if a["elevation_m"] is None:
            continue
        dem_z = float(grid.elevation(np.array(a["utm_easting_m"]), np.array(a["utm_northing_m"])))
        if not np.isfinite(dem_z):
            continue
        a["dem_elevation_m"] = round(dem_z, 1)
        a["elevation_delta_m"] = round(dem_z - a["elevation_m"], 1)
        deltas.append(a["elevation_delta_m"])

    arr = np.asarray(deltas, dtype=float)
    return {
        "n_compared": int(arr.size),
        "mean_delta_m": round(float(arr.mean()), 2) if arr.size else None,
        "abs_max_delta_m": round(float(np.abs(arr).max()), 2) if arr.size else None,
        "interpretation": (
            "DEM elevation minus OurAirports published field elevation, in metres. Residuals "
            "of a few metres are expected (3DEP is a bare-earth surface sampled at 30 m; the "
            "published figure is the highest point of the usable landing surface). Tens of "
            "metres would indicate a datum or projection error."
        ),
    }
