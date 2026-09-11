"""Atmosphere (Open-Meteo pressure levels) -> air density, ceiling, and radar refraction.

Two things the env needs from the atmosphere, and one it did not know it needed:

  1. Air density vs altitude. True airspeed, stall speed and available thrust all scale with
     density, so a "service ceiling" is a density statement, not an altitude constant. The AOI
     floor is already at 1.2 km and the terrain tops 4.4 km, where density is ~62% of sea level.
  2. Winds aloft, which set how much of the fuel budget a route actually costs.
  3. The radar refraction factor k. The LOS model uses an effective-Earth radius to fold
     atmospheric refraction into straight-ray geometry. The usual 4/3 is a global average. The
     measured vertical refractivity gradient gives the *local* k, and over a 100 km theatre the
     difference between k=4/3 and the real value moves the radar horizon by kilometres.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .. import cache
from ..aoi import AOI

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

PRESSURE_LEVELS_HPA = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200)

R_DRY = 287.0528  # J/(kg K)
R_VAPOUR = 461.495
G0 = 9.80665
EARTH_RADIUS_M = 6_371_000.0


def _variables() -> list[str]:
    v = ["temperature_2m", "surface_pressure", "pressure_msl", "relative_humidity_2m", "wind_speed_10m"]
    for lvl in PRESSURE_LEVELS_HPA:
        v += [
            f"temperature_{lvl}hPa",
            f"relative_humidity_{lvl}hPa",
            f"geopotential_height_{lvl}hPa",
            f"wind_speed_{lvl}hPa",
            f"wind_direction_{lvl}hPa",
        ]
    return v


def fetch_profile(aoi: AOI, force: bool = False, past_days: int = 2) -> cache.Artifact:
    """Cache a vertical atmospheric profile over the AOI centre."""
    lat, lon = aoi.center
    params = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": ",".join(_variables()),
        "past_days": past_days, "forecast_days": 1, "timezone": "GMT",
    }
    return cache.fetch(
        key=f"open_meteo/profile/{aoi.name}/{aoi.fingerprint}",
        source_key="open_meteo", url=FORECAST_URL,
        rel_path=f"atmosphere/open_meteo_{aoi.name}_{aoi.fingerprint}_profile.json",
        params=params, force=force, validate=_is_profile,
        note=f"Hourly surface + {len(PRESSURE_LEVELS_HPA)}-level profile over AOI centre {aoi.center}.",
    )


def _is_profile(body: bytes) -> bool:
    """A forecast response, not an error text served with HTTP 200."""
    import json

    try:
        doc = json.loads(body)
    except ValueError:
        return False
    return isinstance(doc, dict) and isinstance(doc.get("hourly"), dict)


def saturation_vapour_pressure_pa(t_k: np.ndarray) -> np.ndarray:
    """Buck (1981) equation over water; adequate to <0.3% over the range flown here."""
    t_c = t_k - 273.15
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def moist_air_density(p_pa: np.ndarray, t_k: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Density of humid air from total pressure, temperature and relative humidity."""
    e = np.clip(rh_pct, 0.0, 100.0) / 100.0 * saturation_vapour_pressure_pa(t_k)
    return (p_pa - e) / (R_DRY * t_k) + e / (R_VAPOUR * t_k)


def refractivity_n_units(p_pa: np.ndarray, t_k: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Radio refractivity N in N-units (ITU-R P.453): N = 77.6/T*(P + 4810*e/T), P and e in hPa."""
    e_hpa = np.clip(rh_pct, 0.0, 100.0) / 100.0 * saturation_vapour_pressure_pa(t_k) / 100.0
    p_hpa = p_pa / 100.0
    return 77.6 / t_k * (p_hpa + 4810.0 * e_hpa / t_k)


def effective_earth_factor(dn_dh_per_km: float) -> float:
    """Effective-Earth radius factor k from the refractivity gradient (ITU-R P.834).

    k = 1 / (1 + Re * dn/dh). With the standard atmosphere gradient of -39 N-units/km this
    returns the familiar 4/3. A steeper (more negative) gradient bends rays further down and
    extends the radar horizon; a positive gradient shortens it.
    """
    dn_dh_per_m = dn_dh_per_km * 1e-6 / 1000.0  # N-units/km -> dimensionless n per metre
    return float(1.0 / (1.0 + EARTH_RADIUS_M * dn_dh_per_m))


def derive_profile(art: cache.Artifact) -> dict[str, Any]:
    """Reduce the hourly forecast to a mean vertical profile plus the derived k factor."""
    doc = json.loads(art.abs_path.read_text())
    h = doc["hourly"]

    levels = []
    for lvl in PRESSURE_LEVELS_HPA:
        t = np.asarray(h[f"temperature_{lvl}hPa"], dtype=float) + 273.15
        rh = np.asarray(h[f"relative_humidity_{lvl}hPa"], dtype=float)
        z = np.asarray(h[f"geopotential_height_{lvl}hPa"], dtype=float)
        ws = np.asarray(h[f"wind_speed_{lvl}hPa"], dtype=float) / 3.6  # km/h -> m/s
        ok = np.isfinite(t) & np.isfinite(rh) & np.isfinite(z)
        if not ok.any():
            continue
        p_pa = lvl * 100.0
        rho = moist_air_density(p_pa, t[ok], rh[ok])
        n_units = refractivity_n_units(p_pa, t[ok], rh[ok])
        levels.append({
            "pressure_hPa": lvl,
            "geopotential_height_m": round(float(z[ok].mean()), 1),
            "temperature_K": round(float(t[ok].mean()), 2),
            "relative_humidity_pct": round(float(rh[ok].mean()), 1),
            "density_kg_m3": round(float(rho.mean()), 5),
            "density_ratio_to_sea_level": round(float(rho.mean()) / 1.225, 4),
            "refractivity_N": round(float(n_units.mean()), 2),
            "wind_speed_ms_mean": round(float(np.nanmean(ws)), 2) if np.isfinite(ws).any() else None,
            "wind_speed_ms_p95": round(float(np.nanpercentile(ws, 95)), 2) if np.isfinite(ws).any() else None,
        })

    levels.sort(key=lambda d: d["geopotential_height_m"])

    # Refractivity gradient over the lowest kilometre above the AOI floor, which is the layer
    # that sets the radar horizon for low-altitude flight.
    z = np.array([l["geopotential_height_m"] for l in levels])
    n = np.array([l["refractivity_N"] for l in levels])
    low = z <= z.min() + 3000.0
    slope_per_km = float(np.polyfit(z[low] / 1000.0, n[low], 1)[0]) if low.sum() >= 2 else -39.0
    k = effective_earth_factor(slope_per_km)

    sfc_t = np.asarray(h["temperature_2m"], dtype=float) + 273.15
    sfc_p = np.asarray(h["surface_pressure"], dtype=float) * 100.0
    sfc_rh = np.asarray(h["relative_humidity_2m"], dtype=float)
    sfc_rho = moist_air_density(sfc_p, sfc_t, sfc_rh)

    return {
        "site": {"lat": doc["latitude"], "lon": doc["longitude"], "model_elevation_m": doc["elevation"]},
        "hours_sampled": len(h["time"]),
        "time_range_utc": [h["time"][0], h["time"][-1]],
        "surface": {
            "temperature_K_mean": round(float(np.nanmean(sfc_t)), 2),
            "pressure_Pa_mean": round(float(np.nanmean(sfc_p)), 1),
            "density_kg_m3_mean": round(float(np.nanmean(sfc_rho)), 4),
            "density_ratio_to_sea_level": round(float(np.nanmean(sfc_rho)) / 1.225, 4),
        },
        "levels": levels,
        "refraction": {
            "dN_dh_N_units_per_km_lowest_3km": round(slope_per_km, 2),
            "effective_earth_factor_k": round(k, 4),
            "standard_k": round(4.0 / 3.0, 4),
            "note": (
                "k derived from the measured refractivity gradient (ITU-R P.834). The env's LOS "
                "model takes k as a parameter and defaults to this value rather than to 4/3."
            ),
        },
    }
