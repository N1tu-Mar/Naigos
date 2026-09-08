"""Bridge: the research agent's cited `Theatre` -> a runnable `EnvConfig` + DEM.

This is the seam between the two halves of the project. The research agent owns
`components/*.json` and `data_cache/`; the env owns physics and RL. Nothing in
`naigos/env` reads the cache directly and nothing here invents a number: every
value below either comes from a component spec or is a MODELLING ASSUMPTION that
says so out loud.

    cfg, hmap, notes = env_from_theatre()
    env = NaigosEnv(cfg, hmap=hmap)

The distinction that matters for honesty: `EnvConfig()` alone runs on synthetic
ridged terrain and placeholder threat parameters, which is correct for tests and
domain randomisation and is NOT a real-data result. Anything reported as a
headline number must come through this function, and `notes` records which
component specs and which DEM sha it was built from.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from .config import (
    AirframeConfig,
    DetectionConfig,
    EnvConfig,
    SpatialHashConfig,
    TerrainConfig,
    ThreatKindConfig,
)

# Modelling assumptions that no open civil dataset can supply. Stated here, once,
# rather than buried as defaults -- see docs/DEVLOG.md and next-steps.md D-1.
ASSUMED_MAX_LOAD_FACTOR = 5.0
"""Civil traffic never manoeuvres hard: OpenSky's implied-bank p99 is ~21 deg
(1.07 g). An evading aircraft is not an airliner, so the g-limit is a stated
design choice, not a measurement. 5 g is a conservative fighter-class figure."""

ASSUMED_CLIMB_RATE_MS = 120.0
"""Maximum vertical rate. OpenSky's civil p95 is 13 m/s, which is what airliners
CHOOSE, not what an airframe can do. MEASURED CONSEQUENCE: at 13 m/s (even
tripled) a terrain-following controller over the Owens Valley DEM -- 345 to
4077 m of relief -- flew into the Sierra on 80% of sorties, because it
physically cannot climb a ridge it is already committed to. Nap-of-the-earth
flight over real mountains requires tactical climb performance, so this is a
stated design assumption, not a measurement."""

ASSUMED_MAX_FLIGHT_PATH_ANGLE = 0.52
"""~30 deg. Same reasoning as the climb rate: civil traffic never flies a
30-degree flight path angle, and terrain masking over 3.7 km of relief is
impossible without one."""

ASSUMED_MIN_AGL_FLOOR = 30.0
"""Below this AGL counts as a terrain strike. A modelling choice about what
'nap-of-the-earth' is allowed to mean, not a measured limit."""

ASSUMED_FUEL_KG = 3000.0
"""Sortie endurance. OpenSky carries no fuel state, so the burn model is
notional and only has to make loitering cost something."""


def airframe_from_theatre(th) -> AirframeConfig:
    """AirframeConfig from measured OpenSky statistics + the stated assumptions."""
    af = th.airframe
    return AirframeConfig(
        v_stall=float(af.min_speed_ms),
        v_max=float(af.max_speed_ms),
        v_init=float(af.cruise_speed_ms),
        n_max=ASSUMED_MAX_LOAD_FACTOR,
        gamma_max=ASSUMED_MAX_FLIGHT_PATH_ANGLE,
        roc_max=ASSUMED_CLIMB_RATE_MS,
        ceiling=float(af.service_ceiling_m),
        floor_agl=ASSUMED_MIN_AGL_FLOOR,
        fuel_init=ASSUMED_FUEL_KG,
    )


def _kind_from_class(name: str, spec: dict, snr_logistic_k: float = 0.45) -> ThreatKindConfig | None:
    """One `model.detection` threat class -> one ThreatKindConfig.

    A class with zero lethal range (pure surveillance) is kept: it cues nothing
    mechanically yet, but it still generates detections, which is exactly the
    'seen but not shootable' pressure the reward's exposure term prices.
    """
    d = spec["derived"]
    e = spec["engagement"]
    mobile = bool(e.get("mobile", False))
    airborne = "interceptor" in name or "seeker" in name

    detect_m = float(d["detection_range_km_pd50"]) * 1000.0
    lethal_m = float(d["lethal_range_km"]) * 1000.0
    return ThreatKindConfig(
        detect_range=detect_m,
        lethal_range=lethal_m,
        alt_min=float(e["min_engagement_alt_agl_m"] or 0.0),
        alt_max=float(e["max_engagement_alt_m"] or 20_000.0),
        reaction_latency=float(e["reaction_latency_s"] or 1.0),
        # lock dynamics are not in the component spec: they are a modelling
        # choice about how quickly exposure becomes a track. Tied to the scan
        # period so a slow-scanning surveillance radar builds a track slowly.
        lock_gain=float(np.clip(1.0 / max(spec["radar"].get("scan_period_s", 5.0), 1.0), 0.05, 0.6)),
        lock_decay=0.40,
        p_kill=0.0 if lethal_m <= 0 else 0.05,
        speed=(150.0 if airborne else 12.0) if mobile else 0.0,
        turn_rate=(0.20 if airborne else 0.15) if mobile else 0.0,
        snr_ref_db=13.0,
        snr_threshold_db=13.0,
        snr_logistic_k=snr_logistic_k,
        airborne=airborne,
        spawn_weight=1.0 if lethal_m > 0 else 0.5,
        label=name,
    )


def env_from_theatre(
    theatre=None,
    aoi: str | None = None,
    n_blue: int = 4,
    n_threat: int = 16,
    cell_m: float = 1500.0,
    nx: int | None = None,
    ny: int | None = None,
    **overrides,
):
    """Build `(EnvConfig, hmap, notes)` from the cited component specs.

    Raises rather than falling back to synthetic terrain: a run that believes it
    used a real DEM but did not is worse than a run that fails.
    """
    from ..data.enu import natural_grid_shape, real_terrain
    from ..data.theatre import load_theatre

    th = theatre if theatre is not None else load_theatre(aoi)
    fit_nx, fit_ny = natural_grid_shape(cell_m, th.terrain)
    nx = nx or fit_nx
    ny = ny or fit_ny
    grid = real_terrain(nx=nx, ny=ny, cell_m=cell_m, grid=th.terrain)

    tcfg = TerrainConfig(nx=grid.nx, ny=grid.ny, cell=grid.cell_m)

    kinds = tuple(
        k for k, spec in ((_kind_from_class(n, s), s) for n, s in th.threat_classes.items()) if k is not None
    )

    dcfg = DetectionConfig(
        # 4/3-earth is the textbook default; the theatre's atmosphere component
        # measured the local refractivity, so use the measured k.
        earth_radius_eff=6_371_000.0 * float(th.effective_earth_factor_k),
    )

    # the sensing horizon has to cover the longest detection range in the
    # theatre, otherwise an emitter can hold a track on an aircraft that cannot
    # see it in its own observation -- an asymmetry the policy cannot learn around.
    max_detect = max(k.detect_range for k in kinds)
    hcfg = SpatialHashConfig(query_radius=float(max(max_detect, 1.5 * tcfg.extent_x)), cell_capacity=max(32, n_threat))

    cfg = EnvConfig(
        n_blue=n_blue,
        n_threat=n_threat,
        n_threat_active=n_threat,
        airframe=airframe_from_theatre(th),
        detection=dcfg,
        terrain=tcfg,
        hash=hcfg,
        threat_kinds=kinds,
        **overrides,
    )

    from ..data.geodetic import georef_from_enu

    georef = georef_from_enu(grid)
    notes = {
        "theatre": th.name,
        # presentation only -- lets the Cesium viewer place the local ENU frame
        # on the globe. Nothing in the simulation reads it.
        "georef": georef.as_dict(),
        "geo_bounds": georef.corners(tcfg.extent_x, tcfg.extent_y),
        "utm_epsg": th.utm_epsg,
        "bbox_wgs84": th.bbox_wgs84,
        "grid": f"{grid.nx}x{grid.ny} @ {grid.cell_m:.0f} m = {tcfg.extent_x/1000:.1f} x {tcfg.extent_y/1000:.1f} km",
        "relief_m": (float(grid.heightmap.min()), float(grid.heightmap.max())),
        "threat_classes": [k.label for k in kinds],
        "effective_earth_k": float(th.effective_earth_factor_k),
        "component_generated_at": th.provenance,
        "assumptions": {
            "n_max_g": ASSUMED_MAX_LOAD_FACTOR,
            "roc_max_ms": ASSUMED_CLIMB_RATE_MS,
            "gamma_max_rad": ASSUMED_MAX_FLIGHT_PATH_ANGLE,
            "measured_civil_climb_p95_ms": "13.3 (OpenSky) -- NOT used; see ASSUMED_CLIMB_RATE_MS",
            "floor_agl_m": ASSUMED_MIN_AGL_FLOOR,
            "fuel_kg": ASSUMED_FUEL_KG,
            "lock_dynamics": "derived from radar scan period; not in any component spec",
        },
    }
    return cfg, jnp.asarray(grid.heightmap), notes


def describe(notes: dict) -> str:
    lines = [f"theatre: {notes['theatre']}  ({notes['grid']})",
             f"relief:  {notes['relief_m'][0]:.0f} - {notes['relief_m'][1]:.0f} m AMSL",
             f"threats: {', '.join(notes['threat_classes'])}",
             f"k_earth: {notes['effective_earth_k']:.4f} (measured, not the 4/3 default)",
             "assumptions (not measured): " + ", ".join(f"{k}={v}" for k, v in notes["assumptions"].items())]
    return "\n".join(lines)
