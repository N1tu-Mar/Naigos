"""Single load point for everything the env needs, assembled from cited component specs.

The env never reads ``data_cache/`` and never touches the network. It loads a ``Theatre``, and
every field on it traces back through a ``components/*.json`` to a source and a sha256.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from naigos.research.cache import CACHE_DIR
from naigos.research.spec import load_component

from .terrain import TerrainGrid


@dataclass
class Airfield:
    ident: str
    name: str
    easting_m: float
    northing_m: float
    elevation_m: float
    longest_runway_m: float | None
    runway_heading_deg: float | None

    @property
    def usable_as_departure(self) -> bool:
        """A departure point needs a runway with a known length and heading."""
        return self.longest_runway_m is not None and self.runway_heading_deg is not None


@dataclass
class AirframeEnvelope:
    """Point-mass airframe limits, calibrated from measured traffic.

    ``max_bank_deg`` is a design choice, not a measurement: civil traffic never banks hard, so
    the measured p95 bounds what airliners *do*, while the coordinated-turn relation
    omega = g*tan(phi)/V -- validated against that same data -- is what extrapolates to the
    manoeuvre limits an evading aircraft would actually use.
    """

    min_speed_ms: float
    cruise_speed_ms: float
    max_speed_ms: float
    max_climb_rate_ms: float
    max_descent_rate_ms: float
    max_long_accel_ms2: float
    observed_p95_bank_deg: float
    observed_max_bank_deg: float
    max_bank_deg: float
    service_ceiling_m: float

    @property
    def max_load_factor(self) -> float:
        """n = 1/cos(phi) in a coordinated level turn."""
        import math

        return 1.0 / math.cos(math.radians(self.max_bank_deg))

    def max_turn_rate_deg_s(self, speed_ms: float) -> float:
        """omega = g*tan(phi_max)/V, the g-limit expressed as a turn-rate cap."""
        import math

        return math.degrees(9.80665 * math.tan(math.radians(self.max_bank_deg)) / max(speed_ms, 1.0))


@dataclass
class Theatre:
    """Everything the env is parameterized by, with provenance attached."""

    name: str
    utm_epsg: int
    bbox_wgs84: tuple[float, float, float, float]
    terrain: TerrainGrid
    airfields: list[Airfield]
    airframe: AirframeEnvelope
    threat_classes: dict[str, Any]
    effective_earth_factor_k: float
    surface_density_ratio: float
    density_profile: list[dict[str, Any]]
    provenance: dict[str, str]

    def departure_fields(self) -> list[Airfield]:
        return [a for a in self.airfields if a.usable_as_departure]


def _dem_path() -> Path:
    comp = load_component("data.terrain_dem")
    npz = [a for a in comp["cached_artifacts"] if a["path"].endswith(".npz")]
    if not npz:
        raise FileNotFoundError("no derived DEM npz in data.terrain_dem; re-run naigos-research")
    path = CACHE_DIR / npz[0]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `naigos-research` to rebuild the cache")
    return path


def load_theatre() -> Theatre:
    """Assemble the theatre from the emitted component specs."""
    aoi = load_component("env.aoi")
    fields_c = load_component("data.airfields")
    atm = load_component("data.atmosphere")
    det = load_component("model.detection")
    env_c = load_component("data.flight_envelope")

    grid = TerrainGrid.from_npz(_dem_path())

    airfields = []
    for a in fields_c["parameters"]["airfields"]:
        if a["elevation_m"] is None:
            continue
        headings = [r["heading_deg_true"] for r in a["runways"] if r["heading_deg_true"] is not None]
        airfields.append(Airfield(
            ident=a["ident"], name=a["name"],
            easting_m=a["utm_easting_m"], northing_m=a["utm_northing_m"],
            elevation_m=a["elevation_m"], longest_runway_m=a["longest_runway_m"],
            runway_heading_deg=headings[0] if headings else None,
        ))

    ev = env_c["evidence"]
    airframe = AirframeEnvelope(
        min_speed_ms=ev["ground_speed_ms"]["p1"],
        cruise_speed_ms=ev["ground_speed_ms"]["p50"],
        max_speed_ms=ev["ground_speed_ms"]["max"],
        max_climb_rate_ms=ev["climb_rate_ms"]["p95"],
        max_descent_rate_ms=ev["descent_rate_ms"]["p95"],
        max_long_accel_ms2=ev["long_accel_ms2"]["p95"],
        observed_p95_bank_deg=ev["implied_bank_deg"]["p95"],
        observed_max_bank_deg=ev["implied_bank_deg"]["max"],
        max_bank_deg=60.0,  # 2 g; a manoeuvre limit, not a civil observation. See docstring.
        service_ceiling_m=ev["baro_altitude_m"]["max"],
    )

    return Theatre(
        name=aoi["parameters"]["name"],
        utm_epsg=aoi["parameters"]["utm_epsg"],
        bbox_wgs84=tuple(aoi["parameters"]["bbox_wgs84"]),
        terrain=grid,
        airfields=airfields,
        airframe=airframe,
        threat_classes=det["parameters"]["threat_classes"],
        effective_earth_factor_k=atm["parameters"]["effective_earth_factor_k"],
        surface_density_ratio=atm["parameters"]["surface_density_ratio"],
        density_profile=atm["evidence"]["levels"],
        provenance={c: load_component(c)["generated_at"] for c in (
            "env.aoi", "data.terrain_dem", "data.airfields",
            "data.atmosphere", "data.flight_envelope", "model.detection",
        )},
    )
