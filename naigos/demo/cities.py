"""The city presentation standard: one JSON per theatre, and the camera contract.

    naigos/demo/cities/<aoi>.json

A city config says how a theatre is PRESENTED -- never what it simulates. It
names the urban sub-box the visual cache may cover, the atmosphere profile, the
camera presets and where they may stand, and the coarse regions presentation
effects may originate in. The theatre's protected zones are not repeated here:
they come from the AOI definition (``naigos.research.aoi``) that the research
agent records in the cited ``env.aoi`` component, so there is one source of
truth for what a theatre stays out of.

A theatre is added by adding a file; no shared table is edited, so two
theatres developed on two branches cannot collide.

The camera contract
-------------------
Every fixed preset a city declares is checked here, before a page ever sees
it, against four rules. A preset that breaks one is a config error, not a
camera the page quietly moves:

1. **Above ground.** The camera sits at least ``MIN_AGL_M[kind]`` above the
   simulation's own DEM under it, and the sight line from camera to target
   clears the DEM by ``SIGHTLINE_CLEARANCE_M`` along its whole length -- so
   the target is not hidden behind a ridge the camera cannot see past.
2. **Over the declared safe region.** The camera's ground point and its target
   lie inside the city's ``camera.safe_region`` polygon (drawn over land and
   inside the AOI), so no preset opens offshore or outside the theatre.
3. **Out of protected zones.** Neither the camera nor its target is inside any
   zone with the ``camera`` policy, buffered.
4. **Not framing a protected zone.** No corner or centre of a buffered
   ``camera`` zone within ``FRAMING_RANGE_M`` lies inside the camera's
   forward view cone (``VIEW_HALF_ANGLE_DEG`` either side of its heading).

The follow camera cannot be checked ahead of time -- it goes where the
aircraft goes -- so ``safe_follow_heading`` turns it away from a protected
zone at run time, and the page runs the same arithmetic
(``tests/test_city_presentation.py`` executes both and compares them).

Pure Python; imports nothing from the simulation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..research.aoi import AOI, ProtectedZone, get_aoi

CITIES_DIR = Path(__file__).resolve().parent / "cities"

M_PER_DEG_LAT = 111_132.0

#: The camera vocabulary shared by every city. A city declares the subset it
#: can frame safely; the preset keys match naigos.demo.camera.
CITY_PRESET_KEYS = ("urban_overview", "coastal_corridor", "valley_overview", "street_canyon")
#: What kind of shot each is, which sets how high above the ground it must stay.
PRESET_KIND = {"urban_overview": "overview", "coastal_corridor": "overview",
               "valley_overview": "overview", "street_canyon": "street"}
MIN_AGL_M = {"overview": 300.0, "street": 60.0}
SIGHTLINE_CLEARANCE_M = 5.0
SIGHTLINE_STEP_M = 60.0

#: Horizontal half-angle of the forward view that must not contain a protected
#: zone. CesiumJS's default vertical FOV is 60 degrees; at 16:9 the horizontal
#: half-angle is ~46 degrees, so 55 leaves margin for a wide window.
VIEW_HALF_ANGLE_DEG = 55.0
#: Beyond this distance a zone is below a pixel or two and cannot be "framed".
FRAMING_RANGE_M = 30_000.0


class CityConfigError(ValueError):
    """A city config that breaks the presentation contract."""


# --- geometry ----------------------------------------------------------------------------


def m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def offset(lon: float, lat: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Move ``dist_m`` along compass ``bearing_deg`` (flat-earth; fine at city scale)."""
    b = math.radians(bearing_deg)
    return (lon + dist_m * math.sin(b) / m_per_deg_lon(lat),
            lat + dist_m * math.cos(b) / M_PER_DEG_LAT)


def bearing_deg(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """Compass bearing from point 0 to point 1, [0, 360)."""
    dx = (lon1 - lon0) * m_per_deg_lon((lat0 + lat1) / 2.0)
    dy = (lat1 - lat0) * M_PER_DEG_LAT
    return math.degrees(math.atan2(dx, dy)) % 360.0


def distance_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    dx = (lon1 - lon0) * m_per_deg_lon((lat0 + lat1) / 2.0)
    dy = (lat1 - lat0) * M_PER_DEG_LAT
    return math.hypot(dx, dy)


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles, [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def point_in_polygon(lon: float, lat: float, poly) -> bool:
    """Even-odd rule over a ring of [lon, lat] pairs."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x:
                inside = not inside
    return inside


def zone_points(z: ProtectedZone) -> list[tuple[float, float]]:
    """Corners and centre of a zone's buffered box -- what the view cone is tested against."""
    w, s, e, n = z.buffered()
    return [(w, s), (e, s), (e, n), (w, n), ((w + e) / 2.0, (s + n) / 2.0)]


# --- the terrain the camera must clear ---------------------------------------------------


class TerrainSampler:
    """Bilinear lookup over the viewer's own lat/lon grid (row 0 = south).

    The same grid and the same arithmetic as the page's ``sampleGrid`` inside
    the AOI (``naigos.demo.live.build_terrain_grid`` builds it from the
    simulation's heightmap), so a clearance checked here is a clearance on the
    surface the page draws. Outside the box the edge value is held.
    """

    def __init__(self, heights, meta: dict):
        import numpy as np

        self.h = np.asarray(heights, dtype=np.float64).reshape(meta["ny"], meta["nx"])
        self.m = meta

    @classmethod
    def from_bytes(cls, raw: bytes, meta: dict) -> "TerrainSampler":
        import numpy as np

        return cls(np.frombuffer(raw, dtype="<i2"), meta)

    def __call__(self, lon: float, lat: float) -> float:
        m = self.m
        u = min(max((lon - m["west"]) / (m["east"] - m["west"]), 0.0), 1.0)
        v = min(max((lat - m["south"]) / (m["north"] - m["south"]), 0.0), 1.0)
        fx, fy = u * (m["nx"] - 1), v * (m["ny"] - 1)
        x0, y0 = int(math.floor(fx)), int(math.floor(fy))
        x1, y1 = min(x0 + 1, m["nx"] - 1), min(y0 + 1, m["ny"] - 1)
        tx, ty = fx - x0, fy - y0
        h = self.h
        return float((h[y0, x0] * (1 - tx) + h[y0, x1] * tx) * (1 - ty)
                     + (h[y1, x0] * (1 - tx) + h[y1, x1] * tx) * ty)


# --- the config ------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPresetSpec:
    """One declared shot: a target on the ground and where to look at it from."""

    key: str
    lon: float
    lat: float
    heading_deg: float
    pitch_deg: float
    range_m: float
    target_agl_m: float
    label: str


@dataclass(frozen=True)
class AmbienceRegion:
    name: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CityConfig:
    aoi: str
    label: str
    atmosphere_profile: str
    urban_bounds: tuple[float, float, float, float]
    urban_rationale: str
    safe_region: tuple[tuple[float, float], ...]
    presets: dict[str, CameraPresetSpec]
    default_preset: str
    ambience_regions: tuple[AmbienceRegion, ...]
    ambience_standoff_m: float
    ambience_edge_margin_m: float
    notes: str = ""
    source_file: str = field(default="", compare=False)

    @property
    def aoi_def(self) -> AOI:
        return get_aoi(self.aoi)

    def zones(self, policy: str) -> tuple[ProtectedZone, ...]:
        """This theatre's protected zones that carry ``policy``."""
        return tuple(z for z in self.aoi_def.protected_zones if policy in z.policies)

    def page_zones(self) -> list[dict]:
        """Zones the page itself enforces (camera clamp, render cutout). Name + box only."""
        out = []
        for z in self.aoi_def.protected_zones:
            pol = [p for p in z.policies if p in ("camera", "render_cutout")]
            if pol:
                out.append({"name": z.name, "bbox": [round(v, 6) for v in z.buffered()],
                            "policies": pol})
        return out


_REQUIRED = ("aoi", "label", "atmosphere_profile", "urban_bounds", "camera", "ambience")


def load_city(path: Path) -> CityConfig:
    """Parse and structurally validate one city config. Strict: typos are errors."""
    from . import atmosphere

    doc = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED if k not in doc]
    unknown = [k for k in doc if k not in _REQUIRED + ("notes",)]
    if missing or unknown:
        raise CityConfigError(f"{path}: missing {missing}, unknown {unknown}")
    if doc["aoi"] != Path(path).stem:
        raise CityConfigError(f"{path}: aoi {doc['aoi']!r} must match the file name")
    aoi = get_aoi(doc["aoi"])
    atmosphere.get_profile(doc["atmosphere_profile"])      # allowlisted, or raises

    ub = doc["urban_bounds"]
    box = (float(ub["west"]), float(ub["south"]), float(ub["east"]), float(ub["north"]))
    if not (aoi.west <= box[0] < box[2] <= aoi.east and aoi.south <= box[1] < box[3] <= aoi.north):
        raise CityConfigError(f"{path}: urban_bounds {box} must lie inside the AOI {aoi.bbox}")

    cam = doc["camera"]
    presets = {}
    for key, p in cam["presets"].items():
        if key not in CITY_PRESET_KEYS:
            raise CityConfigError(f"{path}: unknown camera preset {key!r}; known {CITY_PRESET_KEYS}")
        presets[key] = CameraPresetSpec(
            key=key, lon=float(p["lon"]), lat=float(p["lat"]),
            heading_deg=float(p["heading_deg"]) % 360.0, pitch_deg=float(p["pitch_deg"]),
            range_m=float(p["range_m"]), target_agl_m=float(p.get("target_agl_m", 20.0)),
            label=p.get("label", key.replace("_", " ")),
        )
    if "urban_overview" not in presets or "street_canyon" not in presets:
        raise CityConfigError(f"{path}: every city declares urban_overview and street_canyon")
    default = cam.get("default", "urban_overview")
    if default not in presets:
        raise CityConfigError(f"{path}: default preset {default!r} is not declared")

    amb = doc["ambience"]
    regions = tuple(AmbienceRegion(r["name"], tuple((float(a), float(b)) for a, b in r["polygon"]))
                    for r in amb["regions"])
    if not regions:
        raise CityConfigError(f"{path}: at least one ambience region is required")

    return CityConfig(
        aoi=doc["aoi"], label=doc["label"], atmosphere_profile=doc["atmosphere_profile"],
        urban_bounds=box, urban_rationale=ub.get("rationale", ""),
        safe_region=tuple((float(a), float(b)) for a, b in cam["safe_region"]),
        presets=presets, default_preset=default,
        ambience_regions=regions,
        ambience_standoff_m=float(amb.get("entity_standoff_m", 3000.0)),
        ambience_edge_margin_m=float(amb.get("edge_margin_m", 3000.0)),
        notes=doc.get("notes", ""), source_file=str(path),
    )


def _discover() -> dict[str, CityConfig]:
    out: dict[str, CityConfig] = {}
    for path in sorted(CITIES_DIR.glob("*.json")) if CITIES_DIR.is_dir() else ():
        c = load_city(path)
        out[c.aoi] = c
    return out


CITIES: dict[str, CityConfig] = _discover()


def get_city(aoi: str | None) -> CityConfig | None:
    """The city config for ``aoi``, or None for a theatre presented without one."""
    return CITIES.get(aoi) if aoi else None


# --- the camera contract ----------------------------------------------------------------


def orbit(spec: CameraPresetSpec, ground_m: float) -> dict:
    """A declared shot as the numbers the page hands CesiumJS, plus the derived camera."""
    pitch = math.radians(spec.pitch_deg)
    target_h = ground_m + spec.target_agl_m
    cam_h = target_h - spec.range_m * math.sin(pitch)
    ground_off = spec.range_m * math.cos(pitch)
    # the camera stands BEHIND its heading: CesiumJS's HeadingPitchRange looks along heading
    clon, clat = offset(spec.lon, spec.lat, spec.heading_deg + 180.0, ground_off)
    return {
        "lon": round(spec.lon, 6), "lat": round(spec.lat, 6), "height_m": round(target_h, 1),
        "heading_deg": round(spec.heading_deg, 2), "pitch_deg": spec.pitch_deg,
        "range_m": round(spec.range_m, 1),
        "camera_height_m": round(cam_h, 1), "camera_ground_offset_m": round(ground_off, 1),
        "camera_lon": round(clon, 6), "camera_lat": round(clat, 6),
        "label": spec.label, "kind": "oblique",
    }


def check_preset(city: CityConfig, spec: CameraPresetSpec, sample=None) -> tuple[dict, list[str]]:
    """The preset as drawn, and every rule it breaks (empty = safe)."""
    problems: list[str] = []
    ground = sample(spec.lon, spec.lat) if sample else 0.0
    o = orbit(spec, ground)
    clon, clat = o["camera_lon"], o["camera_lat"]
    kind = PRESET_KIND[spec.key]

    if sample is not None:
        cam_ground = sample(clon, clat)
        agl = o["camera_height_m"] - cam_ground
        o["camera_agl_m"] = round(agl, 1)
        if agl < MIN_AGL_M[kind]:
            problems.append(f"{spec.key}: camera {agl:.0f} m above the DEM, needs {MIN_AGL_M[kind]:.0f}")
        n = max(2, int(o["range_m"] / SIGHTLINE_STEP_M))
        for i in range(1, n):
            f = i / n
            lon = clon + f * (spec.lon - clon)
            lat = clat + f * (spec.lat - clat)
            h = o["camera_height_m"] + f * (o["height_m"] - o["camera_height_m"])
            if h - sample(lon, lat) < SIGHTLINE_CLEARANCE_M:
                problems.append(f"{spec.key}: terrain hides the target {f * 100:.0f}% along the sight line")
                break

    for label, (lon, lat) in (("camera", (clon, clat)), ("target", (spec.lon, spec.lat))):
        if not point_in_polygon(lon, lat, city.safe_region):
            problems.append(f"{spec.key}: {label} ({lon:.4f}, {lat:.4f}) is outside the safe region")
        for z in city.zones("camera"):
            if z.contains(lon, lat):
                problems.append(f"{spec.key}: {label} is inside protected zone {z.name!r}")

    for z in city.zones("camera"):
        for (zlon, zlat) in zone_points(z):
            if distance_m(clon, clat, zlon, zlat) > FRAMING_RANGE_M:
                continue
            if angle_diff(bearing_deg(clon, clat, zlon, zlat), spec.heading_deg) < VIEW_HALF_ANGLE_DEG:
                problems.append(f"{spec.key}: protected zone {z.name!r} is inside the view cone")
                break
    return o, problems


def city_presets(city: CityConfig, sample=None) -> dict[str, dict]:
    """Every declared preset, checked. Raises on any broken rule: never a silent move."""
    out, problems = {}, []
    for key, spec in city.presets.items():
        o, p = check_preset(city, spec, sample)
        out[key] = o
        problems += p
    if problems:
        raise CityConfigError(f"{city.aoi}: camera presets break the contract: {problems}")
    return out


def safe_follow_heading(cam_lon: float, cam_lat: float, heading: float, zones: list[dict]) -> float:
    """Turn a chase camera away from any protected zone it would frame.

    ``zones`` is ``CityConfig.page_zones()`` (buffered boxes). If a corner or the
    centre of a zone within FRAMING_RANGE_M falls inside the forward cone, the
    heading is rotated -- to the nearer side -- until every such point is at
    least VIEW_HALF_ANGLE_DEG + 5 away. Mirrors ``safeFollowHeading`` in the page.
    """
    h = heading % 360.0
    for _ in range(8):
        worst = None
        for z in zones:
            if "camera" not in z["policies"]:
                continue
            w, s, e, n = z["bbox"]
            for (lon, lat) in ((w, s), (e, s), (e, n), (w, n), ((w + e) / 2, (s + n) / 2)):
                if distance_m(cam_lon, cam_lat, lon, lat) > FRAMING_RANGE_M:
                    continue
                b = bearing_deg(cam_lon, cam_lat, lon, lat)
                d = angle_diff(b, h)
                if d < VIEW_HALF_ANGLE_DEG and (worst is None or d < worst[1]):
                    worst = (b, d)
        if worst is None:
            return round(h, 3)
        b = worst[0]
        side = 1.0 if ((h - b + 360.0) % 360.0) < 180.0 else -1.0
        h = (b + side * (VIEW_HALF_ANGLE_DEG + 5.0)) % 360.0
    return round(h, 3)


def urban_bounds_registry() -> dict[str, dict]:
    """Every city's urban sub-box and its extraction exclusions, for naigos.demo.urban."""
    return {
        c.aoi: {
            "west": c.urban_bounds[0], "south": c.urban_bounds[1],
            "east": c.urban_bounds[2], "north": c.urban_bounds[3],
            "rationale": c.urban_rationale,
            "exclusions": [z.buffered() for z in c.zones("extraction")],
        }
        for c in CITIES.values()
    }
