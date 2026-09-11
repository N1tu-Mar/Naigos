"""Areas of interest: the geographic box every data pull is scoped to.

The AOI is a config, not a constant. The default was chosen for one reason: terrain relief.
Radar line-of-sight masking is the signature mechanic (spec section 2), and it is only
interesting where terrain actually occludes. Owens Valley gives ~3.2 km of relief across ~30 km
of lateral distance -- the Sierra crest to the west, the White/Inyo range to the east, a flat
valley floor between them -- so nap-of-the-earth flight genuinely breaks LOS rather than
nudging a detection probability.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

#: What a protected zone can be excluded FROM. Every consumer checks the zone's
#: own list, so a zone is never excluded from more -- or less -- than it says.
#:   extraction      no visual (OSM) geometry is kept inside it
#:   airfield        no airfield inside it becomes a start point or objective
#:   camera          no camera preset sits in it, aims at it or frames it
#:   ambience        no presentation-only effect originates in it
#:   render_cutout   the viewer does not draw imagery or provider 3D tiles over it
ZONE_POLICIES = ("extraction", "airfield", "camera", "ambience", "render_cutout")

#: Where file-defined AOIs live: one JSON per theatre. A theatre is added by
#: adding a file, never by editing a shared table, so two theatres developed on
#: two branches cannot collide in this module.
AOI_DEFS_DIR = Path(__file__).resolve().parent / "aois"


@dataclass(frozen=True)
class ProtectedZone:
    """A rectangle inside or near an AOI that the project deliberately stays out of.

    Declared in the AOI definition, recorded verbatim in the ``env.aoi``
    component, and enforced separately by every consumer named in ``policies``.
    It records WHY in ``reason`` and nothing else about the place: no attributes,
    no function, no operational detail -- a name and a box.
    """

    name: str
    west: float
    south: float
    east: float
    north: float
    buffer_m: float
    policies: tuple[str, ...]
    reason: str

    def __post_init__(self):
        unknown = [p for p in self.policies if p not in ZONE_POLICIES]
        if unknown:
            raise ValueError(f"protected zone {self.name!r}: unknown policies {unknown}")
        if not (self.west < self.east and self.south < self.north):
            raise ValueError(f"protected zone {self.name!r}: malformed box")
        if self.buffer_m < 0:
            raise ValueError(f"protected zone {self.name!r}: negative buffer")

    def buffered(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) grown by ``buffer_m`` on every side."""
        lat_c = math.radians((self.south + self.north) / 2.0)
        dlat = self.buffer_m / 110_574.0
        dlon = self.buffer_m / (111_320.0 * math.cos(lat_c))
        return (self.west - dlon, self.south - dlat, self.east + dlon, self.north + dlat)

    def contains(self, lon: float, lat: float, buffered: bool = True) -> bool:
        w, s, e, n = self.buffered() if buffered else (self.west, self.south, self.east, self.north)
        return w <= lon <= e and s <= lat <= n

    def as_dict(self) -> dict:
        return {
            "name": self.name, "bbox_wgs84": [self.west, self.south, self.east, self.north],
            "buffer_m": self.buffer_m, "buffered_bbox_wgs84": [round(v, 6) for v in self.buffered()],
            "policies": list(self.policies), "reason": self.reason,
        }


@dataclass(frozen=True)
class AOI:
    """A named geographic bounding box in WGS84 degrees."""

    name: str
    west: float
    south: float
    east: float
    north: float
    country: str
    rationale: str
    dem_source: str = "usgs_3dep"
    #: How the box was chosen and what it deliberately leaves out. File-defined
    #: AOIs must state one; the built-in three predate the field.
    bounds_policy: str = ""
    #: Places the theatre stays out of. See ProtectedZone.
    protected_zones: tuple[ProtectedZone, ...] = ()
    #: Drop airfields whose published name marks them as military. Opt-in, so
    #: the built-in AOIs' airfield lists -- and their snapshots -- are unchanged.
    exclude_military_airfields: bool = False
    #: The theatre's one-line framing, shown wherever it is drawn.
    scenario: str = ""
    #: Where this AOI was defined: None for the built-ins, else the JSON file.
    #: A file-defined AOI builds its components in isolation (see run.main), so
    #: building it never rewrites the top-level ``components/*.json``.
    definition_file: str | None = field(default=None, compare=False)

    @property
    def scoped(self) -> bool:
        """True for a file-defined theatre, whose research run touches only its own snapshot."""
        return self.definition_file is not None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in WGS84 degrees, the py3dep/OGC ordering."""
        return (self.west, self.south, self.east, self.north)

    @property
    def fingerprint(self) -> str:
        """Short hash of the bounds.

        Cache keys embed this. Without it, editing an AOI's box silently reuses the DEM
        cached for the old box -- every downstream LOS ray, spawn point and threat placement
        would then be computed against terrain that is not there.
        """
        raw = f"{self.west:.6f},{self.south:.6f},{self.east:.6f},{self.north:.6f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:10]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)

    def span_km(self) -> tuple[float, float]:
        """Approximate (east-west, north-south) extent in km."""
        import math

        lat0 = math.radians(self.center[0])
        return (
            (self.east - self.west) * 111.320 * math.cos(lat0),
            (self.north - self.south) * 110.574,
        )


AOIS: dict[str, AOI] = {
    "owens_valley": AOI(
        name="owens_valley",
        west=-118.95,
        south=36.55,
        east=-117.85,
        north=37.75,
        country="US",
        rationale=(
            "Sierra Nevada crest (>4200 m) to the west, White/Inyo Mountains (>4000 m) to the "
            "east, Owens Valley floor at ~1200 m. ~3.2 km of relief over ~30 km makes terrain "
            "masking a real mechanic. The box was sized to capture the real airfields along the "
            "valley (Bishop, Mammoth, Lone Pine, Independence) so start points and objectives "
            "are surveyed positions at surveyed elevations, not spawns inside a mountain."
        ),
    ),
    "front_range": AOI(
        name="front_range",
        west=-105.90,
        south=39.30,
        east=-105.10,
        north=40.10,
        country="US",
        rationale=(
            "Colorado Front Range: steep west-to-east relief gradient against plains. Secondary "
            "AOI used to check that the LOS model is not overfit to a single terrain morphology."
        ),
    ),
    "tehran_basin": AOI(
        name="tehran_basin",
        west=51.05,
        south=35.40,
        east=51.95,
        north=36.30,
        country="IR",
        dem_source="copernicus_dem",
        rationale=(
            "Central Alborz range (Tochal ~3960 m) rising directly north of the Tehran basin "
            "floor (~1100-1700 m): ~2.8 km of relief across ~15 km of lateral distance, a "
            "steeper gradient than Owens Valley. Chosen as a terrain morphology -- a dense urban "
            "basin walled by a single high ridge -- that the LOS model has not been exercised "
            "against, since both existing AOIs are open valleys. Outside 3DEP coverage, so this "
            "is also the AOI that forces the Copernicus GLO-30 path to work. "
            "The threat field over this AOI is randomly spawned and parameterised exactly as it "
            "is everywhere else in the project; nothing here models any real air-defence "
            "disposition, and the guardrail in docs/DATA.md applies unchanged."
        ),
    ),
}

#: Keys a file-defined AOI must carry, and the ones it may.
_REQUIRED_KEYS = ("name", "west", "south", "east", "north", "country", "rationale",
                  "dem_source", "bounds_policy", "scenario")
_OPTIONAL_KEYS = ("protected_zones", "exclude_military_airfields", "notes")


def aoi_from_file(path: Path) -> AOI:
    """Load one ``aois/<name>.json``. Strict: a typo is an error, not a default."""
    doc = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED_KEYS if doc.get(k) in (None, "")]
    unknown = [k for k in doc if k not in _REQUIRED_KEYS + _OPTIONAL_KEYS]
    if missing or unknown:
        raise ValueError(f"{path}: missing {missing}, unknown {unknown}")
    if doc["name"] != Path(path).stem:
        raise ValueError(f"{path}: name {doc['name']!r} must match the file name")
    zones = tuple(
        ProtectedZone(
            name=z["name"], west=float(z["west"]), south=float(z["south"]),
            east=float(z["east"]), north=float(z["north"]),
            buffer_m=float(z.get("buffer_m", 0.0)), policies=tuple(z["policies"]),
            reason=z["reason"],
        )
        for z in doc.get("protected_zones", [])
    )
    return AOI(
        name=doc["name"], west=float(doc["west"]), south=float(doc["south"]),
        east=float(doc["east"]), north=float(doc["north"]), country=doc["country"],
        rationale=doc["rationale"], dem_source=doc["dem_source"],
        bounds_policy=doc["bounds_policy"], protected_zones=zones,
        exclude_military_airfields=bool(doc.get("exclude_military_airfields", False)),
        scenario=doc["scenario"], definition_file=str(path),
    )


def _discover(directory: Path = AOI_DEFS_DIR) -> dict[str, AOI]:
    found: dict[str, AOI] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        aoi = aoi_from_file(path)
        found[aoi.name] = aoi
    return found


def _register(extra: dict[str, AOI]) -> None:
    clash = sorted(set(extra) & set(AOIS))
    if clash:
        raise ValueError(f"file-defined AOIs redefine built-in ones: {clash}")
    AOIS.update(extra)


_register(_discover())

DEFAULT_AOI = "owens_valley"


def get_aoi(name: str | None = None) -> AOI:
    key = name or DEFAULT_AOI
    if key not in AOIS:
        raise KeyError(f"unknown AOI {key!r}; known: {sorted(AOIS)}")
    return AOIS[key]
