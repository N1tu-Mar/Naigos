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
from dataclasses import dataclass


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

DEFAULT_AOI = "owens_valley"


def get_aoi(name: str | None = None) -> AOI:
    key = name or DEFAULT_AOI
    if key not in AOIS:
        raise KeyError(f"unknown AOI {key!r}; known: {sorted(AOIS)}")
    return AOIS[key]
