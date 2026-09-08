"""Areas of interest: the geographic box every data pull is scoped to.

The AOI is a config, not a constant. The default was chosen for one reason: terrain relief.
Radar line-of-sight masking is the signature mechanic (spec section 2), and it is only
interesting where terrain actually occludes. Owens Valley gives ~3.2 km of relief across ~30 km
of lateral distance -- the Sierra crest to the west, the White/Inyo range to the east, a flat
valley floor between them -- so nap-of-the-earth flight genuinely breaks LOS rather than
nudging a detection probability.
"""

from __future__ import annotations

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
        west=-118.60,
        south=36.90,
        east=-117.90,
        north=37.60,
        country="US",
        rationale=(
            "Sierra Nevada crest (>4200 m) to the west, White/Inyo Mountains (>4000 m) to the "
            "east, Owens Valley floor at ~1200 m. ~3.2 km of relief over ~30 km makes terrain "
            "masking a real mechanic. Several public airfields (Bishop, Lone Pine, Independence) "
            "supply start points and objectives."
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
}

DEFAULT_AOI = "owens_valley"


def get_aoi(name: str | None = None) -> AOI:
    key = name or DEFAULT_AOI
    if key not in AOIS:
        raise KeyError(f"unknown AOI {key!r}; known: {sorted(AOIS)}")
    return AOIS[key]
