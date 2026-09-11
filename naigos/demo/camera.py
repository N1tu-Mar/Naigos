"""Camera presets for the globe viewer, computed from the AOI rather than hard-coded.

The page used to open with `camera.flyTo(Rectangle)`: straight down at the
whole AOI. That is the one angle from which 3 km of relief is invisible -- the
surface reads as a flat map, the domes as circles and the aircraft as dots. So
the opening view is now an oblique one, and top-down is a preset you choose.

Three presets, all derived from the scene's own bounds and terrain range, so
the same code frames Tehran (80 x 100 km, 1 km basin floor under a 4 km range)
and Owens Valley (100 x 130 km, a valley between two 4 km ranges):

    terrain_overview   the default: looking north across the AOI from its
                       southern edge, pitched down enough that the whole box
                       is in frame and far enough that the relief reads as
                       relief rather than as a wall
    follow_aircraft    chase view behind the first live aircraft
    top_down_analysis  straight down over the AOI -- useful for reading
                       routes against envelopes, never the first impression
    urban_overview     the city: oblique over the densest urban chunk, looking
                       north-north-east so the rooflines fill the foreground
                       and the Alborz rises behind them. The default for
                       --visual urban-presentation, and available in every mode
    street_canyon      low and close over the same focus, among the rooftops

`tests/test_camera_presets.py` pins the geometry for both packaged AOIs: the
opening view is oblique, looks at the AOI, and starts above the highest ground.
Pure Python; imports nothing from the simulation.
"""

from __future__ import annotations

import math

M_PER_DEG_LAT = 111_132.0

PRESETS = ("terrain_overview", "follow_aircraft", "top_down_analysis",
           "urban_overview", "street_canyon")
DEFAULT_PRESET = "terrain_overview"
URBAN_DEFAULT_PRESET = "urban_overview"

#: `--camera` spellings -> preset keys. Hyphenated on the command line, keyed
#: with underscores everywhere else (the page, /scene, the tests).
CLI_PRESETS = {
    "terrain-overview": "terrain_overview",
    "urban-overview": "urban_overview",
    "street-canyon": "street_canyon",
    "follow-aircraft": "follow_aircraft",
    "analysis-topdown": "top_down_analysis",
    "coastal-corridor": "coastal_corridor",
    "valley-overview": "valley_overview",
}

#: Views only a city config can supply, because only the city knows where its
#: coast or its valley is. Asking for one on a theatre that does not declare it
#: is an error, not a guess (naigos.demo.cities).
CITY_ONLY_PRESETS = ("coastal_corridor", "valley_overview")

#: The city views. Oblique, never a rectangle fly-to: pitched shallow enough
#: that walls read as walls and the range behind the basin stays in frame, and
#: close enough that a 20 m footprint spans several pixels. Heading NNE, so the
#: camera sits downhill of the city and looks up the slope toward the ridge.
URBAN_HEADING_DEG = 18.0
URBAN_OVERVIEW_PITCH_DEG = -17.0
URBAN_OVERVIEW_RANGE_M = 5_200.0
STREET_CANYON_PITCH_DEG = -7.0
STREET_CANYON_RANGE_M = 650.0
#: Aim this far above the ground at the focus, so the frame centres on the
#: rooftops rather than the street.
URBAN_TARGET_AGL_M = 25.0
STREET_TARGET_AGL_M = 12.0


def preset_key(name: str | None, urban_mode: bool = False) -> str:
    """A `--camera` value (either spelling) -> a preset key; None -> the mode's default."""
    if name is None:
        return URBAN_DEFAULT_PRESET if urban_mode else DEFAULT_PRESET
    key = CLI_PRESETS.get(name, name)
    if key not in PRESETS + CITY_ONLY_PRESETS:
        raise ValueError(f"unknown camera preset {name!r}; expected one of {', '.join(CLI_PRESETS)}")
    return key

#: Oblique pitch of the opening view. Shallower than ~-25 degrees puts the
#: horizon in the upper half of the frame; steeper than ~-50 flattens the relief
#: back toward the map it replaced.
OVERVIEW_PITCH_DEG = -34.0

#: Compass heading of the overview camera. Due north puts the AOI's east-west
#: sortie direction across the frame, left to right, which is the axis the
#: aircraft fly.
OVERVIEW_HEADING_DEG = 0.0


#: One light for the whole scene: the hillshade draped on the DEM and the
#: directional light the models are lit by. Cartographic convention -- sun in
#: the north-west, 45 degrees up -- so ridges read as raised, not incised.
#: Fixed rather than taken from the clock: a replay's epoch is an arbitrary
#: 2000-01-01T00:00Z, which is night over both AOIs.
SUN_AZIMUTH_DEG = 315.0
SUN_ALTITUDE_DEG = 45.0


def sun_vector_enu() -> tuple[float, float, float]:
    """Unit vector TOWARD the sun in east-north-up."""
    az, el = math.radians(SUN_AZIMUTH_DEG), math.radians(SUN_ALTITUDE_DEG)
    return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))


def lighting() -> dict:
    return {"sun_azimuth_deg": SUN_AZIMUTH_DEG, "sun_altitude_deg": SUN_ALTITUDE_DEG}


def hillshade_rectangle(meta: dict) -> tuple[float, float, float, float]:
    """(west, south, east, north) the hillshade image must be draped over.

    The shading has one pixel per terrain POST, and posts sit on the grid's
    edges -- post 0 at `west`, post nx-1 at `east`. An image stretched over
    exactly [west, east] puts pixel centres half a pixel inside those posts,
    shifting every shaded ridge up to half a cell off the mesh it shades. Half
    a post spacing of margin on each side puts each pixel centre on its post.
    """
    sx = (meta["east"] - meta["west"]) / (meta["nx"] - 1)
    sy = (meta["north"] - meta["south"]) / (meta["ny"] - 1)
    return (meta["west"] - sx / 2, meta["south"] - sy / 2,
            meta["east"] + sx / 2, meta["north"] + sy / 2)


def hillshade(heights, meta: dict, z_factor: float = 1.0):
    """Reference hillshade in [0, 1], (ny, nx), row 0 = SOUTH -- what the page computes.

    Horn gradient; lit as `max(0, n . sun)` with n the surface normal of the
    (optionally exaggerated) DEM. `assets/cesium.html` `shadeValue()` is the same
    arithmetic; the tests check this one's physics and the page's text.
    """
    import numpy as np

    h = np.asarray(heights, dtype=np.float64).reshape(meta["ny"], meta["nx"]) * z_factor
    lat_mid = math.radians((meta["south"] + meta["north"]) / 2)
    ex = (meta["east"] - meta["west"]) / (meta["nx"] - 1) * 111_320.0 * math.cos(lat_mid)
    ey = (meta["north"] - meta["south"]) / (meta["ny"] - 1) * M_PER_DEG_LAT
    p = np.pad(h, 1, mode="edge")
    # rows increase NORTH here, so +1 row is north
    dzdx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8 * ex)
    dzdy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / (8 * ey)
    sx, sy, sz = sun_vector_enu()
    return np.clip((-dzdx * sx - dzdy * sy + sz) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0), 0.0, 1.0)


def extent_m(bounds: dict) -> tuple[float, float]:
    """(east-west, north-south) size of the AOI in metres."""
    lat_mid = math.radians((bounds["south"] + bounds["north"]) / 2.0)
    w = (bounds["east"] - bounds["west"]) * M_PER_DEG_LAT * math.cos(lat_mid)
    h = (bounds["north"] - bounds["south"]) * M_PER_DEG_LAT
    return w, h


def _orbit(lon: float, lat: float, target_h: float, heading: float, pitch_deg: float,
           rng: float) -> dict:
    pitch = math.radians(pitch_deg)
    return {
        "lon": round(lon, 6), "lat": round(lat, 6), "height_m": round(target_h, 1),
        "heading_deg": heading, "pitch_deg": pitch_deg, "range_m": round(rng, 1),
        "camera_height_m": round(target_h - rng * math.sin(pitch), 1),
        "camera_ground_offset_m": round(rng * math.cos(pitch), 1),
    }


def presets(bounds: dict, terrain: dict | None = None, urban: dict | None = None,
            default: str | None = None, city=None, sample=None) -> dict:
    """Every preset as plain numbers the page can hand to CesiumJS unchanged.

    `terrain` is the `/scene` terrain block; its `min_m`/`max_m` set the height
    the overview looks at and the floor the camera must stay above.

    `urban` is the city focus -- ``{"lon", "lat", "ground_m"}``, the densest
    chunk of the local city layer or the centre of the AOI's documented urban
    box -- with the DEM height under it. Without one the city views aim at the
    AOI centre over the lowest ground, which is the honest guess for a basin.
    """
    lo = float((terrain or {}).get("min_m", 0.0))
    hi = float((terrain or {}).get("max_m", 0.0))
    w, h = extent_m(bounds)
    lon_c = (bounds["west"] + bounds["east"]) / 2.0
    lat_c = (bounds["south"] + bounds["north"]) / 2.0
    # Aim a little into the relief rather than at the basin floor, so the
    # ridges sit in the middle of the frame instead of along its top edge.
    target_h = lo + 0.35 * (hi - lo)
    # Far enough that the AOI's longer side fits the frame at this pitch.
    rng = 0.80 * max(w, h)
    pitch = math.radians(OVERVIEW_PITCH_DEG)
    cam_h = target_h - rng * math.sin(pitch)
    overview = {
        "lon": round(lon_c, 6), "lat": round(lat_c, 6), "height_m": round(target_h, 1),
        "heading_deg": OVERVIEW_HEADING_DEG, "pitch_deg": OVERVIEW_PITCH_DEG,
        "range_m": round(rng, 1),
        # derived, for the diagnostic and the tests; the page recomputes it
        "camera_height_m": round(cam_h, 1),
        "camera_ground_offset_m": round(rng * math.cos(pitch), 1),
    }
    top = {
        "west": bounds["west"], "south": bounds["south"],
        "east": bounds["east"], "north": bounds["north"],
        "pitch_deg": -90.0,
    }
    follow = {
        # behind and above, looking along the aircraft's own heading
        "pitch_deg": -18.0, "range_m": 4_500.0, "heading_from": "aircraft",
    }
    focus = urban or {"lon": lon_c, "lat": lat_c, "ground_m": lo}
    ground = float(focus.get("ground_m", lo) if focus.get("ground_m") is not None else lo)
    urban_overview = _orbit(focus["lon"], focus["lat"], ground + URBAN_TARGET_AGL_M,
                            URBAN_HEADING_DEG, URBAN_OVERVIEW_PITCH_DEG, URBAN_OVERVIEW_RANGE_M)
    street = _orbit(focus["lon"], focus["lat"], ground + STREET_TARGET_AGL_M,
                    URBAN_HEADING_DEG, STREET_CANYON_PITCH_DEG, STREET_CANYON_RANGE_M)
    out = {
        "default": preset_key(default) if default else DEFAULT_PRESET,
        "terrain_overview": overview,
        "follow_aircraft": follow,
        "top_down_analysis": top,
        "urban_overview": urban_overview,
        "street_canyon": street,
    }
    if city is not None:
        # A theatre with a city config declares its own city views, each
        # checked against the drawn DEM, its safe region and its protected
        # zones (naigos.demo.cities.city_presets raises rather than move one).
        from . import cities

        out.update(cities.city_presets(city, sample))
        out["city"] = city.aoi
        # Zones the page must keep the free and follow cameras away from, and
        # must not draw imagery or provider tiles over. Name and box only.
        out["protected"] = city.page_zones()
        out["follow_aircraft"] = {**follow, "view_half_angle_deg": cities.VIEW_HALF_ANGLE_DEG,
                                  "framing_range_m": cities.FRAMING_RANGE_M}
    if out["default"] not in out:
        raise ValueError(f"camera preset {out['default']!r} is not available for this theatre; "
                         f"available: {', '.join(k for k in out if k in PRESETS + CITY_ONLY_PRESETS)}")
    out["available"] = [k for k in PRESETS + CITY_ONLY_PRESETS if k in out]
    return out
