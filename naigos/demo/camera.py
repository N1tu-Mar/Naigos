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

`tests/test_camera_presets.py` pins the geometry for both packaged AOIs: the
opening view is oblique, looks at the AOI, and starts above the highest ground.
Pure Python; imports nothing from the simulation.
"""

from __future__ import annotations

import math

M_PER_DEG_LAT = 111_132.0

PRESETS = ("terrain_overview", "follow_aircraft", "top_down_analysis")
DEFAULT_PRESET = "terrain_overview"

#: Oblique pitch of the opening view. Shallower than ~-25 degrees puts the
#: horizon in the upper half of the frame; steeper than ~-50 flattens the relief
#: back toward the map it replaced.
OVERVIEW_PITCH_DEG = -34.0

#: Compass heading of the overview camera. Due north puts the AOI's east-west
#: sortie direction across the frame, left to right, which is the axis the
#: aircraft fly.
OVERVIEW_HEADING_DEG = 0.0


def extent_m(bounds: dict) -> tuple[float, float]:
    """(east-west, north-south) size of the AOI in metres."""
    lat_mid = math.radians((bounds["south"] + bounds["north"]) / 2.0)
    w = (bounds["east"] - bounds["west"]) * M_PER_DEG_LAT * math.cos(lat_mid)
    h = (bounds["north"] - bounds["south"]) * M_PER_DEG_LAT
    return w, h


def presets(bounds: dict, terrain: dict | None = None) -> dict:
    """Every preset as plain numbers the page can hand to CesiumJS unchanged.

    `terrain` is the `/scene` terrain block; its `min_m`/`max_m` set the height
    the overview looks at and the floor the camera must stay above.
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
    return {
        "default": DEFAULT_PRESET,
        "terrain_overview": overview,
        "follow_aircraft": follow,
        "top_down_analysis": top,
    }
