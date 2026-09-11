"""The opening camera is oblique, frames the AOI, and starts above its highest ground.

Checked against both packaged AOIs' own bounding boxes and published peak
elevations, so a preset that happens to suit one theatre and not the other
fails here rather than in a screenshot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from naigos.demo import camera

REPO = Path(__file__).resolve().parents[1]


def _aoi(name):
    d = json.loads((REPO / "components" / "aoi" / name / "env.aoi.json").read_text())
    w, s, e, n = d["parameters"]["bbox_wgs84"]
    return {"west": w, "south": s, "east": e, "north": n}


#: (min, max) relief in metres: basin floor / valley floor to the highest peak
#: in the box, from the AOI DEM components (Tochal ~3960 m in Tehran's box;
#: Owens Valley's box reaches the Sierra crest above 4000 m).
AOIS = {"tehran_basin": (1000.0, 4400.0), "owens_valley": (1100.0, 4420.0)}


@pytest.mark.parametrize("name", sorted(AOIS))
def test_the_default_is_the_oblique_terrain_overview(name):
    p = camera.presets(_aoi(name), {"min_m": AOIS[name][0], "max_m": AOIS[name][1]})
    assert p["default"] == "terrain_overview"
    ov = p["terrain_overview"]
    assert -55.0 < ov["pitch_deg"] < -25.0, "the opening view must be oblique, not top-down"
    assert p["top_down_analysis"]["pitch_deg"] == -90.0


@pytest.mark.parametrize("name", sorted(AOIS))
def test_the_overview_looks_at_the_aoi_from_above_its_highest_ground(name):
    b = _aoi(name)
    lo, hi = AOIS[name]
    ov = camera.presets(b, {"min_m": lo, "max_m": hi})["terrain_overview"]
    assert b["west"] < ov["lon"] < b["east"] and b["south"] < ov["lat"] < b["north"]
    assert lo <= ov["height_m"] <= hi
    assert ov["camera_height_m"] > hi + 10_000.0, "the camera must not start inside a mountain"
    # the camera sits south of the AOI centre (heading north), within a couple
    # of AOI lengths -- far enough to frame it, near enough to see relief
    w, h = camera.extent_m(b)
    assert 0.5 * max(w, h) < ov["camera_ground_offset_m"] < 2.0 * max(w, h)
    # and the geometry is self-consistent
    assert ov["camera_height_m"] == pytest.approx(
        ov["height_m"] - ov["range_m"] * math.sin(math.radians(ov["pitch_deg"])), abs=0.2)


def test_the_follow_preset_is_a_chase_view():
    f = camera.presets(_aoi("tehran_basin"))["follow_aircraft"]
    assert f["heading_from"] == "aircraft"
    assert -40.0 < f["pitch_deg"] < 0.0 and 1_000.0 < f["range_m"] < 20_000.0


def test_every_named_preset_is_present():
    p = camera.presets(_aoi("owens_valley"))
    assert set(camera.PRESETS) <= set(p)
