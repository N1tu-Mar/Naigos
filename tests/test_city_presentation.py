"""The shared city presentation standard: cameras, atmosphere, ambience, scenario.

Parametrised over every shipped city config (``naigos/demo/cities/*.json``), so
a theatre added on its own branch is held to the whole contract by adding its
config and its terrain fixture -- nothing in this file changes.

Offline and deterministic: the terrain the cameras must clear is the viewer's
own lat/lon grid, stored per theatre in ``tests/fixtures/viewer_grid_<aoi>.npz``
(``scripts/make_viewer_grid_fixture.py``). No cache, network, browser or GPU.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from naigos.demo import ambience, atmosphere, cities, scenario

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CITY_KEYS = sorted(cities.CITIES)


def sampler(aoi: str) -> cities.TerrainSampler:
    f = np.load(FIXTURES / f"viewer_grid_{aoi}.npz")
    return cities.TerrainSampler(f["heights"], json.loads(str(f["meta"])))


def fixture_meta(aoi: str) -> dict:
    return json.loads(str(np.load(FIXTURES / f"viewer_grid_{aoi}.npz")["meta"]))


# --- the registry ------------------------------------------------------------------------


def test_there_is_at_least_one_city():
    assert "tehran_basin" in cities.CITIES


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_every_city_has_its_offline_terrain_fixture(aoi):
    meta = fixture_meta(aoi)
    assert meta["aoi"] == aoi and "Copernicus" in meta["credit"]


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_the_urban_box_lies_inside_the_aoi(aoi):
    c = cities.CITIES[aoi]
    a = c.aoi_def
    w, s, e, n = c.urban_bounds
    assert a.west <= w < e <= a.east and a.south <= s < n <= a.north


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_the_urban_box_avoids_every_extraction_zone(aoi):
    c = cities.CITIES[aoi]
    w, s, e, n = c.urban_bounds
    for z in c.zones("extraction"):
        zw, zs, ze, zn = z.buffered()
        assert ze < w or zw > e or zn < s or zs > n, f"{z.name} overlaps the urban box"


def test_a_malformed_city_is_refused(tmp_path):
    doc = json.loads((cities.CITIES_DIR / "tehran_basin.json").read_text())
    doc["atmosphere_profile"] = "apocalypse"
    p = tmp_path / "tehran_basin.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(atmosphere.AtmosphereError):
        cities.load_city(p)
    doc = json.loads((cities.CITIES_DIR / "tehran_basin.json").read_text())
    doc["camera"]["presets"]["orbit_the_target"] = doc["camera"]["presets"]["street_canyon"]
    p.write_text(json.dumps(doc))
    with pytest.raises(cities.CityConfigError, match="unknown camera preset"):
        cities.load_city(p)


# --- the camera contract -------------------------------------------------------------------


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_every_declared_preset_passes_the_camera_contract(aoi):
    c = cities.CITIES[aoi]
    out = cities.city_presets(c, sampler(aoi))       # raises on any broken rule
    assert set(out) == set(c.presets)
    assert c.default_preset == "urban_overview" or c.default_preset in out


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_the_opening_city_view_is_oblique_and_above_ground(aoi):
    c = cities.CITIES[aoi]
    s = sampler(aoi)
    o, problems = cities.check_preset(c, c.presets["urban_overview"], s)
    assert not problems
    assert -40.0 < o["pitch_deg"] < -8.0, "urban-overview must be oblique, not a map"
    assert o["camera_agl_m"] >= cities.MIN_AGL_M["overview"]
    street, _ = cities.check_preset(c, c.presets["street_canyon"], s)
    assert street["camera_agl_m"] >= cities.MIN_AGL_M["street"]


def test_the_contract_has_teeth():
    """A camera buried in a hill, a camera offshore, and one framing a protected
    zone must each be refused -- or the pass above means nothing."""
    c = cities.CITIES["tehran_basin"]
    s = sampler("tehran_basin")
    spec = c.presets["urban_overview"]
    import dataclasses

    buried = dataclasses.replace(spec, pitch_deg=-2.0, range_m=30_000.0, heading_deg=200.0)
    _, p = cities.check_preset(c, buried, s)
    assert any("above the DEM" in x or "terrain hides" in x for x in p)
    outside = dataclasses.replace(spec, lon=c.aoi_def.east + 0.5)
    _, p = cities.check_preset(c, outside, s)
    assert any("outside the safe region" in x for x in p)


def test_the_framing_rule_catches_a_zone_in_view(monkeypatch):
    """Tehran declares no zones, so plant one straight ahead of its overview."""
    from naigos.research.aoi import ProtectedZone

    c = cities.CITIES["tehran_basin"]
    spec = c.presets["urban_overview"]
    zlon, zlat = cities.offset(spec.lon, spec.lat, spec.heading_deg, 4000.0)
    zone = ProtectedZone("planted", zlon - 0.01, zlat - 0.01, zlon + 0.01, zlat + 0.01, 500.0,
                         ("camera",), "test")
    monkeypatch.setattr(type(c), "zones", lambda self, policy: (zone,) if policy == "camera" else ())
    _, p = cities.check_preset(c, spec, None)
    assert any("inside the view cone" in x for x in p)
    # the same zone BEHIND the camera is fine
    o = cities.orbit(spec, 0.0)
    blon, blat = cities.offset(o["camera_lon"], o["camera_lat"], spec.heading_deg + 180.0, 3000.0)
    behind = ProtectedZone("behind", blon - 0.005, blat - 0.005, blon + 0.005, blat + 0.005, 200.0,
                           ("camera",), "test")
    monkeypatch.setattr(type(c), "zones", lambda self, policy: (behind,) if policy == "camera" else ())
    _, p = cities.check_preset(c, spec, None)
    assert not any("view cone" in x for x in p)


def test_the_follow_heading_turns_away_from_a_zone():
    zones = [{"name": "z", "bbox": [10.00, 20.00, 10.02, 20.02], "policies": ["camera"]}]
    cam = (10.01, 19.95)                               # due south of the zone
    h = cities.safe_follow_heading(*cam, 0.0, zones)   # looking straight at it
    for (lon, lat) in ((10.0, 20.0), (10.02, 20.0), (10.02, 20.02), (10.0, 20.02), (10.01, 20.01)):
        b = cities.bearing_deg(*cam, lon, lat)
        assert cities.angle_diff(b, h) >= cities.VIEW_HALF_ANGLE_DEG - 1e-6
    # a camera already looking away is left alone
    assert cities.safe_follow_heading(*cam, 180.0, zones) == 180.0
    # and a zone out of range does not steer anything
    far = (10.01, 19.0)
    assert cities.safe_follow_heading(*far, 0.0, zones) == 0.0


# --- atmosphere --------------------------------------------------------------------------


def test_atmosphere_profiles_are_an_allowlist_with_caps():
    for p in atmosphere.PROFILES.values():
        assert 0.0 <= p.fog_density <= atmosphere.MAX_FOG_DENSITY
        assert 0.0 <= p.dust_alpha <= atmosphere.MAX_DUST_ALPHA
        assert 0.0 <= p.shimmer <= atmosphere.MAX_SHIMMER
    with pytest.raises(ValueError):
        atmosphere.AtmosphereProfile(**{**atmosphere.PROFILES["neutral"].__dict__,
                                        "key": "x", "fog_density": 0.01})
    with pytest.raises(atmosphere.AtmosphereError):
        atmosphere.get_profile("nuclear_winter")


@pytest.mark.parametrize("aoi", CITY_KEYS)
def test_atmosphere_selection_is_deterministic(aoi):
    c = cities.CITIES[aoi]
    for _ in range(3):
        assert atmosphere.resolve("auto", "urban-presentation", c.atmosphere_profile).key \
            == c.atmosphere_profile
    # physics keeps the neutral, evidence-mode look unless a key is asked for
    assert atmosphere.resolve(None, "physics", c.atmosphere_profile).key == "neutral"
    assert atmosphere.resolve("warm_coastal_desert", "physics", c.atmosphere_profile).key \
        == "warm_coastal_desert"


def test_the_neutral_profile_is_the_existing_physics_light():
    from naigos.demo import camera

    assert atmosphere.PROFILES["neutral"].lighting() == camera.lighting()


# --- ambience ------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=CITY_KEYS)
def city_mask(request):
    c = cities.CITIES[request.param]
    return c, ambience.build_mask(c, seed=7)


def test_every_mask_cell_is_an_allowed_origin(city_mask):
    c, mask = city_mask
    assert mask.cells
    for lon, lat in mask.cells:
        assert ambience.excluded(c, lon, lat) is None


def test_the_mask_never_touches_a_protected_zone_or_the_urban_core(city_mask):
    c, mask = city_mask
    w, s, e, n = c.urban_bounds
    for lon, lat in mask.cells:
        assert not (w <= lon <= e and s <= lat <= n)
        for z in c.zones("ambience"):
            assert not z.contains(lon, lat)


def test_the_stream_is_deterministic_and_bucket_independent(city_mask):
    c, mask = city_mask
    st = ambience.get_setting("sustained")
    a = ambience.generate(c, st, mask, 7, 0.0, 600.0)
    b = ambience.generate(c, st, mask, 7, 0.0, 600.0)
    assert a == b and a, "same seed, same stream"
    halves = (ambience.generate(c, st, mask, 7, 0.0, 300.0)
              + ambience.generate(c, st, mask, 7, 300.0, 600.0))
    # the concurrency cap can only drop, and the two halves see fewer overlaps
    assert {e["id"] for e in a} <= {e["id"] for e in halves}
    other = ambience.generate(c, st, mask, 8, 0.0, 600.0)
    assert [e["id"] for e in other] != [e["id"] for e in a] or other != a


def test_the_stream_respects_its_rate_and_concurrency_bounds(city_mask):
    c, mask = city_mask
    for key, st in ambience.SETTINGS.items():
        assert 0 < st.rate_per_min <= ambience.MAX_RATE_PER_MIN
        evs = ambience.generate(c, st, mask, 3, 0.0, 1800.0)
        per_bucket: dict[int, int] = {}
        for e in evs:
            k = int(e["t_s"] // ambience.BUCKET_S)
            per_bucket[k] = per_bucket.get(k, 0) + 1
            assert e["fictional"] is True and e["kind"] in ambience.KINDS
        assert max(per_bucket.values()) <= st.max_per_bucket
        assert len(evs) <= st.max_per_bucket * (1800.0 / ambience.BUCKET_S)
        # never more than the cap drawn at once
        for e in evs:
            live = [o for o in evs if o["t_s"] <= e["t_s"] < o["t_s"] + o["duration_s"]]
            assert len(live) <= st.max_concurrent
        # mean rate within a loose band of the declared one (30 minutes of draws)
        assert len(evs) / 30.0 <= st.rate_per_min * 1.5


def test_no_effect_starts_near_an_entity(city_mask):
    c, mask = city_mask
    st = ambience.get_setting("sustained")
    free = ambience.generate(c, st, mask, 11, 0.0, 300.0)
    # put an "aircraft" on top of every origin the free stream chose
    blockers = [(e["lon"], e["lat"]) for e in free]
    blocked = ambience.generate(c, st, mask, 11, 0.0, 300.0, positions_at=lambda t: blockers)
    for e in blocked:
        for lon, lat in blockers:
            assert cities.distance_m(e["lon"], e["lat"], lon, lat) >= c.ambience_standoff_m


def test_the_stream_block_carries_its_label_and_provenance():
    c = cities.CITIES["tehran_basin"]
    blk = ambience.stream_block(c, ambience.AmbienceConfig(True, "sparse", 5), 120.0)
    assert blk["profile"] == "conflict_ambience" and blk["seed"] == 5
    assert "FICTIONAL" in blk["note"] and "not simulated" in blk["note"].lower()
    assert blk["mask"]["digest"] and blk["mask"]["source"].startswith("synthetic")
    assert json.loads(json.dumps(blk)) == blk
    assert ambience.AmbienceConfig(False).as_dict()["profile"] is None


def test_live_generation_is_the_same_stream_one_bucket_ahead():
    c = cities.CITIES["tehran_basin"]
    cfg = ambience.AmbienceConfig(True, "sustained", 2)
    live = ambience.LiveAmbience(c, cfg)
    got = []
    for t in np.arange(0.0, 120.0, 2.0):
        got += live.advance(float(t), [])
    replay = ambience.generate(c, live.setting, live.mask, 2, 0.0, 130.0)
    assert [e["id"] for e in got] == [e["id"] for e in replay][:len(got)]


def test_ambience_settings_refuse_an_unbounded_rate():
    with pytest.raises(ValueError):
        ambience.AmbienceSetting("wild", 60.0, 5, 5, (1, 1, 1, 1), (1, 2), (1, 2), (1, 2))
    with pytest.raises(ambience.AmbienceError):
        ambience.get_setting("apocalyptic")


# --- the presentation modules cannot reach the simulation ------------------------------------

PRESENTATION_MODULES = ("demo/ambience.py", "demo/atmosphere.py", "demo/cities.py",
                        "demo/scenario.py")


@pytest.mark.parametrize("rel", PRESENTATION_MODULES)
def test_presentation_modules_import_nothing_from_the_simulation(rel):
    tree = ast.parse((REPO / "naigos" / rel).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    bad = {m for m in mods if m.split(".")[0] in ("jax", "flax", "optax")
           or re.search(r"(^|\.)(env|rl)(\.|$)", m.lstrip("."))}
    assert not bad, f"{rel} reaches into the simulation: {bad}"


#: Words only the presentation layer may say. Alphanumeric edges, as in
#: tests/test_imagery_layers.py, so an identifier like fetch_urban() is caught.
PRESENTATION_WORDS = re.compile(
    r"(?<![A-Za-z0-9])(ambience|urban|extru\w*|overpass|openstreetmap|osm|haze|smoke_column|smoke_plume|shimmer|"
    r"atmosphere_profile|city|cities|conflict_ambience)(?![A-Za-z0-9])", re.IGNORECASE)


@pytest.mark.parametrize("line", ["from naigos.demo.ambience import generate",
                                  "    obs = obs + urban_mask", "  load_osm_buildings()",
                                  "    h = extruded_height(b)", "# add smoke_column to detection"])
def test_the_vocabulary_guard_has_teeth(line):
    assert PRESENTATION_WORDS.search(line)


@pytest.mark.parametrize("package", ["env", "rl"])
def test_the_simulation_never_names_the_presentation_layer(package):
    hits = []
    for path in sorted((REPO / "naigos" / package).rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if PRESENTATION_WORDS.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert not hits, hits


# --- scenario and checkpoint disclosure ------------------------------------------------------


def test_the_shipped_checkpoint_is_zero_shot_everywhere_but_owens_valley():
    ck = scenario.checkpoint_theatre(REPO / "checkpoints" / "theatre_1000.pkl")
    assert ck["trained_on"] == "owens_valley"
    for theatre in CITY_KEYS:
        d = scenario.disclosure(ck, theatre)
        assert d["relation"] == "zero_shot" and "ZERO-SHOT" in d["text"]
    assert scenario.disclosure(ck, "owens_valley")["relation"] == "trained_on_theatre"


def test_an_unknown_checkpoint_is_never_assumed_trained(tmp_path):
    p = tmp_path / "ckpt.pkl"
    p.write_bytes(b"not the shipped one")
    ck = scenario.checkpoint_theatre(p)
    assert ck["trained_on"] is None
    assert scenario.disclosure(ck, "tehran_basin")["relation"] == "unknown"
    (tmp_path / "theatre.json").write_text(json.dumps({"theatre": "tehran_basin"}))
    ck = scenario.checkpoint_theatre(p)
    assert scenario.disclosure(ck, "tehran_basin")["relation"] == "trained_on_theatre"


def test_the_scenario_block_says_notional():
    b = scenario.scenario_block("tehran_basin")
    assert b["label"] == "notional contested-airspace simulation" and b["notional"] is True
    assert b["threat_layout"]["procedural"] is True
    assert "evasive" in b["blue"] and "no weapon" in b["blue"]
