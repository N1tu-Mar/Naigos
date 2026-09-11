"""dubai_urban: a bounded civil urban-coastal envelope, run as a notional simulation.

Offline: the committed component snapshot, `tests/fixtures/dem_dubai_urban.npz`
(the theatre's own DEM, block-averaged) and `tests/fixtures/viewer_grid_dubai_urban.npz`
(the viewer's terrain grid). Checks that need the real cache or the local city
layer skip cleanly without them; everything else runs on a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _theatre_fixtures import offline_theatre, viewer_sampler

from naigos.demo import cities, presentation, scenario
from naigos.research import allowlist, cache
from naigos.research.aoi import get_aoi

AOI = "dubai_urban"
REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "components" / "aoi" / AOI


def comp(cid: str) -> dict:
    return json.loads((SNAP / f"{cid}.json").read_text())


# --- the AOI and its components -----------------------------------------------------------


def test_the_definition_is_a_bounded_civil_envelope():
    a = get_aoi(AOI)
    assert a.scoped and a.dem_source == "copernicus_dem" and a.country == "AE"
    w, h = a.span_km()
    assert 40.0 < w < 70.0 and 25.0 < h < 45.0, "a modest envelope, not an open-ended scrape"
    assert a.exclude_military_airfields
    assert a.scenario == "notional contested-airspace simulation"
    assert "not an airport, port or sensitive-site study" in a.bounds_policy


def test_every_component_is_present_cited_and_carries_the_fingerprint():
    a = get_aoi(AOI)
    for cid in ("env.aoi", "data.terrain_dem", "data.airfields", "data.atmosphere",
                "data.flight_envelope", "model.detection", "research.agent", "demo.imagery"):
        c = comp(cid)
        for field in ("id", "role", "decision", "rationale", "license", "sources", "generated_at"):
            assert c.get(field), f"{cid} missing {field}"
        assert all(s["key"] in allowlist.ALLOWLIST for s in c["sources"]), cid
    env = comp("env.aoi")["parameters"]
    assert env["name"] == AOI and env["fingerprint"] == a.fingerprint
    assert env["bbox_wgs84"] == list(a.bbox)
    assert env["utm_epsg"] == 32640
    assert env["bounds_policy"] == a.bounds_policy and env["scenario"] == a.scenario
    assert [z["name"] for z in env["protected_zones"]] == [z.name for z in a.protected_zones]
    # cache keys embed the fingerprint, so a moved box can never reuse this DEM
    for cid in ("data.terrain_dem", "data.atmosphere"):
        keys = [x["key"] for x in comp(cid)["cached_artifacts"]]
        assert keys and all(a.fingerprint in k for k in keys), (cid, keys)
    assert comp("data.terrain_dem")["cached_artifacts"][0]["url"].startswith(
        "https://copernicus-dem-30m.s3.amazonaws.com")


def test_provenance_matches_the_cache_when_it_is_present():
    m = cache.load_manifest()
    arts = comp("data.terrain_dem")["cached_artifacts"]
    if not all(a["key"] in m for a in arts):
        pytest.skip("research cache not built here; run `naigos-research --aoi dubai_urban`")
    for a in arts:
        assert m[a["key"]]["sha256"] == a["sha256"]


def test_a_second_research_run_uses_the_cache(tmp_path, monkeypatch):
    """Every network-backed fetch for this AOI returns the cached artifact when
    the manifest has it, and never touches the network."""
    from naigos.research.roots import research_roots
    from naigos.research.sources import atmosphere, terrain

    a = get_aoi(AOI)
    import requests

    def no_network(*args, **kw):
        raise AssertionError("a cached rerun made a network call")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(terrain, "BUILDERS", {k: (no_network, u, lbl)
                                              for k, (_, u, lbl) in terrain.BUILDERS.items()})
    with research_roots(cache_dir=tmp_path):
        for key, rel in ((f"dem/{AOI}/30m/{a.fingerprint}", "terrain/x.tif"),
                         (f"open_meteo/profile/{AOI}/{a.fingerprint}", "atmosphere/x.json")):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(b"cached")
            cache.record(key, "copernicus_dem" if key.startswith("dem") else "open_meteo",
                         "https://example.invalid", rel, b"cached")
        assert terrain.fetch_dem(a).path == "terrain/x.tif"
        assert atmosphere.fetch_profile(a).path == "atmosphere/x.json"


def test_only_civil_airfields_are_start_geometry():
    af = comp("data.airfields")
    names = [x["name"] for x in af["parameters"]["airfields"]]
    assert names and all("air base" not in n.lower() and "military" not in n.lower() for n in names)
    assert af["evidence"]["dem_cross_check"]["abs_max_delta_m"] < 20.0
    a = get_aoi(AOI)
    for x in af["parameters"]["airfields"]:
        assert a.west <= x["lon"] <= a.east and a.south <= x["lat"] <= a.north


# --- georeference: non-sensitive public references ----------------------------------------


def test_the_theatre_lands_on_the_gulf_coast():
    s = viewer_sampler(AOI)
    m = s.m
    assert 54.9 < m["west"] < 55.1 and 55.4 < m["east"] < 55.6
    assert 25.0 < m["south"] < 25.1 and 25.3 < m["north"] < 25.4


@pytest.mark.parametrize("name,lon,lat,published_m,tol_m", [
    # open water of the Gulf, well offshore: the sea surface
    ("open sea, north-west of the coast", 55.06, 25.22, 0.0, 6.0),
    # OurAirports' published field elevation of the civil international
    # airport (62 ft), a public aeronautical reference -- not a military site
    ("civil international airport reference point", 55.370992, 25.24979, 18.9, 20.0),
])
def test_dem_elevation_matches_public_references(name, lon, lat, published_m, tol_m):
    """The viewer's own terrain grid against independent published figures: a
    datum, zone or axis error shows up here as tens to hundreds of metres."""
    h = viewer_sampler(AOI)(lon, lat)
    assert abs(h - published_m) < tol_m, f"{name}: DEM {h:.1f} m vs published {published_m} m"


def test_the_weather_model_ground_height_agrees_with_the_dem():
    site = comp("data.atmosphere")["evidence"]["site"]
    h = viewer_sampler(AOI)(site["lon"], site["lat"])
    assert abs(h - site["model_elevation_m"]) < 25.0


# --- the same simulation, unchanged ----------------------------------------------------------


@pytest.fixture(scope="module")
def theatre():
    from naigos.env.theatre_bridge import env_from_theatre

    with offline_theatre(AOI):
        return env_from_theatre(aoi=AOI, n_blue=4, n_threat=10, cell_m=1500.0)


def test_the_env_builds_from_the_normal_path(theatre):
    cfg, hmap, notes = theatre
    assert notes["theatre"] == AOI and hmap.shape == (cfg.terrain.ny, cfg.terrain.nx)
    assert float(hmap.max()) < 400.0, "a flat coastal theatre"


def test_threats_are_the_generic_classes_with_generic_ranges(theatre):
    cfg, _, _ = theatre
    tehran = json.loads((REPO / "components" / "aoi" / "tehran_basin" / "model.detection.json")
                        .read_text())["parameters"]["threat_classes"]
    ours = comp("model.detection")["parameters"]["threat_classes"]
    assert set(ours) == set(tehran), "no new or renamed threat kinds"
    for k in ours:
        assert ours[k]["radar"]["design_range_km"] == tehran[k]["radar"]["design_range_km"]
    assert {k.label for k in cfg.threat_kinds} == set(ours)
    assert comp("model.detection")["invariants"][0].startswith("Blue never acts on a threat")


def test_blue_is_evasive_only(theatre):
    from naigos.env.config import BLUE_ACTION_NAMES

    cfg, _, _ = theatre
    assert cfg.action_dim == 3 and BLUE_ACTION_NAMES == ("bank_cmd", "gamma_cmd", "throttle_cmd")


def test_layouts_are_procedural_per_seed_and_reroll(theatre):
    from naigos.env.flight_env import NaigosEnv
    from naigos.env.terrain import sample_height

    cfg, hmap, _ = theatre
    env = NaigosEnv(cfg, hmap=hmap)
    s0, _ = env.reset(jax.random.PRNGKey(0))
    s1, _ = env.reset(jax.random.PRNGKey(1))
    assert not np.allclose(np.asarray(s0.threats.pos), np.asarray(s1.threats.pos))
    s2 = env.reroll_threats(s0, jax.random.PRNGKey(5))
    assert not np.allclose(np.asarray(s0.threats.pos), np.asarray(s2.threats.pos))
    g = sample_height(hmap, cfg.terrain, s0.air.pos[:, 0], s0.air.pos[:, 1])
    assert bool(jnp.all(s0.air.pos[:, 2] > g))
    st, o, *_ = env.step(s0, jnp.zeros((cfg.n_blue, 3)))
    assert bool(jnp.all(jnp.isfinite(st.air.pos)))


# --- presentation ------------------------------------------------------------------------------


def test_the_city_is_presented_under_the_standard():
    c = cities.get_city(AOI)
    assert c.atmosphere_profile == "warm_coastal_desert"
    assert {"urban_overview", "coastal_corridor", "street_canyon"} <= set(c.presets)
    assert c.exclude_religious_buildings and c.max_building_height_m >= 830.0
    out = cities.city_presets(c, viewer_sampler(AOI))
    s = viewer_sampler(AOI)
    for key, o in out.items():
        # never below ground, never offshore: the ground under the camera is land
        assert s(o["camera_lon"], o["camera_lat"]) > 0.5, f"{key} camera over the sea"
        assert o["camera_agl_m"] >= cities.MIN_AGL_M[cities.PRESET_KIND[key]]


def test_the_street_camera_clears_the_buildings_around_it():
    from naigos.demo import urban

    st = urban.load(AOI)
    if not st.available:
        pytest.skip("local city layer not built here; run `python -m naigos.demo.urban --aoi dubai_urban`")
    c = cities.get_city(AOI)
    o, _ = cities.check_preset(c, c.presets["street_canyon"], viewer_sampler(AOI))
    p = st.payload
    for ch in p["chunks"]:
        for rec in ch["b"]:
            pts = urban.decode_coords(rec[1:], tuple(p["origin"]))
            lon = sum(q[0] for q in pts) / len(pts)
            lat = sum(q[1] for q in pts) / len(pts)
            if cities.distance_m(lon, lat, o["camera_lon"], o["camera_lat"]) < 400.0:
                assert rec[0] / 10.0 + 30.0 < o["camera_agl_m"]


def test_the_city_layer_is_bounded_attributed_and_tag_free():
    from naigos.demo import urban

    st = urban.load(AOI)
    if not st.available:
        pytest.skip("local city layer not built here")
    p = st.payload
    w, s_, e, n = cities.get_city(AOI).urban_bounds
    assert p["counts"]["buildings"] > 20_000
    assert p["exclusions"]["places_of_worship_excluded"] is True
    assert "OpenStreetMap" in p["source"]["attribution"] and p["presentation_only"]
    # only geometry reaches the browser: chunks hold integer records and nothing else
    for ch in p["chunks"]:
        assert set(ch) == {"k", "c", "b", "r"}
        assert all(isinstance(v, int) for rec in ch["b"] + ch["r"] for v in rec)
    assert not ({"tags", "elements", "osm3s"} & set(p)), "raw Overpass structure in the payload"
    for ch in p["chunks"][:40]:
        for rec in ch["b"]:
            for lon, lat in urban.decode_coords(rec[1:], tuple(p["origin"])):
                assert w - 1e-6 <= lon <= e + 1e-6 and s_ - 1e-6 <= lat <= n + 1e-6


def test_the_scene_says_notional_and_zero_shot():
    ck = scenario.checkpoint_theatre(REPO / "checkpoints" / "theatre_1000.pkl")
    p = presentation.resolve(AOI, "urban-presentation", ambience="conflict_ambience",
                             visual_seed=3, checkpoint=ck)
    f = p.scene_fields()
    assert f["scenario"]["label"] == "notional contested-airspace simulation"
    assert f["scenario"]["checkpoint"]["relation"] == "zero_shot"
    assert "ZERO-SHOT on dubai_urban" in f["scenario"]["checkpoint"]["text"]
    assert f["atmosphere"]["key"] == "warm_coastal_desert" and f["ambience"]["enabled"]


def test_ambience_keeps_clear_of_the_civil_infrastructure_zones():
    from naigos.demo import ambience

    c = cities.get_city(AOI)
    mask = ambience.build_mask(c, 3)
    evs = ambience.generate(c, ambience.get_setting("sustained"), mask, 3, 0.0, 1200.0)
    assert evs
    for e in evs:
        for z in c.zones("ambience"):
            w, s_, e2, n = z.buffered()
            assert not (w - 0.02 <= e["lon"] <= e2 + 0.02 and s_ - 0.02 <= e["lat"] <= n + 0.02)
