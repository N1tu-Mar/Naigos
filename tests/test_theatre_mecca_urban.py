"""mecca_urban: an outer-urban and relief envelope that stays out of the historic centre.

The theatre's first obligation is what it leaves out. The Masjid al-Haram
complex and its immediate precinct lie outside the AOI by construction, and
they and every nearby place of religious significance are protected zones kept
out of visual-data extraction, camera framing, presentation effects and
rendering. These tests check each of those separately, offline, from the
committed snapshot and fixtures; the ones that need the local city layer skip
cleanly without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _theatre_fixtures import offline_theatre, viewer_sampler

from naigos.demo import ambience, cities, presentation, scenario
from naigos.research import allowlist, cache
from naigos.research.aoi import get_aoi

AOI = "mecca_urban"
REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "components" / "aoi" / AOI
PRECINCT = "Masjid al-Haram complex and immediate precinct"


def comp(cid: str) -> dict:
    return json.loads((SNAP / f"{cid}.json").read_text())


def precinct():
    return next(z for z in get_aoi(AOI).protected_zones if z.name == PRECINCT)


def km_from_box(lon, lat, box) -> float:
    """Distance from a point to the nearest point of a (w, s, e, n) box, km."""
    w, s, e, n = box
    clon, clat = min(max(lon, w), e), min(max(lat, s), n)
    return cities.distance_m(lon, lat, clon, clat) / 1000.0


# --- the exclusion, by construction ----------------------------------------------------------


def test_the_precinct_is_outside_the_aoi_by_more_than_three_km():
    a = get_aoi(AOI)
    z = precinct()
    assert a.east < z.west, "the AOI must lie wholly west of the precinct"
    gap = cities.distance_m(a.east, (z.south + z.north) / 2, z.west, (z.south + z.north) / 2)
    assert gap > 3000.0
    # every AOI corner is well clear of the precinct too
    for lon in (a.west, a.east):
        for lat in (a.south, a.north):
            assert km_from_box(lon, lat, (z.west, z.south, z.east, z.north)) > 3.0


def test_every_protected_zone_enforces_every_policy_and_lies_outside_the_box():
    a = get_aoi(AOI)
    names = {z.name for z in a.protected_zones}
    assert PRECINCT in names and any("pilgrimage" in n for n in names)
    for z in a.protected_zones:
        assert set(z.policies) == {"extraction", "airfield", "camera", "ambience", "render_cutout"}
        assert z.west > a.east or z.east < a.west or z.south > a.north or z.north < a.south, z.name


def test_the_component_records_the_bounds_and_exclusion_policy():
    env = comp("env.aoi")["parameters"]
    a = get_aoi(AOI)
    assert env["fingerprint"] == a.fingerprint and env["bbox_wgs84"] == list(a.bbox)
    assert env["utm_epsg"] == 32637
    assert "Masjid al-Haram" in env["bounds_policy"]
    rec = {z["name"]: z for z in env["protected_zones"]}
    assert rec[PRECINCT]["bbox_wgs84"] == [precinct().west, precinct().south,
                                           precinct().east, precinct().north]
    for cid in ("env.aoi", "data.terrain_dem", "data.airfields", "data.atmosphere",
                "data.flight_envelope", "model.detection", "research.agent", "demo.imagery"):
        c = comp(cid)
        assert c["sources"] and all(s["key"] in allowlist.ALLOWLIST for s in c["sources"])
    for cid in ("data.terrain_dem", "data.atmosphere"):
        assert all(a.fingerprint in x["key"] for x in comp(cid)["cached_artifacts"])


def test_no_airfield_and_no_anchor_inside_the_box():
    af = comp("data.airfields")
    assert af["parameters"]["n_airfields"] == 0
    assert af["evidence"]["excluded_by_policy"] == {}


def test_provenance_matches_the_cache_when_it_is_present():
    m = cache.load_manifest()
    arts = comp("data.terrain_dem")["cached_artifacts"]
    if not all(a["key"] in m for a in arts):
        pytest.skip("research cache not built here; run `naigos-research --aoi mecca_urban`")
    for a in arts:
        assert m[a["key"]]["sha256"] == a["sha256"]


def test_a_second_research_run_uses_the_cache(tmp_path, monkeypatch):
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


# --- georeference ------------------------------------------------------------------------


def test_the_theatre_lands_in_the_hijaz_foothills():
    m = viewer_sampler(AOI).m
    assert 39.4 < m["west"] < 39.5 and 39.7 < m["east"] < 39.8
    assert 21.2 < m["south"] < 21.3 and 21.5 < m["north"] < 21.6
    assert 50.0 < m["min_m"] < 150.0 and 500.0 < m["max_m"] < 900.0


def test_the_weather_model_ground_height_agrees_with_the_dem():
    """Open-Meteo's own terrain height at its grid point -- a public reference
    produced independently of this pipeline's reprojection and resampling. A
    wrong zone or datum shows up as hundreds of metres in this relief."""
    site = comp("data.atmosphere")["evidence"]["site"]
    h = viewer_sampler(AOI)(site["lon"], site["lat"])
    assert abs(h - site["model_elevation_m"]) < 40.0


def test_a_wrong_utm_zone_would_be_caught():
    from naigos.data.geodetic import GeoRef

    meta = viewer_sampler(AOI).m
    g = GeoRef(**meta["georef"])
    bad = GeoRef(**{**meta["georef"], "utm_epsg": meta["georef"]["utm_epsg"] + 1})
    assert abs(float(g.to_wgs84(0.0, 0.0)[0]) - float(bad.to_wgs84(0.0, 0.0)[0])) > 1.0


# --- the same simulation, generic and notional -------------------------------------------------


@pytest.fixture(scope="module")
def theatre():
    from naigos.env.theatre_bridge import env_from_theatre

    with offline_theatre(AOI):
        return env_from_theatre(aoi=AOI, n_blue=4, n_threat=10, cell_m=1500.0)


def test_the_env_builds_from_the_normal_path(theatre):
    cfg, hmap, notes = theatre
    assert notes["theatre"] == AOI
    assert 400.0 < float(hmap.max() - hmap.min()) < 900.0


def test_threats_are_the_generic_classes_and_blue_is_evasive_only(theatre):
    from naigos.env.config import BLUE_ACTION_NAMES

    cfg, _, _ = theatre
    tehran = json.loads((REPO / "components" / "aoi" / "tehran_basin" / "model.detection.json")
                        .read_text())["parameters"]["threat_classes"]
    ours = comp("model.detection")["parameters"]["threat_classes"]
    assert set(ours) == set(tehran)
    for k in ours:
        assert ours[k]["radar"]["design_range_km"] == tehran[k]["radar"]["design_range_km"]
    assert cfg.action_dim == 3 and BLUE_ACTION_NAMES == ("bank_cmd", "gamma_cmd", "throttle_cmd")


def test_layouts_are_procedural_and_stay_inside_the_box(theatre):
    """Entities are placed in the ENU grid of the AOI, and the AOI excludes the
    precinct -- so no draw, on any seed, can put anything there."""
    from naigos.data.geodetic import GeoRef
    from naigos.env.flight_env import NaigosEnv

    cfg, hmap, notes = theatre
    env = NaigosEnv(cfg, hmap=hmap)
    g = GeoRef(**notes["georef"])
    z = precinct()
    seen = []
    for seed in range(6):
        st, _ = env.reset(jax.random.PRNGKey(seed))
        st = env.reroll_threats(st, jax.random.PRNGKey(100 + seed))
        for arr in (np.asarray(st.threats.pos), np.asarray(st.air.pos), np.asarray(st.objective)):
            lon, lat = g.to_wgs84(arr[:, 0], arr[:, 1])
            for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat)):
                assert km_from_box(float(a), float(b), z.buffered()) > 1.0
        seen.append(np.asarray(st.threats.pos).copy())
    assert not np.allclose(seen[0], seen[1])


# --- presentation: cameras, city layer, ambience -----------------------------------------------


def test_every_camera_preset_opens_above_ground_and_looks_away_from_the_precinct():
    c = cities.get_city(AOI)
    assert {"urban_overview", "valley_overview", "street_canyon"} <= set(c.presets)
    out = cities.city_presets(c, viewer_sampler(AOI))   # the full contract
    z = precinct()
    for key, o in out.items():
        assert km_from_box(o["camera_lon"], o["camera_lat"], z.buffered()) > 3.0, key
        assert km_from_box(o["lon"], o["lat"], z.buffered()) > 3.0, key
        b = cities.bearing_deg(o["camera_lon"], o["camera_lat"],
                               (z.west + z.east) / 2, (z.south + z.north) / 2)
        assert cities.angle_diff(b, o["heading_deg"]) > 60.0, f"{key} faces the precinct"


def test_the_page_is_handed_every_zone_to_cut_out_and_guard():
    from naigos.demo import camera

    s = viewer_sampler(AOI)
    p = camera.presets(s.m, s.m, city=cities.get_city(AOI), sample=s)
    zones = {z["name"]: z for z in p["protected"]}
    assert PRECINCT in zones
    assert set(zones[PRECINCT]["policies"]) == {"camera", "render_cutout"}
    # the chase camera turns away from it
    zb = zones[PRECINCT]["bbox"]
    cam_lon, cam_lat = zb[0] - 0.05, (zb[1] + zb[3]) / 2           # ~5 km west, facing east
    h = cities.safe_follow_heading(cam_lon, cam_lat, 90.0, p["protected"])
    assert cities.angle_diff(h, 90.0) >= cities.VIEW_HALF_ANGLE_DEG


def test_the_city_layer_holds_nothing_from_any_protected_zone():
    from naigos.demo import urban

    b = urban.URBAN_BOUNDS[AOI]
    assert b.exclude_religious and b.include_minor_roads and len(b.exclusions) == 4
    q = urban.overpass_query(b)
    assert "place_of_worship" in q and "mosque" in q
    st = urban.load(AOI)
    if not st.available:
        pytest.skip("local city layer not built here; run `python -m naigos.demo.urban --aoi mecca_urban`")
    p = st.payload
    boxes = [z.buffered() for z in get_aoi(AOI).protected_zones]
    for ch in p["chunks"]:
        for rec in ch["b"] + ch["r"]:
            for lon, lat in urban.decode_coords(rec[1:], tuple(p["origin"])):
                for (w, s, e, n) in boxes:
                    assert not (w <= lon <= e and s <= lat <= n)
                assert km_from_box(lon, lat, precinct().buffered()) > 1.0
    assert p["exclusions"]["places_of_worship_excluded"] is True
    assert "OpenStreetMap" in p["source"]["attribution"]
    for ch in p["chunks"]:
        assert set(ch) == {"k", "c", "b", "r"}
        assert all(isinstance(v, int) for rec in ch["b"] + ch["r"] for v in rec)


def test_ambience_origins_are_far_from_every_protected_zone():
    c = cities.get_city(AOI)
    mask = ambience.build_mask(c, 11)
    z = precinct()
    assert min(km_from_box(lon, lat, (z.west, z.south, z.east, z.north)) for lon, lat in mask.cells) > 8.0
    evs = ambience.generate(c, ambience.get_setting("sustained"), mask, 11, 0.0, 1800.0)
    assert evs
    for e in evs:
        for zone in c.zones("ambience"):
            assert km_from_box(e["lon"], e["lat"], zone.buffered()) > 1.4


def test_the_scene_says_notional_and_zero_shot_and_uses_the_restrained_profile():
    ck = scenario.checkpoint_theatre(REPO / "checkpoints" / "theatre_1000.pkl")
    p = presentation.resolve(AOI, "urban-presentation", ambience="conflict_ambience",
                             visual_seed=2, checkpoint=ck)
    f = p.scene_fields()
    assert f["scenario"]["label"] == "notional contested-airspace simulation"
    assert f["scenario"]["checkpoint"]["relation"] == "zero_shot"
    assert f["atmosphere"]["key"] == "hot_dusty_inland"
    with pytest.raises(presentation.PresentationError):
        presentation.resolve(AOI, "physics", ambience="conflict_ambience")
