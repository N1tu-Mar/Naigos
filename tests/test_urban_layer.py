"""`--visual urban-presentation`: the local city layer and the provider proof, offline.

No network, no browser, no GPU, no credentials, no Modal. What is under test:

* the mode contract -- physics and photorealistic unchanged, the third mode
  presentation-only and honest about which geometry it has;
* the OSM source -- allowlisted, bounded to a documented box inside the AOI,
  cached with provenance, idempotent after the first fetch;
* the derivation -- deterministic heights (height tag, then levels, then the
  documented fallback), bad geometry rejected, no raw tags reaching the page;
* the page -- the pure helpers (run under node when it is installed) never
  report provider buildings before a mocked tile is loaded AND on screen, the
  loader respects its cap, and a static export carries the layer inside it;
* isolation -- nothing under naigos/env or naigos/rl can reach any of it.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from naigos.demo import camera, imagery, live, urban
from naigos.research import allowlist
from naigos.research.aoi import AOIS

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
TEHRAN = urban.URBAN_BOUNDS["tehran_basin"]
TOKEN = "fake-ion-token.not-a-real-payload.0123456789abcdef"
GOOGLE_KEY = "AIzaSy-not-a-real-google-key"


# --- fixtures ----------------------------------------------------------------------------


def _square(lon, lat, d=0.0002):
    """A closed ~18 x 22 m square way geometry in Overpass `out geom` shape."""
    pts = [(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d), (lon, lat)]
    return [{"lon": x, "lat": y} for x, y in pts]


def overpass_fixture() -> dict:
    """A small Overpass response: good, bad and out-of-scope elements together."""
    lon, lat = 51.40, 35.72
    els = [
        # explicit height wins over levels
        {"type": "way", "id": 1, "tags": {"building": "yes", "height": "42", "building:levels": "3",
                                          "name": "Some Tower", "addr:street": "X"},
         "geometry": _square(lon, lat)},
        # levels when there is no valid height
        {"type": "way", "id": 2, "tags": {"building": "apartments", "building:levels": "5",
                                          "height": "tall"},
         "geometry": _square(lon + 0.001, lat)},
        # fallback by tag
        {"type": "way", "id": 3, "tags": {"building": "house"}, "geometry": _square(lon + 0.002, lat)},
        # fallback by area ("yes")
        {"type": "way", "id": 4, "tags": {"building": "yes"}, "geometry": _square(lon + 0.003, lat)},
        # rejected: self-intersecting bow tie
        {"type": "way", "id": 5, "tags": {"building": "yes"}, "geometry": [
            {"lon": lon, "lat": lat + 0.002}, {"lon": lon + 0.0003, "lat": lat + 0.0023},
            {"lon": lon + 0.0003, "lat": lat + 0.002}, {"lon": lon, "lat": lat + 0.0023},
            {"lon": lon, "lat": lat + 0.002}]},
        # rejected: open ring
        {"type": "way", "id": 6, "tags": {"building": "yes"}, "geometry": _square(lon, lat + 0.004)[:-1]},
        # rejected: outside the documented box
        {"type": "way", "id": 7, "tags": {"building": "yes"}, "geometry": _square(51.60, 35.72)},
        # rejected: malformed vertex
        {"type": "way", "id": 8, "tags": {"building": "yes"},
         "geometry": [{"lon": "x", "lat": lat}] * 4},
        # rejected even if the query had let it through
        {"type": "way", "id": 9, "tags": {"building": "yes", "military": "barracks"},
         "geometry": _square(lon + 0.005, lat)},
        # a road crossing the box edge: clipped to the in-bounds run
        {"type": "way", "id": 10, "tags": {"highway": "primary", "name": "Some Avenue"},
         "geometry": [{"lon": 51.49, "lat": 35.70}, {"lon": 51.495, "lat": 35.70},
                      {"lon": 51.52, "lat": 35.70}]},
        # a node: ignored
        {"type": "node", "id": 11, "lat": lat, "lon": lon, "tags": {"amenity": "cafe"}},
    ]
    return {"version": 0.6, "osm3s": {"timestamp_osm_base": "2026-09-01T00:00:00Z"}, "elements": els}


SOURCE = {"key": urban.SOURCE_KEY, "attribution": "(c) OpenStreetMap contributors (ODbL)",
          "licence": "ODbL", "raw_sha256": "0" * 64}


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """A throwaway research cache root, so nothing touches data_cache/."""
    from naigos.research import cache as research_cache

    previous = research_cache.set_cache_dir(tmp_path)
    yield tmp_path
    research_cache.set_cache_dir(previous)


def _fake_fetcher(calls):
    def fetch(query):
        calls.append(query)
        return json.dumps(overpass_fixture()).encode()
    return fetch


# --- the mode contract -------------------------------------------------------------------


def test_physics_and_photorealistic_are_unchanged_by_the_new_mode():
    phys = imagery.resolve_visual_config("physics", ion_token=TOKEN)
    assert (phys.mode, phys.base_imagery, phys.evidence_grade) == ("physics", "sentinel2", True)
    assert phys.geometry_source is None and phys.presentation_only is False
    assert phys.separation_note == imagery.LAYER_SEPARATION_NOTE
    photo = imagery.resolve_visual_config("photorealistic", ion_token=TOKEN)
    assert photo.mode == "photorealistic" and photo.tileset_route == "cesium_ion"
    assert photo.separation_note == imagery.PHOTOREALISTIC_EVIDENCE_WARNING
    assert photo.geometry_source == imagery.GEOMETRY_PROVIDER
    # local urban data changes nothing for either
    assert imagery.resolve_visual_config("physics", local_urban=True) == \
        imagery.resolve_visual_config("physics")
    assert imagery.resolve_visual_config("photorealistic", local_urban=True).mode == "physics"
    assert imagery.DEFAULT_VISUAL_MODE == "physics"


def test_urban_presentation_is_accepted_and_says_it_is_presentation_only():
    assert "urban-presentation" in imagery.VISUAL_MODES
    cfg = imagery.resolve_visual_config("urban-presentation", local_urban=True)
    d = cfg.as_dict()
    assert d["mode"] == "urban-presentation" and d["presentation_only"] is True
    assert d["evidence_grade"] is False
    assert d["geometry_source"] == "local_osm_extrusions"
    assert d["requested_geometry_source"] == "local_osm_extrusions"
    assert d["urban_geometry"] == imagery.DEFAULT_URBAN_GEOMETRY == "local"
    assert d["provider_state"] == imagery.PROVIDER_NOT_REQUESTED
    assert d["building_occlusion_default"] is False
    assert d["terrain_source"] == imagery.TERRAIN_SOURCE_SIMULATION
    assert "not used by terrain LOS" in d["building_note"]
    assert "OpenStreetMap" in d["attribution"] and "ODbL" in d["attribution"]
    assert d["fallback_reason"] is None, "local was asked for and local is drawn: nothing fell back"


# --- geometry routing: --urban-geometry local|provider|auto --------------------------------


@pytest.mark.parametrize("creds", [{"ion_token": TOKEN}, {"google_api_key": GOOGLE_KEY},
                                   {"ion_token": TOKEN, "google_api_key": GOOGLE_KEY}])
@pytest.mark.parametrize("policy", [None, "local"])
def test_a_credential_plus_a_local_cache_defaults_to_local_geometry(creds, policy):
    cfg = imagery.resolve_visual_config("urban-presentation", local_urban=True,
                                        urban_geometry=policy, **creds)
    assert cfg.urban_geometry == "local"
    assert cfg.geometry_source == cfg.requested_geometry_source == imagery.GEOMETRY_LOCAL
    # No tileset means the page never enters its provider (PHOTO) path, so the
    # globe stays on and the local city layer is what it draws.
    assert cfg.tileset is None and cfg.tileset_route is None and cfg.tileset_ion_asset is None
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION
    assert cfg.provider_state == imagery.PROVIDER_NOT_REQUESTED
    assert cfg.fallback_reason is None
    assert cfg.tileset_attribution is None and "Google" not in cfg.attribution
    assert cfg.separation_note == imagery.URBAN_LOCAL_WARNING
    # the credential is still truthfully reported as present -- as a boolean
    assert cfg.ion_token_present == ("ion_token" in creds)
    assert cfg.google_api_key_present == ("google_api_key" in creds)
    line = imagery.describe(cfg)
    assert "local cached OpenStreetMap buildings" in line and "--urban-geometry provider" in line


def test_the_provider_is_selected_only_when_explicitly_requested():
    # the credential alone: local, under the default and under an explicit local
    for policy in (None, "local"):
        assert imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN, local_urban=True,
                                             urban_geometry=policy).geometry_source == "local_osm_extrusions"
    # the opt-in: provider, by either route
    cfg = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN, local_urban=True,
                                        urban_geometry="provider")
    assert cfg.urban_geometry == "provider"
    assert cfg.geometry_source == cfg.requested_geometry_source == "provider_3d_tiles"
    assert cfg.tileset == "google_photorealistic" and cfg.tileset_route == "cesium_ion"
    assert cfg.fallback_reason is None
    via_google = imagery.resolve_visual_config("urban-presentation", google_api_key=GOOGLE_KEY,
                                               urban_geometry="provider")
    assert via_google.tileset_route == "google_maps_api"
    # auto keeps the mode's original provider-when-reachable behaviour
    auto = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN, local_urban=True,
                                         urban_geometry="auto")
    assert auto.geometry_source == "provider_3d_tiles" and auto.tileset_route == "cesium_ion"
    auto_bare = imagery.resolve_visual_config("urban-presentation", local_urban=True,
                                              urban_geometry="auto")
    assert auto_bare.geometry_source == auto_bare.requested_geometry_source == "local_osm_extrusions"
    assert auto_bare.tileset is None
    # the policy is validated, and refused outside the mode it applies to
    with pytest.raises(imagery.VisualConfigError):
        imagery.resolve_visual_config("urban-presentation", urban_geometry="google")
    with pytest.raises(imagery.VisualConfigError, match="only applies to --visual urban-presentation"):
        imagery.validate_cli("physics", None, "provider")
    for policy in (None, *imagery.URBAN_GEOMETRY_CHOICES):
        imagery.validate_cli("urban-presentation", None, policy)
    imagery.validate_cli("physics", None, None)
    # other modes ignore the knob entirely
    assert imagery.resolve_visual_config("physics", ion_token=TOKEN, urban_geometry="provider") == \
        imagery.resolve_visual_config("physics", ion_token=TOKEN)


def test_the_provider_path_is_never_active_server_side():
    cfg = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN, local_urban=True,
                                        urban_geometry="provider")
    # The server can only say a route exists; "active" is a browser observation.
    assert cfg.provider_state == imagery.PROVIDER_AWAITING_BROWSER
    assert "active" not in cfg.provider_state
    # Google's attribution and the non-evidence warning are preserved.
    assert "Google" in cfg.attribution and "not used by terrain LOS" in cfg.separation_note
    assert imagery.PHOTOREALISTIC_EVIDENCE_WARNING in cfg.separation_note
    # the local layer still rides along, credited, as the runtime fallback
    assert cfg.local_urban_state == "available" and cfg.geometry_attribution
    # the Google key reaches the page only on the direct route
    via_google = imagery.resolve_visual_config("urban-presentation", google_api_key=GOOGLE_KEY,
                                               urban_geometry="provider")
    assert GOOGLE_KEY in live.render_page(via_google, None, GOOGLE_KEY)
    assert GOOGLE_KEY not in live.render_page(cfg, TOKEN, GOOGLE_KEY)


@pytest.mark.parametrize("policy,creds,local,source,requested,provider,reason", [
    # the default
    ("local", {}, True, "local_osm_extrusions", "local_osm_extrusions", "not_requested", None),
    ("local", {"ion_token": TOKEN}, True, "local_osm_extrusions", "local_osm_extrusions",
     "not_requested", None),
    ("local", {}, False, None, "local_osm_extrusions", "not_requested", "no local urban cache"),
    ("local", {"ion_token": TOKEN}, False, None, "local_osm_extrusions", "not_requested",
     "--urban-geometry provider"),
    # the opt-in
    ("provider", {"ion_token": TOKEN}, False, "provider_3d_tiles", "provider_3d_tiles",
     "awaiting_browser", None),
    ("provider", {}, True, "local_osm_extrusions", "provider_3d_tiles",
     "unavailable_no_credentials", "--urban-geometry provider needs"),
    ("provider", {}, False, None, "provider_3d_tiles", "unavailable_no_credentials",
     "no local urban cache"),
    # the original behaviour
    ("auto", {"google_api_key": GOOGLE_KEY}, True, "provider_3d_tiles", "provider_3d_tiles",
     "awaiting_browser", None),
    ("auto", {}, True, "local_osm_extrusions", "local_osm_extrusions",
     "unavailable_no_credentials", "provider buildings need"),
    ("auto", {}, False, None, "local_osm_extrusions", "unavailable_no_credentials",
     "no local urban cache"),
])
def test_geometry_source_and_fallback_status_are_reported(policy, creds, local, source, requested,
                                                          provider, reason):
    cfg = imagery.resolve_visual_config("urban-presentation", local_urban=local,
                                        urban_geometry=policy, **creds)
    assert (cfg.geometry_source, cfg.requested_geometry_source, cfg.provider_state) == \
        (source, requested, provider)
    assert cfg.local_urban_state == ("available" if local else "unavailable")
    assert cfg.urban_geometry == policy
    if reason is None:
        assert cfg.fallback_reason is None
    else:
        assert reason in cfg.fallback_reason
    # the page's PHOTO switch is `!!VISUAL.tileset`: a tileset exactly when the
    # provider is the source, so the page draws what the config says
    assert (cfg.tileset is not None) == (source == "provider_3d_tiles")
    if source is None:
        # nothing to draw: say so, with the command that fixes it
        assert "urban data unavailable" in cfg.fallback_reason
        assert imagery.URBAN_BUILD_HINT in cfg.fallback_reason
        assert cfg.separation_note == imagery.URBAN_UNAVAILABLE_WARNING
        assert "URBAN DATA UNAVAILABLE" in imagery.describe(cfg)
    # smoke-report parity: the same fields, straight through
    d = cfg.as_dict()
    assert (d["geometry_source"], d["requested_geometry_source"], d["urban_geometry"]) == \
        (source, requested, policy)


def test_missing_data_and_credentials_is_a_labelled_state_not_a_pretend_layer():
    cfg = imagery.resolve_visual_config("urban-presentation")
    assert cfg.mode == "urban-presentation"
    assert cfg.geometry_source is None, "no data, no credential: nothing may claim buildings"
    assert cfg.local_urban_state == "unavailable"
    assert "urban data unavailable" in cfg.fallback_reason
    assert "naigos.demo.urban" in cfg.fallback_reason
    assert cfg.separation_note == imagery.URBAN_UNAVAILABLE_WARNING
    assert cfg.geometry_attribution is None
    # a credential does not paper over a missing cache under the default
    tok = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN)
    assert tok.geometry_source is None and tok.tileset is None
    assert "naigos.demo.urban" in tok.fallback_reason


@pytest.mark.parametrize("policy", [None, *imagery.URBAN_GEOMETRY_CHOICES])
@pytest.mark.parametrize("kw", [{}, {"ion_token": TOKEN}, {"google_api_key": GOOGLE_KEY},
                                {"ion_token": TOKEN, "google_api_key": GOOGLE_KEY, "local_urban": True}])
def test_no_credential_survives_into_the_urban_config(kw, policy):
    cfg = imagery.resolve_visual_config("urban-presentation", urban_geometry=policy, **kw)
    for blob in (json.dumps(cfg.as_dict()), json.dumps(cfg.to_page()), repr(cfg), imagery.describe(cfg)):
        assert TOKEN not in blob and GOOGLE_KEY not in blob and "AIzaSy" not in blob


@pytest.mark.parametrize("policy", [None, "local"])
def test_a_local_urban_page_carries_no_credential_at_all(policy):
    # The page on the local path talks to neither Google nor Cesium ion, so it
    # holds neither secret -- not even through the two substitution points.
    cfg = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN,
                                        google_api_key=GOOGLE_KEY, local_urban=True,
                                        urban_geometry=policy)
    html = live.render_page(cfg, TOKEN, GOOGLE_KEY)
    assert TOKEN not in html and GOOGLE_KEY not in html
    assert "const ION_TOKEN = null;" in html and "const GOOGLE_API_KEY = null;" in html
    # the provider opt-in over ion still gets the token it needs, and only that
    prov = imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN,
                                         google_api_key=GOOGLE_KEY, local_urban=True,
                                         urban_geometry="provider")
    html = live.render_page(prov, TOKEN, GOOGLE_KEY)
    assert TOKEN in html and GOOGLE_KEY not in html
    # physics is untouched: its Sentinel-2 skin still needs the ion token
    phys = imagery.resolve_visual_config("physics", ion_token=TOKEN)
    assert TOKEN in live.render_page(phys, TOKEN, None)


def test_sentinel2_is_refused_under_urban_presentation():
    with pytest.raises(imagery.VisualConfigError):
        imagery.validate_cli("urban-presentation", "sentinel2")
    imagery.validate_cli("urban-presentation", None)
    imagery.validate_cli("urban-presentation", "osm")


def test_the_cli_accepts_the_mode_and_the_city_cameras(capsys):
    with pytest.raises(SystemExit) as e:
        live.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "urban-presentation" in out and "urban-overview" in out and "street-canyon" in out


# --- the source --------------------------------------------------------------------------


def test_the_osm_source_is_allowlisted_visual_only_with_its_licence():
    src = allowlist.ALLOWLIST[urban.SOURCE_KEY]
    assert src.hosts == ("overpass-api.de",)
    assert "ODbL" in src.license or "Open Database License" in src.license
    assert src.attribution_required and "OpenStreetMap contributors" in src.citation
    assert "PRESENTATION ONLY" in src.role and "never read by the simulation" in src.role
    assert allowlist.check_url(urban.OVERPASS_URL, urban.SOURCE_KEY).key == urban.SOURCE_KEY
    with pytest.raises(allowlist.SourceNotAllowed):
        allowlist.check_url("https://overpass.kumi.systems/api/interpreter", urban.SOURCE_KEY)
    with pytest.raises(allowlist.SourceNotAllowed):
        allowlist.check_url("http://overpass-api.de/api/interpreter", urban.SOURCE_KEY)


def test_the_urban_box_is_inside_the_aoi_and_the_query_is_bounded_to_it():
    aoi = AOIS["tehran_basin"]
    assert aoi.west <= TEHRAN.west < TEHRAN.east <= aoi.east
    assert aoi.south <= TEHRAN.south < TEHRAN.north <= aoi.north
    q = urban.overpass_query(TEHRAN)
    # every bbox in the query is the documented box, and nothing is unbounded
    boxes = re.findall(r"\(([-\d.]+,[-\d.]+,[-\d.]+,[-\d.]+)\)", q)
    assert boxes and set(boxes) == {TEHRAN.overpass_bbox}
    assert f"[timeout:{urban.QUERY_TIMEOUT_S}]" in q and f"[maxsize:{urban.QUERY_MAXSIZE_BYTES}]" in q
    assert urban.MAX_RESPONSE_BYTES <= urban.QUERY_MAXSIZE_BYTES
    # civilian only: military land is subtracted, military tags excluded
    assert '[!"military"]' in q and "map_to_area" in q and "- .inmil" in q
    assert "(military|bunker)" in q
    # buildings and major roads, nothing else
    for word in ("amenity", "aeroway", "power", "node[", "name"):
        assert word not in q
    assert urban.query_digest(q) == urban.query_digest(urban.overpass_query(TEHRAN))


def test_the_visual_only_host_does_not_widen_the_snapshot_builds_egress():
    from naigos.pipeline import egress

    assert allowlist.ALLOWLIST[urban.SOURCE_KEY].visual_only
    assert "overpass-api.de" not in egress.allowlisted_hosts()
    assert "api.open-meteo.com" in egress.allowlisted_hosts()


def test_an_aoi_without_a_documented_box_is_refused(cache_dir):
    with pytest.raises(urban.UrbanDataError):
        urban.build("owens_valley", fetcher=_fake_fetcher([]))
    st = urban.load("owens_valley")
    assert st.state == "unavailable" and not st.available


# --- heights -----------------------------------------------------------------------------


def test_height_prefers_the_height_tag_then_levels_then_the_fallback():
    assert urban.building_height({"height": "42", "building:levels": "3"}, 400) == (42.0, "height")
    assert urban.building_height({"height": "12 m"}, 400) == (12.0, "height")
    assert urban.building_height({"height": "100 ft"}, 400) == (30.5, "height")
    # an invalid height falls through to levels, not to a guess
    assert urban.building_height({"height": "tall", "building:levels": "5"}, 400) == \
        (round(5 * urban.FLOOR_HEIGHT_M, 1), "levels")
    assert urban.building_height({"height": "9000", "building:levels": "2"}, 400)[1] == "levels"
    assert urban.building_height({"height": "0.5"}, 400)[1] == "fallback"
    # fallback: by tag, then by area
    assert urban.building_height({"building": "garage"}, 5000) == (urban.FLOOR_HEIGHT_M, "fallback")
    assert urban.building_height({"building": "house"}, 5000) == (
        round(2 * urban.FLOOR_HEIGHT_M, 1), "fallback")
    assert urban.fallback_levels("yes", 50) == 1
    assert urban.fallback_levels("yes", 200) == 3
    assert urban.fallback_levels("yes", 10_000) == 6
    assert urban.fallback_levels(None, 1) == 1


def test_heights_are_deterministic():
    tags = [{"building": "yes"}, {"building:levels": "4"}, {"height": "33.3"}, {"building": "retail"}]
    a = [urban.building_height(t, 321.0) for t in tags]
    b = [urban.building_height(dict(t), 321.0) for t in tags]
    assert a == b
    p1 = urban.derive(overpass_fixture(), TEHRAN, SOURCE)
    p2 = urban.derive(overpass_fixture(), TEHRAN, SOURCE)
    assert p1 == p2 and p1["cache_id"] == p2["cache_id"]
    # element order does not matter
    shuffled = overpass_fixture()
    shuffled["elements"].reverse()
    assert urban.derive(shuffled, TEHRAN, SOURCE)["cache_id"] == p1["cache_id"]


# --- geometry ----------------------------------------------------------------------------


def test_invalid_and_out_of_bounds_geometry_is_rejected_and_counted():
    p = urban.derive(overpass_fixture(), TEHRAN, SOURCE)
    c = p["counts"]
    assert c["buildings"] == 4
    assert c["rejected"] == {"malformed": 1, "excluded_tag": 1, "open_ring": 1,
                             "out_of_bounds": 1, "self_intersecting": 1}
    assert c["height_rule"] == {"height": 1, "levels": 1, "fallback": 2}
    # every drawn vertex is inside the documented box
    for ch in p["chunks"]:
        for rec in ch["b"] + ch["r"]:
            for lon, lat in urban.decode_coords(rec[1:], tuple(p["origin"])):
                assert TEHRAN.contains(lon, lat)
    # the road crossing the east edge survives as its in-bounds run only
    assert c["road_ways"] == 1 and c["roads"] == 1


def test_ring_checks_directly():
    ok, why = urban.clean_ring(_square(51.4, 35.72), TEHRAN)
    assert ok and why is None and len(ok) == 4
    assert urban.clean_ring(_square(51.4, 35.72, d=1e-6), TEHRAN) == (None, "degenerate")
    assert urban.clean_ring(_square(51.4, 35.72, d=0.02), TEHRAN) == (None, "implausible_area")
    assert urban.clean_ring(None, TEHRAN) == (None, "malformed")
    assert urban.self_intersects([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not urban.self_intersects([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_an_empty_or_partial_result_is_an_error_not_an_empty_layer():
    with pytest.raises(urban.UrbanDataError, match="no valid building"):
        urban.derive({"elements": []}, TEHRAN, SOURCE)
    partial = overpass_fixture()
    partial["remark"] = "runtime error: Query timed out in \"query\" at line 5 after 181 seconds."
    with pytest.raises(urban.UrbanDataError, match="partial"):
        urban.derive(partial, TEHRAN, SOURCE)
    with pytest.raises(urban.UrbanDataError):
        urban.derive({"nope": 1}, TEHRAN, SOURCE)


def test_encoding_round_trips_to_within_the_quantum():
    ring = [(51.4001234, 35.7204567), (51.4003, 35.7204567), (51.4003, 35.7206)]
    back = urban.decode_coords(urban.encode_coords(ring, (TEHRAN.west, TEHRAN.south)),
                               (TEHRAN.west, TEHRAN.south))
    assert np.allclose(np.array(back), np.array(ring), atol=urban.QUANTUM_DEG)


# --- the payload and the cache -----------------------------------------------------------


def test_the_payload_carries_provenance_and_attribution_but_no_raw_tags():
    p = urban.derive(overpass_fixture(), TEHRAN, SOURCE)
    assert p["schema"] == urban.SCHEMA and p["presentation_only"] is True
    assert p["source"]["attribution"].startswith("(c) OpenStreetMap contributors")
    assert "not used by terrain LOS" in p["note"]
    blob = json.dumps(p)
    # no tag VALUE anywhere in the payload...
    for leaked in ("Some Tower", "Some Avenue", "apartments", "barracks", "cafe", "addr:street",
                   '"tags"', '"id"'):
        assert leaked not in blob, f"{leaked!r} reached the browser payload"
    # ...and the geometry itself is numbers only (the height RULE is documented
    # in words at the top level, which is why this is scoped to the chunks)
    geometry = json.dumps(p["chunks"])
    assert set(re.findall(r'"(\w+)":', geometry)) <= {"k", "c", "b", "r"}
    for rec in (r for ch in p["chunks"] for r in ch["b"]):
        assert all(isinstance(v, int) for v in rec)


def test_build_caches_raw_and_derived_with_provenance_and_is_offline_after(cache_dir):
    calls = []
    prov = urban.build("tehran_basin", fetcher=_fake_fetcher(calls), log=lambda *a: None)
    assert len(calls) == 1 and calls[0] == urban.overpass_query(TEHRAN)
    root = cache_dir / "visual" / "urban" / "tehran_basin"
    assert root.is_dir()
    raw = root / prov["raw"]["path"]
    assert raw.exists() and prov["raw"]["sha256"] == urban._sha256(raw.read_bytes())
    assert prov["raw"]["query_sha256"] == urban.query_digest(calls[0])
    assert prov["raw"]["url"] == urban.OVERPASS_URL and prov["raw"]["fetched_at"]
    assert "Open Database License" in prov["licence"] and "OpenStreetMap" in prov["citation"]
    derived = urban.payload_path("tehran_basin")
    assert prov["derived"]["sha256"] == urban._sha256(derived.read_bytes())
    # the research manifest is untouched: this cache cannot become a parameter
    assert not (cache_dir / "manifest.json").exists()

    # second run: no network at all
    def refuse(q):
        raise AssertionError("second run touched the network")
    again = urban.build("tehran_basin", fetcher=refuse, log=lambda *a: None)
    assert again["derived"]["cache_id"] == prov["derived"]["cache_id"]
    assert urban.build("tehran_basin", offline=True, fetcher=refuse,
                       log=lambda *a: None)["derived"]["sha256"] == prov["derived"]["sha256"]


def test_offline_without_a_cache_refuses_rather_than_fetching(cache_dir):
    with pytest.raises(urban.UrbanDataError, match="offline"):
        urban.build("tehran_basin", offline=True, fetcher=_fake_fetcher([]), log=lambda *a: None)


def test_a_tampered_raw_file_is_refused(cache_dir):
    prov = urban.build("tehran_basin", fetcher=_fake_fetcher([]), log=lambda *a: None)
    raw = cache_dir / "visual" / "urban" / "tehran_basin" / prov["raw"]["path"]
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(urban.UrbanDataError, match="sha256"):
        urban.build("tehran_basin", fetcher=_fake_fetcher([]), log=lambda *a: None)


def test_load_reports_missing_invalid_and_available(cache_dir):
    st = urban.load("tehran_basin")
    assert st.state == "unavailable" and "naigos.demo.urban" in st.reason
    assert st.summary()["counts"] is None and st.focus  # still knows where the city is
    urban.build("tehran_basin", fetcher=_fake_fetcher([]), log=lambda *a: None)
    st = urban.load("tehran_basin")
    assert st.available and st.summary()["counts"]["buildings"] == 4
    assert st.summary()["cache_id"] == st.payload["cache_id"]
    # a payload for another box is refused, not drawn in the wrong place
    bad = json.loads(urban.payload_path("tehran_basin").read_text())
    bad["bounds"] = {"west": 0, "south": 0, "east": 1, "north": 1}
    urban.payload_path("tehran_basin").write_text(json.dumps(bad))
    st = urban.load("tehran_basin")
    assert st.state == "invalid" and not st.available


# --- the server --------------------------------------------------------------------------


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def available(cache_dir):
    urban.build("tehran_basin", fetcher=_fake_fetcher([]), log=lambda *a: None)
    return urban.load("tehran_basin")


@pytest.mark.parametrize("mode,served", [("physics", False), ("photorealistic", False),
                                         ("urban-presentation", True)])
def test_the_urban_route_serves_the_layer_only_in_its_own_mode(available, mode, served):
    visual = imagery.resolve_visual_config(mode, local_urban=True)
    srv, base = _serve(live.make_handler(None, {}, None, visual=visual, urban_status=available))
    try:
        if served:
            body = json.loads(urllib.request.urlopen(base + "/urban").read())
            assert body["schema"] == urban.SCHEMA and body["counts"]["buildings"] == 4
        else:
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + "/urban")
            assert e.value.code == 404
    finally:
        srv.shutdown()


def test_the_smoke_report_describes_the_city_configuration(available):
    visual = imagery.resolve_visual_config("urban-presentation", local_urban=True)
    meta = {"west": 51.0, "east": 52.0, "south": 35.4, "north": 36.3, "nx": 4, "ny": 4,
            "min_m": 1000.0, "max_m": 4000.0, "grid": "4x4", "cell_m": 500.0}
    tb = np.full(16, 1200, dtype="<i2").tobytes()
    notes = {"theatre": "tehran_basin", "geo_bounds": {k: meta[k] for k in ("west", "east", "south", "north")}}
    r = live.smoke_report(visual, notes, meta, None, available,
                          camera.preset_key(None, True), tb)
    assert r["rendered_pixels"] is False
    assert r["visual_mode"] == "urban-presentation" and r["evidence_grade"] is False
    assert r["imagery"] == "osm"
    assert r["requested_geometry_source"] == r["geometry_source"] == "local_osm_extrusions"
    assert r["provider_readiness"] == imagery.PROVIDER_NOT_REQUESTED
    # a credential in the environment changes nothing under the default
    tok = live.smoke_report(
        imagery.resolve_visual_config("urban-presentation", ion_token=TOKEN, local_urban=True),
        notes, meta, None, available, camera.preset_key(None, True), tb)
    assert tok["geometry_source"] == "local_osm_extrusions" and tok["building_count"] == 4
    assert tok["provider_readiness"] == imagery.PROVIDER_NOT_REQUESTED
    assert tok["credentials_present"] == {"ion_token": True, "google_api_key": False}
    assert TOKEN not in json.dumps(tok)
    assert r["local_cache_id"] == available.payload["cache_id"] and r["local_cache_sha256"]
    assert r["building_count"] == 4 and r["road_count"] == 1
    assert r["camera_preset"] == "urban_overview"
    assert r["model_fallback_count"] == 0
    assert r["render_only_building_occlusion"] is False
    # physics draws no buildings even with the cache present
    phys = live.smoke_report(imagery.resolve_visual_config("physics"), notes, meta, None,
                             available, None, tb)
    assert phys["building_count"] == 0 and phys["geometry_source"] is None
    assert phys["camera_preset"] == "terrain_overview"


# --- camera ------------------------------------------------------------------------------


def test_the_city_cameras_are_oblique_and_above_the_ground():
    bounds = {"west": 51.05, "east": 51.95, "south": 35.40, "north": 36.30}
    terrain = {"min_m": 900.0, "max_m": 4270.0}
    focus = {"lon": 51.405, "lat": 35.725, "ground_m": 1290.0}
    p = camera.presets(bounds, terrain, urban=focus, default="urban-overview")
    assert p["default"] == "urban_overview"
    for key in ("urban_overview", "street_canyon"):
        v = p[key]
        assert -30.0 < v["pitch_deg"] < -3.0, f"{key} is not oblique"
        assert v["camera_height_m"] > focus["ground_m"] + 50
        assert (v["lon"], v["lat"]) == (focus["lon"], focus["lat"])
    assert p["street_canyon"]["range_m"] < p["urban_overview"]["range_m"]
    # the existing presets are untouched, and physics still opens on terrain
    assert camera.presets(bounds, terrain)["default"] == "terrain_overview"
    assert p["top_down_analysis"]["pitch_deg"] == -90.0
    assert camera.preset_key(None, urban_mode=True) == "urban_overview"
    assert camera.preset_key("analysis-topdown") == "top_down_analysis"
    with pytest.raises(ValueError):
        camera.preset_key("orbit")


# --- the page ----------------------------------------------------------------------------


def _pure_js() -> str:
    a = PAGE.index("/*__URBAN_PURE_BEGIN__*/")
    b = PAGE.index("/*__URBAN_PURE_END__*/")
    return PAGE[a:b]


def test_the_pure_helpers_touch_neither_cesium_nor_the_dom():
    js = _pure_js()
    for word in ("Cesium.", "document.", "viewer", "window.", "fetch("):
        assert word not in js


NODE = shutil.which("node")


def _node(expr: str):
    script = _pure_js() + f"\nprocess.stdout.write(JSON.stringify(({expr})));\n"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_provider_is_not_active_until_a_loaded_tile_is_visible():
    base = "{requested: true, urban: true, created: true, shown: true, failed: false, " \
           "tilesLoaded: 0, tilesVisible: 0}"
    cases = _node(f"""[
      providerStatus({{requested: false}}),
      providerStatus({{requested: true, created: false}}),
      providerStatus({base}),
      providerStatus(Object.assign({base}, {{tilesLoaded: 3}})),
      providerStatus(Object.assign({base}, {{tilesVisible: 3}})),
      providerStatus(Object.assign({base}, {{tilesLoaded: 3, tilesVisible: 1, shown: false}})),
      providerStatus(Object.assign({base}, {{tilesLoaded: 3, tilesVisible: 1, failed: true, reason: "403"}})),
      providerStatus(Object.assign({base}, {{tilesLoaded: 3, tilesVisible: 1}})),
      providerStatus(Object.assign({base}, {{tilesLoaded: 3, tilesVisible: 1, urban: false}})),
    ]""")
    states = [c["state"] for c in cases]
    assert states == ["not_requested", "loading", "loading", "loading", "loading", "hidden",
                      "failed", "active", "active"]
    assert [c["active"] for c in cases] == [False] * 7 + [True, True]
    assert cases[7]["label"] == "provider buildings active"
    assert cases[8]["label"] == "provider tiles active"
    assert "403" in cases[6]["label"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_what_is_drawing_the_city_is_decided_from_observed_state():
    r = _node("""[urbanGeometryActive({active: true}, true), urbanGeometryActive({active: false}, true),
                   urbanGeometryActive({active: false}, false), urbanGeometryActive(null, false)]""")
    assert r == ["provider_3d_tiles", "local_osm_extrusions", None, None]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_page_decodes_exactly_what_the_builder_encoded():
    p = urban.derive(overpass_fixture(), TEHRAN, SOURCE)
    rec = next(r for ch in p["chunks"] for r in ch["b"])
    js = _node(f"decodeCoords({json.dumps(rec)}, 1, {json.dumps(p['origin'])}, {p['quantum_deg']})")
    py = [c for pt in urban.decode_coords(rec[1:], tuple(p["origin"])) for c in pt]
    assert np.allclose(js, py, atol=1e-12)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_loader_respects_its_cap_and_its_radius_nearest_first():
    chunks = [{"c": [51.40 + 0.01 * i, 35.72], "b": [[0]] * 1000, "r": []} for i in range(10)]
    plan = _node(f"planChunks({json.dumps(chunks)}, 51.40, 35.72, new Set(), 0, 3500, 1e9)")
    assert plan == [0, 1, 2], "nearest first, and never past the cap"
    plan = _node(f"planChunks({json.dumps(chunks)}, 51.40, 35.72, new Set([0, 1]), 2000, 90000, 2500)")
    assert plan == [2], "only unloaded chunks within the radius"
    # a focus far from every chunk still gets the nearest one, not nothing
    plan = _node(f"planChunks({json.dumps(chunks)}, 10.0, 10.0, new Set(), 0, 5000, 100)")
    assert plan == [0]
    limits = _node("URBAN_LIMITS")
    assert limits["initialBuildings"] <= limits["maxBuildings"] <= 100_000
    assert 0 < limits["sliceBuildings"] <= 5000
    assert limits["cullRadiusM"] > limits["loadRadiusM"]
    assert 0 < limits["seeThroughAlpha"] < 1


def test_building_occlusion_starts_off_and_markers_stay_on_top():
    assert "occlusion: VISUAL.building_occlusion_default === true" in PAGE
    assert "urbanOverlaysOnTop = city.active && !city.occlusion;" in PAGE
    formula = PAGE[PAGE.index("const applyDepthTest = () => {"):PAGE.index("const xrayBtn")]
    assert "urbanOverlaysOnTop" in formula
    # see-through (translucent) unless the render-only occlusion is switched on
    assert "translucent: !city.occlusion" in PAGE
    assert "building occlusion: ON (render-only)" in PAGE


def test_buildings_stand_on_the_drawn_dem_never_on_sea_level():
    block = PAGE[PAGE.index("function buildChunk(ci)"):PAGE.index("function showChunk(rec)")]
    assert "sampleGrid(" in block
    assert "height: gmin - URBAN_LIMITS.footingM" in block
    assert "extrudedHeight: gmax + rec[0] / 10" in block
    assert "asynchronous: true" in block
    assert "classificationType: Cesium.ClassificationType.TERRAIN" in block


def test_the_page_says_what_is_and_is_not_drawn():
    assert "URBAN DATA UNAVAILABLE" in PAGE
    assert "provider buildings active" in PAGE
    # the label comes out of providerStatus() and nowhere else that could fake it
    assert PAGE.count('"provider buildings active"') == 1
    assert "providerProbe.shown = true;" in PAGE
    photo = PAGE[PAGE.index("async function showPhotorealisticContext()"):PAGE.index("photoBtn.onclick =")]
    assert "tileset.tileLoad.addEventListener(" in photo
    assert "tileset.tileVisible.addEventListener(" in photo
    # visibility is re-proven each time the tiles are shown
    assert photo.index("providerProbe.tilesVisible = 0;") < photo.index("tileset.show = true;")
    assert 'id="urbanpanel" hidden' in PAGE and 'id="citybanner" hidden' in PAGE


def test_the_page_never_requests_osm_data_itself():
    code = re.sub(r"//.*", "", PAGE)
    assert "overpass" not in code.lower()
    assert "openstreetmap.org/api" not in code
    block = PAGE[PAGE.index("// --------------------------------------------------------------- city layer"):
                 PAGE.index("// ------------------------------------------------------- visual modes")]
    assert 'getRoute("/urban")' in block and "fetch(" not in block


# --- static export -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    import jax

    from naigos.data.geodetic import GeoRef
    from naigos.demo import replay
    from naigos.env.config import EnvConfig
    from naigos.env.flight_env import NaigosEnv
    from naigos.rl.networks import Actor

    cfg = EnvConfig(n_blue=2, n_threat=4, n_threat_active=3, max_steps=40)
    env = NaigosEnv(cfg)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(cfg).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=2, seed=3)
    notes = {"theatre": "test_basin",
             "georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0,
                        "origin_northing_m": 3_950_000.0}}
    georef = GeoRef(**notes["georef"])
    lon, lat = georef.to_wgs84(np.array([0.0, cfg.terrain.extent_x]),
                               np.array([0.0, cfg.terrain.extent_y]))
    notes["geo_bounds"] = {"west": float(lon[0]), "east": float(lon[1]),
                           "south": float(lat[0]), "north": float(lat[1])}
    out = tmp_path_factory.mktemp("urbandemo") / "demo.json"
    replay.to_json(results, cfg, out, env=env, notes=notes)
    return out


def test_a_static_urban_replay_embeds_the_layer_and_fetches_nothing(recording, available):
    payload = live.static_payload(recording, urban_status=available,
                                  camera_key="urban_overview", embed_urban=True)
    assert set(payload) == {"/scene", "/frames", "/terrain", "/urban"}
    assert payload["/urban"]["counts"]["buildings"] == 4
    assert payload["/scene"]["camera"]["default"] == "urban_overview"
    assert payload["/scene"]["urban"]["cache_id"] == available.payload["cache_id"]
    # the page's route reader never falls through to fetch() for an embedded page
    assert "const getRoute = async (path) => (EMBED ? EMBED[path] : (await fetch(path)).json());" in PAGE
    # a physics export knows where the city is but does not carry it
    phys = live.static_payload(recording, urban_status=available, embed_urban=False)
    assert "/urban" not in phys and phys["/scene"]["camera"]["default"] == "terrain_overview"


def test_the_urban_export_refuses_to_ship_an_empty_city(recording, cache_dir):
    from naigos.demo import viewer

    with pytest.raises(SystemExit, match="embeds the local city layer"):
        viewer.build(recording, recording.with_name("u.html"), visual_mode="urban-presentation")
    with pytest.raises(SystemExit, match="credential-free"):
        viewer.build(recording, recording.with_name("p.html"), visual_mode="photorealistic")


def test_the_static_urban_page_is_credential_free(recording, available, monkeypatch):
    from naigos.demo import viewer

    monkeypatch.setattr(urban, "load", lambda aoi: available)
    monkeypatch.setenv("NAIGOS_CESIUM_ION_TOKEN", TOKEN)
    out = viewer.build(recording, recording.with_name("urban.html"), visual_mode="urban-presentation")
    html = out.read_text()
    assert TOKEN not in html and "AIzaSy" not in html
    embed = json.loads(re.search(r"^const EMBED = (\{.*\});$", html, re.M).group(1))
    assert embed["/urban"]["schema"] == urban.SCHEMA
    assert '"mode": "urban-presentation"' in html and '"evidence_grade": false' in html
    assert '"tileset": null' in html, "a static export never takes the provider path"


# --- isolation ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", ["env", "rl"])
def test_the_simulation_cannot_reach_the_city_layer(package):
    hits = []
    for path in sorted((REPO / "naigos" / package).rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                if any("urban" in n or "demo" in n for n in names):
                    hits.append(f"{path.relative_to(REPO)}:{node.lineno}")
        text = path.read_text().lower()
        for word in ("osm_urban", "overpass", "urban-presentation", "building_height"):
            if word in text:
                hits.append(f"{path.relative_to(REPO)}: mentions {word}")
    assert not hits, f"the simulation reaches the city layer: {hits}"


def test_the_city_module_imports_nothing_from_the_simulation():
    tree = ast.parse((REPO / "naigos" / "demo" / "urban.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert not any(m.lstrip(".").split(".")[0] in ("env", "rl", "jax", "numpy") for m in mods), mods
