"""The city standard in the page, and its separation from the simulation.

Three kinds of check, all offline:

* the page's pure helpers (``/*__CITY_PURE_BEGIN__*/`` ... ``END``) are RUN under
  node and compared, number for number, with their Python twins -- the chase
  camera's zone clamp, and the per-effect ambience state;
* the rendering code is read statically for what it must and must not do
  (depth-tested effects, no random numbers or wall clock in what is drawn, the
  labels, the zone cut-outs and camera guards);
* the simulation is stepped with and without the ambience stream, and every
  leaf of its state is compared -- the proof that the visual layer cannot
  alter simulation state, LOS or detection.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from naigos.data.geodetic import GeoRef
from naigos.demo import ambience, cities, live, presentation, replay, viewer
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.networks import Actor
from naigos.rl.ppo import greedy_policy

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
PURE = PAGE[PAGE.index("/*__CITY_PURE_BEGIN__*/"):PAGE.index("/*__CITY_PURE_END__*/")]
AMB_JS = PAGE[PAGE.index("// ---- fictional ambience: presentation-only VFX"):
              PAGE.index("// ---- replay mode: same renderer, on Cesium's clock")]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_js(body: str):
    """Evaluate ``body`` after the pure block under node; it must print one JSON value."""
    out = subprocess.run([NODE, "-e", PURE + "\n" + body], capture_output=True, text=True,
                         timeout=60, check=True)
    return json.loads(out.stdout)


# --- the pure block is pure --------------------------------------------------------------


def test_the_pure_block_touches_neither_cesium_nor_the_dom_nor_chance():
    code = re.sub(r"//.*", "", PURE)
    for banned in ("Cesium.", "document.", "window.", "Math.random", "Date.now", "performance.now"):
        assert banned not in code, banned


@needs_node
def test_the_page_follow_clamp_is_the_python_one():
    rng = random.Random(4)
    zones = [{"name": "z", "bbox": [39.80, 21.40, 39.86, 21.44], "policies": ["camera"]},
             {"name": "y", "bbox": [39.90, 21.33, 40.0, 21.43], "policies": ["camera", "render_cutout"]},
             {"name": "x", "bbox": [39.6, 21.5, 39.61, 21.51], "policies": ["render_cutout"]}]
    cases = [(39.60 + rng.random() * 0.3, 21.30 + rng.random() * 0.3, rng.random() * 360.0)
             for _ in range(300)]
    js = run_js(f"const Z = {json.dumps(zones)}; const C = {json.dumps(cases)};"
                f"console.log(JSON.stringify(C.map(([a, b, h]) => safeFollowHeading(a, b, h, Z, "
                f"{cities.VIEW_HALF_ANGLE_DEG}, {cities.FRAMING_RANGE_M}))));")
    py = [cities.safe_follow_heading(a, b, h, zones) for a, b, h in cases]
    assert np.allclose(js, py, atol=1e-6)
    assert sum(abs(a - b[2]) > 1e-6 for a, b in zip(py, cases)) > 20, "the clamp never fired"


@needs_node
def test_zone_lookup_matches_policy():
    zones = [{"name": "z", "bbox": [0, 0, 1, 1], "policies": ["camera"]}]
    assert run_js(f"console.log(JSON.stringify([zoneAt(0.5,0.5,{json.dumps(zones)},'camera'),"
                  f"zoneAt(0.5,0.5,{json.dumps(zones)},'render_cutout'), zoneAt(2,2,{json.dumps(zones)},'camera')]))") \
        == ["z", None, None]


def _events():
    c = cities.CITIES["tehran_basin"]
    m = ambience.build_mask(c, 5)
    return ambience.generate(c, ambience.get_setting("sustained"), m, 5, 0.0, 400.0)


@needs_node
def test_effect_state_is_deterministic_bounded_and_honours_reduced_motion():
    evs = _events()
    times = [0.0, 3.0, 17.5, 60.0, 121.0, 250.0, 399.0]
    body = (f"const E = {json.dumps(evs)}; const T = {json.dumps(times)};"
            "const run = (r, p) => T.map(t => E.map(ev => ambienceState(ev, t, r, p)));"
            "console.log(JSON.stringify({a: run(false, false), b: run(false, false),"
            " r: run(true, false), p: run(false, true),"
            " act: T.map(t => ambienceActive(E, t, 4).map(e => e.id))}));")
    out = run_js(body)
    assert out["a"] == out["b"], "the same event at the same sim time must look the same"
    for frame in out["a"]:
        for st in frame:
            for q in st["puffs"]:
                assert 0.0 <= q["alpha"] <= 0.4 and q["r"] > 0
            if st["dust"]:
                assert 0.0 <= st["dust"]["alpha"] <= 0.4
            assert len(st["puffs"]) <= 7
    for frame in out["r"]:
        for st in frame:
            assert st["flash"] is None and st["sparks"] == [], "reduced motion draws no flash or spark"
    for frame in out["p"]:
        for st in frame:
            assert len(st["puffs"]) <= 3 and st["sparks"] == []
    for t, ids in zip(times, out["act"]):
        live_ids = [e["id"] for e in evs if e["t_s"] <= t <= e["t_s"] + max(e["duration_s"], e["flash_s"])]
        assert len(ids) <= 4 and set(ids) <= set(live_ids)
    # an effect is not drawn before it starts
    first = min(evs, key=lambda e: e["t_s"])
    st = run_js(f"console.log(JSON.stringify(ambienceState({json.dumps(first)}, {first['t_s'] - 0.5}, false, false)))")
    assert st["live"] is False


# --- the renderer, statically ------------------------------------------------------------


def test_effects_are_depth_tested_labelled_and_chance_free():
    code = re.sub(r"//.*", "", AMB_JS)
    assert "Math.random" not in code and "Date.now" not in code
    # every ambience graphic is depth-tested: an effect can never draw over an
    # aircraft, a threat, a marker or a label
    assert code.count("disableDepthTestDistance: 0") >= 3
    assert "Number.POSITIVE_INFINITY" not in code
    assert "FICTIONAL AMBIENCE" in code and "not simulated events" in code
    assert "prefers-reduced-motion" in code
    # the replay path runs on the Cesium clock, not a wall clock
    rp = PAGE[PAGE.index("// ---- replay mode: same renderer, on Cesium's clock"):
              PAGE.index("  const seenEvents = new Set();")]
    assert "startAmbience(() => ambEvents," in rp
    assert "secondsDifference(viewer.clock.currentTime, EPOCH)" in rp


def test_the_ambience_tag_is_hidden_unless_the_stream_is_on():
    assert '<div id="ambtag" hidden></div>' in PAGE
    assert "ambTag.hidden = !amb.on;" in AMB_JS


def test_the_scenario_line_is_the_first_thing_in_the_hud():
    hud = PAGE[PAGE.index('<div id="hud">'):PAGE.index("<h2>Aircraft")]
    assert 'id="scenario"' in hud
    assert "scenario: ${SCENARIO.label}" in PAGE
    assert "notional contested-airspace simulation" in PAGE


def test_protected_zones_are_cut_out_clipped_and_guarded():
    assert "const CUTOUTS = ZONES.filter(z => z.policies.includes(\"render_cutout\"));" in PAGE
    assert "new Cesium.SingleTileImageryProvider({" in PAGE[PAGE.index("if (CUTOUTS.length) {"):]
    photo = PAGE[PAGE.index("async function showPhotorealisticContext()"):PAGE.index("photoBtn.onclick =")]
    assert "tileset.clippingPolygons = new Cesium.ClippingPolygonCollection({" in photo
    assert 'zoneAt(lon, lat, ZONES, "camera") === null' in PAGE
    assert "chaseHeading(a.lon, a.lat, a.heading ?? 0)" in PAGE
    # replay hands the camera to entity tracking only when no zone needs clamping
    assert "viewer.trackedEntity = follow && !SAFE_FOLLOW" in PAGE


def test_the_page_still_loads_one_external_script():
    assert re.findall(r'<script[^>]*src="([^"]+)"', PAGE) == [
        "https://cesium.com/downloads/cesiumjs/releases/1.145/Build/Cesium/Cesium.js"]


# --- the simulation cannot be touched by any of this ---------------------------------------

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=120)
NOTES = {"georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0,
                    "origin_northing_m": 3_950_000.0}}


def _sim(with_ambience: bool) -> live.Simulation:
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    amb = (ambience.LiveAmbience(cities.CITIES["tehran_basin"],
                                 ambience.AmbienceConfig(True, "sustained", 1))
           if with_ambience else None)
    return live.Simulation(env=env, policy=greedy_policy(params, CFG),
                           georef=GeoRef(**NOTES["georef"]), speed=1e9, reroll_s=0.0,
                           ambience=amb)


def _advance(s: live.Simulation, n: int):
    for _ in range(n):
        s.key, k = jax.random.split(s.key)
        before = s.state
        s2, o2, terms, info, feas, _ = s._step(s.state, s.obs, k)
        s.state, s.obs = s2, o2
        s.sim_t += s.env.cfg.dt
        s._derive_events(before, s2, jax.device_get(terms), info)
        s._publish(np.asarray(feas), np.asarray(terms.exposure))


def test_the_ambience_stream_cannot_change_the_simulation():
    """Same seed, with and without the stream: every state leaf, every
    observation, and the detection and LOS numbers the frame reports agree."""
    a, b = _sim(True), _sim(False)
    _advance(a, 40)
    _advance(b, 40)
    for x, y in zip(jax.tree.leaves(jax.device_get(a.state)), jax.tree.leaves(jax.device_get(b.state))):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(jax.tree.leaves(jax.device_get(a.obs)), jax.tree.leaves(jax.device_get(b.obs))):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    fa, fb = a.latest(), b.latest()
    for ka, kb in zip(fa["aircraft"], fb["aircraft"]):
        assert ka["tracker"] == kb["tracker"] and ka["los"] == kb["los"] and ka["lock"] == kb["lock"]
    assert fa["ambience"] and fb["ambience"] == []
    assert all(e["fictional"] for e in fa["ambience"])


def test_the_live_stream_is_a_function_of_the_seed():
    a, b = _sim(True), _sim(True)
    _advance(a, 20)
    _advance(b, 20)
    assert [e["id"] for e in a.latest()["ambience"]] == [e["id"] for e in b.latest()["ambience"]]


# --- static export -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=2, seed=7)
    g = GeoRef(**NOTES["georef"])
    lon, lat = g.to_wgs84(np.array([0.0, CFG.terrain.extent_x]), np.array([0.0, CFG.terrain.extent_y]))
    notes = dict(NOTES, theatre="tehran_basin",
                 geo_bounds={"west": float(lon[0]), "east": float(lon[1]),
                             "south": float(lat[0]), "north": float(lat[1])})
    out = tmp_path_factory.mktemp("rec") / "demo.json"
    replay.to_json(results, CFG, out, env=env, notes=notes,
                   provenance={"seed": 7, "checkpoint": {"file": "x.pkl", "trained_on": "owens_valley"}})
    return out


def _embed(html: str) -> dict:
    return json.loads(re.search(r"^const EMBED = (\{.*\});$", html, re.M).group(1))


def test_a_recording_says_it_is_notional_and_who_trained_the_policy(recording):
    d = json.loads(recording.read_text())
    assert d["scenario"]["notional"] is True and d["seed"] == 7
    sc = live.static_payload(recording)["/scene"]
    assert sc["scenario"]["label"] == "notional contested-airspace simulation"
    assert sc["scenario"]["checkpoint"]["relation"] == "zero_shot"


def test_the_export_embeds_a_stable_ambience_stream(recording, tmp_path):
    p = presentation.resolve("tehran_basin", "urban-presentation", ambience="conflict_ambience",
                             visual_seed=9)
    a = live.static_payload(recording, presentation=p)
    b = live.static_payload(recording, presentation=p)
    sa, sb = a["/frames"]["ambience"], b["/frames"]["ambience"]
    assert sa == sb and sa["events"], "re-exporting must reproduce the stream exactly"
    assert sa["seed"] == 9 and "FICTIONAL" in sa["note"]
    assert a["/scene"]["ambience"]["enabled"] is True
    assert a["/scene"]["atmosphere"]["key"] == "high_basin_clear"
    # a different seed is a different stream
    q = presentation.resolve("tehran_basin", "urban-presentation", ambience="conflict_ambience",
                             visual_seed=10)
    assert live.static_payload(recording, presentation=q)["/frames"]["ambience"]["events"] != sa["events"]


def test_physics_refuses_the_ambience(recording, tmp_path):
    with pytest.raises(presentation.PresentationError, match="presentation-only"):
        presentation.resolve("tehran_basin", "physics", ambience="conflict_ambience")
    with pytest.raises(SystemExit, match="presentation-only"):
        viewer.build(recording, tmp_path / "x.html", visual_mode="physics",
                     ambience="conflict_ambience")


def test_a_theatre_without_a_city_cannot_have_ambience():
    with pytest.raises(presentation.PresentationError, match="city config"):
        presentation.resolve("owens_valley", "photorealistic", ambience="conflict_ambience")


def test_physics_keeps_its_light_and_carries_the_scenario():
    p = presentation.resolve("tehran_basin", "physics")
    f = p.scene_fields()
    assert f["atmosphere"]["key"] == "neutral" and f["ambience"]["enabled"] is False
    from naigos.demo import camera

    assert f["lighting"] == camera.lighting()
    assert f["scenario"]["notional"] is True
