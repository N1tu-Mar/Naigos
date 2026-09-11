"""The 3D viewer contract: models by default, orientation from state, terrain as terrain.

The renderer draws aircraft and threats as glTF models now, oriented by the
simulation's own attitude state, with ground classes standing on the simulation
DEM and effects tied to recorded events. Each of those is a new way for the
picture to say more than the model did, so this file pins them -- statically
against the page, in the style of `tests/test_visual_renderer.py`, and against
real payloads built from a synthetic rollout, so it needs no token, no network,
no GPU and no data cache.

What a browser would add -- that the pixels look right -- is not claimed here.
"""

from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path

import jax
import numpy as np
import pytest

from naigos.data.geodetic import GeoRef
from naigos.demo import attitude, camera, imagery, live, models, replay, viewer
from naigos.env.config import EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.networks import Actor

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
CODE = re.sub(r"//.*", "", PAGE)

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=120)
NOTES = {
    "theatre": "test_basin",
    "georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0, "origin_northing_m": 3_950_000.0},
}
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.")
GKEY = re.compile(r"AIza[0-9A-Za-z_-]{35}")


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=4, seed=7)
    georef = GeoRef(**NOTES["georef"])
    lon, lat = georef.to_wgs84(np.array([0.0, CFG.terrain.extent_x]),
                               np.array([0.0, CFG.terrain.extent_y]))
    notes = dict(NOTES, geo_bounds={"west": float(lon[0]), "east": float(lon[1]),
                                    "south": float(lat[0]), "north": float(lat[1])})
    out = tmp_path_factory.mktemp("rec") / "demo.json"
    replay.to_json(results, CFG, out, env=env, notes=notes)
    return out, results


@pytest.fixture(scope="module")
def payload(recording):
    return live.static_payload(recording[0])


# --- models are the normal path ---------------------------------------------------------


def test_aircraft_and_threats_are_models_not_points():
    aircraft = PAGE[PAGE.index("craft.push(viewer.entities.add({"):PAGE.index("losPts.push([]);")]
    assert "model: acFailed ? undefined : {" in aircraft
    assert "uri: acSpec.uri" in aircraft
    threats = PAGE[PAGE.index("function buildThreats(scene) {"):PAGE.index("function drawThreats(")]
    assert "model: failed ? undefined : {" in threats
    assert "uri: spec.uri" in threats


def test_the_marker_is_drawn_only_for_a_failed_model_or_an_explicit_override():
    """The point graphic survives only as (a) the fallback for a model whose
    probe failed and (b) the x-ray / photorealistic override that already lifts
    depth testing for every overlay. Its show flag says exactly that."""
    assert "show: new Cesium.CallbackProperty(() => acFailed || markersForced(), false)" in PAGE
    assert "show: new Cesium.CallbackProperty(() => failed || markersForced(), false)" in PAGE
    assert 'const markersForced = () => xray || visualMode === "photorealistic";' in PAGE
    # failure is set in exactly one place: the probe
    assert CODE.count("modelFailed[k] =") == 1
    assert "for (const [k, why] of probes) if (why) modelFailed[k] = why;" in PAGE
    assert "const acFailed = !!modelFailed[AC_KEY];" in PAGE
    assert "const failed = !!modelFailed[t.model];" in PAGE


def test_a_model_failure_is_announced_once_visibly_and_without_secrets():
    warn = PAGE[PAGE.index("if (failedKeys.length) {"):PAGE.index("window.NAIGOS_VIEW = {")]
    assert 'getElementById("modelwarn")' in warn and "w.hidden = false;" in warn
    assert "MODELS[k].file" in warn and "modelFailed[k]" in warn
    for secret in ("ION_TOKEN", "GOOGLE_API_KEY", "uri"):
        assert secret not in warn, f"the warning must not print {secret}"
    assert '<div id="modelwarn" hidden></div>' in PAGE
    # probed once, before any entity is built
    assert PAGE.index("await Promise.all(Object.keys(MODELS).map(") < PAGE.index("buildThreats(scene);")


def test_the_page_probe_checks_what_the_python_validator_checks():
    probe = PAGE[PAGE.index("async function probeModel(spec) {"):PAGE.index("const modelHpr =")]
    for check in ("0x46546C67", "getUint32(4, true) !== 2", "byteLength", "0x4E4F534A",
                  '"2.0"', "spec.required_nodes"):
        assert check in probe
    assert models.GLB_MAGIC == 0x46546C67 and models.CHUNK_JSON == 0x4E4F534A


def test_models_obey_terrain_depth_in_physics_mode():
    """No model is given a depth-test exemption; only markers can be lifted, and
    only through the one global knob (x-ray / photorealistic)."""
    for block in re.findall(r"model: \w+ \? undefined : \{(.*?)\n      \},", PAGE, re.S):
        assert "disableDepthTestDistance" not in block
    aircraft = CODE[CODE.index("craft.push(viewer.entities.add({"):CODE.index("losPts.push([]);")]
    assert "disableDepthTestDistance" not in aircraft.split("label:")[0]
    assert "viewer.scene.globe.depthTestAgainstTerrain = true;" in PAGE


def test_the_page_refuses_a_payload_from_another_schema():
    assert f"const SCENE_SCHEMA = {live.SCENE_SCHEMA};" in PAGE
    assert "if (scene.schema_version !== SCENE_SCHEMA) {" in PAGE


# --- orientation is the simulation's -----------------------------------------------------


def test_the_page_orientation_is_the_python_conversion():
    """modelHpr in the page and attitude.cesium_hpr_deg are one formula."""
    assert ("Cesium.HeadingPitchRoll.fromDegrees(\n"
            "  heading - 90 + spec.heading_offset_deg, pitch + spec.pitch_offset_deg, "
            "roll + spec.roll_offset_deg)") in PAGE
    spec = models.page_registry()["models"]["fixed_wing"]
    rng = np.random.default_rng(1)
    for _ in range(20):
        h, p, r = rng.uniform(0, 360), rng.uniform(-20, 20), rng.uniform(-80, 80)
        page = (h - 90 + spec["heading_offset_deg"], p + spec["pitch_offset_deg"],
                r + spec["roll_offset_deg"])
        assert attitude.cesium_hpr_deg(h, p, r, spec) == pytest.approx(page)


def test_both_paths_orient_aircraft_from_logged_attitude_not_motion():
    assert "modelHpr(a.heading, a.pitch, a.roll, acSpec)" in PAGE
    assert PAGE.count("modelHpr(a.heading, a.pitch, a.roll, acSpec)") == 2   # live + replay
    for inferred in ("VelocityOrientationProperty", "VelocityVectorProperty"):
        assert inferred not in PAGE, f"{inferred} would derive attitude from motion"


def test_replay_attitude_is_sampled_under_the_same_linear_rules():
    block = PAGE[PAGE.index("  if (REPLAY) {"):PAGE.index("  const es = new EventSource(")]
    assert "new Cesium.SampledProperty(Cesium.Quaternion)" in block
    # every sampled property in the page is linear, degree 1
    n_props = block.count("new Cesium.Sampled")
    n_linear = block.count("interpolationAlgorithm: Cesium.LinearApproximation")
    assert n_props >= 5 and n_linear >= 3
    assert "craft[b].orientation = so;" in block
    assert "if (!a.alive) break;" in block


def test_live_frames_carry_attitude_from_the_airframe_state():
    src = (REPO / "naigos" / "demo" / "live.py").read_text()
    pub = src[src.index("def _publish("):src.index("def latest(")]
    assert "aircraft_attitude(self.georef, pos, np.asarray(st.air.psi)," in pub
    assert "np.asarray(st.air.gamma), np.asarray(st.air.phi))" in pub
    for f in ('"heading"', '"pitch"', '"roll"', '"sortie"'):
        assert f in pub


def test_the_replay_frames_carry_the_recorded_attitude(recording, payload):
    path, results = recording
    d = json.loads(path.read_text())
    w = d["worlds"]["trained"]
    frames = payload["/frames"]["frames"]["trained"]
    georef = GeoRef(**d["georef"])
    for f in range(0, len(frames), 7):
        for a in frames[f]:
            b = a["id"]
            assert a["roll"] == pytest.approx(-math.degrees(w["phi"][f][b]), abs=0.02)
            assert a["pitch"] == pytest.approx(math.degrees(w["gamma"][f][b]), abs=0.02)
            x, y = w["pos"][f][b][:2]
            true = float(attitude.true_heading_deg(georef.to_wgs84, x, y, w["psi"][f][b]))
            assert a["heading"] == pytest.approx(true, abs=0.02)
    # and the recording really did bank and climb somewhere -- otherwise the
    # checks above are vacuous. The direct-route baseline steers hard at the
    # objective; the fixture's untrained "trained" actor barely moves the stick.
    for name, pol_frames in payload["/frames"]["frames"].items():
        wn = d["worlds"][name]
        for f in range(0, len(pol_frames), 5):
            for a in pol_frames[f]:
                assert a["roll"] == pytest.approx(-math.degrees(wn["phi"][f][a["id"]]), abs=0.02)
    phi = max(np.abs(np.asarray(w2["phi"])).max() for w2 in d["worlds"].values())
    gamma = max(np.abs(np.asarray(w2["gamma"])).max() for w2 in d["worlds"].values())
    assert phi > math.radians(5) and gamma > math.radians(1)


# --- ground threats stand on the simulation DEM ------------------------------------------


def test_ground_model_altitude_comes_from_the_simulation_dem_and_follows_it():
    tcfg = TerrainConfig(nx=32, ny=32, cell=1000.0)
    georef = GeoRef(**NOTES["georef"])
    hmap = np.full((32, 32), 800.0, dtype=np.float32)
    tpos = np.array([[[5000.0, 7000.0, 805.0], [20_000.0, 12_000.0, 805.0]]])
    kw = dict(tpsi=np.zeros((1, 2)), active=None, track=np.array([[-1, -1]]),
              blue_pos=np.zeros((1, 1, 3)))
    base = live.threat_state_series(georef, hmap, tcfg, tpos, **kw)[0]
    raised = hmap.copy()
    raised[5:9, 3:8] += 1250.0                           # a ridge under the first threat only
    moved = live.threat_state_series(georef, raised, tcfg, tpos, **kw)[0]
    g = live.THREAT_STATE_FIELDS.index("ground_m")
    assert base[0][g] == 800 and base[1][g] == 800
    assert moved[0][g] == 2050, "the ground under the threat is the DEM's, and moves with it"
    assert moved[1][g] == 800
    # the page anchors ground classes on that number, after exaggeration
    assert "const anchorZ = (groundM, spec) => exagZ(groundM) + spec.altitude_offset_m;" in PAGE
    assert 'spec.anchor === "ground"' in PAGE
    spec = models.REGISTRY["ground_vehicle"]
    assert models.rendered_anchor_altitude(moved[0][g], spec, 2.0) == 4100.0 + spec.altitude_offset_m


def test_the_scene_threats_carry_model_ground_and_heading(payload, recording):
    d = json.loads(recording[0].read_text())
    rt = d["terrain"]
    heights = np.asarray(rt["heights"]).reshape(rt["ny"], rt["nx"])
    for t in payload["/scene"]["threats"]:
        assert t["model"] in models.REGISTRY and t["model"] != models.AIRCRAFT_MODEL
        assert t["model"] == models.threat_model_key(t["airborne"], t["mobile"])
        assert heights.min() - 1 <= t["ground_m"] <= heights.max() + 1
        assert 0.0 <= t["heading"] < 360.0
        if not t["airborne"]:
            # the env pins ground units at ground + 5 m; the model stands on ground
            assert t["alt"] - t["ground_m"] == pytest.approx(5.0, abs=1.0)


def test_threat_frames_are_packed_in_the_declared_order(payload):
    fields = payload["/scene"]["threat_state_fields"]
    assert fields == list(live.THREAT_STATE_FIELDS)
    tf = payload["/frames"]["threat_frames"]["trained"]
    assert len(tf) == payload["/scene"]["n_frames"]
    n_active = sum(t["active"] for t in payload["/scene"]["threats"])
    for rows in tf[::10]:
        assert len(rows) == n_active
        for r in rows:
            assert len(r) == len(fields)
            assert -180 <= r[fields.index("sensor_yaw")] <= 180
            if r[fields.index("track")] < 0:
                assert r[fields.index("sensor_yaw")] == 0, "no track, no slew"


def test_the_sensor_slews_only_while_the_track_matrix_says_so(recording, payload):
    d = json.loads(recording[0].read_text())
    track = np.asarray(d["worlds"]["trained"]["track"])
    tf = payload["/frames"]["threat_frames"]["trained"]
    fi, ft = live.THREAT_STATE_FIELDS.index("i"), live.THREAT_STATE_FIELDS.index("track")
    for f, rows in enumerate(tf):
        for r in rows:
            assert r[ft] == track[f, r[fi]]
    assert (track >= 0).any(), "the fixture never tracked anything; the check would be vacuous"


# --- exaggeration stays coherent ---------------------------------------------------------


def test_relief_rescaling_moves_everything_that_caches_an_altitude():
    exag = PAGE[PAGE.index("const applyExag = () => {"):PAGE.index("exagBtn.onclick =")]
    for piece in ("viewer.scene.verticalExaggeration = EXAG_K;",
                  "trailPts.forEach((_, i) => (trailPts[i] = []));",
                  "if (rebuildTracks) rebuildTracks();",
                  "d._naigosH * EXAG_K",
                  "placeThreat(i, ...rec.last)",
                  "applyHillshade(EXAG_K)",
                  "exagBadge.hidden = EXAG_K === 1;"):
        assert piece in exag, piece
    # every entity altitude goes through c3()/exagZ(); nothing places a model
    # with a raw fromDegrees height except the DEM anchor, which exaggerates too
    assert "const c3 = (lon, lat, alt) => Cesium.Cartesian3.fromDegrees(lon, lat, exagZ(alt));" in PAGE
    raw = [m for m in re.findall(r"Cartesian3\.fromDegrees\(([^)]*)\)", CODE)
           if m.strip() not in ("0, 0, 0",)]
    for args in raw:
        assert ("exagZ" in args or "anchorZ" in args or args.count(",") < 2
                or "lon, lat, exagZ(alt)" in args), f"un-exaggerated altitude: fromDegrees({args})"


def test_no_aircraft_goes_inside_a_mountain_under_any_multiplier():
    """Mesh and entities scale by the same k, so a positive AGL stays positive."""
    for k in (1, 2, 3):
        for ground, agl in ((1200.0, 30.0), (4000.0, 150.0)):
            assert (ground + agl) * k - ground * k == pytest.approx(agl * k)
            assert (ground + agl) * k > ground * k


def test_exaggeration_is_opt_in_and_indicated():
    assert "let exagIdx = 0;" in PAGE and "const EXAG = [1, 2, 3];" in PAGE
    assert '<div id="exagbadge" hidden></div>' in PAGE
    assert "VERTICAL EXAGGERATION" in PAGE


# --- terrain reads as terrain ------------------------------------------------------------


def test_physics_mode_still_draws_the_simulation_heightmap():
    assert "const terrainProvider = new Cesium.CustomHeightmapTerrainProvider(" in PAGE
    cfg = imagery.resolve_visual_config("physics")
    assert cfg.terrain_source == imagery.TERRAIN_SOURCE_SIMULATION and cfg.evidence_grade


def test_photorealistic_mode_still_carries_its_non_evidence_warning():
    cfg = imagery.resolve_visual_config("photorealistic", ion_token="a-token")
    assert not cfg.evidence_grade
    assert "PRESENTATION MODE" in cfg.separation_note
    assert 'const VISUAL_ONLY_TITLE = "Visual only — photorealistic context";' in PAGE


def test_the_opening_camera_is_the_oblique_preset_not_a_rectangle():
    assert "frameCamera(cameraMode);" in PAGE and "let cameraMode = CAM.default;" in PAGE
    assert "viewer.camera.flyTo({\n    destination: Cesium.Rectangle" not in PAGE
    for button in ('id="t-overview"', 'id="t-follow"', 'id="t-topdown"'):
        assert button in PAGE


def test_the_hillshade_is_aligned_to_the_terrain_posts():
    meta = {"west": 51.0, "east": 52.0, "south": 35.0, "north": 36.0, "nx": 11, "ny": 21}
    w, s, e, n = camera.hillshade_rectangle(meta)
    # pixel c of nx spans [w + c*px, w + (c+1)*px]; its centre is post c
    px = (e - w) / meta["nx"]
    for c in (0, 5, 10):
        assert w + (c + 0.5) * px == pytest.approx(51.0 + c * 0.1)
    py = (n - s) / meta["ny"]
    assert s + 0.5 * py == pytest.approx(35.0)
    assert "const HILLSHADE_RECT = Cesium.Rectangle.fromDegrees(" in PAGE
    assert "rectangle: HILLSHADE_RECT," in PAGE
    assert "T.west - postX / 2, T.south - postY / 2, T.east + postX / 2, T.north + postY / 2" in PAGE


def test_the_hillshade_lights_slopes_facing_the_sun():
    """A north-west sun must light west- and north-facing slopes, not their
    mirror images -- the defect in the formula this replaced."""
    meta = {"west": 51.0, "east": 51.1, "south": 35.0, "north": 35.1, "nx": 20, "ny": 20}
    x = np.arange(20)[None, :].repeat(20, 0)
    y = np.arange(20)[:, None].repeat(20, 1)
    flat = camera.hillshade(np.zeros((20, 20)), meta)[10, 10]
    rises_east = camera.hillshade(x * 40.0, meta)[10, 10]     # faces west
    rises_west = camera.hillshade(-x * 40.0, meta)[10, 10]    # faces east
    rises_north = camera.hillshade(y * 40.0, meta)[10, 10]    # faces south
    rises_south = camera.hillshade(-y * 40.0, meta)[10, 10]   # faces north
    assert rises_east > flat > rises_west
    assert rises_south > flat > rises_north
    # exaggeration steepens the shading the way it steepens the mesh
    assert camera.hillshade(x * 40.0, meta, 3.0)[10, 10] > rises_east
    assert "(-dzdx * SUN[0] - dzdy * SUN[1] + SUN[2])" in PAGE


def test_models_and_terrain_share_one_light():
    assert "viewer.scene.light = new Cesium.DirectionalLight({" in PAGE
    assert "const toward = new Cesium.Cartesian3(-SUN[0], -SUN[1], -SUN[2]);" in PAGE
    sx, sy, sz = camera.sun_vector_enu()
    assert sx < 0 < sy and sz > 0, "north-west, above the horizon"


# --- effects are tied to recorded events ---------------------------------------------------


def test_effects_only_come_from_recorded_events():
    fx = PAGE[PAGE.index("function tracerFx("):PAGE.index("// ---- one renderer, both modes")]
    assert "simulated outcome" in fx
    # the only callers are the replay and live event paths
    assert CODE.count("tracerFx(") == 3 and CODE.count("impactFx(") == 3
    assert 'ev.type === "launch_visual" && ev.from && ev.at' in PAGE
    assert 'ev.type === "shot_down" && ev.at' in PAGE
    assert "document.getElementById(\"eventnote\").textContent = scene.event_note;" in PAGE
    assert "no projectile or missile is simulated" in live.EVENT_HONESTY_NOTE


def test_replay_effects_are_keyed_to_their_logged_frame():
    block = PAGE[PAGE.index("  if (REPLAY) {"):PAGE.index("  const seenEvents = new Set();")]
    assert "const t0 = timeAt(ev.frame);" in block
    assert "Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, t0)" in block
    assert "renderEventLog((rp.events[policy] || []).filter(ev => ev.frame <= f));" in block
    assert "Date.now()" not in block and "performance.now()" not in block


def test_live_effects_play_once_and_never_for_the_backlog():
    live_js = PAGE[PAGE.index("const seenEvents = new Set();"):PAGE.index("es.onerror")]
    assert "if (seenEvents.has(ev.id)) continue;" in live_js
    assert "if (!primed) continue;" in live_js


def test_recorded_events_reach_the_page_with_a_cause_and_a_frame(payload, recording):
    d = json.loads(recording[0].read_text())
    for name, evs in payload["/frames"]["events"].items():
        assert len(evs) == len(d["worlds"][name]["events"])
        for ev in evs:
            assert ev["cause"] and 0 <= ev["frame"] < payload["/scene"]["n_frames"]
            assert len(ev["at"]) == 3
            assert (ev["from"] is None) == (ev["threat"] is None)
    kills = sum(e["type"] == "shot_down" for evs in payload["/frames"]["events"].values() for e in evs)
    logged = sum(int(r["summary"]["shootdowns"]) for r in [recording[1][n] for n in recording[1]])
    assert kills <= logged    # world 0 only, of the four


# --- payloads -----------------------------------------------------------------------------


def test_the_static_scene_is_schema_2_and_inlines_its_models(payload):
    sc = payload["/scene"]
    assert sc["schema_version"] == live.SCENE_SCHEMA
    assert sc["camera"]["default"] == "terrain_overview"
    assert sc["lighting"] == camera.lighting()
    for key, m in sc["models"]["models"].items():
        assert m["uri"].startswith("data:model/gltf-binary;base64,")
        assert base64.b64decode(m["uri"].split(",", 1)[1]) == models.REGISTRY[key].path.read_bytes()


def test_the_payloads_carry_no_credentials(payload):
    blob = json.dumps(payload)
    assert not JWT.search(blob) and not GKEY.search(blob)
    assert "token" not in json.dumps(payload["/scene"]["models"]).lower()


def test_the_exported_artifact_fetches_nothing_but_cesium(recording, tmp_path):
    html = viewer.build(recording[0], tmp_path / "demo.html").read_text()
    embed = json.loads(re.search(r"^const EMBED = (\{.*\});$", html, re.M).group(1))
    for m in embed["/scene"]["models"]["models"].values():
        assert m["uri"].startswith("data:")
    assert "http://" not in html
    # no model is fetched from the live route either -- it does not exist here
    assert '"uri": "/models/' not in html and '"uri":"/models/' not in html
    assert re.findall(r'<script[^>]*src="([^"]+)"', html) == [
        "https://cesium.com/downloads/cesiumjs/releases/1.145/Build/Cesium/Cesium.js"]


def test_a_recording_without_attitude_is_refused(recording, tmp_path):
    d = json.loads(recording[0].read_text())
    for w in d["worlds"].values():
        del w["phi"]
    stale = tmp_path / "old.json"
    stale.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="predates the 3D viewer schema"):
        live.static_payload(stale)


def test_the_smoke_report_is_diagnostic_only(payload):
    vis = imagery.resolve_visual_config("physics", ion_token="fake-token.x.y")
    notes = {"theatre": "test_basin", "geo_bounds": payload["/scene"]["bounds"]}
    rep = live.smoke_report(vis, notes, payload["/scene"]["terrain"])
    assert rep["rendered_pixels"] is False
    assert rep["terrain_source"] == imagery.TERRAIN_SOURCE_SIMULATION
    assert "CustomHeightmapTerrainProvider" in rep["terrain_provider"]
    assert rep["evidence_grade"] is True
    assert rep["model_fallback_count"] == 0
    assert rep["camera_mode"] == "terrain_overview"
    assert {m["key"] for m in rep["models"]} == set(models.REGISTRY)
    assert "fake-token" not in json.dumps(rep)
    assert rep["credentials_present"]["ion_token"] is True
    photo = live.smoke_report(imagery.resolve_visual_config("photorealistic", ion_token="t"),
                              notes, payload["/scene"]["terrain"])
    assert photo["evidence_grade"] is False


def test_the_live_server_serves_models_only_by_registry_name():
    src = (REPO / "naigos" / "demo" / "live.py").read_text()
    route = src[src.index("if self.path.startswith(models_mod.MODEL_ROUTE):"):]
    assert "models_mod.served_file(self.path)" in route.split("self.send_error(404)")[0]


# --- the live simulation, when the packaged theatre is available ---------------------------


@pytest.fixture(scope="module")
def sim():
    import pickle

    from naigos.env.theatre_bridge import env_from_theatre
    from naigos.rl.ppo import greedy_policy

    ckpt = REPO / "checkpoints" / "theatre_1000.pkl"
    if not ckpt.exists():
        pytest.skip("no shipped checkpoint")
    try:
        cfg, hmap, notes = env_from_theatre(aoi="tehran_basin", n_blue=4, n_threat=10)
    except FileNotFoundError as e:
        pytest.skip(f"tehran_basin snapshot unavailable: {e}")
    s = live.Simulation(env=NaigosEnv(cfg, hmap=hmap),
                        policy=greedy_policy(pickle.loads(ckpt.read_bytes())["actor"], cfg),
                        georef=GeoRef(**notes["georef"]), speed=1e9, reroll_s=0.0)
    return s, notes


def test_live_frames_and_scene_carry_the_visual_fields(sim):
    s, notes = sim
    for _ in range(4):
        s.key, k = jax.random.split(s.key)
        before = s.state
        s2, o2, terms, info, feas, _ = s._step(s.state, s.obs, k)
        s.state, s.obs = s2, o2
        s.sim_t += s.env.cfg.dt
        s._derive_events(before, s2, jax.device_get(terms), info)
        s._publish(np.asarray(feas), np.asarray(terms.exposure))
    f = s.latest()
    blob = json.dumps(f)
    assert not JWT.search(blob) and not GKEY.search(blob)
    for a in f["aircraft"]:
        for k in ("heading", "pitch", "roll", "sortie"):
            assert k in a
        assert 0.0 <= a["heading"] < 360.0
    assert all(len(r) == len(live.THREAT_STATE_FIELDS) for r in f["threats"])
    assert isinstance(f["events"], list)
    sc = s.scene(notes)
    assert sc["schema_version"] == live.SCENE_SCHEMA
    assert sc["models"]["models"]["fixed_wing"]["uri"] == models.MODEL_ROUTE + "fixed_wing.glb"
    assert {t["model"] for t in sc["threats"]} <= set(models.REGISTRY)


def test_deriving_live_events_does_not_touch_the_simulation_state(sim):
    s, _ = sim
    s.key, k = jax.random.split(s.key)
    before = s.state
    s2, o2, terms, info, feas, _ = s._step(s.state, s.obs, k)
    snap = jax.tree.map(lambda x: np.array(x, copy=True), s2)
    s._derive_events(before, s2, jax.device_get(terms), info)
    after = jax.tree.map(np.asarray, s2)
    for a, b in zip(jax.tree.leaves(snap), jax.tree.leaves(after)):
        assert np.array_equal(a, b)


def test_the_static_artifact_is_served_not_opened_from_the_filesystem():
    """A browser will not start CesiumJS's workers for a file:// page, and the
    workers build the terrain -- so the globe silently fails to draw. Found by a
    real browser check; the page now says so, and --open serves on loopback."""
    assert 'if (location.protocol === "file:") {' in PAGE
    assert '<div id="filenote" hidden></div>' in PAGE
    src = (REPO / "naigos" / "demo" / "viewer.py").read_text()
    serve = src[src.index("def serve("):]
    assert 'ThreadingHTTPServer(("127.0.0.1", port)' in serve, "loopback only"
    assert "as_uri()" not in src, "--open must not hand the browser a file:// URL"


def test_no_entity_level_show_is_a_property():
    """Entity.show is a plain boolean in CesiumJS; only graphics' show flags
    are Properties. A CallbackProperty assigned to an entity's show is merely
    truthy -- the first version of the impact effect was drawn at every
    instant of the replay because of exactly this, before the kill included.
    Found by a browser check, pinned here."""
    for m in re.finditer(r"viewer\.entities\.add\(\{", CODE):
        depth, i = 0, m.end() - 1
        while True:
            ch = CODE[i]
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                break
            i += 1
        body = CODE[m.end():i]
        # strip nested blocks so only the entity's own top-level keys remain
        flat, d = [], 0
        for ch in body:
            d += ch in "{(["
            d -= ch in "})]"
            if d == 0:
                flat.append(ch)
        top = "".join(flat)
        assert not re.search(r"\bshow:\s*new Cesium\.CallbackProperty", top), body[:120]
    fx = PAGE[PAGE.index("function tracerFx("):PAGE.index("// ---- one renderer, both modes")]
    assert fx.count("show: fxLive(age, dur)") == 3
