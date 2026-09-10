"""The static replay export must draw the scene the rollout actually happened in.

The failure mode this guards against is the one that already bit this repo once
(see DEVLOG, the north-south mirror at the env/data seam): the terrain still
looks like terrain, the tracks still look like tracks, and the aircraft are
silently flying over a mirrored or misaligned heightmap. So rather than checking
that a file was written, these tests re-implement the viewer's own bilinear
lookup and assert it reproduces the AGL the environment logged.

Two lookups, because there are two grids. `_enu_ground` is the recording's own
`terrain.heights` in local ENU metres -- what `demo.json` carries. `_page_ground`
is `sampleGrid` from `assets/cesium.html`, over the lat/lon grid the export
embeds at `/terrain`. The second is the one the browser actually draws, so it is
the one that has to agree with the logged AGL.

`naigos.demo.viewer` used to emit a separate three.js page that rendered the same
recordings in a bare ENU box. It is gone: the export is now `cesium.html` with
its routes inlined, so the artifact and the live globe are one renderer.
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
from naigos.demo import replay, viewer
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.networks import Actor

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=120)
REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()

#: A real UTM zone with the AOI origin inside it, so `GeoRef` round-trips
#: through pyproj exactly as it does for a cited theatre. Tehran's zone.
NOTES = {
    "theatre": "test_basin",
    "georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0, "origin_northing_m": 3_950_000.0},
}


@pytest.fixture(scope="module")
def demo_json(tmp_path_factory):
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=4, seed=7)

    # The georef and the bounds the recording needs to be placeable at all.
    # `env_from_theatre` supplies these for a real AOI; a synthetic env has no
    # ground truth, so they are derived from the georef itself.
    georef = GeoRef(**NOTES["georef"])
    lon, lat = georef.to_wgs84(
        np.array([0.0, CFG.terrain.extent_x]), np.array([0.0, CFG.terrain.extent_y]))
    notes = dict(NOTES, geo_bounds={
        "west": float(lon[0]), "east": float(lon[1]),
        "south": float(lat[0]), "north": float(lat[1]),
    })

    out = tmp_path_factory.mktemp("demo") / "demo.json"
    replay.to_json(results, CFG, out, env=env, notes=notes)
    return out


@pytest.fixture(scope="module")
def exported(demo_json, tmp_path_factory):
    return viewer.build(demo_json, tmp_path_factory.mktemp("html") / "demo.html")


def _embedded(html_path: Path) -> dict:
    """Pull the EMBED blob back out of the artifact, the way the page sees it."""
    html = html_path.read_text()
    m = re.search(r"^const EMBED = (\{.*\});$", html, re.M)
    assert m, "the EMBED substitution point was not filled in"
    return json.loads(m.group(1))


def _enu_ground(T, x_m, y_m):
    """The recording's own heightmap, in local ENU metres."""
    cell = T["cell_m"]
    nx, ny, h = T["nx"], T["ny"], T["heights"]
    fx = min(max(x_m / cell, 0.0), nx - 1)
    fy = min(max(y_m / cell, 0.0), ny - 1)
    x0, y0 = int(math.floor(fx)), int(math.floor(fy))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    a, b = h[y0 * nx + x0], h[y0 * nx + x1]
    c, d = h[y1 * nx + x0], h[y1 * nx + x1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def _page_ground(meta, heights, lon_deg, lat_deg):
    """Byte-for-byte the `sampleGrid` in assets/cesium.html, inside the AOI."""
    d_lon = meta["east"] - meta["west"]
    d_lat = meta["north"] - meta["south"]
    u = (lon_deg - meta["west"]) / d_lon
    v = (lat_deg - meta["south"]) / d_lat
    cu, cv = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
    fx, fy = cu * (meta["nx"] - 1), cv * (meta["ny"] - 1)
    x0, y0 = int(math.floor(fx)), int(math.floor(fy))
    x1, y1 = min(x0 + 1, meta["nx"] - 1), min(y0 + 1, meta["ny"] - 1)
    tx, ty = fx - x0, fy - y0
    a, b = heights[y0 * meta["nx"] + x0], heights[y0 * meta["nx"] + x1]
    c, d = heights[y1 * meta["nx"] + x0], heights[y1 * meta["nx"] + x1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


# --- the recording ----------------------------------------------------------------------


def test_export_carries_the_whole_scene(demo_json):
    d = json.loads(demo_json.read_text())
    for k in ("terrain", "threats", "objective", "worlds", "summaries", "dt_s", "extent_m"):
        assert k in d, f"viewer export is missing {k!r}"
    T = d["terrain"]
    assert len(T["heights"]) == T["nx"] * T["ny"]
    assert len(d["threats"]) == CFG.n_threat


def test_viewer_terrain_lookup_reproduces_the_logged_agl(demo_json):
    """THE orientation test. If the heightmap were flipped or transposed on the
    way out, this is where it shows up -- everything else would still look
    plausible."""
    d = json.loads(demo_json.read_text())
    T = d["terrain"]
    w = d["worlds"]["trained"]
    worst = 0.0
    for f in range(0, len(w["pos"]), 3):
        for b in range(d["n_blue"]):
            if not w["alive"][f][b]:
                continue
            x, y, z = w["pos"][f][b]
            worst = max(worst, abs((z - _enu_ground(T, x, y)) - w["agl"][f][b]))
    # the export rounds positions and heights to 0.1 m
    assert worst < 1.0, f"viewer terrain disagrees with the env by up to {worst:.2f} m"


def test_a_transposed_heightmap_would_be_caught(demo_json):
    """Proves the test above has teeth rather than passing vacuously."""
    d = json.loads(demo_json.read_text())
    T = dict(d["terrain"])
    h = np.asarray(T["heights"]).reshape(T["ny"], T["nx"])
    T["heights"] = np.flipud(h).reshape(-1).tolist()  # mirror north-south
    w = d["worlds"]["trained"]
    worst = 0.0
    for f in range(0, len(w["pos"]), 3):
        for b in range(d["n_blue"]):
            if not w["alive"][f][b]:
                continue
            x, y, z = w["pos"][f][b]
            worst = max(worst, abs((z - _enu_ground(T, x, y)) - w["agl"][f][b]))
    assert worst > 10.0, "a mirrored heightmap should be obvious, so the check is not vacuous"


def test_tracks_stop_where_the_aircraft_was_lost(demo_json):
    """The replay must not draw an aircraft that is gone."""
    d = json.loads(demo_json.read_text())
    for name, w in d["worlds"].items():
        alive = np.asarray(w["alive"])
        for b in range(d["n_blue"]):
            col = alive[:, b]
            if col.all():
                continue
            first_dead = int(np.argmin(col))
            assert not col[first_dead:].any(), f"{name}: aircraft {b} came back from the dead"


def test_frame_count_and_clock_agree_with_the_env(demo_json):
    d = json.loads(demo_json.read_text())
    n = len(d["worlds"]["trained"]["pos"])
    assert n == math.ceil(CFG.max_steps / 2)  # default stride 2
    assert d["dt_s"] == pytest.approx(CFG.dt * 2)


# --- the artifact -----------------------------------------------------------------------


def test_the_export_is_the_cesium_page_with_nothing_left_to_fetch(exported):
    html = exported.read_text()
    assert "/*__EMBED__*/null" not in html, "the data placeholder was not substituted"
    assert "const EMBED = {" in html
    # Exactly one external script, and it is the same pinned CesiumJS build the
    # live server serves. There is one renderer now, not two.
    srcs = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert len(srcs) == 1 and "cesium.com/downloads/cesiumjs" in srcs[0]
    assert "three.min.js" not in html, "the three.js viewer is gone; nothing may reload it"
    assert "http://" not in html


def test_the_embed_stands_in_for_exactly_the_routes_the_page_fetches(exported):
    """Keyed by route, so the static artifact and a served session hand the
    renderer the same three shapes."""
    embed = _embedded(exported)
    assert set(embed) == {"/scene", "/frames", "/terrain"}
    assert embed["/scene"]["mode"] == "replay"
    assert embed["/scene"]["n_blue"] == CFG.n_blue
    assert embed["/frames"]["policies"]
    assert isinstance(embed["/terrain"], str)


def test_the_embedded_terrain_reproduces_the_logged_agl(exported):
    """The end-to-end orientation check: the lat/lon grid the BROWSER samples,
    against the AGL the env logged. The ENU test above checks the recording; this
    checks the resample, the georef and the page's own lookup together."""
    embed = _embedded(exported)
    meta = embed["/scene"]["terrain"]
    heights = np.frombuffer(base64.b64decode(embed["/terrain"]), dtype="<i2")
    assert heights.size == meta["nx"] * meta["ny"]

    frames = embed["/frames"]["frames"]["trained"]
    errs = []
    for f in range(0, len(frames), 3):
        for a in frames[f]:
            if not a["alive"]:
                continue
            errs.append(abs((a["alt"] - _page_ground(meta, heights, a["lon"], a["lat"])) - a["agl"]))
    assert errs, "no live aircraft to check"
    # The served grid is a 512x512 resample of the env grid rounded to the metre,
    # so agreement is close but not exact; the mirrored control below is what
    # gives the bound meaning.
    assert np.mean(errs) < 25.0, f"mean AGL disagreement {np.mean(errs):.1f} m"


def test_a_mirrored_embedded_grid_would_be_caught(exported):
    """The control for the test above, on the grid the browser actually reads."""
    embed = _embedded(exported)
    meta = embed["/scene"]["terrain"]
    heights = np.frombuffer(base64.b64decode(embed["/terrain"]), dtype="<i2")
    flipped = np.flipud(heights.reshape(meta["ny"], meta["nx"])).reshape(-1)

    frames = embed["/frames"]["frames"]["trained"]
    errs = []
    for f in range(0, len(frames), 3):
        for a in frames[f]:
            if a["alive"]:
                errs.append(abs((a["alt"] - _page_ground(meta, flipped, a["lon"], a["lat"]))
                                - a["agl"]))
    assert np.mean(errs) > 100.0, "a mirrored grid should be obvious, so the bound is not vacuous"


def test_the_scene_describes_the_recordings_threats_not_a_live_envs(exported, demo_json):
    """A recording's envelopes must come from the recording. The served replay
    mode used to patch terrain over the LIVE env's scene, so the domes drawn were
    whichever sites the running simulation happened to hold."""
    embed = _embedded(exported)
    d = json.loads(demo_json.read_text())
    threats = embed["/scene"]["threats"]
    assert len(threats) == len(d["threats"])
    for i, (t, rec) in enumerate(zip(threats, d["threats"])):
        assert t["i"] == i
        assert t["label"] == rec["label"]
        assert t["lethal_m"] == rec["lethal_m"]
        assert t["active"] == rec["active"]

    # ...and they are placed at the positions the recording logged, not at a
    # live env's. Checked by round-tripping one back through the georef.
    georef = GeoRef(**d["georef"])
    tp = np.asarray(d["worlds"]["trained"]["threat_pos"])[0]
    lon, lat = georef.to_wgs84(tp[:, 0], tp[:, 1])
    for i, t in enumerate(threats):
        assert t["lon"] == pytest.approx(float(lon[i]), abs=1e-5)
        assert t["lat"] == pytest.approx(float(lat[i]), abs=1e-5)


def test_the_artifact_carries_no_credential(exported):
    """It gets committed. The export resolves its visual config with no token and
    no key, which lands on keyless OSM over the simulation's own DEM."""
    html = exported.read_text()
    embed = _embedded(exported)
    assert "const ION_TOKEN = null;" in html
    assert "const GOOGLE_API_KEY = null;" in html
    assert not re.search(r"eyJ[A-Za-z0-9_-]{20,}\.", html), "a JWT-shaped token is in the artifact"
    assert not re.search(r"AIza[0-9A-Za-z_-]{35}", html)
    # ...and it is still the evidence-grade posture.
    visual = re.search(r"^const VISUAL = (\{.*\});$", html, re.M)
    assert visual and json.loads(visual.group(1))["evidence_grade"] is True
    assert embed["/scene"]["terrain"]["grid"]


def test_the_three_js_viewer_is_gone():
    """One renderer. The asset is deleted, nothing loads the library, and no
    module still reaches for the template it used to substitute into."""
    assert not (REPO / "naigos" / "demo" / "assets" / "viewer.html").exists()
    assert re.search(r"three(\.min)?\.js|/three\.js/", PAGE, re.I) is None
    offenders = []
    for path in sorted((REPO / "naigos").rglob("*.py")):
        text = path.read_text()
        if "viewer.html" in text or "__NAIGOS_DATA__" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"still referencing the deleted three.js viewer: {offenders}"
