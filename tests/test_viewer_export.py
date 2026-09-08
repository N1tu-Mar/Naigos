"""The 3D viewer must draw the scene the rollout actually happened in.

The failure mode this guards against is the one that already bit this repo once
(see DEVLOG, the north-south mirror at the env/data seam): the terrain still
looks like terrain, the tracks still look like tracks, and the aircraft are
silently flying over a mirrored or misaligned heightmap. So rather than checking
that a file was written, these tests re-implement the viewer's own bilinear
lookup and assert it reproduces the AGL the environment logged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.demo import replay, viewer
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.networks import Actor

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=120)


@pytest.fixture(scope="module")
def demo_json(tmp_path_factory):
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=4, seed=7)
    out = tmp_path_factory.mktemp("demo") / "demo.json"
    replay.to_json(results, CFG, out, env=env)
    return out


def _viewer_ground(T, x_m, y_m):
    """Byte-for-byte the `groundAt` in assets/viewer.html, in metres."""
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
            worst = max(worst, abs((z - _viewer_ground(T, x, y)) - w["agl"][f][b]))
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
            worst = max(worst, abs((z - _viewer_ground(T, x, y)) - w["agl"][f][b]))
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


def test_viewer_html_is_self_contained(demo_json, tmp_path):
    out = viewer.build(demo_json, tmp_path / "v.html")
    html = out.read_text()
    assert "__NAIGOS_DATA__" not in html, "the data placeholder was not substituted"
    assert "const DATA = {" in html
    # exactly one external script, and it is the pinned three.js CDN build
    import re

    srcs = re.findall(r'<script[^>]*src="([^"]+)"', html)
    assert srcs == ["https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js"]
    assert "http://" not in html


def test_frame_count_and_clock_agree_with_the_env(demo_json):
    d = json.loads(demo_json.read_text())
    n = len(d["worlds"]["trained"]["pos"])
    assert n == math.ceil(CFG.max_steps / 2)  # default stride 2
    assert d["dt_s"] == pytest.approx(CFG.dt * 2)
