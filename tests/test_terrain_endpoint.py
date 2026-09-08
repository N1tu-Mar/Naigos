"""The globe must render the surface the simulation actually used.

Gap E-9 was that Cesium drew either a bare ellipsoid or Cesium World Terrain,
neither of which is the DEM line-of-sight was computed against. The fix serves
the env's own heightmap. These tests assert that what is served reproduces
`naigos.env.terrain.sample_height`, and -- crucially -- that the check cannot
pass vacuously: a deliberately transposed grid must fail it.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pyproj")

from naigos.data.geodetic import GeoRef  # noqa: E402
from naigos.demo.live import Simulation, replay_payload  # noqa: E402
from naigos.env.flight_env import NaigosEnv  # noqa: E402
from naigos.env.terrain import sample_height  # noqa: E402
from naigos.rl.ppo import greedy_policy  # noqa: E402

CKPT = Path("checkpoints/theatre_1000.pkl")


@pytest.fixture(scope="module")
def sim_and_notes():
    from naigos.env.theatre_bridge import env_from_theatre

    if not CKPT.exists():
        pytest.skip("no shipped checkpoint")
    try:
        cfg, hmap, notes = env_from_theatre(aoi="tehran_basin", n_blue=4, n_threat=8, cell_m=500.0)
    except FileNotFoundError as e:
        pytest.skip(f"tehran_basin snapshot unavailable: {e}")
    sim = Simulation(
        env=NaigosEnv(cfg, hmap=hmap),
        policy=greedy_policy(pickle.loads(CKPT.read_bytes())["actor"], cfg),
        georef=GeoRef(**notes["georef"]),
        reroll_s=0.0,
    )
    return sim, cfg, hmap, notes


def _decode(body, meta):
    return np.frombuffer(body, dtype="<i2").reshape(meta["ny"], meta["nx"]).astype(float)


def _bilinear(Z, meta, lon, lat):
    """Byte-for-byte the `sampleGrid` lookup in assets/cesium.html."""
    fx = (lon - meta["west"]) / (meta["east"] - meta["west"]) * (meta["nx"] - 1)
    fy = (lat - meta["south"]) / (meta["north"] - meta["south"]) * (meta["ny"] - 1)
    fx = np.clip(fx, 0, meta["nx"] - 1)
    fy = np.clip(fy, 0, meta["ny"] - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, meta["nx"] - 1), np.minimum(y0 + 1, meta["ny"] - 1)
    tx, ty = fx - x0, fy - y0
    return (Z[y0, x0] * (1 - tx) + Z[y0, x1] * tx) * (1 - ty) + (
        Z[y1, x0] * (1 - tx) + Z[y1, x1] * tx
    ) * ty


def _errors(Z, meta, sim, cfg, hmap, n=4000):
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, cfg.terrain.extent_x, n)
    ys = rng.uniform(0, cfg.terrain.extent_y, n)
    modelled = np.asarray(sample_height(hmap, cfg.terrain, jnp.asarray(xs), jnp.asarray(ys)))
    lon, lat = sim.georef.to_wgs84(xs, ys)
    return np.abs(_bilinear(Z, meta, lon, lat) - modelled)


def test_payload_decodes_at_the_declared_shape(sim_and_notes):
    sim, cfg, _, notes = sim_and_notes
    body, meta = sim.terrain_grid(notes)
    assert len(body) == meta["nx"] * meta["ny"] * 2
    Z = _decode(body, meta)
    assert Z.shape == (meta["ny"], meta["nx"])
    assert meta["cell_m"] == cfg.terrain.cell
    assert meta["min_m"] < meta["max_m"]


def test_served_surface_reproduces_the_modelled_one(sim_and_notes):
    """THE test. If these diverge, the globe is illustrating terrain masking
    rather than showing it."""
    sim, cfg, hmap, notes = sim_and_notes
    body, meta = sim.terrain_grid(notes)
    d = _errors(_decode(body, meta), meta, sim, cfg, hmap)
    assert d.mean() < 5.0, f"mean divergence {d.mean():.1f} m"
    assert np.percentile(d, 95) < 25.0
    assert d.max() < 100.0, f"max divergence {d.max():.1f} m"


def test_a_transposed_grid_would_be_caught(sim_and_notes):
    """Proves the check above is not vacuous -- the same failure class that hit
    the env/data seam once already (see docs/DEVLOG.md)."""
    sim, cfg, hmap, notes = sim_and_notes
    body, meta = sim.terrain_grid(notes)
    Z = _decode(body, meta)
    d = _errors(np.flipud(Z), meta, sim, cfg, hmap)
    assert d.mean() > 50.0, "a north-south mirrored surface should be obvious"


def test_terrain_is_cached_not_rebuilt(sim_and_notes):
    sim, _, _, notes = sim_and_notes
    a = sim.terrain_grid(notes)
    b = sim.terrain_grid(notes)
    assert a[0] is b[0]


def test_scene_advertises_the_terrain_grid(sim_and_notes):
    sim, cfg, _, notes = sim_and_notes
    sc = sim.scene(notes)
    json.dumps(sc)
    assert sc["mode"] == "live"
    for k in ("west", "east", "south", "north", "nx", "ny", "cell_m", "grid"):
        assert k in sc["terrain"]


def test_frame_carries_the_tracker_ray(sim_and_notes):
    """The LOS overlay needs the strongest tracker and its terrain clearance."""
    sim, cfg, _, notes = sim_and_notes
    for _ in range(3):
        sim.key, k = jax.random.split(sim.key)
        s2, o2, terms, info, feas, _ = sim._step(sim.state, sim.obs, k)
        sim.state, sim.obs = s2, o2
        sim._publish(np.asarray(feas), np.asarray(terms.exposure))
    f = sim.latest()
    json.dumps(f)
    for a in f["aircraft"]:
        lon, lat, alt, clearance, pd = a["tracker"]
        assert math.isfinite(clearance)
        assert 0.0 <= pd <= 1.0
        assert 50.0 < lon < 53.0 and 34.0 < lat < 37.0
        # objective must carry its altitude: at height 0 the marker sat ~1200 m
        # under the Tehran basin floor once real terrain arrived
        assert len(a["objective"]) == 3 and a["objective"][2] > 500.0


# ------------------------------------------------------------------- replay ---
def test_replay_refuses_an_untagged_recording(tmp_path, sim_and_notes):
    """A recording is positions in a LOCAL frame. Replaying it against the wrong
    theatre silently relocates the whole sortie to another continent."""
    sim, cfg, _, _ = sim_and_notes
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"worlds": {}, "summaries": {}}))
    with pytest.raises(SystemExit, match="predates theatre tagging"):
        replay_payload(p, sim.georef, cfg)


def test_replay_refuses_a_theatre_mismatch(tmp_path, sim_and_notes):
    sim, cfg, _, _ = sim_and_notes
    p = tmp_path / "owens.json"
    p.write_text(json.dumps({"theatre": "owens_valley", "worlds": {}, "summaries": {}}))
    with pytest.raises(SystemExit, match="recorded on theatre"):
        replay_payload(p, sim.georef, cfg, aoi="tehran_basin")


def test_recorded_rollout_round_trips_into_geodetic_frames(sim_and_notes):
    """The shipped demo.json must place inside its own theatre's bounds."""
    sim, cfg, _, _ = sim_and_notes
    p = Path("runs/demo/demo.json")
    if not p.exists():
        pytest.skip("no recorded rollout; run naigos.demo.replay")
    d = json.loads(p.read_text())
    rp = replay_payload(p, GeoRef(**d["georef"]), cfg, aoi=d["theatre"])
    b = d["geo_bounds"]
    for name in rp["policies"]:
        for ac in rp["frames"][name][0]:
            assert b["west"] - 0.3 <= ac["lon"] <= b["east"] + 0.3
            assert b["south"] - 0.3 <= ac["lat"] <= b["north"] + 0.3
