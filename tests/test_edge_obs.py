"""Edge-distance ego features: behind a flag, appended, body frame, absolute metres."""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.obs import POS_SCALE, edge_distances
from naigos.rl.checkpoint import obs_config_from_blob

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=40)
SHIPPED = Path(__file__).resolve().parents[1] / "checkpoints" / "theatre_1000.pkl"

# a 100 x 40 km box
X_MIN, X_MAX, Y_MIN, Y_MAX = 0.0, 100_000.0, 0.0, 40_000.0


def _obs(cfg):
    return NaigosEnv(cfg).reset(jax.random.PRNGKey(0))[1]


def test_flag_off_is_the_old_ten_wide_ego():
    assert CFG.ego_dim == 10
    off = _obs(CFG)
    on = _obs(CFG.replace(obs_edge_features=True))
    assert off.ego.shape == (CFG.n_blue, 10)
    np.testing.assert_array_equal(np.asarray(off.ego), np.asarray(on.ego[:, :10]))


def test_flag_on_appends_four():
    cfg = CFG.replace(obs_edge_features=True)
    assert cfg.ego_dim == 14
    o = _obs(cfg)
    assert o.ego.shape == (cfg.n_blue, 14)
    assert bool(jnp.all(jnp.isfinite(o.ego)))
    assert bool(jnp.all((o.ego[:, 10:] >= 0.0) & (o.ego[:, 10:] <= 2.0)))


def test_heading_east_and_north():
    x, y = 20_000.0, 15_000.0
    pos = jnp.array([[x, y]])
    east = edge_distances(pos, jnp.array([0.0]), X_MIN, X_MAX, Y_MIN, Y_MAX)[0]
    want = np.clip(np.array([X_MAX - x, Y_MAX - y, y - Y_MIN, x - X_MIN]) / POS_SCALE, 0.0, 2.0)
    np.testing.assert_allclose(np.asarray(east), want, rtol=1e-5)

    # +90 deg: forward is north, left is west, right is east, back is south
    north = edge_distances(pos, jnp.array([jnp.pi / 2]), X_MIN, X_MAX, Y_MIN, Y_MAX)[0]
    f, l, r, b = want
    np.testing.assert_allclose(np.asarray(north), [l, b, f, r], rtol=1e-5)


def test_clipped_to_two():
    # on the west edge facing east, 100 km of room = exactly 2.0; a wider box clips
    pos = jnp.array([[0.0, 20_000.0]])
    d = edge_distances(pos, jnp.array([0.0]), X_MIN, X_MAX, Y_MIN, Y_MAX)[0]
    assert float(d[0]) == pytest.approx(2.0)
    assert float(d[3]) == 0.0
    wide = edge_distances(pos, jnp.array([0.0]), X_MIN, 300_000.0, Y_MIN, Y_MAX)[0]
    assert float(wide[0]) == 2.0


def test_outside_the_box_is_zero():
    pos = jnp.array([[-1_000.0, 20_000.0], [50_000.0, 41_000.0], [101_000.0, -5.0]])
    d = edge_distances(pos, jnp.array([0.0, 1.0, -2.0]), X_MIN, X_MAX, Y_MIN, Y_MAX)
    np.testing.assert_array_equal(np.asarray(d), np.zeros((3, 4)))


def test_shipped_checkpoint_has_no_edge_features():
    with open(SHIPPED, "rb") as f:
        blob = pickle.load(f)
    assert obs_config_from_blob(blob) == {"obs_edge_features": False}
    assert obs_config_from_blob({"actor": {}}) == {"obs_edge_features": False}
    assert obs_config_from_blob({"env_cfg": {"obs_edge_features": True}}) == {"obs_edge_features": True}
