"""Per-world play areas: a random rectangle inside the fixed DEM grid, behind a flag."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env.obs import edge_distances

# non-square on purpose (94.5 x 70.5 km) so an x/y mix-up cannot pass
CFG = EnvConfig(n_blue=4, n_threat=8, n_threat_active=6, max_steps=220, terrain=TerrainConfig(nx=64, ny=48, cell=1_500.0))
ON = CFG.replace(map_randomize=True)
EX, EY = CFG.terrain.extent_x, CFG.terrain.extent_y


def _policy(obs, key):
    # 0,1 steer at the objective; 2 holds a gentle left bank; 3 flies straight and fast
    steer = jnp.clip(2.0 * obs.ego[:, 4], -1.0, 1.0)
    bank = jnp.stack([steer[0], steer[1], jnp.float32(0.15), jnp.float32(0.0)])
    return jnp.stack([bank, jnp.zeros(4), jnp.ones(4)], axis=-1)


def _resets(cfg, n=256):
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    return jax.jit(jax.vmap(NaigosEnv(cfg).reset))(keys)[0]


def test_flag_off_bounds_are_the_full_grid():
    assert CFG.map_randomize is False
    st = _resets(CFG, 8)
    np.testing.assert_array_equal(np.asarray(st.bounds), np.tile([0.0, EX, 0.0, EY], (8, 1)).astype(np.float32))


def test_flag_off_rollout_matches_the_grid_extent_env():
    # Golden values: this exact rollout on the env as it was before bounds existed
    # (git 3d08ea9, geometry hard-wired to the grid extent). One aircraft is lost
    # to the threats, one to terrain/threats, one flies off the map, one arrives.
    final, traj = jax.jit(lambda k: NaigosEnv(CFG).rollout(k, _policy))(jax.random.PRNGKey(7))
    want_pos = np.array(
        [
            [21684.238, 19999.357, 2909.4465],
            [17835.021, 28750.086, 2873.8325],
            [52762.27, 70596.35, 2679.7397],
            [82308.21, 42163.156, 3304.2095],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(final.air.pos), want_pos, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(final.alive), [False, False, False, True])
    np.testing.assert_array_equal(np.asarray(final.reached), [False, False, False, True])
    np.testing.assert_array_equal(np.asarray(traj["terms"].bounds_violation.sum(0)), [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(np.asarray(traj["terms"].edge_proximity.sum(0)), [0.0, 0.0, 4.340307, 0.0], rtol=1e-5)


def test_flag_on_boxes_are_valid_and_varied():
    st = _resets(ON)
    b = np.asarray(st.bounds, dtype=np.float64)
    x0, x1, y0, y1 = b.T
    w, h = x1 - x0, y1 - y0
    tol = 1.0  # m, float32 slack at ~1e5 m
    assert np.all(x0 >= -tol) and np.all(x1 <= EX + tol) and np.all(y0 >= -tol) and np.all(y1 <= EY + tol)
    assert np.all(w >= ON.map_min_extent_m - tol) and np.all(h >= ON.map_min_extent_m - tol)
    lo, hi = ON.map_aspect_range
    aspect = h / w
    clipped = (np.abs(h - ON.map_min_extent_m) < tol) | (np.abs(h - EY) < tol)
    assert np.all((aspect >= lo - 1e-4) & (aspect <= hi + 1e-4) | clipped)
    # clipping the height pushes the aspect out of range only in that direction
    assert np.all(aspect[clipped & (np.abs(h - EY) < tol)] <= hi + 1e-4)
    assert np.all(aspect[clipped & (np.abs(h - ON.map_min_extent_m) < tol)] >= lo - 1e-4)
    # genuinely varied, not a constant box
    assert w.std() > 5_000.0 and aspect.std() > 0.1 and x0.std() > 1_000.0
    assert not np.any(np.all(b == [0.0, EX, 0.0, EY], axis=-1))


def _inside(xy, bounds):
    b = np.asarray(bounds)[:, None, :]
    x, y = np.asarray(xy)[..., 0], np.asarray(xy)[..., 1]
    return (x >= b[..., 0]) & (x <= b[..., 1]) & (y >= b[..., 2]) & (y <= b[..., 3])


def test_flag_on_starts_and_objectives_inside_the_box():
    st = _resets(ON)
    assert np.all(_inside(st.air.pos[..., :2], st.bounds))
    assert np.all(_inside(st.objective[..., :2], st.bounds))
    # the start/objective inset is relative to the box, not the grid
    b = np.asarray(st.bounds)
    w = b[:, 1] - b[:, 0]
    np.testing.assert_allclose(np.asarray(st.air.pos[:, 0, 0]), b[:, 0] + ON.spawn_inset_frac * w, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(st.objective[:, 0, 0]), b[:, 0] + (1 - ON.spawn_inset_frac) * w, rtol=1e-5)


def test_respawn_stays_inside_the_box():
    env = NaigosEnv(ON)
    st = _resets(ON, 64)
    mask = jnp.ones((64, ON.n_blue), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(11), 64)
    st2 = jax.jit(jax.vmap(env.respawn))(st, keys, mask)
    np.testing.assert_array_equal(np.asarray(st2.bounds), np.asarray(st.bounds))
    assert np.all(_inside(st2.air.pos[..., :2], st2.bounds))
    assert np.all(_inside(st2.objective[..., :2], st2.bounds))
    st3 = jax.jit(jax.vmap(env.reroll_threats))(st2, keys)
    np.testing.assert_array_equal(np.asarray(st3.bounds), np.asarray(st.bounds))


# a 50 x 30 km box well inside the 94.5 x 70.5 km grid
BOX = jnp.array([20_000.0, 70_000.0, 15_000.0, 45_000.0], dtype=jnp.float32)


def _place(cfg, xy, psi):
    env = NaigosEnv(cfg)
    st, _ = env.reset(jax.random.PRNGKey(5))
    st = st._replace(bounds=BOX)
    g = st.hmap.max() + 3_000.0  # clear of the terrain everywhere
    pos = jnp.concatenate([jnp.asarray(xy, dtype=jnp.float32), jnp.full((cfg.n_blue, 1), g)], axis=-1)
    st = st._replace(
        air=st.air._replace(pos=pos, psi=jnp.asarray(psi, dtype=jnp.float32)),
        threats=st.threats._replace(active=jnp.zeros_like(st.threats.active)),
    )
    return env, st


def test_out_of_bounds_fires_at_the_box_edge_not_the_grid_edge():
    # 0: 2 km outside the box's east edge, still 22 km inside the grid, heading east
    # 1: 2 km outside the north edge, heading north
    # 2: centre of the box; 3: 1 km inside the west edge, heading west
    xy = [[72_000.0, 30_000.0], [45_000.0, 47_000.0], [45_000.0, 30_000.0], [21_000.0, 30_000.0]]
    psi = [0.0, jnp.pi / 2, 0.0, jnp.pi]
    act = jnp.zeros((CFG.n_blue, 3))

    env, st = _place(ON, xy, psi)
    st2, _, terms, _, _ = env.step(st, act)
    np.testing.assert_array_equal(np.asarray(terms.bounds_violation), [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(st2.alive), [False, False, True, True])
    # the edge ramp is box-relative too: margin = 0.06 * min(50, 30) km = 1.8 km,
    # and aircraft 3 is ~1 km - 360 m of flight from the west edge
    e = np.asarray(terms.edge_proximity)
    assert e[2] == 0.0 and 0.5 < e[3] < 1.0

    # the same positions with the flag off are all on the grid: nothing fires
    env, st = _place(CFG, xy, psi)
    _, _, terms, _, _ = env.step(st, act)
    np.testing.assert_array_equal(np.asarray(terms.bounds_violation), [0.0, 0.0, 0.0, 0.0])


def test_edge_features_match_the_box():
    cfg = ON.replace(obs_edge_features=True)
    env = NaigosEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(9), 32)
    st, obs = jax.jit(jax.vmap(env.reset))(keys)
    b = st.bounds
    want = jax.vmap(edge_distances)(st.air.pos[..., :2], st.air.psi, b[:, 0], b[:, 1], b[:, 2], b[:, 3])
    np.testing.assert_allclose(np.asarray(obs.ego[..., 10:]), np.asarray(want), rtol=1e-5, atol=1e-6)
    # and they are NOT the full-grid distances
    grid = jax.vmap(lambda p, s: edge_distances(p, s, 0.0, EX, 0.0, EY))(st.air.pos[..., :2], st.air.psi)
    assert float(jnp.max(jnp.abs(obs.ego[..., 10:] - grid))) > 0.05

    # a known box: centre of 50 x 30 km heading east sees 25 km ahead and behind,
    # 15 km left and right; 2 km outside the box sees zeros
    env, st = _place(cfg, [[45_000.0, 30_000.0], [72_000.0, 30_000.0], [21_000.0, 30_000.0], [45_000.0, 44_000.0]],
                     [0.0, 0.0, jnp.pi, jnp.pi / 2])
    ego = np.asarray(env.observe(st).ego[:, 10:])
    np.testing.assert_allclose(ego[0], np.array([25_000.0, 15_000.0, 15_000.0, 25_000.0]) / 50_000.0, rtol=1e-5)
    np.testing.assert_array_equal(ego[1], np.zeros(4))
    # 1 km inside the west edge heading west: forward 1 km, back 49 km
    np.testing.assert_allclose(ego[2][[0, 3]], np.array([1_000.0, 49_000.0]) / 50_000.0, rtol=1e-4)
    # 1 km below the north edge heading north: forward 1 km
    np.testing.assert_allclose(ego[3][0], 1_000.0 / 50_000.0, rtol=1e-4)
