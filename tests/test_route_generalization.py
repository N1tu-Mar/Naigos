"""Route geometry diversification and the frozen held-out route bank (E-3, E-6)."""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env import config as C
from naigos.env.config import EnvConfig, TerrainConfig
from naigos.env.flight_env import NaigosEnv
from naigos.env import terrain as terrain_mod
from naigos.rl import train as T

# non-square (142.5 x 106.5 km) so an x/y mix-up cannot pass; big enough that the
# threat scatter (22 km) is not dominated by clipping at the grid edge
TERR = TerrainConfig(nx=96, ny=72, cell=1_500.0)
LEGACY = EnvConfig(n_blue=4, n_threat=12, n_threat_active=12, max_steps=60, terrain=TERR)
DIVERSE = LEGACY.replace(route_mode="diverse")
BOXED = DIVERSE.replace(map_randomize=True)
N = 96

# every bearing the protocol names: training sector centres and edges, and every
# held-out bank bearing
TRAIN_BEARINGS = [
    (f, c + d) for f, c in C.TRAIN_ROUTE_CENTERS_DEG.items() for d in (-DIVERSE.route_bearing_jitter_deg, 0.0,
                                                                     DIVERSE.route_bearing_jitter_deg)
]
HELDOUT_BEARINGS = sorted({(r.family, r.bearing_deg) for r in C.route_bank("heldout")})
ALL_BEARINGS = TRAIN_BEARINGS + HELDOUT_BEARINGS


def _route(family: str, bearing_deg: float):
    return jnp.array([math.radians(bearing_deg), C.ROUTE_FAMILY_NAMES.index(family)], dtype=jnp.float32)


@functools.lru_cache(maxsize=None)
def _batched_reset_route(cfg):
    # one compile per config; the route is a traced argument, not a constant
    return jax.jit(jax.vmap(NaigosEnv(cfg).reset_route, in_axes=(0, None)))


def _reset_route(cfg, family, bearing_deg, n=N, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return _batched_reset_route(cfg)(keys, _route(family, bearing_deg))


def _short_side(bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return np.minimum(b[:, 1] - b[:, 0], b[:, 3] - b[:, 2])


def _edge_distance(xy, bounds):
    b = np.asarray(bounds, dtype=np.float64)[:, None, :]
    x, y = np.asarray(xy, dtype=np.float64)[..., 0], np.asarray(xy, dtype=np.float64)[..., 1]
    return np.minimum(np.minimum(x - b[..., 0], b[..., 1] - x), np.minimum(y - b[..., 2], b[..., 3] - y))


def _ramp(cfg, bounds):
    if cfg.map_randomize:
        return cfg.edge_margin_frac * _short_side(bounds)[:, None]
    return np.full((len(np.asarray(bounds)), 1), cfg.edge_margin)


def _policy(obs, key):
    # steer at the objective: the direct-route reference
    return T.direct_route_policy(obs, key)


# --- legacy / default is unchanged -------------------------------------------------


def test_default_is_legacy_and_the_protocol_is_valid():
    assert EnvConfig().route_mode == "legacy"
    assert C.route_protocol_problems(EnvConfig()) == []
    assert C.route_protocol_problems(EnvConfig(route_mode="diverse")) == []
    assert T.TrainConfig().route_eval is False


def test_legacy_reset_is_the_original_placement():
    # Recomputed here from the original formula and the original key stream, so
    # the legacy branch cannot drift without this failing.
    cfg, key = LEGACY, jax.random.PRNGKey(17)
    st, _ = NaigosEnv(cfg).reset(key)
    _, k_start, k_obj, _, _ = jax.random.split(key, 5)
    w, h, B, inset = cfg.terrain.extent_x, cfg.terrain.extent_y, cfg.n_blue, cfg.spawn_inset_frac
    lat = jnp.linspace(0.25, 0.75, B) * h + jax.random.uniform(k_start, (B,), minval=-0.04, maxval=0.04) * h
    want_s = np.stack([np.full(B, inset * w), np.clip(lat, 1.5 * inset * h, (1 - 1.5 * inset) * h)], -1)
    want_o = np.stack([np.full(B, (1 - inset) * w), jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h], -1)
    np.testing.assert_array_equal(np.asarray(st.air.pos[:, :2]), want_s.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(st.objective[:, :2]), want_o.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(st.route), [0.0, 0.0, 1.0])
    # an explicit legacy config is the default config
    st2, _ = NaigosEnv(cfg.replace(route_mode="legacy")).reset(key)
    for a, b in zip(jax.tree.leaves(st), jax.tree.leaves(st2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_legacy_respawn_is_the_original_placement():
    cfg = LEGACY
    env = NaigosEnv(cfg)
    st, _ = env.reset(jax.random.PRNGKey(2))
    k = jax.random.PRNGKey(5)
    st2 = env.respawn(st, k, jnp.ones((cfg.n_blue,), dtype=bool))
    k_lat, k_obj, _ = jax.random.split(k, 3)
    w, h, B, inset = cfg.terrain.extent_x, cfg.terrain.extent_y, cfg.n_blue, cfg.spawn_inset_frac
    lat = jax.random.uniform(k_lat, (B,), minval=1.5 * inset, maxval=1 - 1.5 * inset) * h
    obj = jax.random.uniform(k_obj, (B,), minval=0.25, maxval=0.75) * h
    np.testing.assert_array_equal(np.asarray(st2.air.pos[:, 0]), np.full(B, inset * w, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(st2.air.pos[:, 1]), np.asarray(lat))
    np.testing.assert_array_equal(np.asarray(st2.objective[:, 1]), np.asarray(obj))


# --- geometry, for every family ----------------------------------------------------


@pytest.mark.parametrize("cfg", [DIVERSE, BOXED], ids=["full_grid", "map_randomize"])
@pytest.mark.parametrize("family,bearing", ALL_BEARINGS, ids=[f"{f}@{b:g}" for f, b in ALL_BEARINGS])
def test_every_route_is_in_bounds_long_reachable_and_headed_at_its_objective(cfg, family, bearing):
    st, obs = _reset_route(cfg, family, bearing)
    b = st.bounds
    start, obj = np.asarray(st.air.pos, np.float64), np.asarray(st.objective, np.float64)

    # inside the play box and clear of the boundary ramp at both ends
    ramp = _ramp(cfg, b)
    for xy in (start[..., :2], obj[..., :2]):
        e = _edge_distance(xy, b)
        assert np.all(e > ramp + 1.0), (family, bearing, (e - ramp).min())
        # the whole inset, not just the ramp: same clearance legacy starts have
        assert np.all(e >= cfg.spawn_inset_frac * _short_side(b)[:, None] - 1.0)

    # long enough to be a routing task, short enough to be flown in time
    d = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
    along = ((obj - start)[..., :2] * d).sum(-1)
    assert np.all(along >= cfg.route_min_length_frac * _short_side(b)[:, None]), along.min()
    length = np.linalg.norm((obj - start)[..., :2], axis=-1)
    # reachable in principle: flyable at v_max inside a default-length episode
    reach = cfg.airframe.v_max * cfg.dt * EnvConfig().max_steps
    assert np.all(length <= reach), length.max()
    assert np.all(length > 4 * cfg.objective_radius)

    # altitudes: start 2500 m and objective 1000 m above the local ground
    hm = np.asarray(st.hmap)
    for i in range(0, N, 17):
        g_s = terrain_mod.sample_height(hm[i], cfg.terrain, start[i, :, 0], start[i, :, 1])
        g_o = terrain_mod.sample_height(hm[i], cfg.terrain, obj[i, :, 0], obj[i, :, 1])
        np.testing.assert_allclose(start[i, :, 2] - np.asarray(g_s), 2_500.0, atol=0.05)
        np.testing.assert_allclose(obj[i, :, 2] - np.asarray(g_o), 1_000.0, atol=0.05)

    # heading: each aircraft points at its own objective, and roughly along the route
    psi = np.asarray(st.air.psi, np.float64)
    to = (obj - start)[..., :2] / length[..., None]
    np.testing.assert_allclose(np.cos(psi) * to[..., 0] + np.sin(psi) * to[..., 1], 1.0, atol=1e-5)
    max_dev = math.asin(cfg.route_lateral_frac)  # widest start/objective offset pair
    assert np.all(np.cos(psi - math.radians(bearing)) >= math.cos(max_dev) - 1e-4)
    # and the observation agrees: heading error zero at reset
    np.testing.assert_allclose(np.asarray(obs.ego[..., 4]), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(obs.ego[..., 5]), 1.0, atol=1e-4)

    # the route is recorded, and it is not the legacy one
    np.testing.assert_allclose(np.cos(np.asarray(st.route[:, 0], np.float64) - math.radians(bearing)), 1.0, atol=1e-9)
    assert np.all(np.abs(np.asarray(st.route[:, 0])) <= math.pi + 1e-6)
    np.testing.assert_array_equal(np.asarray(st.route[:, 1]), C.ROUTE_FAMILY_NAMES.index(family))
    np.testing.assert_array_equal(np.asarray(st.route[:, 2]), 0.0)


def test_cardinal_route_on_the_full_grid_spans_inset_edge_to_inset_edge():
    # due east reproduces the legacy picture's x extent exactly (not its lateral law)
    st, _ = _reset_route(DIVERSE, "west_east", 0.0, n=8)
    ex, inset = TERR.extent_x, DIVERSE.spawn_inset_frac
    np.testing.assert_allclose(np.asarray(st.air.pos[..., 0]), inset * ex, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(st.objective[..., 0]), (1 - inset) * ex, rtol=1e-5)
    # due north: bottom inset edge to top inset edge
    st, _ = _reset_route(DIVERSE, "south_north", 90.0, n=8)
    ey = TERR.extent_y
    np.testing.assert_allclose(np.asarray(st.air.pos[..., 1]), inset * ey, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(st.objective[..., 1]), (1 - inset) * ey, rtol=1e-4)


@pytest.mark.parametrize("family,bearing", [(f, c) for f, c in C.TRAIN_ROUTE_CENTERS_DEG.items()]
                         + [("southeast_northwest", 135.0), ("northwest_southeast", 315.0), ("oblique", 22.5)])
def test_threat_corridor_follows_the_route(family, bearing):
    st, _ = _reset_route(DIVERSE, family, bearing, n=64, seed=4)
    s = np.asarray(st.air.pos[..., :2], np.float64).mean(1)  # (N, 2) corridor start
    o = np.asarray(st.objective[..., :2], np.float64).mean(1)
    axis = o - s
    L = np.linalg.norm(axis, axis=-1, keepdims=True)
    u = axis / L
    n = np.stack([-u[:, 1], u[:, 0]], -1)
    # the corridor itself points along the bearing
    d = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
    assert np.all(u @ d > 0.9)
    rel = np.asarray(st.threats.pos[..., :2], np.float64) - s[:, None, :]
    active = np.asarray(st.threats.active)
    t = ((rel * u[:, None, :]).sum(-1) / L)[active]
    lat = (rel * n[:, None, :]).sum(-1)[active]
    along_m = ((rel * u[:, None, :]).sum(-1) - 0.5 * L)[active]
    # threats sit between the ends, centred on the route, spread ALONG it
    assert 0.35 < t.mean() < 0.65, t.mean()
    assert np.median(np.abs(lat)) < 25_000.0
    assert along_m.var() > lat.var()


# --- reset / respawn / rollout / jit / vmap in diverse mode -------------------------


def test_diverse_reset_is_deterministic_and_draws_only_training_sectors():
    env = NaigosEnv(BOXED)
    keys = jax.random.split(jax.random.PRNGKey(0), 1024)
    f = jax.jit(jax.vmap(env.reset))
    a, _ = f(keys)
    b, _ = f(keys)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    bearings = np.degrees(np.asarray(a.route[:, 0], np.float64))
    fam = np.asarray(a.route[:, 1]).astype(int)
    J = BOXED.route_bearing_jitter_deg
    # every draw is inside its own family's sector...
    for bear, i in zip(bearings, fam):
        c = C.TRAIN_ROUTE_CENTERS_DEG[C.ROUTE_FAMILY_NAMES[i]]
        assert C.angle_between_deg(bear, c) <= J + 1e-3
    # ...every training family is used...
    assert set(fam) == {C.ROUTE_FAMILY_NAMES.index(f) for f in BOXED.route_train_families}
    # ...and none comes near a held-out bearing
    held = [r.bearing_deg for r in C.route_bank("heldout")]
    gap = min(C.angle_between_deg(x, h) for x in bearings for h in held)
    assert gap >= C.ROUTE_SECTOR_GUARD_DEG, gap
    np.testing.assert_array_equal(np.asarray(a.route[:, 2]), 0.0)


def test_diverse_rollout_is_jit_vmap_able_and_deterministic():
    env = NaigosEnv(BOXED)
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    f = jax.jit(jax.vmap(lambda k: env.rollout(k, _policy)))
    final, traj = f(keys)
    assert traj["pos"].shape == (6, BOXED.max_steps, BOXED.n_blue, 3)
    assert np.all(np.isfinite(np.asarray(traj["pos"])))
    final2, traj2 = f(keys)
    np.testing.assert_array_equal(np.asarray(traj["pos"]), np.asarray(traj2["pos"]))
    # the route rides through step unchanged
    st0, _ = jax.vmap(lambda k: env.reset(jax.random.split(k)[0]))(keys)  # rollout resets on split(key)[0]
    np.testing.assert_array_equal(np.asarray(final.route), np.asarray(st0.route))

    # and the route-bank entry point: jit + vmap over (key, route)
    routes = jnp.stack([_route("oblique", 22.5), _route("northwest_southeast", 309.0)] * 3)
    g = jax.jit(jax.vmap(lambda k, r: env.rollout(k, _policy, route=r)))
    fin_r, traj_r = g(keys, routes)
    assert np.all(np.isfinite(np.asarray(traj_r["pos"])))
    dr = np.asarray(fin_r.route[:, 0]) - np.asarray(routes[:, 0])
    np.testing.assert_allclose(np.cos(dr), 1.0, atol=1e-6)


@pytest.mark.parametrize("cfg", [DIVERSE, BOXED, LEGACY], ids=["diverse", "diverse_boxed", "legacy_env"])
def test_respawn_keeps_the_route_and_its_invariants(cfg):
    env = NaigosEnv(cfg)
    n = 32
    keys = jax.random.split(jax.random.PRNGKey(8), n)
    # route-bank worlds in every env, including a legacy one: respawn must follow
    # the episode's own route, not the env's mode
    routes = jnp.stack([_route(*ALL_BEARINGS[(3 * i) % len(ALL_BEARINGS)]) for i in range(n)])
    st, _ = jax.vmap(env.reset_route)(keys, routes)
    # mess the state up: aircraft moved, tracks built up
    st = st._replace(lock=jnp.ones_like(st.lock), dwell=jnp.ones_like(st.dwell),
                     alive=jnp.zeros_like(st.alive))
    mask = jnp.tile(jnp.array([True, False, True, True]), (n, 1))
    st2 = jax.jit(jax.vmap(env.respawn))(st, jax.random.split(jax.random.PRNGKey(9), n), mask)
    m = np.asarray(mask)
    np.testing.assert_array_equal(np.asarray(st2.route), np.asarray(st.route))
    np.testing.assert_array_equal(np.asarray(st2.bounds), np.asarray(st.bounds))
    ramp = _ramp(cfg, st2.bounds)
    for xy in (st2.air.pos[..., :2], st2.objective[..., :2]):
        assert np.all((_edge_distance(xy, st2.bounds) > ramp + 1.0)[m])
    bearing = np.asarray(st2.route[:, 0], np.float64)[:, None]
    psi = np.asarray(st2.air.psi, np.float64)
    assert np.all((np.cos(psi - bearing) >= math.cos(math.asin(cfg.route_lateral_frac)) - 1e-4)[m])
    to = np.asarray(st2.objective - st2.air.pos)[..., :2]
    along = (to * np.stack([np.cos(bearing), np.sin(bearing)], -1)).sum(-1)
    assert np.all((along >= cfg.route_min_length_frac * _short_side(st2.bounds)[:, None])[m])
    # masked aircraft cleared and revived; the rest untouched
    assert np.all(np.asarray(st2.alive)[m]) and not np.any(np.asarray(st2.alive)[~m])
    assert np.all(np.asarray(st2.lock)[:, :, 1] == 1.0) and np.all(np.asarray(st2.lock)[:, :, 0] == 0.0)
    np.testing.assert_array_equal(np.asarray(st2.air.pos)[:, 1], np.asarray(st.air.pos)[:, 1])
    # threats can be re-rolled over the respawned corridor without losing the route
    st3 = jax.jit(jax.vmap(env.reroll_threats))(st2, keys)
    np.testing.assert_array_equal(np.asarray(st3.route), np.asarray(st.route))


# --- the frozen bank and the disjointness rule ------------------------------------


def test_bank_is_frozen_versioned_and_complete():
    rows = C.route_bank()
    assert C.route_bank_digest(rows) == C.ROUTE_BANK_V1_SHA256
    assert len(rows) == 96
    held, ref = C.route_bank("heldout"), C.route_bank("train_geometry")
    assert len(held) == 48 and len(ref) == 48
    assert {r.family for r in held} == set(C.HELDOUT_ROUTE_CENTERS_DEG)
    assert {r.family for r in ref} == set(C.TRAIN_ROUTE_CENTERS_DEG)
    ids = [r.scenario_id for r in rows]
    assert len(set(ids)) == len(ids) and all(i.startswith(C.ROUTE_BANK_VERSION + "/") for i in ids)
    # the bank is a pure function of the code: rebuilding gives identical rows
    assert C._build_route_bank_v1() == rows


def test_an_edited_bank_is_refused(monkeypatch):
    rows = list(C._build_route_bank_v1())
    rows[0] = dataclasses.replace(rows[0], bearing_deg=rows[0].bearing_deg + 1.0)
    monkeypatch.setattr(C, "_build_route_bank_v1", lambda: tuple(rows))
    with pytest.raises(RuntimeError, match="was edited"):
        C.route_bank()
    with pytest.raises(ValueError, match="unknown route bank"):
        C.route_bank(version="route-bank-v0")


def test_training_and_heldout_geometry_and_seeds_cannot_overlap():
    held, ref = C.route_bank("heldout"), C.route_bank("train_geometry")
    J = EnvConfig().route_bearing_jitter_deg
    # geometry: every held-out bearing clears every training sector by the guard
    for r in held:
        for c in C.TRAIN_ROUTE_CENTERS_DEG.values():
            assert C.angle_between_deg(r.bearing_deg, c) - J >= C.ROUTE_SECTOR_GUARD_DEG
    # and the in-distribution reference really is inside the training sectors
    for r in ref:
        assert C.angle_between_deg(r.bearing_deg, C.TRAIN_ROUTE_CENTERS_DEG[r.family]) <= J
    # seeds: unique, split-disjoint, inside their reserved ranges
    seeds = [r.seed for r in held + ref]
    assert len(set(seeds)) == len(seeds)
    for r in held + ref:
        base = C.ROUTE_BANK_SEED_BASE[r.split]
        assert base <= r.seed < base + C.ROUTE_BANK_SEED_SPAN
    # and disjoint from the cloud pipeline's held-out seeds
    pipeline = json.loads((Path(C.__file__).parents[1] / "pipeline" / "default_config.json").read_text())
    assert not set(seeds) & set(pipeline["evaluation"]["heldout_seeds"])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"route_bearing_jitter_deg": 16.0}, "comes within"),
        ({"route_train_families": ("west_east", "southeast_northwest")}, "held-out family"),
        ({"route_train_families": ("west_east", "sideways")}, "not a route family"),
        ({"route_train_families": ()}, "empty"),
        ({"route_mode": "anything"}, "not one of"),
        ({"route_lateral_frac": 0.95}, "only guarantees"),
    ],
)
def test_a_config_that_could_train_on_heldout_geometry_is_refused(override, match):
    cfg = DIVERSE.replace(**override)
    assert any(match in p for p in C.route_protocol_problems(cfg))
    with pytest.raises(ValueError, match="invalid route configuration"):
        NaigosEnv(cfg)


def test_training_seeds_in_the_reserved_ranges_are_refused():
    T.assert_route_bank_disjoint([0, 1, 3, 900_001])
    for bad in (710_000, 710_047, 719_999, 720_003):
        with pytest.raises(T.RouteBankOverlap):
            T.assert_route_bank_disjoint([0, bad])


def test_route_protocol_says_what_is_held_out_and_what_is_shared():
    p = T.route_protocol(DIVERSE)
    assert p["bank"] == C.ROUTE_BANK_VERSION and p["bank_sha256"] == C.ROUTE_BANK_V1_SHA256
    assert set(p["training_bearing_sectors_deg"]) == set(DIVERSE.route_train_families)
    assert set(p["heldout_bearing_centres_deg"]) == set(C.HELDOUT_ROUTE_CENTERS_DEG)
    assert p["shared_with_training"] and p["held_out"]
    assert T.route_protocol(LEGACY)["training_bearing_sectors_deg"] == {"west_east": [0.0, 0.0]}
    json.dumps(p)  # it goes into route_eval.json


# --- per-family metrics --------------------------------------------------------------

TINY = EnvConfig(n_blue=2, n_threat=4, n_threat_active=4, max_steps=24,
                 terrain=TerrainConfig(nx=48, ny=40, cell=1_500.0), route_mode="diverse")


def test_route_family_metrics_are_deterministic_and_consistent():
    env = NaigosEnv(TINY)
    a = T.evaluate_route_bank(env, _policy, training_seeds=[0])
    b = T.evaluate_route_bank(env, _policy, training_seeds=[0])
    assert a == b  # every number, bit for bit
    assert a["bank"] == C.ROUTE_BANK_VERSION and a["bank_sha256"] == C.ROUTE_BANK_V1_SHA256
    for split in ("heldout", "train_geometry"):
        rows = C.route_bank(split)
        res = a[split]
        assert res["n_scenarios"] == len(rows)
        assert set(res["by_family"]) == {r.family for r in rows}
        assert sum(f["n_scenarios"] for f in res["by_family"].values()) == len(rows)
        # the aggregate is the scenario-weighted mean of the families
        for k in ("survival_rate", "objective_rate", "shootdown_rate", "timeout_rate"):
            w = sum(f["n_scenarios"] * f[k] for f in res["by_family"].values()) / len(rows)
            assert abs(w - res["aggregate"][k]) < 1e-5, k
    assert set(a["generalization_gap"]) == set(T.ROUTE_HISTORY_METRICS)
    # a legacy env plays exactly the same bank (the bank overrides route_mode)
    c = T.evaluate_route_bank(NaigosEnv(TINY.replace(route_mode="legacy")), _policy)
    assert c["heldout"] == a["heldout"]
    with pytest.raises(T.RouteBankOverlap):
        T.evaluate_route_bank(env, _policy, training_seeds=[710_001])


def test_run_logs_route_eval_only_when_asked(tmp_path):
    from naigos.rl.ppo import PPOConfig

    ppo = PPOConfig(n_envs=2, n_steps=4, n_minibatches=2)

    def tc(out, **kw):
        return T.TrainConfig(iterations=2, out_dir=str(out), seed=3, eval_every=2, eval_worlds=2,
                             checkpoint_every=2, curriculum_every=2, **kw)

    _, hist_off = T.run(TINY, ppo, tc(tmp_path / "off"))
    assert not (tmp_path / "off" / T.ROUTE_EVAL_FILENAME).exists()
    assert not any(k.startswith("route_") or "_route_" in k for row in hist_off for k in row)

    _, hist_on = T.run(TINY, ppo, tc(tmp_path / "on", route_eval=True))
    log = json.loads((tmp_path / "on" / T.ROUTE_EVAL_FILENAME).read_text())
    assert log["protocol"] == T.route_protocol(TINY)
    assert [(e["iter"], e["policy"]) for e in log["evaluations"]] == [
        (0, "direct"), (0, "avoid_nap"), (1, "learner"), (2, "learner")
    ]
    for e in log["evaluations"]:
        assert set(e["heldout"]["by_family"]) == set(C.HELDOUT_ROUTE_CENTERS_DEG)
    assert "direct_route_heldout_objective_rate" in hist_on[0]
    assert all("route_heldout_objective_rate" in row for row in hist_on[1:])
    # route eval is reported, never trained on: the training curve is identical
    for off, on in zip(hist_off[1:], hist_on[1:]):
        for k in ("reward", "cost", "survival_rate", "objective_rate"):
            assert off[k] == on[k], k
