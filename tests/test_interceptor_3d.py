"""Phase 1 of the 3D airborne-interceptor model.

Before this, `threats.step` moved interceptors horizontally and held altitude
(modulo a terrain-clearance floor), so blue could shake an airborne pursuer with
a pure climb or dive -- a 2.5D loophole, not a tactic. These tests pin the
bounded vertical model that closes it:

* ground threats stay on the DEM, exactly as before;
* only airborne kinds get vertical motion at all;
* climb and descent are rate-limited per `cfg.dt`;
* terrain clearance and the vehicle ceiling are hard bounds;
* vertical pursuit fires only on an existing track, never on an unseen blue.

Everything here is a MODELLING ASSUMPTION about a fictional, parameterised
vehicle. See docs/interceptor-3d-dynamics.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from naigos.env import terrain as terrain_mod
from naigos.env import threats as threats_mod
from naigos.env.config import (
    THREAT_INTERCEPTOR,
    THREAT_MOBILE,
    THREAT_STATIC_SAM,
    EnvConfig,
    ThreatKindConfig,
)
from naigos.env.flight_env import NaigosEnv
from naigos.env.threats import ThreatState

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=40)


# --- helpers -----------------------------------------------------------------
def flat_hmap(cfg: EnvConfig, height: float = 0.0) -> jax.Array:
    return jnp.full((cfg.terrain.ny, cfg.terrain.nx), height, dtype=jnp.float32)


def ramp_hmap(cfg: EnvConfig, lo: float = 0.0, hi: float = 3_000.0) -> jax.Array:
    """Terrain that rises linearly along +x. Flying east is flying uphill."""
    xs = jnp.linspace(lo, hi, cfg.terrain.nx, dtype=jnp.float32)
    return jnp.broadcast_to(xs[None, :], (cfg.terrain.ny, cfg.terrain.nx))


def make_threats(cfg: EnvConfig, kinds, xy, z, psi=0.0, active=True, vz=0.0) -> ThreatState:
    """Hand-built threat field -- no spawn randomness in the physics tests."""
    T = cfg.n_threat
    kind = jnp.array(list(kinds) + [0] * (T - len(kinds)), dtype=jnp.int32)
    xy = jnp.array(list(xy) + [[0.0, 0.0]] * (T - len(xy)), dtype=jnp.float32)
    z = jnp.array(list(z) + [0.0] * (T - len(z)), dtype=jnp.float32)
    act = jnp.arange(T) < len(kinds) if active is True else jnp.array(active)
    params = threats_mod.per_threat_params(cfg, kind)
    return ThreatState(
        pos=jnp.concatenate([xy, z[:, None]], axis=-1),
        psi=jnp.full((T,), psi, dtype=jnp.float32),
        speed=params["speed"],
        kind=kind,
        active=act,
        vz=jnp.full((T,), vz, dtype=jnp.float32),
    )


def kinds_of(cfg: EnvConfig):
    return cfg.threat_kinds


# --- 1. configuration --------------------------------------------------------
def test_airborne_flight_limits_are_separate_from_the_engagement_envelope():
    k = ThreatKindConfig()
    for f in ("climb_rate_max", "descent_rate_max", "terrain_clearance", "vehicle_ceiling"):
        assert hasattr(k, f), f"ThreatKindConfig is missing {f}"
    itc = kinds_of(CFG)[THREAT_INTERCEPTOR]
    assert itc.airborne
    assert itc.climb_rate_max > 0.0
    assert itc.descent_rate_max > 0.0
    assert itc.terrain_clearance > 0.0
    # the vehicle's own ceiling is NOT the engagement envelope's upper edge
    assert itc.vehicle_ceiling != itc.alt_max


def test_flight_limits_reach_the_per_threat_param_table():
    kind = jnp.array([THREAT_INTERCEPTOR, THREAT_STATIC_SAM], dtype=jnp.int32)
    p = threats_mod.per_threat_params(CFG, kind)
    for f in ("climb_rate_max", "descent_rate_max", "terrain_clearance", "vehicle_ceiling"):
        assert f in p and p[f].shape == (2,)


# --- 2. state ----------------------------------------------------------------
def test_threat_state_carries_vertical_velocity():
    assert "vz" in ThreatState._fields
    hmap = flat_hmap(CFG)
    corridor = jnp.array([[10_000.0, 10_000.0, 3_000.0], [100_000.0, 100_000.0, 3_000.0]])
    st = threats_mod.spawn(jax.random.PRNGKey(0), CFG, hmap, corridor)
    assert st.vz.shape == (CFG.n_threat,)
    assert bool(jnp.all(jnp.isfinite(st.vz)))


# --- 3. spawning -------------------------------------------------------------
def test_spawn_places_airborne_above_terrain_and_ground_on_it():
    cfg = CFG.replace(n_threat=64, n_threat_active=64)
    hmap = terrain_mod.synthetic_terrain(jax.random.PRNGKey(5), cfg.terrain)
    corridor = jnp.array([[10_000.0, 60_000.0, 3_000.0], [180_000.0, 120_000.0, 3_000.0]])
    st = threats_mod.spawn(jax.random.PRNGKey(1), cfg, hmap, corridor)
    p = threats_mod.per_threat_params(cfg, st.kind)
    g = terrain_mod.sample_height(hmap, cfg.terrain, st.pos[:, 0], st.pos[:, 1])
    air = p["airborne"] > 0.5
    agl = st.pos[:, 2] - g

    assert bool(jnp.all(jnp.where(air, agl >= p["terrain_clearance"] - 1e-3, True)))
    assert bool(jnp.all(jnp.where(air, st.pos[:, 2] <= p["vehicle_ceiling"] + 1e-3, True)))
    assert bool(jnp.all(jnp.where(~air, jnp.abs(agl - 5.0) < 1e-2, True)))


# --- 4. vertical motion ------------------------------------------------------
def test_ground_threats_stay_pinned_to_the_dem():
    cfg = CFG
    hmap = ramp_hmap(cfg)
    st = make_threats(
        cfg,
        kinds=[THREAT_STATIC_SAM, THREAT_MOBILE],
        xy=[[40_000.0, 40_000.0], [40_000.0, 60_000.0]],
        z=[0.0, 0.0],
    )
    g0 = terrain_mod.sample_height(hmap, cfg.terrain, st.pos[:, 0], st.pos[:, 1])
    st = st._replace(pos=st.pos.at[:, 2].set(g0 + 5.0))

    vz_cmd = jnp.full((cfg.n_threat,), 50.0)  # commanded hard, must be ignored
    out = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)

    g1 = terrain_mod.sample_height(hmap, cfg.terrain, out.pos[:, 0], out.pos[:, 1])
    assert bool(jnp.allclose(out.pos[:2, 2], (g1 + 5.0)[:2], atol=1e-3))
    assert bool(jnp.all(out.vz[:2] == 0.0))


def test_only_airborne_threats_receive_vertical_motion():
    cfg = CFG
    hmap = flat_hmap(cfg, 100.0)
    st = make_threats(
        cfg,
        kinds=[THREAT_MOBILE, THREAT_INTERCEPTOR],
        xy=[[40_000.0, 40_000.0], [40_000.0, 60_000.0]],
        z=[105.0, 4_000.0],
    )
    vz_cmd = jnp.full((cfg.n_threat,), 30.0)
    out = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)
    assert float(out.pos[0, 2]) == pytest.approx(105.0, abs=1e-3)  # ground unit
    assert float(out.pos[1, 2]) > 4_000.0  # interceptor climbed


def test_climb_and_descent_are_bounded_per_timestep():
    cfg = CFG
    hmap = flat_hmap(cfg, 0.0)
    itc = kinds_of(cfg)[THREAT_INTERCEPTOR]
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR] * 2, xy=[[40_000.0, 40_000.0]] * 2, z=[5_000.0, 5_000.0])

    up = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, jnp.full((cfg.n_threat,), 1e6))
    dn = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, jnp.full((cfg.n_threat,), -1e6))

    assert float(up.pos[0, 2] - 5_000.0) == pytest.approx(itc.climb_rate_max * cfg.dt, rel=1e-4)
    assert float(5_000.0 - dn.pos[0, 2]) == pytest.approx(itc.descent_rate_max * cfg.dt, rel=1e-4)
    assert float(up.vz[0]) == pytest.approx(itc.climb_rate_max, rel=1e-4)
    assert float(dn.vz[0]) == pytest.approx(-itc.descent_rate_max, rel=1e-4)


def test_terrain_clearance_holds_while_flying_over_rising_ground():
    cfg = CFG.replace(dt=4.0)
    hmap = ramp_hmap(cfg, 0.0, 6_000.0)
    itc = kinds_of(cfg)[THREAT_INTERCEPTOR]
    x0 = 40_000.0
    g0 = float(terrain_mod.sample_height(hmap, cfg.terrain, jnp.array(x0), jnp.array(40_000.0)))
    st = make_threats(
        cfg,
        kinds=[THREAT_INTERCEPTOR],
        xy=[[x0, 40_000.0]],
        z=[g0 + itc.terrain_clearance],
        psi=0.0,  # due +x, straight uphill
    )
    # dive hard while running uphill: clearance must still win
    vz_cmd = jnp.full((cfg.n_threat,), -1e6)
    for _ in range(12):
        st = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)
        g = terrain_mod.sample_height(hmap, cfg.terrain, st.pos[:, 0], st.pos[:, 1])
        assert float(st.pos[0, 2] - g[0]) >= itc.terrain_clearance - 1e-2


def test_vehicle_ceiling_holds_under_a_sustained_climb():
    cfg = CFG
    hmap = flat_hmap(cfg, 0.0)
    itc = kinds_of(cfg)[THREAT_INTERCEPTOR]
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[5_000.0])
    vz_cmd = jnp.full((cfg.n_threat,), 1e6)
    for _ in range(400):
        st = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)
        assert float(st.pos[0, 2]) <= itc.vehicle_ceiling + 1e-2
    assert float(st.pos[0, 2]) == pytest.approx(itc.vehicle_ceiling, abs=1e-2)


def test_inactive_threats_do_not_move():
    cfg = CFG
    hmap = ramp_hmap(cfg)
    st = make_threats(
        cfg,
        kinds=[THREAT_INTERCEPTOR, THREAT_INTERCEPTOR],
        xy=[[40_000.0, 40_000.0], [60_000.0, 40_000.0]],
        z=[5_000.0, 5_000.0],
        active=[False] * cfg.n_threat,
    )
    out = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, jnp.full((cfg.n_threat,), 40.0))
    assert bool(jnp.allclose(out.pos, st.pos))
    assert bool(jnp.all(out.vz == 0.0))


# --- 5. vertical pursuit command --------------------------------------------
def _blue(cfg, z):
    pos = jnp.stack(
        [jnp.full((cfg.n_blue,), 45_000.0), jnp.full((cfg.n_blue,), 40_000.0), jnp.array(z, dtype=jnp.float32)],
        axis=-1,
    )
    return pos


def test_vertical_pursuit_needs_an_existing_track():
    cfg = CFG
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[3_000.0])
    blue = _blue(cfg, [9_000.0] * cfg.n_blue)
    alive = jnp.ones((cfg.n_blue,), dtype=bool)

    no_track = jnp.zeros((cfg.n_threat, cfg.n_blue))
    assert float(threats_mod.vertical_command(cfg, st, blue, alive, no_track)[0]) == 0.0

    lock = no_track.at[0, 1].set(0.9)
    assert float(threats_mod.vertical_command(cfg, st, blue, alive, lock)[0]) > 0.0


def test_vertical_pursuit_ignores_dead_contacts():
    cfg = CFG
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[3_000.0])
    blue = _blue(cfg, [9_000.0] * cfg.n_blue)
    lock = jnp.zeros((cfg.n_threat, cfg.n_blue)).at[0, 2].set(0.9)
    dead = jnp.zeros((cfg.n_blue,), dtype=bool)
    assert float(threats_mod.vertical_command(cfg, st, blue, dead, lock)[0]) == 0.0


def test_vertical_pursuit_is_signed_toward_the_tracked_contact():
    cfg = CFG
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[6_000.0])
    alive = jnp.ones((cfg.n_blue,), dtype=bool)
    lock = jnp.zeros((cfg.n_threat, cfg.n_blue)).at[0, 0].set(0.8)
    itc = kinds_of(cfg)[THREAT_INTERCEPTOR]

    above = threats_mod.vertical_command(cfg, st, _blue(cfg, [10_000.0] * 3), alive, lock)
    below = threats_mod.vertical_command(cfg, st, _blue(cfg, [1_000.0] * 3), alive, lock)
    assert 0.0 < float(above[0]) <= itc.climb_rate_max + 1e-6
    assert -itc.descent_rate_max - 1e-6 <= float(below[0]) < 0.0


def test_ground_threats_never_get_a_vertical_command():
    cfg = CFG
    st = make_threats(
        cfg,
        kinds=[THREAT_STATIC_SAM, THREAT_MOBILE],
        xy=[[40_000.0, 40_000.0], [40_000.0, 41_000.0]],
        z=[5.0, 5.0],
    )
    blue = _blue(cfg, [9_000.0] * cfg.n_blue)
    lock = jnp.ones((cfg.n_threat, cfg.n_blue)) * 0.9
    assert bool(jnp.all(threats_mod.vertical_command(cfg, st, blue, jnp.ones((3,), bool), lock)[:2] == 0.0))


def test_inactive_threats_never_get_a_vertical_command():
    cfg = CFG
    st = make_threats(
        cfg,
        kinds=[THREAT_INTERCEPTOR],
        xy=[[40_000.0, 40_000.0]],
        z=[3_000.0],
        active=[False] * cfg.n_threat,
    )
    blue = _blue(cfg, [9_000.0] * cfg.n_blue)
    lock = jnp.ones((cfg.n_threat, cfg.n_blue)) * 0.9
    assert bool(jnp.all(threats_mod.vertical_command(cfg, st, blue, jnp.ones((3,), bool), lock) == 0.0))


# --- 6. the loophole this closes --------------------------------------------
def test_a_vertical_only_escape_is_no_longer_free():
    """Blue climbs away from a tracking interceptor; the gap must close."""
    cfg = CFG
    hmap = flat_hmap(cfg, 0.0)
    itc = kinds_of(cfg)[THREAT_INTERCEPTOR]
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[3_000.0])
    alive = jnp.ones((cfg.n_blue,), dtype=bool)
    lock = jnp.zeros((cfg.n_threat, cfg.n_blue)).at[0, 0].set(0.9)

    blue_z = 3_000.0
    gap0 = None
    for i in range(20):
        blue_z += 20.0 * cfg.dt  # blue climbs at 20 m/s, inside the interceptor's limit
        blue = _blue(cfg, [blue_z] * cfg.n_blue)
        vz_cmd = threats_mod.vertical_command(cfg, st, blue, alive, lock)
        st = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)
        gap = blue_z - float(st.pos[0, 2])
        if i == 0:
            gap0 = gap
    assert gap <= gap0 + 1e-6
    assert gap < 20.0 * cfg.dt * 20 * 0.5  # nowhere near a free 2.5D escape
    assert float(st.pos[0, 2]) <= itc.vehicle_ceiling + 1e-2


def test_a_diving_blue_is_followed_down():
    cfg = CFG
    hmap = flat_hmap(cfg, 0.0)
    st = make_threats(cfg, kinds=[THREAT_INTERCEPTOR], xy=[[40_000.0, 40_000.0]], z=[9_000.0])
    alive = jnp.ones((cfg.n_blue,), dtype=bool)
    lock = jnp.zeros((cfg.n_threat, cfg.n_blue)).at[0, 0].set(0.9)
    blue = _blue(cfg, [2_000.0] * cfg.n_blue)
    z_prev = float(st.pos[0, 2])
    for _ in range(6):
        vz_cmd = threats_mod.vertical_command(cfg, st, blue, alive, lock)
        st = threats_mod.step(cfg, st, hmap, jnp.zeros((cfg.n_threat,)), st.speed, vz_cmd)
        z = float(st.pos[0, 2])
        assert z < z_prev
        z_prev = z
    assert z_prev >= 2_000.0 - 1e-3


# --- 7. env integration ------------------------------------------------------
def straight(obs, key):
    h = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
    return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.zeros_like(h), jnp.full_like(h, 0.6)], -1)


def test_env_step_moves_airborne_threats_in_3d_and_stays_jit_able():
    env = NaigosEnv(CFG)
    st, _ = env.reset(jax.random.PRNGKey(0))
    st = st._replace(lock=jnp.ones_like(st.lock) * 0.9)
    s2, o2, terms, done, info = jax.jit(env.step)(st, jnp.zeros((CFG.n_blue, 3)))
    p = threats_mod.per_threat_params(CFG, s2.threats.kind)
    air = (p["airborne"] > 0.5) & s2.threats.active
    assert bool(jnp.any(air))
    dz = jnp.abs(s2.threats.pos[:, 2] - st.threats.pos[:, 2])
    assert float(jnp.max(jnp.where(air, dz, 0.0))) > 0.0
    assert bool(jnp.all(jnp.isfinite(s2.threats.pos)))


def test_rollout_is_deterministic_and_vmap_able_with_3d_threats():
    env = NaigosEnv(CFG)
    f = jax.jit(jax.vmap(lambda k: env.rollout(k, straight)))
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    a_final, a_traj = f(keys)
    b_final, b_traj = f(keys)
    assert bool(jnp.allclose(a_traj["threat_pos"], b_traj["threat_pos"]))
    assert bool(jnp.all(jnp.isfinite(a_traj["threat_pos"])))
    assert a_traj["threat_pos"].shape == (3, CFG.max_steps, CFG.n_threat, 3)


def test_threat_altitudes_stay_inside_their_flight_envelope_over_a_rollout():
    env = NaigosEnv(CFG)
    final, traj = jax.jit(lambda k: env.rollout(k, straight))(jax.random.PRNGKey(7))
    p = threats_mod.per_threat_params(CFG, final.threats.kind)
    air = p["airborne"] > 0.5
    z = traj["threat_pos"][..., 2]  # (T_steps, T, )
    g = terrain_mod.sample_height(
        final.hmap, CFG.terrain, traj["threat_pos"][..., 0], traj["threat_pos"][..., 1]
    )
    agl = z - g
    ok_lo = jnp.where(air[None, :] & final.threats.active[None, :], agl >= p["terrain_clearance"][None, :] - 1.0, True)
    ok_hi = jnp.where(air[None, :], z <= p["vehicle_ceiling"][None, :] + 1.0, True)
    assert bool(jnp.all(ok_lo))
    assert bool(jnp.all(ok_hi))


def test_detection_sees_finite_3d_threat_positions():
    from naigos.env import detection as det_mod

    env = NaigosEnv(CFG)
    st, _ = env.reset(jax.random.PRNGKey(0))
    st = st._replace(lock=jnp.ones_like(st.lock) * 0.9)
    s2, *_ = env.step(st, jnp.zeros((CFG.n_blue, 3)))
    tparams = threats_mod.per_threat_params(CFG, s2.threats.kind)
    det = det_mod.detection_probability(
        s2.hmap, CFG.terrain, CFG.detection, s2.air.pos, s2.air.psi, s2.threats.pos, tparams, s2.threats.active
    )
    for k, v in det.items():
        assert bool(jnp.all(jnp.isfinite(v))), k
    assert bool(jnp.all((det["pd"] >= 0.0) & (det["pd"] <= 1.0)))


def test_blue_still_has_no_way_to_touch_a_threat():
    """The 3D interceptor changes red's physics, not blue's action space."""
    from naigos.env.config import BLUE_ACTION_NAMES

    assert BLUE_ACTION_NAMES == ("bank_cmd", "gamma_cmd", "throttle_cmd")
    assert CFG.action_dim == 3
