"""Flight-envelope limits must actually bind."""

from __future__ import annotations

import jax.numpy as jnp

from naigos.env.airframe import AircraftState, decode_action, load_factor, max_bank, step, velocity
from naigos.env.config import G0, AirframeConfig

CFG = AirframeConfig()


def _state(speed=180.0, **kw):
    d = dict(pos=jnp.zeros(3), speed=jnp.float32(speed), psi=jnp.float32(0.0),
             gamma=jnp.float32(0.0), phi=jnp.float32(0.0), fuel=jnp.float32(3000.0))
    d.update(kw)
    return AircraftState(**d)


def test_bank_is_capped_by_the_load_factor_limit():
    phi_cmd, _, _ = decode_action(CFG, jnp.array([1.0, 0.0, 0.0]), jnp.float32(200.0))
    assert float(phi_cmd) <= max_bank(CFG) + 1e-6
    assert float(load_factor(phi_cmd)) <= CFG.n_max + 1e-4


def test_speed_never_goes_below_stall_or_above_vmax():
    st = _state(speed=CFG.v_stall)
    for a in (-1.0, 1.0):
        s = st
        for _ in range(50):
            s = step(CFG, s, jnp.array([0.0, 0.0, a]), 1.0)
        assert CFG.v_stall - 1e-3 <= float(s.speed) <= CFG.v_max + 1e-3


def test_climb_rate_limit_binds_at_low_speed():
    """gamma is limited by BOTH gamma_max and roc_max/V, so a slow aircraft
    cannot command the same flight-path angle as a fast one."""
    _, g_slow, _ = decode_action(CFG, jnp.array([0.0, 1.0, 0.0]), jnp.float32(CFG.v_stall))
    _, g_fast, _ = decode_action(CFG, jnp.array([0.0, 1.0, 0.0]), jnp.float32(CFG.v_max))
    assert float(g_slow) >= float(g_fast)
    assert float(CFG.v_max * jnp.sin(g_fast)) <= CFG.roc_max + 1.0


def test_turn_rate_matches_the_coordinated_turn_relation():
    st = _state(speed=200.0)
    s2 = step(CFG, st, jnp.array([1.0, 0.0, 0.0]), 0.001)  # tiny dt, lag ~ frozen
    # after one tiny step bank is still ~0, so heading should barely move
    assert abs(float(s2.psi)) < 1e-3

    st = _state(speed=200.0, phi=jnp.float32(max_bank(CFG)))
    dt = 0.5
    s2 = step(CFG, st, jnp.array([1.0, 0.0, 0.0]), dt)
    expected = G0 * jnp.tan(jnp.float32(max_bank(CFG))) / 200.0 * dt
    assert abs(float(s2.psi) - float(expected)) < 0.05


def test_ceiling_clamps_altitude():
    st = _state(pos=jnp.array([0.0, 0.0, CFG.ceiling - 10.0]))
    for _ in range(20):
        st = step(CFG, st, jnp.array([0.0, 1.0, 1.0]), 1.0)
    assert float(st.pos[2]) <= CFG.ceiling + 1e-3


def test_fuel_monotonically_decreases_and_floors_at_zero():
    st = _state()
    prev = float(st.fuel)
    for _ in range(200):
        st = step(CFG, st, jnp.array([1.0, 0.0, 1.0]), 5.0)
        assert float(st.fuel) <= prev + 1e-6
        prev = float(st.fuel)
    assert float(st.fuel) >= 0.0


def test_manoeuvring_costs_more_fuel_than_flying_straight():
    a = _state()
    b = _state()
    for _ in range(30):
        a = step(CFG, a, jnp.array([0.0, 0.0, 0.5]), 1.0)
        b = step(CFG, b, jnp.array([1.0, 0.0, 0.5]), 1.0)
    assert float(b.fuel) < float(a.fuel)


def test_velocity_magnitude_equals_speed():
    st = _state(speed=210.0, psi=jnp.float32(1.1), gamma=jnp.float32(0.2))
    v = velocity(st)
    assert abs(float(jnp.linalg.norm(v)) - 210.0) < 1e-2
