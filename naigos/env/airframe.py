"""Point-mass 3D flight dynamics with a setpoint action interface.

Following the Nomos choice of *setpoints, not raw torques*: the policy commands
a bank angle, a flight-path angle and a throttle; first-order actuator lags and
hard envelope limits turn those into achievable motion. That keeps the action
space small and bounded while the airframe stays physically plausible.

State (per aircraft):
    pos    (3,)  local ENU metres, z is AMSL
    speed  ()    true airspeed, m/s
    psi    ()    heading, rad (0 = +x, CCW positive)
    gamma  ()    flight-path angle, rad
    phi    ()    bank angle, rad
    fuel   ()    kg

The coordinated-turn relation `psi_dot = g tan(phi) / V` is what couples the
g-limit to the achievable turn rate: at high speed the same bank buys you less
heading rate, so evasive turns cost energy. That coupling is the whole reason a
2D bicycle model would not do.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import G0, AirframeConfig


class AircraftState(NamedTuple):
    pos: jax.Array  # (..., 3)
    speed: jax.Array  # (...,)
    psi: jax.Array  # (...,)
    gamma: jax.Array  # (...,)
    phi: jax.Array  # (...,)
    fuel: jax.Array  # (...,)


def max_bank(cfg: AirframeConfig) -> float:
    """Bank angle at the load-factor limit: n = 1/cos(phi)."""
    return float(jnp.arccos(1.0 / cfg.n_max))


def load_factor(phi: jax.Array) -> jax.Array:
    """Load factor for a coordinated level turn at bank `phi`."""
    return 1.0 / jnp.maximum(jnp.cos(phi), 1e-3)


def decode_action(cfg: AirframeConfig, action: jax.Array, speed: jax.Array):
    """Map a bounded action in [-1, 1]^3 to physical setpoints.

    action[0] -> bank command, scaled by the g-limit
    action[1] -> flight-path-angle command, scaled by BOTH gamma_max and the
                 rate-of-climb limit (which bites at low speed)
    action[2] -> speed command between stall and v_max
    """
    a = jnp.clip(action, -1.0, 1.0)
    phi_lim = jnp.arccos(1.0 / cfg.n_max)
    phi_cmd = a[..., 0] * phi_lim

    gamma_from_roc = jnp.arcsin(jnp.clip(cfg.roc_max / jnp.maximum(speed, cfg.v_stall), -1.0, 1.0))
    gamma_lim = jnp.minimum(cfg.gamma_max, gamma_from_roc)
    gamma_cmd = a[..., 1] * gamma_lim

    speed_cmd = cfg.v_stall + 0.5 * (a[..., 2] + 1.0) * (cfg.v_max - cfg.v_stall)
    return phi_cmd, gamma_cmd, speed_cmd


def step(cfg: AirframeConfig, st: AircraftState, action: jax.Array, dt: float) -> AircraftState:
    """Integrate one step. Semi-implicit Euler; dt of ~1 s is stable at these lags."""
    phi_cmd, gamma_cmd, speed_cmd = decode_action(cfg, action, st.speed)

    # first-order actuator lags
    phi = st.phi + (phi_cmd - st.phi) * jnp.minimum(dt / cfg.tau_bank, 1.0)
    gamma = st.gamma + (gamma_cmd - st.gamma) * jnp.minimum(dt / cfg.tau_gamma, 1.0)
    speed = st.speed + (speed_cmd - st.speed) * jnp.minimum(dt / cfg.tau_speed, 1.0)

    # envelope clamps (violations are recorded by the verifier, not silently ok)
    speed = jnp.clip(speed, cfg.v_stall, cfg.v_max)
    gamma = jnp.clip(gamma, -cfg.gamma_max, cfg.gamma_max)

    # coordinated turn
    psi = st.psi + dt * (G0 * jnp.tan(phi) / jnp.maximum(speed, cfg.v_stall))
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    cg = jnp.cos(gamma)
    vel = jnp.stack([speed * cg * jnp.cos(psi), speed * cg * jnp.sin(psi), speed * jnp.sin(gamma)], axis=-1)
    pos = st.pos + dt * vel
    pos = pos.at[..., 2].set(jnp.minimum(pos[..., 2], cfg.ceiling))

    throttle = (speed - cfg.v_stall) / (cfg.v_max - cfg.v_stall)
    n = load_factor(phi)
    burn = cfg.burn_base + cfg.burn_throttle * throttle + cfg.burn_maneuver * jnp.abs(n - 1.0)
    fuel = jnp.maximum(st.fuel - dt * burn, 0.0)

    # out of fuel: the airframe glides, it does not hover
    dry = (fuel <= 0.0).astype(jnp.float32)
    speed = speed * (1 - dry) + jnp.maximum(speed - 5.0 * dt, cfg.v_stall) * dry

    return AircraftState(pos=pos, speed=speed, psi=psi, gamma=gamma, phi=phi, fuel=fuel)


def velocity(st: AircraftState) -> jax.Array:
    """Inertial velocity vector (..., 3)."""
    cg = jnp.cos(st.gamma)
    return jnp.stack(
        [st.speed * cg * jnp.cos(st.psi), st.speed * cg * jnp.sin(st.psi), st.speed * jnp.sin(st.gamma)],
        axis=-1,
    )
