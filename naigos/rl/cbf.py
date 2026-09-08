"""Higher-order CBF-QP runtime safety backstop.

`a_exec = filter(a_policy, state)` -- exactly the Nomos contract. Two barrier
families, both relative-degree 2 in the controls (the controls enter through the
accelerations, not the positions, so a plain first-order CBF cannot express them):

  1. **Lethal-envelope keep-out** (horizontal). h = |p_xy - c| - r for each
     *known* envelope. Actuated by bank (lateral acceleration) and throttle
     (longitudinal acceleration).
  2. **Terrain floor** (vertical). h = alt_agl - floor. Actuated by the
     flight-path angle, with the terrain gradient along track included so the
     barrier anticipates rising ground instead of reacting to it.

HOCBF construction, per barrier:
    psi0 = h
    psi1 = hdot + a1 * psi0
    enforce:  psi1dot + a2 * psi1 >= 0
which expands to a constraint **linear in the control accelerations**.

HONEST CAVEAT (the same one Nomos carries): this guarantees safety only while
the QP is feasible. At high threat density the intersection of the half-spaces
with the actuator box can be empty -- overlapping envelopes with no gap simply
cannot be flown out of. The filter's value is bounded above by how good the
learned policy already is. It is a backstop, not the plan. `filter_action`
returns a `feasible` flag; do not silently ignore it.

SOLVER: the QP is
    min ||u - u_pol||^2  s.t.  A u >= b,  u in box
solved by fixed-iteration cyclic projection onto the half-spaces and the box
(Dykstra-free alternating projections). That converges for a feasible convex
intersection, is jit-able with static shapes, and needs no external QP library.
It is NOT an exact active-set solver -- see next-steps.md gap S-1.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from ..env.config import G0, EnvConfig
from ..env import terrain as terrain_mod


@dataclasses.dataclass(frozen=True)
class CBFConfig:
    alpha1: float = 0.01  # 1/s, outer class-K gain. alpha1 * h is the closing
    # speed the barrier will tolerate, so 0.01 starts biting ~20 km out at 200 m/s.
    alpha2: float = 0.15  # 1/s, inner class-K gain
    w_long: float = 4.0  # cost weight on longitudinal (throttle) corrections
    # relative to lateral (bank) ones. Turning around a SAM is nearly free;
    # decelerating in its envelope is not, so the QP should prefer to turn.
    margin: float = 1_500.0  # m, inflate every envelope by this before filtering
    n_iters: int = 24  # projection iterations
    a_long_max: float = 25.0  # m/s^2 achievable longitudinal acceleration
    enable_terrain: bool = True
    enable_envelope: bool = True


def _envelope_constraints(cbf: CBFConfig, pos, vel, centers, radii, active):
    """Linear constraints A u >= b from the horizontal keep-out barriers.

    Distance form, h = |d| - r, rather than the squared form. The squared form
    is algebraically simpler but numerically awful here: h and hdot then scale
    with the square of a 10^4-metre range, so a class-K gain that behaves at
    1 km is off by four orders of magnitude at 100 km and the filter either does
    nothing or saturates. The distance form keeps every term in metres and
    metres/second, so alpha1 and alpha2 have units of 1/s and mean what they say.

    u = (a_long, a_lat) in the body frame. Returns A (T, 2), b (T,).
    """
    d = pos[:2] - centers[:, :2]  # (T, 2)
    rng = jnp.linalg.norm(d, axis=-1) + 1e-6
    dhat = d / rng[:, None]
    r = radii + cbf.margin
    h = rng - r

    v = vel[:2]
    hdot = jnp.sum(dhat * v[None, :], axis=-1)

    speed2 = jnp.sum(v**2)
    t_hat = v / (jnp.linalg.norm(v) + 1e-6)
    n_hat = jnp.stack([-t_hat[1], t_hat[0]])

    # hddot = (|v|^2 - hdot^2)/rng + dhat . a
    A = jnp.stack([jnp.sum(dhat * t_hat[None, :], axis=-1), jnp.sum(dhat * n_hat[None, :], axis=-1)], axis=-1)
    psi1 = hdot + cbf.alpha1 * h
    const = (speed2 - hdot**2) / rng + cbf.alpha1 * hdot + cbf.alpha2 * psi1
    b = -const

    # inactive / unknown envelopes impose nothing: A=0, b=-inf-ish
    keep = active.astype(A.dtype)[:, None]
    A = A * keep
    b = jnp.where(active, b, -1e9)
    return A, b


def _terrain_constraint(cbf: CBFConfig, cfg: EnvConfig, hmap, pos, psi, speed, gamma):
    """Vertical barrier on AGL clearance. Returns (a_gamma_min,) as a scalar
    lower bound on the commanded flight-path angle rate contribution.

    h    = z - ground(x, y) - floor
    hdot = V sin(gamma) - V cos(gamma) * (dh/dx cos psi + dh/dy sin psi)
    The control (commanded gamma) enters hdot's derivative through the actuator
    lag, so this is again relative degree 2 and yields a lower bound on the
    commanded gamma.
    """
    ground = terrain_mod.sample_height(hmap, cfg.terrain, pos[0], pos[1])
    gx, gy = terrain_mod.terrain_gradient(hmap, cfg.terrain, pos[0], pos[1])
    slope = gx * jnp.cos(psi) + gy * jnp.sin(psi)  # m per m along track

    h = pos[2] - ground - cfg.airframe.floor_agl
    hdot = speed * (jnp.sin(gamma) - jnp.cos(gamma) * slope)
    psi1 = hdot + cbf.alpha1 * h

    # required hddot >= -alpha1*hdot - alpha2*psi1; with gammadot = (g_cmd - g)/tau
    # and hddot ~= speed*cos(gamma)*gammadot, solve for g_cmd.
    need = -cbf.alpha1 * hdot - cbf.alpha2 * psi1
    denom = speed * jnp.cos(gamma) / jnp.maximum(cfg.airframe.tau_gamma, 1e-3)
    gamma_min = gamma + need / jnp.maximum(denom, 1e-3)
    return jnp.clip(gamma_min, -cfg.airframe.gamma_max, cfg.airframe.gamma_max)


def _project_qp(u0, A, b, lo, hi, n_iters: int, w=None):
    """min ||W^(1/2)(u - u0)||^2 s.t. A u >= b, lo <= u <= hi.

    Cyclic projection in the metric induced by the diagonal weight `w`: the
    projection onto {a.u >= b} becomes u += viol * (W^-1 a) / (a' W^-1 a).
    """
    if w is None:
        w = jnp.ones_like(u0)
    winv = 1.0 / w

    def body(u, _):
        def one(u, i):
            a = A[i]
            na2 = jnp.sum(winv * a**2) + 1e-9
            viol = b[i] - jnp.dot(a, u)
            u = u + jnp.where(viol > 0, viol / na2, 0.0) * (winv * a)
            return u, None

        u, _ = jax.lax.scan(one, u, jnp.arange(A.shape[0]))
        return jnp.clip(u, lo, hi), None

    u, _ = jax.lax.scan(body, jnp.clip(u0, lo, hi), None, length=n_iters)
    feasible = jnp.all(A @ u >= b - 1e-3)
    return u, feasible


def filter_action(
    cbf: CBFConfig,
    cfg: EnvConfig,
    hmap,
    action: jnp.ndarray,  # (3,) policy action for ONE aircraft
    pos: jnp.ndarray,  # (3,)
    psi,
    gamma,
    phi,
    speed,
    known_centers: jnp.ndarray,  # (T, 3)
    known_radii: jnp.ndarray,  # (T,)
    known_active: jnp.ndarray,  # (T,) bool -- only envelopes the aircraft KNOWS
):
    """Return `(safe_action, feasible)`. Safe action is the closest admissible
    action to the policy's, in the same [-1, 1]^3 space.
    """
    af = cfg.airframe
    phi_lim = jnp.arccos(1.0 / af.n_max)
    a_lat_max = G0 * jnp.tan(phi_lim)

    # policy action -> requested accelerations
    phi_cmd = action[0] * phi_lim
    a_lat_pol = G0 * jnp.tan(phi_cmd)
    speed_cmd = af.v_stall + 0.5 * (action[2] + 1.0) * (af.v_max - af.v_stall)
    a_long_pol = (speed_cmd - speed) / af.tau_speed

    vel = jnp.array([speed * jnp.cos(gamma) * jnp.cos(psi), speed * jnp.cos(gamma) * jnp.sin(psi), speed * jnp.sin(gamma)])

    if cbf.enable_envelope:
        A, b = _envelope_constraints(cbf, pos, vel, known_centers, known_radii, known_active)
    else:
        A = jnp.zeros((known_centers.shape[0], 2))
        b = jnp.full((known_centers.shape[0],), -1e9)

    lo = jnp.array([-cbf.a_long_max, -a_lat_max])
    hi = jnp.array([cbf.a_long_max, a_lat_max])
    w = jnp.array([cbf.w_long, 1.0])
    u, feasible = _project_qp(jnp.array([a_long_pol, a_lat_pol]), A, b, lo, hi, cbf.n_iters, w)

    # accelerations -> action
    a_long, a_lat = u[0], u[1]
    phi_safe = jnp.arctan(a_lat / G0)
    a0 = jnp.clip(phi_safe / phi_lim, -1.0, 1.0)
    speed_safe = jnp.clip(speed + a_long * af.tau_speed, af.v_stall, af.v_max)
    a2 = jnp.clip(2.0 * (speed_safe - af.v_stall) / (af.v_max - af.v_stall) - 1.0, -1.0, 1.0)

    a1 = action[1]
    if cbf.enable_terrain:
        gamma_from_roc = jnp.arcsin(jnp.clip(af.roc_max / jnp.maximum(speed, af.v_stall), -1.0, 1.0))
        gamma_lim = jnp.minimum(af.gamma_max, gamma_from_roc)
        gamma_min = _terrain_constraint(cbf, cfg, hmap, pos, psi, speed, gamma)
        a1 = jnp.maximum(a1, jnp.clip(gamma_min / gamma_lim, -1.0, 1.0))

    return jnp.stack([a0, a1, a2]), feasible


def make_filter(cbf: CBFConfig, cfg: EnvConfig):
    """Vectorised over the blue cohort. `known_*` come from the observation, so
    the filter only ever uses envelopes the aircraft has actually sensed --
    it is not allowed to cheat with ground truth.
    """

    def filt(hmap, actions, air, known_centers, known_radii, known_active):
        return jax.vmap(filter_action, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
            cbf, cfg, hmap, actions, air.pos, air.psi, air.gamma, air.phi, air.speed,
            known_centers, known_radii, known_active,
        )

    return filt
