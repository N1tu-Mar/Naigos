"""The CBF backstop must turn away from envelopes, hold the terrain floor, and
report infeasibility instead of pretending."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from naigos.env.config import EnvConfig
from naigos.env.terrain import sample_height, synthetic_terrain
from naigos.rl.cbf import CBFConfig, filter_action

CFG = EnvConfig()
CB = CBFConfig()
HMAP = synthetic_terrain(jax.random.PRNGKey(0), CFG.terrain)
NONE = (jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros(1, dtype=bool))


def _filt(action, pos, psi=0.0, speed=200.0, centers=None, radii=None, active=None, cbf=CB):
    c, r, a = NONE if centers is None else (centers, radii, active)
    return filter_action(
        cbf, CFG, HMAP, action, pos, jnp.float32(psi), jnp.float32(0.0), jnp.float32(0.0),
        jnp.float32(speed), c, r, a,
    )


def test_no_known_threat_leaves_the_action_untouched():
    a = jnp.array([0.4, 0.1, 0.5])
    safe, feasible = _filt(a, jnp.array([50_000.0, 50_000.0, 6_000.0]))
    assert bool(feasible)
    assert float(jnp.abs(safe[0] - a[0])) < 1e-4
    assert float(jnp.abs(safe[2] - a[2])) < 1e-4


def test_filter_banks_away_from_a_lethal_envelope_it_would_clip():
    centers = jnp.array([[100_000.0, 50_000.0, 1_000.0]])
    radii = jnp.array([18_000.0])
    active = jnp.array([True])
    # flying due east, 8 km south of the site: on course to clip the envelope
    safe, feasible = _filt(
        jnp.array([0.0, 0.0, 0.5]), jnp.array([65_000.0, 42_000.0, 4_000.0]),
        centers=centers, radii=radii, active=active,
    )
    assert bool(feasible)
    assert float(safe[0]) < -0.05, "should bank away from the site, not fly through it"


def test_filter_ignores_an_envelope_it_will_miss():
    centers = jnp.array([[100_000.0, 50_000.0, 1_000.0]])
    radii = jnp.array([18_000.0])
    active = jnp.array([True])
    safe, _ = _filt(
        jnp.array([0.0, 0.0, 0.5]), jnp.array([65_000.0, 5_000.0, 4_000.0]),
        centers=centers, radii=radii, active=active,
    )
    assert abs(float(safe[0])) < 1e-3


def test_unknown_envelopes_impose_nothing():
    """The filter may only use envelopes the aircraft has SENSED."""
    centers = jnp.array([[70_000.0, 50_000.0, 1_000.0]])
    radii = jnp.array([18_000.0])
    a = jnp.array([0.0, 0.0, 0.5])
    seen, _ = _filt(a, jnp.array([60_000.0, 44_000.0, 4_000.0]), centers=centers, radii=radii,
                    active=jnp.array([True]))
    unseen, _ = _filt(a, jnp.array([60_000.0, 44_000.0, 4_000.0]), centers=centers, radii=radii,
                      active=jnp.array([False]))
    assert float(jnp.abs(unseen[0] - a[0])) < 1e-4
    assert float(jnp.abs(seen[0] - a[0])) > 1e-3


def test_terrain_floor_converts_a_commanded_dive_into_a_climb():
    g = float(sample_height(HMAP, CFG.terrain, 30_000.0, 30_000.0))
    safe, _ = _filt(jnp.array([0.0, -1.0, 0.5]), jnp.array([30_000.0, 30_000.0, g + 40.0]))
    assert float(safe[1]) > 0.0


def test_terrain_floor_does_not_interfere_at_altitude():
    g = float(sample_height(HMAP, CFG.terrain, 30_000.0, 30_000.0))
    safe, _ = _filt(jnp.array([0.0, -1.0, 0.5]), jnp.array([30_000.0, 30_000.0, g + 6_000.0]))
    assert float(safe[1]) < -0.5


def test_head_on_at_close_range_is_reported_infeasible_not_faked():
    """Exactly head-on the lateral constraint coefficient is zero; the filter has
    to say so rather than returning an action that does not satisfy the barrier."""
    centers = jnp.array([[70_000.0, 50_000.0, 1_000.0]])
    radii = jnp.array([18_000.0])
    safe, feasible = _filt(
        jnp.array([0.0, 0.0, 0.5]), jnp.array([50_500.0, 50_000.0, 4_000.0]),
        centers=centers, radii=radii, active=jnp.array([True]),
    )
    assert not bool(feasible)
    assert bool(jnp.all(jnp.isfinite(safe)))


def test_output_stays_inside_the_action_box():
    key = jax.random.PRNGKey(0)
    for i in range(30):
        k1, k2, key = jax.random.split(key, 3)
        a = jax.random.uniform(k1, (3,), minval=-1.0, maxval=1.0)
        p = jax.random.uniform(k2, (3,)) * jnp.array([90_000.0, 90_000.0, 8_000.0])
        centers = jnp.array([[45_000.0, 45_000.0, 1_000.0]])
        safe, _ = _filt(a, p, centers=centers, radii=jnp.array([20_000.0]), active=jnp.array([True]))
        assert bool(jnp.all(safe >= -1.0 - 1e-5)) and bool(jnp.all(safe <= 1.0 + 1e-5))


def test_filter_is_jit_able():
    f = jax.jit(lambda a, p: _filt(a, p))
    safe, feasible = f(jnp.array([0.2, 0.0, 0.5]), jnp.array([40_000.0, 40_000.0, 5_000.0]))
    assert safe.shape == (3,)


def test_policy_filter_only_uses_sensed_envelopes():
    """The backstop reads envelopes out of the OBSERVATION, never ground truth."""
    import inspect

    from naigos.rl.cbf import make_policy_filter

    src = inspect.getsource(make_policy_filter)
    assert "known_envelopes" in src
    assert "state.threats" not in src, "the filter must not read the ground-truth threat list"


def test_policy_filter_holds_the_terrain_floor_in_a_rollout():
    """A/B on identical seeds. Asserted on minimum AGL rather than on the
    terrain-loss count: the synthetic map's relief is mild enough that a naive
    nap-of-the-earth controller does not actually hit it, so a loss count would
    be 0 on both sides and prove nothing. The real-theatre loss reduction is
    measured in tests/test_theatre_bridge.py."""
    from naigos.env.flight_env import NaigosEnv
    from naigos.rl.cbf import make_policy_filter

    cfg = EnvConfig(n_blue=4, n_threat=10, n_threat_active=8, max_steps=250)
    env = NaigosEnv(cfg)

    def nap(o, k):
        h = jnp.arctan2(o.ego[:, 4], o.ego[:, 5])
        agl = o.ego[:, 2] * 5000.0
        return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.clip((150.0 - agl) / 300.0, -1, 1),
                          jnp.full_like(h, 0.6)], -1)

    keys = jax.random.split(jax.random.PRNGKey(5), 16)
    afilter = make_policy_filter(CB, cfg)
    fin_off, off = jax.jit(jax.vmap(lambda k: env.rollout(k, nap)))(keys)
    fin_on, on = jax.jit(jax.vmap(lambda k: env.rollout(k, nap, action_filter=afilter)))(keys)
    assert float(on["alt_agl"].min()) > float(off["alt_agl"].min())
    assert float(on["alt_agl"].min()) >= cfg.airframe.floor_agl
    assert float(on["terms"].terrain_violation.sum()) <= float(off["terms"].terrain_violation.sum())
    # and the envelope barrier should not be making survival worse
    assert float(fin_on.alive.mean()) >= float(fin_off.alive.mean())


def test_rollout_logs_feasibility_so_it_can_be_reported():
    from naigos.env.flight_env import NaigosEnv
    from naigos.rl.cbf import make_policy_filter

    cfg = EnvConfig(n_blue=2, n_threat=6, n_threat_active=4, max_steps=30)
    env = NaigosEnv(cfg)
    afilter = make_policy_filter(CB, cfg)
    _, traj = env.rollout(jax.random.PRNGKey(0), lambda o, k: jnp.zeros((2, 3)), action_filter=afilter)
    assert traj["cbf_feasible"].shape == (30, 2)
    assert traj["cbf_feasible"].dtype == jnp.bool_
