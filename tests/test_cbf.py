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
