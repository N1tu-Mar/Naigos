"""The drawn LOS ray must be the curve the model marched, not a straight line.

`naigos.env.terrain.los_clearance` returns one number: the minimum clearance
along a ray whose every sample has been dropped for 4/3-earth refraction. The
viewer used to draw a two-point line between the emitter and the aircraft, which
is a different curve -- over an 80 km ray the drop reaches ~100 m at the
midpoint, so the drawn line could clear a ridge the number said it did not.

`naigos.demo.los` reconstructs the sampled ray. It shares no code with the env,
on the same reasoning as `naigos/rl/verifier.py`: a reconstruction that called
the env's own functions would agree by construction and would prove nothing. So
the agreement is asserted here instead, numerically, and this file is where the
two drifting apart shows up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.demo import los
from naigos.env.config import DetectionConfig, TerrainConfig
from naigos.env.terrain import los_clearance, sample_height, synthetic_terrain

TCFG = TerrainConfig()
DCFG = DetectionConfig()


@pytest.fixture(scope="module")
def hmap():
    return synthetic_terrain(jax.random.PRNGKey(3), TCFG)


def _rays(n=64, seed=11):
    """Emitter-to-aircraft pairs spread across the map, at realistic heights."""
    rng = np.random.default_rng(seed)
    ex = rng.uniform(0, TCFG.extent_x, n)
    ey = rng.uniform(0, TCFG.extent_y, n)
    ax = rng.uniform(0, TCFG.extent_x, n)
    ay = rng.uniform(0, TCFG.extent_y, n)
    return ex, ey, ax, ay


def _endpoints(hmap, ex, ey, ax, ay):
    """Emitter at ground+10 m, aircraft at ground+150 m -- the live geometry."""
    gh_e = np.asarray(sample_height(hmap, TCFG, jnp.asarray(ex), jnp.asarray(ey)))
    gh_a = np.asarray(sample_height(hmap, TCFG, jnp.asarray(ax), jnp.asarray(ay)))
    p_from = np.stack([ex, ey, gh_e + 10.0], axis=-1)
    p_to = np.stack([ax, ay, gh_a + 150.0], axis=-1)
    return p_from, p_to


def _ground_at_samples(hmap, p_from, p_to):
    """Sample the DEM where `los.profile` says the samples are."""
    s = los.sample_fractions(DCFG.los_samples)[None, :]
    seg = p_to - p_from
    gx = p_from[:, None, 0] + s * seg[:, None, 0]
    gy = p_from[:, None, 1] + s * seg[:, None, 1]
    return np.asarray(sample_height(hmap, TCFG, jnp.asarray(gx), jnp.asarray(gy)))


# --- the agreement ----------------------------------------------------------------------


def test_the_reconstructed_minimum_is_the_envs_clearance(hmap):
    """THE test. The polyline's worst sample is the number the model reduced to."""
    p_from, p_to = _endpoints(hmap, *_rays())
    ground = _ground_at_samples(hmap, p_from, p_to)

    prof = los.profile(p_from, p_to, ground,
                       los_samples=DCFG.los_samples, earth_radius_eff=DCFG.earth_radius_eff)
    env_min = np.asarray(los_clearance(hmap, TCFG, DCFG,
                                       jnp.asarray(p_from), jnp.asarray(p_to)))

    worst = float(np.abs(prof["min_clearance"] - env_min).max())
    # float32 in the env against float64 here, over kilometre-scale heights
    assert worst < 0.5, f"the drawn ray disagrees with the model by up to {worst:.3f} m"


def test_the_pinch_index_is_where_the_minimum_actually_is(hmap):
    p_from, p_to = _endpoints(hmap, *_rays(n=32, seed=5))
    ground = _ground_at_samples(hmap, p_from, p_to)
    prof = los.profile(p_from, p_to, ground,
                       los_samples=DCFG.los_samples, earth_radius_eff=DCFG.earth_radius_eff)
    for i, pinch in enumerate(prof["pinch"]):
        assert prof["clearance"][i, pinch] == prof["min_clearance"][i]


# --- the refraction is real, and a straight line would not have it ----------------------


def test_a_straight_line_would_disagree_on_a_long_ray(hmap):
    """Proves the whole exercise is not a rounding argument.

    An 80 km ray is ordinary here -- the Tehran AOI is ~80 km across and the
    long-range threat classes reach right over it.
    """
    p_from = np.array([[0.0, 0.0, 1000.0]])
    p_to = np.array([[80_000.0, 0.0, 3000.0]])
    ground = np.zeros((1, DCFG.los_samples))
    prof = los.profile(p_from, p_to, ground,
                       los_samples=DCFG.los_samples, earth_radius_eff=DCFG.earth_radius_eff)

    drop_peak = float(prof["drop"].max())
    assert 80.0 < drop_peak < 120.0, f"expected ~94 m of drop at 80 km, got {drop_peak:.1f} m"

    # the straight chord, for contrast: it sits a full drop_peak higher mid-ray
    s = los.sample_fractions(DCFG.los_samples)
    straight = p_from[0, 2] + s * (p_to[0, 2] - p_from[0, 2])
    assert float((straight - prof["z"][0]).max()) == pytest.approx(drop_peak, rel=1e-9)


def test_the_drop_vanishes_at_the_ends_and_peaks_in_the_middle():
    s = los.sample_fractions(96)
    drop = los.curvature_drop(s, np.array([50_000.0]), DCFG.earth_radius_eff)
    assert drop[0] < drop[len(drop) // 2] > drop[-1]
    assert drop[0] == pytest.approx(drop[-1], rel=1e-9)
    # d1*d2/(2Re) at the midpoint of a 50 km ray
    assert float(drop.max()) == pytest.approx(
        (50_000.0 / 2) ** 2 / (2 * DCFG.earth_radius_eff), rel=2e-3)


def test_a_short_ray_is_essentially_straight():
    """The correction must not be a constant fudge: at 5 km it is centimetres."""
    s = los.sample_fractions(96)
    drop = los.curvature_drop(s, np.array([5_000.0]), DCFG.earth_radius_eff)
    assert float(drop.max()) < 1.0


def test_the_sample_positions_match_the_envs_midpoint_rule():
    """`(i + 0.5) / n`. If this drifts, every clearance above is measured at
    different places than the model measured it."""
    s = los.sample_fractions(4)
    np.testing.assert_allclose(s, [0.125, 0.375, 0.625, 0.875])
    assert los.sample_fractions(96)[0] > 0.0
    assert los.sample_fractions(96)[-1] < 1.0


# --- what actually gets sent ------------------------------------------------------------


def test_the_drawn_vertices_always_include_the_pinch():
    """A smooth, plausible polyline missing the one vertex that carries the
    claim is the exact failure this subsampling could introduce."""
    for pinch in (0, 1, 17, 48, 95):
        idx = los.draw_indices(96, pinch)
        assert pinch in idx
        assert len(idx) <= los.DRAW_SAMPLES + 1
        assert list(idx) == sorted(set(idx)), "out-of-order vertices draw a bowtie"


def test_the_drawn_ray_keeps_both_ends():
    idx = los.draw_indices(96, 40)
    assert idx[0] == 0 and idx[-1] == 95


def test_a_short_march_is_sent_whole():
    assert list(los.draw_indices(8, 3)) == list(range(8))


def test_the_bands_are_the_sigmoid_scale_not_a_step():
    assert los.band(-500.0) == "masked"
    assert los.band(0.0) == "grazing"
    assert los.band(59.0) == "grazing"
    assert los.band(500.0) == "clear"
    assert los.CLEARANCE_GRAZING == DCFG.los_clearance_scale
    assert los.CLEARANCE_MASKED == -DCFG.los_clearance_scale


# --- the frame carries it ---------------------------------------------------------------


def test_the_module_does_not_reach_into_the_simulation():
    """The independence that makes the agreement test above mean something.

    Import statements only -- the module docstring names both packages on
    purpose, to explain why it does not import them.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "naigos" / "demo" / "los.py").read_text()
    modules = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
    assert modules == {"numpy", "__future__"}, f"unexpected dependencies: {sorted(modules)}"


def test_the_page_draws_the_sampled_ray_and_marks_the_block():
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1]
            / "naigos" / "demo" / "assets" / "cesium.html").read_text()
    # the polyline is built from the server's sampled vertices...
    assert "ray.pts.map(p => c3(p[0], p[1], p[2]))" in page
    # ...and never from a two-point emitter-to-aircraft line again
    assert "losPts[a.id] = [c3(tk[0], tk[1], tk[2] + 10), pos];" not in page
    # the pinch point is drawn, and only when the ray is actually cut
    assert "losBlocks" in page
    assert "if (ray.blocked && ray.pinch) {" in page
    # the colour bands come from the server, so they cannot drift from the number
    assert 'ray.band === "masked"' in page
