"""The terrain-resolution benchmark's pure functions.

`naigos/bench/terrain_resolution.py` is split so that everything deciding what a
measurement *means* is NumPy-only, with JAX, the DEM cache and the clock confined
to the measurement half. That split is what makes these tests possible without a
cache, and it is asserted here too: a benchmark whose arithmetic is only
exercised by running the benchmark is a benchmark whose arithmetic is untested.

The important cases are not the happy paths. They are:

  * `recommend` must be a pure function of the records plus a stated policy, and
    must never claim a configuration passed on an AOI it was not measured on;
  * `PUBLISHED_DEFAULTS` must actually match the shipped defaults, so that
    changing a default without re-running the published results fails here;
  * `los_agreement` must keep false-visible and false-masked apart, because they
    are not symmetric errors and averaging them into an accuracy number is how
    the coarse-grid bias becomes invisible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from naigos.bench.terrain_resolution import (
    PUBLISHED_DEFAULTS,
    GridPlan,
    ResolutionPolicy,
    diagonal_m,
    height_agreement,
    los_agreement,
    plan_grid,
    ray_step_m,
    recommend,
    sample_ray_endpoints,
    sampling_adequacy,
    soft_visibility,
    throughput_sensitivity,
    throughput_stats,
)

# --- grid planning ------------------------------------------------------------


def test_plan_grid_floors_so_the_window_never_leaves_the_dem():
    # 99.7 km at 500 m is 199.4 cells. Rounding up would place the eastern column
    # outside the cached raster, which `TheatreTooSmall` exists to prevent.
    p = plan_grid(99_700.0, 134_600.0, 500.0)
    assert (p.nx, p.ny) == (199, 269)
    assert p.extent_x_m == 198 * 500.0
    assert p.extent_y_m == 268 * 500.0
    assert p.cells == 199 * 269
    assert p.heightmap_bytes == 199 * 269 * 4


def test_plan_grid_scales_memory_with_the_square_of_refinement():
    coarse = plan_grid(100_000.0, 100_000.0, 1000.0)
    fine = plan_grid(100_000.0, 100_000.0, 100.0)
    assert fine.heightmap_bytes == pytest.approx(coarse.heightmap_bytes * 100, rel=0.05)


def test_plan_grid_refuses_a_cell_size_that_leaves_no_surface():
    with pytest.raises(ValueError, match="not a surface"):
        plan_grid(1_000.0, 1_000.0, 900.0)
    with pytest.raises(ValueError, match="positive"):
        plan_grid(1_000.0, 1_000.0, 0.0)


def test_grid_plan_round_trips_through_its_dict():
    p = plan_grid(50_000.0, 40_000.0, 250.0)
    assert GridPlan(**p.as_dict()) == p


# --- sampling adequacy --------------------------------------------------------


def test_ray_step_and_sampling_adequacy_are_the_documented_ratio():
    # The failure the DEVLOG records: a 128 km ray at 24 samples steps 5.3 km.
    assert ray_step_m(128_000.0, 24) == pytest.approx(5_333.3, abs=1.0)
    # At 1500 m cells that march skips ~3.6 cells per sample.
    assert sampling_adequacy(128_000.0, 24, 1500.0) == pytest.approx(3.56, abs=0.01)
    # Refining the grid makes adequacy *worse* at a fixed sample count, which is
    # exactly why the two knobs have to be chosen together.
    assert sampling_adequacy(128_000.0, 24, 100.0) > sampling_adequacy(128_000.0, 24, 1500.0)


def test_sampling_adequacy_improves_linearly_with_sample_count():
    a = sampling_adequacy(100_000.0, 96, 500.0)
    b = sampling_adequacy(100_000.0, 192, 500.0)
    assert b == pytest.approx(a / 2.0)


def test_ray_step_rejects_a_march_with_no_samples():
    with pytest.raises(ValueError):
        ray_step_m(1000.0, 0)


def test_diagonal_is_the_longest_ray_a_rectangle_holds():
    assert diagonal_m(3.0, 4.0) == pytest.approx(5.0)
    assert diagonal_m(99_000.0, 134_000.0) == pytest.approx(math.hypot(99_000.0, 134_000.0))


# --- throughput ---------------------------------------------------------------


def test_throughput_headline_is_best_of_n_not_the_median():
    """Interference only ever adds time, so the fastest run is the estimate."""
    s = throughput_stats([0.10, 0.20, 0.40], n_worlds=10, n_steps=100, n_blue=4)
    assert s["estimator"] == "best-of-n"
    assert s["best_s"] == 0.10
    assert s["world_steps_per_s"] == pytest.approx(1000 / 0.10)
    # the median is still reported, and it is a different number
    assert s["median_agent_steps_per_s"] == pytest.approx(1000 * 4 / 0.20)
    assert s["agent_steps_per_s"] > s["median_agent_steps_per_s"]


def test_throughput_agent_steps_are_blue_times_world_steps():
    s = throughput_stats([0.5], n_worlds=8, n_steps=64, n_blue=6)
    assert s["agent_steps_per_s"] == pytest.approx(s["world_steps_per_s"] * 6)


def test_throughput_spread_makes_a_contended_run_visible():
    quiet = throughput_stats([0.100, 0.101, 0.102], 1, 1, 1)
    noisy = throughput_stats([0.100, 0.180, 0.260], 1, 1, 1)
    assert quiet["spread_frac"] < 0.05
    assert noisy["spread_frac"] > 1.0
    # ... and does not move the headline, which is what best-of-n buys.
    assert quiet["world_steps_per_s"] == pytest.approx(noisy["world_steps_per_s"])


def test_throughput_rejects_empty_and_nonpositive_samples():
    with pytest.raises(ValueError, match="no timing samples"):
        throughput_stats([], 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        throughput_stats([0.1, 0.0], 1, 1, 1)


# --- line-of-sight agreement --------------------------------------------------


def test_soft_visibility_is_the_envs_sigmoid():
    assert soft_visibility(np.array([0.0]), 60.0)[0] == pytest.approx(0.5)
    assert soft_visibility(np.array([600.0]), 60.0)[0] > 0.99
    assert soft_visibility(np.array([-600.0]), 60.0)[0] < 0.01
    # monotone in clearance: the gradient the reward shaping relies on
    xs = np.linspace(-300, 300, 50)
    assert np.all(np.diff(soft_visibility(xs, 60.0)) > 0)


def test_los_agreement_keeps_false_visible_and_false_masked_apart():
    # ref: masked, masked, clear, clear.  grid: says clear on the first (the
    # coarse-grid failure mode) and masked on the last (the opposite error).
    ref = np.array([-100.0, -100.0, 100.0, 100.0])
    grid = np.array([50.0, -100.0, 100.0, -50.0])
    a = los_agreement(ref, grid, clearance_scale_m=60.0)
    assert a["n_rays"] == 4
    assert a["false_visible_rate"] == 0.25
    assert a["false_masked_rate"] == 0.25
    assert a["sign_agreement"] == 0.5
    assert a["ref_visible_fraction"] == 0.5
    assert a["grid_visible_fraction"] == 0.5
    # equal and opposite here, so the visible-fraction error cancels -- which is
    # precisely why the two rates are reported instead of just this one.
    assert a["visible_fraction_error_pp"] == pytest.approx(0.0)


def test_los_agreement_reports_the_coarse_grid_bias_direction():
    """A grid that smooths ridges away over-reports visibility, and says so."""
    ref = np.full(100, -50.0)
    grid = ref.copy()
    grid[:10] = 25.0  # ten rays the coarse surface fails to block
    a = los_agreement(ref, grid)
    assert a["false_visible_rate"] == pytest.approx(0.10)
    assert a["false_masked_rate"] == 0.0
    assert a["clearance_bias_m"] > 0
    assert a["soft_visibility_bias"] > 0
    assert a["visible_fraction_error_pp"] == pytest.approx(10.0)


def test_los_agreement_drops_no_data_rays_rather_than_filling_them():
    ref = np.array([10.0, np.nan, -10.0])
    grid = np.array([10.0, 5.0, np.nan])
    a = los_agreement(ref, grid)
    assert a["n_rays"] == 1
    assert a["n_discarded"] == 2
    assert a["sign_agreement"] == 1.0


def test_los_agreement_rejects_mismatched_and_empty_inputs():
    with pytest.raises(ValueError, match="shape mismatch"):
        los_agreement(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="no finite"):
        los_agreement(np.array([np.nan]), np.array([np.nan]))


def test_perfect_agreement_scores_perfectly():
    ref = np.linspace(-500, 500, 101)
    a = los_agreement(ref, ref.copy())
    assert a["sign_agreement"] == 1.0
    assert a["false_visible_rate"] == 0.0
    assert a["clearance_mae_m"] == 0.0
    assert a["soft_visibility_mae"] == 0.0


def test_height_agreement_reports_the_tail_not_just_the_mean():
    err = np.zeros(100)
    err[-1] = 200.0  # one bad cell, e.g. a ridge crest lost to resampling
    a = height_agreement(np.zeros(100), err)
    assert a["mean_abs_m"] == pytest.approx(2.0)
    assert a["max_abs_m"] == 200.0
    assert a["p95_abs_m"] < a["max_abs_m"]


# --- ray sampling -------------------------------------------------------------


def test_ray_endpoints_are_identical_across_cell_sizes():
    """The same geometry must be reused, or resolutions differ by sampling noise."""
    a = sample_ray_endpoints(100_000.0, 80_000.0, 64, seed=7)
    b = sample_ray_endpoints(100_000.0, 80_000.0, 64, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert sample_ray_endpoints(100_000.0, 80_000.0, 64, seed=8)[0].tolist() != a[0].tolist()


def test_ray_endpoints_stay_off_the_clamping_edge():
    ex, ey = 100_000.0, 80_000.0
    emitter, target = sample_ray_endpoints(ex, ey, 500, seed=0, inset_frac=0.02)
    for pts in (emitter, target):
        assert pts.shape == (500, 2)
        assert pts[:, 0].min() >= 0.02 * ex
        assert pts[:, 0].max() <= 0.98 * ex
        assert pts[:, 1].min() >= 0.02 * ey
        assert pts[:, 1].max() <= 0.98 * ey


def test_ray_endpoints_requires_at_least_one_ray():
    with pytest.raises(ValueError):
        sample_ray_endpoints(1000.0, 1000.0, 0, seed=0)


# --- the recommendation -------------------------------------------------------


def _record(aoi, cell_m, los_samples, *, agent_steps, false_visible, soft_mae=0.001,
            endpoint_p95=1.0, endpoint_build_s=0.01, payload=1024, endpoint_ns=(512,),
            ray_step_cells=1.0):
    return {
        "aoi": aoi,
        "cell_m": float(cell_m),
        "grid": {
            "nx": 10, "ny": 10, "cells": 100,
            "extent_x_m": 1e5, "extent_y_m": 1e5, "diagonal_m": 1.4e5,
            "heightmap_bytes": 400, "relief_m": [0.0, 1.0],
            "los_samples": int(los_samples),
            "ray_step_m_diagonal": 1000.0,
            "ray_step_cells_diagonal": float(ray_step_cells),
        },
        "throughput": {"agent_steps_per_s": float(agent_steps), "spread_frac": 0.01},
        "los": {"false_visible_rate": float(false_visible), "soft_visibility_mae": float(soft_mae)},
        "endpoints": [
            {"n": int(n), "build_s": float(endpoint_build_s), "payload_bytes": int(payload),
             "agreement": {"p95_abs_m": float(endpoint_p95)}}
            for n in endpoint_ns
        ],
    }


def test_recommendation_prefers_the_fastest_passing_configuration():
    records = [
        _record("a", 1500, 96, agent_steps=100_000, false_visible=0.05),  # too inaccurate
        _record("a", 500, 96, agent_steps=99_000, false_visible=0.010),  # passes, fast
        _record("a", 500, 384, agent_steps=25_000, false_visible=0.005),  # passes, slow
    ]
    r = recommend(records)
    assert (r["physics_cell_m"], r["physics_los_samples"]) == (500.0, 96)


def test_recommendation_breaks_a_throughput_tie_toward_the_coarser_grid():
    records = [
        _record("a", 500, 96, agent_steps=100_000, false_visible=0.01),
        _record("a", 100, 96, agent_steps=100_000, false_visible=0.01),
    ]
    assert recommend(records)["physics_cell_m"] == 500.0


def test_a_configuration_must_pass_on_every_aoi_measured():
    records = [
        _record("a", 500, 96, agent_steps=100_000, false_visible=0.010),
        _record("b", 500, 96, agent_steps=100_000, false_visible=0.050),  # fails here
        _record("a", 100, 96, agent_steps=100_000, false_visible=0.010),
        _record("b", 100, 96, agent_steps=100_000, false_visible=0.010),
    ]
    r = recommend(records)
    assert r["physics_cell_m"] == 100.0
    failed = next(c for c in r["physics_candidates"] if c["cell_m"] == 500.0)
    assert not failed["passes"] and any("b:" in f for f in failed["failures"])


def test_a_configuration_measured_on_one_aoi_only_cannot_pass():
    records = [
        _record("a", 500, 96, agent_steps=100_000, false_visible=0.001),
        _record("a", 100, 96, agent_steps=100_000, false_visible=0.001),
        _record("b", 100, 96, agent_steps=100_000, false_visible=0.001),
    ]
    r = recommend(records)
    partial = next(c for c in r["physics_candidates"] if c["cell_m"] == 500.0)
    assert not partial["passes"]
    assert any("not measured on b" in f for f in partial["failures"])
    assert r["physics_cell_m"] == 100.0


def test_throughput_is_judged_against_the_baseline_not_the_fastest_row():
    """A ratio taken against a sweep maximum follows whichever row ran cleanest."""
    policy = ResolutionPolicy(min_throughput_frac=0.80)
    records = [
        # baseline configuration, deliberately not the fastest row in the sweep
        _record("a", 1500, 96, agent_steps=100_000, false_visible=0.05),
        _record("a", 500, 96, agent_steps=90_000, false_visible=0.01),
        _record("a", 250, 96, agent_steps=200_000, false_visible=0.01),
    ]
    r = recommend(records, policy)
    assert r["throughput_baseline"]["a"]["agent_steps_per_s"] == 100_000
    assert r["throughput_baseline"]["a"]["is_the_published_configuration"] is True
    # 90k is 0.90x the baseline (passes) but only 0.45x the sweep maximum, which
    # would have rejected it.
    assert all(c["passes"] for c in r["physics_candidates"] if c["cell_m"] in (500.0, 250.0))


def test_baseline_falls_back_and_says_so_when_it_was_not_measured():
    records = [_record("a", 500, 96, agent_steps=100_000, false_visible=0.01)]
    r = recommend(records)
    assert r["throughput_baseline"]["a"]["is_the_published_configuration"] is False


def test_ray_step_is_advisory_and_never_rejects_a_configuration():
    """The march-step heuristic is a mechanism diagnostic, not the outcome."""
    records = [_record("a", 100, 96, agent_steps=100_000, false_visible=0.01,
                       ray_step_cells=17.4)]
    r = recommend(records)
    cand = r["physics_candidates"][0]
    assert cand["passes"]
    assert cand["advisories"] and "not resolved" in cand["advisories"][0]


def test_presentation_takes_the_finest_passing_grid_and_the_smallest_payload():
    records = [
        _record("a", 500, 96, agent_steps=1, false_visible=0.9, endpoint_p95=5.0,
                endpoint_ns=(512, 1024)),
        _record("a", 100, 96, agent_steps=1, false_visible=0.9, endpoint_p95=9.0,
                endpoint_ns=(512, 1024)),
    ]
    r = recommend(records)
    assert r["presentation_cell_m"] == 100.0
    assert r["presentation_endpoint_n"] == 512


def test_presentation_rejects_a_surface_that_renders_below_the_agl_floor():
    policy = ResolutionPolicy(max_endpoint_p95_error_m=15.0)
    records = [
        _record("a", 500, 96, agent_steps=1, false_visible=0.9, endpoint_p95=13.0),
        _record("a", 100, 96, agent_steps=1, false_visible=0.9, endpoint_p95=25.5),
    ]
    r = recommend(records, policy)
    assert r["presentation_cell_m"] == 500.0


def test_presentation_counts_each_endpoint_once_not_once_per_march_length():
    records = [
        _record("a", 500, s, agent_steps=1, false_visible=0.9, endpoint_p95=99.0)
        for s in (96, 192, 384)
    ]
    cand = recommend(records)["presentation_candidates"]
    assert len(cand) == 1
    assert len(cand[0]["failures"]) == 1  # not three copies of the same failure


def test_no_recommendation_is_made_when_nothing_passes():
    records = [_record("a", 1500, 96, agent_steps=100_000, false_visible=0.9,
                       soft_mae=0.9, endpoint_p95=999.0)]
    r = recommend(records)
    assert r["physics_cell_m"] is None
    assert r["presentation_cell_m"] is None


def test_the_policy_is_recorded_so_the_recommendation_can_be_disagreed_with():
    r = recommend([_record("a", 500, 96, agent_steps=1, false_visible=0.001)])
    assert r["policy"]["max_false_visible_rate"] == ResolutionPolicy().max_false_visible_rate
    assert r["policy"]["baseline_cell_m"] == ResolutionPolicy().baseline_cell_m


def test_recommend_needs_records():
    with pytest.raises(ValueError, match="no records"):
        recommend([])


# --- sensitivity --------------------------------------------------------------


def test_sensitivity_separates_the_grid_axis_from_the_march_axis():
    records = [
        _record("a", 1500, 96, agent_steps=100_000, false_visible=0.01),
        _record("a", 100, 96, agent_steps=98_000, false_visible=0.01),
        _record("a", 1500, 384, agent_steps=25_000, false_visible=0.01),
    ]
    s = throughput_sensitivity(records)
    by_cell = next(e for e in s["vs_cell_at_fixed_los_samples"] if e["los_samples"] == 96)
    assert by_cell["reference_cell_m"] == 1500.0
    assert [p["relative"] for p in by_cell["points"]] == pytest.approx([1.0, 0.98])
    by_march = next(e for e in s["vs_los_samples_at_fixed_cell"] if e["cell_m"] == 1500.0)
    assert by_march["reference_los_samples"] == 96
    assert [p["relative"] for p in by_march["points"]] == pytest.approx([1.0, 0.25])


# --- guards -------------------------------------------------------------------


def test_published_defaults_still_match_the_shipped_configuration():
    """Changing a default without re-running the published results fails here.

    `PUBLISHED_DEFAULTS` is what the recommendation is contrasted against. If it
    drifts from the code, the benchmark starts reporting "unchanged" about a
    configuration that changed, which is worse than reporting nothing.
    """
    import inspect

    from naigos.env.config import DetectionConfig, TerrainConfig
    from naigos.env.theatre_bridge import env_from_theatre

    assert TerrainConfig().cell == PUBLISHED_DEFAULTS["physics_cell_m"]
    assert DetectionConfig().los_samples == PUBLISHED_DEFAULTS["los_samples"]
    bridge_default = inspect.signature(env_from_theatre).parameters["cell_m"].default
    assert bridge_default == PUBLISHED_DEFAULTS["theatre_bridge_cell_m"]


def test_the_live_demo_cell_size_is_what_the_benchmark_says_it_is():
    import inspect

    pytest.importorskip("pyproj")
    from naigos.demo.live import build_terrain_grid, main  # noqa: F401
    from naigos.demo import live

    src = inspect.getsource(live.main)
    assert f'default={PUBLISHED_DEFAULTS["live_demo_cell_m"]}' in src, (
        "naigos.demo.live --cell-m no longer defaults to the value "
        "PUBLISHED_DEFAULTS records; the benchmark would misreport what shipped"
    )
    assert (
        inspect.signature(build_terrain_grid).parameters["n"].default
        == PUBLISHED_DEFAULTS["terrain_endpoint_n"]
    )


def test_the_env_does_not_depend_on_the_benchmark():
    """A benchmark must be able to measure the shipped configuration, not be it."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = [
        p.relative_to(root)
        for d in ("naigos/env", "naigos/rl", "naigos/data", "naigos/demo")
        for p in (root / d).rglob("*.py")
        if "naigos.bench" in p.read_text() or "from ..bench" in p.read_text()
    ]
    assert not offenders, f"benchmark imported by shipped code: {offenders}"


def test_the_benchmarks_pure_half_imports_without_jax(monkeypatch):
    """The NumPy-only split is load-bearing: these tests must not need a backend."""
    import inspect

    import naigos.bench.terrain_resolution as mod

    src = inspect.getsource(mod)
    header = src[: src.index("# --- measurement ---")]
    assert "import jax" not in header
    assert "\nimport numpy as np" in header
