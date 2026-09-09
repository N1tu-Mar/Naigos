"""Measure what a finer terrain grid actually costs and actually buys.

Terrain cell size is a single number that currently does three different jobs:
it sets the surface line-of-sight is computed against, it sets the size of every
DEM gather in the hot loop, and -- because `naigos.demo.live` passes one
`--cell-m` to `env_from_theatre` -- it also sets the surface the globe renders.
Those three jobs have different right answers, and until they are measured
separately the choice is a guess.

This module measures, per (AOI, cell size):

  * JAX compile time, split into trace/lower and backend compile;
  * steady-state throughput, reported as world-steps/s and agent-steps/s;
  * host RSS and (where the backend exposes it) device memory;
  * line-of-sight agreement against the **source 30 m DEM**, using the pure-numpy
    reference implementation in `naigos.data.terrain` rather than a coarser copy
    of the thing under test;
  * `/terrain` endpoint generation time, payload size and the disagreement
    between what the browser would interpolate and what `sample_height` returns.

Two deliberate constraints.

**Nothing here changes a default.** Every published number in this repo was
produced at a specific `cell_m`/`los_samples` pair; silently refining the grid
would invalidate them without re-running anything. The module reports a
recommendation and records the shipped defaults next to it.

**The reference is a different implementation, not a finer instance of the same
one.** `naigos.data.terrain.los_margin` is pure NumPy, marches one sample per
half pixel of the native 30 m raster, and shares no code with the JAX ray march
it is checking. Comparing the JAX path at 500 m against the JAX path at 100 m
would only measure self-consistency.

The pure functions (`plan_grid`, `throughput_stats`, `los_agreement`,
`sampling_adequacy`, `recommend`, ...) are NumPy-only and are unit-tested in
`tests/test_bench_terrain.py`. Everything that needs JAX, a DEM or a clock lives
below the `--- measurement ---` divider and imports its dependencies locally.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SCHEMA = "naigos.bench.terrain_resolution/1"

# The configuration every committed result in this repo was produced at. Recorded
# so a benchmark run can be compared against the published numbers rather than
# quietly replacing them. See docs/DEVLOG.md and next-steps.md.
PUBLISHED_DEFAULTS = {
    "physics_cell_m": 1500.0,  # naigos.env.config.TerrainConfig.cell
    "theatre_bridge_cell_m": 1500.0,  # naigos.env.theatre_bridge.env_from_theatre
    "live_demo_cell_m": 500.0,  # naigos.demo.live --cell-m
    "los_samples": 96,  # naigos.env.config.DetectionConfig.los_samples
    "terrain_endpoint_n": 512,  # naigos.demo.live.build_terrain_grid
    "note": (
        "docs/artifacts and next-steps.md report evaluation at 500 m / 96 samples; "
        "the library defaults remain 1500 m / 96 samples. A benchmark does not "
        "change either."
    ),
}


# --- pure: grid planning ------------------------------------------------------


@dataclass(frozen=True)
class GridPlan:
    """The ENU grid a cell size implies inside a DEM window, and what it costs."""

    cell_m: float
    nx: int
    ny: int
    extent_x_m: float
    extent_y_m: float
    cells: int
    heightmap_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)


def plan_grid(dem_width_m: float, dem_height_m: float, cell_m: float, dtype_bytes: int = 4) -> GridPlan:
    """Largest ENU grid at ``cell_m`` that fits inside a DEM window.

    Mirrors `naigos.data.enu.natural_grid_shape` -- floor division, so the grid
    never extends past the cached DEM -- but takes plain metres so it can be
    reasoned about, and tested, without a raster on disk.
    """
    if cell_m <= 0:
        raise ValueError("cell_m must be positive")
    nx = int(dem_width_m // cell_m)
    ny = int(dem_height_m // cell_m)
    if nx < 2 or ny < 2:
        raise ValueError(f"cell_m={cell_m} leaves a {nx}x{ny} grid, which is not a surface")
    return GridPlan(
        cell_m=float(cell_m),
        nx=nx,
        ny=ny,
        extent_x_m=(nx - 1) * float(cell_m),
        extent_y_m=(ny - 1) * float(cell_m),
        cells=nx * ny,
        heightmap_bytes=nx * ny * dtype_bytes,
    )


def ray_step_m(ray_len_m: float, los_samples: int) -> float:
    """Spacing between consecutive samples of one line-of-sight ray."""
    if los_samples < 1:
        raise ValueError("los_samples must be >= 1")
    return float(ray_len_m) / float(los_samples)


def sampling_adequacy(ray_len_m: float, los_samples: int, cell_m: float) -> float:
    """Ray-march step measured in terrain cells.

    Below 1.0 the march visits every cell the ray crosses. Above 1.0 it steps
    over whole cells, so refining the grid buys resolution the ray never looks
    at -- which is the trap `los_samples = 24` fell into (a 128 km ray sampled
    every 5.3 km walked straight past ridges) and the reason grid size and
    sample count have to be chosen together rather than one at a time.
    """
    return ray_step_m(ray_len_m, los_samples) / float(cell_m)


def diagonal_m(extent_x_m: float, extent_y_m: float) -> float:
    """Longest ray a theatre can contain -- the worst case for `sampling_adequacy`."""
    return float(math.hypot(extent_x_m, extent_y_m))


# --- pure: timing summary -----------------------------------------------------


def throughput_stats(
    wall_s: list[float] | np.ndarray,
    n_worlds: int,
    n_steps: int,
    n_blue: int,
) -> dict:
    """Steady-state throughput from repeated timings of one compiled rollout.

    ``wall_s`` must contain steady-state samples only -- the caller drops the
    first, compile-inclusive call.

    The headline is **best-of-N, not the mean or the median.** Interference only
    ever adds time: a scheduler preemption, a thermal step, another process on
    the same laptop. So the fastest observed run is the closest estimate of what
    the configuration costs, and the slower ones measure the machine rather than
    the code. This is not a cosmetic choice here -- an early version of this
    benchmark reported medians while a training job shared the machine, and the
    same configuration came back at 184k, 153k and 139k agent-steps/s across
    three runs, which is enough spread to invert a resolution decision. The
    median is still reported next to it, and ``spread_frac`` says how far apart
    they were, so a noisy run is visible rather than averaged into confidence.

    Two throughput definitions are reported side by side because the repo has
    quoted both: a *world-step* advances one environment by `dt`; an
    *agent-step* is one aircraft-decision, i.e. `n_blue` per world-step.
    """
    w = np.asarray(wall_s, dtype=np.float64)
    if w.size == 0:
        raise ValueError("no timing samples")
    if not np.all(w > 0):
        raise ValueError("timing samples must be positive")
    world_steps = float(n_worlds * n_steps)
    best = float(w.min())
    median = float(np.median(w))
    return {
        "n_samples": int(w.size),
        "estimator": "best-of-n",
        "best_s": best,
        "median_s": median,
        "max_s": float(w.max()),
        "spread_frac": float((w.max() - best) / best),
        "world_steps_per_s": world_steps / best,
        "agent_steps_per_s": world_steps * float(n_blue) / best,
        "median_world_steps_per_s": world_steps / median,
        "median_agent_steps_per_s": world_steps * float(n_blue) / median,
    }


# --- pure: line-of-sight agreement -------------------------------------------


def soft_visibility(clearance_m: np.ndarray, clearance_scale_m: float) -> np.ndarray:
    """The env's soft LOS factor, in NumPy.

    Deliberately re-derived here rather than imported from `naigos.env.terrain`:
    the benchmark compares a JAX path against a NumPy reference, and a shared
    helper would let one bug cancel itself out on both sides of the comparison.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(clearance_m, dtype=np.float64) / float(clearance_scale_m)))


def los_agreement(
    ref_clearance_m: np.ndarray,
    grid_clearance_m: np.ndarray,
    clearance_scale_m: float = 60.0,
) -> dict:
    """Compare modelled line-of-sight clearance against the source-DEM reference.

    ``ref_clearance_m`` comes from `naigos.data.terrain.los_margin` on the native
    30 m raster; ``grid_clearance_m`` from `naigos.env.terrain.los_clearance` on
    the resampled ENU grid, for the identical pair of 3D endpoints.

    ``false_visible_rate`` is the metric that matters and it is not symmetric
    with its opposite. A coarse grid smooths ridgelines away, so its errors are
    biased toward declaring a masked aircraft visible -- which inflates measured
    exposure, makes terrain masking look weaker than it is, and cannot be
    corrected for after the fact. ``false_masked_rate`` is the reverse error and
    is reported separately rather than folded into one accuracy number.
    """
    ref = np.asarray(ref_clearance_m, dtype=np.float64)
    grid = np.asarray(grid_clearance_m, dtype=np.float64)
    if ref.shape != grid.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {grid.shape}")
    finite = np.isfinite(ref) & np.isfinite(grid)
    ref, grid = ref[finite], grid[finite]
    n = int(ref.size)
    if n == 0:
        raise ValueError("no finite ray pairs to compare")

    ref_vis = ref > 0.0
    grid_vis = grid > 0.0
    err = grid - ref
    soft_err = soft_visibility(grid, clearance_scale_m) - soft_visibility(ref, clearance_scale_m)

    return {
        "n_rays": n,
        "n_discarded": int(finite.size - n),
        "ref_visible_fraction": float(ref_vis.mean()),
        "grid_visible_fraction": float(grid_vis.mean()),
        "visible_fraction_error_pp": float((grid_vis.mean() - ref_vis.mean()) * 100.0),
        "sign_agreement": float((ref_vis == grid_vis).mean()),
        "false_visible_rate": float((grid_vis & ~ref_vis).mean()),
        "false_masked_rate": float((~grid_vis & ref_vis).mean()),
        "clearance_bias_m": float(err.mean()),
        "clearance_mae_m": float(np.abs(err).mean()),
        "clearance_p95_abs_m": float(np.percentile(np.abs(err), 95)),
        "clearance_max_abs_m": float(np.abs(err).max()),
        "soft_visibility_mae": float(np.abs(soft_err).mean()),
        "soft_visibility_bias": float(soft_err.mean()),
        "clearance_scale_m": float(clearance_scale_m),
    }


def height_agreement(ref_m: np.ndarray, grid_m: np.ndarray) -> dict:
    """Elevation-only agreement, for the presentation surface.

    A renderer does not care about ray geometry, only about whether the pixel it
    draws sits where the model thinks the ground is. Against a 30 m AGL floor,
    a p95 of tens of metres is the difference between an aircraft skimming a
    ridge and an aircraft inside it.
    """
    ref = np.asarray(ref_m, dtype=np.float64)
    grid = np.asarray(grid_m, dtype=np.float64)
    if ref.shape != grid.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {grid.shape}")
    err = np.abs(grid - ref)
    err = err[np.isfinite(err)]
    if err.size == 0:
        raise ValueError("no finite height pairs to compare")
    return {
        "n": int(err.size),
        "mean_abs_m": float(err.mean()),
        "p95_abs_m": float(np.percentile(err, 95)),
        "max_abs_m": float(err.max()),
    }


# --- pure: the recommendation -------------------------------------------------


def throughput_sensitivity(records: list[dict]) -> dict:
    """How throughput responds to each axis, separately.

    The two axes are not symmetric and conflating them is how the wrong knob
    gets turned. Refining the grid enlarges a gathered-from array; lengthening
    the ray march multiplies the number of gathers. Only the second shows up in
    wall clock, so "we cannot afford a finer DEM" and "we cannot afford a longer
    march" are different statements and only one of them is true here.

    Reported as ratios against the coarsest/cheapest measured point on each axis,
    per AOI, so the numbers are read as "x times slower" without needing the
    machine this ran on.
    """
    out: dict = {"vs_cell_at_fixed_los_samples": [], "vs_los_samples_at_fixed_cell": []}

    for (aoi, los), rows in sorted(_group(records, lambda r: (r["aoi"], int(r["grid"]["los_samples"]))).items()):
        rows = sorted(rows, key=lambda r: -float(r["cell_m"]))
        ref = rows[0]["throughput"]["agent_steps_per_s"]
        out["vs_cell_at_fixed_los_samples"].append({
            "aoi": aoi,
            "los_samples": los,
            "reference_cell_m": float(rows[0]["cell_m"]),
            "points": [
                {
                    "cell_m": float(r["cell_m"]),
                    "cells": int(r["grid"]["cells"]),
                    "agent_steps_per_s": r["throughput"]["agent_steps_per_s"],
                    "relative": r["throughput"]["agent_steps_per_s"] / ref,
                    "within_row_spread_frac": r["throughput"]["spread_frac"],
                }
                for r in rows
            ],
        })

    for (aoi, cell), rows in sorted(_group(records, lambda r: (r["aoi"], float(r["cell_m"]))).items()):
        rows = sorted(rows, key=lambda r: int(r["grid"]["los_samples"]))
        ref = rows[0]["throughput"]["agent_steps_per_s"]
        out["vs_los_samples_at_fixed_cell"].append({
            "aoi": aoi,
            "cell_m": cell,
            "reference_los_samples": int(rows[0]["grid"]["los_samples"]),
            "points": [
                {
                    "los_samples": int(r["grid"]["los_samples"]),
                    "agent_steps_per_s": r["throughput"]["agent_steps_per_s"],
                    "relative": r["throughput"]["agent_steps_per_s"] / ref,
                    "within_row_spread_frac": r["throughput"]["spread_frac"],
                }
                for r in rows
            ],
        })
    return out


@dataclass(frozen=True)
class ResolutionPolicy:
    """Thresholds that turn a table of measurements into a resolution choice.

    These are stated values, not measurements. They exist so that the
    recommendation is a function of the data plus an explicit policy, and so
    that disagreeing with the recommendation means disagreeing with a number
    written down here rather than with a judgement call.
    """

    # Line-of-sight accuracy the physics configuration must reach against the
    # 30 m DEM. `max_false_visible_rate` is the binding one: a configuration
    # that reports a masked aircraft as visible inflates measured exposure and
    # understates the mechanic the whole project is about.
    max_false_visible_rate: float = 0.02
    max_soft_visibility_mae: float = 0.05
    # A configuration retaining less than this fraction of the *baseline*
    # configuration's throughput is rejected however accurate it is: training
    # throughput is the budget. The baseline is the configuration the committed
    # results were produced at, not the fastest row in the sweep -- ratios taken
    # against a sweep maximum move with whichever row the laptop happened to
    # schedule well, and 0.85-vs-0.85 is not a decision anyone should ship.
    min_throughput_frac: float = 0.80
    baseline_cell_m: float = 1500.0
    baseline_los_samples: int = 96
    # Presentation: what the globe may spend building one /terrain payload, and
    # how far the rendered ground may sit from the modelled ground. The error
    # budget is set by the 30 m AGL floor -- past roughly half of it, a
    # nap-of-the-earth aircraft renders inside the hill it is hugging.
    max_endpoint_build_s: float = 1.0
    max_endpoint_payload_bytes: int = 2 * 1024 * 1024
    max_endpoint_p95_error_m: float = 15.0
    # Reported, not gated. The ray-march step in terrain cells is a *mechanism*
    # diagnostic; `false_visible_rate` is the outcome it causes, measured
    # end-to-end against the source DEM. Gating on both would count the same
    # defect twice and would reject configurations whose measured error is fine.
    advisory_ray_step_cells: float = 1.5


def _config_key(r: dict) -> tuple[float, int]:
    return (float(r["cell_m"]), int(r["grid"]["los_samples"]))


def _group(records: list[dict], key) -> dict:
    out: dict = {}
    for r in records:
        out.setdefault(key(r), []).append(r)
    return out


def recommend(records: list[dict], policy: ResolutionPolicy | None = None) -> dict:
    """Pick a physics configuration and a presentation cell size from measurements.

    The two answers are separate on purpose, and they are separate *questions*.

    Physics is a **(cell size, los_samples) pair**, not a cell size. Refining the
    grid without lengthening the march buys detail the ray steps over; length-
    ening the march on a coarse grid resolves a surface whose ridges were already
    smoothed away. Grading them jointly against the source DEM is the only way
    the trade is visible. Among configurations that pass, the fastest wins, with
    ties broken toward the coarser grid -- cheaper memory, cheaper compile, and
    no benefit measured for the extra cells.

    Presentation is a cell size alone, graded on how far the surface the browser
    interpolates sits from the surface `sample_height` returns, and on what it
    costs to serve. The finest passing grid wins, because here resolution is the
    product.

    A configuration must pass on **every** AOI measured. One theatre passing is
    not evidence about a resolution; it is evidence about that theatre.
    """
    policy = policy or ResolutionPolicy()
    if not records:
        raise ValueError("no records to recommend from")

    aois = sorted({r["aoi"] for r in records})

    # Throughput is judged against the baseline configuration on the same AOI,
    # falling back to the fastest measured row when the baseline was not part of
    # this sweep. `baseline_is_measured` says which, because the two mean
    # different things and a reader must not have to guess.
    baseline_throughput, baseline_is_measured = {}, {}
    for aoi in aois:
        rows = [r for r in records if r["aoi"] == aoi]
        base = [
            r for r in rows
            if float(r["cell_m"]) == policy.baseline_cell_m
            and int(r["grid"]["los_samples"]) == policy.baseline_los_samples
        ]
        baseline_is_measured[aoi] = bool(base)
        pool = base or rows
        baseline_throughput[aoi] = max(r["throughput"]["agent_steps_per_s"] for r in pool)

    physics = []
    for (cell_m, los_samples), rows in sorted(_group(records, _config_key).items()):
        failures, advisories, fracs = [], [], []
        covered = sorted({r["aoi"] for r in rows})
        if covered != aois:
            failures.append(f"not measured on {', '.join(a for a in aois if a not in covered)}")

        for r in rows:
            los, thr, grid = r["los"], r["throughput"], r["grid"]
            frac = thr["agent_steps_per_s"] / baseline_throughput[r["aoi"]]
            if los["false_visible_rate"] > policy.max_false_visible_rate:
                failures.append(
                    f"{r['aoi']}: false-visible {los['false_visible_rate']:.4f} "
                    f"> {policy.max_false_visible_rate:.4f}"
                )
            if los["soft_visibility_mae"] > policy.max_soft_visibility_mae:
                failures.append(
                    f"{r['aoi']}: soft-LOS MAE {los['soft_visibility_mae']:.4f} "
                    f"> {policy.max_soft_visibility_mae:.4f}"
                )
            fracs.append(frac)
            if frac < policy.min_throughput_frac:
                failures.append(
                    f"{r['aoi']}: throughput {frac:.2f}x baseline < {policy.min_throughput_frac:.2f}x"
                )
            if grid["ray_step_cells_diagonal"] > policy.advisory_ray_step_cells:
                advisories.append(
                    f"{r['aoi']}: the march steps {grid['ray_step_cells_diagonal']:.1f} cells "
                    f"per sample on the theatre diagonal, so some of this grid is not resolved"
                )

        physics.append({
            "cell_m": cell_m,
            "los_samples": los_samples,
            "passes": not failures,
            "failures": failures,
            "advisories": advisories,
            "min_agent_steps_per_s": min(r["throughput"]["agent_steps_per_s"] for r in rows),
            "min_throughput_frac_vs_baseline": min(fracs) if fracs else None,
            "worst_false_visible_rate": max(r["los"]["false_visible_rate"] for r in rows),
            "worst_soft_visibility_mae": max(r["los"]["soft_visibility_mae"] for r in rows),
        })

    # The /terrain payload does not depend on los_samples, so it is measured once
    # per (AOI, cell size, endpoint n) and deduplicated here rather than counted
    # once per ray-march row.
    seen: set = set()
    endpoint_rows: dict = {}
    for r in records:
        for end in r.get("endpoints") or []:
            key = (float(r["cell_m"]), int(end["n"]))
            tag = (r["aoi"], *key)
            if tag in seen:
                continue
            seen.add(tag)
            endpoint_rows.setdefault(key, []).append((r["aoi"], end))

    presentation = []
    for (cell_m, endpoint_n), rows in sorted(endpoint_rows.items()):
        failures = []
        covered = sorted({aoi for aoi, _ in rows})
        if covered != aois:
            failures.append(f"not measured on {', '.join(a for a in aois if a not in covered)}")
        for aoi, end in rows:
            if end["build_s"] > policy.max_endpoint_build_s:
                failures.append(
                    f"{aoi}: endpoint build {end['build_s']:.3f}s > {policy.max_endpoint_build_s:.2f}s"
                )
            if end["payload_bytes"] > policy.max_endpoint_payload_bytes:
                failures.append(
                    f"{aoi}: payload {end['payload_bytes']} B > {policy.max_endpoint_payload_bytes} B"
                )
            if end["agreement"]["p95_abs_m"] > policy.max_endpoint_p95_error_m:
                failures.append(
                    f"{aoi}: rendered-vs-modelled p95 {end['agreement']['p95_abs_m']:.1f} m "
                    f"> {policy.max_endpoint_p95_error_m:.1f} m"
                )
        presentation.append({
            "cell_m": cell_m,
            "endpoint_n": endpoint_n,
            "passes": not failures,
            "failures": failures,
            "worst_p95_abs_m": max(end["agreement"]["p95_abs_m"] for _, end in rows),
            "worst_build_s": max(end["build_s"] for _, end in rows),
            "payload_bytes": max(end["payload_bytes"] for _, end in rows),
        })

    # Fastest passing configuration; ties (throughput is nearly flat in cell
    # size, which is itself the headline finding) break toward the coarser grid.
    winners = [c for c in physics if c["passes"]]
    chosen = max(winners, key=lambda c: (c["min_agent_steps_per_s"], c["cell_m"])) if winners else None
    # Finest passing grid -- here resolution is the product -- and among ties the
    # smallest payload that still carries it.
    pres_ok = [c for c in presentation if c["passes"]]
    pres_chosen = min(pres_ok, key=lambda c: (c["cell_m"], c["endpoint_n"])) if pres_ok else None

    return {
        "policy": asdict(policy),
        "aois": aois,
        "throughput_baseline": {
            aoi: {
                "agent_steps_per_s": baseline_throughput[aoi],
                "is_the_published_configuration": baseline_is_measured[aoi],
            }
            for aoi in aois
        },
        "physics_cell_m": chosen["cell_m"] if chosen else None,
        "physics_los_samples": chosen["los_samples"] if chosen else None,
        "presentation_cell_m": pres_chosen["cell_m"] if pres_chosen else None,
        "presentation_endpoint_n": pres_chosen["endpoint_n"] if pres_chosen else None,
        "physics_candidates": physics,
        "presentation_candidates": presentation,
        "published_defaults": PUBLISHED_DEFAULTS,
        "applied": False,
        "applied_note": (
            "Advisory only. No default under naigos/ was changed by this run; the "
            "committed RL results were produced at the defaults recorded above, "
            "and adopting a recommendation means re-running them."
        ),
    }


# --- measurement --------------------------------------------------------------
# Everything below needs JAX, a cached DEM and a clock. Imports are local so the
# pure half of this module stays importable with NumPy alone.


def _host_rss_bytes() -> int | None:
    """Peak resident set size, in bytes, or None if the platform will not say."""
    import resource
    import sys

    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):  # pragma: no cover - platform dependent
        return None
    # macOS reports bytes; Linux reports kilobytes. Nothing in the struct says
    # which, so it is keyed off the platform rather than guessed from magnitude.
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def _device_memory_stats() -> dict | None:
    """Device allocator stats where the backend exposes them (CPU usually will not)."""
    import jax

    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # pragma: no cover - backend dependent
        return None
    if not stats:
        return None
    keep = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    return {k: int(v) for k, v in stats.items() if k in keep}


def sample_ray_endpoints(
    extent_x_m: float,
    extent_y_m: float,
    n_rays: int,
    seed: int,
    inset_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (emitter_xy, target_xy) pairs in local ENU metres.

    Seeded and inset from the boundary so the identical geometry is reused for
    every cell size and every AOI, and so no ray starts or ends in the half cell
    where `sample_height` clamps. Comparing resolutions on independently drawn
    ray sets would put sampling noise into the difference being measured.
    """
    if n_rays < 1:
        raise ValueError("n_rays must be >= 1")
    rng = np.random.default_rng(seed)
    lo_x, hi_x = inset_frac * extent_x_m, (1.0 - inset_frac) * extent_x_m
    lo_y, hi_y = inset_frac * extent_y_m, (1.0 - inset_frac) * extent_y_m
    emitter = np.stack([rng.uniform(lo_x, hi_x, n_rays), rng.uniform(lo_y, hi_y, n_rays)], axis=-1)
    target = np.stack([rng.uniform(lo_x, hi_x, n_rays), rng.uniform(lo_y, hi_y, n_rays)], axis=-1)
    return emitter, target


def measure_los_agreement(
    enu,
    theatre,
    tcfg,
    dcfg,
    n_rays: int = 2000,
    seed: int = 0,
    emitter_agl_m: float = 10.0,
    target_agl_m: float = 150.0,
) -> dict:
    """Ray-by-ray agreement between the ENU grid and the source 30 m DEM.

    Both ends of every ray are placed at a fixed AGL above the **reference**
    terrain, so the two models are asked about the same physical geometry rather
    than about their own resampled idea of where the ground is. Rays whose
    endpoints fall on DEM no-data are dropped, not filled.
    """
    import jax.numpy as jnp

    from ..data.terrain import los_margin
    from ..env.terrain import los_clearance

    emitter_xy, target_xy = sample_ray_endpoints(tcfg.extent_x, tcfg.extent_y, n_rays, seed)
    e_east, e_north = enu.to_utm(emitter_xy[:, 0], emitter_xy[:, 1])
    t_east, t_north = enu.to_utm(target_xy[:, 0], target_xy[:, 1])

    ref_grid = theatre.terrain
    k = float(theatre.effective_earth_factor_k)
    e_ground = ref_grid.elevation(e_east, e_north)
    t_ground = ref_grid.elevation(t_east, t_north)
    ok = np.isfinite(e_ground) & np.isfinite(t_ground)

    e_z = e_ground + emitter_agl_m
    t_z = t_ground + target_agl_m

    ref = np.full(n_rays, np.nan)
    for i in np.flatnonzero(ok):
        ref[i] = los_margin(
            ref_grid,
            (float(e_east[i]), float(e_north[i]), float(e_z[i])),
            (float(t_east[i]), float(t_north[i]), float(t_z[i])),
            k=k,
        )[0]

    p_from = jnp.asarray(np.stack([emitter_xy[:, 0], emitter_xy[:, 1], np.nan_to_num(e_z)], axis=-1))
    p_to = jnp.asarray(np.stack([target_xy[:, 0], target_xy[:, 1], np.nan_to_num(t_z)], axis=-1))
    grid = np.asarray(los_clearance(jnp.asarray(enu.heightmap), tcfg, dcfg, p_from, p_to))
    grid = np.where(ok, grid, np.nan)

    out = los_agreement(ref, grid, clearance_scale_m=dcfg.los_clearance_scale)
    out.update(
        reference="naigos.data.terrain.los_margin on the native 30 m raster",
        reference_pixel_m=float(ref_grid.pixel_m),
        emitter_agl_m=float(emitter_agl_m),
        target_agl_m=float(target_agl_m),
        effective_earth_k=k,
        los_samples=int(dcfg.los_samples),
        seed=int(seed),
        mean_ray_len_m=float(np.hypot(t_east - e_east, t_north - e_north)[ok].mean()),
    )
    return out


def measure_terrain_endpoint(hmap, tcfg, georef, geo_bounds: dict, n: int, repeats: int = 3,
                             n_probe: int = 4000, seed: int = 0) -> dict:
    """Time `/terrain` payload generation and measure what the browser would draw.

    The agreement figure re-implements the client-side bilinear lookup from
    `naigos/demo/assets/cesium.html` and compares it against `sample_height` at
    random points -- the same check `tests/test_terrain_endpoint.py` makes, run
    here as a measurement across cell sizes instead of as a single assertion.
    """
    import time

    import jax.numpy as jnp

    from ..demo.live import build_terrain_grid
    from ..env.terrain import sample_height

    timings = []
    body = meta = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        body, meta = build_terrain_grid(np.asarray(hmap), tcfg, georef, geo_bounds, n=n)
        timings.append(time.perf_counter() - t0)

    Z = np.frombuffer(body, dtype="<i2").reshape(meta["ny"], meta["nx"]).astype(np.float64)

    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, tcfg.extent_x, n_probe)
    ys = rng.uniform(0.0, tcfg.extent_y, n_probe)
    modelled = np.asarray(sample_height(jnp.asarray(hmap), tcfg, jnp.asarray(xs), jnp.asarray(ys)))
    lon, lat = georef.to_wgs84(xs, ys)
    drawn = _client_bilinear(Z, meta, lon, lat)

    return {
        "n": int(n),
        # best-of-n, for the same reason as `throughput_stats`: interference only
        # ever adds time, so the fastest run is the closest estimate of the cost.
        "build_s": float(np.min(timings)),
        "build_s_median": float(np.median(timings)),
        "build_s_samples": [float(t) for t in timings],
        "payload_bytes": int(len(body)),
        "agreement": height_agreement(modelled, drawn),
        "meta": {k: meta[k] for k in ("nx", "ny", "min_m", "max_m", "inside_fraction", "cell_m", "grid")},
    }


def _client_bilinear(Z: np.ndarray, meta: dict, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The `sampleGrid` lookup in `naigos/demo/assets/cesium.html`, in NumPy."""
    fx = np.clip((lon - meta["west"]) / (meta["east"] - meta["west"]) * (meta["nx"] - 1), 0, meta["nx"] - 1)
    fy = np.clip((lat - meta["south"]) / (meta["north"] - meta["south"]) * (meta["ny"] - 1), 0, meta["ny"] - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, meta["nx"] - 1), np.minimum(y0 + 1, meta["ny"] - 1)
    tx, ty = fx - x0, fy - y0
    top = Z[y0, x0] * (1 - tx) + Z[y0, x1] * tx
    bot = Z[y1, x0] * (1 - tx) + Z[y1, x1] * tx
    return top * (1 - ty) + bot * ty


def constant_policy(cfg):
    """A zero-action policy, so the timing measures the env and not a network.

    Throughput here is deliberately *not* comparable to a training iteration: it
    excludes the actor forward pass and the PPO update. It is comparable across
    cell sizes, which is the only comparison this benchmark makes.
    """
    import jax.numpy as jnp

    def policy(obs, key):
        return jnp.zeros((cfg.n_blue, cfg.action_dim), dtype=jnp.float32)

    return policy


def measure_rollout(cfg, hmap, n_worlds: int, n_steps: int, repeats: int, seed: int = 0) -> dict:
    """Compile time and steady-state throughput for a vmapped rollout.

    Compile time is taken from an explicit ahead-of-time `lower()` / `compile()`
    rather than by timing the first call, because a first call also traces,
    allocates and executes; those are three different costs and a single number
    for all of them is not actionable.
    """
    import time

    import jax

    from ..env.flight_env import NaigosEnv

    env = NaigosEnv(cfg, hmap=hmap)
    policy = constant_policy(cfg)

    def one(key):
        _, traj = env.rollout(key, policy, n_steps=n_steps)
        # Reduce inside the jit: returning the full trajectory would time the
        # device-to-host transfer of a buffer that grows with n_steps, not the
        # simulation.
        return traj["alive"].sum(), traj["alt_agl"].mean()

    fn = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.PRNGKey(seed), n_worlds)

    t0 = time.perf_counter()
    lowered = fn.lower(keys)
    t_lower = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    jax.block_until_ready(compiled(keys))
    t_first = time.perf_counter() - t0

    wall = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(keys))
        wall.append(time.perf_counter() - t0)

    return {
        "compile": {
            "trace_lower_s": float(t_lower),
            "backend_compile_s": float(t_compile),
            "total_s": float(t_lower + t_compile),
            "first_call_s": float(t_first),
        },
        "throughput": throughput_stats(wall, n_worlds, n_steps, cfg.n_blue),
        "shape": {
            "n_worlds": int(n_worlds),
            "n_steps": int(n_steps),
            "n_blue": int(cfg.n_blue),
            "n_threat": int(cfg.n_threat),
            "los_samples": int(cfg.detection.los_samples),
        },
    }


def benchmark_cell(
    aoi: str,
    cell_m: float,
    *,
    n_worlds: int,
    n_steps: int,
    n_blue: int,
    n_threat: int,
    repeats: int,
    n_rays: int,
    endpoint_ns: list[int],
    seed: int,
    los_samples: int | None = None,
) -> dict:
    """One (AOI, cell size) row: grid, compile, throughput, memory, LOS, endpoint."""
    import dataclasses

    from ..data.enu import real_terrain
    from ..data.geodetic import georef_from_enu
    from ..data.theatre import load_theatre
    from ..env.theatre_bridge import env_from_theatre

    theatre = load_theatre(aoi)
    cfg, hmap, notes = env_from_theatre(
        theatre=theatre, n_blue=n_blue, n_threat=n_threat, cell_m=cell_m
    )
    if los_samples is not None:
        cfg = cfg.replace(detection=dataclasses.replace(cfg.detection, los_samples=int(los_samples)))

    tcfg = cfg.terrain
    enu = real_terrain(nx=tcfg.nx, ny=tcfg.ny, cell_m=tcfg.cell, grid=theatre.terrain)
    diag = diagonal_m(tcfg.extent_x, tcfg.extent_y)

    rss_before = _host_rss_bytes()
    rollout = measure_rollout(cfg, hmap, n_worlds, n_steps, repeats, seed=seed)
    rss_after = _host_rss_bytes()

    los = measure_los_agreement(
        enu, theatre, tcfg, cfg.detection, n_rays=n_rays, seed=seed
    )

    try:
        georef = georef_from_enu(enu)
        endpoints = [
            measure_terrain_endpoint(hmap, tcfg, georef, notes["geo_bounds"], n=n, seed=seed)
            for n in endpoint_ns
        ]
    except ImportError as e:  # pyproj is a demo-side dependency, not an env one
        endpoints = []
        endpoint_skip = f"{type(e).__name__}: {e}"
    else:
        endpoint_skip = None

    return {
        "aoi": aoi,
        "cell_m": float(tcfg.cell),
        "grid": {
            "nx": int(tcfg.nx),
            "ny": int(tcfg.ny),
            "cells": int(tcfg.nx * tcfg.ny),
            "extent_x_m": float(tcfg.extent_x),
            "extent_y_m": float(tcfg.extent_y),
            "diagonal_m": diag,
            "heightmap_bytes": int(np.asarray(hmap).nbytes),
            "relief_m": list(notes["relief_m"]),
            "los_samples": int(cfg.detection.los_samples),
            "ray_step_m_diagonal": ray_step_m(diag, cfg.detection.los_samples),
            "ray_step_cells_diagonal": sampling_adequacy(diag, cfg.detection.los_samples, tcfg.cell),
        },
        "compile": rollout["compile"],
        "throughput": rollout["throughput"],
        "rollout_shape": rollout["shape"],
        "memory": {
            "heightmap_bytes": int(np.asarray(hmap).nbytes),
            "host_rss_before_bytes": rss_before,
            "host_rss_peak_bytes": rss_after,
            "host_rss_delta_bytes": None if (rss_before is None or rss_after is None) else rss_after - rss_before,
            "device": _device_memory_stats(),
            "note": (
                "host_rss_peak is the process high-water mark, so it is monotonic "
                "across rows in one run and only the delta is attributable to this "
                "row. Device stats are null on the CPU backend, which does not "
                "expose an allocator."
            ),
        },
        "los": los,
        "endpoints": endpoints,
        "endpoint_skipped": endpoint_skip,
    }


def environment_report() -> dict:
    """What this run was measured on. A throughput number without it is not reproducible."""
    import os
    import platform
    import sys

    import jax

    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "devices": [str(d) for d in jax.devices()],
        "default_backend": jax.default_backend(),
    }


def run(
    aois: list[str],
    cell_sizes: list[float],
    *,
    n_worlds: int = 64,
    n_steps: int = 128,
    n_blue: int = 4,
    n_threat: int = 16,
    repeats: int = 5,
    n_rays: int = 2000,
    endpoint_ns: list[int] | None = None,
    seed: int = 0,
    los_samples: list[int] | None = None,
    policy: ResolutionPolicy | None = None,
    on_row=None,
) -> dict:
    """Run the whole (AOI x cell size x los_samples) sweep and return the report.

    ``los_samples=None`` measures the shipped `DetectionConfig.los_samples` only.
    Passing a list sweeps it, which is what makes the physics recommendation a
    configuration rather than a cell size.

    A cell size that does not fit the cached DEM is recorded as a skip carrying
    the raising message, not dropped: "100 m was not benchmarked" and "100 m does
    not fit this AOI" are different facts, and only one of them is about the AOI.
    """
    import datetime as _dt

    from ..data.enu import TheatreTooSmall

    sample_counts: list[int | None] = list(los_samples) if los_samples else [None]
    endpoint_ns = list(endpoint_ns) if endpoint_ns else [512]

    records, skipped = [], []
    for aoi in aois:
        for cell in cell_sizes:
            for s_count in sample_counts:
                try:
                    row = benchmark_cell(
                        aoi, cell,
                        n_worlds=n_worlds, n_steps=n_steps, n_blue=n_blue, n_threat=n_threat,
                        repeats=repeats, n_rays=n_rays, endpoint_ns=endpoint_ns, seed=seed,
                        los_samples=s_count,
                    )
                except (TheatreTooSmall, ValueError, FileNotFoundError) as e:
                    skipped.append({
                        "aoi": aoi, "cell_m": float(cell), "los_samples": s_count,
                        "reason": f"{type(e).__name__}: {e}",
                    })
                    if on_row is not None:
                        on_row(None, skipped[-1])
                    continue
                records.append(row)
                if on_row is not None:
                    on_row(row, None)

    return {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_report(),
        "settings": {
            "aois": list(aois),
            "cell_sizes_m": [float(c) for c in cell_sizes],
            "n_worlds": n_worlds,
            "n_steps": n_steps,
            "n_blue": n_blue,
            "n_threat": n_threat,
            "repeats": repeats,
            "n_rays": n_rays,
            "endpoint_ns": endpoint_ns,
            "seed": seed,
            "los_samples": sample_counts,
        },
        "records": records,
        "skipped": skipped,
        "throughput_sensitivity": throughput_sensitivity(records) if records else None,
        "recommendation": recommend(records, policy) if records else None,
    }
