#!/usr/bin/env python
"""Benchmark terrain resolution: what a finer grid costs and what it buys.

    uv run python scripts/bench_terrain.py                     # shipped sweep
    uv run python scripts/bench_terrain.py --quick             # ~30 s smoke run
    uv run python scripts/bench_terrain.py --aoi tehran_basin --cell-m 500 250

Writes a machine-readable report to `docs/artifacts/terrain_bench.json` and
prints a table. It changes no default anywhere in `naigos/`: the recommendation
it prints is advisory, and the configuration every committed result was produced
at is recorded in the report next to it.

Requires the DEM cache (`uv run naigos-research --aoi <name>`); AOI/cell
combinations that do not fit the cached DEM are recorded as skips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from naigos.bench.terrain_resolution import ResolutionPolicy, run  # noqa: E402

DEFAULT_OUT = Path("docs/artifacts/terrain_bench.json")
ROW = ("{aoi:<13} {cell:>6} {los:>4} {grid:>10} {step:>6} {compile:>8} {steps:>13} "
       "{vis:>8} {fv:>9} {mae:>7} {end:>8}")


def _header() -> str:
    return ROW.format(
        aoi="aoi", cell="cell m", los="S", grid="grid", step="ray/c", compile="cmpl s",
        steps="agent-steps/s", vis="vis frac", fv="false-vis", mae="softMAE", end="endpt s",
    )


def _row(r: dict) -> str:
    end = r.get("endpoint") or {}
    return ROW.format(
        aoi=r["aoi"][:13],
        cell=f"{r['cell_m']:.0f}",
        los=f"{r['grid']['los_samples']}",
        grid=f"{r['grid']['nx']}x{r['grid']['ny']}",
        step=f"{r['grid']['ray_step_cells_diagonal']:.1f}",
        compile=f"{r['compile']['total_s']:.2f}",
        steps=f"{r['throughput']['agent_steps_per_s']:,.0f}",
        vis=f"{r['los']['grid_visible_fraction']:.4f}",
        fv=f"{r['los']['false_visible_rate']:.4f}",
        mae=f"{r['los']['soft_visibility_mae']:.4f}",
        end="-" if "build_s" not in end else f"{end['build_s']:.3f}",
    )


def _fmt_cell(cell_m) -> str:
    """A cell size, or why there is not one. `None` means nothing passed the policy."""
    return "none passed the policy thresholds" if cell_m is None else f"{cell_m:.0f} m"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_terrain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", nargs="+", default=["owens_valley", "tehran_basin"])
    ap.add_argument("--cell-m", nargs="+", type=float, default=[1500.0, 500.0, 250.0, 100.0])
    ap.add_argument("--worlds", type=int, default=64, help="parallel envs in the vmapped rollout")
    ap.add_argument("--steps", type=int, default=128, help="steps per rollout")
    ap.add_argument("--blue", type=int, default=4)
    ap.add_argument("--threats", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5, help="steady-state timing samples per row")
    ap.add_argument("--rays", type=int, default=2000, help="line-of-sight rays per row")
    ap.add_argument("--endpoint-n", type=int, default=512, help="/terrain lat/lon grid side")
    ap.add_argument("--los-samples", nargs="+", type=int, default=[96, 192, 384],
                    help="ray-march sample counts to sweep (shipped default is 96)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="small sweep for a smoke check; not a publishable measurement")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stdout", action="store_true", help="also print the full JSON")
    a = ap.parse_args(argv)

    if a.quick:
        a.worlds, a.steps, a.repeats, a.rays, a.endpoint_n = 8, 32, 2, 200, 128
        a.los_samples = a.los_samples[:1]

    print(_header())
    print("-" * len(_header()))

    def on_row(row, skip):
        print(_row(row) if row else
              f"{skip['aoi']:<13} {skip['cell_m']:>6.0f} {str(skip['los_samples']):>4}  "
              f"SKIPPED  {skip['reason']}")

    report = run(
        a.aoi, a.cell_m,
        n_worlds=a.worlds, n_steps=a.steps, n_blue=a.blue, n_threat=a.threats,
        repeats=a.repeats, n_rays=a.rays, endpoint_n=a.endpoint_n, seed=a.seed,
        los_samples=a.los_samples, policy=ResolutionPolicy(), on_row=on_row,
    )

    rec = report["recommendation"]
    if rec:
        print()
        print(f"recommended physics grid:      {_fmt_cell(rec['physics_cell_m'])}"
              f" at los_samples={rec['physics_los_samples']}")
        print(f"recommended presentation cell: {_fmt_cell(rec['presentation_cell_m'])}")
        d = rec["published_defaults"]
        print(f"shipped defaults UNCHANGED by this run: {d['theatre_bridge_cell_m']:.0f} m physics, "
              f"{d['live_demo_cell_m']:.0f} m live demo, los_samples={d['los_samples']}")
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {a.out}  ({len(report['records'])} rows, {len(report['skipped'])} skipped)")
    if a.stdout:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
