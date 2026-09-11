#!/usr/bin/env python
"""Offline CBF margin calibration sweep.

    uv run python scripts/sweep_cbf.py --quick            # ~8 cells, laptop-sized
    uv run python scripts/sweep_cbf.py \
        --margin 0 750 1500 3000 --threats 6 12 --red-level 0.2 0.6 \
        --n-seeds 32 --out docs/artifacts/cbf_sweep.json --markdown docs/artifacts/cbf_sweep.md

Measures the tradeoff between CBF margin, threat density and red difficulty on
one held-out seed set that is identical in every grid cell and in both arms
(filtered and unfiltered). Writes a versioned JSON result and prints a table
sorted by margin.

It changes no default anywhere in `naigos/`. `naigos/rl/cbf.py` is imported and
used as shipped; the margins on the command line are the sweep's axis, not a new
setting. Nothing here recommends a value -- read the table.

The environment configuration defaults to the one rebuilt from the checkpoint,
because a theatre-trained actor has threat-slot and ego widths that a default
synthetic `EnvConfig` does not build. Size overrides (`--blue`,
`--threat-capacity`, `--steps`) are applied on top of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from naigos.rl.cbf import CBFConfig  # noqa: E402
from naigos.rl.cbf_sweep import (  # noqa: E402
    SweepConfigError,
    SweepGrid,
    format_table,
    markdown_summary,
    run_sweep,
    validate_seeds,
    write_result,
)

DEFAULT_CKPT = Path("checkpoints/theatre_1000.pkl")
DEFAULT_OUT = Path("docs/artifacts/cbf_sweep.json")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sweep_cbf",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--margin", nargs="+", type=float, default=[0.0, 750.0, 1500.0, 3000.0],
                    help="metres each known envelope is inflated by before filtering")
    ap.add_argument("--threats", nargs="+", type=int, default=[6, 12],
                    help="active threat counts; each must be <= --threat-capacity")
    ap.add_argument("--red-level", nargs="+", type=float, default=[0.2, 0.6],
                    help="RedCurriculum levels in [0, 1]")

    ap.add_argument("--n-seeds", type=int, default=32, help="held-out scenarios per cell")
    ap.add_argument("--base-seed", type=int, default=20_000,
                    help="first held-out seed; the plan is this plus 0..n-1")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="explicit seed list, overriding --n-seeds/--base-seed")

    ap.add_argument("--blue", type=int, default=4)
    ap.add_argument("--threat-capacity", type=int, default=16,
                    help="EnvConfig.n_threat, the padded slot count. Fixed across the sweep so "
                         "every cell has the same threat draw and density is a prefix of it.")
    ap.add_argument("--steps", type=int, default=None,
                    help="rollout length; defaults to the env config's max_steps")

    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the unfiltered arm. Halves the cost and removes the control the "
                         "margin question needs; use only for a smoke run.")
    ap.add_argument("--arm", choices=("cbf", "no_cbf"), default="cbf", help="arm to tabulate")
    ap.add_argument("--notes", default=None, help="free text recorded in the result")

    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--markdown", type=Path, default=None, help="also write a Markdown summary")
    ap.add_argument("--quick", action="store_true",
                    help="2x2x2 grid, 8 seeds, 64 steps -- a few minutes on a CPU")
    return ap


def _apply_quick(args) -> None:
    args.margin = [0.0, 1_500.0]
    args.threats = [6, 12]
    args.red_level = [0.2, 0.6]
    args.n_seeds = 8
    args.steps = 64
    args.blue = 2


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        _apply_quick(args)

    try:
        grid = SweepGrid(
            margins=tuple(args.margin),
            threat_counts=tuple(args.threats),
            red_levels=tuple(args.red_level),
        )
        seeds = validate_seeds(args.seeds) if args.seeds else None

        def on_cell(i, total, row):
            m = row["arms"]["cbf"]
            print(
                f"[{i:>3}/{total}] {row['cell']['key']}  surv={m['survival_rate']:.3f} "
                f"obj={m['objective_rate']:.3f} infeas={m['cbf_infeasible_rate']:.3f}",
                flush=True,
            )

        payload = run_sweep(
            grid=grid,
            checkpoint=args.checkpoint,
            base_cbf_cfg=CBFConfig(),
            seeds=seeds,
            base_seed=args.base_seed,
            n_seeds=args.n_seeds,
            n_steps=args.steps,
            include_no_cbf=not args.no_baseline,
            env_overrides={
                "n_blue": args.blue,
                "n_threat": args.threat_capacity,
                **({"max_steps": args.steps} if args.steps else {}),
            },
            notes=args.notes,
            on_cell=on_cell,
        )
    except SweepConfigError as e:
        print(f"sweep_cbf: {e}", file=sys.stderr)
        return 2

    write_result(args.out, payload)
    md = markdown_summary(payload, arm=args.arm)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)

    print()
    print(format_table(payload, arm=args.arm))
    print()
    print(f"wrote {args.out}" + (f" and {args.markdown}" if args.markdown else ""))
    print(
        "This is a measurement, not a recommendation. The margin that looks best here was "
        "measured on these seeds, this checkpoint and this theatre; nothing in naigos/ changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
