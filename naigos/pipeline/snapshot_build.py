"""Child process for one snapshot build: guarded egress, snapshot-scoped roots.

    python -m naigos.pipeline.snapshot_build --staging DIR --aoi NAME --result FILE

The egress guard is installed *before* the research modules are imported, so
nothing they import at module load can resolve an unlisted host either. The
cache and component roots are pointed at ``DIR/cache`` and ``DIR/components``,
``DATA.md`` is rendered to ``DIR/DATA.md``, and nothing under the repository is
written. Called by ``naigos.pipeline.snapshot.run_research_subprocess``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging", required=True)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--skip-flights", action="store_true")
    ap.add_argument("--flight-snapshots", type=int, default=26)
    ap.add_argument("--no-egress-guard", action="store_true",
                    help="tests only; the scheduled worker always guards")
    a = ap.parse_args(argv)

    from naigos.pipeline import egress

    allowed = None if a.no_egress_guard else egress.install()

    from naigos.research import roots, spec
    from naigos.research import run as research_run

    staging = Path(a.staging)
    with roots.research_roots(cache_dir=staging / "cache", components_dir=staging / "components"):
        result = research_run.build(a.aoi, force=False, skip_flights=a.skip_flights,
                                    n_snapshots=a.flight_snapshots, data_doc=staging / "DATA.md")
        spec.snapshot_aoi(a.aoi)
    result = {
        "components": [Path(p).name for p in result.get("components", [])],
        "artifacts": list(result.get("artifacts", [])),
        "egress_guard": sorted(allowed) if allowed is not None else None,
    }
    Path(a.result).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
