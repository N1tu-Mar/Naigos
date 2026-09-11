"""Write the offline terrain fixture a theatre's camera and landmark tests read.

    uv run python scripts/make_viewer_grid_fixture.py --aoi dubai_urban

The viewer never sees the 30 m DEM: it draws `naigos.demo.live.build_terrain_grid`,
the simulation's own heightmap resampled to a lat/lon grid. This writes that
same grid -- built from the theatre's component snapshot and the cached DEM,
at the live viewer's default cell size -- as a small compressed fixture under
tests/fixtures/, so the camera contract and the georeference checks run on a
fresh clone with no cache and no network.

Derived from Copernicus GLO-30 (ESA/Airbus, attribution required): the
fixture carries that credit in its metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CREDIT = ("Derived from the Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus "
          "Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union "
          "and ESA; resampled by the Naigos simulation.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--n", type=int, default=256, help="grid posts per side")
    ap.add_argument("--cell-m", type=float, default=500.0, help="env cell size (live default)")
    a = ap.parse_args(argv)

    from naigos.data.geodetic import GeoRef
    from naigos.demo.live import build_terrain_grid
    from naigos.env.theatre_bridge import env_from_theatre

    cfg, hmap, notes = env_from_theatre(aoi=a.aoi, cell_m=a.cell_m)
    raw, meta = build_terrain_grid(np.asarray(hmap), cfg.terrain, GeoRef(**notes["georef"]),
                                   notes["geo_bounds"], n=a.n)
    meta = {**meta, "aoi": a.aoi, "georef": notes["georef"], "credit": CREDIT,
            "built_from": f"components/aoi/{a.aoi} @ cell {a.cell_m:g} m"}
    out = REPO / "tests" / "fixtures" / f"viewer_grid_{a.aoi}.npz"
    np.savez_compressed(out, heights=np.frombuffer(raw, dtype="<i2").reshape(a.n, a.n),
                        meta=json.dumps(meta))
    print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size / 1e3:.0f} kB, "
          f"{meta['min_m']:.0f}-{meta['max_m']:.0f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
