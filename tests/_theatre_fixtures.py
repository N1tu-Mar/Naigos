"""Build a theatre's env with no data cache: the component snapshot plus a coarse DEM fixture.

The component snapshot (`components/aoi/<aoi>/`) is committed; the 30 m DEM it
cites lives in the git-ignored cache. `offline_theatre` points the research
cache at a temporary directory holding `tests/fixtures/dem_<aoi>.npz` at the
exact relative path the `data.terrain_dem` component names, so
`env_from_theatre(aoi=...)` runs the normal load path -- component -> npz ->
ENU grid -> EnvConfig -- on a fresh clone.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def offline_theatre(aoi: str):
    from naigos.research.roots import research_roots

    comp = json.loads((REPO / "components" / "aoi" / aoi / "data.terrain_dem.json").read_text())
    rel = next(a["path"] for a in comp["cached_artifacts"] if a["path"].endswith(".npz"))
    with tempfile.TemporaryDirectory(prefix=f"naigos-{aoi}-") as tmp:
        dest = Path(tmp) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "tests" / "fixtures" / f"dem_{aoi}.npz", dest)
        with research_roots(cache_dir=tmp):
            yield Path(tmp)


def viewer_sampler(aoi: str):
    import numpy as np

    from naigos.demo.cities import TerrainSampler

    f = np.load(REPO / "tests" / "fixtures" / f"viewer_grid_{aoi}.npz")
    return TerrainSampler(f["heights"], json.loads(str(f["meta"])))
