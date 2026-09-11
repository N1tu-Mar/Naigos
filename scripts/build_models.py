"""Rebuild the viewer's generic glTF models from `naigos/demo/modelgen.py`.

    uv run python scripts/build_models.py           # write naigos/demo/assets/models/*.glb
    uv run python scripts/build_models.py --check   # exit 1 if a committed file drifted

Offline and deterministic: no download, no third-party asset. The committed
GLBs are exactly this script's output, which `tests/test_model_assets.py`
re-asserts on every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naigos.demo import modelgen  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed files instead of writing")
    a = ap.parse_args(argv)
    bad = 0
    for name in modelgen.BUILDERS:
        path = modelgen.MODELS_DIR / f"{name}.glb"
        data = modelgen.build(name)
        if a.check:
            same = path.exists() and path.read_bytes() == data
            print(f"{'ok   ' if same else 'DRIFT'} {path}  ({len(data)} bytes)")
            bad += not same
        else:
            modelgen.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            print(f"wrote {path}  ({len(data)} bytes)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
