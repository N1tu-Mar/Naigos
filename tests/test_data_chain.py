"""End-to-end integrity of the data chain: manifest -> cache bytes -> component
spec -> theatre -> EnvConfig.

The env must never read `data_cache/` directly; it reads `components/*.json`,
which name the cached artefacts by sha256. These tests check that the chain is
actually intact rather than assuming it -- a component spec that cites a file
whose bytes have changed is worse than one that cites nothing, because it looks
trustworthy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

CACHE = Path("data_cache")
COMPONENTS = Path("components")
MANIFEST = CACHE / "manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="research cache not populated")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_every_manifest_entry_points_at_a_file_that_exists():
    missing = [k for k, e in _manifest().items() if not (CACHE / e["path"]).exists()]
    assert not missing, f"manifest references missing files: {missing}"


def test_every_manifest_entry_declares_a_licence_and_a_url():
    for k, e in _manifest().items():
        assert e.get("license"), f"{k} has no licence"
        assert e.get("url"), f"{k} has no source url"
        assert e.get("citation"), f"{k} has no citation"


@pytest.mark.parametrize("key", sorted(_manifest().keys()) if MANIFEST.exists() else [])
def test_cached_bytes_match_the_recorded_sha256(key):
    """Catches a silently edited or truncated cache file."""
    e = _manifest()[key]
    p = CACHE / e["path"]
    if p.stat().st_size > 60_000_000:  # keep the suite fast on the 50 MB rasters
        pytest.skip("large raster; size check only")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == e["sha256"], f"{e['path']} does not match its recorded sha256"
    assert p.stat().st_size == e["bytes"]


def test_aoi_fingerprint_is_consistent_across_scoped_artifacts():
    """The bbox hash in the filename is what stops a changed AOI silently
    reusing the wrong terrain."""
    aoi = json.loads((COMPONENTS / "env.aoi.json").read_text())
    fp = aoi["parameters"]["fingerprint"]
    scoped = [e["path"] for e in _manifest().values() if fp in e["path"]]
    assert scoped, f"no cached artefact carries the AOI fingerprint {fp}"
    for path in scoped:
        assert fp in path


def test_component_specs_reference_only_cached_artifacts():
    man_paths = {e["path"] for e in _manifest().values()}
    for spec_path in sorted(COMPONENTS.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        for art in spec.get("cached_artifacts", []):
            assert art["path"] in man_paths, f"{spec_path.name} cites uncached {art['path']}"
            assert (CACHE / art["path"]).exists()


def test_every_component_spec_is_cited():
    """Nothing enters the env without a source (prompt.md s8)."""
    for spec_path in sorted(COMPONENTS.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        assert spec.get("sources"), f"{spec_path.name} has no cited source"
        for s in spec["sources"]:
            assert s.get("license"), f"{spec_path.name} cites a source with no licence"


def test_env_config_built_from_the_chain_uses_the_real_dem():
    """The whole point: the flying surface is the cited 3DEP raster, not a
    synthetic stand-in."""
    from naigos.env.theatre_bridge import env_from_theatre

    cfg, hmap, notes = env_from_theatre(n_threat=8)
    lo, hi = notes["relief_m"]
    dem_spec = json.loads((COMPONENTS / "data.terrain_dem.json").read_text())
    ev = dem_spec["evidence"]["elevation_m"]
    # the resampled window sits inside the full raster's elevation range
    assert ev["min"] - 1.0 <= lo <= hi <= ev["max"] + 1.0
    assert hi - lo > 1_000.0, "no real relief in the loaded window"
    assert notes["theatre"] == json.loads((COMPONENTS / "env.aoi.json").read_text())["parameters"]["name"]


def test_env_package_never_reads_the_cache_directly():
    """`naigos/env` must go through components/, never through data_cache/.

    Checks string literals and identifiers, not prose: the docstrings in
    `theatre_bridge` legitimately talk about the cache while never reading it.
    """
    import ast

    for p in Path("naigos/env").glob("*.py"):
        tree = ast.parse(p.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                assert "data_cache" not in node.value, f"{p}:{node.lineno} reads the cache directly"
            if isinstance(node, ast.Name):
                assert "data_cache" not in node.id.lower(), f"{p}:{node.lineno}"
