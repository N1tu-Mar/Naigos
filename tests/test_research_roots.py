"""Cache and component roots: repository defaults, environment and explicit overrides.

The cloud pipeline builds each snapshot in its own directory, so the research
layer must be pointable at a root other than the repository -- without the
local default moving for anyone who is not asking for that.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from naigos.research import cache, roots, spec

REPO = Path(__file__).resolve().parents[1]


def _roots_in_subprocess(env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("NAIGOS_CACHE_DIR", "NAIGOS_COMPONENTS_DIR")}
    env.update(env_extra)
    code = (
        "import json; from naigos.research import cache, spec; "
        "print(json.dumps({'cache': str(cache.CACHE_DIR), 'manifest': str(cache.MANIFEST_PATH), "
        "'components': str(spec.COMPONENTS_DIR)}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_local_defaults_are_the_repository_directories():
    got = _roots_in_subprocess({})
    assert got["cache"] == str(REPO / "data_cache")
    assert got["manifest"] == str(REPO / "data_cache" / "manifest.json")
    assert got["components"] == str(REPO / "components")
    assert roots.default_roots().cache_dir == REPO / "data_cache"
    assert roots.default_roots().components_dir == REPO / "components"


def test_environment_overrides_both_roots(tmp_path):
    got = _roots_in_subprocess({
        "NAIGOS_CACHE_DIR": str(tmp_path / "c"),
        "NAIGOS_COMPONENTS_DIR": str(tmp_path / "k"),
    })
    assert got == {
        "cache": str(tmp_path / "c"),
        "manifest": str(tmp_path / "c" / "manifest.json"),
        "components": str(tmp_path / "k"),
    }


def test_an_empty_environment_value_keeps_the_default():
    got = _roots_in_subprocess({"NAIGOS_CACHE_DIR": "", "NAIGOS_COMPONENTS_DIR": ""})
    assert got["cache"] == str(REPO / "data_cache")
    assert got["components"] == str(REPO / "components")


def test_explicit_roots_redirect_writes_and_reads_then_restore(tmp_path):
    before = roots.current_roots()
    with roots.research_roots(cache_dir=tmp_path / "cache",
                              components_dir=tmp_path / "components") as r:
        assert r.cache_dir == tmp_path / "cache"
        art = cache.produce("t/root", "ourairports", "https://ourairports.com/data/",
                            "x/one.bin", lambda: b"snapshot bytes")
        assert (tmp_path / "cache" / "x" / "one.bin").read_bytes() == b"snapshot bytes"
        assert art.abs_path == tmp_path / "cache" / "x" / "one.bin"
        assert "t/root" in json.loads((tmp_path / "cache" / "manifest.json").read_text())

        path = spec.write_component(
            "test.rooted", role="r", inputs=["i"], outputs=["o"], decision="d",
            rationale="why", source_keys=["ourairports"], artifacts=[art],
        )
        assert path == tmp_path / "components" / "test.rooted.json"
        assert spec.load_component("test.rooted")["cached_artifacts"][0]["sha256"] == art.sha256
    assert roots.current_roots() == before
    # nothing leaked into the repository's own directories
    assert not (before.components_dir / "test.rooted.json").exists()
    assert "t/root" not in cache.load_manifest()


def test_roots_are_restored_after_an_exception(tmp_path):
    before = roots.current_roots()
    with pytest.raises(RuntimeError):
        with roots.research_roots(cache_dir=tmp_path):
            raise RuntimeError("snapshot build failed")
    assert roots.current_roots() == before


def test_one_root_can_be_overridden_alone(tmp_path):
    before = roots.current_roots()
    with roots.research_roots(components_dir=tmp_path) as r:
        assert r.components_dir == tmp_path
        assert r.cache_dir == before.cache_dir


def test_the_theatre_loader_follows_the_cache_root(tmp_path):
    """`naigos.data.theatre` used to bind CACHE_DIR at import; it must not."""
    from naigos.data import theatre

    comp_dir = tmp_path / "components"
    comp_dir.mkdir()
    (comp_dir / "data.terrain_dem.json").write_text(json.dumps({
        "cached_artifacts": [{"path": "terrain/dem.npz"}],
    }))
    with roots.research_roots(cache_dir=tmp_path / "cache", components_dir=comp_dir):
        with pytest.raises(FileNotFoundError) as e:
            theatre._dem_path()
        assert str(tmp_path / "cache" / "terrain" / "dem.npz") in str(e.value)
        (tmp_path / "cache" / "terrain").mkdir(parents=True)
        (tmp_path / "cache" / "terrain" / "dem.npz").write_bytes(b"x")
        assert theatre._dem_path() == tmp_path / "cache" / "terrain" / "dem.npz"
