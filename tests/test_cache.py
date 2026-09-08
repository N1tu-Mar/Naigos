"""Cache behaviour: idempotent, hashed, and offline after the first run."""

from __future__ import annotations

import json

import pytest

from naigos.research import cache
from naigos.research.aoi import AOIS, get_aoi


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "MANIFEST_PATH", tmp_path / "manifest.json")
    return tmp_path


def test_produce_writes_bytes_and_records_provenance(temp_cache):
    art = cache.produce(
        "t/one", "ourairports", "https://ourairports.com/data/", "sub/one.bin",
        lambda: b"hello",
    )
    assert (temp_cache / "sub/one.bin").read_bytes() == b"hello"
    assert art.sha256 == cache.sha256_bytes(b"hello")
    assert art.license == "Public domain (dedicated to the public domain by OurAirports)"
    assert art.fetched_at.endswith("Z")


def test_a_second_call_does_not_rebuild(temp_cache):
    """Offline-after-first-run: the builder must not be invoked again."""
    calls = []

    def build() -> bytes:
        calls.append(1)
        return b"payload"

    cache.produce("t/two", "ourairports", "https://ourairports.com/data/", "two.bin", build)
    cache.produce("t/two", "ourairports", "https://ourairports.com/data/", "two.bin", build)
    assert len(calls) == 1


def test_force_rebuilds(temp_cache):
    calls = []
    cache.produce("t/three", "ourairports", "https://ourairports.com/data/", "three.bin",
                  lambda: (calls.append(1), b"a")[1])
    cache.produce("t/three", "ourairports", "https://ourairports.com/data/", "three.bin",
                  lambda: (calls.append(1), b"b")[1], force=True)
    assert len(calls) == 2
    assert (temp_cache / "three.bin").read_bytes() == b"b"


def test_a_deleted_payload_is_re_fetched_even_though_the_manifest_remembers_it(temp_cache):
    cache.produce("t/four", "ourairports", "https://ourairports.com/data/", "four.bin",
                  lambda: b"first")
    (temp_cache / "four.bin").unlink()
    assert cache.get_artifact("t/four") is None
    art = cache.produce("t/four", "ourairports", "https://ourairports.com/data/", "four.bin",
                        lambda: b"second")
    assert art.abs_path.read_bytes() == b"second"


def test_the_manifest_is_stable_json(temp_cache):
    cache.produce("t/b", "ourairports", "https://ourairports.com/data/", "b.bin", lambda: b"b")
    cache.produce("t/a", "ourairports", "https://ourairports.com/data/", "a.bin", lambda: b"a")
    keys = list(json.loads(cache.MANIFEST_PATH.read_text()))
    assert keys == sorted(keys)


# --- AOI fingerprinting ----------------------------------------------------------------


def test_the_aoi_fingerprint_changes_with_the_bounds():
    """The bug this guards: editing an AOI silently reusing the previous box's DEM."""
    import dataclasses

    aoi = get_aoi("owens_valley")
    moved = dataclasses.replace(aoi, north=aoi.north + 0.1)
    assert aoi.fingerprint != moved.fingerprint


def test_the_aoi_fingerprint_is_stable_across_calls():
    assert get_aoi("owens_valley").fingerprint == get_aoi("owens_valley").fingerprint


def test_every_aoi_box_is_well_formed():
    for name, aoi in AOIS.items():
        assert aoi.west < aoi.east, name
        assert aoi.south < aoi.north, name
        assert aoi.rationale, f"{name} has no stated rationale"
        w, h = aoi.span_km()
        assert 20.0 < w < 400.0 and 20.0 < h < 400.0, name


def test_unknown_aoi_is_rejected():
    with pytest.raises(KeyError):
        get_aoi("atlantis")
