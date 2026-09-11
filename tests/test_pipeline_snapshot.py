"""Snapshots: validated before publication, immutable after, re-verified at use.

Offline: the research build is replaced by a fixture that writes the same tree
shape with small fake bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from _pipeline_fixtures import fake_runner, write_research_tree
from naigos.pipeline import egress, layout, snapshot
from naigos.pipeline import config as pcfg
from naigos.research import roots

REPO = Path(__file__).resolve().parents[1]
SID = "s-20260911-0123456789ab"
SID2 = "s-20260912-0123456789ab"


@pytest.fixture
def tree(tmp_path):
    write_research_tree(tmp_path / "t")
    return tmp_path / "t"


def _build(lay, sid, calls=None, **kw):
    cfg = pcfg.load_default()
    return snapshot.build(lay, sid=sid, attempt=1, cfg=cfg, window=sid.split("-")[1],
                          idempotency_key="k" * 64, config_digest="c", stage_digest="d",
                          code={"commit": "abc", "dirty": False},
                          research_runner=fake_runner(calls, **kw))


def test_a_well_formed_tree_validates(tree):
    assert snapshot.validate_tree(tree, "owens_valley") == []


def test_missing_bytes_are_rejected(tree):
    (tree / "cache" / "atmosphere").joinpath(
        next(p.name for p in (tree / "cache" / "atmosphere").iterdir())).unlink()
    assert any("bytes missing" in p for p in snapshot.validate_tree(tree, "owens_valley"))


def test_a_hash_mismatch_is_rejected(tree):
    f = next((tree / "cache" / "flights").iterdir())
    f.write_bytes(b'{"snapshots": [1]}')
    problems = snapshot.validate_tree(tree, "owens_valley")
    assert any("sha256 mismatch" in p for p in problems)


def test_uncited_bytes_in_the_cache_are_rejected(tree):
    (tree / "cache" / "stray.bin").write_bytes(b"from nowhere")
    assert any("uncited bytes" in p for p in snapshot.validate_tree(tree, "owens_valley"))


@pytest.mark.parametrize("sources", [[], [{"key": "some_random_blog"}]])
def test_an_uncited_or_non_allowlisted_component_is_rejected(tree, sources):
    for d in (tree / "components", tree / "components" / "aoi" / "owens_valley"):
        p = d / "data.atmosphere.json"
        doc = json.loads(p.read_text())
        doc["sources"] = sources
        p.write_text(json.dumps(doc))
    problems = snapshot.validate_tree(tree, "owens_valley")
    assert any("uncited" in p or "non-allowlisted" in p or "missing required" in p for p in problems)


def test_a_component_naming_bytes_the_manifest_disagrees_with_is_rejected(tree):
    p = tree / "components" / "data.flight_envelope.json"
    doc = json.loads(p.read_text())
    doc["cached_artifacts"][0]["sha256"] = "0" * 64
    p.write_text(json.dumps(doc))
    problems = snapshot.validate_tree(tree, "owens_valley")
    assert any("differs from the manifest" in p for p in problems)


def test_an_aoi_mismatch_is_rejected(tree):
    problems = snapshot.validate_tree(tree, "front_range")
    assert any("env.aoi names AOI" in p for p in problems)
    assert any("components/aoi/front_range/ is missing" in p for p in problems)


def test_a_tampered_aoi_fingerprint_is_rejected(tree):
    for d in (tree / "components", tree / "components" / "aoi" / "owens_valley"):
        p = d / "env.aoi.json"
        doc = json.loads(p.read_text())
        doc["parameters"]["fingerprint"] = "ffffffffff"
        p.write_text(json.dumps(doc))
    assert any("fingerprint" in p for p in snapshot.validate_tree(tree, "owens_valley"))


def test_a_stale_aoi_scoped_copy_is_rejected(tree):
    p = tree / "components" / "aoi" / "owens_valley" / "data.atmosphere.json"
    p.write_text(p.read_text().replace('"d"', '"older decision"'))
    assert any("differs from components" in p for p in snapshot.validate_tree(tree, "owens_valley"))


def test_the_guardrail_and_evasive_invariant_are_mandatory(tree):
    for d in (tree / "components", tree / "components" / "aoi" / "owens_valley"):
        p = d / "model.detection.json"
        doc = json.loads(p.read_text())
        doc["caveats"] = ["free-space model"]
        doc["invariants"] = ["Blue engages threats."]
        p.write_text(json.dumps(doc))
    problems = snapshot.validate_tree(tree, "owens_valley")
    assert any("GUARDRAIL" in p for p in problems)
    assert any("never acts on a threat" in p for p in problems)


def test_unsafe_manifest_paths_are_rejected(tree):
    mpath = tree / "cache" / "manifest.json"
    m = json.loads(mpath.read_text())
    next(iter(m.values()))["path"] = "../../etc/passwd"
    mpath.write_text(json.dumps(m))
    assert any("unsafe path" in p for p in snapshot.validate_tree(tree, "owens_valley"))


# --- build, publish, immutability ------------------------------------------------


def test_first_build_publishes_a_complete_snapshot(tmp_path):
    lay = layout.Layout(tmp_path)
    rec = _build(lay, SID)
    d = lay.snapshot_dir(SID)
    assert d.is_dir() and (d / "snapshot.json").exists()
    assert rec["parents"] == {"snapshot_id": None}
    assert rec["code"]["commit"] == "abc" and rec["config_digest"] == "c"
    assert rec["created_utc"].endswith("Z") and rec["aoi"]["name"] == "owens_valley"
    assert rec["content"]["sha256"] and "cache/manifest.json" in rec["content"]["files"]
    assert snapshot.verify_completed(lay, SID) == []
    assert snapshot.latest_completed(lay, aoi_name="owens_valley") == SID
    assert not (tmp_path / "pipeline" / "snapshots" / ".staging" / f"{SID}.1").exists()


def test_a_second_snapshot_seeds_from_the_first_and_refreshes_only_what_is_asked(tmp_path):
    lay = layout.Layout(tmp_path)
    _build(lay, SID)
    before = snapshot.content_digest(lay.snapshot_dir(SID))
    calls: list = []
    rec = _build(lay, SID2, calls=calls, payload=b'{"hourly": {"refreshed": true}}')
    assert rec["parents"] == {"snapshot_id": SID}
    assert calls == ["open_meteo/profile/owens_valley/" + rec["aoi"]["fingerprint"]]
    assert rec["refresh"]["dropped_artifacts"] == calls
    # the parent is untouched, byte for byte
    assert snapshot.content_digest(lay.snapshot_dir(SID)) == before
    assert snapshot.verify_completed(lay, SID) == []
    assert snapshot.latest_completed(lay, aoi_name="owens_valley") == SID2


def test_a_published_snapshot_is_never_rewritten(tmp_path):
    lay = layout.Layout(tmp_path)
    _build(lay, SID)
    before = snapshot.content_digest(lay.snapshot_dir(SID))
    with pytest.raises(layout.ImmutableRecordError):
        _build(lay, SID)
    assert snapshot.content_digest(lay.snapshot_dir(SID)) == before


def test_a_failed_build_publishes_nothing(tmp_path):
    lay = layout.Layout(tmp_path)
    with pytest.raises(RuntimeError, match="503"):
        _build(lay, SID, fail=True)
    assert not lay.snapshot_dir(SID).exists()
    assert snapshot.latest_completed(lay, aoi_name="owens_valley") is None


def test_an_invalid_build_publishes_nothing(tmp_path):
    lay = layout.Layout(tmp_path)

    def bad_runner(**kw):
        out = write_research_tree(kw["staging"], kw["aoi"])
        (kw["staging"] / "cache" / "stray.bin").write_bytes(b"x")
        return out

    with pytest.raises(snapshot.SnapshotError, match="uncited"):
        snapshot.build(lay, sid=SID, attempt=1, cfg=pcfg.load_default(), window="20260911",
                       idempotency_key="k" * 64, config_digest="c", stage_digest="d",
                       code={}, research_runner=bad_runner)
    assert not lay.snapshot_dir(SID).exists()


def test_tampering_after_publication_is_caught_at_use(tmp_path):
    lay = layout.Layout(tmp_path)
    _build(lay, SID)
    f = next((lay.snapshot_dir(SID) / "cache" / "terrain").glob("*.npz"))
    os.chmod(f, 0o644)
    f.write_bytes(b"different")
    problems = snapshot.verify_completed(lay, SID)
    assert any("content hash differs" in p for p in problems)
    assert snapshot.latest_completed(lay, aoi_name="owens_valley") is None


def test_a_directory_without_a_record_is_not_a_snapshot(tmp_path):
    lay = layout.Layout(tmp_path)
    write_research_tree(lay.snapshot_dir(SID))
    assert "not a completed snapshot" in snapshot.verify_completed(lay, SID)[0]
    assert snapshot.latest_completed(lay, aoi_name="owens_valley") is None


def test_snapshot_build_leaves_the_repository_roots_alone(tmp_path):
    before = roots.current_roots()
    _build(layout.Layout(tmp_path), SID)
    assert roots.current_roots() == before


# --- egress guard ------------------------------------------------------------------


def test_egress_guard_refuses_unlisted_hosts(monkeypatch):
    seen = []
    monkeypatch.setattr(egress, "_ORIGINAL_GETADDRINFO", lambda host, *a, **k: seen.append(host) or [])
    try:
        egress.install(frozenset({"api.open-meteo.com"}))
        assert egress.installed() == frozenset({"api.open-meteo.com"})
        socket.getaddrinfo("API.open-meteo.com.", 443)
        assert seen == ["API.open-meteo.com."]
        for host in ("example.com", "169.254.169.254", "", b"evil.example"):
            with pytest.raises(egress.EgressBlocked):
                socket.getaddrinfo(host, 443)
        assert seen == ["API.open-meteo.com."]
    finally:
        egress.uninstall()
    assert egress.installed() is None


def test_the_guard_covers_every_allowlisted_fetch_host_and_nothing_else():
    hosts = egress.allowlisted_hosts()
    assert "api.open-meteo.com" in hosts and "elevation.nationalmap.gov" in hosts
    assert "api.modal.com" not in hosts and "www.google.com" not in hosts


def test_a_guarded_child_process_cannot_resolve_an_unlisted_host():
    code = ("import socket; from naigos.pipeline import egress; egress.install(); "
            "import urllib.request\n"
            "try:\n urllib.request.urlopen('https://example.com', timeout=5)\n"
            "except Exception as e:\n print(type(e).__name__, 'refused' in str(e))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         timeout=60)
    assert "True" in out.stdout, out.stdout + out.stderr


def test_snapshot_build_entry_installs_the_guard_before_research_imports():
    src = (REPO / "naigos" / "pipeline" / "snapshot_build.py").read_text()
    assert src.index("egress.install()") < src.index("from naigos.research import")


# --- native (GDAL) network and the bootstrap seed ---------------------------------------


def test_the_guarded_child_turns_gdal_http_off(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["env"] = kw["env"]
        (tmp_path / ".build_result.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(snapshot.subprocess, "run", fake_run)
    snapshot.run_research_subprocess(staging=tmp_path, aoi="owens_valley", skip_flights=True,
                                     flight_snapshots=1, timeout_s=10)
    assert seen["env"]["GDAL_HTTP_PROXY"] == "127.0.0.1:9"
    assert "NAIGOS_CACHE_DIR" not in seen["env"]


def _two_aoi_local_cache(root):
    write_research_tree(root, "front_range")
    write_research_tree(root, "owens_valley")
    return root / "cache", root / "components"


def test_a_seed_takes_one_aoi_from_a_local_cache_and_validates(tmp_path):
    cache_dir, comp_dir = _two_aoi_local_cache(tmp_path / "local")
    dest = tmp_path / "seed"
    assert snapshot.prepare_seed(cache_dir, comp_dir, "owens_valley", dest) == []
    keys = json.loads((dest / "cache" / "manifest.json").read_text())
    assert keys and not any("front_range" in k for k in keys)
    assert not (dest / "components" / "aoi" / "front_range").exists()


def test_an_invalid_local_cache_is_refused_before_upload(tmp_path):
    cache_dir, comp_dir = _two_aoi_local_cache(tmp_path / "local")
    next((cache_dir / "flights").iterdir()).write_bytes(b"tampered")
    problems = snapshot.prepare_seed(cache_dir, comp_dir, "owens_valley", tmp_path / "seed")
    assert any("sha256 mismatch" in p for p in problems)
    assert snapshot.prepare_seed(cache_dir, comp_dir, "tehran_basin", tmp_path / "x")[0].endswith("first")


def test_a_published_seed_bootstraps_the_chain(tmp_path):
    from _pipeline_fixtures import Harness
    from naigos.pipeline import admin

    h = Harness(tmp_path / "vol")
    cache_dir, comp_dir = _two_aoi_local_cache(tmp_path / "local")
    tree = tmp_path / "seed"
    assert snapshot.prepare_seed(cache_dir, comp_dir, "owens_valley", tree) == []
    sid = snapshot.seed_id(snapshot.content_digest(tree)["sha256"], h.svc.code, now=h.clock())
    shutil.copytree(tree, h.lay.staging_dir(sid, 1))  # what `pipeline.py seed` uploads
    out = admin.handle(h.svc, "publish-seed", {"snapshot_id": sid, "aoi": "owens_valley"})
    assert out["origin"] == "operator-seed"
    assert snapshot.verify_completed(h.lay, sid) == []
    # The seed is a normal parent: the next scheduled snapshot copies its DEM
    # instead of fetching one, and training runs on it.
    h.clock.advance(days=1)
    h.tick("snapshot")
    h.drain()
    new = [s for s in h.lay.snapshot_ids() if s != sid]
    rec = json.loads(h.lay.snapshot_record(new[0]).read_text())
    assert rec["parents"]["snapshot_id"] == sid
    # ...and a second seed is refused once the chain exists.
    shutil.copytree(tree, h.lay.staging_dir(SID, 1))
    with pytest.raises(snapshot.SnapshotError, match="bootstrapping"):
        admin.handle(h.svc, "publish-seed", {"snapshot_id": SID, "aoi": "owens_valley"})
    with pytest.raises(admin.AdminError, match="trains on"):
        admin.handle(h.svc, "publish-seed", {"snapshot_id": SID, "aoi": "front_range"})
