"""File-defined theatres: one JSON per AOI, isolated research runs, protected zones.

A theatre is added by adding ``naigos/research/aois/<name>.json`` -- never by
editing a shared table -- and building it writes only that theatre's own
component snapshot and provenance doc. Both properties exist so that two
theatres developed on two branches cannot collide, and so that building a new
theatre can never rewrite the files the default theatre (and every test and
checkpoint that reads it) depends on.

Offline: no cache, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naigos.research import aoi as aoi_mod
from naigos.research import run, spec
from naigos.research.sources import airports

REPO = Path(__file__).resolve().parents[1]


def _doc(name="test_city", **over):
    d = {
        "name": name, "west": 10.0, "south": 20.0, "east": 10.4, "north": 20.3,
        "country": "XX", "dem_source": "copernicus_dem",
        "rationale": "a test box", "bounds_policy": "chosen for the test",
        "scenario": "notional contested-airspace simulation",
        "exclude_military_airfields": True,
        "protected_zones": [{
            "name": "test precinct", "west": 10.1, "south": 20.1, "east": 10.12,
            "north": 20.12, "buffer_m": 500, "policies": ["extraction", "camera", "ambience"],
            "reason": "test",
        }],
    }
    d.update(over)
    return d


def _write(tmp_path, doc):
    p = tmp_path / f"{doc['name']}.json"
    p.write_text(json.dumps(doc))
    return p


# --- the built-in theatres are untouched ------------------------------------------------


def test_the_built_in_aois_keep_their_fingerprints():
    """Cache keys embed these. A changed fingerprint would orphan every cached DEM."""
    assert aoi_mod.get_aoi("tehran_basin").fingerprint == "38a6bdd22d"
    assert aoi_mod.get_aoi("owens_valley").fingerprint == "47b7c1c4e7"
    for name in ("owens_valley", "front_range", "tehran_basin"):
        a = aoi_mod.get_aoi(name)
        assert not a.scoped and a.protected_zones == () and not a.exclude_military_airfields


def test_built_in_aois_emit_no_new_component_parameters():
    """Their env.aoi output stays byte-for-byte what it was."""
    assert run._scoped_aoi_parameters(aoi_mod.get_aoi("tehran_basin")) == {}


def test_the_military_filter_is_opt_in():
    """The built-ins' airfield lists (and snapshots) must not change under them."""
    tehran = aoi_mod.get_aoi("tehran_basin")
    assert airports.excluded_reason(tehran, "Some Air Base", 51.4, 35.7) is None


# --- file-defined theatres ----------------------------------------------------------------


def test_a_file_defined_aoi_loads_strictly(tmp_path):
    a = aoi_mod.aoi_from_file(_write(tmp_path, _doc()))
    assert a.scoped and a.name == "test_city"
    assert a.protected_zones[0].policies == ("extraction", "camera", "ambience")
    params = run._scoped_aoi_parameters(a)
    assert params["protected_zones"][0]["name"] == "test precinct"
    assert params["scenario"] == "notional contested-airspace simulation"
    assert params["bounds_policy"]


@pytest.mark.parametrize("bad", [
    {"rationale": ""},                      # missing required text
    {"colour": "blue"},                     # unknown key: a typo is an error
    {"name": "not_the_file_name"},
])
def test_a_malformed_definition_is_refused(tmp_path, bad):
    doc = _doc(**bad)
    path = tmp_path / "test_city.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        aoi_mod.aoi_from_file(path)


def test_a_zone_with_an_unknown_policy_is_refused(tmp_path):
    doc = _doc()
    doc["protected_zones"][0]["policies"] = ["extraction", "targeting"]
    with pytest.raises(ValueError, match="unknown policies"):
        aoi_mod.aoi_from_file(_write(tmp_path, doc))


def test_a_zone_buffer_grows_the_box_by_its_metres():
    z = aoi_mod.ProtectedZone("z", 10.0, 20.0, 10.1, 20.1, 1000.0, ("camera",), "t")
    w, s, e, n = z.buffered()
    assert (20.0 - s) * 110_574.0 == pytest.approx(1000.0, rel=1e-6)
    assert z.contains(10.0 - 0.005, 20.05) and not z.contains(10.0 - 0.005, 20.05, buffered=False)
    assert not z.contains(9.9, 20.05)


def test_a_file_aoi_cannot_shadow_a_built_in(tmp_path, monkeypatch):
    monkeypatch.setattr(aoi_mod, "AOIS", dict(aoi_mod.AOIS))
    clash = aoi_mod.aoi_from_file(_write(tmp_path, _doc(name="tehran_basin")))
    with pytest.raises(ValueError, match="redefine built-in"):
        aoi_mod._register({"tehran_basin": clash})


def test_every_shipped_definition_file_is_registered():
    for path in sorted(aoi_mod.AOI_DEFS_DIR.glob("*.json")):
        a = aoi_mod.get_aoi(path.stem)
        assert a.scoped and Path(a.definition_file).name == path.name
        assert a.scenario and a.bounds_policy
        w, h = a.span_km()
        assert 20.0 < w < 400.0 and 20.0 < h < 400.0, a.name


def test_military_names_are_dropped_for_an_opted_in_theatre(tmp_path):
    a = aoi_mod.aoi_from_file(_write(tmp_path, _doc()))
    for name in ("Example Air Base", "Example Airbase", "Example Air Force Station",
                 "Example Naval Air Station", "Example AFB", "Army Airfield Example"):
        assert airports.excluded_reason(a, name, 10.3, 20.2) == "military_name", name
    assert airports.excluded_reason(a, "Example International Airport", 10.3, 20.2) is None


def test_a_protected_zone_only_drops_airfields_when_it_says_so(tmp_path):
    a = aoi_mod.aoi_from_file(_write(tmp_path, _doc()))
    # the zone above has no "airfield" policy, so a field inside it is kept
    assert airports.excluded_reason(a, "Field", 10.11, 20.11) is None
    doc = _doc()
    doc["protected_zones"][0]["policies"].append("airfield")
    b = aoi_mod.aoi_from_file(_write(tmp_path, doc))
    assert airports.excluded_reason(b, "Field", 10.11, 20.11) == "protected_zone:test precinct"


# --- isolated research runs ---------------------------------------------------------------


def test_a_scoped_build_writes_only_its_own_snapshot_and_doc(tmp_path, monkeypatch):
    """The legacy path rewrites components/*.json and docs/DATA.md. A file-defined
    theatre must not: those files belong to the default theatre."""
    a = aoi_mod.aoi_from_file(_write(tmp_path, _doc()))
    monkeypatch.setitem(aoi_mod.AOIS, a.name, a)
    comp_root = tmp_path / "components"
    comp_root.mkdir()
    (comp_root / "env.aoi.json").write_text('{"id": "env.aoi", "untouched": true}')
    seen = {}

    def fake_build(name, force, skip_flights, n_snapshots, data_doc=None):
        seen["components_dir"] = spec.components_dir()
        seen["data_doc"] = data_doc
        spec.write_component("env.aoi", role="r", inputs=["i"], outputs=["o"], decision="d",
                             rationale="why", source_keys=["ourairports"])
        Path(data_doc).parent.mkdir(parents=True, exist_ok=True)
        Path(data_doc).write_text("doc")
        return {"components": [], "artifacts": []}

    monkeypatch.setattr(run, "build", fake_build)
    doc = tmp_path / "docs" / "theatres" / a.name / "DATA.md"
    out = run.build_scoped(a.name, components_root=comp_root, data_doc=doc)

    assert seen["components_dir"] != comp_root, "built straight into the shared root"
    assert json.loads((comp_root / "env.aoi.json").read_text())["untouched"] is True
    snap = comp_root / "aoi" / a.name / "env.aoi.json"
    assert snap.exists() and out["aoi_snapshot"] == str(comp_root / "aoi" / a.name)
    assert doc.read_text() == "doc"
    # and the component root in effect afterwards is the one before
    assert spec.components_dir() != Path(seen["components_dir"])


def test_the_theatre_doc_lives_beside_its_theatre():
    assert run.theatre_doc_path("x_city") == REPO / "docs" / "theatres" / "x_city" / "DATA.md"


def test_a_built_in_is_refused_by_the_scoped_path():
    with pytest.raises(ValueError):
        run.build_scoped("tehran_basin")
