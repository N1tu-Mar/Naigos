"""The viewer's 3D models: present, licensed, generated, and total over threat kinds.

Offline and GPU-free. Nothing here claims a model *renders* -- that needs WebGL.
What it pins is everything around the render that can silently rot: a file that
went missing, a licence record that stopped matching, a committed GLB that is
no longer what the generator makes, a threat kind with no model, and a marker
fallback that has become the normal path.
"""

from __future__ import annotations

import json
import re
import struct

import pytest

from naigos.demo import modelgen, models
from naigos.env.config import ThreatKindConfig, default_threat_kinds

REQUIRED_CLASSES = {"fixed_wing", "ground_vehicle", "sensor_site", "interceptor_drone"}


# --- the files ------------------------------------------------------------------------


def test_every_required_model_class_is_registered():
    assert set(models.REGISTRY) == REQUIRED_CLASSES
    assert models.AIRCRAFT_MODEL == "fixed_wing"


@pytest.mark.parametrize("key", sorted(REQUIRED_CLASSES))
def test_every_model_exists_locally_and_is_a_valid_glb(key):
    spec = models.REGISTRY[key]
    assert spec.path.exists(), f"{spec.file} is not in the repository"
    assert spec.path.parent == models.MODELS_DIR
    data = spec.path.read_bytes()
    assert models.validate_glb(data, spec.required_nodes) == []
    # small enough to inline into the static artifact without thinking about it
    assert len(data) < 64 * 1024


@pytest.mark.parametrize("key", sorted(REQUIRED_CLASSES))
def test_every_model_has_a_licence_and_attribution_record(key):
    spec = models.REGISTRY[key]
    readme = models.MODELS_README.read_text()
    assert f"`{spec.file}`" in readme, f"{spec.file} is not documented"
    for heading in ("Source", "Licence", "Attribution", "Modifications"):
        assert f"**{heading}:**" in readme
    assert spec.licence == models.MODEL_LICENCE and models.MODEL_LICENCE in readme
    # the licence also travels inside the file, so a copied GLB still carries it
    data = spec.path.read_bytes()
    jlen = struct.unpack("<I", data[12:16])[0]
    asset = json.loads(data[20:20 + jlen])["asset"]
    assert "CC0-1.0" in asset["copyright"]
    assert asset["generator"] == modelgen.GENERATOR


@pytest.mark.parametrize("key", sorted(REQUIRED_CLASSES))
def test_the_committed_file_is_exactly_the_generators_output(key):
    """The provenance record. A GLB pasted in from elsewhere, or edited by hand,
    fails here -- the repository can only carry what modelgen.py builds."""
    spec = models.REGISTRY[key]
    assert spec.path.read_bytes() == modelgen.build(key), (
        f"{spec.file} drifted from naigos/demo/modelgen.py; "
        "run: uv run python scripts/build_models.py")


def test_the_generator_is_deterministic():
    for key in REQUIRED_CLASSES:
        assert modelgen.build(key) == modelgen.build(key)


def test_models_are_generic_and_unmarked():
    """No texture (so no insignia or photo can ride along), and no node or
    material named after anything but its generic role."""
    for spec in models.REGISTRY.values():
        data = spec.path.read_bytes()
        jlen = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + jlen])
        assert "images" not in gltf and "textures" not in gltf
        names = [n["name"] for n in gltf["nodes"]] + [m["name"] for m in gltf["materials"]]
        assert all(re.fullmatch(r"[a-z_]+", n) for n in names), names


def test_the_models_follow_the_documented_axes():
    """Nose/front at +Z and up at +Y: the frame CesiumJS's default glTF 2.0
    correction expects, which is why every registry offset is zero."""
    for key in ("fixed_wing", "interceptor_drone"):
        tris = [t for n in modelgen.BUILDERS[key]()["nodes"] for p in n["parts"] for t in p.tris]
        import numpy as np

        pts = np.concatenate(tris)
        nose = pts[pts[:, 2].argmax()]
        assert abs(nose[0]) < 1e-6, "the nose is on the centreline, at +Z"
        assert pts[:, 2].max() > abs(pts[:, 0]).max(), "longer than it is wide"
    for spec in models.REGISTRY.values():
        assert (spec.forward_axis, spec.up_axis) == ("+Z", "+Y")
        assert (spec.heading_offset_deg, spec.pitch_offset_deg, spec.roll_offset_deg) == (0, 0, 0)


# --- the registry covers every threat kind ----------------------------------------------


@pytest.mark.parametrize("kind", default_threat_kinds(), ids=lambda k: k.label)
def test_every_default_threat_kind_has_a_model(kind):
    key = models.model_for_kind(kind)
    assert key in models.REGISTRY and key != models.AIRCRAFT_MODEL


def test_the_default_kinds_land_on_the_expected_classes():
    by_label = {k.label: models.model_for_kind(k) for k in default_threat_kinds()}
    assert by_label == {"static_sam": "sensor_site", "mobile_ground": "ground_vehicle",
                        "interceptor": "interceptor_drone"}


def test_every_theatre_threat_class_has_a_model():
    """Theatre kinds are DATA -- five classes from components/model.detection.json
    today, and whatever the research agent writes tomorrow. The mapping reads
    fields, not labels, so none of them can reach the page without a model."""
    from naigos.env.theatre_bridge import _kind_from_class

    comp = json.loads((models.MODELS_DIR.parents[2].parent / "components"
                       / "model.detection.json").read_text())
    classes = comp["parameters"]["threat_classes"]
    assert classes
    seen = set()
    for name, spec in classes.items():
        kind = _kind_from_class(name, spec)
        if kind is None:
            continue
        key = models.model_for_kind(kind)
        assert key in models.REGISTRY, name
        seen.add(key)
    assert "interceptor_drone" in seen and "sensor_site" in seen


@pytest.mark.parametrize("airborne", [False, True])
@pytest.mark.parametrize("speed", [0.0, 12.0, 150.0])
def test_the_mapping_is_total(airborne, speed):
    kind = ThreatKindConfig(airborne=airborne, speed=speed)
    assert models.model_for_kind(kind) in models.REGISTRY


# --- fallback, and only as a fallback ---------------------------------------------------


def test_the_normal_path_uses_no_fallback():
    audit = models.audit()
    assert {a["key"] for a in audit} == REQUIRED_CLASSES
    assert all(a["valid"] and a["documented"] and not a["fallback"] for a in audit), audit


def test_a_broken_file_is_reported_as_a_fallback(tmp_path, monkeypatch):
    originals = {k: s.path.read_bytes() for k, s in models.REGISTRY.items()}
    for k, s in models.REGISTRY.items():
        (tmp_path / s.file).write_bytes(originals[k] if k != "ground_vehicle" else b"not a glb")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    audit = {a["key"]: a for a in models.audit()}
    assert audit["ground_vehicle"]["fallback"] and not audit["ground_vehicle"]["valid"]
    assert not audit["sensor_site"]["fallback"]


def test_validation_catches_each_structural_failure():
    good = models.REGISTRY["sensor_site"].path.read_bytes()
    assert models.validate_glb(b"") != []
    assert models.validate_glb(b"x" * 40) == ["not a binary glTF (bad magic)"]
    assert any("missing node" in e for e in models.validate_glb(good, ("turret",)))
    truncated = good[:-8]
    assert any("length" in e for e in models.validate_glb(truncated))


def test_every_spec_carries_an_explicit_fallback_marker():
    for spec in models.REGISTRY.values():
        assert spec.fallback_pixel_size > 0
        assert re.fullmatch(r"#[0-9a-f]{6}", spec.fallback_color)


# --- what the page is handed ------------------------------------------------------------


def test_the_page_registry_points_at_the_route_or_inlines_the_bytes():
    served = models.page_registry(inline=False)
    assert served["aircraft"] == "fixed_wing"
    for key, m in served["models"].items():
        assert m["uri"] == models.MODEL_ROUTE + models.REGISTRY[key].file
        for field in ("scale", "minimum_pixel_size", "maximum_scale", "altitude_offset_m",
                      "heading_offset_deg", "pitch_offset_deg", "roll_offset_deg",
                      "turret_node", "required_nodes", "fallback_color", "anchor"):
            assert field in m, f"{key} missing {field}"
    inline = models.page_registry(inline=True)
    for m in inline["models"].values():
        assert m["uri"].startswith("data:model/gltf-binary;base64,")
    json.dumps(inline)


def test_the_route_serves_registry_files_and_nothing_else():
    for spec in models.REGISTRY.values():
        assert models.served_file(models.MODEL_ROUTE + spec.file) == spec.path
    for bad in ("/models/../live.py", "/models/README.md", "/models/", "/models/x.glb",
                "/models/%2e%2e/cesium.html", "/terrain", "/models/sub/fixed_wing.glb"):
        assert models.served_file(bad) is None, bad


def test_model_bytes_carry_nothing_credential_shaped():
    """The GLBs are committed and inlined into a committed artifact. The repo's
    secret scans read every tracked file; this is the same check aimed at the
    base64 the page will actually carry."""
    blob = json.dumps(models.page_registry(inline=True))
    assert not re.search(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.", blob)
    assert not re.search(r"AIza[0-9A-Za-z_-]{35}", blob)


def test_ground_models_rescale_with_the_relief_and_keep_their_offset():
    spec = models.REGISTRY["ground_vehicle"]
    assert models.rendered_anchor_altitude(1200.0, spec, 1.0) == 1200.0 + spec.altitude_offset_m
    assert models.rendered_anchor_altitude(1200.0, spec, 3.0) == 3600.0 + spec.altitude_offset_m
