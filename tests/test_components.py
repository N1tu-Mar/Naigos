"""The citation contract, and the emitted specs checked against the real cached data.

Integration tests skip cleanly when the cache has not been built, so the unit suite stays
standalone (spec section 9).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from naigos.research import spec
from naigos.research.allowlist import ALLOWLIST
from naigos.research.cache import CACHE_DIR, MANIFEST_PATH, load_manifest

COMPONENT_IDS = [
    "env.aoi", "data.terrain_dem", "data.airfields",
    "data.atmosphere", "data.flight_envelope", "model.detection", "research.agent",
    # Parameterizes nothing in the env -- it records the demo's imagery licence and
    # the imagery/terrain separation. Cited like everything else precisely because
    # it is the one layer with no downstream consumer.
    "demo.imagery",
]

needs_cache = pytest.mark.skipif(
    not (spec.COMPONENTS_DIR / "env.aoi.json").exists(),
    reason="research cache not built; run `naigos-research`",
)


# --- the citation contract (no cache needed) -------------------------------------------


def test_a_component_without_sources_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "COMPONENTS_DIR", tmp_path)
    with pytest.raises(spec.UncitedComponent):
        spec.write_component(
            "test.uncited", role="r", inputs=["i"], outputs=["o"],
            decision="d", rationale="why", source_keys=[],
        )


def test_a_component_citing_a_non_allowlisted_source_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "COMPONENTS_DIR", tmp_path)
    with pytest.raises(spec.UncitedComponent):
        spec.write_component(
            "test.bad_source", role="r", inputs=["i"], outputs=["o"],
            decision="d", rationale="why", source_keys=["some_random_blog"],
        )


def test_a_valid_component_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, "COMPONENTS_DIR", tmp_path)
    path = spec.write_component(
        "test.ok", role="r", inputs=["i"], outputs=["o"],
        decision="d", rationale="why", source_keys=["ourairports"],
    )
    doc = json.loads(path.read_text())
    assert doc["sources"][0]["key"] == "ourairports"
    assert doc["license"] == ALLOWLIST["ourairports"].license
    assert doc["generated_at"].endswith("Z")


# --- the emitted specs -----------------------------------------------------------------


@needs_cache
@pytest.mark.parametrize("cid", COMPONENT_IDS)
def test_every_component_is_complete_and_cited(cid):
    doc = spec.load_component(cid)
    for field in spec.REQUIRED_FIELDS:
        assert doc.get(field), f"{cid} missing {field}"
    assert doc["sources"], f"{cid} has no sources"
    for src in doc["sources"]:
        assert src["key"] in ALLOWLIST
        assert src["license"] and src["citation"]


@needs_cache
def test_every_cached_artifact_referenced_by_a_component_exists_and_matches_its_hash():
    """Provenance is only worth something if the bytes still match the recorded sha256."""
    import hashlib

    for cid in COMPONENT_IDS:
        for art in spec.load_component(cid)["cached_artifacts"]:
            path = CACHE_DIR / art["path"]
            assert path.exists(), f"{cid} references missing {art['path']}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == art["sha256"], f"{art['path']} changed since it was cited"


@needs_cache
def test_the_manifest_covers_every_referenced_artifact():
    manifest = load_manifest()
    assert MANIFEST_PATH.exists()
    for cid in COMPONENT_IDS:
        for art in spec.load_component(cid)["cached_artifacts"]:
            assert art["key"] in manifest


@needs_cache
def test_the_detection_component_never_grants_blue_a_weapon():
    """The hard invariant, asserted in the data layer where threats are described."""
    doc = spec.load_component("model.detection")
    blob = json.dumps(doc).lower()
    for forbidden in ("blue_weapon", "fire_action", "engage_action", "strike_action"):
        assert forbidden not in blob
    assert any("blue never acts on a threat" in inv.lower() for inv in doc["invariants"])


@needs_cache
def test_threat_classes_can_always_see_further_than_they_can_shoot():
    classes = spec.load_component("model.detection")["parameters"]["threat_classes"]
    for name, c in classes.items():
        assert c["derived"]["lethal_range_km"] < c["derived"]["detection_range_km_pd50"], name


@needs_cache
def test_the_guardrail_is_recorded_in_the_shipped_specs():
    for cid in ("model.detection", "research.agent"):
        caveats = " ".join(spec.load_component(cid)["caveats"]).lower()
        assert "parameterized abstractions" in caveats or "parameterizes instead" in caveats


# --- the real theatre ------------------------------------------------------------------


@needs_cache
def test_the_theatre_loads_and_is_internally_consistent():
    from naigos.data.theatre import load_theatre

    t = load_theatre()
    min_x, min_y, max_x, max_y = t.terrain.bounds

    assert t.terrain.pixel_m == 30.0
    assert t.airfields, "no airfields in the theatre"
    for a in t.airfields:
        assert min_x <= a.easting_m <= max_x, f"{a.ident} outside the DEM in easting"
        assert min_y <= a.northing_m <= max_y, f"{a.ident} outside the DEM in northing"

    assert t.departure_fields(), "no airfield usable as a departure point"
    assert 1.0 < t.effective_earth_factor_k < 1.6
    assert 0.5 < t.surface_density_ratio < 1.0


@needs_cache
def test_airfield_elevations_agree_with_the_dem():
    """Two independent sources must agree, or the projection chain is wrong."""
    from naigos.data.theatre import load_theatre

    t = load_theatre()
    deltas = []
    for a in t.airfields:
        dem_z = float(t.terrain.elevation(np.array(a.easting_m), np.array(a.northing_m)))
        if np.isfinite(dem_z):
            deltas.append(dem_z - a.elevation_m)
    assert deltas
    assert max(abs(d) for d in deltas) < 30.0, f"elevation disagreement too large: {deltas}"


@needs_cache
def test_the_terrain_actually_masks_low_flight():
    """The project's premise, asserted against the real DEM: low flight buys concealment."""
    from naigos.data.terrain import masked_fraction
    from naigos.data.theatre import load_theatre

    t = load_theatre()
    z = t.terrain.z
    r, c = np.unravel_index(int(np.nanargmax(z)), z.shape)
    sensor = (
        t.terrain.origin_x + (c + 0.5) * t.terrain.pixel_m,
        t.terrain.origin_y - (r + 0.5) * t.terrain.pixel_m,
        float(z[r, c]) + 10.0,
    )
    low = masked_fraction(t.terrain, sensor, target_agl_m=100.0, n_samples=250, seed=1)
    high = masked_fraction(t.terrain, sensor, target_agl_m=3000.0, n_samples=250, seed=1)
    assert low > 0.4, f"terrain masks too little at 100 m AGL ({low:.2f}) for the mechanic to matter"
    assert low > high


@needs_cache
def test_the_airframe_envelope_is_physically_ordered():
    from naigos.data.theatre import load_theatre

    af = load_theatre().airframe
    assert af.min_speed_ms < af.cruise_speed_ms < af.max_speed_ms
    assert af.max_climb_rate_ms > 0 and af.max_descent_rate_ms > 0
    assert af.max_load_factor > 1.0
    # Turn rate must fall as speed rises: omega = g*tan(phi)/V.
    assert af.max_turn_rate_deg_s(100.0) > af.max_turn_rate_deg_s(250.0)


@needs_cache
def test_measured_bank_angles_land_in_the_civil_range():
    """Validates omega = g*tan(phi)/V against real traffic.

    The bank angles implied by measured ADS-B turn rates must fall in the range airliners
    actually fly. If they did not, the coordinated-turn relation the airframe model is built
    on would be wrong, and every g-limit downstream with it.
    """
    from naigos.data.theatre import load_theatre

    af = load_theatre().airframe
    assert 0.0 < af.observed_p95_bank_deg < 20.0
    assert af.observed_max_bank_deg < 45.0
