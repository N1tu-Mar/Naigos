"""Offline fixtures for the pipeline tests: a snapshot tree with no network and no cache.

Built with the real `cache.record` and `spec.write_component`, pointed at a
temporary root, so the tree has exactly the shape the research agent writes --
only the bytes are small and fake.
"""

from __future__ import annotations

from pathlib import Path

from naigos.research import cache, roots, spec
from naigos.research.allowlist import GUARDRAIL
from naigos.research.aoi import get_aoi


def write_research_tree(staging: Path, aoi_name: str = "owens_valley", *,
                        atmosphere_payload: bytes | None = None, calls: list | None = None) -> dict:
    """What `naigos.research.run.build` + `snapshot_aoi` produce, in miniature.

    Honours the cache contract: an artifact already in the manifest is reused
    (the builder is not called), so seeded snapshots re-fetch only what was
    dropped. `calls` records which builders actually ran.
    """
    aoi = get_aoi(aoi_name)
    calls = calls if calls is not None else []

    def produce(key, source, url, rel, payload):
        def build():
            calls.append(key)
            return payload
        return cache.produce(key, source, url, rel, build)

    with roots.research_roots(cache_dir=staging / "cache", components_dir=staging / "components"):
        fp = aoi.fingerprint
        dem = produce(f"dem/{aoi.name}/30m/{fp}", "usgs_3dep",
                      "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
                      f"terrain/{aoi.name}_{fp}_dem_30m_utm.tif", b"tif-bytes")
        npz = produce(f"dem_npz/{aoi.name}/{fp}", "usgs_3dep", dem.url,
                      f"terrain/{aoi.name}_{fp}_dem_30m_utm.npz", b"npz-bytes")
        ap = produce("ourairports/airports", "ourairports",
                     "https://davidmegginson.github.io/ourairports-data/airports.csv",
                     "airspace/ourairports_airports.csv", b"ident,name\n")
        atm = produce(f"open_meteo/profile/{aoi.name}/{fp}", "open_meteo",
                      "https://api.open-meteo.com/v1/forecast",
                      f"atmosphere/open_meteo_{aoi.name}_{fp}_profile.json",
                      atmosphere_payload or b'{"hourly": {}}')
        fl = produce("opensky/states/2x12s", "opensky", "https://opensky-network.org/api/states/all",
                     "flights/opensky_states_2x12s.json", b'{"snapshots": []}')
        common = dict(role="r", inputs=["i"], outputs=["o"], decision="d", rationale="why")
        spec.write_component("env.aoi", source_keys=["usgs_3dep"], artifacts=[dem], parameters={
            "name": aoi.name, "fingerprint": fp, "bbox_wgs84": list(aoi.bbox)}, **common)
        spec.write_component("data.terrain_dem", source_keys=["usgs_3dep"], artifacts=[dem, npz], **common)
        spec.write_component("data.airfields", source_keys=["ourairports", "usgs_3dep"], artifacts=[ap], **common)
        spec.write_component("data.atmosphere", source_keys=["open_meteo"], artifacts=[atm], **common)
        spec.write_component("data.flight_envelope", source_keys=["opensky"], artifacts=[fl], **common)
        spec.write_component(
            "model.detection", source_keys=["radar_theory", "open_meteo"],
            invariants=["Blue never acts on a threat. These envelopes are environment, not targets."],
            caveats=["free-space model", GUARDRAIL.strip()], **common)
        spec.write_component("research.agent", source_keys=["usgs_3dep", "open_meteo"],
                             caveats=[GUARDRAIL.strip()], **common)
        spec.snapshot_aoi(aoi.name)
    (staging / "DATA.md").write_text("# DATA.md - provenance (fixture)\n")
    return {"components": ["env.aoi.json"], "artifacts": sorted(cache.load_manifest()),
            "egress_guard": None}


def fake_runner(calls: list | None = None, *, fail: bool = False, payload: bytes | None = None):
    """A `research_runner` for `snapshot.build` that never touches the network."""

    def run(*, staging, aoi, skip_flights, flight_snapshots, timeout_s):
        if fail:
            raise RuntimeError("upstream returned 503")
        return write_research_tree(staging, aoi, calls=calls, atmosphere_payload=payload)

    return run
