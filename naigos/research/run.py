"""The research agent entry point: fetch -> cache -> cite -> emit specs.

Idempotent. A second run makes no network calls: everything already in ``data_cache`` is reused,
so training never depends on a live network. ``--force`` re-pulls.

    naigos-research                     # default AOI, reuse cache
    naigos-research --aoi front_range   # different theatre
    naigos-research --force             # re-pull everything
    naigos-research --skip-flights      # OpenSky sampling takes minutes; skip during iteration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import cache, spec
from .allowlist import ALLOWLIST, GUARDRAIL
from .aoi import AOIS, get_aoi
from .sources import airports, atmosphere, flights, radar, terrain


def _terrain_npz(art: cache.Artifact, aoi) -> cache.Artifact:
    """Derive the compact numpy grid the JAX env loads (no rasterio at training time)."""
    from naigos.data.terrain import TerrainGrid

    def build() -> bytes:
        import io

        grid = TerrainGrid.from_geotiff(art.abs_path)
        buf = io.BytesIO()
        np.savez_compressed(
            buf, z=grid.z, origin_x=grid.origin_x, origin_y=grid.origin_y,
            pixel_m=grid.pixel_m, crs=grid.crs,
        )
        return buf.getvalue()

    return cache.produce(
        key=f"dem_npz/{aoi.name}/{aoi.fingerprint}",
        source_key=aoi.dem_source, url=art.url,
        rel_path=f"terrain/{aoi.name}_{aoi.fingerprint}_dem_30m_utm.npz",
        builder=build,
        note="Derived from the cached GeoTIFF; the env's load path, numpy-only.",
    )


def build(
    aoi_name: str | None,
    force: bool,
    skip_flights: bool,
    n_snapshots: int,
    data_doc: Path | None = None,
) -> dict[str, Any]:
    """Run the whole research pipeline against the cache and component roots in effect.

    ``data_doc`` is where ``DATA.md`` is rendered; the default is the repository's
    ``docs/DATA.md``. A snapshot build passes a path inside its own snapshot so a
    scheduled refresh never rewrites a tracked file.
    """
    aoi = get_aoi(aoi_name)
    written: list[str] = []
    print(f"AOI {aoi.name} bbox={aoi.bbox} fingerprint={aoi.fingerprint}", file=sys.stderr)

    # --- terrain -----------------------------------------------------------------------
    print("[1/6] terrain (USGS 3DEP)", file=sys.stderr)
    dem = terrain.fetch_dem(aoi, force=force)
    dem_npz = _terrain_npz(dem, aoi)
    dem_summary = terrain.summarize(dem)
    masked = dem_summary["masking_potential"]["masked_fraction_by_height_above_ground"]

    written.append(str(spec.write_component(
        "env.aoi",
        role="Defines the theatre every other component is scoped to.",
        inputs=["WGS84 bounding box", "USGS 3DEP elevation"],
        outputs=["AOI bounds", "local UTM frame", "terrain relief statistics"],
        decision=(
            f"Use {aoi.name} ({aoi.bbox}) as the primary theatre, working in EPSG:"
            f"{terrain.utm_epsg(*aoi.center)} metres."
        ),
        rationale=aoi.rationale + (
            " The AOI was accepted only after the true line-of-sight test confirmed a monotonic "
            "masking gradient with height above ground; a flat theatre would make the signature "
            "mechanic unlearnable no matter how good the policy is."
        ),
        source_keys=["usgs_3dep"], artifacts=[dem],
        parameters={
            "name": aoi.name,
            "bbox_wgs84": list(aoi.bbox), "fingerprint": aoi.fingerprint,
            "utm_epsg": terrain.utm_epsg(*aoi.center),
            "span_km": [round(x, 1) for x in aoi.span_km()],
        },
        evidence={
            "relief_m": dem_summary["elevation_m"]["relief"],
            "mean_slope_deg": dem_summary["slope_deg"]["mean"],
            "masked_fraction_by_height_above_ground": masked,
        },
        invariants=[
            "Cache keys embed the AOI bounds fingerprint; changing the box invalidates the DEM.",
            "All env geometry is metric UTM; no per-step geodesy.",
        ],
    )))

    written.append(str(spec.write_component(
        "data.terrain_dem",
        role="Elevation grid backing radar line-of-sight masking and terrain-following flight.",
        inputs=["AOI bounds"],
        outputs=["north-up UTM elevation grid (GeoTIFF + npz)", "LOS clearance queries"],
        decision=(
            "Pull 3DEP at 30 m, reproject once to the AOI's UTM zone, clip to the AOI envelope, "
            "and expose LOS as a continuous clearance margin rather than a boolean."
        ),
        rationale=(
            "3DEP is public domain and seamless over the AOI. Reprojecting once at ingest keeps "
            "slant range a plain Euclidean distance in the env's inner loop. LOS returns a signed "
            "clearance in metres because a hard 0/1 mask gives the policy no gradient to climb or "
            "descend along -- the sign is the constraint, the magnitude is the learning signal."
        ),
        source_keys=["usgs_3dep"], artifacts=[dem, dem_npz],
        parameters={
            "resolution_m": terrain.RESOLUTION_M, "crs": dem_summary["crs"],
            "grid_shape": dem_summary["grid_shape"], "extent_km": dem_summary["extent_km"],
            "origin_easting_m": dem_summary["origin_easting_m"],
            "origin_northing_m": dem_summary["origin_northing_m"],
        },
        evidence={
            "elevation_m": dem_summary["elevation_m"], "slope_deg": dem_summary["slope_deg"],
            "masking": dem_summary["masking_potential"],
        },
        invariants=[
            "Grid is north-up and unrotated; rotated rasters are rejected at load.",
            "No-data is treated as 'no terrain evidence' (non-blocking), never as cover.",
        ],
        caveats=[
            "3DEP is bare-earth: vegetation and structures are absent, so masking is a floor, "
            "not a ceiling.",
            "CONUS-only. A non-US AOI must switch to the Copernicus GLO-30 fallback.",
        ],
    )))

    # --- airfields ---------------------------------------------------------------------
    print("[2/6] airfields (OurAirports)", file=sys.stderr)
    ap_arts = airports.fetch_tables(force=force)
    airfields = airports.extract_airfields(aoi, ap_arts)
    reconcile = airports.reconcile_with_dem(airfields, dem)

    written.append(str(spec.write_component(
        "data.airfields",
        role="Surveyed start points and objectives inside the theatre.",
        inputs=["AOI bounds", "OurAirports airports.csv + runways.csv", "3DEP elevation"],
        outputs=["airfield list with UTM position, field elevation and runway geometry"],
        decision="Spawn and task aircraft from real airfields at surveyed positions and elevations.",
        rationale=(
            "Random spawn points inside a mountain range put aircraft underground or in "
            "unrecoverable attitudes. Real fields also give a real initial heading (the runway) "
            "and a real initial altitude (field elevation), so episode start states are "
            "physically consistent with the terrain the aircraft is about to fly over."
        ),
        source_keys=["ourairports", "usgs_3dep"], artifacts=list(ap_arts.values()),
        parameters={"n_airfields": len(airfields), "airfields": airfields},
        evidence={"dem_cross_check": reconcile},
        invariants=["Every airfield's UTM position lies inside the DEM grid bounds."],
        caveats=[
            "Only fixed-wing field types are kept; heliports and seaplane bases are excluded.",
            "Some minor fields carry no runway record, so they are usable as objectives but not "
            "as departure points.",
        ],
    )))

    # --- atmosphere --------------------------------------------------------------------
    print("[3/6] atmosphere (Open-Meteo)", file=sys.stderr)
    atm_art = atmosphere.fetch_profile(aoi, force=force)
    profile = atmosphere.derive_profile(atm_art)
    k_earth = profile["refraction"]["effective_earth_factor_k"]
    sfc_density_ratio = profile["surface"]["density_ratio_to_sea_level"]

    written.append(str(spec.write_component(
        "data.atmosphere",
        role="Air density vs altitude, winds aloft, and the local radar refraction factor.",
        inputs=["AOI centre coordinates", "Open-Meteo pressure-level forecast"],
        outputs=["density profile", "wind profile", "effective-Earth factor k"],
        decision=(
            f"Parameterize airframe performance by measured density ratio, and set the LOS "
            f"model's effective-Earth factor to the measured k={k_earth} rather than the "
            "textbook 4/3."
        ),
        rationale=(
            "Stall speed, available thrust and true airspeed all scale with density, and the AOI "
            "floor already sits near 1.2 km with terrain to 4.4 km, so a sea-level atmosphere "
            "would misstate the whole flight envelope. The refraction factor was an unplanned "
            "find: the measured refractivity gradient over this dry high-desert air is "
            f"{profile['refraction']['dN_dh_N_units_per_km_lowest_3km']} N-units/km, giving "
            f"k={k_earth} against the standard 1.3333. That shortens the radar horizon, and "
            "over a 100 km theatre the difference is kilometres of detection range."
        ),
        source_keys=["open_meteo"], artifacts=[atm_art],
        parameters={
            "pressure_levels_hPa": list(atmosphere.PRESSURE_LEVELS_HPA),
            "effective_earth_factor_k": k_earth,
            "surface_density_ratio": sfc_density_ratio,
        },
        evidence=profile,
        caveats=[
            "A 3-day forecast window at one point, not a climatology. It fixes the order of "
            "magnitude and the vertical shape; it is not a seasonal distribution.",
        ],
    )))

    # --- flight kinematics -------------------------------------------------------------
    print("[4/6] flight kinematics (OpenSky)", file=sys.stderr)
    envelope: dict[str, Any] | None = None
    if not skip_flights:
        fl_art = flights.sample_states(n_snapshots=n_snapshots, force=force)
        envelope = flights.derive_envelope(fl_art)
        written.append(str(spec.write_component(
            "data.flight_envelope",
            role="Measured airframe operating envelope used to calibrate the point-mass model.",
            inputs=["OpenSky /states/all snapshots over a busy calibration box"],
            outputs=["speed, climb, descent, turn-rate and implied-bank distributions"],
            decision=(
                "Bound the point-mass airframe's speed and climb limits by measured civil "
                "traffic, and validate the coordinated-turn relation omega = g*tan(phi)/V "
                "against observed turn rates instead of asserting a g-limit."
            ),
            rationale=(
                "The spec's requirement is that g-limits and speeds be realistic rather than "
                "invented. Consecutive ADS-B snapshots of the same aircraft give a finite-"
                "difference turn rate that the instantaneous state vector does not carry, which "
                "is what makes the bank-angle check possible at all."
            ),
            source_keys=["opensky"], artifacts=[fl_art],
            parameters={
                "calibration_box": flights.CALIBRATION_BOX,
                "snapshot_interval_s": flights.SNAPSHOT_INTERVAL_S,
            },
            evidence=envelope,
            caveats=[
                envelope["caveat"],
                "Snapshot sampling at 12 s cannot resolve manoeuvres shorter than that, so peak "
                "turn rates are understated.",
            ],
        )))
    else:
        print("      skipped (--skip-flights)", file=sys.stderr)

    # --- detection model ---------------------------------------------------------------
    print("[5/6] detection model (radar range equation)", file=sys.stderr)
    threats = radar.derive_threat_classes(density_ratio=sfc_density_ratio, k_earth=k_earth)

    written.append(str(spec.write_component(
        "model.detection",
        role="Detection probability and lethal envelopes -- the signature mechanic's physics.",
        inputs=["radar range equation", "terrain LOS clearance", "measured density and k"],
        outputs=["per-threat-class detection range, Pd curve, lethal envelope, reaction latency"],
        decision=(
            "Derive every threat envelope from the monostatic range equation with Swerling-1 "
            "detection statistics. Each class declares a notional role-based design range; the "
            "transmit power is then solved from the equation rather than invented."
        ),
        rationale=(
            "Solving for power instead of guessing it means the only free choice per class is a "
            "deliberately round, generic engagement distance, and everything else -- the "
            "RCS^(1/4) range scaling, the frequency-dependent gaseous attenuation, the shape of "
            "the Pd ramp -- follows from physics. Swerling 1 rather than a non-fluctuating model "
            "because an aircraft's RCS swings by an order of magnitude with small aspect changes, "
            "and a non-fluctuating model turns detection on far too sharply to give the policy a "
            "usable exposure gradient."
        ),
        source_keys=["radar_theory", "open_meteo"],
        parameters={
            "reference_rcs_m2": radar.REFERENCE_RCS_M2,
            "effective_earth_factor_k": k_earth,
            "surface_density_ratio": sfc_density_ratio,
            "threat_classes": threats,
        },
        evidence={
            "pd_curve_example": {
                "class": "medium_sam_acquisition", "rcs_m2": 1.0,
                "range_km": [20, 40, 60, 80, 100, 110, 120, 150, 200],
                "pd": [
                    round(float(x), 4) for x in radar.detection_probability(
                        radar.snr_at_range(
                            radar.calibrate_power(
                                radar.RADAR_CLASSES["medium_sam_acquisition"], 1.0
                            ),
                            np.array([20, 40, 60, 80, 100, 110, 120, 150, 200]) * 1000.0, 1.0,
                        ),
                        radar.RADAR_CLASSES["medium_sam_acquisition"].prob_false_alarm,
                    )
                ],
            },
        },
        invariants=[
            "Blue never acts on a threat. These envelopes are environment, not targets.",
            "Detection range always exceeds lethal range, so 'detected but not engaged' is a "
            "state the policy can still act from.",
            "No parameter here describes any real fielded system.",
        ],
        caveats=[
            "Free-space plus terrain LOS. Multipath, sidelobe and ground-clutter effects are not "
            "modelled; the pattern-propagation factor is folded into the design range.",
            GUARDRAIL.strip(),
        ],
    )))

    # --- the demo's visual skin, kept explicitly apart from the physics -----------------
    written.append(str(_write_imagery_component()))

    # --- the agent itself --------------------------------------------------------------
    print("[6/6] provenance", file=sys.stderr)
    written.append(str(spec.write_component(
        "research.agent",
        role="The data layer's own contract: what may be fetched, and what may not.",
        inputs=["fixed source allowlist"],
        outputs=["cached raw artifacts", "cited component specs", "docs/DATA.md"],
        decision=(
            "Every fetch is host-checked against a fixed allowlist, cached with a sha256 and a "
            "citation, and distilled into a component spec. There is no open-ended web search."
        ),
        rationale=(
            "Reproducibility and scope control. The allowlist makes the data layer auditable and "
            "offline-after-first-run; the guardrail keeps threat modelling parameterized, which "
            "is both the responsible choice and the correct one -- the RL problem depends on the "
            "shape of the exposure-versus-survival tradeoff, not on real-system accuracy."
        ),
        source_keys=sorted(ALLOWLIST),
        parameters={
            "allowlisted_hosts": sorted({h for s in ALLOWLIST.values() for h in s.hosts}),
            "cache_dir": "data_cache/", "manifest": "data_cache/manifest.json",
        },
        invariants=[
            "Non-allowlisted hosts raise SourceNotAllowed; nothing enters the cache uncited.",
            "write_component refuses a component with no sources.",
            "Re-running makes no network calls unless --force is passed.",
        ],
        caveats=[GUARDRAIL.strip()],
    )))

    write_data_doc(aoi, dem_summary, airfields, reconcile, profile, envelope, threats,
                   out=data_doc)
    return {"components": written, "artifacts": sorted(cache.load_manifest())}


def _write_imagery_component():
    """Emit `demo.imagery`: the one component that parameterizes nothing.

    It is here because the licence question is real and the separation is
    load-bearing, not because the env reads it. Satellite pixels are a skin over
    a globe; the DEM is what the detection model consumes. Recording that as a
    cited component means the distinction is auditable in the same place as
    every other design decision, instead of living only in a comment.
    """
    from ..demo import imagery as imagery_mod

    return spec.write_component(
        "demo.imagery",
        role=(
            "Visual providers for the demo globe. Cosmetic only -- never observed by the "
            "policy."
        ),
        inputs=[
            "Cesium ion asset 3954 (Copernicus Sentinel-2)",
            "Cesium ion asset 2275207 (Google Photorealistic 3D Tiles), optional",
        ],
        outputs=["viewer base imagery layer", "optional 3D tileset", "on-screen attribution"],
        decision=(
            "Two visual modes behind one --visual flag. physics (the default) drapes "
            "Copernicus Sentinel-2 (Cesium ion asset 3954) over the simulation's own terrain, "
            "as a layer strictly separate from the DEM. photorealistic streams Google "
            "Photorealistic 3D Tiles through CesiumJS instead, which replaces the drawn "
            "surface with the provider's geometry and is therefore marked "
            "evidence_grade=false. Credentials come from explicit environment variables and "
            "are never committed; with none present, photorealistic falls back to physics and "
            "Sentinel-2 falls back to keyless OpenStreetMap."
        ),
        rationale=(
            "Two separate reasons. Licensing: Cesium's default base imagery is Bing Aerial -- "
            "third-party commercial data, metered by session, under Microsoft's terms rather "
            "than Cesium's. Sentinel-2 is ESA/Copernicus open data, free to use rather than "
            "merely free to view, which removes the question from a portfolio project. "
            "Correctness: imagery and elevation must not come from the same provider, because "
            "the globe's relief is evidence -- it is the surface line-of-sight was computed "
            "against. Taking terrain from an imagery provider is the defect this viewer "
            "already shipped once (next-steps E-9). Imagery is therefore a skin with no "
            "downstream consumer at all. The photorealistic mode exists because the same "
            "argument runs the other way: Google's 3D Tiles are the best-looking globe "
            "available and bring their own geometry, so they are offered as an explicitly "
            "non-evidential presentation mode rather than smuggled in as a prettier default."
        ),
        source_keys=["copernicus_sentinel2", "google_photorealistic_3d_tiles"],
        parameters={
            "visual_modes": list(imagery_mod.VISUAL_MODES),
            "default_visual_mode": imagery_mod.DEFAULT_VISUAL_MODE,
            "ion_asset_id": imagery_mod.SENTINEL2_ION_ASSET,
            "google_3d_tiles_ion_asset_id": imagery_mod.GOOGLE_3D_TILES_ION_ASSET,
            "token_env_vars": list(imagery_mod.TOKEN_ENV_VARS),
            "google_api_key_env_vars": list(imagery_mod.GOOGLE_API_KEY_ENV_VARS),
            "fallback": "OpenStreetMap (keyless)",
            "attribution": imagery_mod.SENTINEL2_ATTRIBUTION,
            "google_attribution": imagery_mod.GOOGLE_3D_TILES_ATTRIBUTION,
        },
        evidence={
            "layers_are_separate": (
                "In physics mode the viewer builds imagery from IonImageryProvider and terrain "
                "from CustomHeightmapTerrainProvider over /terrain; "
                "tests/test_imagery_layers.py asserts no ion terrain provider is ever "
                "constructed."
            ),
            "mode_resolution_and_fallbacks": (
                "naigos.demo.imagery.resolve_visual_config resolves the requested mode against "
                "the credentials present and degrades toward physics; "
                "tests/test_visual_modes.py covers resolution, missing credentials, the safe "
                "fallback and token non-persistence."
            ),
            "no_pixels_in_the_observation": (
                "No module under naigos/env or naigos/rl references imagery, ion or "
                "Sentinel-2; asserted by test."
            ),
        },
        invariants=[
            "The detection model consumes the DEM only. No satellite pixel enters an observation.",
            "Terrain never comes from an imagery provider; the globe renders the env's heightmap.",
            "No Cesium ion token is committed to the repository; it is read from the environment.",
            "Sentinel-2 attribution is displayed on screen whenever the imagery is used.",
            "physics is the default visual mode and is the only evidence-grade one.",
            "No credential is ever stored in a config object, written to disk or sent to the "
            "page; only a boolean saying whether one was found.",
            "photorealistic falls back to physics when its credentials are absent, never the "
            "other way round.",
        ],
        caveats=[
            "Imagery is decorative. It establishes nothing about the simulation and is not "
            "registered to the DEM beyond both being georeferenced to WGS84.",
            "Cesium ion Community tier covers individual and non-commercial use only.",
            "In photorealistic mode the drawn surface is Google's, not the simulation's DEM, "
            "so nothing in that mode is evidence about terrain masking. VisualConfig marks it "
            "evidence_grade=false and the viewer HUD says so.",
            "Google Photorealistic 3D Tiles are metered under Google Maps Platform terms; "
            "unlike Sentinel-2 they are free to look at, not free to use.",
        ],
    )


#: The headings that bound the allowlist-derived half of docs/DATA.md. The
#: refresh path below rewrites exactly this span and nothing else.
SOURCES_HEADING = "## Sources"
AFTER_SOURCES_HEADING = "## Cached artifacts"


def source_sections() -> list[str]:
    """The Sources and Attribution sections of ``docs/DATA.md``, as lines.

    Split out of :func:`write_data_doc` because it depends on nothing but
    ``ALLOWLIST`` -- no cache, no network, no AOI. Attribution is a licence
    obligation, and an obligation that can only be restated by a full research
    run is one the tree is silently out of date on between runs. This is what
    ``--refresh-docs`` writes and what ``tests/test_data_doc.py`` checks the
    committed file against.
    """
    lines = [
        SOURCES_HEADING, "",
        "| source | license | role |", "| --- | --- | --- |",
    ]
    for s in ALLOWLIST.values():
        lines.append(f"| {s.name} | {s.license} | {s.role} |")

    # Attribution is a licence obligation, so it is generated rather than left to
    # whoever remembers. Sources flagged attribution_required name themselves here.
    lines += [
        "", "## Attribution", "",
        "Sources marked *attribution required* above must be credited wherever their data is "
        "shown. The demo globe displays: **Contains modified Copernicus Sentinel data** "
        "(Sentinel-2 imagery, served as Cesium ion asset 3954), alongside Cesium's own credit "
        "display, which is left visible on purpose.", "",
        "Note what that imagery is *not*: it is a skin. Elevation -- the only geospatial "
        "quantity the detection model consumes -- comes from the DEM rows above, never from an "
        "imagery provider.", "",
        "The optional `--visual photorealistic` mode adds Google Photorealistic 3D Tiles, which "
        "carry their own geometry: in that mode the surface on screen is the provider's and not "
        "the simulation's DEM, so it is presentation only and is not evidence about terrain "
        "masking. Google's attribution and the per-tile credits stay on screen for as long as "
        "the tiles are drawn, and the physics mode that the viewer boots into -- and falls back "
        "to -- never requests them at all.", "",
        "The optional `--visual urban-presentation` mode draws OpenStreetMap building footprints "
        "and major-road centrelines, fetched once by `naigos.demo.urban` into "
        "`data_cache/visual/urban/` and extruded over the simulation's DEM. They are presentation "
        "only -- not terrain, not radar cover, never read by LOS or detection -- and are credited "
        "on screen as (c) OpenStreetMap contributors under the ODbL wherever they are drawn.", "",
        "| source requiring attribution | citation |", "| --- | --- |",
    ]
    for s in ALLOWLIST.values():
        if s.attribution_required:
            lines.append(f"| {s.name} | {s.citation} |")
    return lines


def refresh_source_sections() -> Path:
    """Rewrite only the allowlist-derived span of ``docs/DATA.md``, in place.

    The rest of the doc reports measurements and needs the cache to regenerate;
    these two sections need only the allowlist, so a licence change can be
    landed with the commit that makes it rather than waiting on a research run.
    """
    out = cache.REPO_ROOT / "docs" / "DATA.md"
    text = out.read_text()
    head, _, rest = text.partition(SOURCES_HEADING)
    if not rest:
        raise ValueError(f"{out} has no {SOURCES_HEADING!r} section to refresh")
    tail = AFTER_SOURCES_HEADING + rest.partition(AFTER_SOURCES_HEADING)[2]
    out.write_text(head + "\n".join(source_sections()) + "\n\n" + tail)
    return out


def write_data_doc(aoi, dem_summary, airfields, reconcile, profile, envelope, threats,
                   out: Path | None = None) -> None:
    """Render docs/DATA.md (or ``out``), the human-readable provenance log."""
    m = cache.load_manifest()
    lines = [
        "# DATA.md - provenance", "",
        "Generated by `naigos-research`. Every number the simulation uses traces to a row here.",
        "Raw bytes live in `data_cache/` (gitignored, regenerable); "
        "`data_cache/manifest.json` records the sha256 of each.", "",
        f"**Theatre:** {aoi.name} `{aoi.bbox}` "
        f"({' x '.join(str(round(x)) for x in aoi.span_km())} km), fingerprint `{aoi.fingerprint}`", "",
    ]
    lines += source_sections()

    lines += ["", AFTER_SOURCES_HEADING, "",
              "| key | path | size | sha256 | fetched |", "| --- | --- | --- | --- | --- |"]
    for key in sorted(m):
        a = m[key]
        lines.append(
            f"| `{key}` | `{a['path']}` | {a['bytes'] / 1e6:.1f} MB | "
            f"`{a['sha256'][:12]}` | {a['fetched_at']} |"
        )

    e = dem_summary["elevation_m"]
    masked = dem_summary["masking_potential"]["masked_fraction_by_height_above_ground"]
    lines += [
        "", "## What the data established", "",
        "### Terrain masking is a real, learnable mechanic here",
        "",
        f"The DEM spans {e['min']}-{e['max']} m ({e['relief']} m relief), mean slope "
        f"{dem_summary['slope_deg']['mean']} deg. Running the true line-of-sight test "
        "(4/3-Earth geometry, per-ray terrain sampling) from a sensor on the AOI high point "
        "against 1500 random points:", "",
        "| height above ground | fraction masked from the high-point sensor |", "| --- | --- |",
    ]
    for k, v in masked.items():
        lines.append(f"| {k.replace('m_agl', ' m')} | {v:.1%} |")
    lines += [
        "",
        "That monotonic gradient is the whole premise of the project: flying low genuinely buys "
        "concealment, and it is measured from real terrain, not asserted.", "",
        "### The projection chain is independently validated", "",
        f"DEM elevation was compared against published field elevations at "
        f"{reconcile['n_compared']} airfields: mean difference "
        f"{reconcile['mean_delta_m']} m, worst case {reconcile['abs_max_delta_m']} m. Two "
        "independent sources agreeing to a few metres means the WGS84 -> UTM reprojection every "
        "LOS ray depends on is correct.", "",
        "### The local atmosphere is not the textbook atmosphere", "",
        f"Measured refractivity gradient over the lowest 3 km: "
        f"{profile['refraction']['dN_dh_N_units_per_km_lowest_3km']} N-units/km, giving an "
        f"effective-Earth factor k = {profile['refraction']['effective_earth_factor_k']} against "
        "the standard 1.3333. Dry high-desert air refracts less, so the radar horizon is shorter "
        "than the default assumption. Surface air density is "
        f"{profile['surface']['density_ratio_to_sea_level']:.3f} of sea level at the valley floor.",
        "",
    ]

    if envelope:
        gs, cr, tr = envelope["ground_speed_ms"], envelope["climb_rate_ms"], envelope["turn_rate_deg_s"]
        bank = envelope["implied_bank_deg"]
        lines += [
            "### The airframe envelope is measured, not invented", "",
            f"{envelope['n_airborne_states']} airborne states across "
            f"{envelope['n_snapshots']} snapshots, {envelope['n_tracked_pairs']} consecutive "
            "pairs of the same aircraft:", "",
            "| quantity | p5 | p50 | p95 | max |", "| --- | --- | --- | --- | --- |",
            f"| ground speed (m/s) | {gs.get('p5')} | {gs.get('p50')} | {gs.get('p95')} | {gs.get('max')} |",
            f"| climb rate (m/s) | {cr.get('p5')} | {cr.get('p50')} | {cr.get('p95')} | {cr.get('max')} |",
            f"| turn rate (deg/s) | {tr.get('p5')} | {tr.get('p50')} | {tr.get('p95')} | {tr.get('max')} |",
            f"| implied bank (deg) | {bank.get('p5')} | {bank.get('p50')} | {bank.get('p95')} | {bank.get('max')} |",
            "", envelope["caveat"], "",
        ]

    lines += ["### Threat envelopes are derived, not looked up", "",
              "Each class declares a notional role-based design range; the range equation solves "
              "the transmit power. Everything else follows from physics.", "",
              "| class | design range | Pd=0.5 | Pd=0.9 | lethal | reaction |",
              "| --- | --- | --- | --- | --- | --- |"]
    for k, v in threats.items():
        d, en = v["derived"], v["engagement"]
        lines.append(
            f"| {k} | {v['radar']['design_range_km']:.0f} km | "
            f"{d['detection_range_km_pd50']:.0f} km | {d['detection_range_km_pd90']:.0f} km | "
            f"{d['lethal_range_km']:.0f} km | {en['reaction_latency_s']:.0f} s |"
        )
    lines += ["", "## Scope guardrail", "", "```", GUARDRAIL.strip(), "```", ""]

    out = Path(out) if out is not None else cache.REPO_ROOT / "docs" / "DATA.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"      wrote {out}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Naigos research agent: fetch -> cache -> cite.")
    ap.add_argument("--aoi", default=None, choices=sorted(AOIS), help="theatre to build")
    ap.add_argument("--force", action="store_true", help="re-pull everything, ignoring the cache")
    ap.add_argument("--skip-flights", action="store_true", help="skip OpenSky sampling (slow)")
    ap.add_argument("--snapshots", type=int, default=26, help="OpenSky snapshots to collect")
    ap.add_argument(
        "--refresh-docs", action="store_true",
        help="rewrite only the allowlist-derived sections of docs/DATA.md and exit (no cache needed)",
    )
    args = ap.parse_args()

    # The licence half of the doc, without a full run. A source added to
    # ALLOWLIST changes an obligation immediately; the measurements it will
    # eventually be cited beside can wait for the next fetch.
    if args.refresh_docs:
        out = refresh_source_sections()
        print(f"wrote {out.relative_to(cache.REPO_ROOT)}", file=sys.stderr)
        return 0

    result = build(args.aoi, args.force, args.skip_flights, args.snapshots)

    # Snapshot this AOI's component set. Without it a run for a second theatre
    # overwrites the first, and a checkpoint trained on the first becomes
    # unreproducible while every file on disk still looks perfectly valid.
    from .aoi import DEFAULT_AOI
    from .spec import snapshot_aoi

    dest = snapshot_aoi(args.aoi or DEFAULT_AOI)
    result["aoi_snapshot"] = str(dest.relative_to(cache.REPO_ROOT))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
