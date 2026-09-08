"""Imagery and terrain are two layers, and the difference is a correctness claim.

The globe's relief is evidence: it is the surface every line-of-sight ray was
computed against, so a viewer that takes elevation from an imagery provider is
illustrating terrain masking rather than demonstrating it. That defect shipped
once already (next-steps E-9). These tests pin the separation in place:

  * imagery comes from Cesium ion asset 3954 (Copernicus Sentinel-2), terrain
    from the env's own heightmap, and no ion terrain provider is constructed;
  * nothing under naigos/env or naigos/rl knows imagery exists, so no satellite
    pixel can reach an observation;
  * the ion token is read from the environment and is not in the repository;
  * the Sentinel-2 attribution required by Cesium ion's Content Usage guide is
    rendered on screen.

They are static and structural on purpose -- they must pass with no token, no
network and no browser, which is exactly when this rule is easiest to break.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from naigos.demo import imagery

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()


# --- token handling ---------------------------------------------------------------------


def test_token_comes_from_the_environment():
    env = {"NAIGOS_CESIUM_ION_TOKEN": "from-naigos-var"}
    assert imagery.resolve_ion_token(None, env) == "from-naigos-var"
    assert imagery.resolve_ion_token(None, {"CESIUM_ION_TOKEN": "generic"}) == "generic"


def test_the_project_specific_variable_wins_over_the_generic_one():
    env = {"NAIGOS_CESIUM_ION_TOKEN": "ours", "CESIUM_ION_TOKEN": "someone-elses"}
    assert imagery.resolve_ion_token(None, env) == "ours"


def test_an_explicit_flag_overrides_the_environment():
    assert imagery.resolve_ion_token("cli", {"NAIGOS_CESIUM_ION_TOKEN": "env"}) == "cli"


def test_a_blank_token_is_treated_as_absent():
    """An empty `export CESIUM_ION_TOKEN=` would otherwise be sent to ion and
    rejected, which surfaces as a blank globe rather than as a missing token."""
    assert imagery.resolve_ion_token(None, {"CESIUM_ION_TOKEN": "   "}) is None
    assert imagery.resolve_ion_token("  ", {}) is None
    assert imagery.resolve_ion_token(None, {}) is None


def test_no_ion_token_is_committed_to_the_repository():
    """ion tokens are JWTs, so they start `eyJ` and are unmistakable. Scanning
    tracked files means a token pasted into the page template fails here rather
    than reaching a public repo.

    Vendored dependencies are excluded: CesiumJS embeds its own default demo
    token in its distributed bundle, which is Cesium's to publish and not a leak
    of ours. Everything we author is in scope.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    jwt = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
    offenders = []
    for rel in tracked:
        if rel.startswith(("node_modules/", "data_cache/")):
            continue
        path = REPO / rel
        if not path.is_file() or path.suffix in (".png", ".jpg", ".pkl", ".tif", ".npz"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if jwt.search(text):
            offenders.append(rel)
    assert not offenders, f"JWT-shaped secret committed in: {offenders}"


def test_the_page_ships_with_no_token_baked_in():
    assert "/*__ION_TOKEN__*/null" in PAGE, "the server's substitution point is gone"
    assert "/*__IMAGERY__*/" in PAGE


# --- the imagery layer ------------------------------------------------------------------


def test_sentinel2_is_the_shipped_skin_when_a_token_exists():
    cfg = imagery.imagery_config("a-token")
    assert cfg["mode"] == "sentinel2"
    assert cfg["ion_asset"] == 3954
    assert "Copernicus Sentinel" in cfg["attribution"]


def test_without_a_token_it_degrades_to_a_keyless_provider_rather_than_failing():
    cfg = imagery.imagery_config(None)
    assert cfg["mode"] == "osm"
    assert cfg["ion_asset"] is None
    assert cfg["osm_url"].startswith("https://")


def test_osm_can_be_forced_even_with_a_token():
    assert imagery.imagery_config("a-token", prefer="osm")["mode"] == "osm"


def test_an_unknown_imagery_mode_is_refused():
    with pytest.raises(ValueError):
        imagery.imagery_config("a-token", prefer="bing")


def test_the_default_base_imagery_is_not_bing():
    """Cesium's default base layer is Bing Aerial: third-party commercial data,
    metered by session. The shipped demo must not reach for it."""
    assert "createWorldImagery" not in PAGE
    assert "BingMapsImageryProvider" not in PAGE
    blob = json.dumps(imagery.imagery_config("a-token")).lower()
    assert "bing" not in blob


def test_the_page_builds_sentinel2_through_the_supported_api():
    """`new IonImageryProvider(...)` was removed in CesiumJS 1.107 and yields a
    provider that never resolves; `fromAssetId` is the current form."""
    assert "Cesium.IonImageryProvider.fromAssetId(IMAGERY.ion_asset)" in PAGE
    assert "new Cesium.IonImageryProvider(" not in PAGE


# --- the separation ---------------------------------------------------------------------


def test_terrain_never_comes_from_an_imagery_provider():
    """THE test. The globe's surface must stay the simulation's own heightmap."""
    assert "CustomHeightmapTerrainProvider" in PAGE
    for ion_terrain in ("createWorldTerrain", "CesiumTerrainProvider", "fromIonAssetId"):
        assert ion_terrain not in PAGE, f"{ion_terrain} would replace the modelled surface"


def test_the_two_layers_enter_the_viewer_through_different_options():
    opts = PAGE[PAGE.index("const opts = {"):PAGE.index("const viewer = new Cesium.Viewer")]
    assert "terrainProvider," in opts and "baseLayer," in opts


@pytest.mark.parametrize("package", ["env", "rl"])
def test_no_satellite_pixel_can_reach_the_policy(package):
    """The observation is built from state and the heightmap. If the simulation
    packages cannot even name imagery, they cannot be reading it."""
    # Whole words that can only mean the viewer's skin. Plain "sentinel" is a
    # sentinel index in the spatial hash and "Cesium" appears in a comment about
    # what the georef is FOR, neither of which is a data path.
    forbidden = re.compile(
        r"\bimagery\b|\bsentinel[-_ ]?2\b|\bbasemap\b|\bion_token\b|"
        r"\bIonImageryProvider\b|\bsatellite\b|\borthophoto\b",
        re.IGNORECASE,
    )
    hits = []
    for path in sorted((REPO / "naigos" / package).rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert not hits, f"simulation code references imagery: {hits}"

    # And the import graph agrees: no simulation module reaches the demo package.
    imports = []
    for path in sorted((REPO / "naigos" / package).rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"^\s*(from|import)\s+.*\bdemo\b", line):
                imports.append(f"{path.relative_to(REPO)}:{n}")
    assert not imports, f"simulation imports the demo package: {imports}"


def test_the_observation_is_unchanged_by_imagery():
    """A direct check to go with the structural one: obs width follows the env
    config, and nothing in it is keyed to a viewer setting."""
    import jax

    from naigos.env.config import EnvConfig
    from naigos.env.flight_env import NaigosEnv

    cfg = EnvConfig(n_blue=2, n_threat=6, n_threat_active=4, max_steps=20)
    _, obs = NaigosEnv(cfg).reset(jax.random.PRNGKey(0))
    assert obs.ego.shape[0] == cfg.n_blue
    assert obs.threats.shape[:2] == (cfg.n_blue, cfg.n_threat)
    assert obs.threats.shape[-1] < 64, "an observation this wide could be hiding a raster"


# --- attribution ------------------------------------------------------------------------


def test_the_sentinel2_attribution_is_rendered_on_screen():
    """Cesium ion's Content Usage guide requires the provider attribution to be
    visible. The credit display carries the authoritative version; the HUD line
    means a screenshot carries it too."""
    assert 'id="imagerynote"' in PAGE
    assert "imagerynote" in PAGE.split("const viewer = new Cesium.Viewer")[1]
    assert "IMAGERY.separation_note" in PAGE
    text = imagery.SENTINEL2_ATTRIBUTION
    assert "Copernicus Sentinel" in text and "Cesium ion" in text
    assert "modified Copernicus Sentinel data" in text


def test_the_credit_display_is_not_suppressed():
    """Hiding Cesium's credit container would break the attribution requirement
    the asset is served under."""
    assert "creditContainer" not in PAGE
    assert "cesium-widget-credits" not in PAGE


def test_the_source_is_allowlisted_with_a_licence_and_attribution_flag():
    from naigos.research.allowlist import ALLOWLIST

    src = ALLOWLIST["copernicus_sentinel2"]
    assert src.attribution_required
    assert "Copernicus" in src.license and src.citation
    # Nothing server-side fetches it; the browser streams the tiles.
    assert src.hosts == ()


def test_the_component_records_the_separation():
    doc = json.loads((REPO / "components" / "demo.imagery.json").read_text())
    assert doc["sources"][0]["key"] == "copernicus_sentinel2"
    invariants = " ".join(doc["invariants"]).lower()
    assert "no satellite pixel enters an observation" in invariants
    assert "read from the environment" in invariants
