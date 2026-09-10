"""The browser contract for visual modes: what the viewer is allowed to draw.

`tests/test_visual_modes.py` pins the server-side half -- which mode is resolved
from which credentials, and that no credential survives into the config object.
This file pins the half that runs in the browser, because that is where the
claim is actually made: the picture is what a reader believes.

The rule the whole file exists to enforce is one sentence. *Exactly one surface
is drawn at a time, and the page says which.* Photorealistic mode brings the
provider's own geometry -- ground, buildings, trees -- none of which the
simulation has ever seen, so while it is on:

  * the globe carrying the modelled surface is switched OFF, so there are never
    two surfaces competing to be the truth (and, incidentally, never two to
    z-fight);
  * a banner says so continuously, and is hidden by exactly one thing, leaving
    the mode;
  * overlays are not depth-tested against the provider's mesh, since an
    occlusion the simulation never computed must not be drawn as though it had;
  * physics terrain is one click away, and is where a provider failure lands by
    itself.

Static and structural, like `tests/test_imagery_layers.py`: they must pass with
no credential, no network and no browser, which is exactly the condition under
which a viewer quietly starts claiming more than it measured.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from naigos.demo import imagery, live

REPO = Path(__file__).resolve().parents[1]
PAGE = (REPO / "naigos" / "demo" / "assets" / "cesium.html").read_text()
SERVER = (REPO / "naigos" / "demo" / "live.py").read_text()

#: The mode machinery, isolated from the rest of the page. Several tests below
#: are about what this block does *not* contain, which only means anything if
#: the block is really the whole of it.
BLOCK = PAGE[PAGE.index("// ------------------------------------------------------- visual modes"):
             PAGE.index("// --------------------------------------------------------------- stream")]

PHOTO_PATH = BLOCK[BLOCK.index("async function showPhotorealisticContext()"):
                   BLOCK.index("photoBtn.onclick =")]
PHYSICS_PATH = BLOCK[BLOCK.index("function showPhysicsTerrain("):
                     BLOCK.index("async function showPhotorealisticContext()")]

#: The block with its comments stripped. The identifier scans below are about
#: what the code does, and a comment naming `craft` to promise it is left alone
#: must not read as touching it.
CODE = re.sub(r"//.*", "", BLOCK)


# --- physics is the default and the destination -----------------------------------------


def test_the_page_boots_into_physics_terrain():
    """Whatever the server resolved, the first surface drawn is the modelled one.

    A page that booted straight into the provider's geometry would put the
    non-evidence picture on screen before the banner explaining it, which is the
    one ordering that cannot be allowed."""
    assert 'let visualMode = "physics";' in PAGE
    assert BLOCK.index("showPhysicsTerrain();") < BLOCK.index("if (PHOTO) {")


def test_the_runtime_toggle_returns_to_physics_terrain():
    assert "photoBtn.onclick" in BLOCK
    toggle = BLOCK[BLOCK.index("photoBtn.onclick ="):BLOCK.index("showPhysicsTerrain();\n")]
    assert 'visualMode === "photorealistic"' in toggle
    assert "showPhysicsTerrain(" in toggle
    # And the button says where it goes, rather than what it is.
    assert 'photoBtn.textContent = "physics terrain";' in PHOTO_PATH


def test_the_toggle_is_disabled_rather_than_dead_when_no_route_was_resolved():
    """Availability is a server-side question -- it needs the credentials. A
    button that silently did nothing would read as a broken viewer."""
    assert "photoBtn.disabled = true;" in BLOCK
    assert "VISUAL.fallback_reason" in BLOCK


# --- exactly one surface ----------------------------------------------------------------


def test_only_one_surface_is_ever_drawn():
    """THE test. Two surfaces on screen is two competing claims about the same
    ground, and the prettier one is the one that is not evidence."""
    assert "viewer.scene.globe.show = false;" in PHOTO_PATH
    assert "viewer.scene.globe.show = true;" in PHYSICS_PATH
    # ...and nowhere else, so no other code path can put both up at once.
    assert PAGE.count("viewer.scene.globe.show =") == 2


def test_the_photorealistic_tileset_never_becomes_the_terrain_provider():
    """The globe's surface stays the simulation's heightmap in the mode that
    claims to be showing it. Restated from tests/test_imagery_layers.py because
    a 3D tileset is a new way to reintroduce next-steps E-9."""
    assert "CustomHeightmapTerrainProvider" in PAGE
    for ion_terrain in ("createWorldTerrain", "CesiumTerrainProvider", "fromIonAssetId"):
        assert ion_terrain not in PAGE, f"{ion_terrain} would replace the modelled surface"
    # The terrain provider is built unconditionally, outside the mode machinery,
    # so /terrain is available the instant the toggle comes back.
    assert "const terrainProvider = new Cesium.CustomHeightmapTerrainProvider(" in PAGE
    assert "CustomHeightmapTerrainProvider" not in BLOCK


def test_the_tileset_is_built_through_the_supported_api():
    assert "Cesium.createGooglePhotorealistic3DTileset(" in PHOTO_PATH
    # The factory is the supported entry point; hand-rolling the tileset would
    # skip the credit plumbing the provider's terms depend on.
    assert "new Cesium.Cesium3DTileset(" not in PAGE
    assert "onlyUsingWithGoogleGeocoder: true" in PHOTO_PATH
    # ...which is honest only because the viewer ships no geocoder at all.
    assert "geocoder: false" in PAGE


# --- the disclaimer ---------------------------------------------------------------------


def test_the_visual_only_banner_exists_and_ships_hidden():
    assert 'id="visualonly"' in PAGE
    markup = PAGE[PAGE.index('<div id="visualonly"'):PAGE.index('<div id="status">')]
    assert "hidden" in markup.split(">")[0], "the banner must not be visible in physics mode"
    for slot in ("vo-title", "vo-body", "vo-credit"):
        assert f'id="{slot}"' in markup


def test_the_banner_is_shown_by_the_mode_and_hidden_by_exactly_one_thing():
    assert "banner.hidden = false;" in PHOTO_PATH
    assert "banner.hidden = true;" in PHYSICS_PATH
    assert PAGE.count("banner.hidden =") == 2, "the banner has exactly one way in and one out"


def test_the_banner_carries_the_configs_own_evidence_warning():
    """The wording is the server's, so it cannot drift from what /scene reports."""
    assert "VISUAL.separation_note" in BLOCK
    cfg = imagery.resolve_visual_config("photorealistic", ion_token="a-token")
    assert cfg.separation_note == imagery.PHOTOREALISTIC_EVIDENCE_WARNING
    assert not cfg.evidence_grade
    warning = cfg.separation_note.lower()
    assert "provider" in warning and "not the simulation" in warning


def test_the_terrain_note_stops_claiming_the_modelled_surface_while_the_mode_is_on():
    """The HUD line saying the globe *is* the simulation's heightmap is false in
    photorealistic mode. It is replaced there, and restored verbatim after."""
    assert "const PHYSICS_TERRAIN_NOTE = terrainNote.textContent;" in BLOCK
    assert "terrainNote.textContent = VISUAL_ONLY_BODY;" in PHOTO_PATH
    assert "terrainNote.textContent = PHYSICS_TERRAIN_NOTE;" in PHYSICS_PATH


# --- attribution ------------------------------------------------------------------------


def test_the_providers_credits_are_preserved_in_both_modes():
    # Cesium's credit display is where the authoritative per-tile data
    # attributions land; suppressing it would break the terms the tiles are
    # served under.
    assert "creditContainer" not in PAGE
    assert "cesium-widget-credits" not in PAGE
    # showCreditsOnScreen puts Google's per-tile credits in that display rather
    # than behind a "Data attribution" popup.
    assert "showCreditsOnScreen: true," in PHOTO_PATH
    # And the banner restates them, so a screenshot carries them too.
    assert 'document.getElementById("vo-credit").textContent = VISUAL.attribution' in BLOCK


def test_both_providers_are_credited_when_both_are_on_screen():
    """The tileset does not cover the whole globe; the base layer shows through
    at the edges, so both providers are visible and both are named."""
    cfg = imagery.resolve_visual_config("photorealistic", ion_token="a-token")
    assert "Google" in cfg.attribution
    assert cfg.base_attribution in cfg.attribution
    # The base-layer HUD line credits the base layer only, so it always matches
    # the pixels actually under it.
    assert cfg.to_page()["attribution"] == cfg.base_attribution


# --- failure lands on physics -----------------------------------------------------------


def test_a_provider_failure_returns_to_physics_terrain():
    """A rejected key, an offline provider or a quota refusal must land on the
    mode whose surface is the modelled one -- not on a blank globe."""
    assert "try {" in PHOTO_PATH and "} catch (e) {" in PHOTO_PATH
    catch = PHOTO_PATH[PHOTO_PATH.index("} catch (e) {"):]
    assert "showPhysicsTerrain(" in catch
    assert "tileset = null;" in catch


def test_a_provider_that_fails_mid_session_also_returns_to_physics_terrain():
    """Creating the tileset succeeding says nothing about the next thousand tile
    requests. A half-drawn backdrop under a banner implying there is something to
    look at is worse than the modelled surface."""
    assert "tileset.tileFailed.addEventListener(" in PHOTO_PATH
    failed = PHOTO_PATH[PHOTO_PATH.index("tileset.tileFailed.addEventListener("):]
    assert "PHOTO_LIMITS.tileFailureBudget" in failed
    assert "showPhysicsTerrain(" in failed


def test_returning_to_physics_needs_nothing_from_the_network():
    """/terrain is served identically in both modes, and the terrain provider is
    never torn down -- so the fallback cannot itself fail."""
    assert "fetch(" not in BLOCK
    assert "removeAll" not in BLOCK
    assert "primitives.remove" not in BLOCK


# --- LOD and request budget -------------------------------------------------------------


def test_lod_and_request_limits_are_set_rather_than_inherited():
    """CesiumJS's defaults assume the tileset owns the page: a 1.5 GB cache with
    1 GB of overflow, plus collision geometry. Here it is a backdrop on a page
    that is also holding an SSE stream open and stepping a simulation."""
    limits = PAGE[PAGE.index("const PHOTO_LIMITS = {"):PAGE.index("async function boot()")]
    values = {
        k: int(eval(v, {"__builtins__": {}}))  # noqa: S307 -- our own literal arithmetic
        for k, v in re.findall(r"^\s*(\w+): ([\d *]+),$", limits, re.M)
    }
    assert values["maximumScreenSpaceError"] >= 16, "a lower error refines further, not less"
    assert values["cacheBytes"] < 1536 * 1024 * 1024, "CesiumJS's default cache, uncapped"
    assert values["maximumCacheOverflowBytes"] < 1024 * 1024 * 1024
    assert 0 < values["maximumRequestsPerServer"] <= 18
    assert values["tileFailureBudget"] > 0

    for key in ("maximumScreenSpaceError", "cacheBytes", "maximumCacheOverflowBytes"):
        assert f"{key}: PHOTO_LIMITS.{key}," in PHOTO_PATH
    assert "Cesium.RequestScheduler.maximumRequestsPerServer = PHOTO_LIMITS" in PHOTO_PATH
    # Nothing here picks against the mesh or drives on it.
    assert "enableCollision: false," in PHOTO_PATH


def test_the_request_budget_is_handed_back_on_the_way_out():
    """maximumRequestsPerServer is global. Leaving it throttled would slow the
    imagery of the mode that is actually evidence."""
    assert "const DEFAULT_REQUESTS_PER_SERVER = Cesium.RequestScheduler.maximumRequestsPerServer;" in BLOCK
    assert "Cesium.RequestScheduler.maximumRequestsPerServer = DEFAULT_REQUESTS_PER_SERVER;" in PHYSICS_PATH


# --- the overlays keep working ----------------------------------------------------------


@pytest.mark.parametrize(
    "overlay",
    ["craft", "trailPts", "trails", "losLines", "losPts", "dropLines",
     "threatEntities", "lethalDomes", "detectRings", "objMarks", "renderCounters"],
)
def test_the_mode_switch_does_not_touch_the_overlays(overlay):
    """Aircraft, trails, threat envelopes, LOS rays and the counters are entities
    in world space. They are drawn identically in both modes; only the surface
    under them changes, and the switch has no business reaching into them."""
    assert re.search(rf"\b{overlay}\b", CODE) is None, f"the mode switch touches {overlay}"


def test_the_overlays_are_not_depth_tested_against_the_providers_mesh():
    """In physics mode an aircraft that vanishes behind a ridge vanished in the
    model too -- that occlusion is the demo. Against Google's buildings it would
    be an occlusion the simulation never computed, drawn as though it had."""
    assert "const applyDepthTest = () => {" in PAGE
    formula = PAGE[PAGE.index("const applyDepthTest = () => {"):PAGE.index("const xrayBtn")]
    assert 'visualMode === "photorealistic"' in formula
    assert "Number.POSITIVE_INFINITY" in formula
    # The only place the depth-test distance is set, so x-ray cannot get around it.
    assert PAGE.count("minimumDisableDepthTestDistance =") == 2  # the default, and this


def test_vertical_exaggeration_is_reset_so_entities_and_the_mesh_agree():
    """Entity altitudes go through exagZ(); the tileset is unexaggerated. A
    relief multiplier would float every aircraft above a mesh that never moved --
    the same class of bug as exaggerating the globe but not the aircraft."""
    assert "if (EXAG_K !== 1) { exagIdx = 0; applyExag(); }" in PHOTO_PATH
    # And the globe-only controls are disabled rather than left as dead buttons.
    assert "globeOnlyButtons.forEach(b => (b.disabled = true));" in PHOTO_PATH
    assert "globeOnlyButtons.forEach(b => (b.disabled = false));" in PHYSICS_PATH


# --- credentials ------------------------------------------------------------------------


def test_the_google_key_travels_by_its_own_substitution_point():
    """Same rule as the ion token: one credential, one path, greppable."""
    assert "/*__GOOGLE_API_KEY__*/null" in PAGE
    assert SERVER.count("/*__GOOGLE_API_KEY__*/null") == 1, "more than one path into the page"
    # What is substituted is the ROUTE-NARROWED value, not the raw key: on the
    # ion route, and in physics mode, the page has no use for it and gets null.
    # (This assertion used to name `google_api_key` and had gone stale against
    # that narrowing, so it was passing on nothing.)
    assert '"/*__GOOGLE_API_KEY__*/null", json.dumps(page_google_key)' in SERVER
    assert 'if visual.tileset_route == "google_maps_api" else None' in SERVER
    for mode, route in (("physics", None), ("photorealistic", "cesium_ion")):
        cfg = imagery.resolve_visual_config(mode, ion_token="a-token")
        assert cfg.tileset_route == route
        html = live.render_page(cfg, "a-token", google_api_key="AIza-a-real-looking-key")
        assert "AIza-a-real-looking-key" not in html, f"{mode} leaked the Google key"
    # It is used on the direct route only -- with no key CesiumJS reaches the
    # same tileset through ion, which the ion token already covers.
    assert 'VISUAL.tileset_route === "google_maps_api"' in PHOTO_PATH
    assert "Cesium.GoogleMaps.defaultApiKey" not in PAGE, "a global default leaks into ion runs"


def test_the_served_page_carries_each_credential_exactly_once():
    key = "AIzaSy-not-a-real-key-0123456789abcdef"
    token = "not-a-real-ion-token"
    cfg = imagery.resolve_visual_config("photorealistic", ion_token=token)
    html = (PAGE
            .replace("/*__ION_TOKEN__*/null", json.dumps(token))
            .replace("/*__GOOGLE_API_KEY__*/null", json.dumps(key))
            .replace('/*__VISUAL__*/{mode: "physics", evidence_grade: true}',
                     json.dumps(cfg.as_dict())))
    assert html.count(token) == 1
    assert html.count(key) == 1
    # Neither ever reaches the config blob, which is what /scene serves.
    assert token not in json.dumps(cfg.as_dict())
    assert key not in json.dumps(cfg.as_dict())


def test_no_google_api_key_is_committed_to_the_repository():
    """Google Maps keys are `AIza` plus 35 characters, so they are as
    recognisable as a JWT. Same scan, same reason: a key pasted into the page
    template fails here rather than reaching a public repo."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    pattern = re.compile(r"AIza[0-9A-Za-z_-]{35}")
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
        if pattern.search(text):
            offenders.append(rel)
    assert not offenders, f"Google-API-key-shaped secret committed in: {offenders}"


# --- the mode cannot reach the simulation -----------------------------------------------


def test_the_page_still_loads_exactly_one_external_script():
    """The tileset is streamed by the CesiumJS already on the page. A second
    <script> would be a second thing to trust with the same origin."""
    srcs = re.findall(r'<script[^>]*src="([^"]+)"', PAGE)
    assert srcs == ["https://cesium.com/downloads/cesiumjs/releases/1.145/Build/Cesium/Cesium.js"]


@pytest.mark.parametrize("package", ["env", "rl"])
def test_no_provider_geometry_can_reach_the_policy(package):
    """The detection model consumes the DEM. If the simulation packages cannot
    name a tileset, they cannot be sampling one."""
    forbidden = re.compile(
        r"\bphotorealistic\b|\b3d[-_ ]?tiles\b|\btileset\b|\bcesium3dtileset\b|"
        r"\bgoogle[-_ ]?maps\b|\bgoogle_api_key\b",
        re.IGNORECASE,
    )
    hits = []
    for path in sorted((REPO / "naigos" / package).rglob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert not hits, f"simulation code references a visual provider: {hits}"


# --- what the mode costs for the rest of the session -------------------------------------


def test_the_page_says_sentinel2_is_gone_for_the_session_rather_than_missing():
    """`resolve_visual_config("photorealistic", ...)` forces `base_imagery="osm"`
    -- correct, since Sentinel-2 tiles under an opaque tileset would be metered
    against the ion quota and never seen. But the runtime toggle brings the
    *surface* back while the base layer stays OSM, so a session started with
    `--visual photorealistic` and a valid ion token can never show Sentinel-2.

    That is a design consequence, not a bug. Unexplained, it reads as a broken
    imagery layer: the operator has a working token, the HUD said Sentinel-2 was
    available, and the skin is OSM anyway. The page says so on the control they
    would reach for, and only when a token is actually present -- without one,
    OSM is what physics mode would have drawn too, so there is nothing lost to
    report."""
    assert "VISUAL.ion_token_present" in BLOCK
    note = BLOCK[BLOCK.index("VISUAL.ion_token_present") - 400:]
    assert "imgBtn.title" in note
    lowered = note[:900].lower()
    assert "sentinel-2" in lowered
    assert "restart" in lowered
    # It is the mode that costs it, so the mode is what the note names.
    assert "photorealistic" in lowered


def test_the_session_note_does_not_promise_a_layer_the_page_could_switch_to():
    """The alternative fix -- build both layers and toggle visibility -- costs an
    ion request per tile in the mode designed to avoid exactly that. The page
    tells the truth about the restart instead, so nothing here may quietly start
    constructing the Sentinel-2 provider inside the mode block."""
    assert "IonImageryProvider" not in BLOCK
    assert "baseLayer" not in CODE, "the mode switch must not rebuild the skin"
