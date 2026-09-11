"""Visual providers for the globe viewer -- the demo's SKIN, and nothing else.

The viewer draws two things that are deliberately kept apart:

    imagery   this module.  Optical pixels draped over the globe.
              Cosmetic. Read by human eyes only.
    terrain   ``naigos.demo.live.build_terrain_grid``.  The simulation's own
              heightmap, the surface every line-of-sight ray was computed
              against, and the only elevation data the detection model consumes.

Keeping them separate is a correctness property, not a stylistic one. If the
globe's relief came from a visual provider's own geometry (Cesium World Terrain,
Google's 3D Tiles) the picture would look better and would no longer be
evidence: what occludes on screen would not be what occluded in the model. That
defect had already been shipped once and removed (next-steps E-9), so the split
is asserted by ``tests/test_imagery_layers.py`` rather than left to discipline.

**No satellite pixel ever enters an observation.** ``naigos/env/obs.py`` builds
its vectors from state and the heightmap; this module is imported by the demo
server only, and the test suite asserts that nothing under ``naigos/env`` or
``naigos/rl`` references imagery at all.

Visual modes
------------
One knob, ``--visual``, chooses between three whole postures. They differ in what
the drawn surface *is*, which is why this is a mode and not another imagery
option:

``physics`` (the default, and the only evidence-grade one)
    Surface: the simulation's own DEM, served from ``/terrain``.
    Skin:    Copernicus Sentinel-2 via Cesium ion, or keyless OpenStreetMap.
    What occludes on screen is what occluded in the model.

``photorealistic``
    Surface: Google Photorealistic 3D Tiles, streamed through CesiumJS from
             Cesium ion (asset 2275207) or from Google's own endpoint with a
             Maps API key.
    Skin:    the tileset carries its own texture; OSM stays underneath it.
    The tileset brings its own geometry -- buildings, trees, provider relief --
    so the drawn surface is NO LONGER the modelled one. Nothing in this mode
    demonstrates terrain masking; it is a presentation mode, and
    ``VisualConfig.evidence_grade`` is ``False`` to say so in one field rather
    than in a paragraph nobody reads.

``urban-presentation``
    A dense 3D city for context, from one of two geometry sources:

    local      Extruded OpenStreetMap building footprints and major-road
               centrelines from the local visual cache (``naigos.demo.urban``),
               standing on the simulation's own DEM. The layer Naigos owns and
               styles itself.
    provider   Google Photorealistic 3D Tiles, exactly as ``photorealistic``
               draws them. The page reports ``provider buildings active`` only
               once the tileset has put a tile with content on screen -- never
               because a token exists. The local layer, when cached, stays
               loaded as the runtime fallback.

    Which one is chosen by ``urban_geometry`` (``--urban-geometry``), never by
    which credentials happen to be in the shell:

    ``local`` (the default)   the local cache. A credential does not change it.
    ``provider``              the provider tiles: an explicit opt-in. Without a
                              credential it degrades to the local cache and says
                              why in ``fallback_reason``.
    ``auto``                  provider when a credential exists, else local --
                              the mode's original behaviour, kept for anyone who
                              wants the provider whenever it is reachable.

    Presentation only either way, and ``evidence_grade`` is ``False``: building
    geometry is not used by terrain LOS, by detection, or by any simulation
    result. The page defaults render-only building occlusion OFF so no building
    can hide an aircraft on screen and read as a blocked radar line. When the
    chosen source has nothing to draw the mode starts in a labelled
    ``urban data unavailable`` state, with the command that fixes it, instead
    of pretending buildings are there. ``docs/urban-geometry-routing.md`` has
    the whole routing table.

All modes fall back to keyless OpenStreetMap rather than failing, and
``photorealistic`` falls back to the whole ``physics`` mode when its credentials
are absent: a demo that opens is worth more than a demo that is right about why
it did not.

Why Sentinel-2 rather than Cesium's default
-------------------------------------------
Cesium ion's default base imagery is Bing Aerial: third-party commercial data,
metered per session, and its terms are Microsoft's rather than Cesium's. This is
a portfolio project, so the shipped demo uses Cesium ion asset 3954 -- Copernicus
Sentinel-2 -- which is ESA open data: free to *use*, not merely free to look at.
That removes the licensing question from the demo entirely.

Credentials
-----------
Read from explicit, named environment variables and nowhere else:

    NAIGOS_CESIUM_ION_TOKEN, CESIUM_ION_TOKEN            Cesium ion
    NAIGOS_GOOGLE_MAPS_API_KEY, GOOGLE_MAPS_API_KEY      Google Maps Tiles API

There is no config file, no dotenv load, no credential discovery. Nothing
token-shaped is written to disk, and :class:`VisualConfig` stores only *whether*
a credential was found, never its value -- so a config object can be logged,
serialised into ``/scene`` or pasted into an issue without leaking anything.
``--ion-token`` remains as a one-off override for a single run; there is
deliberately no CLI flag for the Google key, because a key on argv is a key in
the shell history.

Without any credential the viewer falls back to OpenStreetMap, which needs no
key -- the terrain, and therefore every claim the viewer makes, is identical
either way.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

#: Cesium ion asset id for Copernicus Sentinel-2 imagery.
SENTINEL2_ION_ASSET = 3954

#: Cesium ion asset id for Google Photorealistic 3D Tiles.
GOOGLE_3D_TILES_ION_ASSET = 2275207

#: The visual postures. ``physics`` is the default and the only one whose drawn
#: surface is the surface the simulation computed against; the other two are
#: presentation modes and say so in ``evidence_grade``.
VISUAL_MODES = ("physics", "photorealistic", "urban-presentation")
DEFAULT_VISUAL_MODE = "physics"
URBAN_MODE = "urban-presentation"

#: Where the building geometry on screen comes from. ``None`` in physics mode,
#: which draws no buildings at all.
GEOMETRY_PROVIDER = "provider_3d_tiles"
GEOMETRY_LOCAL = "local_osm_extrusions"

#: urban-presentation's geometry policy (``--urban-geometry``). ``local`` is the
#: default so a credential in the environment never silently swaps the Naigos
#: city layer for the provider's; ``provider`` is the opt-in; ``auto`` keeps the
#: provider-when-reachable behaviour. See the module docstring.
URBAN_GEOMETRY_LOCAL = "local"
URBAN_GEOMETRY_PROVIDER = "provider"
URBAN_GEOMETRY_AUTO = "auto"
URBAN_GEOMETRY_CHOICES = (URBAN_GEOMETRY_LOCAL, URBAN_GEOMETRY_PROVIDER, URBAN_GEOMETRY_AUTO)
DEFAULT_URBAN_GEOMETRY = URBAN_GEOMETRY_LOCAL

#: The one command that builds the local city layer, quoted wherever its
#: absence is reported.
URBAN_BUILD_HINT = "uv run python -m naigos.demo.urban --aoi <aoi>"

#: Provider readiness as far as the SERVER can know it. It can only ever say
#: whether a route exists; whether the tileset actually drew anything is a
#: browser fact, reported by the page (``NAIGOS_VIEW.provider()``) and never
#: inferred from a credential being present.
PROVIDER_NOT_REQUESTED = "not_requested"
PROVIDER_AWAITING_BROWSER = "awaiting_browser"
PROVIDER_UNAVAILABLE = "unavailable_no_credentials"

#: Said wherever buildings are drawn, in every mode that draws them.
BUILDING_LOS_NOTE = (
    "Building geometry is presentation only: it is not used by terrain LOS, "
    "detection, or any simulation result."
)

#: The local city layer's credit. The data is ODbL; see naigos.demo.urban.
OSM_BUILDINGS_ATTRIBUTION = (
    "Buildings and roads: (c) OpenStreetMap contributors, ODbL -- cached locally "
    "and extruded for presentation."
)

#: Base-imagery skins available in ``physics`` mode.
IMAGERY_MODES = ("sentinel2", "osm")

#: Checked in order; the first non-empty one wins.
TOKEN_ENV_VARS = ("NAIGOS_CESIUM_ION_TOKEN", "CESIUM_ION_TOKEN")

#: Same rule, for the Google Maps Tiles API key. No CLI flag: see the module
#: docstring.
GOOGLE_API_KEY_ENV_VARS = ("NAIGOS_GOOGLE_MAPS_API_KEY", "GOOGLE_MAPS_API_KEY")

#: Keyless fallback. No token, no account, no attribution beyond the ODbL line
#: Cesium's own credit display already emits for this provider.
OSM_TILE_URL = "https://tile.openstreetmap.org/"

#: Per Cesium ion's Content Usage guide, ion-hosted content must carry its
#: provider attribution on screen. CesiumJS's credit display does that
#: automatically from the asset's own credit, so the viewer leaves that display
#: visible on purpose and treats it as authoritative. This string is the static
#: restatement that goes in the HUD and in the docs, so the requirement is met
#: even at a glance at a screenshot.
SENTINEL2_ATTRIBUTION = (
    "Imagery: Copernicus Sentinel-2, served by Cesium ion (asset 3954). "
    "Contains modified Copernicus Sentinel data. Cesium's own credit display "
    "(bottom right) carries the authoritative per-provider attribution and is "
    "left visible deliberately."
)

OSM_ATTRIBUTION = "Imagery: OpenStreetMap contributors, ODbL."

#: Google's Photorealistic 3D Tiles policy requires the Google attribution and
#: the per-tile data credits to stay visible. CesiumJS surfaces the tileset's
#: own credits into the credit display, which this viewer never suppresses; this
#: is the static restatement that also survives a screenshot.
GOOGLE_3D_TILES_ATTRIBUTION = (
    "Imagery and 3D geometry: Google Photorealistic 3D Tiles (© Google). "
    "Streamed by CesiumJS; the per-tile data credits in Cesium's credit display "
    "(bottom right) are authoritative and are left visible deliberately."
)

#: Said in the HUD so a viewer cannot mistake the pretty layer for the modelled
#: one. This is the whole reason the module exists.
LAYER_SEPARATION_NOTE = (
    "Imagery is a skin. The relief under it -- and every line-of-sight test -- "
    "comes from the simulation's own DEM, not from this provider."
)

#: The counterpart for photorealistic mode, where the sentence above stops being
#: true. Stated at least as loudly, because this is the mode that can mislead.
PHOTOREALISTIC_EVIDENCE_WARNING = (
    "PRESENTATION MODE. Google's 3D Tiles bring their own geometry, so the "
    "surface drawn here is the provider's, not the simulation's DEM. Nothing on "
    "screen in this mode is evidence about terrain masking -- run --visual "
    "physics for that."
)

#: urban-presentation over the local cache: the surface IS the simulation's DEM,
#: but the buildings on it are not in the model, so a screenshot of this mode is
#: still not evidence.
URBAN_LOCAL_WARNING = (
    "PRESENTATION MODE. OpenStreetMap buildings and roads are extruded over the "
    "simulation's DEM for context. " + BUILDING_LOS_NOTE + " Run --visual physics "
    "for evidence about terrain masking."
)

#: urban-presentation when the selected geometry source has nothing to draw.
#: Worded for every such case (no local cache, or --urban-geometry provider with
#: no credential and no cache); the specific cause is ``fallback_reason``.
URBAN_UNAVAILABLE_WARNING = (
    "PRESENTATION MODE -- urban data unavailable. No building layer is drawn: the "
    "selected urban geometry source has nothing to draw. The surface is the "
    "simulation's DEM; run --visual physics for evidence."
)

#: Where the drawn surface comes from, per mode. Only the first is evidence.
TERRAIN_SOURCE_SIMULATION = "simulation_dem"
TERRAIN_SOURCE_PROVIDER = "provider_3d_tiles"


class VisualConfigError(ValueError):
    """A visual-mode request that cannot be honoured and must not be guessed at."""


@dataclass(frozen=True)
class VisualConfig:
    """The public visual-provider contract: one object, no credentials in it.

    Everything the server, the page and the operator need to know about what the
    globe is showing, and nothing they must not see. Credentials are represented
    as the booleans ``ion_token_present`` / ``google_api_key_present`` -- the
    values themselves stay in the caller's local variable and never enter this
    object, so it is safe to print, to serialise into ``/scene``, and to paste
    into a bug report. ``tests/test_visual_modes.py`` asserts that.

    Frozen because the mode is resolved once, at startup, and a viewer whose
    posture can change under it is a viewer whose screenshots cannot be trusted.
    """

    #: The mode actually in force after credentials were checked. In VISUAL_MODES.
    mode: str
    #: What the operator asked for. Differs from ``mode`` exactly when a
    #: fallback fired, which is what ``fallback_reason`` explains.
    requested_mode: str
    #: The base imagery layer: "sentinel2" or "osm". Present in both modes --
    #: photorealistic keeps a keyless layer underneath its tileset.
    base_imagery: str
    #: ion asset backing the base imagery layer, or None when keyless.
    ion_asset: int | None
    #: The 3D tileset draped over everything, or None in physics mode.
    tileset: str | None
    #: ion asset for that tileset, when it is reached through Cesium ion.
    tileset_ion_asset: int | None
    #: How the tileset is reached: "cesium_ion", "google_maps_api", or None.
    tileset_route: str | None
    #: TERRAIN_SOURCE_SIMULATION or TERRAIN_SOURCE_PROVIDER.
    terrain_source: str
    #: True only when the drawn surface is the surface the model used. This is
    #: the single field that says whether a screenshot means anything.
    evidence_grade: bool
    #: Attribution for the base imagery layer.
    base_attribution: str
    #: Attribution the tileset provider requires, or None.
    tileset_attribution: str | None
    #: Why the resolved mode is not the requested one, or None.
    fallback_reason: str | None
    ion_token_present: bool
    google_api_key_present: bool
    #: True in every mode but physics. The page shows the presentation banner
    #: exactly when this is set.
    presentation_only: bool = False
    #: GEOMETRY_PROVIDER, GEOMETRY_LOCAL, or None (no buildings drawn). The
    #: geometry the page will try first; it may still fall back at runtime.
    geometry_source: str | None = None
    #: What the mode would have used had everything been available.
    requested_geometry_source: str | None = None
    #: PROVIDER_* above. Never "active": only the browser can observe that.
    provider_state: str = PROVIDER_NOT_REQUESTED
    #: urban-presentation only: "available", "unavailable" or "not_requested".
    local_urban_state: str = "not_requested"
    #: Whether buildings may hide simulation entities on screen at start. Off,
    #: always: a building covering an aircraft reads as a blocked radar line.
    building_occlusion_default: bool = False
    #: Credit for building geometry drawn from a local cache, or None.
    geometry_attribution: str | None = None
    #: urban-presentation only: the URBAN_GEOMETRY_* policy that was asked for
    #: ("local", "provider" or "auto"); None in every other mode.
    urban_geometry: str | None = None

    # --- derived, credential-free views ---------------------------------------

    @property
    def imagery(self) -> str:
        """What is on top: the tileset if there is one, else the base layer."""
        return "google_3d_tiles" if self.tileset else self.base_imagery

    @property
    def osm_url(self) -> str:
        return OSM_TILE_URL

    @property
    def osm_attribution(self) -> str:
        return OSM_ATTRIBUTION

    @property
    def attribution(self) -> str:
        """Every attribution this mode must show, in one string.

        Both, in photorealistic mode: the tileset does not cover the whole globe
        and the base layer shows through at the edges, so both providers are on
        screen and both are credited.
        """
        parts = [self.tileset_attribution, self.base_attribution, self.geometry_attribution]
        return " ".join(p for p in parts if p)

    @property
    def separation_note(self) -> str:
        """The layer-separation sentence, or the warning that replaces it."""
        if self.evidence_grade:
            return LAYER_SEPARATION_NOTE
        if self.mode != URBAN_MODE:
            return PHOTOREALISTIC_EVIDENCE_WARNING
        if self.geometry_source == GEOMETRY_PROVIDER:
            return f"{PHOTOREALISTIC_EVIDENCE_WARNING} {BUILDING_LOS_NOTE}"
        if self.geometry_source == GEOMETRY_LOCAL:
            return URBAN_LOCAL_WARNING
        return URBAN_UNAVAILABLE_WARNING

    @property
    def building_note(self) -> str | None:
        """The HUD's building disclaimer, in every mode that can draw buildings."""
        return BUILDING_LOS_NOTE if self.presentation_only else None

    def to_page(self) -> dict:
        """The base-imagery blob, substituted at ``__IMAGERY__``. Credential-free.

        Deliberately describes the BASE LAYER only -- never the tileset -- so the
        HUD credit line it feeds always names the provider of the pixels actually
        under the cursor. In photorealistic mode that is OpenStreetMap, showing
        through wherever the tileset has no coverage; crediting Google there would
        put Google's name over pixels Google did not supply, which is the opposite
        of satisfying an attribution requirement. The tileset's own attribution is
        the banner's job, and Cesium's credit display carries the authoritative
        per-provider version either way.

        The mode-level fields -- the tileset, its route, ``evidence_grade`` --
        travel in :meth:`as_dict` through the separate ``__VISUAL__`` point.

        The ion token is injected separately again, through its own substitution
        point, so the one string that must not be logged travels by exactly one
        path and can be grepped for on that basis.
        """
        return {
            "mode": self.base_imagery,
            "ion_asset": self.ion_asset,
            "attribution": self.base_attribution,
            "separation_note": LAYER_SEPARATION_NOTE,
            "osm_url": OSM_TILE_URL,
            "osm_attribution": OSM_ATTRIBUTION,
        }

    def as_dict(self) -> dict:
        """The whole contract, for the page, for ``/scene`` and for logs.

        Still credential-free: the booleans are the only trace of a token.
        """
        d = asdict(self)
        d.update(
            imagery=self.imagery,
            attribution=self.attribution,
            separation_note=self.separation_note,
            building_note=self.building_note,
            # the page's two urban banners, worded here so they cannot drift
            local_urban_note=URBAN_LOCAL_WARNING if self.mode == URBAN_MODE else None,
            urban_unavailable_note=URBAN_UNAVAILABLE_WARNING if self.mode == URBAN_MODE else None,
            osm_url=OSM_TILE_URL,
            osm_attribution=OSM_ATTRIBUTION,
        )
        return d


def resolve_ion_token(cli_token: str | None = None, env: dict[str, str] | None = None) -> str | None:
    """Return the ion token from ``--ion-token`` or the environment, else ``None``.

    An explicit flag wins so a one-off run can override a shell export. Blank and
    whitespace-only values are treated as absent: an empty ``CESIUM_ION_TOKEN=``
    left in a shell profile would otherwise be sent to ion and rejected, which
    surfaces as a blank globe rather than as a missing token.
    """
    return _resolve_credential(TOKEN_ENV_VARS, cli_token, env)


def resolve_google_api_key(env: dict[str, str] | None = None) -> str | None:
    """Return the Google Maps Tiles API key from the environment, else ``None``.

    Environment only, by design: there is no CLI flag, because a key passed on
    argv is a key in the shell history and in every ``ps`` listing on the box.
    """
    return _resolve_credential(GOOGLE_API_KEY_ENV_VARS, None, env)


def _resolve_credential(
    names: tuple[str, ...], cli_value: str | None, env: dict[str, str] | None
) -> str | None:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env = os.environ if env is None else env
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def resolve_visual_config(
    mode: str = DEFAULT_VISUAL_MODE,
    *,
    ion_token: str | None = None,
    google_api_key: str | None = None,
    imagery: str | None = None,
    local_urban: bool = False,
    urban_geometry: str | None = None,
) -> VisualConfig:
    """Resolve the requested visual mode against the credentials actually present.

    The two arguments that are secrets are consumed here and not retained: what
    comes back records only whether each was found.

    Unknown values raise rather than defaulting. A typo in ``--visual`` that
    silently selected ``physics`` would be indistinguishable from a working run,
    and the whole point of the mode is that the two are not interchangeable.

    Missing credentials do not raise: ``photorealistic`` degrades to the full
    ``physics`` mode and says why, because the fallback is strictly the safer
    posture -- it is the one whose surface is the modelled one. Within physics,
    a missing token degrades Sentinel-2 to keyless OpenStreetMap on the same
    reasoning.

    ``urban-presentation`` never degrades to another mode -- its camera and its
    disclaimers are the point of asking for it -- only between geometry sources,
    by the ``urban_geometry`` policy (None means DEFAULT_URBAN_GEOMETRY, i.e.
    ``local``): the local OSM cache when ``local_urban`` says one was built,
    provider tiles only on ``provider`` or ``auto`` with a credential, else a
    labelled no-buildings state. It never resolves into ``photorealistic``, and
    it is never evidence-grade. ``urban_geometry`` is ignored by the other modes
    (``validate_cli`` refuses it there on the command line).
    """
    if mode not in VISUAL_MODES:
        raise VisualConfigError(
            f"unknown visual mode {mode!r}; expected one of {', '.join(VISUAL_MODES)}"
        )
    if urban_geometry is None:
        urban_geometry = DEFAULT_URBAN_GEOMETRY
    if urban_geometry not in URBAN_GEOMETRY_CHOICES:
        raise VisualConfigError(
            f"unknown urban geometry {urban_geometry!r}; expected one of "
            f"{', '.join(URBAN_GEOMETRY_CHOICES)}"
        )
    if imagery is None:
        imagery = default_imagery_for(mode)
    if imagery not in IMAGERY_MODES:
        raise VisualConfigError(
            f"unknown imagery mode {imagery!r}; expected one of {', '.join(IMAGERY_MODES)}"
        )

    has_ion = bool(ion_token and ion_token.strip())
    has_google = bool(google_api_key and google_api_key.strip())

    fallback_reason = None
    if mode == URBAN_MODE:
        return _resolve_urban(has_ion, has_google, bool(local_urban), urban_geometry)
    if mode == "photorealistic":
        # ion first: the same token the rest of the demo already uses, and the
        # route that keeps every credential on one account.
        route = "cesium_ion" if has_ion else ("google_maps_api" if has_google else None)
        if route is not None:
            # The base layer under an opaque tileset stays keyless on purpose:
            # metering Sentinel-2 tiles nobody can see is pure cost.
            return VisualConfig(
                mode="photorealistic",
                requested_mode=mode,
                base_imagery="osm",
                ion_asset=None,
                tileset="google_photorealistic",
                tileset_ion_asset=(GOOGLE_3D_TILES_ION_ASSET if route == "cesium_ion" else None),
                tileset_route=route,
                terrain_source=TERRAIN_SOURCE_PROVIDER,
                evidence_grade=False,
                base_attribution=OSM_ATTRIBUTION,
                tileset_attribution=GOOGLE_3D_TILES_ATTRIBUTION,
                fallback_reason=None,
                ion_token_present=has_ion,
                google_api_key_present=has_google,
                presentation_only=True,
                geometry_source=GEOMETRY_PROVIDER,
                requested_geometry_source=GEOMETRY_PROVIDER,
                provider_state=PROVIDER_AWAITING_BROWSER,
            )
        fallback_reason = (
            "photorealistic needs a Cesium ion token (" + " or ".join(TOKEN_ENV_VARS) + ") "
            "or a Google Maps Tiles API key (" + " or ".join(GOOGLE_API_KEY_ENV_VARS) + "); "
            "neither is set, so the evidence-grade physics mode is used instead"
        )
        # The fallback is physics, and physics wants the best skin it can get.
        imagery = default_imagery_for("physics")

    # physics: the modelled surface, skinned with whatever the credentials allow.
    if imagery == "sentinel2" and has_ion:
        return VisualConfig(
            mode="physics",
            requested_mode=mode,
            base_imagery="sentinel2",
            ion_asset=SENTINEL2_ION_ASSET,
            tileset=None,
            tileset_ion_asset=None,
            tileset_route=None,
            terrain_source=TERRAIN_SOURCE_SIMULATION,
            evidence_grade=True,
            base_attribution=SENTINEL2_ATTRIBUTION,
            tileset_attribution=None,
            fallback_reason=fallback_reason,
            ion_token_present=has_ion,
            google_api_key_present=has_google,
        )
    return VisualConfig(
        mode="physics",
        requested_mode=mode,
        base_imagery="osm",
        ion_asset=None,
        tileset=None,
        tileset_ion_asset=None,
        tileset_route=None,
        terrain_source=TERRAIN_SOURCE_SIMULATION,
        evidence_grade=True,
        base_attribution=OSM_ATTRIBUTION,
        tileset_attribution=None,
        fallback_reason=fallback_reason,
        ion_token_present=has_ion,
        google_api_key_present=has_google,
    )


def _credential_hint() -> str:
    return ("a Cesium ion token (" + " or ".join(TOKEN_ENV_VARS) + ") or a Google Maps "
            "Tiles API key (" + " or ".join(GOOGLE_API_KEY_ENV_VARS) + ")")


def _resolve_urban(has_ion: bool, has_google: bool, local: bool,
                   policy: str = DEFAULT_URBAN_GEOMETRY) -> VisualConfig:
    """urban-presentation: route the building geometry by ``policy``.

    ============  ===================  ======================  ===================
    policy        credential + cache   credential, no cache    no credential
    ============  ===================  ======================  ===================
    local         local                unavailable             local / unavailable
    provider      provider             provider                local / unavailable
    auto          provider             provider                local / unavailable
    ============  ===================  ======================  ===================

    ("local / unavailable": local when the cache exists, else the labelled
    no-buildings state.) A credential alone never selects the provider: only
    ``provider`` or ``auto`` does.

    The base layer is keyless OSM on every path. Under the provider it is what
    shows through at the tileset's edges (and Sentinel-2 there would be metered
    and unseen); on the local path the city layer is the subject and the skin
    is toned down under it, so metering Sentinel-2 buys nothing.
    """
    route = "cesium_ion" if has_ion else ("google_maps_api" if has_google else None)
    common = dict(
        mode=URBAN_MODE, requested_mode=URBAN_MODE, base_imagery="osm", ion_asset=None,
        evidence_grade=False, base_attribution=OSM_ATTRIBUTION,
        ion_token_present=has_ion, google_api_key_present=has_google,
        presentation_only=True, building_occlusion_default=False,
        local_urban_state="available" if local else "unavailable",
        # The local layer rides along as the runtime fallback, so it is
        # credited whenever the page holds it.
        geometry_attribution=OSM_BUILDINGS_ATTRIBUTION if local else None,
        urban_geometry=policy,
    )
    no_tileset = dict(tileset=None, tileset_ion_asset=None, tileset_route=None,
                      terrain_source=TERRAIN_SOURCE_SIMULATION, tileset_attribution=None)
    build = f"Build it once with: {URBAN_BUILD_HINT}"

    if policy == URBAN_GEOMETRY_LOCAL:
        # The default. The provider is not asked for, so its state says exactly
        # that -- whether or not a credential happens to be in the environment.
        if local:
            reason = None
        else:
            reason = f"urban data unavailable: no local urban cache for this AOI. {build}"
            if route is not None:
                reason += (" -- or opt in to Google Photorealistic 3D Tiles with "
                           "--urban-geometry provider (a credential is set)")
        return VisualConfig(
            **common, **no_tileset,
            fallback_reason=reason,
            geometry_source=GEOMETRY_LOCAL if local else None,
            requested_geometry_source=GEOMETRY_LOCAL,
            provider_state=PROVIDER_NOT_REQUESTED,
        )

    if route is not None:
        return VisualConfig(
            **common,
            tileset="google_photorealistic",
            tileset_ion_asset=GOOGLE_3D_TILES_ION_ASSET if route == "cesium_ion" else None,
            tileset_route=route,
            terrain_source=TERRAIN_SOURCE_PROVIDER,
            tileset_attribution=GOOGLE_3D_TILES_ATTRIBUTION,
            fallback_reason=None,
            geometry_source=GEOMETRY_PROVIDER,
            requested_geometry_source=GEOMETRY_PROVIDER,
            provider_state=PROVIDER_AWAITING_BROWSER,
        )
    # provider or auto, and no credential: the local layer if there is one.
    # `provider` was an explicit request that could not be honoured, so it
    # stays the requested source and the reason says so; `auto` asked for
    # "provider if reachable", and local is simply what that resolves to.
    asked = ("--urban-geometry provider needs" if policy == URBAN_GEOMETRY_PROVIDER
             else "provider buildings need")
    if local:
        reason = (f"{asked} {_credential_hint()}; neither is set, so the "
                  "local cached OpenStreetMap building layer is drawn over the simulation DEM")
    else:
        reason = (f"urban data unavailable: {asked} {_credential_hint()}; neither is set, "
                  f"and there is no local urban cache. {build}")
    return VisualConfig(
        **common, **no_tileset,
        fallback_reason=reason,
        geometry_source=GEOMETRY_LOCAL if local else None,
        requested_geometry_source=(GEOMETRY_PROVIDER if policy == URBAN_GEOMETRY_PROVIDER
                                   else GEOMETRY_LOCAL),
        provider_state=PROVIDER_UNAVAILABLE,
    )


def validate_cli(mode: str, imagery: str | None, urban_geometry: str | None = None) -> None:
    """Reject an impossible ``--visual``/``--imagery``/``--urban-geometry`` set early.

    argparse's ``choices`` already covers a bad single value; this covers the
    combination, and gives a caller that builds the args itself the same check.
    ``imagery=None`` means "let the mode pick", which is always resolvable, and
    ``urban_geometry=None`` means DEFAULT_URBAN_GEOMETRY.
    """
    if mode not in VISUAL_MODES:
        raise VisualConfigError(
            f"unknown visual mode {mode!r}; expected one of {', '.join(VISUAL_MODES)}"
        )
    if imagery is not None and imagery not in IMAGERY_MODES:
        raise VisualConfigError(
            f"unknown imagery mode {imagery!r}; expected one of {', '.join(IMAGERY_MODES)}"
        )
    if urban_geometry is not None:
        if urban_geometry not in URBAN_GEOMETRY_CHOICES:
            raise VisualConfigError(
                f"unknown urban geometry {urban_geometry!r}; expected one of "
                f"{', '.join(URBAN_GEOMETRY_CHOICES)}"
            )
        if mode != URBAN_MODE:
            # A flag that silently does nothing reads as a flag that worked.
            raise VisualConfigError(
                f"--urban-geometry only applies to --visual {URBAN_MODE}; "
                f"--visual {mode} draws no city layer"
            )
    if mode == URBAN_MODE and imagery == "sentinel2":
        # Same reasoning as below: with an ion token the provider path is taken
        # and Sentinel-2 would sit unseen under it; without one it cannot load.
        raise VisualConfigError(
            "--visual urban-presentation keeps a keyless base layer; --imagery sentinel2 "
            "would be metered under provider tiles or cannot load without a token. "
            "Drop --imagery, or use --visual physics for the Sentinel-2 skin."
        )
    if mode == "photorealistic" and imagery == "sentinel2":
        # Only reached when --imagery sentinel2 was passed EXPLICITLY (the flag
        # defaults to None). It is a request that cannot be honoured rather than
        # a preference: the tileset is opaque where it has coverage, so a
        # Sentinel-2 layer under it would be metered against the ion quota and
        # never seen. Refuse instead of silently dropping it.
        raise VisualConfigError(
            "--visual photorealistic draws Google's 3D Tiles over a keyless base layer; "
            "--imagery sentinel2 would be metered and never visible. Drop --imagery, "
            "or use --visual physics for the Sentinel-2 skin."
        )


def default_imagery_for(mode: str) -> str:
    """The base-imagery skin a mode picks when ``--imagery`` was not given.

    Photorealistic keeps the keyless layer underneath its tileset; physics wants
    the best skin the credentials allow, and degrades to OSM on its own.
    """
    return "osm" if mode in ("photorealistic", URBAN_MODE) else "sentinel2"


def describe(config: VisualConfig | dict, token: str | None = None) -> str:
    """One line for stdout at startup. Accepts a VisualConfig or a page dict."""
    if isinstance(config, VisualConfig):
        d = config.as_dict()
        has_token = bool(token) or config.ion_token_present
    else:
        d = config
        has_token = bool(token)

    if d.get("mode") == URBAN_MODE:
        src = d.get("geometry_source")
        if src == GEOMETRY_PROVIDER:
            what = ("Google Photorealistic 3D Tiles if the browser can load them (the page "
                    "reports 'provider buildings active' only once tiles are on screen), "
                    + ("else the local OSM building cache"
                       if d.get("local_urban_state") == "available" else "no local fallback"))
        elif src == GEOMETRY_LOCAL:
            what = "local cached OpenStreetMap buildings and roads over the simulation DEM"
            if (d.get("urban_geometry") == URBAN_GEOMETRY_LOCAL
                    and (d.get("ion_token_present") or d.get("google_api_key_present"))):
                what += (" (a provider credential is set but not used: Google Photorealistic "
                         "3D Tiles are opt-in with --urban-geometry provider)")
        else:
            what = "URBAN DATA UNAVAILABLE -- no buildings drawn"
        return f"visuals: urban-presentation -- {what}. PRESENTATION MODE: {BUILDING_LOS_NOTE}"
    if d.get("imagery", d["mode"]) == "google_3d_tiles":
        via = ("Cesium ion asset {}".format(d["tileset_ion_asset"])
               if d.get("tileset_route") == "cesium_ion" else "the Google Maps Tiles API")
        return (
            f"visuals: Google Photorealistic 3D Tiles via {via} -- PRESENTATION MODE. "
            "The surface drawn is Google's, not the simulation's DEM: nothing here is "
            "evidence about terrain masking."
        )
    if d.get("base_imagery", d["mode"]) == "sentinel2":
        return (
            f"imagery: Copernicus Sentinel-2 via Cesium ion asset {d['ion_asset']} "
            "(ESA open data). Terrain is the simulation's own DEM, not ion's."
        )
    hint = (
        "set NAIGOS_CESIUM_ION_TOKEN (free Community-tier key at ion.cesium.com) "
        "for Copernicus Sentinel-2"
        if not has_token else "forced with --imagery osm"
    )
    return f"imagery: OpenStreetMap -- {hint}. Terrain is the simulation's own DEM either way."
