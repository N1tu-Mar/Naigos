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
One knob, ``--visual``, chooses between two whole postures. They differ in what
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

Both modes fall back to keyless OpenStreetMap rather than failing, and
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

#: The two visual postures. ``physics`` is the default and the only one whose
#: drawn surface is the surface the simulation computed against.
VISUAL_MODES = ("physics", "photorealistic")
DEFAULT_VISUAL_MODE = "physics"

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
        if self.tileset_attribution:
            return f"{self.tileset_attribution} {self.base_attribution}"
        return self.base_attribution

    @property
    def separation_note(self) -> str:
        """The layer-separation sentence, or the warning that replaces it."""
        return LAYER_SEPARATION_NOTE if self.evidence_grade else PHOTOREALISTIC_EVIDENCE_WARNING

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
    """
    if mode not in VISUAL_MODES:
        raise VisualConfigError(
            f"unknown visual mode {mode!r}; expected one of {', '.join(VISUAL_MODES)}"
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


def validate_cli(mode: str, imagery: str | None) -> None:
    """Reject an impossible ``--visual``/``--imagery`` pair before anything starts.

    argparse's ``choices`` already covers a bad single value; this covers the
    combination, and gives a caller that builds the args itself the same check.
    ``imagery=None`` means "let the mode pick", which is always resolvable.
    """
    if mode not in VISUAL_MODES:
        raise VisualConfigError(
            f"unknown visual mode {mode!r}; expected one of {', '.join(VISUAL_MODES)}"
        )
    if imagery is not None and imagery not in IMAGERY_MODES:
        raise VisualConfigError(
            f"unknown imagery mode {imagery!r}; expected one of {', '.join(IMAGERY_MODES)}"
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
    return "osm" if mode == "photorealistic" else "sentinel2"


def imagery_config(token: str | None, prefer: str = "sentinel2") -> dict:
    """Describe the imagery layer for the browser. Terrain is not in this dict.

    The pre-``--visual`` entry point, kept because it is the narrow question --
    "which skin, given this token" -- that most callers actually have.
    ``prefer="osm"`` forces the keyless provider even when a token is present,
    which is how the token path gets exercised against a control.
    """
    if prefer not in IMAGERY_MODES:
        raise ValueError(f"unknown imagery mode {prefer!r}; expected 'sentinel2' or 'osm'")
    return resolve_visual_config("physics", ion_token=token, imagery=prefer).to_page()


def describe(config: VisualConfig | dict, token: str | None = None) -> str:
    """One line for stdout at startup. Accepts a VisualConfig or a page dict."""
    if isinstance(config, VisualConfig):
        d = config.as_dict()
        has_token = bool(token) or config.ion_token_present
    else:
        d = config
        has_token = bool(token)

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
