"""Imagery for the globe viewer -- the demo's SKIN, and nothing else.

The viewer draws two layers that are deliberately kept apart:

    imagery   this module.  Sentinel-2 optical pixels draped over the globe.
              Cosmetic. Read by human eyes only.
    terrain   ``naigos.demo.live.build_terrain_grid``.  The simulation's own
              heightmap, the surface every line-of-sight ray was computed
              against, and the only elevation data the detection model consumes.

Keeping them separate is a correctness property, not a stylistic one. If the
globe's relief came from an imagery provider's own terrain (Cesium World
Terrain, say) the picture would look better and would no longer be evidence:
what occludes on screen would not be what occluded in the model. That defect
had already been shipped once and removed (next-steps E-9), so the split is
asserted by ``tests/test_imagery_layers.py`` rather than left to discipline.

**No satellite pixel ever enters an observation.** ``naigos/env/obs.py`` builds
its vectors from state and the heightmap; this module is imported by the demo
server only, and the test suite asserts that nothing under ``naigos/env`` or
``naigos/rl`` references imagery at all.

Why Sentinel-2 rather than Cesium's default
-------------------------------------------
Cesium ion's default base imagery is Bing Aerial: third-party commercial data,
metered per session, and its terms are Microsoft's rather than Cesium's. This is
a portfolio project, so the shipped demo uses Cesium ion asset 3954 -- Copernicus
Sentinel-2 -- which is ESA open data: free to *use*, not merely free to look at.
That removes the licensing question from the demo entirely.

Token
-----
Read from the environment (``NAIGOS_CESIUM_ION_TOKEN``, then ``CESIUM_ION_TOKEN``),
never committed. A free ion Community-tier account covers individual and
non-commercial use, which is what this is. Without a token the viewer falls back
to OpenStreetMap, which needs no key -- the terrain, and therefore every claim
the viewer makes, is identical either way.
"""

from __future__ import annotations

import os

#: Cesium ion asset id for Copernicus Sentinel-2 imagery.
SENTINEL2_ION_ASSET = 3954

#: Checked in order; the first non-empty one wins.
TOKEN_ENV_VARS = ("NAIGOS_CESIUM_ION_TOKEN", "CESIUM_ION_TOKEN")

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

#: Said in the HUD so a viewer cannot mistake the pretty layer for the modelled
#: one. This is the whole reason the module exists.
LAYER_SEPARATION_NOTE = (
    "Imagery is a skin. The relief under it -- and every line-of-sight test -- "
    "comes from the simulation's own DEM, not from this provider."
)


def resolve_ion_token(cli_token: str | None = None, env: dict[str, str] | None = None) -> str | None:
    """Return the ion token from ``--ion-token`` or the environment, else ``None``.

    An explicit flag wins so a one-off run can override a shell export. Blank and
    whitespace-only values are treated as absent: an empty ``CESIUM_ION_TOKEN=``
    left in a shell profile would otherwise be sent to ion and rejected, which
    surfaces as a blank globe rather than as a missing token.
    """
    if cli_token and cli_token.strip():
        return cli_token.strip()
    env = os.environ if env is None else env
    for name in TOKEN_ENV_VARS:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def imagery_config(token: str | None, prefer: str = "sentinel2") -> dict:
    """Describe the imagery layer for the browser. Terrain is not in this dict.

    ``prefer="osm"`` forces the keyless provider even when a token is present,
    which is how the token path gets exercised against a control.
    """
    if prefer not in ("sentinel2", "osm"):
        raise ValueError(f"unknown imagery mode {prefer!r}; expected 'sentinel2' or 'osm'")
    if prefer == "sentinel2" and token:
        return {
            "mode": "sentinel2",
            "ion_asset": SENTINEL2_ION_ASSET,
            "attribution": SENTINEL2_ATTRIBUTION,
            "separation_note": LAYER_SEPARATION_NOTE,
            "osm_url": OSM_TILE_URL,          # fallback if the ion request fails
            "osm_attribution": OSM_ATTRIBUTION,
        }
    return {
        "mode": "osm",
        "ion_asset": None,
        "attribution": OSM_ATTRIBUTION,
        "separation_note": LAYER_SEPARATION_NOTE,
        "osm_url": OSM_TILE_URL,
        "osm_attribution": OSM_ATTRIBUTION,
    }


def describe(config: dict, token: str | None) -> str:
    """One line for stdout at startup."""
    if config["mode"] == "sentinel2":
        return (
            f"imagery: Copernicus Sentinel-2 via Cesium ion asset {config['ion_asset']} "
            "(ESA open data). Terrain is the simulation's own DEM, not ion's."
        )
    hint = (
        "set NAIGOS_CESIUM_ION_TOKEN (free Community-tier key at ion.cesium.com) "
        "for Copernicus Sentinel-2"
        if not token else "forced with --imagery osm"
    )
    return f"imagery: OpenStreetMap -- {hint}. Terrain is the simulation's own DEM either way."
