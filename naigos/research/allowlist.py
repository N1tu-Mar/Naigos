"""Fixed source allowlist and the data-scope guardrail.

Spec section 8: the research agent may only pull from an explicit allowlist. There is no
open-ended web search. Every fetch in this package routes through :func:`check_url`, so a
source that is not listed here cannot enter the cache, and therefore cannot enter the env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class Source:
    """One allowlisted upstream data source."""

    key: str
    name: str
    hosts: tuple[str, ...]
    license: str
    license_url: str
    citation: str
    role: str
    attribution_required: bool = False
    notes: str = ""
    #: Fetched for the demo globe only, never by the research/snapshot build.
    #: Such a source is allowlisted for its own fetcher, and deliberately NOT
    #: among the hosts the pipeline's egress guard lets a snapshot resolve.
    visual_only: bool = False
    docs: tuple[str, ...] = field(default_factory=tuple)


ALLOWLIST: dict[str, Source] = {
    "usgs_3dep": Source(
        key="usgs_3dep",
        name="USGS 3D Elevation Program (3DEP)",
        hosts=("elevation.nationalmap.gov", "www.usgs.gov", "apps.nationalmap.gov"),
        license="Public domain (US Government work, USGS)",
        license_url="https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
        citation=(
            "U.S. Geological Survey, 3D Elevation Program (3DEP) 1/3 arc-second Digital "
            "Elevation Model. USGS National Map 3DEP Dynamic Service."
        ),
        role="Terrain elevation grid for radar line-of-sight masking and terrain-following routes.",
        docs=("https://www.usgs.gov/3d-elevation-program",),
        notes="CONUS coverage only. Non-US areas of interest must fall back to Copernicus GLO-30.",
    ),
    "copernicus_dem": Source(
        key="copernicus_dem",
        name="Copernicus DEM GLO-30",
        hosts=(
            "portal.opentopography.org",
            "opentopography.org",
            # ESA's own open-data distribution on AWS. No key, no signing, and it
            # serves Cloud-Optimized GeoTIFFs, so a windowed read pulls only the
            # bytes covering the AOI instead of a 1-degree tile per corner.
            "copernicus-dem-30m.s3.amazonaws.com",
            "copernicus-dem-30m.s3.eu-central-1.amazonaws.com",
        ),
        license="Copernicus DEM free/open licence (ESA/Airbus), attribution required",
        license_url="https://spacedata.copernicus.eu/documents/20123/121286/CSCDA_ESA_Mission-specific+Annex.pdf",
        citation=(
            "European Space Agency, Copernicus DEM GLO-30 Global 30m Digital Surface Model, "
            "distributed by OpenTopography."
        ),
        role="Global DEM fallback where 3DEP has no coverage.",
        attribution_required=True,
        docs=("https://portal.opentopography.org/apidocs/",),
        notes=(
            "Primary path is the unauthenticated AWS open-data bucket (COG, windowed read). "
            "The OpenTopography portal is an alternative that needs a free API key in "
            "NAIGOS_OPENTOPO_KEY. Used for AOIs outside 3DEP's CONUS footprint."
        ),
    ),
    "ourairports": Source(
        key="ourairports",
        name="OurAirports",
        hosts=("davidmegginson.github.io", "ourairports.com"),
        license="Public domain (dedicated to the public domain by OurAirports)",
        license_url="https://ourairports.com/data/",
        citation="OurAirports open data (airports.csv, runways.csv), public domain.",
        role="Airfield and runway geometry -> sortie start points, objectives, no-fly structure.",
        docs=("https://ourairports.com/data/",),
    ),
    "opensky": Source(
        key="opensky",
        name="OpenSky Network REST API",
        hosts=("opensky-network.org", "auth.opensky-network.org"),
        license="Free for non-commercial / research use under the OpenSky Network terms",
        license_url="https://opensky-network.org/about/terms-of-use",
        citation=(
            "Schafer, M., Strohmeier, M., Lenders, V., Martinovic, I., Wilhelm, M. (2014). "
            "Bringing up OpenSky: A large-scale ADS-B sensor network for research. IPSN 2014, "
            "pp. 83-94."
        ),
        role="Real civil aircraft kinematics -> calibrate airframe speed/climb/turn-rate envelopes.",
        attribution_required=True,
        docs=("https://openskynetwork.github.io/opensky-api/rest.html",),
        notes="Anonymous access is rate limited. Sampling is snapshot-based and cached.",
    ),
    "open_meteo": Source(
        key="open_meteo",
        name="Open-Meteo Forecast API",
        hosts=("api.open-meteo.com", "open-meteo.com"),
        license="CC BY 4.0 (Open-Meteo), free for non-commercial use without an API key",
        license_url="https://open-meteo.com/en/license",
        citation="Open-Meteo.com free weather API, CC BY 4.0.",
        role="Surface pressure/temperature/humidity/wind -> air density and ceiling modelling.",
        attribution_required=True,
        docs=("https://open-meteo.com/en/docs",),
    ),
    "copernicus_sentinel2": Source(
        key="copernicus_sentinel2",
        name="Copernicus Sentinel-2 (via Cesium ion asset 3954)",
        # Empty on purpose. Nothing in this package fetches Sentinel-2: the tiles
        # are streamed by the BROWSER straight from Cesium ion while the demo is
        # open, and no pixel ever reaches data_cache/, a component parameter, or
        # an observation. This entry exists to carry the licence, the citation
        # and the attribution requirement -- the same role radar_theory plays for
        # data that is derived rather than downloaded.
        hosts=(),
        license="Copernicus open licence (ESA/Copernicus Sentinel data), free to use, attribution required",
        license_url="https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
        citation=(
            "European Space Agency / Copernicus, Sentinel-2 MSI optical imagery, served as "
            "Cesium ion asset 3954. Contains modified Copernicus Sentinel data."
        ),
        role=(
            "Visual base imagery for the demo globe only. Cosmetic: the detection model "
            "consumes the DEM, never satellite pixels."
        ),
        attribution_required=True,
        docs=(
            "https://cesium.com/legal/terms-of-service/",
            "https://sentinels.copernicus.eu/copernicus/sentinel-2",
        ),
        notes=(
            "Chosen over Cesium's default Bing Aerial base layer, which is third-party "
            "commercial data metered by session under Microsoft's terms. Sentinel-2 is ESA "
            "open data -- free to use, not merely free to view. Access needs a Cesium ion "
            "token, read from NAIGOS_CESIUM_ION_TOKEN; the free Community tier covers "
            "individual and non-commercial use. Without a token the demo falls back to "
            "keyless OpenStreetMap and no claim changes."
        ),
    ),
    "google_photorealistic_3d_tiles": Source(
        key="google_photorealistic_3d_tiles",
        # Empty for the same reason as Sentinel-2: nothing in this package fetches
        # the tileset. The BROWSER streams it from Cesium ion (asset 2275207) or
        # from Google's Map Tiles API while the demo is open, and no byte of it
        # reaches data_cache/, a component parameter, or an observation. The entry
        # exists to carry the licence, the citation and the attribution flag.
        hosts=(),
        name="Google Photorealistic 3D Tiles (via CesiumJS / Cesium ion asset 2275207)",
        license="Google Maps Platform Terms of Service; attribution and credit display required",
        license_url="https://cloud.google.com/maps-platform/terms",
        citation=(
            "Google Photorealistic 3D Tiles, streamed via CesiumJS (Cesium ion asset 2275207 "
            "or the Google Map Tiles API). Imagery and 3D geometry (c) Google."
        ),
        role=(
            "Optional presentation-only skin for the demo globe (--visual photorealistic). "
            "NOT evidence: the tileset carries its own geometry, so the surface drawn in that "
            "mode is the provider's rather than the simulation's DEM. The detection model "
            "consumes the DEM and never this."
        ),
        attribution_required=True,
        docs=(
            "https://developers.google.com/maps/documentation/tile/3d-tiles",
            "https://cesium.com/platform/cesiumjs/photorealistic-3d-tiles/",
        ),
        notes=(
            "Credentials are read from explicit environment variables only -- "
            "NAIGOS_CESIUM_ION_TOKEN/CESIUM_ION_TOKEN for the ion route, "
            "NAIGOS_GOOGLE_MAPS_API_KEY/GOOGLE_MAPS_API_KEY for the direct route -- and are "
            "never committed. Without either, --visual photorealistic falls back to the "
            "evidence-grade physics mode. Google's terms require the Google attribution and "
            "the per-tile data credits to stay visible, so the viewer never hides Cesium's "
            "credit display."
        ),
    ),
    "osm_urban_visual": Source(
        key="osm_urban_visual",
        name="OpenStreetMap civilian buildings and roads (Overpass API, visual only)",
        # The one Overpass instance run by the OSM community's main operator.
        # Queried by `naigos.demo.urban` alone, with a fixed, bounded query.
        hosts=("overpass-api.de",),
        license="Open Database License (ODbL) 1.0; attribution and share-alike required",
        license_url="https://www.openstreetmap.org/copyright",
        citation=(
            "(c) OpenStreetMap contributors. Building footprints and major-road centrelines "
            "retrieved through the Overpass API, available under the Open Database License."
        ),
        role=(
            "PRESENTATION ONLY: civilian building footprints and major-road centrelines for "
            "the --visual urban-presentation city layer. Never terrain, never radar cover, "
            "never read by the simulation: the detection model and LOS consume the DEM only."
        ),
        attribution_required=True,
        visual_only=True,
        docs=(
            "https://wiki.openstreetmap.org/wiki/Overpass_API",
            "https://opendatacommons.org/licenses/odbl/1-0/",
        ),
        notes=(
            "Bounded to a documented urban sub-box inside the AOI (naigos.demo.urban."
            "URBAN_BOUNDS), with server-side timeout and maxsize and a client-side byte cap. "
            "Military-tagged buildings and anything inside landuse=military are excluded by "
            "the query; no names or other tags reach the browser. Cached under "
            "data_cache/visual/urban/, outside the research manifest, so it cannot become a "
            "component parameter. Derived data stays ODbL: credited on screen wherever drawn."
        ),
    ),
    "radar_theory": Source(
        key="radar_theory",
        name="Open radar propagation and detection literature",
        hosts=(),  # derived, not fetched
        license="N/A - the model is derived from published first-principles equations",
        license_url="",
        citation=(
            "Skolnik, M. I. (2008). Radar Handbook, 3rd ed., McGraw-Hill (radar range equation, "
            "ch. 1-2). "
            "Barton, D. K. (2013). Radar Equations for Modern Radar, Artech House. "
            "Blake, L. V. (1986). Radar Range-Performance Analysis, Artech House (pattern-"
            "propagation factor, atmospheric attenuation). "
            "ITU-R P.526-15 (2019), Propagation by diffraction (terrain diffraction, knife-edge). "
            "ITU-R P.676-13 (2022), Attenuation by atmospheric gases. "
            "Swerling, P. (1960). Probability of detection for fluctuating targets, IRE Trans. IT-6."
        ),
        role="Physics basis for the detection-probability model (range, RCS, terrain LOS, Swerling).",
        notes=(
            "Derived, never fetched. Parameters are generic and notional by construction; see "
            "the guardrail in this module."
        ),
    ),
}


class SourceNotAllowed(RuntimeError):
    """Raised when a fetch targets a host outside the allowlist."""


def check_url(url: str, source_key: str) -> Source:
    """Validate that ``url`` belongs to the allowlisted ``source_key``. Returns the source."""
    if source_key not in ALLOWLIST:
        raise SourceNotAllowed(f"unknown source key {source_key!r}; allowlist={sorted(ALLOWLIST)}")
    src = ALLOWLIST[source_key]
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SourceNotAllowed(f"{url!r} is not https; refusing plaintext fetch")
    host = (parsed.hostname or "").lower()
    if host not in src.hosts:
        raise SourceNotAllowed(
            f"host {host!r} is not allowlisted for source {source_key!r} (allowed: {src.hosts})"
        )
    return src


# --- Data-scope guardrail (spec section 8) ------------------------------------------------

GUARDRAIL = """\
Threat models in Naigos are PARAMETERIZED ABSTRACTIONS: a detection range scale, an altitude
band, a reaction latency, a detection-probability curve, and a lethal-envelope radius. They are
derived from the open radar range equation with generic, notional parameters chosen to produce
a usable exposure-vs-survival gradient for the RL problem.

The research agent does not assemble, and this repository does not contain, a current or precise
capability database for any specific fielded weapon system. The RL problem depends on the SHAPE
of the tradeoff, not on real-world accuracy against a real system, so no such data is needed.

If a data request drifts toward "precise current capabilities of a specific system in order to
defeat it", the agent declines the request and parameterizes instead.
"""

# Substrings that indicate a request has drifted out of scope. Matched case-insensitively
# against a free-text request string by `assert_in_scope`.
_OUT_OF_SCOPE_MARKERS = (
    "defeat the",
    "targeting-grade",
    "kill chain against",
    "actual performance of the",
    "real-world capabilities of the",
    "classified",
    "export-controlled",
)


class OutOfScope(RuntimeError):
    """Raised when a data request drifts toward a real-system capability database."""


def assert_in_scope(request: str) -> None:
    """Refuse a free-text data request that drifts toward real-system capability lookup."""
    low = request.lower()
    for marker in _OUT_OF_SCOPE_MARKERS:
        if marker in low:
            raise OutOfScope(
                f"request matched out-of-scope marker {marker!r}. {GUARDRAIL}"
            )
