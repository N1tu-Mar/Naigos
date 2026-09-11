"""Presentation-only atmosphere profiles: sun, sky, haze and dust for the globe viewer.

A profile is a MOOD, chosen from the explicit allowlist below by name -- never
generated in the browser, never random, never read from weather data. It sets
where the light comes from, how the sky and the ground atmosphere are tinted,
how much distance haze the renderer adds, and whether a low dust veil is drawn
near the ground. Nothing else.

What a profile can never do
---------------------------
* Change what the simulation computes. This module imports nothing from
  ``naigos.env`` or ``naigos.rl`` and nothing there imports it
  (``tests/test_demo_isolation.py``); a haze density is a shader constant, not
  an attenuation term. Detection, LOS, observations and rewards are identical
  under every profile, which ``tests/test_city_presentation.py`` checks by
  stepping the env under each one.
* Conceal a simulation entity. Haze is capped (``MAX_FOG_DENSITY``,
  ``MAX_DUST_ALPHA``) so aircraft and threat models, their markers and their
  labels stay legible at the ranges the camera presets use.
* Claim anything about real conditions. The Dubai profile is warm and hazy
  because that reads as a coastal desert, not because a forecast said so; the
  cited atmosphere that DOES enter the physics (density, refraction) is the
  ``data.atmosphere`` component, and it is untouched by all of this.

Physics mode keeps ``neutral``: its lighting is the cartographic sun the
hillshade convention needs, and its screenshots are evidence. Profiles are for
the presentation modes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

#: Hard caps. Tested; a profile over either is refused at import.
MAX_FOG_DENSITY = 2.5e-4      # Cesium scene.fog.density; the library default is 2.0e-4
MAX_DUST_ALPHA = 0.22         # the near-ground dust veil's peak opacity
MAX_SHIMMER = 0.0015          # heat-shimmer UV displacement; 0 disables the stage


@dataclass(frozen=True)
class AtmosphereProfile:
    """One allowlisted look. Every field is a renderer constant."""

    key: str
    description: str
    #: Where the light comes from, in the AOI's local frame (compass azimuth,
    #: elevation above the horizon). Drives the hillshade and the model light.
    sun_azimuth_deg: float
    sun_altitude_deg: float
    #: The directional light's colour and intensity.
    light_color: str
    light_intensity: float
    #: Sky and ground-atmosphere tint, as Cesium hue/saturation/brightness shifts.
    sky_hue_shift: float
    sky_saturation_shift: float
    sky_brightness_shift: float
    #: Distance haze. Density is a renderer constant, capped above.
    fog_density: float
    fog_min_brightness: float
    #: Colour of the low dust veil and of dust-burst effects.
    haze_color: str
    dust_alpha: float
    #: Heat-shimmer strength near the ground; 0 = off. Dropped entirely in
    #: reduced-motion and performance mode.
    shimmer: float

    def __post_init__(self):
        if not (0.0 <= self.fog_density <= MAX_FOG_DENSITY):
            raise ValueError(f"{self.key}: fog_density {self.fog_density} exceeds {MAX_FOG_DENSITY}")
        if not (0.0 <= self.dust_alpha <= MAX_DUST_ALPHA):
            raise ValueError(f"{self.key}: dust_alpha {self.dust_alpha} exceeds {MAX_DUST_ALPHA}")
        if not (0.0 <= self.shimmer <= MAX_SHIMMER):
            raise ValueError(f"{self.key}: shimmer {self.shimmer} exceeds {MAX_SHIMMER}")
        if not (5.0 <= self.sun_altitude_deg <= 85.0):
            raise ValueError(f"{self.key}: a sun below 5 degrees would light models from underneath")

    def sun_vector_enu(self) -> tuple[float, float, float]:
        """Unit vector TOWARD the sun in east-north-up (cf. camera.sun_vector_enu)."""
        az, el = math.radians(self.sun_azimuth_deg), math.radians(self.sun_altitude_deg)
        return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))

    def lighting(self) -> dict:
        """The `/scene` lighting block this profile implies."""
        return {"sun_azimuth_deg": self.sun_azimuth_deg,
                "sun_altitude_deg": self.sun_altitude_deg}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["presentation_only"] = True
        return d


#: The allowlist. A city config names one of these; nothing else is accepted.
PROFILES: dict[str, AtmosphereProfile] = {
    "neutral": AtmosphereProfile(
        key="neutral",
        description=("The evidence-mode look: cartographic sun from the north-west at 45 degrees, "
                     "Cesium's default sky and haze, no dust, no shimmer."),
        sun_azimuth_deg=315.0, sun_altitude_deg=45.0,
        light_color="#ffffff", light_intensity=2.2,
        sky_hue_shift=0.0, sky_saturation_shift=0.0, sky_brightness_shift=0.0,
        fog_density=2.0e-4, fog_min_brightness=0.03,
        haze_color="#c8c8c8", dust_alpha=0.0, shimmer=0.0,
    ),
    "high_basin_clear": AtmosphereProfile(
        key="high_basin_clear",
        description=("A high, dry basin under a range: crisp late-morning light from the "
                     "south-east so the ridge behind the city is side-lit, light blue haze."),
        sun_azimuth_deg=135.0, sun_altitude_deg=38.0,
        light_color="#fff4e6", light_intensity=2.3,
        sky_hue_shift=0.0, sky_saturation_shift=-0.05, sky_brightness_shift=0.02,
        fog_density=1.6e-4, fog_min_brightness=0.05,
        haze_color="#b9c3cf", dust_alpha=0.06, shimmer=0.0,
    ),
    "warm_coastal_desert": AtmosphereProfile(
        key="warm_coastal_desert",
        description=("Warm coastal desert: low late-afternoon sun from the west-south-west over "
                     "the water, golden light, sand-coloured haze and a faint heat shimmer."),
        sun_azimuth_deg=250.0, sun_altitude_deg=24.0,
        light_color="#ffd9a8", light_intensity=2.4,
        sky_hue_shift=-0.02, sky_saturation_shift=-0.12, sky_brightness_shift=0.03,
        fog_density=2.3e-4, fog_min_brightness=0.08,
        haze_color="#d8c3a0", dust_alpha=0.12, shimmer=0.0008,
    ),
    "hot_dusty_inland": AtmosphereProfile(
        key="hot_dusty_inland",
        description=("Hot, dry mountain basin: high hard sun from the south-south-west, bleached "
                     "sky, mild ochre dust haze that deepens the relief. Restrained."),
        sun_azimuth_deg=200.0, sun_altitude_deg=56.0,
        light_color="#fff0d6", light_intensity=2.5,
        sky_hue_shift=-0.03, sky_saturation_shift=-0.2, sky_brightness_shift=0.05,
        fog_density=2.1e-4, fog_min_brightness=0.08,
        haze_color="#cdb48e", dust_alpha=0.10, shimmer=0.0006,
    ),
}

DEFAULT_PROFILE = "neutral"
#: `--atmosphere` values besides a profile key.
AUTO = "auto"


class AtmosphereError(ValueError):
    """An atmosphere request that is not on the allowlist."""


def get_profile(key: str) -> AtmosphereProfile:
    if key not in PROFILES:
        raise AtmosphereError(f"unknown atmosphere profile {key!r}; allowlist: {sorted(PROFILES)}")
    return PROFILES[key]


def resolve(requested: str | None, visual_mode: str, city_profile: str | None) -> AtmosphereProfile:
    """The profile in force: explicit key > the city's own (presentation modes) > neutral.

    ``auto`` (or None) keeps physics on ``neutral`` -- its look is part of what
    makes a physics screenshot comparable to every other one -- and gives a
    presentation mode the city's declared profile. Deterministic: the same
    inputs always select the same profile.
    """
    if requested not in (None, AUTO):
        return get_profile(requested)
    if visual_mode != "physics" and city_profile:
        return get_profile(city_profile)
    return PROFILES[DEFAULT_PROFILE]
