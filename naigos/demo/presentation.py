"""Everything about HOW a scene is presented that is not the visual provider.

``naigos.demo.imagery`` resolves what the drawn surface and buildings are.
This resolves the rest, once, at startup, the same way for the live server,
the served replay and the static export:

* the atmosphere profile (``naigos.demo.atmosphere``) -- the city's own under a
  presentation mode, ``neutral`` under physics unless one is named;
* whether the opt-in ``conflict_ambience`` stream runs, with which setting and
  visual seed (``naigos.demo.ambience``);
* the scenario framing and the checkpoint disclosure (``naigos.demo.scenario``);
* the theatre's city config, if it has one (``naigos.demo.cities``).

Two refusals, both before anything starts:

* ambience under ``--visual physics``. Physics is the evidence mode: every
  flash on screen there is a simulated outcome (``naigos.demo.events``), and a
  fictional flash beside them would make a physics screenshot say something
  the simulation did not. The ambience is a presentation-mode feature.
* ambience on a theatre with no city config: its origins come from the city's
  declared regions and protected zones, and without them there is no honest
  place to put anything.

Credential-free and simulation-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ambience as ambience_mod
from . import atmosphere as atmosphere_mod
from . import cities as cities_mod
from . import scenario as scenario_mod

AMBIENCE_CHOICES = ("off", ambience_mod.PROFILE_KEY)


class PresentationError(ValueError):
    """A presentation request that must be refused rather than approximated."""


@dataclass
class Presentation:
    theatre: str | None
    visual_mode: str
    atmosphere: atmosphere_mod.AtmosphereProfile
    ambience: ambience_mod.AmbienceConfig
    scenario: dict
    city: cities_mod.CityConfig | None

    def scene_fields(self) -> dict:
        """What `/scene` carries about the presentation. Same shape live and replay."""
        return {
            "lighting": self.atmosphere.lighting(),
            "atmosphere": self.atmosphere.as_dict(),
            "ambience": self.ambience.as_dict(),
            "scenario": self.scenario,
            "city": ({"aoi": self.city.aoi, "label": self.city.label} if self.city else None),
        }

    def live_ambience(self):
        """The generator a live simulation advances, or None when ambience is off."""
        if not self.ambience.enabled:
            return None
        return ambience_mod.LiveAmbience(self.city, self.ambience)

    def replay_ambience(self, duration_s: float, positions_at) -> dict | None:
        """The whole stream for a recording, or None when ambience is off."""
        if not self.ambience.enabled:
            return None
        return ambience_mod.stream_block(self.city, self.ambience, duration_s, positions_at)

    def summary(self) -> dict:
        """For the smoke report and stdout: what was chosen and why it is safe to show."""
        out = {
            "atmosphere_profile": self.atmosphere.key,
            "atmosphere_presentation_only": True,
            "ambience": self.ambience.as_dict(),
            "scenario": self.scenario["label"],
            "checkpoint": self.scenario["checkpoint"].get("text"),
            "checkpoint_relation": self.scenario["checkpoint"].get("relation"),
            "city_config": self.city.aoi if self.city else None,
        }
        if self.ambience.enabled:
            mask = ambience_mod.build_mask(self.city, self.ambience.seed)
            first = ambience_mod.generate(self.city, ambience_mod.get_setting(self.ambience.setting),
                                          mask, self.ambience.seed, 0.0, 600.0)
            out["ambience"].update(mask=mask.summary(), events_first_10_min=len(first))
        return out


def resolve(theatre: str | None, visual_mode: str, *, atmosphere: str | None = None,
            ambience: str | None = None, ambience_setting: str | None = None,
            visual_seed: int = 0, checkpoint: dict | None = None,
            layout_seed: int | None = None) -> Presentation:
    """Resolve the presentation for one run. Raises PresentationError on a refusal."""
    city = cities_mod.get_city(theatre)
    try:
        profile = atmosphere_mod.resolve(atmosphere, visual_mode,
                                         city.atmosphere_profile if city else None)
    except atmosphere_mod.AtmosphereError as e:
        raise PresentationError(str(e)) from e

    wants = ambience not in (None, "off")
    if wants and ambience != ambience_mod.PROFILE_KEY:
        raise PresentationError(f"unknown ambience {ambience!r}; expected one of {AMBIENCE_CHOICES}")
    if wants and visual_mode == "physics":
        raise PresentationError(
            "--ambience conflict_ambience is presentation-only and is refused under --visual "
            "physics: the evidence mode draws only effects the simulation produced. Use "
            "--visual urban-presentation (or photorealistic).")
    if wants and city is None:
        raise PresentationError(
            f"--ambience needs a city config for {theatre!r} (naigos/demo/cities/{theatre}.json): "
            "effect origins come from its declared regions and protected zones")
    if wants:
        try:
            ambience_mod.get_setting(ambience_setting or ambience_mod.DEFAULT_SETTING)
        except ambience_mod.AmbienceError as e:
            raise PresentationError(str(e)) from e
    amb = ambience_mod.AmbienceConfig(
        enabled=wants,
        setting=(ambience_setting or ambience_mod.DEFAULT_SETTING) if wants else None,
        seed=int(visual_seed))

    aoi_scenario = None
    if theatre:
        try:
            from ..research.aoi import get_aoi

            aoi_scenario = get_aoi(theatre).scenario or None
        except KeyError:
            aoi_scenario = None
    block = scenario_mod.scenario_block(theatre, checkpoint, aoi_scenario, layout_seed)
    return Presentation(theatre=theatre, visual_mode=visual_mode, atmosphere=profile,
                        ambience=amb, scenario=block, city=city)
