# City theatres

Every theatre here is a **notional contested-airspace simulation** over real
terrain. It is not a digital twin and it makes no claim about any real force,
facility, event or current condition. Blue aircraft are evasive only and carry
no weapon; red ground and air entities are the project's generic, procedurally
generated hazards, drawn at random per seed and re-rolled -- never placed from
real-world data or at named locations. No city shown here is a battlefield.

One theatre, one set of files. Nothing below is a shared table, so a theatre
developed on its own branch merges without touching another's files.

| file | what it is | who writes it |
| --- | --- | --- |
| `naigos/research/aois/<aoi>.json` | the AOI box, rationale, bounds policy, protected zones | hand-written, reviewed |
| `components/aoi/<aoi>/*.json` | the cited component snapshot the env reads | `naigos-research --aoi <aoi>` |
| `docs/theatres/<aoi>/DATA.md` | the theatre's provenance doc | `naigos-research --aoi <aoi>` |
| `naigos/demo/cities/<aoi>.json` | how the theatre is presented: urban box, atmosphere, cameras, ambience regions | hand-written, reviewed |
| `tests/fixtures/viewer_grid_<aoi>.npz` | the viewer's terrain grid, for offline camera and georeference tests | `scripts/make_viewer_grid_fixture.py --aoi <aoi>` |
| `docs/theatres/<aoi>.md` | this theatre's own notes: bounds, sources, commands, limitations | hand-written |
| `tests/test_theatre_<aoi>.py` | the theatre's own checks (landmarks, zones, configuration) | hand-written |

`tests/test_city_presentation.py` and `tests/test_city_page.py` hold every
city config to the shared standard automatically; a new theatre adds its files
and changes no test.

## Commands, for any theatre `<aoi>`

```bash
# Research: DEM, atmosphere, civil airfields -- allowlisted, cached, cited.
# Credentials never go in arguments. A second run makes no network call.
uv run naigos-research --aoi <aoi>

# The bounded, visual-only city cache (OSM buildings and roads). Presentation only.
uv run python -m naigos.demo.urban --aoi <aoi>

# Evidence-grade terrain/LOS viewer (the simulation's own DEM).
uv run python -m naigos.demo.live --aoi <aoi> \
  --checkpoint checkpoints/theatre_1000.pkl --visual physics --camera urban-overview --open

# Dense city presentation; never LOS evidence. Add the fictional ambience if wanted.
uv run python -m naigos.demo.live --aoi <aoi> \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --ambience conflict_ambience --visual-seed 7 --open

# Record a rollout on the theatre, and export a self-contained replay.
uv run python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl --aoi <aoi> --out runs/<aoi>
uv run python -m naigos.demo.viewer runs/<aoi>/demo.json --visual urban-presentation \
  --ambience conflict_ambience --visual-seed 7 --open

# What the viewer would draw, without a browser.
uv run python -m naigos.demo.live --aoi <aoi> --visual urban-presentation --smoke-render
```

## What the standard guarantees

**Bounds and zones.** The AOI is a documented rectangle; its `bounds_policy`
says how it was chosen and what it leaves out. Protected zones are boxes the
project stays out of, each with the consumers that enforce it:

| policy | enforced by |
| --- | --- |
| `extraction` | `naigos.demo.urban`: no footprint with a vertex inside is kept; roads are cut at the edge |
| `airfield` | `naigos.research.sources.airports`: no start point inside |
| `camera` | `naigos.demo.cities`: no preset sits in, aims at or frames it; the page keeps the free camera out and turns the chase camera away |
| `ambience` | `naigos.demo.ambience`: no effect originates within the zone, its buffer and a 1.5 km margin |
| `render_cutout` | the page: covered on the imagery and clipped out of provider 3D tiles |

**Cameras.** Every city declares `urban_overview` and `street_canyon`, and may
declare `coastal_corridor` or `valley_overview`. Each is checked before the page
sees it: above the simulation DEM (300 m for overviews, 60 m for street views),
a sight line the terrain does not cut, camera and target inside the city's safe
region (on land, inside the theatre), and no protected zone within 30 km inside
a 55 degree half-angle of the heading. `follow-aircraft` and `analysis-topdown`
are available everywhere.

**Atmosphere.** A profile from the allowlist in `naigos/demo/atmosphere.py`:
sun direction, light colour, sky tint, capped fog, a depth-aware haze veil and
optional heat shimmer. Physics keeps `neutral` unless one is named. Profiles are
renderer constants; the cited `data.atmosphere` component is what the physics
uses, and it is untouched.

**Fictional ambience.** `--ambience conflict_ambience` (opt-in, presentation
modes only) draws recurring distant flashes, smoke columns, dust bursts and
sparks. It is an art-directed VFX stream, not a munition, strike, target or
damage model. Every effect is a pure function of the visual seed, the setting
and the time bucket; its origin comes from a synthetic value-noise mask inside
the city's coarse, hand-drawn open-land regions, clear of the urban core,
every protected zone and every aircraft and threat. The stream is capped (at
most 10 per minute, 3 per 10 s bucket, 12 at once), exported with a static
replay, drawn on the replay clock, labelled fictional on screen, and dropped to
a still rendering under `prefers-reduced-motion`. Nothing in it can reach the
simulation: `tests/test_city_page.py` steps the env with and without it and
compares every state leaf.

**Framing.** The HUD's first line is `scenario: notional contested-airspace
simulation`, and the second says whether the checkpoint was trained on this
theatre or is flying it zero-shot. The same block is in `/scene`, in every
recording and in every export.

## What a city never does

- use real conflict data, current events, news imagery or live data;
- place anything at, or name, a real military site, a religious site, a
  pilgrimage venue, a hospital, a school, or any other sensitive location;
- render civilians, crowds, casualties, damage, flags or religious or political
  symbols;
- let a building, an effect or a haze setting enter LOS, detection,
  observations, rewards, training data or the verifier.
