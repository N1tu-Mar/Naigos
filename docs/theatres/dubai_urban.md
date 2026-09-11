# dubai_urban

A **notional contested-airspace simulation** over a flat Gulf-coast city. It is
a generic simulation theatre on real terrain, not a digital twin, and it makes
no claim about any real force, facility, event or current condition. Blue is
evasive only; red ground and air entities are the project's generic hazards,
drawn at random per seed and re-rolled -- never placed from real-world data or
at named locations.

## Bounds policy

`naigos/research/aois/dubai_urban.json`: **55.00--55.80 E, 25.05--25.38 N**
(about 81 x 37 km, UTM 40N / EPSG:32640, fingerprint `eb66169b03`).

A modest rectangular civilian urban-and-coastal test envelope: a stretch of
shoreline, a dense band of ordinary urban fabric and ~40 km of the inland dunes
behind it, and nothing chosen for its own sake. The first box (55 x 32 km) sat
inside most of the generic lethal envelopes -- every aircraft in a recorded world
was lost within ~30 s -- so it was deepened inland until flights stay legible. It is not an airport, port or sensitive-site
study. The south edge (25.05 N) is placed so the box stays clear of the military
airfield south of the city and of most of the Jebel Ali port basin. The civil
airports and port areas it does contain are protected zones with the `ambience`
policy: no presentation effect originates within them, their buffer, or a
further 1.5 km. Airfields whose published name marks them military are dropped
from start geometry (`exclude_military_airfields`); none fell inside the box.

Changing the box changes the fingerprint, which every cached artifact is keyed by.

## Data, licences, attribution

All inputs come through the allowlisted, cached, cited research pipeline
(`uv run naigos-research --aoi dubai_urban`); a second run makes no network call
(verified with every proxy pointed at a dead port). Provenance:
[`dubai_urban/DATA.md`](dubai_urban/DATA.md), components in
`components/aoi/dubai_urban/`.

| input | source | licence |
| --- | --- | --- |
| terrain | Copernicus DEM GLO-30 (AWS open-data COG, windowed read) | Copernicus DEM licence; attribution required |
| atmosphere | Open-Meteo forecast at the AOI centre | CC BY 4.0 |
| civil airfields | OurAirports | public domain |
| flight envelope | OpenSky (shared calibration sample) | OpenSky terms, non-commercial |
| city layer (visual only) | OpenStreetMap via Overpass, `naigos.demo.urban` | ODbL 1.0 -- (c) OpenStreetMap contributors |

Measured: 41.5 m mean elevation, 221 m relief (inland dunes and the DSM's
structure signal), mean slope 1.6 deg. Terrain masking is nearly absent (3.8%
masked from the high point at 100 m AGL, 0% above); the radar horizon -- earth
curvature over the measured refraction factor k = 1.566 -- carries the geometry.
Four civil airfields (DEM agrees with published field elevations within 15.6 m). The GLO-30 DSM
carries some structure height at 30 m; resampled to the env grid (>= 500 m) it
averages out. The local building extrusions are presentation only and never enter
LOS.

Georeference checks (`tests/test_theatre_dubai_urban.py`): open Gulf water reads
0 m +/- 6 m; the civil international airport's published field elevation (62 ft)
agrees within 20 m; Open-Meteo's own terrain height at its grid point agrees
within 25 m at the point the request named.

## Presentation

`naigos/demo/cities/dubai_urban.json`:

- **Urban box:** 55.18--55.31 E, 25.13--25.23 N (13 x 11 km). 49,804 building
  footprints and 4,334 road pieces (OSM, cache `35b8dd00fb6d3e44`); places of
  worship left out; the height cap raised to 900 m for supertall towers (1,328
  heights from tags, 580 from levels, the rest by the documented fallback).
- **Atmosphere:** `warm_coastal_desert` -- low late-afternoon sun from the
  west-south-west, golden light, sand haze, faint heat shimmer.
- **Cameras:** `urban-overview` (opening) looks south-west from above the high-rise
  core, 960 m up, with the shoreline on the right and the southern desert --
  where the fictional ambience originates -- on the left; `coastal-corridor` runs south-west along the
  shore; `street-canyon` sits 140 m above mid-rise streets (no tagged building
  within 400 m is taller than 20 m). All over the land-only safe region.
- **Ambience regions:** the inland dune belt south-east of the city and the open
  desert south of it -- coarse, hand-drawn open land.
- **Checkpoint:** `theatre_1000.pkl` was trained on `owens_valley`; every Dubai
  view says **ZERO-SHOT**. Zero-shot rollout (8 worlds x 4 aircraft, seed 999):
  0.438 of sorties survive against 0.188 for the direct route and 0.219 for the
  avoid+nap heuristic -- a small, untrained-on-Dubai sample, not a result. The
  replay draws world 1 of 8 (`--world 1`), a mixed outcome; the table covers all 8.

## Commands

```bash
uv run naigos-research --aoi dubai_urban
uv run python -m naigos.demo.urban --aoi dubai_urban

# Evidence-grade terrain/LOS viewer.
uv run python -m naigos.demo.live --aoi dubai_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual physics --camera urban-overview --open

# Dense city presentation; never LOS evidence.
uv run python -m naigos.demo.live --aoi dubai_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --open

# ... with the fictional ambience.
uv run python -m naigos.demo.live --aoi dubai_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --ambience conflict_ambience --visual-seed 7 --open

# A recording and its self-contained replay.
uv run python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl \
  --aoi dubai_urban --worlds 8 --world 1 --out runs/dubai_urban
uv run python -m naigos.demo.viewer runs/dubai_urban/demo.json --visual urban-presentation \
  --ambience conflict_ambience --visual-seed 7 --open
```

No Dubai training command is added: the offline and cloud candidate workflows
take `--aoi`, and a Dubai run through them would record its theatre in
`theatre.json`, which the viewer then reports as trained-on-theatre.

## Limitations

- Flat terrain: terrain masking is not a learnable mechanic here; the theatre
  exercises the curvature/refraction term and the presentation standard.
- The checkpoint is zero-shot; the numbers above are a 32-sortie sample, and the
  policy still loses aircraft to the ground here (6 terrain losses).
- OSM building heights are mostly the documented fallback (47,896 of 49,804).
- The provider path (Google Photorealistic 3D Tiles) needs a credential and was
  not exercised here; without one the local layer is drawn.
