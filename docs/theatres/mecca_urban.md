# mecca_urban

A **notional contested-airspace simulation** over the rugged relief and outer
districts west of Mecca. It is a generic simulation theatre on real terrain, not
a digital twin; it does not depict, predict or facilitate any real conflict and
makes no claim about any real force, facility, security arrangement, event or
current condition. Blue is evasive only; red ground and air entities are the
project's generic hazards, drawn at random per seed and re-rolled -- never placed
from real-world data or at named locations. There is no attack objective over the
city: waypoints are anonymous coordinates drawn by the theatre generator.

## Bounds and exclusion policy

`naigos/research/aois/mecca_urban.json`: **39.30--39.78 E, 21.15--21.68 N**
(about 50 x 59 km, UTM 37N / EPSG:32637, fingerprint `d0b3750c15`).

The box is placed entirely **west** of the historic centre, reaching into the
western foothills and wadis. (A first, 34 x 36 km box sat inside most generic
lethal envelopes and every recorded aircraft was lost within ~30 s; it was
widened west, north and south -- never east -- until flights stay legible.) Its east edge is set
so the **Masjid al-Haram complex and its immediate precinct lie outside the AOI
by more than 3 km**: nothing in the simulation can be placed there (entities live
in the AOI's own grid), no terrain or visual data is extracted there, and no
camera or effect can use it. The pilgrimage venues to the east (Mina,
Muzdalifah, Arafat) and Jabal al-Nour lie further outside still.

Four protected zones, each carrying every policy -- `extraction`, `airfield`,
`camera`, `ambience`, `render_cutout` -- with a 1--2 km buffer:

| zone | why |
| --- | --- |
| Masjid al-Haram complex and immediate precinct | religious importance |
| Masjid Aisha (Tan'eem) | place of religious significance near the box edge |
| Jabal al-Nour | place of religious significance |
| pilgrimage venues (Mina, Muzdalifah, Arafat) | pilgrimage venues |

What that means in practice:

- **extraction** -- the city layer keeps no footprint with a vertex in any zone
  and cuts roads at its edge (none reached the zones: the urban box is outside
  every buffer);
- **camera** -- every preset stands more than 3 km from the precinct buffer and
  faces more than 60 deg away from it; the page keeps the free camera out of the
  zones and turns the chase camera away from them;
- **render_cutout** -- the page covers every zone with a plain neutral patch over
  the imagery and clips it out of provider 3D tiles, so it is never drawn;
- **ambience** -- no effect originates within 8 km of the precinct;
- **airfield** -- no start point inside (there is no airfield in the box at all).

No religious site, pilgrimage venue, hospital, school or other sensitive civilian
location is a start point, waypoint, camera focus or backdrop. No crowds,
civilians or evacuations are modelled or drawn. Places of worship are left out of
the city layer entirely.

## Data, licences, attribution

Allowlisted, cached, cited (`uv run naigos-research --aoi mecca_urban`; a second
run makes no network call, verified with every proxy pointed at a dead port).
Provenance: [`mecca_urban/DATA.md`](mecca_urban/DATA.md); components in
`components/aoi/mecca_urban/`.

| input | source | licence |
| --- | --- | --- |
| terrain | Copernicus DEM GLO-30 (AWS open-data COG) | Copernicus DEM licence; attribution required |
| atmosphere | Open-Meteo forecast at the AOI centre | CC BY 4.0 |
| airfields | OurAirports (none inside the box) | public domain |
| flight envelope | OpenSky (shared calibration sample) | OpenSky terms, non-commercial |
| city layer (visual only) | OpenStreetMap via Overpass, `naigos.demo.urban` | ODbL 1.0 -- (c) OpenStreetMap contributors |

Measured: 9--765 m (756 m relief), mean slope 5.6 deg; 16.1% of points masked at
100 m AGL from the high point, 6.0% at 300 m -- many short ridges, masking close
to the aircraft. Refraction factor k = 1.674. Georeference check: Open-Meteo's own
terrain height at the requested point agrees with the viewer's DEM within 40 m, and
a perturbed UTM zone is caught. (Open-Meteo first answered this request with an
error text served as HTTP 200; the research cache now refuses such a body and
retries rather than caching it.)

## Presentation

`naigos/demo/cities/mecca_urban.json`:

- **Urban box:** 39.60--39.77 E, 21.30--21.50 N (unchanged by the widening). OpenStreetMap maps few building
  footprints in these districts: the layer has **931 buildings** and **3,402 road
  pieces** (residential streets included, because the street grid is mapped where
  the buildings are not; cache `5b90972f2ad5030a`). That is what the data holds,
  drawn honestly -- not padded with invented buildings. The provider path (Google
  Photorealistic 3D Tiles, with every zone clipped out) gives a dense city when a
  credential is present.
- **Atmosphere:** `hot_dusty_inland` -- high hard sun from the south-south-west,
  bleached sky, mild ochre dust that deepens the relief. Restrained.
- **Cameras:** `urban-overview` (opening) looks west-north-west across the most
  densely mapped outer district to the hills; `valley-overview` looks across the western relief from
  2.7 km; `street-canyon` sits among the outer streets.
- **Ambience regions:** the rugged hills west of the districts and the ridges to
  the south -- the far side of the city from the historic centre.
- **Checkpoint:** `theatre_1000.pkl` was trained on `owens_valley`; every Mecca
  view says **ZERO-SHOT**. Zero-shot rollout (8 worlds x 4 aircraft, seed 999):
  0.344 of sorties survive against 0.219 for the direct route and 0.375 for the
  avoid+nap heuristic -- a small sample, not a result; the policy loses 12
  aircraft to the ground in relief it never trained on. The replay draws world 1
  of 8 (`--world 1`), a mixed outcome; the table covers all 8.

## Commands

```bash
uv run naigos-research --aoi mecca_urban
uv run python -m naigos.demo.urban --aoi mecca_urban

# Evidence-grade model terrain/LOS
uv run python -m naigos.demo.live --aoi mecca_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual physics --camera valley-overview --open

# Presentation-only dense city scene
uv run python -m naigos.demo.live --aoi mecca_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --open

# ... with the restrained fictional ambience
uv run python -m naigos.demo.live --aoi mecca_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --ambience conflict_ambience --ambience-setting sparse --visual-seed 7 --open

# A recording and its self-contained replay
uv run python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl \
  --aoi mecca_urban --worlds 8 --world 1 --out runs/mecca_urban
uv run python -m naigos.demo.viewer runs/mecca_urban/demo.json --visual urban-presentation --open
```

## Limitations

- Sparse OSM building coverage (931 footprints); heights are the documented
  fallback for all but one.
- The checkpoint is zero-shot; the numbers above are a 32-sortie sample.
- The provider path was not exercised here (no credential).
