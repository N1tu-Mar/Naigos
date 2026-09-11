# tehran_basin -- presentation notes

The research box, its rationale and its DEM are unchanged from the built-in
theatre (`naigos/research/aoi.py`, `components/aoi/tehran_basin/`, `docs/DATA.md`).
This file records only how the city standard presents it
(`naigos/demo/cities/tehran_basin.json`).

- **Urban box:** 51.30--51.50 E, 35.66--35.81 N, the box the local city cache was
  first built for; the query and its cache are byte-identical under the
  standard.
- **Protected zones:** none declared (a built-in AOI).
- **Atmosphere:** `high_basin_clear` -- crisp light from the south-east so the
  ridge behind the city is side-lit, light blue haze.
- **Cameras:** `urban-overview` looks north-north-east from inside the city up to
  the Alborz at -12 degrees and 4.2 km, so rooflines, streets and the ridge
  horizon share the frame; `street-canyon` is 180 m above the street.
- **Ambience regions:** the high range north of the basin, and the open plains
  south, west and east of the city -- coarse hand-drawn open land, far from the
  urban box. No place, facility or road was consulted.
- **Checkpoint:** the shipped `theatre_1000.pkl` was trained on `owens_valley`;
  every Tehran view says **zero-shot**.
