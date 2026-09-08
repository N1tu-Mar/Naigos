# components/

One cited JSON per design decision. The env reads these; it never reads `data_cache/` directly.

Each file carries:

| field | meaning |
| --- | --- |
| `id` | component identifier, matching the filename |
| `role` | what this component is for, in one line |
| `inputs` / `outputs` | what it consumes and what it produces |
| `decision` | the design decision this component settles |
| `rationale` | why that decision, including what was rejected |
| `parameters` | the values the env consumes |
| `evidence` | measurements supporting the decision |
| `invariants` | properties that must hold, asserted in `tests/` |
| `caveats` | what this data does *not* establish |
| `license` / `sources` | provenance; a component with no source is refused at write time |
| `cached_artifacts` | path, size and sha256 of the raw bytes behind it |
| `generated_at` | UTC timestamp of emission |

Regenerate with `naigos-research`. Idempotent: a second run makes no network calls.

## Current components

| id | settles |
| --- | --- |
| `env.aoi` | which theatre, and the evidence that terrain masking is learnable there |
| `data.terrain_dem` | the elevation grid and how line-of-sight is computed on it |
| `data.airfields` | where sorties start and what they are tasked against |
| `data.atmosphere` | air density vs altitude, winds aloft, radar refraction factor |
| `data.flight_envelope` | measured airframe speed, climb and turn limits |
| `model.detection` | detection probability and lethal envelopes, derived from the range equation |
| `research.agent` | the data layer's own contract: what may be fetched, and what may not |
| `demo.imagery` | the globe's visual skin (Sentinel-2 via Cesium ion) and why it is kept apart from the DEM |
