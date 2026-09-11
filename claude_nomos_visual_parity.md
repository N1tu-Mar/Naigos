# Claude Code task — Nomos-style dense urban 3D visuals for Naigos

Implement a **visually obvious, dense urban 3D presentation mode** for Naigos that has the same visual character as the supplied Nomos San Francisco reference: oblique fly-through camera, satellite/aerial context, textured or photorealistic building massing, streets, and moving simulation entities clearly embedded in a city.

The initial target is the existing `tehran_basin` AOI, because it already contains an urban basin. Do not substitute San Francisco or another city. The imagery references are visual requirements only, not instructions to copy assets, data, source code, or proprietary map tiles from Nomos.

## Why the current result does not meet this bar

Naigos currently has a real 3D terrain surface from its own DEM and a Cesium viewer, but it has no dense building geometry in its physics mode; aircraft and threats have also been rendered as points. As a result, a top-down/large-AOI camera can read as a 2D tactical map. The optional Google Photorealistic 3D Tiles route is a provider context, not a demonstrated simulation surface, and may be unavailable without credentials.

This work must result in a user-observable change. Do not hand back only a plan, a screenshot of a third-party map, a new flag that does nothing, or documentation claiming visual fidelity without a working renderer.

## Required modes and honesty contract

Keep these two modes separate in code, UI, docs, and tests:

| Mode | What it is for | What may occlude simulation/LOS? |
| --- | --- | --- |
| `physics` (default) | Demonstrating the Naigos terrain/LOS simulation | Only the existing simulation DEM. Never buildings, trees, provider meshes, or imagery. |
| `urban-presentation` | Nomos-style cinematic situational awareness | Visual buildings and provider geometry are allowed, but the UI must label it presentation-only. They must not be passed into the environment or presented as LOS evidence. |

`physics` must remain the existing evidence-grade mode. If an overlay becomes visually hidden behind a presentation building, label that as render-only occlusion, not simulated cover. Do not solve the visual task by secretly changing the simulated terrain or detection model.

## Deliverable

Add an explicit `--visual urban-presentation` option to the live viewer and replay exporter. It must open at an oblique, low-enough overview of Tehran’s urban fabric and show:

- realistic satellite/aerial context where credentials allow;
- dense, geographically aligned 3D buildings, with roofs, believable height variation, and streets visibly running between them;
- the existing terrain relief visible beyond/around the city;
- actual simulation aircraft, ground threats, fixed sites, and airborne threats rendered as 3D model entities, using the work specified in `claude_3d_visual_fidelity.md` if it has not already been implemented;
- trajectories, objectives, tracks, detection/engagement overlays, and counters still accurately linked to the simulation;
- a clear HUD badge: `URBAN PRESENTATION — buildings and provider geometry are visual only; terrain LOS uses the Naigos DEM`.

The presentation should look like the reference in *composition and spatial depth*: a city you can fly through and above, not a 2D basemap with a few boxes. It does not need to replicate a specific proprietary photogrammetry source.

## Implementation requirements

### 1. Inspect and reuse the Nomos approach where legal and accessible

First inspect the current repository. If the sibling Nomos repository is available locally, inspect its renderer, asset provenance, data flow, and camera choices to learn the technique. Do not copy code blindly; do not import an asset whose licence is unknown; do not introduce a runtime dependency on the Nomos repository.

Document exactly which design ideas were reused versus newly implemented.

### 2. Urban geometry: robust provider path plus local fallback

Implement a provider-first presentation path using the project’s existing, credential-safe Google Photorealistic 3D Tiles / Cesium ion configuration when available. It should be the highest-fidelity option and retain provider-required attribution.

Also implement an independent cached local fallback so the core urban presentation does not collapse to a blank terrain view when provider credentials are absent:

- Add a **visual-only** source entry for OpenStreetMap building/road data, with correct ODbL attribution, explicit host allowlisting, caching, source URLs, hashes, and a short data-provenance document.
- Fetch only a bounded region corresponding to the selected AOI; never query open-ended locations or collect military/defence-location data.
- Produce a versioned, cached derived artifact for the browser: building footprint polygons, height/levels where tagged, and road geometry. It must be separate from `naigos/env` inputs.
- Render local buildings as real Cesium geometry (e.g. extruded polygon primitives/entities, batched where appropriate), ground-clamped to the visual terrain representation. Preserve footprint shape and tagged height. For genuinely missing height tags, use a deterministic, clearly documented presentation heuristic based on ordinary civilian building-type/area—not random height per frame and not a claim of surveyed truth.
- Render major roads visibly enough to establish urban structure. Do not build a duplicate traffic simulation.
- Use renderer LOD/culling/batching and a bounded viewport/AOI so the browser stays responsive. Publish a performance budget and show a concise diagnostic including building count, road count, tile/provider route, and fallback mode.

The fallback must be visibly city-like in Tehran: a dense field of varied extruded buildings and roads—not a dozen isolated boxes. It may be lower-fidelity than provider photogrammetry, but must still have a working oblique-city visual verification artifact.

### 3. Satellite imagery and credentials

- Retain Sentinel-2 through Cesium ion as the default aerial skin when `NAIGOS_CESIUM_ION_TOKEN`/`CESIUM_ION_TOKEN` is present. Retain keyless OSM fallback.
- Use `NAIGOS_GOOGLE_MAPS_API_KEY`/`GOOGLE_MAPS_API_KEY` or the existing Cesium ion path only through the established single secret-injection route. Do not add a CLI `--key` parameter or write tokens into a static replay artifact.
- Keep all required provider credits visible in the Cesium credit display and in the HUD where needed.
- State plainly that satellite pixels, provider mesh, and building extrusions are visual-only and never model inputs.

### 4. Camera, light, and readability

Create reusable camera presets for both live and replay:

- `urban-overview` — default in `urban-presentation`; an oblique, city-scale view that shows depth and terrain context;
- `street-canyon` — a close oblique fly-through composition for clear building scale, while avoiding clipping and retaining controls;
- `follow-aircraft` — tracks a real simulated aircraft while maintaining a useful look-ahead distance;
- `analysis-topdown` — the explicit tactical/map posture; never the default in urban-presentation.

Use physically coherent Cesium lighting, fog/atmosphere, shadows if performant, and level-of-detail settings. No fake screenshot background or pre-rendered video is acceptable. Avoid permanently hiding aircraft/overlays behind visual-only building geometry in a way a user would mistake for simulated terrain masking; either draw a clearly labelled render-only occlusion state or offer a presentation-building-occlusion toggle that defaults off.

### 5. Simulation integration without changing simulation claims

- Use only the existing environment’s positions, headings, altitude, state transitions, and event stream for entity placement and animation.
- Ground entities must be visually mounted on the presentation surface but retain their recorded/simulated coordinates. Aircraft must retain their actual simulated altitude—not be snapped to roofs or streets.
- Visual models, streets, buildings, satellite imagery, and optional effects must never feed observations, rewards, collision/LOS, policy actions, verifier outputs, or live run metrics.
- Show the active visual posture, terrain source, imagery source, geometry source, and `evidence_grade` in `/scene`, the HUD, `--smoke-render`, logs, and exported replay metadata.

### 6. Tests and acceptance evidence

Add offline tests that verify:

- `urban-presentation` is a valid and separately labelled visual mode;
- physics mode remains DEM-only for LOS and does not import visual building data under `naigos/env` or `naigos/rl`;
- visual OSM data uses an explicit allowlist, is cached/provenanced, is bounded to the AOI, and has attribution;
- derived building/road artifacts are deterministic and location-aligned;
- missing heights use the documented deterministic fallback;
- building/road entities are only built in urban-presentation;
- entity model coordinates and simulation coordinates remain consistent;
- provider credential values never appear in serialized scene data, static exports, or logs;
- UI/metadata correctly expose presentation-only status and the evidence warning;
- existing physics terrain, LOS, imagery, replay, and 3D-model tests remain green.

If browser access is available, perform a real visual acceptance check after implementation:

1. open `tehran_basin` in `urban-presentation` at `urban-overview`;
2. capture one screenshot showing dense buildings, streets, terrain, and simulation entities;
3. open physics mode at the same approximate camera framing and capture a second screenshot showing that its terrain remains the actual DEM;
4. include paths to the generated artifacts in the handoff and state which provider/fallback supplied the buildings.

Do not claim this check happened if browser inspection was unavailable. A DOM/unit test is not a visual acceptance check.

## Suggested verification commands

Keep commands aligned with the implementation, but the finished README must include an equivalent to:

```bash
# Provider-enhanced, presentation-only urban scene
export NAIGOS_CESIUM_ION_TOKEN=<your-token>
uv run python -m naigos.demo.live \
  --aoi tehran_basin \
  --checkpoint checkpoints/theatre_1000.pkl \
  --visual urban-presentation \
  --camera urban-overview \
  --open

# Evidence-grade simulation DEM/LOS view
uv run python -m naigos.demo.live \
  --aoi tehran_basin \
  --checkpoint checkpoints/theatre_1000.pkl \
  --visual physics \
  --camera urban-overview \
  --open
```

## Handoff requirements

Report changed files, visual data/asset licences, exact run commands, tests, browser acceptance evidence, fallback behavior, runtime performance observations, and any constraints remaining. Be exact about the boundary: the city visuals make the simulation legible and cinematic; they do not model buildings as tactical cover unless and until the physics environment explicitly and separately adds that capability.
