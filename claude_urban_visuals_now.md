# Claude Code — implement the remaining Nomos-style urban visual gap now

Work in the current Naigos repository. This is a focused implementation task, not a fresh redesign. The goal is a **working, visibly dense 3D Tehran urban scene** with the same cinematic spatial feeling as the supplied Nomos city reference, while retaining Naigos’s simulation-truth boundaries.

## Current code facts — verify these before changing anything

Do not repeat already completed work. The current branch already contains:

- `naigos/demo/assets/cesium.html`: one shared Cesium renderer for live and replay;
- `naigos/demo/live.py`: schema v2 scene/frame payloads, recorded attitude, threat state, visual events, and `/models/` serving;
- `naigos/demo/models.py` and `naigos/demo/assets/models/`: locally bundled CC0 GLB assets and a model registry;
- `naigos/demo/events.py`: presentation-only events derived from existing simulation state;
- `naigos/demo/imagery.py`: exactly two visual modes today: `physics` and credentialed `photorealistic`;
- `physics`: the only evidence-grade visual mode, using `CustomHeightmapTerrainProvider` from the simulation’s own DEM;
- `photorealistic`: Google 3D Tiles when credentials are present, intentionally labelled non-evidence because it brings provider terrain/buildings;
- existing tests including `test_visual_fidelity.py`, `test_model_assets.py`, `test_visual_renderer.py`, `test_visual_modes.py`, terrain/LOS/replay tests.

Run the current focused tests first. If an assertion above is inaccurate, report the mismatch and adapt to the actual code rather than reverting completed renderer/model/event work.

## The exact remaining problem

The current app does not have a first-class, credential-independent **urban city presentation** for `tehran_basin`. Without Google/Cesium provider credentials, it falls back to imagery over the DEM, so the city reads as terrain/map rather than the dense 3D streets-and-buildings scene in the Nomos reference. The current `photorealistic` path also needs an explicit, inspectable proof that it actually loaded provider geometry rather than silently falling back.

## Deliverable in this task

Add a third visual mode named exactly `urban-presentation`.

It must work in both live server mode and exported static replay, and must have two resolution paths:

1. **Provider-enhanced path:** use the existing Google Photorealistic 3D Tiles mechanism when configured. Preserve attribution and the current non-evidence warning. Add a clear runtime status: `provider buildings active` only after the tileset actually becomes ready/visible; never claim it merely because a token exists.
2. **Local cached fallback path:** when the provider route is unavailable, render a dense, locally cached Tehran building-and-road layer using OpenStreetMap-derived civilian data. It must be a real Cesium 3D geometry layer, not a background screenshot or a handful of decorative blocks.

The mode must default to an oblique city view where the user can immediately recognize rooflines, streets, height variation, terrain beyond the basin, and Naigos’s already-existing 3D simulation entities. It is presentation-only: render-only building occlusion must default **off**, and the HUD must say that building geometry is not used by terrain LOS.

## Implement in this order

### A. Baseline and actual provider verification

1. Run the focused existing viewer/model tests before editing.
2. Run a normal `tehran_basin` physics viewer once, using the existing documented command.
3. If environment credentials are available, run `--visual photorealistic` and inspect the browser. Verify through a renderer state, not a token check, whether the Google tileset loaded and added geometry. Capture/record this as a manual check only if it actually happened.
4. Do not require the contributor to own credentials or spend money. Automated tests must use mocked state and the local fallback.

### B. Visual-mode contract

Extend `naigos/demo/imagery.py` carefully:

- Include `urban-presentation` in `VISUAL_MODES`; keep `physics` the default.
- Do not replace or weaken the existing `physics` and `photorealistic` contracts.
- Add explicit fields to the credential-free visual scene/config payload: `geometry_source` (`provider_3d_tiles` or `local_osm_extrusions`), `presentation_only: true`, provider readiness/state, and a human-readable fallback reason where applicable.
- Update argument parsing, validation, HUD, scene metadata, static export, README, and `docs/STACK.md` so they all recognize the new mode.
- A static replay must contain enough local urban geometry data to render offline. It must not contain any secret/token or make an external OSM/Overpass request at view time.

### C. Local urban data pipeline (bounded, visual-only, cached)

Implement a small module such as `naigos/demo/urban.py` plus a CLI entry point. Keep this out of `naigos/env/` and `naigos/rl/`.

1. Add an explicit **visual-only** OSM/Overpass source entry to `naigos/research/allowlist.py`, with its correct licence, attribution, allowed HTTPS hosts, and clear role: Tehran civilian building footprints and roads for presentation only.
2. Add a bounded fetch/build command, for example:

   ```bash
   uv run python -m naigos.demo.urban --aoi tehran_basin
   ```

   It must query only the existing AOI bounds (or a documented smaller urban sub-bound), have a sensible response-size/timeout limit, and cache raw response plus a derived artifact under a dedicated visual-cache namespace. It must not fetch force dispositions, named military facilities, weapon data, or any unrelated geography.
3. Reuse the project’s provenance style: cache raw bytes, source URL/query digest, fetch time, sha256, licence, and attribution. Make repeated runs idempotent/offline after the first successful fetch.
4. Derive a versioned browser payload with only what the renderer needs:
   - building exterior rings in WGS84;
   - a deterministic height in metres: use valid `height` first, then `building:levels * documented floor height`, then a deterministic ordinary-civilian fallback based on tag/footprint area;
   - major road centerlines;
   - no raw tags or unrelated OSM attributes passed to the browser.
5. Reject malformed/self-intersecting/out-of-bounds geometry and clearly report empty data rather than rendering an unlabeled empty layer.

Do not make network availability a requirement of starting `naigos.demo.live`. If the local derived artifact is absent, `urban-presentation` must start in a labelled `urban data unavailable` state and offer the existing provider route if credentials permit; it must never pretend buildings are present.

### D. Cesium rendering and performance

In the existing shared renderer, add an urban layer only when `visual.mode === "urban-presentation"`:

- Convert each cached footprint to an extruded Cesium polygon or batched primitive, with base height from the visual terrain position and extrusion to the deterministic building height. Do not use sea level.
- Render roads as a low-profile, performant geometry/material that makes the street grid clear without obscuring satellite imagery.
- Use progressive loading/chunking, distance culling, and/or tiles/chunks. Set and document an initial target appropriate for a normal laptop browser (e.g. cap initial visible buildings; add more with camera proximity). Do not create tens of thousands of entities in one blocking synchronous loop.
- Give local extrusions neutral, non-tactical materials. Satellite imagery remains the source of photoreal surface detail; local buildings provide spatial form.
- Keep the terrain DEM and current GLB aircraft/threat models/effects. Do not regress their orientation, event timing, depth-testing in physics mode, or static replay behavior.
- Add city camera presets in the existing camera module/renderer: `urban-overview` (default for this mode), `street-canyon`, `follow-aircraft`, and `analysis-topdown`. `urban-overview` must be oblique, not the current map-style rectangle fly-to.
- In `urban-presentation`, visual-building depth testing/occlusion must be off by default or clearly labelled/toggleable. No user should infer that a building blocked radar LOS.

### E. A narrow, testable acceptance bar

Add tests that run without network, browser, credentials, GPU, or Modal:

- local default behavior remains unchanged in `physics` and `photorealistic` modes;
- `urban-presentation` is accepted and exposes its presentation-only/evidence metadata;
- OSM visual source is explicitly allowlisted and its derived data is bounded to `tehran_basin`;
- derived building height choices are deterministic and correctly prefer explicit height, then levels, then documented fallback;
- invalid/out-of-bounds geometry is rejected;
- cached payload includes attribution/provenance but no secret or arbitrary raw OSM tags;
- local building geometry is used only in `urban-presentation`, never imported into `naigos/env` or `naigos/rl`;
- provider state does not report active until a mocked ready/visible state occurs;
- missing local data/provider credentials produces an honest HUD/scene fallback state;
- static replay embeds the local urban payload and does not perform browser-time data fetches;
- model/event/terrain/LOS/replay tests stay green.

Add a `--smoke-render` diagnostic report containing: active visual mode, evidence grade, imagery source, requested vs actual geometry source, provider readiness, local cache ID/hash, building/road counts, camera preset, model fallback count, and render-only-building-occlusion state. It must describe configuration, not claim pixels were rendered.

## Explicitly do not do these things

- Do not alter the RL environment, observation, rewards, detection model, LOS, verifier, or blue’s evasive-only action space.
- Do not claim OSM buildings are terrain, radar cover, or tactical truth.
- Do not use an open-ended web search, an unbounded Overpass query, an undocumented asset, a third-party model CDN, an API key in an argument, or a screenshot/video as the implementation.
- Do not rebuild current GLB models, attitude handling, effects, Modal pipeline, or photorealistic tiles integration unless a focused bug blocks this task.
- Do not make the presentation mode the default. `physics` stays default and evidence-grade.

## Handoff

Run the focused tests, then full `pytest`. If possible, open `tehran_basin` in both modes and inspect screenshots; state exactly which path rendered buildings (provider vs local fallback). Report changed files, OSM/provider attribution, cache setup command, exact live/replay commands, performance numbers or observations, and any unfulfilled acceptance item.

The two commands users should be able to run after implementation are:

```bash
# One-time local visual data build; no influence on simulation physics
uv run python -m naigos.demo.urban --aoi tehran_basin

# Dense-city presentation mode
uv run python -m naigos.demo.live --aoi tehran_basin \
  --checkpoint checkpoints/theatre_1000.pkl \
  --visual urban-presentation --camera urban-overview --open

# The same terrain as the physics model; use for LOS evidence
uv run python -m naigos.demo.live --aoi tehran_basin \
  --checkpoint checkpoints/theatre_1000.pkl \
  --visual physics --camera urban-overview --open
```
