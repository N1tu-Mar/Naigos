# Claude Code — add the Dubai urban theatre and the cross-city contested-airspace presentation standard

Work in the current Naigos repository. Implement this fully; do not merely write a plan. The result is a new, reproducible **Dubai** AOI which runs the same Naigos evasive-flight RL environment and shows a dense, cinematic but clearly notional urban contested-airspace scene.

This is a simulation/presentation task, not a reconstruction of a real conflict. Blue aircraft stay strictly evasive; red entities remain generic, randomly spawned simulated hazards. Do not collect, infer, encode, or depict real force locations, current military activity, weapons performance, facilities, routes, targets, or personnel.

## Establish the baseline first

Before editing, inspect and run the focused tests for the current code. Verify the actual state rather than assuming an earlier prompt was completed:

- `components/aoi/tehran_basin/` is the component/provenance format to follow;
- `naigos/env/theatre_bridge.py`, the research pipeline, component schemas, guards, and verifier define how an AOI becomes an environment;
- `naigos/demo/live.py`, `assets/cesium.html`, `camera.py`, `imagery.py`, models, and events define the shared live/replay renderer;
- `physics` is evidence-grade because it renders the simulation DEM; any provider 3D tiles or local building extrusions are explicitly presentation-only;
- the existing GLB aircraft, generic ground/mobile threat, airborne threat models, attitudes, tracks, effects, and static replay must be reused, not replaced;
- the current visual modes and the current status of `urban-presentation` (if present) are the source of truth. Adapt to the code you find.

Do not alter old AOIs, existing snapshots, or a trained checkpoint to force them to work with Dubai.

## 1. Create a reproducible `dubai_urban` AOI

Create `components/aoi/dubai_urban/` using the project’s normal AOI/research/component workflow. It must be a modest, documented **civilian urban-and-coastal test envelope** around Dubai—not an open-ended city scrape and not an airport, port, or sensitive-site study. Choose and document a bounded rectangular AOI with enough coastline, dense ordinary urban fabric, and height variation to make flight visualization legible. Keep its exact bounds in the AOI component/provenance metadata.

- Obtain DEM, atmosphere, civil-airfield/start geometry, and generic flight-envelope inputs only through the existing allowlisted, cache/provenance-controlled pipeline. Add a source only when it is essential, correctly licensed, bounded, and has an explicit non-sensitive role.
- Preserve the component hashes, raw-cache/provenance records, AOI fingerprint checks, and `naigos-research --aoi dubai_urban` idempotence behaviour. A second run must use the cache.
- Add a small, stable landmark-elevation/georeference sanity check from a non-sensitive public terrain reference, following the existing test style; do not use a military facility as a reference point.
- Do not embed API keys, request current conflict data, scrape map search, or make runtime network access mandatory. Tests must run from checked-in fixtures/caches or mocks.
- Update theatre listings and docs with a plain description of the terrain/data source, bounds policy, and the fact that this is a generic simulation theatre—not a digital twin or a claim about real conditions.

## 2. Use the exact same simulation system, without making it operational

`dubai_urban` must run through the normal `env_from_theatre`/training/demo paths. Keep the existing generic threat kinds and parameter ranges. The screen must make their roles readable—generic fixed ground sensor/emitter, generic mobile ground sensor/emitter, and generic airborne adversary—without using names, liveries, values, silhouettes, or placement intended to represent any real military platform.

Implement sensible theatre configuration only where the existing environment already supports it:

- Blue has the existing evasive-only action space. No firing, targeting, selection, weapon release, or attack planning controls may be added.
- Red ground and air hazards are procedural/random per seed and reroll; never place them from live data or at named real-world locations. Seed/replay metadata must say they are notional.
- Existing LOS/detection/kill math must continue to consume only the simulation terrain and generic configs. Local city buildings, smoke, haze, imagery, vehicle visuals, and provider geometry must never enter LOS, observations, rewards, training data, or verification.
- Do not make a zero-shot checkpoint look trained on Dubai. Update labelling so a Dubai run visibly says whether the chosen checkpoint was trained for this AOI. Add a Dubai training command/path only if it follows the existing offline/cloud candidate workflow and preserves run provenance.

## 3. Dubai’s urban scene: real spatial form, not a 2D map

Make `--visual urban-presentation` (or the current equivalent if that feature has not landed—implement it first rather than inventing a fourth overlapping mode) work for `dubai_urban` in live and exported replay.

The desired look is comparable to the Nomos reference: an oblique, flyable, dense 3D city with visible streets, varied rooflines, coast/water, terrain horizon, and properly oriented aircraft/threat models. It must be visibly a 3D scene, not a flat map with markers.

1. When Google Photorealistic 3D Tiles is legitimately configured through the existing credential-safe path, use it as an optional provider-enhanced presentation. Keep mandatory attribution and report `provider buildings active` only after renderer readiness/visibility—not merely when a token exists.
2. Without provider credentials, use the existing bounded/cache-first OSM visual pipeline for actual extruded ordinary civilian building footprints and low-profile road geometry. If the Tehran local pipeline exists, generalize it by AOI; do not duplicate it. Payloads embedded in static replay must be bounded, attribution/provenance-bearing, and contain no raw tags or secrets.
3. Preserve the simulation DEM underneath every physics visualization. In urban-presentation, label the city layer `PRESENTATION ONLY — building geometry is not used by terrain LOS`; render-only building occlusion defaults off or is unmistakably toggleable.
4. Add/update city camera presets for the local geography: `urban-overview` should be the opening oblique view; `coastal-corridor`, `street-canyon`, `follow-aircraft`, and `analysis-topdown` must not accidentally put the camera below ground or offshore. Apply the same camera-contract improvement to Tehran and Mecca once their AOIs exist.
5. Use batching/chunking, progressive load, distance culling, and reasonable caps. Do not synchronously create an unbounded entity per building. The smoke-render report must include geometry source, actual provider readiness, cache hash, building/road counts, camera preset, model fallback count, and presentation/evidence state.

## 4. A shared atmosphere and sustained fictional-conflict ambience contract for Tehran, Dubai, and Mecca

Create reusable, deterministic **presentation-only** atmosphere profiles, selected from an explicit local allowlist/config rather than random browser effects. Apply the same system to `tehran_basin` now; make `dubai_urban` use a warm coastal/desert profile; prepare it so `mecca_urban` can use a hot, dusty inland profile when that AOI is added.

The goal is a credible fictional training-simulator mood, not a claim that any city is presently a warzone. Add a first-class, opt-in `conflict_ambience` presentation profile with a sustained setting so the scene can carry recurring fictional distant impacts throughout a replay/live session:

- daylight/time-of-day, neutral sky, sun direction, terrain-aware haze/dust, subtle heat shimmer if performant, and bounded event-only smoke/dust can add scale and tension;
- the sustained profile may produce recurring **fictional** distant blast flashes, rising smoke plumes, dust bursts, rumble/siren-style ambience if audio already exists, and generic non-graphic debris/spark effects. It is an art-directed VFX stream, not a munition, strike, projectile, or damage model;
- generate that VFX stream deterministically from a visual seed and a declared rate/intensity profile. It must be replayable, pausable, reduced-motion/performance-aware, and exportable with the replay; a random browser timer is not acceptable;
- choose effect origins from a synthetic, presentation-only scene mask—not map search, real events, named buildings, facilities, roads, or live data. Keep all origins away from aircraft/threats and away from protected/excluded areas. Effects must never identify what was hit or assert that anything was hit;
- effects may be derived from already recorded/simulated generic state or from the separately recorded synthetic ambience stream; both paths must be visibly labelled simulated and neither may influence simulation state;
- do not render civilians, real damage, casualties, flags, religious/political symbols, current news imagery, or identifiable real conflict footage;
- no smoke/fire/atmosphere effect may conceal entities, alter visibility/detection/LOS, or make a generic simulation assertion about actual conditions;
- honour reduced-motion/performance mode and provide deterministic seed/time inputs so replay frames are stable.

Add an obvious HUD field: `scenario: notional contested-airspace simulation`; retain the visual evidence-grade label and source attribution. The visual tone may be dramatic, but never imply that Dubai, Tehran, or Mecca is an active real battlefield.

## 5. Verification and handoff

Add focused, offline tests for:

- the `dubai_urban` component schema, AOI fingerprint, provenance, and cached rerun behaviour;
- generic randomized ground/air threat configuration and blue evasive-only invariants;
- no city visual/atmospheric code imported by `naigos/env` or `naigos/rl`;
- physics mode retaining simulation DEM/evidence-grade behaviour;
- provider readiness and honest fallback state;
- bounded urban payload, deterministic building heights, attribution, static replay embedding, and no browser-time data fetch;
- city camera presets and deterministic atmosphere profile selection;
- deterministic sustained fictional-conflict ambience, including rate bounds, replay stability, protected-mask exclusion, and proof that it cannot modify simulation state or the LOS/detection path;
- non-sensitive/notional warning appearing in scene metadata/HUD/export;
- existing terrain, LOS, renderer, model, event, replay, demo-isolation, and guardrail tests remaining green.

Run focused tests, then full `pytest`. Perform `--smoke-render` in physics and urban-presentation modes. If a browser is available, inspect a Dubai live run and a static replay; state which geometry path actually rendered (local cache vs provider) and do not claim a visual check you did not perform.

Document the exact commands. The intended shape is:

```bash
# Build/reuse normal research components; credentials never go in arguments.
uv run naigos-research --aoi dubai_urban

# Build/reuse the bounded visual-only urban cache, if this CLI exists.
uv run python -m naigos.demo.urban --aoi dubai_urban

# Evidence-grade terrain/LOS viewer.
uv run python -m naigos.demo.live --aoi dubai_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual physics --camera urban-overview --open

# Dense city presentation; never treat this as LOS evidence.
uv run python -m naigos.demo.live --aoi dubai_urban \
  --checkpoint checkpoints/theatre_1000.pkl --visual urban-presentation \
  --camera urban-overview --open
```

In the final handoff list changed files, exact AOI bounds policy, source licences/attribution, commands run, test results, actual geometry status, performance observations, and remaining limitations. Do not hand off a placeholder AOI, a 2D map, or a prompt-only change.
