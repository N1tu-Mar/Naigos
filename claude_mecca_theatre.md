# Claude Code — add the Mecca urban theatre using the established Naigos city standard

Work in the current Naigos repository. Implement this fully, after reading the existing Dubai/Tehran city work and the current code. The goal is a new, reproducible **Mecca-area** AOI that uses the same Naigos RL evasive-flight system and a dense 3D presentation—but only as a **notional, generic contested-airspace simulation**.

This must not depict, predict, or facilitate a real conflict in Saudi Arabia. Blue aircraft stay evasive only. Ground and airborne red entities remain generic, notional, procedurally generated hazards. Do not collect or encode real military locations, force dispositions, weapons, targeting data, current conditions, security arrangements, routes, or people.

## Read this before changing anything

The expected implementation order is:

1. `claude_urban_visuals_now.md` establishes local/provider urban rendering and its physics-vs-presentation boundary.
2. `claude_dubai_theatre.md` establishes the reusable city/AOI/atmosphere standard.
3. This task adds `mecca_urban` by reusing that standard, never by forking or duplicating the renderer/pipeline.

Those files are requirements only where they match the actual repository. Inspect the current branch and test suite first. If a prerequisite has not landed, implement the smallest shared abstraction needed, then use it for Mecca; do not build a parallel one-off system.

Existing invariants that must survive:

- `physics` is the default and the sole evidence-grade viewer mode: the visible terrain is the same simulation DEM used for LOS.
- Provider 3D tiles and OSM building/road extrusions are visual-only. They must not affect environment state, LOS, detection, rewards, observations, training data, or verifier results.
- Current GLB aircraft and generic threats, model attitudes, tracks, effects, replay, and camera infrastructure are shared across AOIs.
- The guardrail/allowlist/provenance structure permits bounded public civilian terrain/visual data, not unrestricted web research or sensitive data.

## 1. Create `mecca_urban` with a respectful, bounded scope

Create `components/aoi/mecca_urban/` through the project’s normal research/component system. Use a modest, documented **outer urban and surrounding-relief envelope** appropriate for generic terrain-following simulation and city visualization.

Respect the city’s religious importance:

- Exclude the Grand Mosque / Masjid al-Haram complex and its immediate precinct from the AOI and from visual-data extraction. Do not use it as a landmark, objective, launch point, threat location, camera focus, or visual backdrop.
- Use no religious sites, pilgrimage venues, hospitals, schools, or other sensitive civilian locations as scenario anchors. Do not model crowds, civilians, or evacuations.
- The presentation must have no attack objective over the city. Use generic non-target waypoints at anonymous coordinates chosen by the theatre generator, and describe them as simulation-only.

Obtain the DEM, generic atmosphere, and any civil-airfield/start inputs through the existing allowlisted, cached, provenance-checked sources. Give the new AOI an explicit bounds/fingerprint and preserve idempotent `naigos-research --aoi mecca_urban` behaviour. Add a non-sensitive public topographic/georeference sanity check in the same pattern as other AOIs. Never require network access at viewer or test time, and never put credentials in code, recordings, or CLI arguments.

## 2. Reuse the exact generic aircraft-versus-threat system

Run Mecca through normal `env_from_theatre`, RL, rollout, record, and demo paths. Reuse—not re-specify—the generic simulated entity schema:

- blue: existing evasive-only aircraft agents; no weapon or targeting action space;
- red ground: existing generic fixed/mobile ground sensor-emitter models, visualized as non-identifying equipment;
- red air: existing generic airborne sensor/emitter adversaries, visualized with the existing non-identifying aircraft model;
- layouts: procedural random draws and rerolls, keyed by seed and logged as notional; never read from real-world placement data.

Keep the generic, deliberately non-operational parameter ranges. Do not add real platform names, mission profiles, attack logic, current capabilities, loadouts, response procedures, or named-site placement. The visible status/metadata must say `notional contested-airspace simulation` and state whether a checkpoint is zero-shot or trained for `mecca_urban`.

## 3. Deliver a real 3D city scene, including offline fallback

Use the shared `urban-presentation` implementation for `mecca_urban` in both live viewer and static replay:

- On a provider-enabled machine, use the existing credential-safe Google Photorealistic 3D Tiles route only when actual tileset readiness/visibility proves it loaded; retain provider attribution and the non-evidence warning.
- Without credentials, use the generalized bounded/cache-first OSM pipeline to construct actual local civilian building extrusions and road geometry. Ensure extraction excludes the protected precinct above. Embed only the derived bounded, attribution/provenance-bearing payload in offline static replay—no raw OSM tags, secrets, or browser-time data requests.
- The `urban-overview` camera must be oblique and frame ordinary urban rooflines, streets, the surrounding rugged terrain, and live simulation models. Add geography-safe `valley-overview`, `street-canyon`, `follow-aircraft`, and `analysis-topdown` presets, clamping all cameras away from excluded precinct bounds.
- Use terrain-relative extrusion bases, progressive batching/chunking, culling, and a laptop-appropriate initial cap. A missing visual cache must produce an honest labelled fallback, never a fake city.
- Keep renderer building occlusion off by default, or clearly toggleable and labelled. City geometry is never radar/LOS cover in Naigos.

## 4. Apply the shared atmosphere contract—sustained fictional conflict ambience, never a claim of war

Extend the reusable presentation-only atmosphere profiles across `tehran_basin`, `dubai_urban`, and `mecca_urban`. Include the shared opt-in `conflict_ambience` profile with a sustained setting: recurrent **fictional**, non-graphic distant blast flashes, smoke columns, dust bursts, and generic VFX that make a replay feel like a high-intensity simulated conflict scene. It is art-directed ambience only—not a munition, projectile, strike, target, or damage model.

For Mecca use a restrained, deterministic hot/dry mountain-basin profile: sun/sky, mild dust haze, relief depth, and bounded generic event effects. Generate all sustained-ambience origins from a synthetic visual mask and seed/rate configuration, not from map search, current news, named sites, real incidents, or live data. Keep effect origins away from all excluded/protected precincts, simulation entities, and ordinary named locations. Export the seed/profile/event stream for static replay; respect pause/reduced-motion/performance settings; and never modify simulation state, visibility, LOS, detection, rewards, or training.

Do not render current news scenes, real damage, people, crowds, casualties, religious/political symbols, flags, or imagery that implies real events. Never call the city an active battleground. The desired atmosphere is a polished fictional training-simulator scene, always labelled as such.

## 5. Testing, documentation, and required evidence

Add offline, deterministic tests for:

- `mecca_urban` component validity, AOI fingerprint/provenance, cache rerun, and excluded-precinct geometry filter;
- the generic/procedural red ground and air entities and blue evasive-only invariant;
- generic simulation labels plus zero-shot/trained-on-theatre disclosure;
- no presentation module dependency from `naigos/env` or `naigos/rl`;
- `physics` remaining the simulation-DEM/evidence-grade path;
- urban local payload bounds, exclusion filter, height determinism, attribution, static replay embedding, and honest provider/cache fallback state;
- camera presets staying outside excluded bounds and opening above terrain;
- deterministic atmosphere selection and no effect on LOS/detection/rewards;
- deterministic sustained fictional-conflict ambience, including profile/rate bounds, static-replay stability, excluded-precinct enforcement, and proof that the visual layer cannot alter simulation state;
- existing terrain, LOS, model, viewer, event, replay, guardrail, and demo-isolation tests remaining green.

Run focused tests and then full `pytest`. Execute smoke-render diagnostics in `physics` and `urban-presentation`; report actual geometry source, cache ID/hash, building/road counts, atmosphere profile, evidence state, models/fallbacks, and excluded-area enforcement. Browser inspection is optional; report it only when actually performed.

The final implementation must support commands of this form:

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
```

Handoff with changed files, AOI bounds/exclusion policy, source licence/attribution, exact commands/test results, visual status and limitations. Do not substitute a 2D map, a blank urban layer, a mock-only theatre, or real-world conflict data for the requested implementation.
