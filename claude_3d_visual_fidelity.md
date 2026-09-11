# Claude Code task — make the Naigos Cesium experience visibly 3D

Implement this in the current Naigos repository. Do not stop at a plan, a README claim, a mockup, a new command-line flag, or point-marker styling: the finished live and replay viewers must visibly render 3D terrain, 3D aircraft, and 3D generic threat vehicles.

## First, establish the actual baseline

Read `README.md`, `docs/STACK.md`, `next-steps.md`, `naigos/demo/{live,viewer,imagery,los}.py`, `naigos/demo/assets/cesium.html`, `naigos/env/{threats,flight_env,config,detection}.py`, and all existing `test_*visual*`, `test_*terrain*`, `test_los_profile.py`, and viewer tests.

The current viewer is not flat internally: it sends the simulation heightmap through `Cesium.CustomHeightmapTerrainProvider`, so it is a real 3D DEM surface. However, aircraft and threats are currently rendered as Cesium **points**. That is the primary defect to fix. Also note that `docs/artifacts/learning_delta.png` is intentionally a 2D analytical figure; it is not the Cesium viewer. Do not mistake it for the 3D deliverable.

## Non-negotiable truth and safety rules

- Blue stays purely evasive. Do not add blue weapons, targeting, attack controls, or an offensive objective.
- Threat locations and threat types remain randomly spawned, generic, parameterized simulation abstractions. Do not introduce real force dispositions, current weapon-system databases, targeting-grade performance, or real military insignia.
- A visual effect must never be presented as a physics result it did not simulate. The current detection/lock/lethal-envelope model remains the source of truth for a shootdown. Tracers, launch flashes, impacts, smoke, and explosions are presentation effects tied to existing recorded simulation events only.
- Preserve the current evidence-grade rule: `--visual physics` displays the exact terrain surface used for LOS and is the default. Google Photorealistic 3D Tiles may remain an explicitly labelled presentation context, but must never replace the physics DEM in an evidence-grade claim.
- Do not put API keys, Cesium tokens, or asset-provider secrets in source, artifacts, logs, URLs, or command-line arguments.

## Definition of done

At ordinary viewing distance and an oblique camera angle, a user can immediately see:

1. mountain/valley relief with shaded, occluding 3D terrain;
2. aircraft with recognizable fixed-wing 3D silhouettes that bank, climb/descend, and follow their actual simulation trajectory;
3. generic mobile ground-threat vehicles mounted on the actual DEM, with their heading and turret/sensor direction visually tied to their simulated state where that state exists;
4. generic stationary radar/air-defense-site assets mounted on the actual DEM;
5. generic airborne interceptor/drone assets where the existing threat kind is airborne;
6. a clearly labelled visual kill/track event only when the simulation reports one, without inventing a projectile simulation;
7. all existing terrain LOS overlays, evidence banners, replay timing, and live/replay shared-renderer guarantees still working.

## Required design

### 1. Licensed, local 3D assets

Add a small, curated asset set under a repository-owned path such as `naigos/demo/assets/models/`:

- one generic fixed-wing aircraft;
- one generic wheeled/tracked ground vehicle;
- one generic stationary sensor/launcher site;
- one generic airborne interceptor/drone;
- optional small, generic visual-effect assets only if they materially improve the result.

Use glTF/GLB assets with a clear redistributable licence (prefer CC0, otherwise a licence compatible with this repository). Add `naigos/demo/assets/models/README.md` listing each file, source, licence, attribution, scale, forward/up axes, and any modifications. Store assets locally so the viewer works after the app is launched; never make a third-party model CDN a runtime dependency.

Do not download an unlicensed asset, copy a recognizable real-world military vehicle, or use a realistic model merely because its thumbnail looks good. If a suitable asset cannot be licensed and committed, create a clean generic glTF model/assembly and document that it is generic; do not leave point markers in place.

### 2. Replace point markers with Cesium Model entities/primitives

Refactor the single shared renderer in `cesium.html` so live and replay use `Cesium.ModelGraphics`/glTF models (or an equally appropriate Cesium model primitive) for aircraft and threats.

- Aircraft position must be its actual recorded/live position, not an estimated or cosmetic route.
- Aircraft orientation must use the simulation state. Extend the scene/frame schema only with state already produced by the physics simulation (heading, flight-path/pitch, and bank/roll as available). Convert the coordinate convention explicitly and test it. Do not infer a generic velocity vector that erases bank or makes a climbing aircraft look level.
- Threat model choice comes from the already-available threat kind fields: stationary site, mobile ground unit, or airborne interceptor. Ground threats must sit at the appropriate DEM-derived ground clearance, not a fixed ellipsoid height or sea level. Movers use the actual simulated heading.
- Add robust per-model scale, altitude offset, and axis-correction metadata. These must be centralized in a typed/configured model registry, not scattered magic numbers in HTML.
- Preserve labels, but make them optional/toggleable and visually secondary to the models. Retain threat-envelope and detection overlays.
- Models must obey terrain depth testing in `physics` mode. The current x-ray toggle may intentionally override it. Do not make models permanently visible through a ridge.
- Retain point/billboard fallback only for a missing or failed model load; display a one-time, visible non-secret warning. The normal path must never use the fallback.

### 3. 3D terrain should read as terrain

Keep the physics DEM as the source for `CustomHeightmapTerrainProvider`; do not solve this by switching the default to a provider mesh. Improve legibility without changing what the simulator used:

- retain the same-heightmap hillshade, but tune it for oblique viewing and ensure it remains aligned with the mesh;
- choose a compelling initial oblique camera framing (not top-down rectangle view) that shows relief and is valid for both AOIs;
- add a camera preset/toggle such as `terrain overview`, `follow aircraft`, and `top-down analysis`; top-down is useful but must not be the opening impression;
- keep vertical exaggeration opt-in/clearly indicated and rescale entities/overlays consistently, preserving the current no-aircraft-inside-mountain invariant;
- with Sentinel-2 credentials, maintain Sentinel-2 as a cosmetic imagery skin over the physics mesh. Without credentials, use OSM but keep the same 3D terrain quality;
- ensure terrain, models, LOS rays, objectives, threat domes, and shadows/lighting are coherent in a WebGL-capable browser.

Do not claim buildings/vegetation are physics when the DEM does not model them. If photorealistic 3D Tiles are enabled, keep the existing non-evidence warning and avoid using their building occlusion for any simulation overlay conclusion.

### 4. Honest visual event layer

Create a small event schema produced by the existing live/replay data path. It may include `detected`, `lock_acquired`, `launch_visual`, `shot_down`, and `terrain_masked`, but each event must cite exactly which existing state transition or verifier result caused it.

- A ground threat may rotate its generic sensor/turret toward the tracked aircraft only as a visual representation of an existing track/detection selection.
- Add an optional, short-lived stylized tracer/launch/impact effect only after the existing model reaches its lethal/shotdown outcome. It must be labelled in the HUD/docs as a **visualization of the simulated outcome**, not an independently simulated round/missile.
- No effects may alter reward, detection, lock, action selection, state transitions, or verifier results.
- Replay must reproduce an event at the same logged frame/time; live may show it once. Never emit repeated effects after an aircraft is already lost or retasked.

### 5. Contracts, tests, and verification

Keep one renderer for live and static replay. Update the scene/frame schema carefully and version it if that reduces ambiguity.

Add focused tests that run offline and do not require API tokens, WebGL, network access, a GPU, or a Modal account. At minimum verify:

- every required model asset exists locally and has a documented licence/attribution record;
- the model registry covers every existing threat kind and has explicit fallback behavior;
- live and replay scene payloads have required visual orientation/event fields and no secrets;
- a ground-threat model altitude is computed from the simulation terrain and changes when the DEM height changes;
- aircraft orientation conversion preserves heading and nonzero bank/climb semantics;
- models are the normal rendering path, while marker fallback activates only on an explicit load failure;
- model/LOS/entity altitude rescaling remains coherent when vertical exaggeration changes;
- event derivation is deterministic, does not modify simulation state, and cannot repeat after loss/retask;
- the physics mode still instantiates `CustomHeightmapTerrainProvider` and photorealistic mode still carries its non-evidence warning;
- existing viewer, terrain, imagery, LOS, replay-clock, invariant, and demo-isolation tests remain green.

Add a `--smoke-render` or equivalent non-interactive diagnostic that reports: selected terrain source, resolved imagery source, asset registry results, model fallback count, chosen camera mode, and whether a run is physics evidence-grade. This must be diagnostic only; it must not pretend a headless test rendered pixels.

If your environment supports a browser, additionally run the live viewer over a real packaged AOI and inspect a screenshot from the opening camera position. Only claim this visual check if you actually performed it. Otherwise give precise local commands for the human to run.

## Required user-facing documentation

Update `README.md` and `docs/STACK.md` with:

- the distinction between the 2D analytical `learning_delta.png` and the 3D Cesium viewer;
- an exact physics-terrain command, e.g. `uv run python -m naigos.demo.live --aoi tehran_basin --checkpoint checkpoints/theatre_1000.pkl --visual physics --imagery sentinel2 --open` (with the documented token environment variable and keyless fallback);
- camera controls/presets, vertical exaggeration behavior, model fallback behavior, and the visual-event honesty caveat;
- asset licence/attribution location;
- a concise statement that terrain physics and visual models are separate: the DEM determines physics; models/imagery make that physics legible but do not add hidden tactical input.

## Workflow and handoff

Implement in small, testable changes. Preserve unrelated worktree changes. Do not launch paid Modal work or perform unapproved network downloads as part of this task.

Before handoff, run the relevant focused tests and then the full `pytest` suite. Report changed files, asset provenance, exact command to open the realistic 3D physics viewer, tests run, whether a real browser screenshot was inspected, and any deliberately deferred fidelity work. Do not claim the model has real-world ballistic or targeting realism; call it a visually legible rendering of Naigos’s existing generic simulation.
