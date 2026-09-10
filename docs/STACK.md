# STACK

What each layer is, what it depends on, and how to verify it on its own. The
layering is deliberate: every slice below can be checked before the next one
pulls it in, which is what `pyproject.toml`'s extras are for.

```
components/*.json + data_cache/        <- research agent (allowlisted, cited, sha256'd)
        |  naigos/data/{theatre,terrain,enu}.py
        v
naigos/env/theatre_bridge.py           <- the seam. EnvConfig + DEM, assumptions declared
        |
        v
naigos/env/  flight_env  airframe  terrain  detection  threats  obs  spatial_hash
        |                              (jit / vmap, fixed shapes, ~450k env-steps/s CPU)
        v
naigos/rl/   networks (DeepSets+attn, CTDE)   reward   verifier (pure numpy)
             ppo (MAPPO + Lagrangian)         red_team   cbf (HOCBF-QP)
        |
        v
naigos/rl/train.py -> modal_train.py           naigos/demo/replay.py -> demo.json
  (perf/run/manifest.json)  \-> runmeta.py -> scripts/modal_runs.py       |
        \-> checkpoint.py             (submit / status / logs / resume)    |
                                                     \                     v
                                                      -> live.py -> assets/cesium.html
                                                         viewer.py -^  (one renderer:
                                                                        served, or static
                                                                        with routes inlined)
```

## Layers

| layer | module | extra | verify with |
| ----- | ------ | ----- | ----------- |
| data provenance | `naigos/research/` | `research` | `pytest tests/test_allowlist.py tests/test_cache.py tests/test_components.py tests/test_data_chain.py` |
| geospatial | `naigos/data/` | `data` | `pytest tests/test_dem_enu.py tests/test_terrain_geometry.py` |
| detection physics | `naigos/env/detection.py` | core | `pytest tests/test_detection_physics.py` |
| env | `naigos/env/` | core | `pytest tests/test_env_contract.py tests/test_airframe.py tests/test_spatial_hash.py` |
| constraints | `naigos/rl/verifier.py` | core (numpy only) | `pytest tests/test_verifier_cmdp.py` |
| learning | `naigos/rl/` | `rl` | `pytest tests/test_networks_ctde.py tests/test_reward_shaping.py tests/test_cbf.py` |
| presentation | `naigos/demo/` | `demo` | `pytest tests/test_viewer_export.py tests/test_terrain_endpoint.py tests/test_geodetic_live.py tests/test_imagery_layers.py tests/test_visual_modes.py tests/test_visual_renderer.py tests/test_replay_clock.py tests/test_los_profile.py tests/test_cesium_version.py tests/test_demo_isolation.py` |
| invariant | everywhere | core | `pytest tests/test_invariant.py` |
| benchmarks | `naigos/bench/` | `rl` + `data` | `pytest tests/test_bench_terrain.py` |

## Key contracts

**Env.** `reset(key) -> (state, obs)` and `step(state, action) -> (state, obs, terms, done, info)`,
both pure and both `jit`/`vmap`-able. Fixed padded shapes; no data-dependent control flow.
`action` is `(n_blue, 3)` — bank, flight-path angle, throttle. There is no fourth axis.

**Reward.** The env emits raw `RewardTerms`. `naigos/rl/reward.py` weights them.
Splitting the two is what lets the curriculum re-weight without touching physics
and lets the verifier recheck the constraint channel independently.

**Verifier.** Pure NumPy, shares no code with the env. It verifies a logged trace;
it does not re-simulate. A verifier that reuses the env's functions cannot catch a
bug in those functions.

**Data.** `naigos/env` never reads `data_cache/`. It reads `components/*.json`
through `naigos/data/theatre.py`, and every component names its cached bytes by
sha256. `tests/test_data_chain.py` enforces both directions.

**Globe.** The live globe has separate terrain and imagery layers. `/terrain`
serves the env's own heightmap, so displayed relief is the surface used by LOS;
Cesium ion Sentinel-2 or OpenStreetMap imagery is a cosmetic skin only. It is
never an environment observation or an RL input.
`naigos/demo/imagery.py` owns the skin and reads the ion token from
`NAIGOS_CESIUM_ION_TOKEN`; `tests/test_imagery_layers.py` asserts that no ion
terrain provider is ever constructed, that no token is committed, and that
nothing under `naigos/env` or `naigos/rl` names imagery at all.

**One renderer.** `assets/cesium.html` is the only 3D surface in the project.
`live.py` serves it and fills `/scene`, `/frames` and `/terrain`; `viewer.py`
exports the same page with those three routes inlined at `__EMBED__`, so a static
artifact and a served session hand the renderer identical shapes. A separate
three.js replay viewer used to draw the same recordings in a local ENU box with
no georeferencing, no threat envelopes and no LOS rays; it is deleted, and
`docs/artifacts/replay.html` is now the Cesium export. `tests/test_viewer_export.py`
checks the AGL twice -- once against the recording's ENU heightmap and once
against the lat/lon grid the browser samples -- each with a mirrored control.

**Replay time.** A recording has every frame, so replay runs on `viewer.clock`
with Cesium's animation and timeline widgets. Three rules keep a log from
becoming an animation: one sample per logged frame at the simulation timestep,
`LinearApproximation` at degree 1 (the library default is a Lagrange fit, which
would invent curvature between logged states), and availability ending at the
step an aircraft was lost. Live mode never touches the clock -- there is no next
frame to scrub to. `tests/test_replay_clock.py` pins all three.

**LOS rays.** `los_clearance` drops every sample for 4/3-earth refraction, ~94 m
at the midpoint of an 80 km ray. `naigos/demo/los.py` reconstructs that sampled
curve for drawing and finds the sample the minimum came from, so the ray on
screen is the geometry the number was computed from and the pinch point is
marked. It imports numpy and nothing else, on the same reasoning as the
verifier: a reconstruction that called the env's sampler would agree by
construction. `tests/test_los_profile.py` asserts the agreement instead (max
0.5 m over 64 rays).

**Cesium version.** The CDN URL in `assets/cesium.html` is the source of truth --
it is the build the browser loads. `package.json` pins the same version exactly,
`node_modules/` is not tracked, and nothing imports the npm package;
`tests/test_cesium_version.py` holds those together, including the npm-to-CDN
form difference (`1.145.0` on npm, `1.145` on the CDN).

**Visual modes.** `--visual` selects between two postures, and `VisualConfig` in
`naigos/demo/imagery.py` is the public contract for both:

| mode | surface | `evidence_grade` |
| ---- | ------- | ---------------- |
| `physics` (default) | the simulation's DEM, from `/terrain` | `True` |
| `photorealistic` | Google Photorealistic 3D Tiles, via CesiumJS | `False` |

Three rules hold across them. Physics is the default. Every fallback moves
*toward* physics — a missing credential downgrades photorealistic to physics and
Sentinel-2 to keyless OSM, never the reverse. And credentials come from explicit
environment variables only (`NAIGOS_CESIUM_ION_TOKEN`/`CESIUM_ION_TOKEN`,
`NAIGOS_GOOGLE_MAPS_API_KEY`/`GOOGLE_MAPS_API_KEY`), are consumed at resolution
and never retained: `VisualConfig` is frozen and holds booleans, so it is safe to
print, to serve at `/scene` and to paste into an issue. Each credential reaches
the browser through exactly one substitution point in `assets/cesium.html`.
`tests/test_visual_modes.py` pins mode resolution, missing credentials, the
fallback direction and token non-persistence.

**Terrain resolution.** `naigos/bench/terrain_resolution.py` measures what a
cell size costs and what it buys — compile time, throughput, memory,
line-of-sight agreement against the source 30 m DEM, and `/terrain` generation
time — across AOIs, cell sizes and `los_samples`.

```bash
uv run python scripts/bench_terrain.py            # full sweep -> docs/artifacts/terrain_bench.json
uv run python scripts/bench_terrain.py --quick    # ~30 s smoke check, not a publishable number
```

It is advisory and changes nothing. Measured recommendation: **100 m cells at
`los_samples=96` for physics**, and **100 m served at `/terrain n=1024` for
presentation** — two answers because they are two questions, and the second is
constrained by the endpoint resample rather than by the DEM. The shipped
defaults remain 1500 m / 96 (500 m for the live demo), which is what every
committed result was produced at; `tests/test_bench_terrain.py` fails if one of
them moves without the published numbers being re-run. Nothing under
`naigos/env`, `naigos/rl`, `naigos/data` or `naigos/demo` may import
`naigos.bench`, and that is asserted too.

## Two ways to run, and the difference matters

```bash
uv run python scripts/train_local.py     # synthetic ridged terrain. tests, CI, domain randomisation.
uv run python scripts/train_theatre.py   # the cited DEM + calibrated radar classes.
```

Only the second produces a real-data result. It requires the raw cache to have
been populated with `uv run naigos-research --aoi <theatre>`. `env_from_theatre`
raises rather than silently falling back to synthetic terrain, because a run
that believes it used a real DEM but did not is worse than a run that fails.
