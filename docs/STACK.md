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
| cloud learning pipeline | `naigos/pipeline/`, `naigos/rl/modal_pipeline.py`, `scripts/pipeline.py` | core (stdlib); Modal to deploy | `pytest tests/test_pipeline_*.py tests/test_research_roots.py` |

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

## Cloud learning pipeline

Scheduled, unattended candidate learning on Modal. **Deploying is the only step
that needs the laptop**; after `modal deploy` the crons belong to Modal and keep
firing with every client closed.

```text
daily cron  -> snapshot: refresh allowlisted inputs -> immutable, provenance-checked snapshot
nightly cron-> candidate: train on the newest *new* snapshot (GPU, network blocked)
            -> held-out evaluation + independent verifier (CPU, network blocked)
            -> decision.json: eligible | rejected | inconclusive  (shadow by default)
weekly cron -> a longer candidate, only if the latest nightly passed
operator    -> scripts/pipeline.py promote <id>: re-validate, re-evaluate, then
               atomically move the champion pointer
```

"Learning" here means periodically retraining on refreshed, allowlisted,
non-sensitive simulation inputs and fixed scenario seeds. It never modifies a
deployed policy or its weights mid-rollout, never adds a source (the fixed
allowlist and `GUARDRAIL` are enforced on every snapshot), and blue stays
purely evasive.

### Volume layout

Everything lives under `/pipeline` on the existing `naigos-runs` Volume, beside
the detached runs (`pipeline` is a reserved run name so they cannot collide).
Written by `naigos/pipeline/layout.py`, whose docstring is the reference:

```text
/pipeline/config/current.json                versioned runtime config (atomic replace)
/pipeline/config/history/v<N>.json           every published version (write-once)
/pipeline/snapshots/<snapshot-id>/           cache/, components/, DATA.md, snapshot.json (immutable)
/pipeline/snapshots/.staging/<id>.<attempt>/ a snapshot being built; renamed into place when valid
/pipeline/candidates/<candidate-id>/         candidate.json, run.json, manifest/history/perf,
                                             ckpt_*.pkl, evaluation.json, decision.json
/pipeline/champions/current.json             champion pointer: candidate, checkpoint sha256, generation
/pipeline/champions/history/g<N>.json        every pointer generation (write-once)
/pipeline/locks/<lease>.json                 mirror of the atomic lease store (modal.Dict)
/pipeline/status/jobs/<stage>__<id>.json     one record per stage attempt: status, call id, reason
/pipeline/status/coordinator.json            last tick and its decisions (heartbeat)
```

Records that describe something that happened (`snapshot.json`,
`candidate.json`, `evaluation.json`, `decision.json`, config and champion
history) are write-once; each carries UTC timestamps, the code commit, a config
digest, parent IDs, the AOI, the seeds and content hashes. The two pointers
that are meant to move are replaced atomically (temp file + `os.replace`). IDs
are validated before they become paths and every path is checked to stay
inside `/pipeline`.

### Separation of workers

| Function | image | network | does |
| --- | --- | --- | --- |
| `snapshot_tick` / `nightly_tick` / `weekly_tick` | stdlib + numpy | yes (Modal API) | `coordinator.tick`, one container at a time |
| `snapshot_worker` | research deps only | allowlisted hosts only | seeds from the previous snapshot, re-fetches `refresh_sources`, validates, publishes |
| `train_candidate` | JAX CUDA, no baked data | **blocked** | trains on a re-hashed container-local copy of one snapshot |
| `evaluate_candidate` | JAX CPU | **blocked** | held-out eval, verifier, decision; the only automatic promoter |
| `promote_candidate` / `rollback_champion` | JAX CPU | **blocked** | the only manual pointer movers |
| `admin` | stdlib + numpy | yes | status, pause/resume, retry, unlock, config, prune |

The snapshot's research build runs in a child process whose DNS resolution is
restricted to the hosts in `naigos/research/allowlist.py`
(`naigos/pipeline/egress.py`), because Modal's platform-level domain allowlist
exists only for Sandboxes. GDAL does its own HTTP in C, beneath that guard, so
the child also points GDAL's HTTP at a closed port.

**Why a seed is needed.** py3dep 0.19 fetches 30 m 3DEP through GDAL from
`prd-tnm.s3.amazonaws.com` (USGS's staged-products bucket), which is not one of
the `usgs_3dep` allowlisted hosts; the locally cached DEM was fetched that way
while its manifest records the declared service URL. The allowlist is not
widened by this pipeline, so a DEM fetch in the cloud fails closed. The default
refresh re-pulls only `open_meteo`, and every later snapshot copies the DEM
from its parent; the chain starts from `pipeline.py seed`, which uploads one
AOI's slice of this clone's cited cache and has the deployed app validate it
exactly like any snapshot before publishing it (`origin: operator-seed`).
Whether to add the USGS bucket host to the allowlist is an open decision
(next-steps.md P-1).

Snapshot validation rejects missing bytes, hash
mismatches, uncited files or components, non-allowlisted sources, an AOI
mismatch, a stale AOI-scoped copy, and a missing `GUARDRAIL` or blue-evasive
invariant. `modal_train.py`'s offline image is unchanged.

### Default cadence (UTC) and cost controls

| stage | cron | runs only if |
| --- | --- | --- |
| snapshot | `0 6 * * *` (06:00 daily) | not paused; deployed code is a clean commit |
| nightly | `0 8 * * *` (08:00 daily) | a new verified snapshot exists; under `max_candidates_per_day` (1); smoke gate passes for this commit |
| weekly | `0 10 * * 0` (10:00 Sunday) | the latest nightly candidate was eligible or promoted |

Profiles default to `short` (nightly, 200 iterations) and `short` with 600
iterations (weekly), on the `NAIGOS_MODAL_GPU` card (A10G), each capped by a
4 h timeout (`NAIGOS_PIPELINE_TRAIN_TIMEOUT_S`). Retries are zero everywhere; a
failed stage waits for an operator. **No GPU run has happened yet**, so there is
no measured cost per candidate; the first nightly's `perf.json` is where that
number will come from (next-steps.md C-1). No remote pipeline run has been
performed either: the binding has been loaded against modal 1.5.5 and every
decision path is tested offline, nothing more. The coordinator ticks and admin calls
are seconds of CPU.

To change the cadence, edit `schedule` in `naigos/pipeline/default_config.json`,
commit, and `modal deploy naigos/rl/modal_pipeline.py` again. Modal fixes a
`modal.Cron` at deploy time, which is why the Volume config refuses a
`schedule` key. Everything else -- pause, gates, seeds, profiles, budget,
`auto_promote` -- is changed with `scripts/pipeline.py config set --file ...`
and takes effect at the next tick without a redeploy.

### Idempotency, leases and failure

Stage IDs come from an idempotency key over the stage's config digest, the
input snapshot and its content hash, and the code commit (plus the cron window
for snapshots). A duplicate cron delivery, an overlapping old invocation or a
manual `run-now` computes the same ID and skips. A lease is an atomic
create-if-absent in a named `modal.Dict` (Volumes are last-write-wins with no
compare-and-swap). A lease whose Modal call is terminal is reclaimed; a live
call holds; with no answer from Modal, a recent heartbeat holds and an old one
is **ambiguous and fails closed** -- the stage does not run and its job record
says why. Every tick first reconciles job records whose calls ended without
their workers recording it.

### Promotion gates

Strict by default (`gates` in the config). A candidate is **eligible** only if
every check passes on measured numbers; any failed check makes it **rejected**;
any check that cannot be decided (missing, non-finite or out-of-range metric,
no verifier result, an incompatible champion) makes it **inconclusive**, which
is never promoted. Checks: provenance (run, snapshot and checkpoint verify;
clean commit; GPU actually used); held-out seeds recorded and disjoint from
training seeds; the verifier found no mismatch on sampled episodes; survival >=
0.60 and objective >= 0.50; no worse than the avoid-plus-nap baseline on
survival (tolerance 0) and objective (0.05); and no regression against the
current champion re-evaluated on the same episodes (exposure, shootdowns,
terrain and out-of-bounds losses within +0.02; survival within -0.0, objective
within -0.02). "Held out" means held-out *seeds* from the same scenario
generator, not a held-out distribution (next-steps.md E-6).

`auto_promote` is **false** by default: an eligible decision is recorded with
`action: shadow` and the champion does not move. `scripts/pipeline.py promote
<id>` re-verifies provenance and hashes, re-runs the held-out evaluation (CPU,
so it must reproduce the record within `max_reproduction_error`), re-runs every
gate against the champion *as it is now*, and only then publishes. Publishing
takes the `champion` lease, compares-and-swaps on the generation, writes the
history entry, then replaces the pointer atomically; every refusal leaves the
pointer byte-for-byte unchanged.

### Operating it

```bash
uv pip install modal && modal token new     # auth: ~/.modal.toml, or MODAL_TOKEN_ID/SECRET in CI
modal deploy naigos/rl/modal_train.py       # the existing detached-run app, for the smoke gate
uv run python scripts/modal_runs.py submit --profile smoke   # arms the gate for this commit
modal deploy naigos/rl/modal_pipeline.py    # from a clean tree; laptop can close after this
uv run python scripts/pipeline.py seed      # once: first snapshot from this clone's cited cache
uv run python scripts/pipeline.py status    # scheduled/queued/running/completed/failed/
                                            # skipped/rejected/inconclusive/promoted
uv run python scripts/pipeline.py list-candidates
uv run python scripts/pipeline.py inspect c-...           # or s-..., g<N>, champion
uv run python scripts/pipeline.py promote c-... --approver <name>
```

- **Pause safely:** `pipeline.py pause --reason "..."` publishes a config
  version with `paused: true`. Schedules cannot be paused on Modal, so the crons
  keep firing, see the flag, record `skipped: paused` and do no costly work.
  Jobs already running finish (cancel them in the Modal dashboard if needed).
  `pipeline.py resume` lifts it. Stopping the schedules outright means `modal
  app stop naigos-pipeline`.
- **Recover a failed stage:** `status` shows the job's failure reason. Fix the
  cause, then `pipeline.py retry <id>`; training resumes from the candidate's
  last valid checkpoint if it has one, and a changed commit is refused (a new
  commit makes a new candidate). An ambiguous lease is cleared with
  `pipeline.py unlock <key> --yes` only after confirming its holder is gone.
- **Roll back:** `pipeline.py rollback --generation <N> --approver <name>`
  publishes a *new* generation pointing at generation N's candidate, after
  re-checking its artifacts. History stays append-only.
- **Retention:** `pipeline.py prune` lists snapshots and candidates beyond
  `retention.keep_snapshots` (14) and `keep_candidates` (30); `--apply` deletes
  them. Current and historical champions, their snapshots and anything with a
  live job are never deleted. Snapshots copy their parent's cache, so each is
  roughly one AOI's slice of `data_cache/` (the local cache holding two AOIs
  is ~170 MB).
- **Credentials:** none in code, artifacts, logs or arguments. The sources
  need no key; if one ever does, it goes in a `modal.Secret` on
  `snapshot_worker` only. Logs on the Volume pass through `runmeta.redact`.
