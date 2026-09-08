# Naigos

Survivable routing in contested 3D airspace. Aircraft learn to reach an
objective through a field of active threats — radar/SAM sites, mobile ground
units, interceptor drones — while an adaptive red team hunts them.

**The blue agent is purely evasive. It has no weapon and no offensive action, ever.**
Its entire action space is three flight controls: bank, flight-path angle,
throttle. It survives by routing, timing, terrain masking and manoeuvre — never
by removing a threat. This is enforced in code (`config.BLUE_ACTION_NAMES`) and
by `tests/test_invariant.py`, not by a comment.

The measurable claim: **detection probability and shootdowns down,
objectives-reached and sorties-preserved up, against a naive direct-route
baseline.**

## Result

1000 MAPPO-Lagrangian iterations on the cited Owens Valley theatre
(`checkpoints/theatre_1000.pkl`, shipped). Evaluated on held-out demo seeds, 48 worlds x 4 aircraft, with identical
seeds and an identical threat field across all four policies:

| | untrained | **trained** | naive direct route | avoid+nap heuristic |
| --- | --- | --- | --- | --- |
| sorties surviving | 0.172 | **0.677** | 0.240 | 0.646 |
| objectives reached | 0.177 | **0.672** | 0.240 | 0.646 |
| shootdowns (of 192) | 133 | **25** | 134 | 43 |
| mean detection probability | 0.376 | **0.224** | 0.268 | 0.157 |

Against the naive direct route: shootdowns fall 5.4x, objectives-reached rise
2.8x, detection probability falls 0.268 → 0.224 — the tradeoff `prompt.md` asks
to be measured, in the direction it asks for. The demo prints this table itself;
it is not transcribed by hand.

Measured at 500 m terrain cells with `los_samples=96`. Earlier published figures
used 1500 m / 24, which over-reported visibility by ~50% relative at low
altitude; see [next-steps.md](next-steps.md) for the convergence data. The
correction lowers detection for every policy and, notably, flips the comparison
with the hand-written heuristic.

![learning delta](docs/artifacts/learning_delta.png)

Raw numbers: [`docs/artifacts/summary.json`](docs/artifacts/summary.json),
[`summary_cbf.json`](docs/artifacts/summary_cbf.json), and the full training curve in
[`history.json`](docs/artifacts/history.json). `runs/` is gitignored; these are the
committed copies.

Three caveats that belong next to that table, not in a footnote:

- A **hand-written avoid-plus-nap heuristic remains the real bar.** Under the
  corrected line-of-sight model the policy now edges it on survival (0.677 vs
  0.646) and on terrain losses (15 vs 25), but the heuristic still flies
  markedly quieter (detection 0.157 vs 0.224). It is a first-class baseline in
  the demo and in every training eval, because beating only the naive route
  would be a weak claim.
- **Single seed, single theatre, single route geometry.**
- The policy wins by **climbing above the short-range engagement ceilings**, not
  by terrain masking. That is a legitimate tactic it found on its own, but it is
  not the signature mechanic the design intended. Diagnosed in detail in
  [next-steps.md](next-steps.md) V-1.

## Watch it live on a globe (CesiumJS)

```bash
uv run python -m naigos.demo.live --aoi tehran_basin --open
```

Steps the environment continuously and streams it to CesiumJS over Server-Sent
Events at `http://localhost:8765`, **over real 3D terrain**.

The globe's surface is the simulation's own heightmap, served from `/terrain` and
fed to Cesium through `CustomHeightmapTerrainProvider` — so what occludes on
screen is what occluded in the model, verified to a mean of 1.65 m against the
env's own sampler. Aircraft are depth-tested: they genuinely vanish behind
ridges, which is the visual proof of the mechanic (press **x-ray** to see them
through terrain). A **LOS ray** is drawn to whichever threat has the best look at
each aircraft, coloured red when the ray is clear, amber when grazing, and green
when a ridge is cutting it. Hillshade is computed in the browser from that same
height array, so the shading cannot disagree with the geometry.

To scrub a recorded rollout on the same globe:

```bash
uv run python -m naigos.demo.live --aoi owens_valley --replay runs/demo/demo.json --open
```

Recordings are tagged with their theatre; replaying one against a different AOI
is refused rather than silently relocating the sortie to the wrong continent. This is a **live simulation, not a replay**:
a worker thread runs `NaigosEnv` forever with the trained policy, and aircraft
are re-tasked through `env.respawn` the instant a sortie ends, so the theatre
never empties. Cumulative counters (sorties launched, objectives reached, shot
down, terrain and out-of-bounds losses, success rate) accumulate as it runs.

```bash
--aoi owens_valley     # the other theatre
--speed 60             # sim seconds per wall-clock second (default 10)
--blue 6 --threats 14  # cohort and threat-field size
--red-level 0.6        # red curriculum difficulty, 0-1
--cbf                  # run the HOCBF-QP backstop
--reroll 1200          # re-draw the threat field every N sim seconds
--ion-token <token>    # Cesium World Terrain; without it, OSM on the ellipsoid
--port 8765
```

**No Cesium ion token is needed.** Without one the viewer uses OpenStreetMap
imagery on the WGS84 ellipsoid, which is enough to see the routing. Pass
`--ion-token` (free key at ion.cesium.com) for Cesium World Terrain, which
renders the actual Alborz relief the aircraft are masking against.

`--reroll` matters more than it looks. Aircraft respawn but the threat layout
did not, so a long session was reporting a single draw: one benign layout showed
a 100% success rate against 40.8% measured across seven draws. The threat field
is now re-rolled on a cadence and the viewer rebuilds its envelopes to match.

### Theatres

| AOI | terrain | DEM source |
| --- | ------- | ---------- |
| `owens_valley` | Sierra crest / valley floor, 345–4077 m | USGS 3DEP 30 m |
| `tehran_basin` | Central Alborz over an urban basin. North third mean 2880 m, south third mean 1202 m — 3.5 km of relief over ~15 km, a steeper gradient than Owens Valley | Copernicus GLO-30 |

Each AOI has its own component snapshot under `components/aoi/<name>/`, so
running the research agent for one theatre cannot silently invalidate a
checkpoint trained on another.

**The shipped checkpoint was trained on Owens Valley.** Everything shown on
Tehran is zero-shot transfer — it holds up (40.8% success against a ~22%
direct-route baseline on that theatre) but no Tehran number here is a
trained-on-Tehran number. See [next-steps.md](next-steps.md) E-4.

Georeferencing is verified against published landmark elevations rather than
assumed: central Tehran and the Mehrabad apron agree with the DEM to 26–66 m,
and the Tochal massif peaks at 3956 m against a published 3964 m. A test asserts
this and a companion test perturbs the UTM zone to prove the check has teeth.

**Threat placement over any AOI is randomly spawned and parameterised**, exactly
as it is everywhere else in this project. Nothing here models any real
air-defence disposition, and the guardrail below applies unchanged.

## Watch a recorded rollout

The trained policy ships with the repo (`checkpoints/theatre_1000.pkl`, 1.9 MB),
so nothing has to be trained to see it fly.

```bash
uv venv --python 3.12
uv pip install -e '.[all]'

# 1. fly the rollouts and log them          (~14 s)
uv run python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl

# 2. build the 3D replay and open it        (instant)
uv run python -m naigos.demo.viewer runs/demo/demo.json --open
```

That gives you an **animated 3D replay over the real Owens Valley DEM**: the
terrain at ×3 vertical exaggeration, translucent red domes for the lethal
engagement envelopes, and the four aircraft flying the exact positions they were
logged at. Aircraft turn amber then red as a threat's track on them hardens, and
a wireframe marker is dropped wherever one was lost.

Controls: drag to orbit, scroll to zoom, <kbd>space</kbd> to play/pause,
<kbd>1</kbd>–<kbd>4</kbd> to switch between untrained / trained / direct route /
avoid+nap on the same seeds, <kbd>←</kbd><kbd>→</kbd> to step frame by frame.
Live counters (airborne, objectives reached, mean detection probability, lowest
AGL) update as it plays.

The page is a single self-contained HTML file — no server, no build step, and
the only external request is the pinned three.js CDN script. Switching policies
mid-playback is the learning delta: same terrain, same threat field, same seeds,
different policy.

**Prebuilt copy:** [`docs/artifacts/replay.html`](docs/artifacts/replay.html) is
the same page, already built from the shipped checkpoint. Open it directly and
skip both steps.

### Just the numbers

Step 1 on its own prints the comparison table and writes the static plan view.
It rolls out four policies — **untrained**, **trained**, the **naive direct
route** and a **hand-written avoid+nap heuristic** — on identical seeds over the
identical threat field, then writes `runs/demo/learning_delta.png` (the plan view
at the top of this README) and `demo.json` (the raw logged trajectories the
viewer reads).

```
  metric                      untrained        TRAINED   direct route      avoid+nap
  ----------------------------------------------------------------------------------
  sorties surviving               0.172          0.677          0.240          0.646
  objectives reached              0.177          0.672          0.240          0.646
  shootdowns                        133             25            134             43
  terrain losses                     14             15             12             25
  out-of-bounds losses               12             22              0              0
  mean detection prob             0.376          0.224          0.268          0.157
  mean track quality              0.448          0.250          0.318          0.196

  vs the naive direct route:
    shootdowns          134 -> 25   (5.4x better)
    objectives reached  0.240 -> 0.672   (2.8x better)
    detection prob      0.268 -> 0.224
```

Nothing in it is scripted. Every track is a real `env.rollout`, the untrained
side is a freshly initialised network rather than a strawman, and the run is
deterministic — the same command reproduces these numbers exactly.

Useful flags:

```bash
--worlds 96        # more episodes (default 48 worlds x 4 aircraft = 192 sorties)
--cbf              # add the HOCBF-QP safety backstop; see the CBF row below
--synthetic        # synthetic terrain instead of the cited 3DEP DEM
--seed 1234        # a different held-out seed set
```

With `--cbf` the same command shows the backstop's real tradeoff against the
trained policy: terrain losses 20 → **0**, but survival 0.651 → 0.568, objectives
0.651 → **0.240**, out-of-bounds losses 28 → 60 and shootdowns 19 → 23, at a QP
infeasibility rate of **0.227** (printed as a row, not omitted). The filter's
conservative margin closes corridors the policy had learned to thread, and where
the margin conflicts with the map edge the QP has no feasible action at all.
Against a naive nap-of-the-earth controller the same filter nearly doubles
survival (0.31 → 0.61). That is `prompt.md` §6's *"a backstop, not the plan, and
its value is bounded by how good the learned policy already is"* as a
measurement rather than a disclaimer.

## Verify it

```bash
uv run pytest -q                    # 205 tests, ~35 s, no network, no GPU
uv run pytest tests/test_invariant.py -q      # blue has no weapon
uv run pytest tests/test_verifier_cmdp.py -q  # constraints, pure NumPy
uv run pytest tests/test_data_chain.py -q     # manifest -> sha256 -> spec -> env
```

The suite runs offline. Tests that need the research cache skip cleanly if
`data_cache/` is absent.

## Train it

```bash
# synthetic ridged terrain: fast, no cache needed. ~2 s/iteration at these sizes.
uv run python scripts/train_local.py --iterations 200

# the cited 3DEP DEM. This is the path any reported number must come through.
uv run python scripts/train_theatre.py --iterations 1000 --envs 64 --steps 128

# with the safety backstop active during evaluation
uv run python scripts/train_theatre.py --iterations 1000 --cbf
```

Checkpoints and `history.json` land in `runs/<name>/`. The committed result came
from `scripts/train_theatre.py --iterations 1000 --envs 64 --steps 128`
(~2.5 h on a laptop CPU). Point the demo at any checkpoint it produces:

```bash
uv run python -m naigos.demo.replay --checkpoint runs/theatre/ckpt_001000.pkl
```

`env_from_theatre` **raises** rather than silently falling back to synthetic
terrain, so a run cannot quietly believe it used a real DEM when it did not.

## Rebuild the data layer

Not required — `data_cache/` and `components/` are committed, and the env reads
the specs, never the cache. To re-pull from scratch (the only step that touches
the network):

```bash
uv run naigos-research --aoi owens_valley      # fetch -> cache -> cited specs
uv run naigos-research --aoi front_range       # a second theatre
uv run naigos-research --skip-flights          # skip the slow OpenSky sampling
```

Idempotent: a second run hits the cache and changes nothing. AOI-scoped artefacts
carry a bbox fingerprint in the filename so changing the AOI cannot silently
reuse the wrong terrain.

## What is real

| claim | status |
| ----- | ------ |
| Terrain | Real. USGS 3DEP 30 m over Owens Valley, reprojected to UTM 11N, 345–4077 m of relief in the flown window. Cached, sha256'd, cited. |
| Radar detection | Derived from the monostatic radar range equation with Swerling-1 fluctuation. No product data sheet anywhere in the repo. |
| Atmosphere | Real. Open-Meteo pressure profile gives a *measured* effective-earth factor k = 1.256, not the textbook 4/3. |
| Airframe speeds and climb | Calibrated from 1654 OpenSky ADS-B states. |
| Airframe g-limit and tactical climb | **Assumed**, and labelled `ASSUMED_*` in `theatre_bridge.py`. Civil traffic never manoeuvres hard, so no open civil dataset can supply these. |
| Threat envelopes | **Parameterised abstractions** — a range, an altitude band, a reaction latency, a Pd curve. Not a capability database, by design (see the guardrail below). |
| Globe terrain | Real, and it is the **same surface the model used** — `/terrain` serves the env's own heightmap, verified against `sample_height` to mean 1.65 m. Hillshade is derived from that same array. |
| Training | Real MAPPO-Lagrangian runs with a reproducible learning curve, on CPU (1000 iterations). No GPU run yet. |
| CBF backstop | Implemented and wired behind `--cbf`. Measured both ways: it nearly doubles a naive controller's survival (0.31 → 0.61) and *costs* the trained policy objective rate (0.651 → 0.240). QP infeasibility (0.227) is reported, not hidden. |
| Learned red / self-play | **Not implemented.** `LearnedRedStub` raises rather than falling back. |

## The guardrail

Threat envelopes are parameterised abstractions built from open physics and
nominal open figures for generic classes of system. This repository does not
contain, and does not need, a current or targeting-grade capability database:
the RL problem depends on the *shape* of the exposure-versus-survival tradeoff,
not on real-world accuracy against any fielded system. The research agent's
allowlist refuses requests that drift that way
(`naigos/research/allowlist.py::check_request`).

## Layout

```
naigos/env/       JAX 3D flight env: airframe, terrain+LOS, detection, threats, obs, spatial hash
naigos/rl/        MAPPO + PPO-Lagrangian, DeepSets+attention nets, HOCBF-QP filter,
                  pure-numpy CMDP verifier, red team, Modal wrapper
naigos/data/      DEM / airspace loaders (consume the research cache via component specs)
naigos/research/  the research sub-agent: allowlisted fetch -> cache -> cited JSON
naigos/demo/      replay.py (logged rollouts), viewer.py (three.js 3D replay),
                  live.py + assets/cesium.html (live CesiumJS globe stream)
components/       one cited JSON per design decision and per data source
data_cache/       raw fetched bytes + manifest.json (sha256, licence, url, fetch time)
checkpoints/      the shipped trained policy the demo runs from
scripts/          train_local.py (synthetic), train_theatre.py (cited DEM), emit helpers
docs/             DEVLOG.md, DATA.md (provenance), STACK.md, artifacts/
tests/            205 tests; offline, no GPU
```

## Honesty check

- **Is it really training?** Yes — the full curve for the shipped policy is in
  [`docs/artifacts/history.json`](docs/artifacts/history.json), with the
  death-cause breakdown logged alongside the headline rate at every eval so a
  falling shootdown rate cannot hide a rising terrain-collision rate. New runs
  write the same file to `runs/<name>/`.
- **Is the demo real?** Yes — `naigos/demo/replay.py` replays logged rollouts from
  `env.rollout` with identical seeds and threat fields. The untrained side is a
  freshly initialised network, not a strawman.
- **Is the validation a tradeoff, not a claim?** Yes — detection probability and
  shootdown rate against objectives-reached and sorties-preserved, versus a naive
  direct route.
- **Did the invariant hold?** Yes. `tests/test_invariant.py`.

See [next-steps.md](next-steps.md) for what is not done and what would break first.
