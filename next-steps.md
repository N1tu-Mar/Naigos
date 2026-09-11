# next-steps.md

What exists, what does not, and what would break first. Written against
`prompt.md`'s build order (§10) and its honesty check (§12).

Every gap below has an ID, a **why it matters**, and a **what "done" looks like**.
Gaps marked **BLOCKING** stand between the current repo and a defensible headline
result; the rest are upgrades.

---

## 0. Where the build actually is

| prompt.md §10 step | state |
| ------------------ | ----- |
| 1. Terrain + airspace ground truth | **Done.** Real 3DEP DEM (Owens Valley, UTM 11N, 345–4077 m relief in the flown window), OurAirports geometry, OpenSky ADS-B, Open-Meteo profile. All cached, sha256'd, cited in `components/*.json`, offline after first run. |
| 2. 3D flight env | **Done.** Point-mass airframe, g/stall/ceiling/climb limits, `jit`/`vmap` verified, ~450k env-steps/s on CPU. Speeds calibrated from ADS-B; g-limit and tactical climb are declared assumptions (D-1). |
| 3. Detection + terrain-LOS | **Done and calibrated before rewarding.** Range equation + Swerling-1, soft LOS with measured effective-earth k, aspect-dependent RCS. Measured: flying low cuts mean detection probability from 0.27 to 0.14 on the real DEM. |
| 4. Threats v1 + reward/CMDP v1 | **Done.** Scripted lead-pursuit red, five theatre-calibrated classes, PPO-Lagrangian with a pure-NumPy verifier that agrees with the env to 1.8e-6. |
| 5. MARL learning curve | **Partial.** A 1000-iteration CPU run exists (below). The Modal path is now instrumented, profiled and verifiable but **has still not been run** (C-1); training has not reached full curriculum difficulty or been replicated across seeds (C-2, L-3). |
| 6. CBF backstop | **Done.** HOCBF-QP implemented, unit-tested, and wired into `rollout` behind `--cbf`. A/B measured; QP infeasibility rate reported (S-2, S-4). |
| 7. Red team v2 / v3 | **v2 done, v3 not started.** Difficulty curriculum runs and advances. Learned red raises `NotImplementedError` (R-1). |
| 8. Learning-delta demo | **Done.** Four-policy replay of real logged rollouts, a static plan view, and one CesiumJS globe renderer serving both a live stream with continuously re-tasked aircraft and a clock-driven replay -- shipped static with its routes inlined. Terrain orientation and geodetic placement are both verified by test, on the recording's grid and on the one the browser samples. |

### The learning curve that exists today

The shipped `checkpoints/theatre_1000.pkl` — real Owens Valley theatre, 1000
MAPPO-Lagrangian iterations, 64 worlds × 128 steps, CPU, single seed. The red
curriculum reached level 0.20; the corresponding curve is committed as
`docs/artifacts/history.json`.

**Held-out demo seeds** (`python -m naigos.demo.replay`, 48 worlds × 4 aircraft,
seed 999, identical seeds and identical threat field across all three policies):

| | untrained | **trained** | direct route | avoid+nap heuristic |
| --- | --- | --- | --- | --- |
| survival | 0.172 | **0.677** | 0.240 | 0.646 |
| objective reached | 0.177 | **0.672** | 0.240 | 0.646 |
| shootdowns (of 192) | 133 | **25** | 134 | 43 |
| mean detection probability | 0.376 | **0.224** | 0.268 | 0.157 |
| mean track quality | 0.448 | **0.250** | 0.318 | 0.196 |

**The headline claim holds on this seed set.** Against the naive direct route:
shootdowns fall 134 → 25 (5.4×), survival rises 0.240 → 0.677,
objectives-reached rises 0.240 → 0.672, and mean detection probability falls
0.268 → 0.224. Detection *and* shootdowns down, objectives *and*
sorties-preserved up — which is exactly what `prompt.md` asks to be measured.

**What is not yet clean, stated plainly:**

- **The hand-written avoid-plus-nap heuristic remains a serious baseline.** The
  policy now leads it on survival (0.677 vs 0.646), but is more detectable
  (0.224 vs 0.157). It is a first-class replay and training-evaluation baseline,
  not a calibration-only probe (E-7).
- **Single seed, single theatre, single route geometry.** See L-3, E-3, E-4.
- **The detection improvement is smaller than the survival improvement**, and on
  the *training-distribution* eval (`exposure_early`, an unbiased window) the
  trained policy is actually worse than the direct route: 0.715 vs 0.389. The two
  evals disagree because they measure different windows; see V-1.
- **Residual loss channels:** terrain 0.066, out-of-bounds 0.176 at the end of
  training. Roughly one sortie in five is still lost to leaving the map.

---

## 1. BLOCKING gaps

### C-1 — No GPU run has happened — PATH BUILT, STILL NOT RUN
`naigos/rl/modal_train.py` defines the image, the Volume and the entrypoint, and
mounts `components/` and `data_cache/` so the worker stays offline. **It has
still never been executed.** Every number in this repo is from CPU, and this
repo therefore states no GPU throughput and no speedup, because it has not
measured one.

**Why it matters.** §7 asks for a reproducible learning curve from Modal, and the
run lengths that would produce a converged policy (thousands of iterations at
256+ parallel worlds) are not practical on a laptop.

**What now exists, so that the first run produces a defensible number rather
than an anecdote:**

- **Instrumentation.** `train.py` writes `perf.json` beside `history.json`:
  backend and device kind, first-iteration wall time labelled as compile plus
  one step, median and p90 seconds per iteration over the last 50 timed
  iterations, env-steps/s and agent-steps/s, recompile count and cost, and peak
  device memory. Timing closes after the metrics are pulled to the host,
  because JAX dispatch is asynchronous and that is the real sync point. On CPU
  the memory field is null rather than zero: the backend does not implement
  `memory_stats`, and an unmeasured quantity is not a measured zero.
- **A CPU baseline to compare a GPU run against.** 0.54 s/iteration and
  3.8k env-steps/s at 32 worlds × 64 steps on the development laptop, from that
  same `perf.json`. Without it a GPU number is a number rather than a speedup.
- **A smoke profile.** Three iterations at 16 × 32. `--profile full` is refused
  unless a smoke run verified against the same commit is recorded on the
  Volume. The failures that only appear remotely — a dependency missing from
  the image, an unmounted cache, an unwritable Volume, no GPU actually attached
  — should cost minutes, not six hours.
- **Periodic persistence.** The Volume was committed once, at the end, so a run
  that hit the six-hour timeout left *nothing*. It is now committed after every
  history, perf and checkpoint write.
- **Detached execution.** `scripts/modal_runs.py submit` spawns a *deployed*
  Modal function and returns a job id immediately; the job then belongs to
  Modal's queue and not to the launching process, so the laptop can be closed.
  `status`, `logs`, `cancel`, `fetch` and `resume` address the run by name
  afterwards. `modal run` is kept for debugging the image and is explicitly not
  the detached path.
- **Trustworthy status.** A `manifest.json` per run — job id, timestamps,
  requested GPU, actual backend and device, commit, termination reason, one
  entry per attempt — merged with Modal's call state into one verdict across
  queued / running / completed / timed out / failed / cancelled, plus a separate
  `resumable` flag. The case this exists for: a container killed by its own
  timeout never updates its manifest, so the manifest says `running` forever and
  trusting it alone reports a dead job as live.
- **Resume.** Checkpoints now carry complete recovery state (both optimizer
  states, the multiplier's optimizer state, the RNG key, both curricula, the
  accumulated history), and `modal_runs.py resume` continues from the most
  recent *valid* checkpoint — walking backwards, because the newest file is the
  one a killed container was most likely mid-write on. Resume is refused across
  a commit, seed, theatre, shape or length change unless the specific key is
  named. An interrupted run reaching bit-identical final state to an
  uninterrupted one is asserted in `tests/test_resume.py`.
- **A smoke run that says what it proved.** Six named preflight checks — image,
  GPU backend, data cache, Volume write, checkpoint read-back, artifact
  retrieval — recorded in the manifest, so a smoke run that completed without
  proving one of them does not arm the gate in front of `short` and `full`.
- **No auto-retry.** `retries=0`, deliberately: Modal's retry restarts from
  scratch, which would pay for the same iterations twice and put a second writer
  in one run directory. Resume is an operator decision.
- **Run isolation and immutable metadata.** Validated run names defaulting to
  `<profile>-s<seed>-<timestamp>`, a write-once `run.json`, and a writer lock on
  the run directory; pointing a different configuration at an existing run
  directory is refused, and so is a second job for a run that is already live.
- **Verification.** `scripts/modal_runs.py verify` reports a truncated run, two
  runs interleaved into one directory, a real-theatre run with no provenance, a
  dirty tree, and a run launched on Modal whose `perf.json` says the backend was
  CPU. JAX falls back silently, so that last one is the specific way a CPU
  number gets published as a GPU number.

**Done looks like.** One `modal_runs.py submit --profile smoke` completing with
all six preflight checks green, then `--profile short`, with `perf.json` fetched
off the Volume and its env-steps/s quoted in the DEVLOG next to the CPU figure
above. Not before.

### C-2 — Training has not been run to convergence
The longest committed run is 1000 iterations at 64 worlds × 128 steps. It ends
at red curriculum level 0.20, well below the target level of 1.0, and is only
one seed; it is evidence of learning, not a convergence result.

**Done looks like.** A run where the curriculum reaches level 1.0 and the eval
metrics plateau, with the plateau visible in `history.json`.

### V-1 — The policy climbs over threats instead of hiding under terrain — DIAGNOSED
**The diagnosis ran.** Mean AGL under the trained policy rises from 2878 m at
iteration 1 to ~5400 m at convergence, and `exposure_early` rises with it
(0.389 direct-route baseline → 0.715 trained). The policy is not flying low and
being seen anyway; it is deliberately climbing.

**And it is a legitimate strategy, not a reward hack.** Three of the five
theatre-calibrated threat classes have engagement ceilings between 4.5 and 6 km
(`short_range_point_defense` 6000 m, `mobile_short_range` 4500 m,
`interceptor_seeker` 12000 m). Climbing above them makes those classes unable to
engage at all, at the cost of being brightly visible to the two long-range
classes that cannot reach you either. The policy found "trade detection for
un-engageability", which is a real air-defence tactic and which the reward, as
written, correctly prices as better than nap-of-the-earth.

**So this is a specification question, not a bug.** The measured fact that low
flight halves detection probability (0.27 → 0.14) confirms the terrain-masking
mechanic works and is available. The policy is declining to use it because
exposure costs 0.15/step against an airframe loss of 300, so ~500 steps of full
exposure is worth roughly a sixth of one shootdown.

**Done looks like — pick one, deliberately:**
1. If low observability is the actual objective, raise the exposure and lock
   weights until integrated exposure is comparable to a fraction of an airframe
   loss, and re-measure. The risk is re-crossing the line where shaping exceeds
   the task reward, which `test_reward_shaping.py` now guards.
2. If survival is the actual objective, **report the altitude finding as a
   result** rather than a defect — the policy discovered an engagement-ceiling
   exploit from local observations alone — and drop the "terrain masking is the
   signature mechanic" framing to "terrain masking is one of two strategies the
   env supports, and the agent chose the other one".

Option 2 is more honest about what was actually built. Option 1 is what
`prompt.md` §5 implies. They should not be blurred.

### ~~S-2 — The CBF is not wired into training or the demo~~ — DONE
Now wired. `NaigosEnv.rollout` takes an `action_filter(state, obs, action) ->
(action, feasible)`; `naigos.rl.cbf.make_policy_filter` builds one that reads
envelopes out of the **observation** (so the backstop is limited to what the
aircraft has sensed and never touches the ground-truth threat list) and reads
ego kinematics and the DEM from the state. `--cbf` on `scripts/train_theatre.py`
and `python -m naigos.demo.replay`.

**Historical calibration**, real theatre, curriculum level 0, 48 worlds,
identical seeds, naive nap-of-the-earth controller:

| | CBF off | CBF on |
| --- | --- | --- |
| survival | 0.312 | **0.609** |
| objective reached | 0.312 | **0.604** |
| terrain losses | 81 | **42** |
| shootdowns | 51 | **25** |
| QP infeasibility rate | — | **0.540** |

**The infeasibility rate is the honest part and it is not small.** For 54% of
live agent-steps the half-space intersection with the actuator box is empty —
overlapping envelopes with no gap simply cannot be flown out of, which is exactly
the caveat §6 requires. The filter still helps a great deal, because even an
infeasible projection returns the closest admissible-ish action rather than
nothing, but "the CBF guarantees keep-out" would be a false claim here. What is
true: it roughly halves both terrain losses and shootdowns for a naive policy.

**Second measured necessity, found the same way.** The first wired version had no
map-boundary barrier, so the filter satisfied envelope keep-out by pushing
aircraft off the map: bounds losses tripled (28 → 88 of 192 sorties) and net
survival *fell* from 0.651 to 0.474. A keep-out filter with an incomplete barrier
set discharges the constraint into whatever it was not told about. Four
half-space edge barriers were added; that took infeasibility from 0.41 to 0.23
and terrain losses to 0.

**Current held-out artifact: the CBF hurts the trained policy's completion.**
Against `checkpoints/theatre_1000.pkl` on the committed demo seeds:

| trained policy | CBF off | CBF on |
| --- | --- | --- |
| survival | **0.677** | 0.568 |
| objective reached | **0.672** | 0.240 |
| shootdowns | 25 | **23** |
| terrain losses | 15 | **0** |
| bounds losses | 22 | 60 |
| QP infeasibility | — | 0.227 |

The filter removes terrain losses entirely and cuts shootdowns further, but its
conservative `margin` keeps the aircraft out of corridors the trained policy had
learned to thread, so objective rate collapses. That is precisely the §6 claim —
*a backstop, not the plan, and its value is bounded by how good the learned
policy already is* — showing up as a number rather than as a disclaimer.

**Residual work (now S-4).** Anneal `CBFConfig.margin` with the red curriculum,
and re-measure. A filter whose margin is tuned for an untrained policy is the
wrong filter for a trained one.

---

## 2. Data and calibration

### D-1 — The g-limit and tactical climb performance are assumptions, not measurements
`ASSUMED_MAX_LOAD_FACTOR = 5.0`, `ASSUMED_CLIMB_RATE_MS = 120.0`,
`ASSUMED_MAX_FLIGHT_PATH_ANGLE = 0.52` in `naigos/env/theatre_bridge.py`.

OpenSky gives implied bank angles at p95 9.1° and max 26.8° — which validates the
coordinated-turn relation the model is built on, but bounds only what airliners
*choose* to fly. Using the measured civil climb rate (13.3 m/s) made a
terrain-following controller crash into the Sierra on ~80% of sorties.

**Why it matters.** These three constants set how sharply the aircraft can react,
which directly sets how survivable the task is. They are the least-grounded
numbers in the project and they are load-bearing.

**Done looks like.** Either an open source for tactical-aircraft manoeuvre limits
cited in `components/`, or — better — a **sensitivity sweep**: survival and
objective rate across n_max ∈ {3, 5, 7} and roc_max ∈ {60, 120, 200}, published,
so the result is reported as a band rather than a point that depends on a guess.

### D-2 — No airspace structure
`components/data.airfields.json` has airfield and runway geometry, but there are
no restricted areas, corridors, or no-fly volumes. Start points and objectives are
currently placed by grid fraction (`spawn_inset_frac`), not at real airfields, even
though the airfield data is loaded.

**Done looks like.** Sorties departing from and routing to the six AOI airfields
that have runway geometry; optionally OpenAIP airspace volumes as soft-constraint
regions.

### D-3 — Lock dynamics are invented
`lock_gain` is derived from the radar scan period and `lock_decay` is a flat 0.40.
Neither is in any component spec. The track-build/decay timescale is what makes
pop-up-and-remask tactics viable, so it directly shapes the optimal policy.

**Done looks like.** Either a cited basis for track-formation timescales, or an
explicit `components/model.tracking.json` that states it is a modelling choice and
records a sensitivity sweep.

### D-4 — Weather affects nothing dynamic
The atmosphere component supplies air density and the effective-earth factor, both
of which are used. Wind is fetched and unused; there is no ceiling, no
visibility, no density effect on airframe performance with altitude.

**Done looks like.** Wind as a drift term on the ground track (cheap, and it makes
timing genuinely harder), and density-altitude degradation of climb rate.

---

## 3. Environment and physics

### E-2 — Interceptors are 2.5D
`threats.step` holds an airborne threat's altitude at `max(current, ground+200)`.
Interceptors chase in the horizontal plane and never climb or dive at their
target, so a simple vertical jink defeats them for free.

**Why it matters.** It is a hole the policy can and eventually will find, and it
undercuts the "3D pursuit-evasion" framing.

**Done looks like.** A vertical channel in `scripted_red` (proportional navigation
in the vertical plane, with a climb-rate limit per class), plus a test that a pure
vertical evasion no longer produces a clean escape.

### E-3 — One shared objective geometry, west-to-east
Every episode routes across the short axis of the map with start and objective at
fixed grid fractions. Threats scatter around that one corridor.

**Why it matters.** The policy may be learning "fly east, dodge" rather than a
transferable routing behaviour. There is no held-out geometry to test that.

**Done looks like.** Randomised start/objective pairs (including north-south and
diagonal routes) and a held-out set of geometries the policy never trained on,
evaluated separately.

### E-4 — Two theatres now, but the policy is still trained on one
`tehran_basin` was added alongside `owens_valley` (Copernicus GLO-30, since it is
outside 3DEP coverage), and each AOI has its own component snapshot so a research
run for one cannot silently invalidate a checkpoint trained on the other. Its
morphology is genuinely different from both existing AOIs: a dense urban basin
walled by a single high ridge (north third mean 2880 m, south third mean 1202 m)
rather than an open valley.

**Still open, and this is the one to read carefully.** The shipped checkpoint was
trained entirely on Owens Valley. Everything shown on Tehran is **zero-shot
transfer**, and the live viewer runs it that way by default. It transfers
respectably -- 40.8% success across seven threat draws, against a direct-route
baseline of ~22% on the same theatre -- but no number reported on Tehran is a
trained-on-Tehran number, and nothing in the README should be read as claiming
one.

**Done looks like.** A checkpoint trained on `tehran_basin`, and a 2x2 table with
each policy evaluated on both theatres, so transfer is measured rather than
asserted.

### ~~E-9 — Cesium World Terrain is not the DEM the simulation used~~ — DONE
`/terrain` serves the env's own heightmap as an int16 lat/lon grid and
`CustomHeightmapTerrainProvider` makes it the globe's actual terrain. Verified
against `sample_height`: mean 1.65 m, p95 8.4 m, max 37.5 m. Sampling the raw
30 m DEM instead was measured to diverge by up to 2094 m -- the same defect in a
different costume. Hillshade is derived from the same array (no vertex normals
from this provider), and aircraft are depth-tested so occlusion is real.

**Residual (E-12).** The 512x512 lat/lon resample carries up to ~37 m of
disagreement, against a 30 m AGL floor -- so at nap-of-the-earth altitude the
render can put an aircraft slightly underground. The fix is to ship the ENU grid
exactly plus a small lon/lat->ENU warp table (a degree-3 polynomial or a 17x17
bilinear table is accurate to under a metre over the AOI), which also shrinks the
payload from 512 KB to 65 KB and makes the client sampler a line-for-line port of
`sample_height`. Until then the equality claim carries a ~40 m qualifier.

**Follow-on, also done.** The imagery layer was separated from the terrain layer
outright. Imagery is now Copernicus Sentinel-2 (Cesium ion asset 3954) built
through `IonImageryProvider`, terrain stays `CustomHeightmapTerrainProvider` over
`/terrain`, and no ion terrain provider is constructed anywhere. That closes the
route by which E-9 could return as a config change, and swaps Cesium's default
Bing Aerial base layer — commercial, session-metered — for ESA open data. The
token moved to `NAIGOS_CESIUM_ION_TOKEN`. `tests/test_imagery_layers.py` pins all
of it, including that nothing under `naigos/env` or `naigos/rl` so much as names
imagery.

### E-9-OLD — original writeup, kept for the record
With `--ion-token` the globe renders Cesium World Terrain. The simulation
computed every line-of-sight ray against the cached Copernicus GLO-30 grid
resampled to 1500 m cells. Those are two different surfaces, so an aircraft can
appear to clear a ridge it was masked by, or the reverse.

**Why it matters.** The point of the viewer is to make terrain masking legible.
If the terrain on screen is not the terrain in the model, the viewer illustrates
the mechanic rather than showing it. Without a token the globe is a bare
ellipsoid, which is honest but shows no relief at all.

**Done looks like.** Render the env's own heightmap as a Cesium primitive -- a
`GroundPrimitive` mesh, or a quantized-mesh provider generated from the npz -- so
the displayed surface is the one LOS was computed against, with ion terrain
demoted to an optional backdrop. *(Superseded: ion terrain is not a backdrop
either. It is gone, and imagery is the only thing ion supplies.)*

### E-5 — Fuel and endurance are notional
`ASSUMED_FUEL_KG = 3000` with a hand-picked burn model. No sortie ends from fuel
exhaustion in practice, so the fuel term is currently decorative.

**Done looks like.** Either calibrate endurance so fuel actually binds on a
long-way-round route (which is what makes the routing tradeoff interesting), or
remove the term and say so.

---

## 4. Learning

### R-1 — Learned red / self-play is not implemented
`LearnedRedStub` raises `NotImplementedError` by design, so a half-wired run
cannot be mistaken for a real one. `prompt.md` §4 calls v3 the strongest result
and correctly stages it after a clean v1/v2 curve.

**Done looks like.** A separate red actor-critic with its own optimizer,
alternating updates against a frozen blue, and a reported exploitability-style
metric (how much a freshly trained red gains against a frozen blue). Do not start
this before C-2 and V-1 — self-play on top of an unconverged blue produces noise.

### ~~L-1 — The Lagrange multiplier only rises~~ — DONE
λ climbed monotonically (1.00 → 2.98) across every run and never came back,
because the measured cost never fell below the budget. **Root cause, found by
reading the units:** the cost channel summed `aircraft_lost` — a once-per-episode
terminal event — with `envelope_dwell`, a per-step indicator. Over a 500-step
episode the dwell term dominates by two orders of magnitude, so "expected
violations per agent-episode <= 0.05" was arithmetically unmeetable and dual
ascent could only integrate upward. λ was a slowly growing penalty weight
wearing a constraint's name.

**Fixed.** The cost channel is terminal-only, so it reads as a rate in [0, 1]:
"expected airframe losses per agent-episode". Budget 0.10. Envelope dwell is
still penalised, as *shaping*, in the reward, where a per-step quantity belongs.
A proportional term (Stooke et al.'s PID-Lagrangian) sits outside the optimizer
so it vanishes the moment the constraint is met, and λ is hard-capped.

**Observed after the fix:** cost 0.43 → 0.28 over the first 20 iterations and λ
moving *down* (1.17 → 1.15). The multiplier can now decrease, which is the whole
point of pricing a constraint rather than weighting a penalty.

`verifier.py` was updated to match — a verifier checking a different constraint
from the one being optimised is worse than no verifier, and a test now asserts
the two channels agree.

### L-2 — The critic is per-world but the advantage is per-agent
`ppo.py` computes one centralized value per world and then broadcasts it across
all `n_blue` agents when forming minibatches. That is a legitimate MAPPO variant
(a shared team value), but it means an individual aircraft's advantage carries its
teammates' outcomes, which slows credit assignment.

**Done looks like.** Either an agent-conditioned centralized critic (global state
plus an agent-ID / ego-view embedding) or an explicit statement that the team
value is intentional, with an A/B run supporting it.

### L-3 — No hyperparameter search, single seed
Every result is one seed. Learning rate, clip epsilon, entropy coefficient,
GAE lambda and network width were set to reasonable defaults and never swept.

**Done looks like.** At least 3 seeds per reported configuration with the spread
shown, and a small sweep over learning rate and entropy coefficient.

### L-4 — Curriculum promotion is coarse (and had a real bug, now fixed)
**BUG, found in a training run and fixed:** promotion was driven by
`1.0 - shootdown_rate` as a stand-in for survival. Those are not the same
quantity. A policy that had traded being shot down for flying into a ridge had a
*low* shootdown rate, was promoted to red level 0.5, and survival collapsed from
0.51 to 0.10 with terrain losses at 0.28 — the run never recovered. Promotion now
reads the measured `survival_rate`, and a test asserts it.

**Still open.** Promotion is a single noisy eval against a hard threshold, with a
wide dead band (demote below 0.35, promote above 0.75) that the policy sits
inside for long stretches.

**Done looks like.** Promotion on a statistic smoothed over several evals, and a
curriculum trace in `history.json` so a plateau can be attributed to the
curriculum rather than to the policy.

---

## 5. Safety

### S-1 — The QP is a projection, not an exact solver
`_project_qp` runs 24 cyclic projections onto the half-spaces and the box. That
converges for a feasible convex intersection and is `jit`-able with static shapes,
but it is not an active-set QP and gives no optimality certificate.

**Done looks like.** Either a comparison against a reference solver (`qpax`, or
CVXPY offline) on a few thousand sampled states, reporting the distribution of
suboptimality, or a swap to a differentiable QP layer if the gap turns out to
matter.

### S-4 — Infeasibility remains high under a trained policy
The trained-policy A/B above measures a 0.227 infeasibility rate. That is lower
than the historical naive-controller calibration (0.540), but still means over a
fifth of live agent-steps have no jointly feasible projected action.

**Done looks like.** A sensitivity sweep of `CBFConfig.margin` by red level and
threat density, reporting completion, loss channels and infeasibility together.

### S-3 — Head-on geometry is a documented degeneracy
Exactly nose-on to an envelope centre, the lateral constraint coefficient is zero
and the filter can only decelerate; it reports `feasible=False`. This is a real
property of the relative-degree-2 barrier, not a bug, and it is tested — but it is
also the geometry an aggressive direct-route policy produces most often.

**Done looks like.** A tie-breaking term that injects a small lateral preference
as the geometry approaches head-on, plus a measurement of how often the head-on
case actually arises under the trained policy.

---

## 6. Evaluation and demo

### ~~E-1 — No 3D viewer, demo not run against a converged checkpoint~~ — DONE
`naigos/demo/replay.py` now rolls out four policies (untrained, trained, direct
route, avoid+nap) on identical seeds and an identical threat field, prints a
comparison table, draws the four-panel plan view, and exports the full scene —
terrain heightmap, per-threat envelopes, per-frame threat motion, per-frame
exposure and track quality — to `demo.json`.

`naigos/demo/viewer.py` turns that into a **self-contained animated 3D replay**
(`docs/artifacts/replay.html`, also shipped prebuilt). It is the CesiumJS globe
page with `/scene`, `/frames` and `/terrain` inlined: the simulation's own DEM as
the drawn surface, translucent lethal domes, refracted LOS rays with their pinch
points, aircraft coloured by how hard they are being tracked, tracks that end
where the aircraft was lost, and a policy switch on the same seeds. Play, scrub
and rate are Cesium's clock widgets. One external request (pinned CesiumJS), no
server, no build step, no credential in the artifact.

Both run against the shipped 1000-iteration checkpoint; that checkpoint is not
presented as a converged policy.

**Guarded, not just written.** `tests/test_viewer_export.py` re-implements the
viewer's own bilinear terrain lookup and asserts it reproduces the AGL the env
logged (agreement: 0.11 m, which is the export's rounding). A companion test
mirrors the heightmap north-south and asserts the check fails, so it cannot pass
vacuously. This is the exact bug class that already bit the repo once at the
env/data seam: the terrain still looks like terrain and the tracks still look
like tracks while the aircraft fly over a mirrored map.

**~~Residual (E-8)~~ — closed by the consolidation.** The three.js replay viewer
drew lethal envelopes but not detection envelopes or the LOS rays, so *why* a
given aircraft was or was not seen could only be inferred. The fix was not to add
them there: that viewer is deleted, and the export is now the Cesium page, which
already draws detection rings and the per-frame ray to the strongest tracker.

### E-6 — No held-out evaluation protocol
Evaluation uses fresh keys but the same generator, so start points, objectives and
threat layouts come from the same distribution the policy trained on.

**Done looks like.** A frozen evaluation set of N scenarios (seeded once, stored),
reported separately from training-distribution eval.

**Partly addressed by P-1.** Pipeline candidates are evaluated on held-out
*seeds* fixed in the versioned config and refused if any overlaps a training
seed. The generator is still the training generator, so this is reproducible
held-out episodes, not a held-out distribution.

### E-7 — A competent baseline is present, but needs broader evaluation
The hand-written avoid-plus-nap controller is now a first-class replay and
training-evaluation baseline, alongside the naive direct route. On the current
held-out seeds the learned policy leads it on survival (0.677 vs 0.646), while
the heuristic remains less detectable (0.157 vs 0.224).

**Why it matters.** Beating a naive baseline is a weak claim. Beating a competent
hand-written heuristic is the claim worth making.

**Done looks like.** The learned-versus-heuristic comparison repeated across
multiple seeds, theatres and held-out route geometries.

---

### ~~E-11 — The live viewer has no LOS ray~~ — DONE
Each aircraft draws a ray to whichever threat has the best look at it, in three
clearance bands keyed to `los_clearance_scale`. Selected by detection probability
rather than lock: lock decays to zero the moment an aircraft is masked, so an
earlier lock-gated version switched the feature off in exactly the case it
exists to show. Measured over 240 live aircraft-steps: 5% masked, 56% grazing,
40% clear.

**~~Residual (E-13)~~ — DONE.** The ray was drawn as a straight 2-point line
while the model applied a 4/3-earth refraction drop of up to ~94 m at the
midpoint of an 80 km ray, so on grazing long-range geometry the drawn line could
clear a ridge the model said it did not. `naigos/demo/los.py` now reconstructs
the sampled, dropped profile and the frame carries it; the page draws that curve
and marks the pinch point with its depth. 32 of the 96 vertices are sent, with
the pinch index always forced into the selection. The module imports numpy and
nothing else, so `tests/test_los_profile.py` can assert it agrees with
`env.terrain.los_clearance` (max 0.5 m over 64 rays) and have that mean
something.

### E-11-OLD — original writeup, kept for the record
The Cesium page shows lethal domes, detection rings and a per-aircraft track
meter, so you can see *that* an aircraft is being tracked. You cannot see *why*
-- whether the ray is clear or the ridge is breaking it.

**Done looks like.** A per-frame polyline from each aircraft to its strongest
tracker, coloured by the LOS clearance the env already computes, so the moment a
ridge cuts the ray is visible rather than deduced from a falling meter.

## 7. Engineering

### G-1 — Two overlapping data layers were built in parallel
An env-side `naigos/data/dem.py` and `naigos/data/cache.py` were written at the
same time as the research layer, duplicating the cited cache and the DEM path —
and carrying a north-south mirror bug the research agent caught. Both are deleted;
`theatre_bridge` now goes through `naigos/data/enu.py::real_terrain`. Recorded
here because the *process* failure (two agents on adjacent surfaces without a
declared owner) is more likely to recur than the bug.

**Done looks like.** A one-line ownership note per package in `docs/STACK.md`
before any further parallel work.

### G-2 — No CI
257 tests are collected on the current checkout. The suite is designed for no
network and no GPU; cache-dependent checks skip cleanly when raw data is absent.
Nothing runs it automatically.

**Done looks like.** A GitHub Actions workflow running `pytest -q` on push, with
the `data`/`rl` extras installed and the cache-dependent tests skipping cleanly
when `data_cache/` is absent (they already do).

### G-3 — Checkpoints are pickles
`train.py` pickles the raw param pytree. Fine locally, brittle across JAX or Flax
versions, and unsafe to load from an untrusted source.

**Done looks like.** `orbax` checkpointing, or at minimum `safetensors` for the
arrays plus JSON for the config.

### G-4 — `EnvConfig` changes force a full recompile
Because the config is a static argument, every curriculum promotion rebuilds the
jitted train step. At the current cadence that is a ~2 s cost every 10 iterations
— tolerable now, wasteful on a GPU run of thousands of iterations.

**Done looks like.** Move the curriculum knobs (`red_*_scale`, `n_threat_active`)
out of the static config into traced arrays carried in `EnvState`, so annealing
does not retrigger compilation.

**Now measured rather than estimated.** `perf.json` reports `recompiles` and
`recompile_s_total`, and those iterations are booked as compilation instead of
inflating reported throughput. The first GPU run will say what this actually
costs there; the ~2 s figure above is a CPU estimate.

### P-1 — Cloud-scheduled candidate learning: BUILT, NOT RUN
`naigos/pipeline/` plus `naigos/rl/modal_pipeline.py` and `scripts/pipeline.py`
schedule snapshot → candidate training → held-out evaluation and verifier →
gated, shadow-by-default promotion on Modal, so learning continues with the
laptop closed (docs/STACK.md, "Cloud learning pipeline"). **No remote run has
happened**: the binding loads against modal 1.5.5 and every decision path is
tested offline; nothing has been deployed, and no GPU has been billed.

Open items, in the order they would bite:

- **The DEM host is outside the allowlist.** py3dep 0.19 fetches 30 m 3DEP
  through GDAL from `prd-tnm.s3.amazonaws.com`; the `usgs_3dep` allowlist names
  `elevation.nationalmap.gov` and two USGS web hosts. The local cache's
  manifest records the declared URL, not the host the bytes came from. The
  pipeline fails a cloud DEM fetch closed and bootstraps from `pipeline.py
  seed` instead. Deciding whether USGS's staged-products bucket belongs on the
  allowlist (and recording the real fetch URL in the manifest) is a data-layer
  decision, deliberately not taken by a scheduler.
- **`block_network=True` on training and evaluation is unverified remotely.**
  Modal documents Dict/Function access as governed by `restrict_modal_access`,
  not `block_network`, so leases should keep working; if they do not, redeploy
  with `NAIGOS_PIPELINE_BLOCK_NETWORK=0` and the code path is still offline.
- **Cost per candidate is unknown** until the first nightly writes `perf.json`
  (C-1). The defaults are one `short` candidate a day and one 600-iteration
  candidate a week.
- **The smoke gate is shared with `modal_runs.py`.** It proves `modal_train.py`'s
  image for the commit, not the pipeline's training image, which differs only
  in carrying no baked data.

**Done looks like.** `modal deploy`, `pipeline.py seed`, one nightly reaching a
recorded decision, and `pipeline.py status` from a second machine showing it.

---

## 8. Suggested order

1. **C-1 → C-2** get onto Modal and run to full curriculum difficulty.
2. **L-3, E-6, E-7** repeat learned-versus-heuristic evaluation across seeds, frozen scenarios, theatres and routes.
3. **V-1** choose whether observability or survivability is the primary objective, then rebalance if needed.
4. **S-4** sweep the CBF margin rather than treating one filter configuration as definitive.
5. **D-1** run the sensitivity sweep that turns the assumed g-limit into a reported band.
6. **E-2, E-3, E-4** close the 2.5D interceptor hole, randomise geometry, and evaluate the second theatre.
7. **R-1** add self-play only after the blue policy has been replicated at full difficulty.

---

## 9. What would break first

If this had to be defended tomorrow, the three things that would not survive
scrutiny, in order:

1. **Single seed, single theatre, single geometry** (L-3, E-4, E-3). Every number
   is one sample from one map.
2. **The g-limit is a guess that the whole difficulty curve hangs on** (D-1) —
   defensible only as a reported band, which does not exist yet.
3. **The policy has not reached full red difficulty or a demonstrated plateau**
   (C-2), so the shipped result is a learning result, not a convergence claim.

Everything else is an upgrade. These three are the difference between "a real
system with real learning" and "a real system with a defensible result".
