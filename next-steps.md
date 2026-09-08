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
| 5. MARL learning curve | **Partial.** Real curves on CPU (below). Not run on Modal (C-1), and not run long enough to be a headline number (C-2). |
| 6. CBF backstop | **Done.** HOCBF-QP implemented, unit-tested (12 tests), and wired into `rollout` behind `--cbf`. A/B measured; QP infeasibility rate reported (S-2, S-4). |
| 7. Red team v2 / v3 | **v2 done, v3 not started.** Difficulty curriculum runs and advances. Learned red raises `NotImplementedError` (R-1). |
| 8. Learning-delta demo | **Partial.** `naigos/demo/replay.py` exists and replays real logged rollouts; it has not been run against a converged checkpoint (E-1). |

### The learning curve that exists today

Real theatre, curriculum level 0 → 0.20, 64 worlds × 4 aircraft, CPU:

| | naive direct route | trained (iter ~160) |
| --- | --- | --- |
| survival | 0.41 | 0.45 *(against a harder red)* |
| objective reached | 0.42 | 0.45 |
| shootdown rate | 0.51 | 0.26 |
| mean detection probability | 0.22 | 0.37 |

**Read this honestly.** The shootdown rate halves and the policy holds survival
while the red curriculum is being turned up, which is real learning. But it is
not yet a clean headline result: mean detection probability is *higher* than the
baseline (the policy is trading exposure for standoff rather than hiding), and
~10% of sorties are still lost to terrain. Both are addressed below (V-1, S-2).

For reference, a hand-written avoid-plus-nap-of-the-earth controller reaches 0.78
survival at level 0. That is the headroom the policy has not yet closed.

---

## 1. BLOCKING gaps

### C-1 — No GPU run has happened
`naigos/rl/modal_train.py` defines the image, the Volume and the entrypoint, and
mounts `components/` and `data_cache/` so the worker stays offline. It has never
been executed. Every number in this repo is from CPU.

**Why it matters.** §7 asks for a reproducible learning curve from Modal, and the
run lengths that would produce a converged policy (thousands of iterations at
256+ parallel worlds) are not practical on a laptop.

**Done looks like.** One `modal run` completing, `history.json` on the Volume, and
a throughput figure (env-steps/s and wall-clock per iteration) in the DEVLOG.

### C-2 — Training has not been run to convergence
Longest run so far: ~200 iterations at 64 worlds × 128 steps. The reward is still
climbing and the red curriculum is still advancing when the run ends.

**Done looks like.** A run where the curriculum reaches level 1.0 and the eval
metrics plateau, with the plateau visible in `history.json`.

### V-1 — The policy trades exposure for standoff instead of hiding
Mean detection probability under the trained policy (0.37) is *above* the direct
baseline (0.22) even though the shootdown rate halves. The policy has learned to
route around lethal envelopes, which is correct and valuable, but it has not
learned to use terrain — which is supposed to be the signature mechanic.

**Diagnosis to run first, before changing weights.** Log the altitude-AGL
distribution alongside the exposure. If the policy is flying high and wide, the
exposure weight is too cheap relative to the progress reward; if it is flying low
but in the open, the LOS term is not reaching the policy. The measured fact that
low flight halves exposure (0.27 → 0.14) says the mechanic works, so this is a
reward-balance or exploration problem, not a physics one.

**Done looks like.** Trained mean detection probability below the direct-route
baseline, with mean-minimum-AGL to match.

### ~~S-2 — The CBF is not wired into training or the demo~~ — DONE
Now wired. `NaigosEnv.rollout` takes an `action_filter(state, obs, action) ->
(action, feasible)`; `naigos.rl.cbf.make_policy_filter` builds one that reads
envelopes out of the **observation** (so the backstop is limited to what the
aircraft has sensed and never touches the ground-truth threat list) and reads
ego kinematics and the DEM from the state. `--cbf` on `scripts/train_theatre.py`
and `python -m naigos.demo.replay`.

**Measured**, real theatre, curriculum level 0, 48 worlds, identical seeds, naive
nap-of-the-earth controller:

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

**Residual work (now S-4).** The infeasibility rate should fall as the policy
improves, since a good policy does not fly into a pocket with no exit. Re-measure
it against a converged checkpoint (C-2); if it stays above ~20%, the envelope
`margin` is too aggressive for this threat density and should be annealed with
the curriculum.

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

### E-4 — Only one theatre
Owens Valley only. A policy trained on one 97×132 km patch of the Sierra may have
memorised that terrain.

**Done looks like.** A second AOI with different relief character (rolling rather
than alpine) fetched by the research agent, and a cross-theatre transfer number.

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

### S-4 — Infeasibility rate under a trained policy is unmeasured
See the S-2 table: 54% under a naive controller. Whether that is a property of
the threat density or of the controller is not yet known.

**Done looks like.** The same A/B against a converged checkpoint, and if the rate
stays high, `CBFConfig.margin` annealed alongside the red curriculum.

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

### E-1 — The demo has not been run against a converged checkpoint
`naigos/demo/replay.py` rolls out untrained, trained and direct-route policies on
identical seeds and the identical threat field, dumps the real logged trajectories
to JSON, and draws a plan view with lethal envelopes over the DEM. It has been
written but not run end-to-end on a converged policy, and there is **no 3D
viewer** — the current figure is a 2D plan view with altitude only implicit.

**Done looks like.** The figure and `demo.json` regenerated from a C-2 checkpoint,
plus a 3D replay (a small three.js page reading `demo.json`, or a matplotlib 3D
animation) showing the terrain-masking behaviour, which a plan view cannot.

### E-6 — No held-out evaluation protocol
Evaluation uses fresh keys but the same generator, so start points, objectives and
threat layouts come from the same distribution the policy trained on.

**Done looks like.** A frozen evaluation set of N scenarios (seeded once, stored),
reported separately from training-distribution eval.

### E-7 — The baseline could be stronger
The reported comparison is against a direct great-circle route. The hand-written
avoid-plus-nap controller is much stronger (0.78 vs 0.39 survival at level 0) and
is currently only used as a calibration probe.

**Why it matters.** Beating a naive baseline is a weak claim. Beating a competent
hand-written heuristic is the claim worth making.

**Done looks like.** The avoid+nap controller promoted to a first-class baseline
in `train.py::baseline` and reported alongside the direct route in every table.

---

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
192 tests exist and pass locally in ~37 s with no network and no GPU. Nothing runs
them automatically.

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

---

## 8. Suggested order

1. ~~**S-2** wire the CBF~~ — done; survival 0.31 → 0.61, terrain losses halved
2. **V-1** diagnose exposure-vs-standoff, then rebalance
3. **E-7** promote the avoid+nap baseline so progress is measured against something real
4. **C-1 → C-2** get onto Modal and run to convergence
5. ~~**L-1** fix the multiplier windup~~ — done; cost channel is terminal-only, λ now moves both ways
6. **E-1** regenerate the demo, add the 3D view
7. **L-3, E-6** seeds and a held-out set — everything before this is a single-seed anecdote
8. **D-1** the sensitivity sweep that turns the assumed g-limit into a reported band
9. **E-2, E-3, E-4** close the 2.5D interceptor hole, randomise geometry, add a second theatre
10. **R-1** self-play, last, on a converged blue

---

## 9. What would break first

If this had to be defended tomorrow, the three things that would not survive
scrutiny, in order:

1. **Mean detection probability is worse than the baseline** (V-1). The headline
   claim is "detection down *and* shootdowns down". Only half of it currently holds.
2. **Single seed, single theatre, single geometry** (L-3, E-4, E-3). Every number
   is one sample from one map.
3. **The g-limit is a guess that the whole difficulty curve hangs on** (D-1) —
   defensible only as a reported band, which does not exist yet.

Everything else is an upgrade. These three are the difference between "a real
system with real learning" and "a real system with a defensible result".
