# Naigos — build spec for Claude Code

**Naigos** is the [Nomos](https://github.com/N1tu-Mar/nomos) engine lifted into 3D contested airspace. Where Nomos learned decentralized _traffic coordination_ (cars sharing a real road graph without crashing), Naigos learns _survivable routing_: aircraft reaching their objectives through a battlefield of active threats — SAM/radar envelopes, mobile tanks, drone interceptors — while an **adaptive red team** actively hunts them. It is a two-sided multi-agent pursuit-evasion problem.

The measurable win: **detection probability and shootdowns down, objectives-reached and sorties-preserved up, vs. a naive direct-route baseline.**

---

## 0. Hard invariant — read first, never violate

**The blue agent is purely evasive. It has NO weapon, NO fire/strike/engage action, ever.**

This is a non-negotiable design constraint, not a phase-2 feature:

- The blue action space contains **only** flight controls (heading/pitch/bank/throttle setpoints). There is no "engage", "fire", "suppress", "retaliate", or "target" action — not now, not behind a flag, not as an option.
- The agent survives through **routing, timing, terrain masking, and maneuver** — never by removing a threat.
- The red team (threats) is the _environment_. Blue never controls red and never attacks it.
- If any part of the build starts to add an offensive/target-selection capability to blue, **stop and flag it** rather than implement it. The constraint is what makes the RL problem hard and the project defensible; removing it is a regression, not a feature.

Everything below assumes this invariant holds.

---

## 1. Architecture — reuse the Nomos spine

Nomos's proven stack transfers almost 1:1. Keep the parts that worked:

| Nomos                                                                      | Naigos                                                                      |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Vectorized JAX kinematic env (cars on OSM graph)                           | Vectorized JAX flight env (aircraft in 3D airspace over real terrain)       |
| Decentralized Dec-POMDP, local obs                                         | Same — each aircraft sees a local, partial view                             |
| **MAPPO/IPPO under CTDE** (shared actor, centralized critic)               | Same                                                                        |
| Permutation-invariant **Deep Sets + attention** over variable neighbor set | Same, over the variable **threat + friendly** set                           |
| Multi-objective reward + **curriculum annealing**                          | Same (reward table in §5)                                                   |
| **CMDP**: hard constraints via a deterministic **verifier** cost channel   | Same — "shootdown / lethal-envelope entry" replaces "crash"                 |
| **CBF-QP** runtime safety backstop                                         | Same — keep-out of known lethal envelopes                                   |
| **Learning-delta demo** (untrained gridlock → trained smooth flow)         | Same — untrained gets shot down → trained survives and reaches objective    |
| Real data layer (OSMnx / proximity / traffic)                              | Real data layer (terrain DEM / airspace / flight tracks / weather) — see §8 |

**Critical path (the spine):**
`terrain+airspace data → 3D flight env → MARL (+reward/CMDP) → Modal GPU training → trained evasion policy → CBF backstop → learning-delta demo`

**What's genuinely new vs. Nomos (the hard, interesting parts):**

1. **3D kinematics** — a fixed-wing/rotary point-mass airframe with g-limits, climb/turn-rate limits, min speed (stall). Harder than the 2D bicycle model.
2. **Terrain-masked detection** — line-of-sight radar occlusion off a real elevation model. This is the signature mechanic: nap-of-the-earth flight to break radar LOS.
3. **Adaptive red team** — the threats are a _second learning/curriculum-driven half_, so blue evades a moving target, not a static field.

---

## 2. Environment spec (`naigos/env/`)

Build a **vectorized JAX** env, pure `reset`/`step` over one world, `vmap` over worlds — exactly the Nomos contract (`jit`/`vmap`-able, fixed-size padded arrays, host-side routing handed in as arrays).

**Blue airframe (point-mass 3D).** State: position `(x,y,z)`, velocity vector, heading, pitch, bank, speed, fuel. Actions (bounded setpoints, matching Nomos's "setpoints not raw torque" choice): target heading-rate, target climb-rate/pitch, target speed/throttle, bank. Enforce:

- max load factor (g-limit) → caps turn rate at a given speed,
- stall speed (min airspeed), service ceiling (max alt), climb/descent rate limits,
- fuel burn as a function of throttle and maneuvering.

**Threats (the red environment).** Parameterized, **not** a real weapons database (see §8 guardrail):

- **Static SAM/radar sites** — a position, a detection model (range + altitude envelope + a detection-probability curve that ramps with exposure and shrinks with terrain masking / low altitude), and a lethal engagement envelope.
- **Mobile threats** (tanks, mobile SAMs) — the airspace analog of Nomos's moving pedestrians/cars: dynamic ground units with their own kinematics and shorter-range envelopes.
- **Interceptor drones** — mobile pursuers with their own simple kinematics that vector toward detected blue aircraft (the adaptive core, see §4).

**Detection model (the signature mechanic).**

- Detection probability per threat per step = f(slant range, aspect, blue altitude, **terrain line-of-sight**). Break LOS with terrain → detection drops. This is where the real DEM earns its place (§8).
- Accumulated exposure feeds a **track/lock** state; sustained lock inside a lethal envelope → shootdown (the hard constraint).

**Observation (per blue agent, local + partial).** Ego vector (speed, alt, heading err, fuel, distance/progress to objective) + a **variable-length set** of the K nearest known threats (ego-frame relative position/velocity, threat type, estimated envelope, current detection/lock state) + nearby friendlies. Partial observability: a threat the aircraft can't currently sense (masked / out of range) is not in the obs — this is a Dec-POMDP.

**Termination.** Objective reached (arrive), shot down (hard-constraint terminal), out of fuel, or bounds/ceiling violation. Mirror Nomos's finite-cohort-then-freeze vs. continuous-respawn choice — start finite (one sortie per aircraft), add persistent tasking later if useful.

**Spatial hashing** for neighbor/threat search → O(N·C), so it scales to many aircraft + many threats like Nomos scaled to thousands of cars.

---

## 3. Network (`naigos/rl/networks.py`)

Copy Nomos's design directly: **shared-parameter actor + centralized critic (CTDE)**, with a **permutation-invariant encoder** (Deep Sets: masked mean+max pool; optional ego-query attention) over the variable threat/friendly set so the policy is agnostic to how many threats are nearby. The centralized critic sees the full scene at train time (all blue states + all threat states); each aircraft acts from local obs at run time. This is what lets anticipation/evasion emerge without a separate "predict the threat" model.

---

## 4. Adaptive red team (the two-sided game)

This is what replaces "add a fire button" — the tension comes from a threat side that gets _smarter_, not from blue shooting back.

Design it as a **curriculum from scripted → adaptive**:

1. **Scripted red (v1):** static SAMs + interceptors using a simple pure-pursuit / proportional-navigation vector toward the nearest detected blue. Enough to make evasion non-trivial.
2. **Parameterized-difficulty red (v2):** curriculum knobs — detection range, reaction latency, interceptor speed, number of threats — annealed up as blue improves (mirrors Nomos's collision-weight-then-efficiency curriculum).
3. **Learned red / self-play (v3, stretch):** red interceptors are their own RL policy trained to maximize blue detection+shootdown, blue trained to minimize it. Alternating / population-based self-play. This is the richest version and the strongest résumé line — competitive multi-agent pursuit-evasion — but treat it as an upgrade after v1/v2 give a clean learning curve.

Keep red and blue as **separate policies with separate optimizers**; never let blue's action space leak into red's job or vice-versa.

---

## 5. Reward + CMDP (`naigos/rl/reward.py`, `verifier.py`)

Multi-objective weighted sum, **curriculum-annealed** (survival dominates first, efficiency fades in once shootdown-rate drops) — the exact Nomos pattern.

| component                                       | sign         | why                                                                                 |
| ----------------------------------------------- | ------------ | ----------------------------------------------------------------------------------- |
| **not-shot-down / stay out of lethal envelope** | − (dominant) | the hard constraint; routed through the CMDP cost channel                           |
| **detection-probability / radar-exposure**      | −            | continuous ramp (the APF-spacing analog) — punishes exposure _before_ a lock        |
| **progress to objective**                       | +            | reward progress toward the goal, not raw speed (Nomos's anti-circle-farming lesson) |
| **objective reached**                           | +            | sparse arrival bonus                                                                |
| **maneuver feasibility / fuel**                 | +            | stay within g-limits, don't waste fuel; keeps flight realistic                      |

**CMDP via a deterministic verifier** (copy Nomos's `verifier.py` principle: _verify the trace, don't re-simulate_). Shootdown, ceiling/stall/bounds violations, and lethal-envelope dwell are **constraints** with their own cost channel, fed to a **PPO-Lagrangian** objective so the collision (shootdown) cost is a learned hard constraint, not a hand-tuned reward weight.

**Reward-hacking watchlist** (Nomos-style):

- Fly to the map edge forever to trivially "survive" → progress-to-objective + fuel/time penalty fixes it.
- Circle in a safe corner → progress, not raw survival-time, is rewarded.
- Miscalibrated detection footprint → phantom/missed shootdowns; **calibrate the detection + LOS model first**, before tuning weights (Nomos learned this the hard way with collision-radius vs. lane-width).

---

## 6. Safety backstop (`naigos/rl/cbf.py`)

A **CBF-QP filter** as the last-resort runtime backstop, exactly as Nomos uses it for cars: `a_exec = filter(a_policy, state)`. Barrier per known lethal envelope: keep the aircraft outside it; the QP returns the action _closest to the policy's_ subject to the barrier, using turn + climb + throttle jointly. Use a **higher-order CBF** (the controls enter through the 2nd derivative, as in Nomos's bicycle HOCBF). Same honest caveat applies: the filter only guarantees safety _while the QP is feasible_ — at high threat density the feasible set can be empty, so the filter's value is bounded by how good the learned policy is. It's a backstop, not the plan.

---

## 7. Compute + training

Modal GPU, exactly like Nomos: rollout workers + learner + checkpoints to a Volume, curriculum-annealed, producing a **reproducible learning curve** (shootdown-rate down, objectives-reached up). Keep the fast kinematic env as the training substrate so a run finishes in reasonable time. Any GPU.

---

## 8. Research agent — realistic data layer (`naigos/research/`, `naigos/data/`)

Naigos should ground itself in **real, open, physics-based data**, the way Nomos grounded itself in real OSM/traffic data and cited every source in `components/*.json`. Build a **research sub-agent** (a Claude Code workflow/sub-agent) whose entire job is to fetch, cache, and cite that data, then emit structured specs the env consumes.

**Design of the research agent:**

- **Source allowlist (fixed).** The agent may only pull from an explicit allowlist of open sources (below). No open-ended "search the whole web for weapon specs." This keeps the data reproducible, legal, and non-sensitive.
- **Fetch → cache → cite.** Every pull is cached to `data_cache/` (raw) and distilled into one `components/<source>.json` spec (schema copied from Nomos `components/README.md`: `id/role/inputs/outputs/decision/license/sources/...`). **Nothing enters the env without a cited source.**
- **Structured output.** The agent's deliverable is these cited JSON specs + cached files + a short `docs/DATA.md` provenance log — not prose.
- **Idempotent + offline-after-first-run.** Cache so training never depends on a live network (Nomos's "cache the graphml" lesson).

**Allowlisted data sources (the good, open, non-sensitive ones):**

- **Terrain elevation (the key one)** — SRTM / Copernicus DEM / USGS 3DEP via `py3dep` (Nomos already used `py3dep`). Feeds **radar line-of-sight masking** and terrain-following routes. This is the highest-leverage dataset in the project.
- **Airspace + airfields** — OurAirports (CSV, public domain) and/or OpenAIP for airport/waypoint geometry → objective and start locations, no-fly structure.
- **Real flight kinematics** — OpenSky Network API → real aircraft speeds, climb rates, turn behavior to **calibrate the airframe model** (so g-limits/speeds are realistic, not invented).
- **Atmosphere/weather** — an open weather API → ceiling, and air density for propagation/performance modeling.
- **Detection physics** — the open **radar range equation** and published propagation/terrain-masking literature → build the detection-probability model from first principles, parameterized (transmit power, aperture, RCS, terrain-LOS) rather than copied from any product.

**Guardrail for the research agent (bake this in):** threat envelopes are **parameterized abstractions** — range, altitude band, reaction latency, detection-probability curve — calibrated only from _open, published, nominal_ figures for generic/legacy systems where needed. The agent does **not** assemble a current, precise, targeting-grade capability database, and the simulation never needs one: the RL problem cares about the _shape_ of the tradeoff (exposure vs. survival), not real-world accuracy against a specific fielded system. If a data request drifts toward "precise current capabilities of a specific weapon to defeat it," the agent declines and parameterizes instead.

---

## 9. Engineering practices

- **Commit frequently.** Make a git commit after every meaningful unit of progress — each vertical slice, each passing test batch, each env/reward change, each bug fix — with a short descriptive message. Small, frequent, legible commits over big infrequent ones. Treat the commit history as part of the deliverable (it tells the story of the build, the way Nomos's DEVLOG does).
- **Test as you go.** Mirror Nomos's ~170-test suite: smoke tests that `reset`/`step` are `jit`/`vmap`-able and produce correct shapes; verifier/CMDP tests are pure-numpy (no JAX) so they run standalone; env-geometry tests (spawn non-overlap, detection-LOS correctness) _before_ trusting the reward.
- **DEVLOG.** Keep `docs/DEVLOG.md` — one ~two-sentence entry per learning/architectural-change/improvement, newest at bottom, exactly like Nomos. It's where the "why" lives.
- **Layered installs.** `pyproject.toml` extras (`data` / `rl` / `demo` / `research`) so each slice verifies on its own before the next pulls in.
- **Components-as-graph.** One `components/*.json` per design decision (env, reward, network, cbf, red-team, each data source), cited — so the design is reasoned over as a graph like Nomos.

---

## 10. Build order

1. **Terrain + airspace ground truth** (research agent) — DEM pull + cache, airfield/objective geometry, provenance JSON. (hours)
2. **3D flight env** — point-mass airframe, g/stall/ceiling limits, calibrated from OpenSky. Smoke-test `vmap` over worlds. (2–3 days)
3. **Detection + terrain-LOS model** — the signature mechanic; **calibrate before rewarding**. Tests that masking actually drops detection. (1–2 days)
4. **Threats v1 (scripted red)** + **reward/CMDP v1** — survival-dominant, PPO-Lagrangian. Start few aircraft / few threats. (2–3 days)
5. **MARL on Modal** — shared-param MAPPO/CTDE, curriculum-anneal exposure/efficiency → **learning curve**. (hours to wire)
6. **CBF backstop** — HOCBF keep-out of lethal envelopes. (1–2 days)
7. **Red team v2/v3** — difficulty curriculum, then self-play if time. (open-ended)
8. **Learning-delta demo** — 3D replay of _actual logged rollouts_: untrained aircraft gets detected + shot down; trained aircraft terrain-masks, evades, reaches objective. Live counters (shootdowns=0, detection-prob, objectives-reached). Render real rollouts, nothing scripted (Nomos's honesty rule). (1–2 days)

**Long poles to start day 1:** the DEM/LOS pipeline and the vectorized 3D env.

---

## 11. Repo layout (target)

```
naigos/
  env/         # JAX 3D flight env, airframe kinematics, detection+LOS, spatial hash
  rl/          # ppo.py (MAPPO + PPO-Lagrangian), cbf.py (HOCBF), networks.py (DeepSets+attn),
               # verifier.py (deterministic CMDP cost), red_team.py, modal_train.py
  data/        # DEM / airspace / flight-track loaders (consume research-agent cache)
  research/    # the research sub-agent: allowlisted fetch → cache → cited JSON specs
  demo/        # 3D viewer + learning-delta artifact (untrained vs trained rollouts)
components/    # one cited JSON per design decision + per data source
docs/          # DEVLOG.md, DATA.md (provenance), STACK.md, architecture notes
scripts/ tests/  # eval/export utilities + the test suite
data_cache/    # cached raw data (idempotent, offline-after-first-run)
```

---

## 12. The honesty check (Nomos-style — answer these before claiming done)

- **Is this really training something?** Yes — a real evasion policy with a reproducible learning curve (shootdown-rate ↓, objectives-reached ↑), optionally a real red policy via self-play.
- **Is the demo real?** Yes — logged rollouts replayed, not scripted. Untrained → shot down; trained → survives.
- **Is the validation a tradeoff, not a claim?** Yes — detection-probability / shootdown-rate vs. objectives-reached / sorties-preserved, against a naive-direct-route baseline. (Matches the performance/safety-tradeoff framing.)
- **Did the invariant hold?** Blue never gained a weapon. If it did, that's a bug, not a feature.
