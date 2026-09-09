# gaps.md — terrain-fidelity benchmark

Open issues in the work in `naigos/bench/terrain_resolution.py`,
`scripts/bench_terrain.py` and `docs/artifacts/terrain_bench.json`. Written
while the benchmark was being built, so it records what the measurement does
*not* establish as well as what it does.

Each gap has an ID, why it matters, and what "done" looks like — the same shape
as `next-steps.md`, which is where these should migrate once triaged.

---

## What the benchmark now establishes

Measured on two AOIs (`owens_valley`, `tehran_basin`) × four cell sizes
(1500 / 500 / 250 / 100 m) × three ray-march lengths (`los_samples` 96 / 192 /
384), 64 worlds × 128 steps, best-of-9 timings, 2000 line-of-sight rays per row
against the native 30 m DEM. macOS arm64, 12 CPU cores, JAX 0.11.1, CPU backend.

1. **Grid resolution is very nearly free; the ray march is what costs.**
   At `los_samples=96`, throughput is flat within noise from 1500 m to 100 m
   cells (relative 1.00 / 0.98 / 1.00 / 0.93 on Owens, 1.00 / 1.03 / 0.99 / 1.03
   on Tehran) while the heightmap grows 230×. Doubling `los_samples` roughly
   halves throughput (1.00 / 0.54 / 0.30 and 1.00 / 0.52 / 0.25 at 500 m cells).
2. **The coarse grid, not the march, is the binding line-of-sight error.**
   At a fixed 96 samples, refining 1500 → 100 m cuts the false-visible rate from
   3.65% to 1.05% (Owens) and 5.35% to 1.40% (Tehran) at no throughput cost. At
   a fixed 1500 m grid, raising 96 → 384 samples moves the same rate only
   3.65% → 3.15% and 5.35% → 4.90%, for 3.4× the compute. Sampling a ridge that
   has already been averaged away does not recover it.
3. **Physics and presentation want different answers**, which is why the
   recommendation names two: 100 m at `los_samples=96` for physics, and 100 m
   served at `/terrain n=1024` for presentation. The presentation constraint is
   the endpoint's lat/lon resample, not the DEM: at the shipped `n=512`, a 100 m
   env grid renders with a p95 of 25.5 m against a 30 m AGL floor.

Nothing above changed a default. Every committed RL number was produced at
1500 m / 96 samples (500 m / 96 for the published re-measure and the live demo)
and still is.

---

## B-1 — The recommendation is measured on line-of-sight geometry, not on policy outcomes — BLOCKING for adoption

The benchmark grades a resolution by how often it disagrees with the source DEM
about whether a ray is cut. It does **not** measure what changes downstream:
survival, objective rate, mean detection probability, the altitude the policy
settles at.

**Why it matters.** A 1.4%-vs-3.65% false-visible rate is a statement about the
sensor model, not about the task. The repo has already been bitten by the
converse — `docs/DEVLOG.md` records that an earlier resolution conclusion was
drawn from high-altitude rollouts where grid refinement genuinely does nothing,
and was wrong for the low-altitude case. A geometry benchmark can make the same
mistake in the other direction.

**Done looks like.** The committed checkpoint evaluated on the demo seeds at
1500 / 500 / 250 / 100 m, reported as a table beside the geometry numbers. If
survival and detection move by less than seed noise, the honest conclusion is
that the resolution choice is not load-bearing for the published claim, and the
benchmark should say so.

## B-2 — `false_visible_rate` thresholds are asserted, not derived

`ResolutionPolicy.max_false_visible_rate = 0.02` and
`max_soft_visibility_mae = 0.05` are numbers I chose. They are stated in one
place and the recommendation is a pure function of them, which is the most that
can be said for them.

**Why it matters.** 250 m at 96 samples fails on Tehran by 0.0210 against 0.0200.
One threshold digit moves the recommendation from 100 m to 250 m. A reader is
entitled to know the answer is that sensitive.

**Done looks like.** Either B-1 supplies an outcome-anchored budget (what
false-visible rate perturbs survival by less than seed noise), or the report
publishes the recommendation across a threshold sweep so the sensitivity is
visible rather than implicit.

## B-3 — The `ray_step_cells` heuristic predicts the opposite of what was measured

The advisory diagnostic says that at 100 m cells and 96 samples the march steps
13.5–17.4 cells per sample along the theatre diagonal, i.e. most of the grid is
never looked at. The end-to-end measurement says that same configuration is the
most accurate one at full throughput.

**Why it matters.** Both are reported and they disagree, so at least one is
being misread. The likely explanation is that a bilinear 100 m surface preserves
ridge *height* even when sampled sparsely, whereas a 1500 m surface has already
averaged the ridge into the valley and no sampling density recovers it — but
that is a hypothesis, not a measurement, and it is currently written into the
module as a comment rather than tested.

**Done looks like.** A ridge-crossing test: synthetic terrain with a single
known ridge of known width, sampled at each cell size, showing what fraction of
the ridge height survives resampling. That separates "the march missed it" from
"the grid never had it".

## B-4 — Only the shipped endpoint resample is measured, not the fix already proposed

Presentation is graded on `build_terrain_grid`'s lat/lon resample at
n ∈ {256, 512, 1024}. `next-steps.md` E-12 already proposes replacing it with
the ENU grid plus a small lon/lat→ENU warp table, which would be accurate to
under a metre and *shrink* the payload from 512 KB to 65 KB.

**Why it matters.** The presentation recommendation (100 m at n=1024, a 2 MB
payload) is the best answer available inside the current design. It is a worse
answer than the design change that is already written down. Recommending 2 MB
without saying that would be misleading.

**Done looks like.** E-12 implemented, then re-benchmarked; the expected result
is that the presentation cell size stops being constrained by the endpoint at
all.

## B-5 — CPU only, one machine, and it was a shared one

Every number is from a single macOS arm64 laptop on the CPU backend, and part of
the sweep ran while another job was training on the same machine. That is why
`throughput_stats` reports best-of-N with the spread beside it rather than a
median: the same configuration came back at 186k, 153k and 139k agent-steps/s
across three medians-based runs, which is enough spread to invert the decision.

**Why it matters.** The cell-size-is-free finding is a statement about this CPU's
cache behaviour. On a GPU, a 5.4 MB heightmap and a 14 KB one have very
different residency characteristics, and the conclusion could reverse.

**Done looks like.** The same sweep run on the Modal GPU worker (`C-1`), with
`device` memory stats populated — they are `null` here because the CPU backend
implements no allocator — and the two conclusions compared.

## B-6 — Line-of-sight rays are drawn uniformly, not from the flown distribution

`sample_ray_endpoints` draws both ends uniformly over the theatre, at a fixed
10 m AGL emitter and 150 m AGL target. Real rays run from threat sites, which
are placed by spawn geometry, to aircraft, which fly where the policy takes them
— and the trained policy climbs to ~5400 m (`next-steps.md` V-1), where masking
barely operates.

**Why it matters.** The measured false-visible rates describe a low-altitude
ray population. That is the right population for the mechanic the project is
about, but it is not the population the current policy generates, so the numbers
overstate how much this resolution choice affects *this* checkpoint.

**Done looks like.** A second ray set drawn from a logged rollout's actual
(threat, aircraft) pairs, reported alongside the uniform one. The gap between
the two is itself the interesting number.

## B-7 — `TerrainConfig.cell` and the demo's `--cell-m` are still one knob

`naigos.demo.live` passes a single `--cell-m` to `env_from_theatre`, so raising
presentation fidelity necessarily rebuilds the simulation grid, and vice versa.
The benchmark now shows the two want different values.

**Why it matters.** As long as they are one number, "make the globe look better"
silently changes the surface line-of-sight is computed against — which is the
class of defect E-9 already cost this repo once.

**Done looks like.** A separate presentation cell size threaded to
`build_terrain_grid` only, with a test asserting the physics grid is unaffected.
Deliberately **not** done here: the task was to improve the decision, not to
change the configuration behind the published results.

## B-8 — No CI, so the benchmark can rot silently

`scripts/bench_terrain.py` is run by hand. `tests/test_bench_terrain.py` covers
the pure functions, so the arithmetic cannot rot, but nothing re-runs the sweep
or checks that `docs/artifacts/terrain_bench.json` still matches the code that
produced it.

**Done looks like.** `--quick` wired into the CI workflow that `next-steps.md`
G-2 already asks for, asserting only that the sweep completes and the schema
validates — not the timings, which are not reproducible on a shared runner.
