# Curriculum stability

The red curriculum raises red's difficulty as blue learns to survive. It used to
do that from a single evaluation: one measured survival rate, compared against
`promote_survival` and `demote_survival`, one level change. This document
describes what replaced that, what stayed the same, and what a `history.json`
row now tells you.

Nothing here changes the simulation, the reward, the difficulty mapping, or what
a `red_level` of 0.5 means. It changes only **when** the level is allowed to
move and **what is written down** when it does.

> No claim is made that these defaults train a better policy. That is a claim
> about a training run and no new training run has been done. What is
> demonstrated is that the level no longer moves on one noisy measurement, and
> that every movement is auditable and reproducible across a resume.

## The problem

`evaluate()` measures survival over `eval_worlds` worlds of `n_blue` agents --
64 x 4 = 256 Bernoulli sorties by default. A policy whose true survival rate is
0.72 therefore produces measurements with a standard error near 0.03, and a
useful fraction of those land above `promote_survival` (0.75). Under the old
rule that draw *was* a promotion: red got harder, and the next evaluation
measured a policy that had been moved into a task it had not earned. The
opposite draw demoted a policy that was fine.

The same noise makes the fix's other half necessary. Damping alone is not
enough if the level can move again on the very next evaluation, so there is also
a gate on how many evaluations must agree.

## The rule

Two knobs stand between a measurement and a level change.

**Smoothing.** Each evaluation is folded into a state; the thresholds are
compared against the smoothed statistic, never against the newest measurement.

| `smoothing` | statistic |
|-------------|-----------|
| `window` (default) | mean of the last `window_size` raw survival measurements |
| `ewma` | `ewma_alpha * new + (1 - ewma_alpha) * previous`, seeded by the first measurement |
| `none` | the newest measurement -- no smoothing (this is the legacy statistic) |

**The evidence gate.** `min_evaluations` evaluations must have been observed
since the level last *moved* before it may move again. Below that the thresholds
are not consulted at all and the row says `insufficient_evidence` rather than
`held` -- "we did not look" and "we looked and it was inside the dead band" are
different facts about a run.

### Defaults

| field | default | why |
|-------|---------|-----|
| `smoothing` | `"window"` | a plain mean is the statistic that is easiest to read off a history row and to reason about at a threshold |
| `window_size` | `5` | at 3 a single catastrophic evaluation carries a third of the weight, which is on its own enough to drag a policy at 0.50 survival below `demote_survival`; at 5 it moves the mean by a fifth |
| `ewma_alpha` | `0.4` | only used when `smoothing="ewma"` |
| `min_evaluations` | `3` | three agreeing evaluations, which at the default `eval_every=20` is 60 iterations between level changes |
| `promote_survival` | `0.75` | **unchanged** |
| `demote_survival` | `0.35` | **unchanged** |
| `step_size` | `0.05` | **unchanged** |

The thresholds and the difficulty mapping (`detect_lo/hi`, `lethal_lo/hi`,
`latency_lo/hi`, `speed_lo/hi`, `n_lo/hi`) are deliberately untouched, so a
checkpoint resumes against the dead band it was trained under and a stored
`red_level` still means what it always meant.

### The dead band is still wide

`demote_survival=0.35` to `promote_survival=0.75` is a 0.40-wide hold band. A
policy that settles anywhere inside it never moves the level in either
direction, and smoothing does not change that -- it makes the band *more*
binding, because a lucky draw out of it no longer counts. If a run plateaus at,
say, 0.6 survival and sits there, that is the dead band, not the smoothing, and
narrowing it is the knob:

```python
run(..., red_curriculum=RedCurriculum(promote_survival=0.6, demote_survival=0.4))
```

Whether a narrower band trains better is an empirical question this change does
not answer, which is why the shipped defaults do not move it.

### A level change clears the window

When the level moves, the window and the EWMA accumulator are cleared and
`n_since_change` resets to zero. Samples taken before a level change measured a
*different task*: red's detection reach, lethal radius, reaction latency, speed
and active count are all functions of the level. Averaging across that boundary
would smooth two difficulties together, and is the one way a smoothed curriculum
can be more wrong than an unsmoothed one.

The consequence is that the first `min_evaluations` evaluations after any level
change report `insufficient_evidence`.

### What smoothing does not buy

A window mean is not a robust statistic. An outlier still moves it by
`1/window_size`, so a bad draw arriving when the policy is *already* close to a
threshold can still tip it. What the window buys is that one draw from a policy
in the middle of the dead band cannot -- which is the case that was actually
breaking runs. `tests/test_curriculum_stability.py` asserts both halves of that,
including the limit.

### Survival, not `1 - shootdown_rate`

Unchanged and load-bearing. A policy that has swapped being shot down for flying
into a ridge has a *low* shootdown rate and a low survival rate; thresholding
`1 - shootdown_rate` promoted exactly such a policy, survival collapsed from
0.51 to 0.10, and the run never recovered. Both the smoothed and the legacy
paths threshold measured survival.

## What a history row says

Every row the curriculum touches carries the audit. Rows at an evaluation
boundary carry what was measured; rows where the curriculum ticked also carry
the verdict. When both happen at the same iteration they are one row.

| key | meaning |
|-----|---------|
| `red_level` | the level the row's metrics were **measured at** |
| `curriculum_raw_survival` | this evaluation's survival rate, undamped |
| `curriculum_smoothed_survival` | the statistic the thresholds were compared against; `null` before the first evaluation |
| `curriculum_smoothing` | `window`, `ewma` or `none` |
| `curriculum_window_n` | how many evaluations that statistic rests on |
| `curriculum_evals_total` | evaluations observed over the life of the run |
| `curriculum_evals_since_change` | evaluations observed since the level last moved |
| `curriculum_min_evaluations` | the gate in force |
| `curriculum_reason` | `promoted`, `demoted`, `held` or `insufficient_evidence` |
| `red_level_before` / `red_level_after` | the tick's effect |
| `curriculum_state` | the complete post-event state (see below) |

Both the raw and the smoothed number are recorded because a row showing only the
smoothed one would leave an audit unable to tell a genuine plateau from a window
that is merely lagging.

`red_level` is *not* restamped by a tick merged onto an evaluation row. If it
were, every promotion row would attribute the old theatre's metrics to the new
one.

A curriculum tick that falls between evaluations (`curriculum_every` and
`eval_every` need not divide each other) appends its own row, tagged
`"phase": "curriculum"`, rather than going unrecorded.

## Resume

`curriculum_state` is not only telemetry -- it is the recovery carrier.

```json
{"level": 0.15, "window": [0.71, 0.68], "ewma": 0.694,
 "n_observations": 11, "n_since_change": 2,
 "last_observed_iteration": 220, "last_reason": "insufficient_evidence"}
```

The state changes only at an evaluation and at a curriculum tick, and both write
a row, so **the newest row that has a `curriculum_state` is the live state**. A
resume reads it back with `train.curriculum_state_from_history` off the
checkpoint's own `recovery["history"]`. There is deliberately one copy: a
parallel checkpoint field could disagree with the audit trail, and then neither
could be trusted.

The state is plain JSON scalars and survives a `json.dumps`/`json.loads` round
trip exactly, so a resumed run's accumulator is bit-identical and it crosses
thresholds on the same evaluations the uninterrupted run would have.

### Resuming a checkpoint written before this change

Such a checkpoint carries no smoothing state. Its stored `red_level` was
produced under a rule where one evaluation could move the level; continuing it
under a rule that needs several is a different run. That is not decided
silently -- the resume is refused and names both ways forward:

```
ckpt_000200.pkl carries no red-curriculum smoothing state (it was written before
the curriculum kept one), and this run is configured with 'window' smoothing and
min_evaluations=3. Resuming would change what the stored red level 0.35 means [...]
  * run(..., red_curriculum=RedCurriculum.legacy()) to continue under the
    single-evaluation rule the checkpoint was trained with; or
  * TrainConfig(curriculum_resume='reset') to adopt the smoothed rule,
    starting from an empty window.
```

`curriculum_resume='reset'` prints what it is doing and starts an empty window,
so the level cannot move until `min_evaluations` fresh evaluations have been
observed.

Independently of this, `checkpoint.curriculum_compatibility` still reports a
curriculum whose *fields* differ from the checkpoint's, which is what
`scripts/modal_runs.py resume` gates on.

## Legacy behaviour

`RedCurriculum.legacy()` is `smoothing="none"`, `window_size=1`,
`min_evaluations=1`: one evaluation, one decision, identical to the rule this
replaced. It is never the default and has to be asked for:

```python
run(env_cfg, ppo_cfg, train_cfg, red_curriculum=RedCurriculum.legacy())
```

The stateless `RedCurriculum.update(level, survival_rate)` is kept verbatim and
is the function that path calls. A legacy run writes the same audit columns as a
smoothed one -- it is a different rule, not a different amount of
accountability -- so one history format reads both.

## Validation

`RedCurriculum.__post_init__` refuses a configuration that cannot behave, at
construction time rather than at the first tick: an unknown `smoothing`, a
survival threshold outside `[0, 1]`, a `demote_survival` at or above
`promote_survival` (an inverted or empty dead band would promote and demote on
the same measurement), a non-positive `step_size`, a `window_size` below 1, an
`ewma_alpha` outside `(0, 1]`, a `min_evaluations` below 1, or `n_lo > n_hi`.
`observe` refuses a survival rate outside `[0, 1]`. `TrainConfig.curriculum_resume`
is checked before any training happens.

An unmeasured statistic is `None`, never `0.0`. `demote_survival` is 0.35, so a
zero would read as a measured collapse and demote a run for not having been
evaluated yet.

## JAX

Curriculum bookkeeping is host-side Python and always was. Nothing here is
traced, and `RedCurriculum.apply` -- the part the env is a function of -- is
unchanged.

## Tests

```
uv run pytest tests/test_curriculum_stability.py tests/test_resume.py -q
```

`tests/test_curriculum_stability.py` is the state machine: outliers, repeated
evidence, the gates, the bounds, the statistic, the telemetry and the
serialization. `tests/test_resume.py` adds the claims that need a real
interrupted run: the state survives the interruption, the resumed run's rows
match, and the pre-smoothing migration is explicit in both directions.
