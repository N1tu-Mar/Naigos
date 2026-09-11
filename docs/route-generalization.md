# Route generalization: diverse training routes and a frozen held-out route bank

Closes the infrastructure half of next-steps.md **E-3** (one shared west-to-east
route geometry) and **E-6** (no held-out evaluation protocol). This change builds
scenario and evaluation machinery only. No policy was retrained and no
performance claim is made. The retraining experiment comes next (see
"Running it").

## The problem

Every episode used to start on the west inset line and end on the east one, with
threats scattered around that one corridor. Evaluation, including the pipeline's
held-out seeds (P-1), drew from the same generator. So a held-out number meant
"fresh random seeds from the training distribution". It could not show whether
the policy had learned to route or only to fly east and dodge.

One nuance matters for reading any later result. The observation is ego-frame
(`naigos/env/obs.py`): the policy never sees an absolute heading. Route
direction still changes the task, though. Relief on a real DEM runs one way
(Owens Valley is a north-south valley). The map edges sit at different
body-frame distances, which the edge features see when `obs_edge_features` is on.
Route length and box shape interact, and the threat corridor is built along the
route. A policy can over-fit to all of that, and the held-out bank is designed to
measure it.

## What changed

| where | what |
|---|---|
| `EnvConfig.route_mode` | `"legacy"` (default): the original placement, code and key stream unchanged. `"diverse"`: each world draws one training route family. |
| `EnvConfig.route_train_families` | families training may draw (default: all six training families) |
| `EnvConfig.route_bearing_jitter_deg` | half-width of each family's bearing sector (default 10°) |
| `EnvConfig.route_lateral_frac`, `route_min_length_frac` | cross-track offset law and the guaranteed length (below) |
| `EnvState.route` | `(3,)`: bearing (rad), family index, legacy flag. Read only by `respawn` and per-family evaluation. |
| `NaigosEnv.reset_route(key, route)` | a reset along a given `[bearing, family]`, jit/vmap-safe; how the bank is played |
| `NaigosEnv.rollout(..., route=...)` | the same thing for a whole episode |
| `config.route_bank()` | the frozen bank, digest-checked |
| `train.evaluate_route_bank` | plays the bank; per-split, per-family metrics |
| `TrainConfig.route_eval` | opt-in: run the bank at every eval boundary, write `route_eval.json` |
| `scripts/train_theatre.py` | `--route-mode {legacy,diverse}`, `--route-eval` |

Nothing in `step`, the reward, the constraint channel, detection, LOS, terrain or
the observation reads the route. Simulation rules stay independent of any visual
data. Demo, pipeline and visual code are untouched.

## Route geometry (`route_mode="diverse"`, and every bank scenario)

Bearings are the direction of travel in the ENU grid, in the same convention as
`psi`: 0° flies toward +x (east) and 90° toward +y (north). Families are named
origin_destination.

1. Shrink the play box (the full grid, or this world's `map_randomize` box) by
   `spawn_inset_frac` (10%) on **every** side. The boundary ramp is
   `edge_margin_frac` (6%) of the short side, so both ends of every route sit
   outside the ramp in any direction. The validator refuses a config where the
   inset does not exceed the ramp.
2. Each aircraft's start lies on a line through the inset box's centre, parallel
   to the bearing and offset across the track by `s_i`, at the point where that
   line enters the inset box. Its objective lies on a line offset by `o_i`, at
   the point where that line leaves it. Due east, this is the legacy picture:
   west inset edge to east inset edge. Starts are spread evenly across the track
   with jitter. Objectives are drawn independently, as in legacy.
3. `|s_i|, |o_i| <= route_lateral_frac * m`, where `m` is the inset box's half
   short side. Every line then crosses the box's inscribed disk, so the
   along-track length is at least `2 m sqrt(1 - f^2)`. With the defaults that is
   **0.64 × the play box's short side**. `route_min_length_frac` (0.5) is checked
   against this bound when the env is built, and tests check it on every family.
4. The along-track reach is capped at the inset box's half long side, so a
   diagonal is never longer than a cardinal route along the long axis. The cap
   never binds on a cardinal route. On the default 192 km grid the worst-case
   route is about 179 km, similar to the legacy worst case, and short enough to
   fly at `v_max` inside a default-length episode.
5. The rest matches legacy: start 2,500 m and objective 1,000 m above local
   ground, initial heading straight at the objective, and threats spawned by
   the existing corridor rule (the mean start to the mean objective), which
   works in any orientation.
6. `respawn` re-tasks along the episode's own bearing. The threat field was
   built around that corridor. `EnvState.route[2]` tells it whether the episode
   used legacy placement, so a legacy env respawning a route-bank world still
   follows the bank route. Legacy worlds respawn bit-identically.

## Families and sectors

| family | bearing | used by |
|---|---|---|
| west_east | 0° ± 10° | training |
| south_north | 90° ± 10° | training |
| east_west | 180° ± 10° | training |
| north_south | 270° ± 10° | training |
| southwest_northeast | 45° ± 10° | training |
| northeast_southwest | 225° ± 10° | training |
| southeast_northwest | 135° (±6° in bank) | **held out**: the untrained diagonal axis (extrapolation) |
| northwest_southeast | 315° (±6° in bank) | **held out**: the untrained diagonal axis (extrapolation) |
| oblique | 22.5° + 45°k (±2° in bank) | **held out**: halfway between trained directions (interpolation) |

The rule (`config.route_protocol_problems`): no bearing that training can draw
may come within `ROUTE_SECTOR_GUARD_DEG` (5°) of a held-out bank bearing. The
tightest pair under the defaults is the 20.5° oblique against the west_east
sector edge at 10°, which clears the guard by 5.5°. `NaigosEnv` refuses to build
a config that breaks the rule, for example a jitter above 15.5°, a held-out
family in `route_train_families`, or an unknown family. It refuses rather than
warns.

## The frozen bank: `route-bank-v1`

`config._build_route_bank_v1()` enumerates the bank as a plain loop with no RNG.
`route_bank()` refuses to return it unless its canonical JSON hashes to
`ROUTE_BANK_V1_SHA256` (`d3b44303…86e7fa`). An edited bank therefore cannot keep
the v1 name. Changing the bank means adding a v2.

| split | scenarios | contents | seeds |
|---|---|---|---|
| `heldout` | 48 | southeast_northwest and northwest_southeast at 4 bearings × 4 seeds each (32), and 8 obliques × 2 bearings (16) | 710000–710047 |
| `train_geometry` | 48 | the six training families, 4 in-sector bearings × 2 seeds each | 720000–720047 |

A scenario is `(scenario_id, split, family, bearing_deg, seed)`. The seed
becomes `jax.random.PRNGKey(seed)`, and that key fixes everything the bearing
does not: synthetic terrain (when no DEM is pinned), the play box (when
`map_randomize` is on), cross-track offsets, the threat field and the red
team's dice. The ranges `[710000, 720000)` and `[720000, 730000)` are reserved.
`assert_route_bank_disjoint` refuses a training seed inside either range, and
`run(route_eval=True)` and `evaluate_route_bank(training_seeds=...)` both apply
it. The ranges are also disjoint from the pipeline's held-out seeds (900001+),
and a test checks that.

`train_geometry` is the in-distribution reference: training bearings on seeds
training never used. The difference between it and `heldout`,
`generalization_gap` in every result, is the route-generalization number. Only
that gap separates the geometry effect from the seed effect.

## What is held out, and what is still shared

Held out:

- **Route bearing.** Held-out scenarios fly directions that no training sector
  reaches, with a 5° guard. The untrained diagonal axis never occurs in
  training. The obliques never occur within their bank bearings ±5°.
- **Scenario seeds.** The seeds come from reserved ranges that no training run
  may use.

Still shared with training. This is why the result is a held-out *route
geometry* test and not a held-out *world*:

- the theatre DEM (or the synthetic terrain generator), the grid and the
  play-box rule, including `map_randomize` if training used it
- the threat generator: kinds, spawn weights, count, corridor scatter (22 km)
  and along-track placement law
- the scripted red policy and the red curriculum level at evaluation time (each
  `route_eval.json` entry records `red_level`)
- the start/objective construction rule, the cross-track offset law and the
  altitudes
- the airframe, detection, reward and constraint definitions

Other limits to state plainly:

- The bank's randomness comes from `PRNGKey(seed)` streams, while training draws
  keys by splitting `PRNGKey(train_seed)`. A key collision between the two is
  astronomically unlikely but not structurally impossible. The structural
  guarantee is the bearing rule, not the keys.
- A `legacy`-mode policy has trained on due east only. For it, every bank
  family except `west_east` is out of distribution, `train_geometry` included.
  Read its numbers per family.
- The pipeline's held-out evaluation (`naigos/pipeline/evaluation.py`) still
  uses the legacy generator. Wiring the bank into the pipeline is a separate
  change in files this work does not own.

## Reading the output

With `TrainConfig.route_eval=True` (`--route-eval`):

- `route_eval.json` holds the protocol descriptor (`train.route_protocol`), then
  one entry per evaluation: iteration 0 for both baselines (`direct`,
  `avoid_nap`), then the greedy learner at every eval boundary. Each entry holds
  `heldout` and `train_geometry`, each with `aggregate` and `by_family`, where
  every block is the same `rollout_metrics` dictionary `evaluate` reports. The
  entry also holds `generalization_gap`.
- `history.json` rows gain flat scalars `route_heldout_<metric>` and
  `route_train_geometry_<metric>` for survival, objective, shootdown, terrain,
  bounds, timeout and early exposure. The baseline row gains
  `direct_route_…` and `avoid_nap_route_…`.
- Route evaluation is reported and never used for training. It draws no key from
  the training chain and feeds neither curriculum, so a run's training curve is
  identical with it on or off, and a test checks that.
- Results are a deterministic function of the policy and the env config. Two
  calls return identical dictionaries, and a test checks that.

Without `route_eval`, a run writes exactly the files and history rows it always
did.

## Running it

Train on diverse routes with held-out route evaluation (local, real theatre):

```bash
uv run python scripts/train_theatre.py --out runs/routes-diverse \
    --route-mode diverse --route-eval
```

For the control, use the same command with `--route-mode legacy --route-eval`.
Compare `heldout.by_family` and `generalization_gap` between the two, and
against the baselines' iteration-0 entries.

To evaluate an existing checkpoint on the bank:

```python
import pickle
from naigos.env.flight_env import NaigosEnv
from naigos.env.theatre_bridge import env_from_theatre
from naigos.rl.checkpoint import obs_config_from_blob
from naigos.rl.ppo import greedy_policy
from naigos.rl.red_team import RedCurriculum
from naigos.rl.train import evaluate_route_bank

blob = pickle.load(open("runs/theatre/ckpt_000800.pkl", "rb"))
cfg, hmap, _ = env_from_theatre(n_threat=16, cell_m=1500.0)
cfg = RedCurriculum().apply(cfg, 0.0).replace(**obs_config_from_blob(blob))
env = NaigosEnv(cfg, hmap=hmap)
res = evaluate_route_bank(env, greedy_policy(blob["actor"], cfg), training_seeds=[0])
print(res["heldout"]["by_family"], res["generalization_gap"])
```

The env's `route_mode` does not matter here, because the bank sets every
route. Record the red level you evaluate at.

## Migration notes

- **Defaults are unchanged.** `route_mode="legacy"` keeps the original
  placement, its key stream, the golden rollout in
  `tests/test_map_randomization.py`, baselines and checkpoints.
- **`EnvState` gained a trailing `route` field.** Code that builds `EnvState(...)`
  by hand must pass it. `_replace`, tree maps and field access are unaffected.
  Inside this repository only `flight_env.py` builds one.
- **Checkpoints** store `dataclasses.asdict(env_cfg)`, so new ones carry the
  `route_*` keys. Older blobs lack them, which means legacy.
- **`run.json`** records `config.route_mode` / `config.route_eval` only when they
  are not the default. A default run's identity is unchanged. A resume or
  re-run that switches either one into an existing run directory is refused as
  a different run.
