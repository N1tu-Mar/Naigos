# CBF calibration sweep

Offline evaluation infrastructure for the runtime safety backstop in
`naigos/rl/cbf.py`. It measures the tradeoff between **CBF margin**, **active
threat density** and **red curriculum level** on one held-out seed set, and
writes a versioned JSON result plus a Markdown summary.

It changes no CBF behaviour. `cbf.py`, `train.py`, `red_team.py` and
`naigos/env/**` are imported and used as shipped. No constant is tuned here, no
policy is retrained, and nothing in this document claims an optimal margin --
the tool produces the evidence, and the evidence has to actually be run.

## Why

`CBFConfig.margin` defaults to 1500 m. Nothing in the repository measured that.
What *is* measured is that the trained policy's CBF-QP is infeasible 22.7% of
the time, and `cbf.py` carries the honest caveat that the filter guarantees
safety only while the QP is feasible. A wider margin inflates every envelope, so
it should push the aircraft further from lethal range *and* shrink the feasible
set. Which of those dominates is an empirical question with three inputs, so the
answer is a surface.

## The pairing property

This is the part that makes the table worth reading. Every cell of the grid, and
both arms within each cell, consume the same explicitly recorded seed list.
Three facts about the environment turn that into a genuinely paired comparison:

- `naigos/env/threats.py::spawn` draws all `n_threat` positions and *then* masks
  them with `active = arange(T) < n_threat_active`. Raising the density **adds**
  threats to a scenario; it does not move the ones already there.
- `RedCurriculum.apply` scales detection range, lethal range, reaction latency
  and speed. It changes what the threats can do, not where they are.
- The margin is a *filter* parameter. It never reaches `reset`. Two cells that
  differ only in margin are flown on byte-identical worlds.

So a margin column is an exact paired comparison, and a density or red column
holds terrain, start point, objective and threat placement fixed while varying
one thing. `tests/test_cbf_sweep.py` asserts all three.

Each cell is also flown twice: `cbf` (the filter in the loop) and `no_cbf`
(`action_filter=None`), on those same seeds. Comparing a filtered run against a
survival number remembered from some other run is how a filter gets credit for a
lucky draw.

## Usage

```bash
# laptop-sized: 2x2x2 grid, 8 seeds, 64 steps
uv run python scripts/sweep_cbf.py --quick

# the real thing
uv run python scripts/sweep_cbf.py \
    --checkpoint checkpoints/theatre_1000.pkl \
    --margin 0 750 1500 3000 6000 \
    --threats 6 12 16 \
    --red-level 0.2 0.6 1.0 \
    --n-seeds 32 --base-seed 20000 \
    --out docs/artifacts/cbf_sweep.json \
    --markdown docs/artifacts/cbf_sweep.md
```

Cost is `len(margins) x len(threats) x len(red_levels) x 2 arms` vmapped rollout
batches of `n_seeds` worlds. Keep the grid small first and confirm the shape of
the answer before paying for resolution.

As a library:

```python
from naigos.rl.cbf_sweep import SweepGrid, run_sweep, markdown_summary, write_result

payload = run_sweep(
    grid=SweepGrid(margins=(0.0, 1500.0, 3000.0), threat_counts=(6, 12), red_levels=(0.2, 0.6)),
    checkpoint="checkpoints/theatre_1000.pkl",
    n_seeds=32,
    base_seed=20_000,
)
write_result("docs/artifacts/cbf_sweep.json", payload)
print(markdown_summary(payload))
```

`run_sweep` also accepts `make_policy=(EnvConfig) -> policy` instead of a
checkpoint, for reference controllers and tests. It then *requires*
`policy_provenance`: a result whose policy cannot be identified is not a result.

## What each cell reports

Per arm, every key in `METRIC_KEYS`:

| key | meaning |
| --- | --- |
| `survival_rate` | fraction of sorties still alive at the end |
| `objective_rate` | fraction that reached the objective |
| `shootdown_rate` | killed by a threat |
| `terrain_rate` | flew below the AGL floor |
| `bounds_rate` | left the play area |
| `timeout_rate` | alive but never arrived |
| `fuel_loss_rate` | ran dry at any point (see note) |
| `ceiling_rate`, `stall_rate` | the remaining envelope violations |
| `mean_exposure`, `exposure_early`, `exposure_successful`, `cumulative_exposure_per_sortie` | the four exposure statistics `train.py` defines, with its biases documented there |
| `mean_lock`, `mean_min_agl`, `mean_agl_live` | track quality and clearance |
| `cbf_infeasible_rate` | fraction of live aircraft-steps where the QP had no solution |

Everything except the fuel/ceiling/stall row comes from
`naigos.rl.train.rollout_metrics`, called verbatim. That is deliberate: a sweep
that computed survival with its own formula would be comparing formulas with the
training history rather than comparing policies.

**The fuel note.** `out_of_fuel` is not a one-shot event. Running dry does not
kill the aircraft -- `flight_env.step` excludes only shot, terrain and bounds
from `alive` -- so the flag stays raised on every subsequent unfrozen step.
Summing it over time reports a "rate" of several hundred percent. The sweep
reports the fraction of sorties that ran dry *at any point*.

**Reading `cbf_infeasible_rate`.** On the `no_cbf` arm it is 0.0 by
construction: `rollout` fills `cbf_feasible` with `True` when there is no
filter. That is a fact about there being no filter, not evidence of feasibility.
On the `cbf` arm, a margin that buys survival while driving this number up has
bought it somewhere other than the barrier, and the result should be read as
such.

Each cell also carries `delta` (`cbf` minus `no_cbf`) for the headline keys, the
complete `env_config` it was flown on, and the complete `cbf_config`.

## The result file

```
schema             1
kind               "cbf_calibration_sweep"
created_utc        ISO-8601, second resolution        (volatile)
runtime.wall_s     wall clock                         (volatile)
notes              free text from --notes
grid               the three axes and the cell count
seeds              the full list, n, base, and the shared-across-* flags
policy             checkpoint path, sha256, bytes, iteration, trained env_cfg,
                   obs config -- or the callable's provenance
base_env_config    complete EnvConfig, nested dataclasses included
base_cbf_config    complete CBFConfig
red_curriculum     complete RedCurriculum
rollout            steps, worlds, blue, sorties per arm, arms, fixed_hmap
metric_keys        the metric schema, in report order
code               git commit and dirty flag
cells[]            one row per grid cell
```

`serialize` sorts keys and ends with a newline. `canonical_json` drops
`created_utc` and `runtime`, and two runs of the same sweep must produce the same
string from it -- that is what lets you diff two result files and see only what
actually changed. `load_result` refuses a schema it does not read.

## Checkpoint compatibility

The actor's input widths are functions of the environment config: the ego vector
is 10 or 14 wide depending on `obs_edge_features`, and each threat slot is
`9 + len(threat_kinds)` wide. The shipped `checkpoints/theatre_1000.pkl` was
trained on a five-class theatre, so evaluating it against a default synthetic
`EnvConfig` (three classes) is a shape error, not a worse measurement.

So `run_sweep` rebuilds the env config from the checkpoint's own `env_cfg`
(`env_config_from_blob`) unless you pass `base_env_cfg` explicitly, and then
checks both widths (`require_actor_compatible`). A mismatch raises
`SweepConfigError` naming both numbers rather than producing a table.

Size overrides (`--blue`, `--threat-capacity`, `--steps`) are applied on top of
the rebuilt config. `--threat-capacity` sets `EnvConfig.n_threat`, the padded
slot count, and is held fixed across the whole sweep so that the threat draw is
the same everywhere and density is a prefix of it.

## Errors

Everything below raises `SweepConfigError` *before* anything is compiled or
flown, so a long sweep does not fail on its last cell. The CLI turns it into
exit code 2 with the message on stderr.

- an empty, duplicated, negative or non-finite axis
- a red level outside `[0, 1]`
- a threat count above `EnvConfig.n_threat`
- an empty, duplicated or negative seed list
- neither or both of `checkpoint` and `make_policy`
- `make_policy` without `policy_provenance`
- a checkpoint path that does not exist
- an actor whose ego or threat-slot width does not match the config
- `load_result` on an unknown schema

## Scope

Evaluation only. This tool does not tune `CBFConfig`, does not change the
learned policy, and does not write a recommendation anywhere. Anyone quoting a
margin from it should quote the result file's commit, checkpoint sha256 and seed
list alongside it.
