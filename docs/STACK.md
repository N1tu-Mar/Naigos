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
naigos/rl/train.py -> modal_train.py           naigos/demo/replay.py
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
| invariant | everywhere | core | `pytest tests/test_invariant.py` |

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

## Two ways to run, and the difference matters

```bash
python scripts/train_local.py     # synthetic ridged terrain. tests, CI, domain randomisation.
python scripts/train_theatre.py   # the cited 3DEP DEM + calibrated radar classes.
```

Only the second produces a real-data result. `env_from_theatre` raises rather
than silently falling back to synthetic terrain, because a run that believes it
used a real DEM but did not is worse than a run that fails.
