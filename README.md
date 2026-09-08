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
| sorties surviving | 0.156 | **0.651** | 0.240 | 0.688 |
| objectives reached | 0.156 | **0.651** | 0.240 | 0.688 |
| shootdowns (of 192) | 141 | **19** | 135 | 57 |
| mean detection probability | 0.478 | **0.300** | 0.333 | 0.202 |
| mean min AGL (m) | 1207 | -50 | 1255 | 262 |

Against the naive direct route: shootdowns fall 7.1x, objectives-reached rise
2.7x, detection probability falls — the tradeoff `prompt.md` asks to be measured,
in the direction it asks for. The demo prints this table itself; it is not
transcribed by hand.

![learning delta](docs/artifacts/learning_delta.png)

Raw numbers: [`docs/artifacts/summary.json`](docs/artifacts/summary.json),
[`summary_cbf.json`](docs/artifacts/summary_cbf.json), and the full training curve in
[`history.json`](docs/artifacts/history.json). `runs/` is gitignored; these are the
committed copies.

Three caveats that belong next to that table, not in a footnote:

- A **hand-written avoid-plus-nap heuristic is still competitive** — it edges the
  policy on survival (0.688 vs 0.651) and clearly wins on detection probability
  (0.202 vs 0.300) and on terrain losses (0 vs 20). It is a first-class baseline
  in the demo and in every training eval, because beating only the naive route
  would be a weak claim.
- **Single seed, single theatre, single route geometry.**
- The policy wins by **climbing above the short-range engagement ceilings**, not
  by terrain masking. That is a legitimate tactic it found on its own, but it is
  not the signature mechanic the design intended. Diagnosed in detail in
  [next-steps.md](next-steps.md) V-1.

## Demo it in one command

The trained policy ships with the repo (`checkpoints/theatre_1000.pkl`, 1.9 MB),
so nothing has to be trained to see the result.

```bash
uv venv --python 3.12
uv pip install -e '.[all]'

uv run python -m naigos.demo.replay --checkpoint checkpoints/theatre_1000.pkl
```

~14 s on a laptop CPU. It rolls out four policies — **untrained**, **trained**,
the **naive direct route** and a **hand-written avoid+nap heuristic** — on
identical seeds over the identical threat field, prints the comparison table
reproduced above, and writes `runs/demo/learning_delta.png` (the plan view) plus
`demo.json` with the raw logged trajectories.

```
  metric                      untrained        TRAINED   direct route      avoid+nap
  ----------------------------------------------------------------------------------
  sorties surviving               0.156          0.651          0.240          0.688
  objectives reached              0.156          0.651          0.240          0.688
  shootdowns                        141             19            135             57
  terrain losses                     14             20             11              0
  out-of-bounds losses                7             28              0              3
  mean detection prob             0.478          0.300          0.333          0.202
  mean track quality              0.549          0.318          0.384          0.244
  mean min AGL (m)                 1207            -50           1255            262

  vs the naive direct route:
    shootdowns          135 -> 19   (7.1x better)
    objectives reached  0.240 -> 0.651   (2.7x better)
    detection prob      0.333 -> 0.300
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
| Training | Real MAPPO-Lagrangian runs with a reproducible learning curve, on CPU (1000 iterations, ~450k env-steps/s). No GPU run yet. |
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
naigos/demo/      learning-delta replay of logged rollouts
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
