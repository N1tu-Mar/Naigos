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

`runs/theatre5/` — 1000 MAPPO-Lagrangian iterations on the cited Owens Valley
theatre. Evaluated on held-out demo seeds, 48 worlds x 4 aircraft, with identical
seeds and an identical threat field across all three policies:

| | untrained | **trained** | naive direct route |
| --- | --- | --- | --- |
| sorties surviving | 0.156 | **0.651** | 0.240 |
| objectives reached | 0.156 | **0.651** | 0.240 |
| shootdowns (of 192) | 141 | **19** | 135 |
| mean detection probability | 0.478 | **0.300** | 0.333 |

Shootdowns fall 7x, objectives-reached rise 2.7x, detection probability falls —
the tradeoff `prompt.md` asks to be measured, in the direction it asks for.

![learning delta](docs/artifacts/learning_delta.png)

Raw numbers: [`docs/artifacts/summary.json`](docs/artifacts/summary.json),
[`summary_cbf.json`](docs/artifacts/summary_cbf.json), and the full training curve in
[`history.json`](docs/artifacts/history.json). `runs/` is gitignored; these are the
committed copies.

Three caveats that belong next to that table, not in a footnote:

- A **hand-written avoid-plus-nap heuristic still beats the policy** (0.816
  survival, 0.268 detection). It is a first-class baseline in every eval.
- **Single seed, single theatre, single route geometry.**
- The policy wins by **climbing above the short-range engagement ceilings**, not
  by terrain masking. That is a legitimate tactic it found on its own, but it is
  not the signature mechanic the design intended. Diagnosed in detail in
  [next-steps.md](next-steps.md) V-1.

## Quick start

```bash
uv venv && uv pip install -e '.[rl,dev]'
pytest -q                                       # 192 tests, no network, no GPU
python scripts/train_local.py --iterations 200  # synthetic terrain
python scripts/train_theatre.py                 # the cited 3DEP DEM
python -m naigos.demo.replay --checkpoint runs/theatre1/ckpt_000700.pkl
```

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
| CBF backstop | Implemented and wired. Measured both ways: it roughly halves losses for a naive controller and *costs* objective rate for a trained one. QP infeasibility is reported, not hidden. |
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
docs/             DEVLOG.md, DATA.md (provenance), STACK.md
```

## Honesty check

- **Is it really training?** Yes — reproducible learning curves in `runs/*/history.json`,
  with the death-cause breakdown alongside the headline rate so a falling
  shootdown rate cannot hide a rising terrain-collision rate.
- **Is the demo real?** Yes — `naigos/demo/replay.py` replays logged rollouts from
  `env.rollout` with identical seeds and threat fields. The untrained side is a
  freshly initialised network, not a strawman.
- **Is the validation a tradeoff, not a claim?** Yes — detection probability and
  shootdown rate against objectives-reached and sorties-preserved, versus a naive
  direct route.
- **Did the invariant hold?** Yes. `tests/test_invariant.py`.

See [next-steps.md](next-steps.md) for what is not done and what would break first.
