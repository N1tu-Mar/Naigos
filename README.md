# Naigos

Survivable routing in contested 3D airspace. Aircraft learn to reach objectives through active
threat envelopes using routing, timing, terrain masking and manoeuvre.

> **Hard invariant.** The blue agent is purely evasive. Its action space contains only flight
> controls — heading rate, climb rate, speed, bank. There is no engage, fire, suppress or target
> action, and there never will be. Survival comes from routing and terrain, never from removing a
> threat. See §0 of `prompt.md`.

## Status

The research and data layer (spec §8) is built. The 3D flight env, MARL training and demo are not.

## The data layer

```bash
uv venv && uv pip install -e '.[research,dev]'
naigos-research          # fetch -> cache -> cite; idempotent, offline after first run
pytest -q                # 81 tests
```

This pulls from a fixed six-source allowlist, caches every artifact with a sha256, and emits
`components/*.json` — the cited specs the env consumes — plus `docs/DATA.md`.

### What the data established

- **Terrain masking is a real, learnable mechanic in the chosen theatre.** True line-of-sight
  from a summit sensor over real 3DEP terrain: **76% of the area is masked at 100 m above
  ground, 64% at 300 m, 31% at 1 km, 0% at 3 km.** Flying low genuinely buys concealment, and
  that gradient is measured rather than asserted.
- **The projection chain is independently validated.** DEM elevations and published airfield
  elevations agree to within 3.6 m across the theatre.
- **The local atmosphere is not the textbook atmosphere.** The measured refractivity gradient
  gives an effective-Earth factor of **k = 1.256**, not the standard 4/3 — dry high-desert air
  refracts less, so the radar horizon is shorter than the default assumption.
- **The airframe envelope is measured.** 1459 consecutive ADS-B pairs put implied bank angles at
  p95 9.1° and max 26.8°, validating ω = g·tan(φ)/V against real traffic.
- **Threat envelopes are derived, not looked up.** Each class declares a notional role-based
  design range; the radar range equation solves the transmit power.

Full provenance in [`docs/DATA.md`](docs/DATA.md); the reasoning trail is in
[`docs/DEVLOG.md`](docs/DEVLOG.md).

### Scope guardrail

Threat models are parameterized abstractions — a detection range scale, an altitude band, a
reaction latency, a Pd curve, a lethal radius — derived from the open radar range equation with
generic, notional parameters. This repository does not contain, and the research agent will not
assemble, a capability database for any real fielded system. The RL problem depends on the shape
of the exposure-versus-survival tradeoff, not on real-world accuracy against a real system.

## Layout

```
naigos/research/   allowlisted fetch -> cache -> cited JSON specs
naigos/data/       loaders the env consumes (terrain + LOS, theatre assembly)
components/        one cited JSON per design decision
docs/              DATA.md (provenance), DEVLOG.md (the why)
data_cache/        raw cached data, gitignored; manifest.json records every sha256
tests/             81 tests, pure numpy, no network
```
