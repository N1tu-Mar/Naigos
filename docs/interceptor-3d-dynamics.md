# 3D airborne-interceptor dynamics (Phase 1)

**Status:** landed. Physics layer only — no learned world model, no vision, no
self-play, no analytical lead predictor. Those are Phase 2 and beyond.

**Nature of the model:** every number here is a knob on a *fictional,
parameterised* vehicle, chosen to make the evasion problem well-posed. None of
it is a model of, or a claim about, any real interceptor. It is the same
guardrail `ThreatKindConfig` already carries for the sensor and envelope
parameters (prompt.md s8).

---

## The defect this closes

`threats.step` advanced threats horizontally and then wrote

```python
z = jnp.where(airborne, jnp.maximum(st.pos[:, 2], ground + 200.0), ground + 5.0)
```

An airborne interceptor therefore held whatever altitude it spawned at, forever,
except when terrain pushed it up. It could never descend, and it could never
climb toward a target.

That is a **2.5D loophole**. Blue's action space is `(bank, gamma, throttle)`;
`gamma` is a free axis. Against a pursuer that cannot move in that axis, a pure
climb or a pure dive is an escape that costs nothing but fuel and that no amount
of red horizontal skill can answer. Any policy trained against it learns that
altitude is a safe place to hide, which is an artefact of the simulator rather
than a tactic.

Ground kinds were never part of the defect and are unchanged: they stay pinned
to the DEM at `ground + 5 m`.

---

## What was added

### 1. Flight limits, separate from the engagement envelope

Four fields on `ThreatKindConfig`:

| field | default | meaning |
|---|---|---|
| `climb_rate_max` | 40 m/s | best sustained rate of climb |
| `descent_rate_max` | 60 m/s | max rate of descent, positive magnitude |
| `terrain_clearance` | 200 m AGL | hard floor the vehicle will not fly below |
| `vehicle_ceiling` | 11 000 m AMSL | hard flight ceiling of the vehicle |

These are the vehicle's **own** limits. `alt_min` / `alt_max` are untouched and
keep their existing meaning: the altitude band in which this kind's *weapon* can
engage. The two are deliberately different quantities, and for the default
interceptor they are deliberately different numbers — `alt_max` is 13 km,
`vehicle_ceiling` is 11 km. The vehicle cannot fly to the top of its own
engagement envelope, which is exactly the kind of gap a policy should be able to
find and exploit.

The fields are read **only when `airborne` is true**. A ground kind carrying the
defaults still never leaves the DEM. This also means a theatre-derived airborne
class coming out of `theatre_bridge` inherits usable limits with no JSON change.

### 2. Vertical state

`ThreatState` gains `vz: (T,)` — vertical velocity in m/s, `+up`. Vertical
velocity rather than flight-path angle, because the limits that matter here are
rate limits (m/s) and the clamp against terrain and ceiling is a position
clamp; carrying `gamma` would mean converting through speed twice per step and
would go singular at zero ground speed. Ground kinds and inactive slots hold
`vz = 0`.

`vz` records the rate **achieved**, not the rate commanded. If the clamp against
terrain or ceiling ate part of the commanded climb, that shows up in the state
instead of being silently discarded.

### 3. Spawning

* airborne: placed at `SPAWN_AGL = 3000 m` above local terrain, then clamped
  into `[ground + terrain_clearance, vehicle_ceiling]`. `vz = 0` — on station,
  nothing tracked yet, nothing to climb toward.
* ground: `ground + 5 m`, exactly as before.
* inactive/padded: inert, and stay inert.

### 4. The vertical pursuit command

`threats.vertical_command(cfg, st, blue_pos, blue_alive, lock) -> (T,)`.

It lives in `threats.py`, not `red_team.py`, so the `RedPolicy` interface stays
`(psi_cmd, speed_cmd)` and a learned red can be dropped in later without
inheriting the physics.

**Information discipline — the important part.** The contact is chosen exactly
the way `scripted_red` chooses its horizontal target:

```python
score = jnp.where(blue_alive[None, :], lock, -1.0)
tgt = jnp.argmax(score, axis=-1)
has_contact = jnp.max(score, axis=-1) > CONTACT_THRESHOLD   # 0.05, red_team's number
```

Same threshold, same matrix, same start-of-step snapshot. The vertical channel
cannot see a blue the horizontal channel cannot. **With no contact the command
is 0** and the interceptor holds altitude — it does not drift toward an
undetected aircraft. Without that rule the model would hand red omniscience in
altitude and make blue's vertical axis worthless in the opposite direction.

The law itself is a saturated one-step closure:

```python
vz = clip((blue_z - threat_z) / cfg.dt, -descent_rate_max, climb_rate_max)
```

No lead term. A vertical lead/intercept predictor is Phase 2, and should be
built and measured on top of a physics layer that is already correct.

Horizontal target selection and horizontal pursuit are untouched.

### 5. Integration

`threats.step(cfg, st, hmap, psi_cmd, speed_cmd, vz_cmd=None)` — `vz_cmd`
defaults to level flight so callers predating the 3D model still work.

Order of operations, per step:

1. heading and speed as before; horizontal position updated and clipped to the
   grid;
2. terrain sampled at the **updated** `(x, y)` — the clearance floor that
   matters is the ground the vehicle is about to be over, not the one it just
   left. This is what makes flying at a ridge push the vehicle up rather than
   through it;
3. `z += clip(vz_cmd, -descent_rate_max, climb_rate_max) * cfg.dt`;
4. clamp into `[ground + terrain_clearance, vehicle_ceiling]`;
5. ground kinds overwrite with `ground + 5`; inactive slots keep their old `z`;
6. `vz` recorded as the achieved rate.

Shapes are fixed, there is no data-dependent control flow, and the whole step
stays `jit`/`vmap`-safe.

**Bound tie-break.** Where terrain rises far enough that
`ground + terrain_clearance > vehicle_ceiling`, the two bounds cross. Terrain
wins: the ceiling is lifted to the floor. A threat inside a mountain is a worse
modelling failure than one briefly above its published ceiling, and `clip` needs
a well-ordered interval to stay jit-safe. At the default limits (200 m clearance,
11 km ceiling) this needs terrain above 10.8 km AMSL and does not occur on any
theatre in the repo.

---

## What was NOT changed

Confirmed by reading, not assumed:

* **Detection / radar mathematics** — `detection.py` is untouched. It already
  consumed `threats.pos` as a 3D point for slant range, LOS ray-marching and
  the earth-curvature drop, so the updated altitude flows through with no code
  change. That is the whole point of having done it in the position.
* **Engagement** — `det_mod.engagement` untouched; it reads the same slant
  range and the same `alt_min`/`alt_max` gate.
* **Blue observations** — `obs.py` untouched. `vz` is deliberately NOT exposed
  to blue: adding a feature would change `threat_feat_dim` and invalidate every
  checkpoint's input width. Blue sees the interceptor's 3D position moving,
  which is what a sensor would give it.
* **Reward weights, CBF, renderer, red policy behaviour** — untouched.
* **Blue's action space** — still `(bank_cmd, gamma_cmd, throttle_cmd)`. Blue
  never attacks or controls a threat. `tests/test_invariant.py` still enforces
  it and `tests/test_interceptor_3d.py` re-asserts it.

---

## Modelling assumptions, stated plainly

1. **Rate-limited vertical motion, no vertical acceleration state.** The vehicle
   reaches its commanded vertical rate within one `cfg.dt` (2 s). Real vertical
   dynamics have a lag; this model has none. It makes red slightly stronger in
   the vertical than a lagged model would.
2. **Climb and descent are asymmetric** — 40 m/s up, 60 m/s down. Gravity is
   free and thrust is not. This makes diving after blue cheaper than chasing it
   up, so blue's two vertical escapes are not equally good.
3. **The ceiling is hard, not soft.** No degraded performance band below it. The
   vehicle simply stops at 11 km.
4. **Terrain clearance is hard and instantaneous.** The vehicle never descends
   below 200 m AGL and is pushed up by rising ground with no lag and no stall.
   This is generous to red in mountainous terrain.
5. **One-step vertical closure, no lead.** The command aims at where blue *is*,
   not where it will be. Against a fast-climbing blue this under-leads. Phase 2.
6. **Vertical pursuit is gated on horizontal track quality.** There is no
   separate vertical sensor and no altitude-only detection: `lock` is the only
   contact channel, so the vertical channel inherits exactly red's horizontal
   information.
7. **Speed is unchanged by climb.** A climbing interceptor keeps its full
   horizontal speed. Real vehicles trade one for the other. This is generous to
   red, and it is the assumption most worth revisiting in Phase 2.
8. **Ground kinds have no vertical model at all** — not a slow one, none. They
   are pinned to the DEM.

---

## Retraining impact

**Any checkpoint trained before this commit is valid only for the old, 2.5D
dynamics. Do not use one to make a comparative claim after this change.**

The environment's transition function changed for every world containing an
active airborne kind. Specifically:

* a climb or dive that used to break a lock now may not;
* the exposure and lock trajectories a policy learned to predict are different;
* survival rates measured before and after are **not comparable**, and any
  pre-change number quoted next to a post-change number is a false comparison.

Observation and action shapes did **not** change (`ego_dim`, `threat_feat_dim`,
`action_dim` are all identical), so an old checkpoint will load and run without
error. That is the trap: it will produce plausible-looking numbers against
dynamics it never saw. Retrain from scratch before reporting anything.

Expect measured survival to drop on the first run after this change. That drop
is the loophole closing, not a regression.

---

## Tests

`tests/test_interceptor_3d.py`, 22 tests:

ground threats pinned to the DEM · only airborne kinds move vertically · climb
and descent bounded per timestep · terrain clearance holds while flying over
rising ground · vehicle ceiling holds under a sustained climb · inactive threats
inert · vertical pursuit requires an existing track · dead contacts ignored ·
command signed toward the contact · ground and inactive kinds never commanded ·
a climbing blue is no longer a free escape · a diving blue is followed down ·
env step is jit-able and moves airborne threats in 3D · rollout deterministic
and vmap-able · altitudes stay inside the envelope across a full rollout ·
detection receives finite 3D positions · blue still has no weapon.

```
uv run pytest tests/test_interceptor_3d.py tests/test_env_contract.py \
  tests/test_detection_physics.py -q
```
