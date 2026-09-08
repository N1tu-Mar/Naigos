"""The adaptive red team: what replaces "give blue a gun".

prompt.md s4 stages this as a curriculum, and all three stages share one
interface so the env never has to know which is running:

    psi_cmd, speed_cmd = policy(key, cfg, threats, blue, sensed, params)

v1  `scripted_red`  -- pure pursuit / proportional navigation onto the nearest
                       *detected* blue; static sites simply do not move.
v2  `RedCurriculum` -- difficulty knobs (detection reach, reaction latency,
                       interceptor speed, threat count) annealed as blue improves.
                       Implemented as a rewrite of EnvConfig, so v1 behaviour is
                       unchanged and only the numbers move.
v3  `learned_red`   -- hook for a separate red policy with its own optimizer.
                       NOT trained yet; see next-steps.md.

Red and blue never share parameters or an optimizer. Blue's action space is
flight controls only and nothing in this module changes that.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..env.config import EnvConfig
from ..env.threats import ThreatState, per_threat_params

# psi_cmd (T,), speed_cmd (T,)
RedPolicy = Callable[..., tuple[jax.Array, jax.Array]]


def scripted_red(
    key: jax.Array,
    cfg: EnvConfig,
    threats: ThreatState,
    blue_pos: jax.Array,  # (B, 3)
    blue_vel: jax.Array,  # (B, 3)
    blue_alive: jax.Array,  # (B,) bool
    lock: jax.Array,  # (T, B) track quality -- red only chases what it senses
):
    """v1 red. Lead-pursuit onto the highest-track blue contact.

    Uses a constant-bearing lead point (a first-order proportional-navigation
    approximation) rather than pure pursuit, because pure pursuit against a
    faster target degenerates into a tail chase the interceptor can never win --
    which would make evasion trivially easy and teach blue nothing.
    """
    params = per_threat_params(cfg, threats.kind)
    T = threats.pos.shape[0]

    # pick the best contact per threat: highest lock among living blue
    score = jnp.where(blue_alive[None, :], lock, -1.0)  # (T, B)
    tgt = jnp.argmax(score, axis=-1)  # (T,)
    has_contact = jnp.max(score, axis=-1) > 0.05

    tp = blue_pos[tgt]  # (T, 3)
    tv = blue_vel[tgt]  # (T, 3)

    rel = tp - threats.pos
    rng = jnp.linalg.norm(rel[:, :2], axis=-1) + 1e-6
    closing = jnp.maximum(params["speed"], 1.0)
    t_go = jnp.clip(rng / closing, 0.0, 120.0)
    lead = tp[:, :2] + tv[:, :2] * t_go[:, None]

    psi_cmd = jnp.arctan2(lead[:, 1] - threats.pos[:, 1], lead[:, 0] - threats.pos[:, 0])

    # no contact: interceptors fly a slow racetrack, ground units hold position
    patrol = threats.psi + 0.05
    psi_cmd = jnp.where(has_contact, psi_cmd, patrol)

    airborne = params["airborne"] > 0.5
    speed_cmd = jnp.where(
        has_contact,
        params["speed"],
        jnp.where(airborne, 0.6 * params["speed"], 0.2 * params["speed"]),
    )
    speed_cmd = jnp.where(threats.active, speed_cmd, 0.0)
    return psi_cmd, speed_cmd


@dataclasses.dataclass(frozen=True)
class RedCurriculum:
    """v2: anneal red difficulty on blue's measured survival.

    Mirrors the Nomos pattern (collision weight first, efficiency second): red
    only gets harder once blue is actually surviving, so the learning signal
    never collapses. `level` in [0, 1].
    """

    detect_lo: float = 0.55
    detect_hi: float = 1.15
    # Lethal reach is annealed too. On the real Owens Valley theatre (97 km wide)
    # a single medium-SAM class has a 49.5 km lethal radius, so at full scale two
    # sites blanket the map and level 0 is already unsurvivable -- there is no
    # gradient for the policy to climb.
    lethal_lo: float = 0.45
    lethal_hi: float = 1.0
    latency_lo: float = 2.0  # x slower reaction = easier
    latency_hi: float = 0.8
    speed_lo: float = 0.6
    speed_hi: float = 1.15
    n_lo: int = 5
    n_hi: int = 16
    promote_survival: float = 0.75  # promote once survival rate exceeds this
    demote_survival: float = 0.35
    step_size: float = 0.05

    def apply(self, cfg: EnvConfig, level: float) -> EnvConfig:
        L = float(min(max(level, 0.0), 1.0))
        lerp = lambda a, b: a + (b - a) * L  # noqa: E731
        return cfg.replace(
            red_detect_scale=lerp(self.detect_lo, self.detect_hi),
            red_lethal_scale=lerp(self.lethal_lo, self.lethal_hi),
            red_latency_scale=lerp(self.latency_lo, self.latency_hi),
            red_speed_scale=lerp(self.speed_lo, self.speed_hi),
            n_threat_active=min(cfg.n_threat, int(round(lerp(self.n_lo, self.n_hi)))),
        )

    def update(self, level: float, survival_rate: float) -> float:
        """One curriculum tick. Returns the new level."""
        if survival_rate > self.promote_survival:
            level += self.step_size
        elif survival_rate < self.demote_survival:
            level -= self.step_size
        return float(min(max(level, 0.0), 1.0))


class LearnedRedStub(NamedTuple):
    """v3 placeholder.

    A learned red policy plugs in here with the same signature as
    `scripted_red`, its own parameters and its own optimizer. Deliberately not
    implemented: prompt.md s4 says land a clean v1/v2 learning curve first.
    Calling it raises rather than silently falling back, so a half-wired
    self-play run cannot be mistaken for a real one.
    """

    params: object = None

    def __call__(self, *a, **kw):
        raise NotImplementedError(
            "Learned red (self-play) is not implemented. See next-steps.md, gap R-1. "
            "Use scripted_red with a RedCurriculum until then."
        )
