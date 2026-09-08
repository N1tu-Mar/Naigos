"""Multi-objective reward with curriculum annealing.

The env emits raw `RewardTerms`; this module turns them into a scalar. Keeping
the two apart means the curriculum can re-weight without touching the physics,
and the CMDP cost channel can be recomputed independently by `verifier.py`.

Curriculum (the Nomos pattern): survival dominates first; efficiency terms fade
in only once the shootdown rate has dropped. Annealing the other way round
produces a policy that flies a beautiful fuel-optimal line straight into a SAM.

Reward-hacking watchlist (prompt.md s5), and what blocks each one here:
  * "fly to the map edge and loiter forever"  -> `progress` is metres closed on
    the objective, not survival time; `step_cost` makes idling strictly negative.
  * "circle in a safe corner"                 -> same: zero progress, and
    `progress` is signed, so backtracking is punished.
  * "dive into the dirt to break every radar" -> `terrain` is a hard constraint
    in the cost channel, not a reward term, so the Lagrange multiplier prices it.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from ..env.flight_env import RewardTerms


@dataclasses.dataclass(frozen=True)
class RewardWeights:
    # --- survival / exposure (dominant early) ---
    exposure: float = 1.0  # continuous ramp: punish being *seen*, before any lock
    lock: float = 2.0  # punish a hard track
    envelope: float = 1.5  # punish dwell inside a lethal envelope
    shotdown: float = 200.0  # terminal; ALSO in the CMDP cost channel

    # --- task ---
    progress: float = 1.0  # per km closed on the objective
    arrived: float = 150.0

    # --- efficiency / feasibility (fades in) ---
    fuel: float = 0.02  # per kg
    g_excess: float = 0.05  # per g above 1
    step_cost: float = 0.05  # per step, kills loitering

    # --- flight-envelope violations (also constraints; small shaping here) ---
    terrain: float = 50.0
    ceiling: float = 5.0
    stall: float = 5.0
    bounds: float = 50.0
    out_of_fuel: float = 20.0

    def scaled(self, survival_w: float, efficiency_w: float) -> "RewardWeights":
        """Apply the two curriculum dials."""
        s, e = survival_w, efficiency_w
        return dataclasses.replace(
            self,
            exposure=self.exposure * s,
            lock=self.lock * s,
            envelope=self.envelope * s,
            shotdown=self.shotdown * s,
            fuel=self.fuel * e,
            g_excess=self.g_excess * e,
            step_cost=self.step_cost * e,
        )


@dataclasses.dataclass(frozen=True)
class RewardCurriculum:
    """survival_w 1 -> 1, efficiency_w 0 -> 1 as the shootdown rate falls."""

    efficiency_on_below: float = 0.35  # shootdown rate at which efficiency starts
    efficiency_full_below: float = 0.10
    survival_floor: float = 0.6  # survival weight never anneals below this

    def weights_for(self, base: RewardWeights, shootdown_rate: float) -> tuple[RewardWeights, float, float]:
        lo, hi = self.efficiency_full_below, self.efficiency_on_below
        e = (hi - shootdown_rate) / max(hi - lo, 1e-6)
        e = float(min(max(e, 0.0), 1.0))
        s = 1.0 - (1.0 - self.survival_floor) * e
        return base.scaled(s, e), s, e


def compute(terms: RewardTerms, w: RewardWeights, alive_mask=None) -> jnp.ndarray:
    """Per-agent scalar reward, shape (B,)."""
    r = (
        w.progress * (terms.progress / 1_000.0)  # per km
        + w.arrived * terms.arrived
        - w.exposure * terms.exposure
        - w.lock * terms.lock_level
        - w.envelope * terms.envelope_dwell
        - w.shotdown * terms.shotdown
        - w.fuel * terms.fuel_used
        - w.g_excess * jnp.maximum(terms.g_excess, 0.0)
        - w.step_cost
        - w.terrain * terms.terrain_violation
        - w.ceiling * terms.ceiling_violation
        - w.stall * terms.stall_violation
        - w.bounds * terms.bounds_violation
        - w.out_of_fuel * terms.out_of_fuel
    )
    if alive_mask is not None:
        r = r * alive_mask.astype(r.dtype)
    return r


def constraint_cost(terms: RewardTerms) -> jnp.ndarray:
    """The CMDP cost channel, shape (B,). Separate from the reward on purpose.

    PPO-Lagrangian learns the price of this channel instead of us hand-tuning
    `shotdown=200.0` against everything else. Each element is an indicator of a
    *hard* violation, so the constraint reads as "expected violations per
    episode <= budget".
    """
    return (
        terms.shotdown
        + terms.terrain_violation
        + terms.bounds_violation
        + terms.envelope_dwell  # dwelling in a lethal envelope is itself a violation
    )
