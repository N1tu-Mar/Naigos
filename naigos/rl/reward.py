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
    # SCALE DISCIPLINE: these are PER-STEP and an episode is ~500 steps, so the
    # integrated shaping cost must stay well under the achievable task return
    # (progress + arrival, ~640 here). At exposure=1.0/lock=2.0 the integral was
    # ~700 and the optimal policy was to crash on step 1 -- shaping that exceeds
    # the task reward does not shape the task, it replaces it.
    exposure: float = 0.15  # continuous ramp: punish being *seen*, before any lock
    lock: float = 0.40  # punish a hard track
    envelope: float = 0.80  # punish dwell inside a lethal envelope
    # Every way of losing the airframe costs the same. MEASURED BUG: with
    # terrain=50 against shotdown=200 the policy learned to dive into a ridge to
    # break radar lock -- crashing was literally cheaper than being seen. A
    # loss is a loss; the reward must not rank them.
    aircraft_loss: float = 300.0  # terminal; ALSO in the CMDP cost channel

    # --- task ---
    progress: float = 2.0  # per km closed on the objective
    arrived: float = 300.0

    # --- efficiency / feasibility (fades in) ---
    fuel: float = 0.02  # per kg
    g_excess: float = 0.05  # per g above 1
    step_cost: float = 0.02  # per step, kills loitering

    # --- non-terminal flight-envelope violations (shaping only) ---
    ceiling: float = 5.0
    stall: float = 5.0

    def scaled(self, survival_w: float, efficiency_w: float) -> "RewardWeights":
        """Apply the two curriculum dials."""
        s, e = survival_w, efficiency_w
        return dataclasses.replace(
            self,
            exposure=self.exposure * s,
            lock=self.lock * s,
            envelope=self.envelope * s,
            aircraft_loss=self.aircraft_loss * s,
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


def aircraft_lost(terms: RewardTerms) -> jnp.ndarray:
    """Any terminal loss of the airframe, however it happened. (B,)."""
    return jnp.clip(
        terms.shotdown + terms.terrain_violation + terms.bounds_violation + terms.out_of_fuel,
        0.0,
        1.0,
    )


def compute(terms: RewardTerms, w: RewardWeights, alive_mask=None) -> jnp.ndarray:
    """Per-agent scalar reward, shape (B,)."""
    r = (
        w.progress * (terms.progress / 1_000.0)  # per km
        + w.arrived * terms.arrived
        - w.exposure * terms.exposure
        - w.lock * terms.lock_level
        - w.envelope * terms.envelope_dwell
        - w.aircraft_loss * aircraft_lost(terms)
        - w.fuel * terms.fuel_used
        - w.g_excess * jnp.maximum(terms.g_excess, 0.0)
        - w.step_cost
        - w.ceiling * terms.ceiling_violation
        - w.stall * terms.stall_violation
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
    return aircraft_lost(terms) + terms.envelope_dwell
