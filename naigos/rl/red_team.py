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


class CurriculumError(ValueError):
    """A curriculum configuration or a stored curriculum state is not usable."""


#: Every reason a curriculum tick can give for the level it left behind.
#: `held` means the smoothed statistic was inside the dead band (or the level
#: was already at a bound); `insufficient_evidence` means the gate had not been
#: met yet and no threshold was even consulted.
CURRICULUM_REASONS = ("promoted", "demoted", "held", "insufficient_evidence")

SMOOTHING_MODES = ("window", "ewma", "none")


@dataclasses.dataclass(frozen=True)
class CurriculumState:
    """Everything the curriculum remembers between ticks.

    Deliberately a plain frozen dataclass of JSON scalars: it is written into
    `history.json` verbatim (see `RedCurriculum.decide`) and that history is
    what a resume reads its smoothing state back out of, so it has to survive a
    JSON round trip unchanged.

    `window` holds RAW survival measurements, newest last. `ewma` is the
    exponentially weighted accumulator; it is None until the first observation,
    which is not the same thing as zero -- an unmeasured statistic must never
    read as a measured collapse.

    `n_since_change` counts evaluations observed since the level last MOVED, and
    the window and the accumulator are both cleared at that moment. Samples taken
    before a level change measured a different task: red's detection reach,
    lethal radius, reaction latency, speed and active count all changed under
    them. Averaging across that boundary would smooth two different difficulties
    together and is the one way a smoothed curriculum can be more wrong than an
    unsmoothed one.
    """

    level: float = 0.0
    window: tuple[float, ...] = ()
    ewma: float | None = None
    n_observations: int = 0  # evaluations observed over the life of the run
    n_since_change: int = 0  # evaluations observed since the level last moved
    last_observed_iteration: int | None = None
    last_reason: str | None = None

    def to_dict(self) -> dict:
        """JSON-safe, key-ordered, no numpy scalars."""
        return {
            "level": float(self.level),
            "window": [float(x) for x in self.window],
            "ewma": None if self.ewma is None else float(self.ewma),
            "n_observations": int(self.n_observations),
            "n_since_change": int(self.n_since_change),
            "last_observed_iteration": (
                None if self.last_observed_iteration is None else int(self.last_observed_iteration)
            ),
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumState":
        if not isinstance(d, dict):
            raise CurriculumError(f"curriculum state must be a mapping, got {type(d).__name__}")
        missing = [k for k in ("level", "window", "n_observations", "n_since_change") if k not in d]
        if missing:
            raise CurriculumError(f"curriculum state is missing {missing}")
        ewma = d.get("ewma")
        it = d.get("last_observed_iteration")
        return cls(
            level=float(d["level"]),
            window=tuple(float(x) for x in d["window"]),
            ewma=None if ewma is None else float(ewma),
            n_observations=int(d["n_observations"]),
            n_since_change=int(d["n_since_change"]),
            last_observed_iteration=None if it is None else int(it),
            last_reason=d.get("last_reason"),
        )


@dataclasses.dataclass(frozen=True)
class CurriculumDecision:
    """One curriculum tick, with the evidence that produced it.

    `telemetry()` is what lands in the history row. It carries the raw
    measurement AND the smoothed one, because a row that showed only the
    smoothed number would make an audit unable to tell a genuine plateau from a
    smoothing window that is simply lagging.
    """

    state: CurriculumState  # state AFTER the decision
    level_before: float
    level_after: float
    reason: str
    raw_survival: float | None
    smoothed_survival: float | None
    window_n: int
    n_observations: int
    n_since_change: int
    min_evaluations: int
    smoothing: str

    @property
    def changed(self) -> bool:
        return self.level_after != self.level_before

    def telemetry(self) -> dict:
        # No `red_level`: in a merged eval row that key means "the level the
        # metrics were measured at", which is `level_before`. Overwriting it
        # with the post-tick level would misattribute the whole row.
        return {
            "red_level_before": float(self.level_before),
            "red_level_after": float(self.level_after),
            "curriculum_reason": self.reason,
            "curriculum_raw_survival": self.raw_survival,
            "curriculum_smoothed_survival": self.smoothed_survival,
            "curriculum_smoothing": self.smoothing,
            "curriculum_window_n": int(self.window_n),
            "curriculum_evals_total": int(self.n_observations),
            "curriculum_evals_since_change": int(self.n_since_change),
            "curriculum_min_evaluations": int(self.min_evaluations),
            "curriculum_state": self.state.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class RedCurriculum:
    """v2: anneal red difficulty on blue's measured survival.

    Mirrors the Nomos pattern (collision weight first, efficiency second): red
    only gets harder once blue is actually surviving, so the learning signal
    never collapses. `level` in [0, 1].

    Difficulty moves on a SMOOTHED survival statistic, not on a single
    evaluation. `eval_worlds` is 64 by default and survival is a mean over
    64 x n_blue Bernoulli sorties, so one evaluation's standard error is several
    percent; a single draw landing either side of `promote_survival` was enough
    to move the level under the old rule. Two gates now stand between a
    measurement and a level change:

      * `smoothing` / `window_size` / `ewma_alpha` -- the statistic the
        thresholds are compared against is an average of recent evaluations, not
        the newest one.
      * `min_evaluations` -- at least this many evaluations must have been
        observed since the level last moved before it may move again.

    Both gates are configurable and `RedCurriculum.legacy()` turns them off,
    which reproduces the pre-smoothing single-evaluation rule exactly.

    The dead band between `demote_survival` and `promote_survival` is
    deliberately wide and is UNCHANGED from the pre-smoothing defaults, so an
    existing checkpoint resumes against the thresholds it was trained under.
    Narrowing it is a separate, empirical decision; see
    docs/curriculum-stability.md.
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
    promote_survival: float = 0.75  # promote once SMOOTHED survival exceeds this
    demote_survival: float = 0.35
    step_size: float = 0.05

    # --- stability gates ---------------------------------------------------
    #: "window" = mean of the last `window_size` raw survival measurements.
    #: "ewma"   = exponentially weighted mean with `ewma_alpha`.
    #: "none"   = the newest measurement, i.e. no smoothing at all (legacy).
    smoothing: str = "window"
    window_size: int = 3
    ewma_alpha: float = 0.4
    #: Evaluations that must be observed since the level last moved before it
    #: may move again. 1 disables the gate.
    min_evaluations: int = 3

    def __post_init__(self) -> None:
        """Refuse a curriculum that cannot behave, at construction time.

        These are cheap checks and every one of them describes a configuration
        whose failure mode is silent: an inverted dead band promotes and demotes
        on the same measurement, a zero step never moves, a window of zero has
        no statistic to threshold against.
        """
        if self.smoothing not in SMOOTHING_MODES:
            raise CurriculumError(
                f"smoothing must be one of {SMOOTHING_MODES}, got {self.smoothing!r}"
            )
        for name in ("promote_survival", "demote_survival"):
            v = getattr(self, name)
            if not 0.0 <= float(v) <= 1.0:
                raise CurriculumError(f"{name} must be a survival rate in [0, 1], got {v}")
        if self.demote_survival >= self.promote_survival:
            raise CurriculumError(
                f"demote_survival ({self.demote_survival}) must be below promote_survival "
                f"({self.promote_survival}); an inverted or empty dead band would promote and "
                f"demote on the same measurement"
            )
        if not self.step_size > 0.0:
            raise CurriculumError(f"step_size must be positive, got {self.step_size}")
        if int(self.window_size) < 1:
            raise CurriculumError(f"window_size must be at least 1, got {self.window_size}")
        if not 0.0 < float(self.ewma_alpha) <= 1.0:
            raise CurriculumError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")
        if int(self.min_evaluations) < 1:
            raise CurriculumError(
                f"min_evaluations must be at least 1 (1 means no gate), got {self.min_evaluations}"
            )
        if int(self.n_lo) > int(self.n_hi):
            raise CurriculumError(f"n_lo ({self.n_lo}) must not exceed n_hi ({self.n_hi})")

    # --- the legacy rule ---------------------------------------------------

    @classmethod
    def legacy(cls, **overrides) -> "RedCurriculum":
        """The pre-smoothing curriculum: one evaluation, one decision.

        Kept because a run started under it can only be resumed under it, and
        because a test that wants to demonstrate the instability needs to be
        able to ask for it. It is never the default.
        """
        return cls(smoothing="none", window_size=1, min_evaluations=1, **overrides)

    @property
    def legacy_single_evaluation(self) -> bool:
        """True when both stability gates are off, so `decide` == `update`."""
        return self.smoothing == "none" and int(self.min_evaluations) <= 1

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
        """One LEGACY curriculum tick on a single evaluation. Returns the new level.

        This is the pre-smoothing rule, kept verbatim. It is reachable from
        `run()` only when the curriculum is explicitly configured as legacy
        (`RedCurriculum.legacy()`); the stateful `observe`/`decide` pair is what
        a default run uses. Promotion is on MEASURED SURVIVAL either way --
        never on `1 - shootdown_rate`, which is a different quantity (see the
        comment at the curriculum tick in `naigos.rl.train.run`).
        """
        if survival_rate > self.promote_survival:
            level += self.step_size
        elif survival_rate < self.demote_survival:
            level -= self.step_size
        return float(min(max(level, 0.0), 1.0))

    # --- the smoothed rule -------------------------------------------------

    def initial_state(self, level: float = 0.0) -> CurriculumState:
        return CurriculumState(level=self.clamp(level))

    @staticmethod
    def clamp(level: float) -> float:
        return float(min(max(float(level), 0.0), 1.0))

    def observe(
        self, state: CurriculumState, survival_rate: float, iteration: int | None = None
    ) -> CurriculumState:
        """Fold ONE evaluation into the smoothing state.

        Called once per evaluation, not once per curriculum tick. The two are on
        different periods (`eval_every` vs `curriculum_every`) and a tick that
        re-read the last evaluation would push the same measurement into the
        window twice, inflating the sample count with evidence that does not
        exist.
        """
        x = float(survival_rate)
        if not 0.0 <= x <= 1.0:
            raise CurriculumError(f"survival_rate must be in [0, 1], got {survival_rate!r}")
        window = (*state.window, x)[-int(self.window_size) :]
        a = float(self.ewma_alpha)
        ewma = x if state.ewma is None else a * x + (1.0 - a) * float(state.ewma)
        return dataclasses.replace(
            state,
            window=window,
            ewma=ewma,
            n_observations=state.n_observations + 1,
            n_since_change=state.n_since_change + 1,
            last_observed_iteration=state.last_observed_iteration if iteration is None else int(iteration),
        )

    def raw(self, state: CurriculumState) -> float | None:
        """The newest raw survival measurement, or None before the first one."""
        return float(state.window[-1]) if state.window else None

    def smoothed(self, state: CurriculumState) -> float | None:
        """The statistic the thresholds are compared against, or None if unmeasured.

        None rather than 0.0 on an empty state: `demote_survival` is 0.35 and a
        zero would read as a measured collapse and demote a run that has simply
        not been evaluated yet.
        """
        if not state.window:
            return None
        if self.smoothing == "none":
            return float(state.window[-1])
        if self.smoothing == "ewma":
            return None if state.ewma is None else float(state.ewma)
        return float(sum(state.window) / len(state.window))

    def evidence_n(self, state: CurriculumState) -> int:
        """How many evaluations the current statistic actually rests on.

        For a rolling window that is the window occupancy; for an EWMA the window
        is only a viewport, so it is the count since the last level change.
        """
        if self.smoothing == "ewma":
            return int(state.n_since_change)
        return len(state.window)

    def decide(self, state: CurriculumState, iteration: int | None = None) -> CurriculumDecision:
        """One curriculum tick against the smoothed statistic.

        Order of the gates matters and is deliberate:

          1. `min_evaluations` -- with too little evidence the thresholds are
             not consulted at all, and the row says `insufficient_evidence`
             rather than `held`. "We did not look" and "we looked and it was in
             the dead band" are different facts about the run.
          2. threshold crossing on the SMOOTHED statistic.
          3. clamping to [0, 1]. A level already at a bound reports `held`,
             because nothing moved -- the row still carries the statistic that
             wanted to move it.

        A level that moves clears the window and the accumulator: see
        `CurriculumState`.
        """
        before = self.clamp(state.level)
        raw = self.raw(state)
        sm = self.smoothed(state)
        n = self.evidence_n(state)

        if n < 1 or sm is None or state.n_since_change < int(self.min_evaluations):
            reason, after = "insufficient_evidence", before
        elif sm > self.promote_survival:
            reason, after = "promoted", self.clamp(before + self.step_size)
        elif sm < self.demote_survival:
            reason, after = "demoted", self.clamp(before - self.step_size)
        else:
            reason, after = "held", before

        if after == before:
            # clamped at a bound: the threshold was crossed but the level could
            # not move, so the run held. Saying "promoted" would be a lie.
            if reason in ("promoted", "demoted"):
                reason = "held"
            new_state = dataclasses.replace(state, level=before, last_reason=reason)
        else:
            new_state = CurriculumState(
                level=after,
                window=(),
                ewma=None,
                n_observations=state.n_observations,
                n_since_change=0,
                last_observed_iteration=state.last_observed_iteration,
                last_reason=reason,
            )

        return CurriculumDecision(
            state=new_state,
            level_before=before,
            level_after=after,
            reason=reason,
            raw_survival=raw,
            smoothed_survival=sm,
            window_n=n,
            n_observations=state.n_observations,
            n_since_change=state.n_since_change,
            min_evaluations=int(self.min_evaluations),
            smoothing=self.smoothing,
        )


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
