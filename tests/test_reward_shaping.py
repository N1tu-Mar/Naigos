"""Reward-hacking guards, written from failures that actually happened."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv, RewardTerms
from naigos.rl.reward import (
    RewardCurriculum,
    RewardWeights,
    aircraft_lost,
    compute,
    constraint_cost,
)

W = RewardWeights()


def _terms(**kw):
    z = jnp.zeros(1)
    d = {f: z for f in RewardTerms._fields}
    d.update({k: jnp.array([v], dtype=jnp.float32) for k, v in kw.items()})
    return RewardTerms(**d)


def test_every_way_of_losing_the_airframe_costs_the_same():
    """MEASURED FAILURE: with terrain priced below shootdown the policy learned
    to dive into a ridge to break radar lock. A loss is a loss."""
    r_shot = float(compute(_terms(shotdown=1.0), W)[0])
    r_terr = float(compute(_terms(terrain_violation=1.0), W)[0])
    r_oob = float(compute(_terms(bounds_violation=1.0), W)[0])
    r_dry = float(compute(_terms(out_of_fuel=1.0), W)[0])
    assert r_shot == r_terr == r_oob == r_dry


def test_simultaneous_losses_are_not_double_counted():
    one = float(compute(_terms(shotdown=1.0), W)[0])
    both = float(compute(_terms(shotdown=1.0, terrain_violation=1.0), W)[0])
    assert one == both
    assert float(aircraft_lost(_terms(shotdown=1.0, terrain_violation=1.0))[0]) == 1.0


def test_integrated_shaping_stays_under_the_achievable_task_return():
    """MEASURED FAILURE: when the per-step exposure/lock cost integrated to more
    than progress+arrival, the optimal policy was to crash on step one."""
    cfg = EnvConfig()
    steps = cfg.max_steps
    # The case that matters is "fly the whole route fully exposed": heavily
    # tracked the entire way, and a third of it inside a lethal envelope. That
    # has to still beat dying on step one. (The absolute pathological maximum --
    # every term pinned at 1.0 for 500 steps including the boundary ramp -- is
    # not a reachable state, since the aircraft would be dead or off-map long
    # before, so bounding against it would over-constrain the weights.)
    sustained = W.exposure * 0.7 + W.lock * 0.7 + W.envelope * 0.3 + W.step_cost
    integrated = steps * sustained
    route_km = cfg.terrain.extent_x / 1000.0
    task_return = W.progress * route_km + W.arrived
    assert integrated < task_return, (
        f"flying the route fully exposed integrates to {integrated:.0f} over {steps} steps but the "
        f"task is only worth {task_return:.0f} -- dying immediately would be optimal"
    )


def test_loitering_is_strictly_negative():
    """Zero progress, zero exposure: the step cost must still bite."""
    assert float(compute(_terms(), W)[0]) < 0.0


def test_backtracking_is_punished():
    fwd = float(compute(_terms(progress=1000.0), W)[0])
    back = float(compute(_terms(progress=-1000.0), W)[0])
    assert back < fwd


def test_boundary_ramp_has_a_gradient_before_the_cliff():
    """The terminal bounds penalty fires only once the aircraft is already gone."""
    near = float(compute(_terms(edge_proximity=0.9), W)[0])
    far = float(compute(_terms(edge_proximity=0.0), W)[0])
    assert near < far


def test_edge_proximity_is_zero_at_spawn():
    """Aircraft must not start inside the ramp and inherit a constant offset."""
    cfg = EnvConfig()
    env = NaigosEnv(cfg)
    st, _ = env.reset(jax.random.PRNGKey(0))
    _, _, terms, _, _ = env.step(st, jnp.zeros((cfg.n_blue, 3)))
    assert float(terms.edge_proximity.max()) == 0.0


def test_constraint_cost_is_separate_from_the_reward():
    t = _terms(shotdown=1.0, envelope_dwell=1.0)
    # the cost channel is terminal only; dwell is shaping (see below)
    assert float(constraint_cost(t)[0]) == 1.0
    # and it must not depend on any reward weight
    from dataclasses import replace

    heavy = replace(W, aircraft_loss=9999.0, envelope=9999.0)
    assert float(constraint_cost(t)[0]) == 1.0
    assert float(compute(t, heavy)[0]) != float(compute(t, W)[0])


def test_curriculum_moves_survival_first_then_efficiency():
    cur = RewardCurriculum()
    hot, s_hot, e_hot = cur.weights_for(W, shootdown_rate=0.9)
    cool, s_cool, e_cool = cur.weights_for(W, shootdown_rate=0.02)
    assert e_hot == 0.0 and e_cool == 1.0
    assert s_hot > s_cool  # survival weight relaxes only after survival is achieved
    assert cool.fuel > hot.fuel  # efficiency terms fade IN
    assert cool.aircraft_loss < hot.aircraft_loss


def test_red_curriculum_promotes_on_survival_not_on_shootdown_rate():
    """BUG FOUND IN TRAINING: (1 - shootdown_rate) is not survival. A policy
    that trades being shot down for flying into a ridge has a low shootdown rate
    and would be promoted into a harder theatre it cannot handle."""
    import inspect

    from naigos.rl import train

    src = inspect.getsource(train.run)
    assert "red_cur.update(level, survival_rate)" in src
    assert "1.0 - shootdown_rate" not in src


def test_curriculum_demotes_when_survival_collapses():
    from naigos.rl.red_team import RedCurriculum

    cur = RedCurriculum()
    assert cur.update(0.5, survival_rate=0.95) > 0.5
    assert cur.update(0.5, survival_rate=0.10) < 0.5
    assert cur.update(0.0, survival_rate=0.10) == 0.0  # clamped
    assert cur.update(1.0, survival_rate=0.95) == 1.0


def test_cost_channel_is_terminal_only():
    """Mixing a per-step indicator with a once-per-episode terminal event makes
    the CMDP budget unmeetable and turns the multiplier into a ratchet."""
    dwell = float(constraint_cost(_terms(envelope_dwell=1.0))[0])
    lost = float(constraint_cost(_terms(shotdown=1.0))[0])
    assert dwell == 0.0, "per-step dwell must be shaping, not a hard constraint"
    assert lost == 1.0


def test_cost_per_episode_is_bounded_by_one():
    """So 'expected violations per agent-episode <= budget' reads as a rate."""
    t = _terms(shotdown=1.0, terrain_violation=1.0, bounds_violation=1.0, out_of_fuel=1.0)
    assert float(constraint_cost(t)[0]) == 1.0


def test_envelope_dwell_is_still_penalised_as_shaping():
    assert float(compute(_terms(envelope_dwell=1.0), W)[0]) < float(compute(_terms(), W)[0])
