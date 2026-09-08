"""The verifier must agree with the env AND catch it when it does not.

Pure NumPy, no JAX import needed for the constraint arithmetic itself -- the
env is only used to produce a trace to check.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl import verifier

CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=90)


def _trace(seed=0, cfg=CFG):
    import jax.numpy as jnp

    env = NaigosEnv(cfg)

    def pol(o, k):
        h = jnp.arctan2(o.ego[:, 4], o.ego[:, 5])
        return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.zeros_like(h), jnp.full_like(h, 0.6)], -1)

    final, traj = env.rollout(jax.random.PRNGKey(seed), pol)
    out = {k: np.asarray(v) for k, v in traj.items() if k != "terms"}
    out["terms"] = type(traj["terms"])(*[np.asarray(x) for x in traj["terms"]])
    return final, out, cfg


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_verifier_agrees_with_the_env(seed):
    final, traj, cfg = _trace(seed)
    rep = verifier.verify_trace(
        cfg, np.asarray(final.hmap), traj,
        np.asarray(final.threats.kind), np.asarray(final.threats.active),
    )
    assert rep.ok, str(rep)
    assert rep.max_pd_error < 1e-3, f"detection probability drifted: {rep.max_pd_error}"


def test_verifier_reports_no_shootdown_without_a_firing_solution():
    """Inject a phantom kill: an aircraft dies on a step where nothing could fire."""
    final, traj, cfg = _trace(0)
    alive = traj["alive"].copy()
    # find an agent alive throughout, then kill it artificially at t=3
    survivors = np.where(alive[-1])[0]
    if len(survivors) == 0:
        pytest.skip("no survivor in this trace to falsify")
    b = int(survivors[0])
    alive[3:, b] = False
    traj = {**traj, "alive": alive}
    rep = verifier.verify_trace(
        cfg, np.asarray(final.hmap), traj,
        np.asarray(final.threats.kind), np.asarray(final.threats.active),
    )
    assert not rep.ok
    assert any("no firing solution" in m for m in rep.mismatches)


def test_verifier_is_pure_numpy():
    """No JAX in the verifier's own module namespace: the constraint channel has
    to be checkable by something that does not share the env's code path."""
    import inspect

    src = inspect.getsource(verifier)
    assert "import jax" not in src
    assert "jax.numpy" not in src


def test_cost_channel_counts_every_hard_violation():
    final, traj, cfg = _trace(1)
    rep = verifier.verify_trace(
        cfg, np.asarray(final.hmap), traj,
        np.asarray(final.threats.kind), np.asarray(final.threats.active),
    )
    assert rep.total_cost == pytest.approx(
        rep.shootdowns + rep.terrain_violations + rep.bounds_violations + rep.envelope_dwell_steps
    )


def test_bilinear_sampler_matches_the_jax_one():
    """The two heightmap samplers are written independently; they must agree."""
    import jax.numpy as jnp

    from naigos.env import terrain as jt

    tc = CFG.terrain
    h = jt.synthetic_terrain(jax.random.PRNGKey(0), tc)
    xs = np.linspace(0, tc.extent_x, 37)
    ys = np.linspace(0, tc.extent_y, 37)
    a = np.asarray(jt.sample_height(h, tc, jnp.asarray(xs), jnp.asarray(ys)))
    b = verifier.sample_height(np.asarray(h), tc, xs, ys)
    assert np.max(np.abs(a - b)) < 1e-2
