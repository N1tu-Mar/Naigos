"""The env must be buildable from the cited theatre, and must say which it used."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

theatre = pytest.importorskip("naigos.data.theatre")
from naigos.env.flight_env import NaigosEnv  # noqa: E402
from naigos.env.theatre_bridge import describe, env_from_theatre  # noqa: E402

try:
    CFG, HMAP, NOTES = env_from_theatre(n_threat=8)
except (FileNotFoundError, KeyError) as e:  # pragma: no cover - needs the research cache
    pytest.skip(f"theatre cache unavailable: {e}", allow_module_level=True)


def test_theatre_grid_matches_the_dem():
    assert HMAP.shape == (CFG.terrain.ny, CFG.terrain.nx)
    assert float(HMAP.min()) >= 0.0
    # real relief, not a synthetic plateau
    assert float(HMAP.max() - HMAP.min()) > 1_000.0


def test_threat_kinds_come_from_the_component_spec():
    labels = [k.label for k in CFG.threat_kinds]
    assert len(labels) == len(set(labels))
    assert any("sam" in n or "surveillance" in n for n in labels)
    assert CFG.threat_feat_dim == 9 + len(labels)


def test_sensing_horizon_covers_the_longest_detection_range():
    """Otherwise an emitter tracks an aircraft that cannot see it in its own obs."""
    assert CFG.hash.query_radius >= max(k.detect_range for k in CFG.threat_kinds)


def test_assumptions_are_declared_not_hidden():
    a = NOTES["assumptions"]
    for k in ("n_max_g", "roc_max_ms", "gamma_max_rad", "floor_agl_m"):
        assert k in a
    assert "OpenSky" in str(a)  # the measured value it deliberately overrides
    assert "assumptions (not measured)" in describe(NOTES)


def test_env_runs_on_the_real_theatre():
    env = NaigosEnv(CFG, hmap=HMAP)
    st, o = env.reset(jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(o.ego)))
    s2, o2, terms, done, info = env.step(st, jnp.zeros((CFG.n_blue, 3)))
    assert bool(jnp.all(jnp.isfinite(s2.air.pos)))


def test_aircraft_and_threats_spawn_above_the_terrain():
    env = NaigosEnv(CFG, hmap=HMAP)
    st, _ = env.reset(jax.random.PRNGKey(1))
    from naigos.env.terrain import sample_height

    g = sample_height(HMAP, CFG.terrain, st.air.pos[:, 0], st.air.pos[:, 1])
    assert bool(jnp.all(st.air.pos[:, 2] > g))
    gt = sample_height(HMAP, CFG.terrain, st.threats.pos[:, 0], st.threats.pos[:, 1])
    assert bool(jnp.all(st.threats.pos[:, 2] >= gt - 1e-3))


def test_spawn_is_outside_the_boundary_ramp_on_this_theatre():
    """The fixed-metre version of this ramp put every aircraft inside it here."""
    env = NaigosEnv(CFG, hmap=HMAP)
    st, _ = env.reset(jax.random.PRNGKey(0))
    _, _, terms, _, _ = env.step(st, jnp.zeros((CFG.n_blue, 3)))
    assert float(terms.edge_proximity.max()) == 0.0


def test_terrain_masking_reduces_detection_on_the_real_dem():
    """The signature mechanic, measured on real relief rather than asserted."""
    from naigos.env import detection as det
    from naigos.env.threats import per_threat_params

    env = NaigosEnv(CFG, hmap=HMAP)
    st, _ = env.reset(jax.random.PRNGKey(2))
    tp = per_threat_params(CFG, st.threats.kind)
    from naigos.env.terrain import sample_height

    ground = sample_height(HMAP, CFG.terrain, st.air.pos[:, 0], st.air.pos[:, 1])

    high = st.air.pos.at[:, 2].set(ground + 6_000.0)
    low = st.air.pos.at[:, 2].set(ground + 120.0)
    pd_high = det.detection_probability(HMAP, CFG.terrain, CFG.detection, high, st.air.psi,
                                        st.threats.pos, tp, st.threats.active)["pd"]
    pd_low = det.detection_probability(HMAP, CFG.terrain, CFG.detection, low, st.air.psi,
                                       st.threats.pos, tp, st.threats.active)["pd"]
    assert float(pd_low.max(axis=0).mean()) < float(pd_high.max(axis=0).mean())


def test_cbf_backstop_cuts_terrain_losses_on_the_real_dem():
    """The Sierra is where the terrain barrier earns its place: a naive
    terrain-following controller loses sorties to the ridgeline without it."""
    import numpy as np

    from naigos.rl.cbf import CBFConfig, make_policy_filter
    from naigos.rl.red_team import RedCurriculum

    cfg = RedCurriculum().apply(CFG, 0.0)
    env = NaigosEnv(cfg, hmap=HMAP)

    def nap(o, k):
        h = jnp.arctan2(o.ego[:, 4], o.ego[:, 5])
        agl = o.ego[:, 2] * 5000.0
        return jnp.stack([jnp.clip(h * 2.0, -1, 1), jnp.clip((250.0 - agl) / 300.0, -1, 1),
                          jnp.full_like(h, 0.6)], -1)

    keys = jax.random.split(jax.random.PRNGKey(11), 16)
    afilter = make_policy_filter(CBFConfig(), cfg)
    fin_off, off = jax.jit(jax.vmap(lambda k: env.rollout(k, nap)))(keys)
    fin_on, on = jax.jit(jax.vmap(lambda k: env.rollout(k, nap, action_filter=afilter)))(keys)

    assert float(on["terms"].terrain_violation.sum()) < float(off["terms"].terrain_violation.sum())
    assert float(fin_on.alive.mean()) > float(fin_off.alive.mean())
    # the honest caveat: the filter only guarantees safety while the QP is
    # feasible, so the infeasibility rate has to be observable, not hidden.
    live = np.asarray(on["alive"]).astype(float)
    infeasible = ((~np.asarray(on["cbf_feasible"])) * live).sum() / max(live.sum(), 1)
    assert 0.0 <= infeasible <= 1.0
