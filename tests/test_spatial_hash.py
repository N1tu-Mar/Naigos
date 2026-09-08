"""The grid must be an EXACT radius-restricted kNN, not an approximation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from naigos.env.config import SpatialHashConfig, TerrainConfig
from naigos.env import spatial_hash as sh

TC = TerrainConfig()


@pytest.mark.parametrize("radius", [15_000.0, 25_000.0, 45_000.0, 90_000.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grid_matches_bruteforce_within_radius(radius, seed):
    hc = SpatialHashConfig(query_radius=radius, cell_capacity=64)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    pts = jax.random.uniform(k1, (60, 2)) * TC.extent_x
    q = jax.random.uniform(k2, (10, 2)) * TC.extent_x
    valid = jnp.arange(60) < 45

    a, ma = sh.topk_neighbours(hc, TC, q, pts, valid, 5)
    b, mb = sh.topk_bruteforce(q, pts, valid, 5, radius=radius)
    assert jnp.all(ma == mb)
    assert jnp.all(jnp.where(ma, a, 0) == jnp.where(mb, b, 0))


def test_no_overflow_at_configured_density():
    """cell_capacity must exceed the points that actually land in one cell."""
    hc = SpatialHashConfig()
    pts = jax.random.uniform(jax.random.PRNGKey(0), (32, 2)) * TC.extent_x
    _, overflow = sh.build(hc, TC, pts, jnp.ones(32, dtype=bool))
    assert int(overflow) == 0


def test_candidates_are_not_duplicated_at_the_grid_edge():
    """Clamping out-of-range neighbour cells would list an edge cell twice."""
    hc = SpatialHashConfig(query_radius=20_000.0, cell_capacity=8)
    pts = jnp.array([[100.0, 100.0], [500.0, 500.0], [900.0, 200.0]])
    table, _ = sh.build(hc, TC, pts, jnp.ones(3, dtype=bool))
    cand = sh.query(hc, TC, table, sh.cell_of(hc, TC, pts[0]))
    present = [int(c) for c in cand if c >= 0]
    assert len(present) == len(set(present))


def test_inactive_points_are_never_returned():
    hc = SpatialHashConfig(query_radius=90_000.0)
    pts = jax.random.uniform(jax.random.PRNGKey(1), (20, 2)) * 50_000.0
    valid = jnp.zeros(20, dtype=bool).at[3].set(True)
    q = jnp.array([[25_000.0, 25_000.0]])
    idx, mask = sh.topk_neighbours(hc, TC, q, pts, valid, 4)
    assert int(mask.sum()) == 1
    assert int(idx[0, 0]) == 3


def test_pair_valid_masks_per_query():
    """Per-query sensing masks must be honoured, not just the global one."""
    hc = SpatialHashConfig(query_radius=90_000.0)
    pts = jnp.array([[10_000.0, 10_000.0], [12_000.0, 10_000.0]])
    q = jnp.array([[11_000.0, 10_000.0], [11_000.0, 10_000.0]])
    valid = jnp.ones(2, dtype=bool)
    pair = jnp.array([[True, False], [False, True]])
    idx, mask = sh.topk_neighbours(hc, TC, q, pts, valid, 2, pair_valid=pair)
    assert int(mask[0].sum()) == 1 and int(idx[0, 0]) == 0
    assert int(mask[1].sum()) == 1 and int(idx[1, 0]) == 1
