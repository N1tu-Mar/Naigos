"""Fixed-capacity uniform grid for neighbour queries.

Naive all-pairs is O(N*M) and fine at the default 4 blue / 16 threats, but the
Nomos lesson is that the env has to keep scaling as the cohort grows. This grid
buckets points into cells of side `query_radius` and answers a query from the
3x3 cell neighbourhood, giving O(N*C) with C = 9*capacity, independent of the
total point count.

SEMANTICS -- read this before using it. The 3x3 neighbourhood covers every point
within `query_radius` of the query and *some* points beyond it. So the grid is
an exact k-nearest search **restricted to a radius**, not a global kNN. That is
the right primitive here: an aircraft's obs should contain nearby threats, and a
threat 150 km away is not information the policy should be handed for free.
`topk_bruteforce(..., radius=...)` is the reference the grid is tested against.

Everything is fixed-shape: overflow past `cell_capacity` is dropped and counted
so tests can assert it stays at zero for the configured densities. Empty slots
carry the sentinel index -1.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .config import SpatialHashConfig, TerrainConfig

EMPTY = -1


def grid_dims(hcfg: SpatialHashConfig, tcfg: TerrainConfig) -> tuple[int, int]:
    """Static grid shape: one cell per query radius, at least 1x1."""
    gx = max(1, int(math.ceil(tcfg.extent_x / hcfg.query_radius)))
    gy = max(1, int(math.ceil(tcfg.extent_y / hcfg.query_radius)))
    return gx, gy


def cell_of(hcfg: SpatialHashConfig, tcfg: TerrainConfig, xy: jax.Array) -> jax.Array:
    """(..., 2) local metres -> flat cell index (...,). Clamped to the grid."""
    gx, gy = grid_dims(hcfg, tcfg)
    cx = jnp.clip((xy[..., 0] / hcfg.query_radius).astype(jnp.int32), 0, gx - 1)
    cy = jnp.clip((xy[..., 1] / hcfg.query_radius).astype(jnp.int32), 0, gy - 1)
    return cy * gx + cx


def build(hcfg: SpatialHashConfig, tcfg: TerrainConfig, xy: jax.Array, active: jax.Array):
    """Scatter points into the grid.

    Returns `(table, overflow)` where table is (gx*gy, cell_capacity) of point
    indices (or EMPTY) and overflow counts active points that did not fit.
    """
    gx, gy = grid_dims(hcfg, tcfg)
    n = xy.shape[0]
    n_cells = gx * gy
    cells = jnp.where(active, cell_of(hcfg, tcfg, xy), n_cells)  # inactive -> sink cell

    # slot = rank of this point within its cell. N here is the padded agent or
    # threat count (tens), so this O(N^2) rank is cheap, deterministic and
    # jit-friendly -- no sort, no dynamic shapes.
    idx = jnp.arange(n)
    same = (cells[:, None] == cells[None, :]) & (idx[None, :] < idx[:, None])
    slot = jnp.sum(same, axis=1)

    fits = (slot < hcfg.cell_capacity) & active
    overflow = jnp.sum(active.astype(jnp.int32)) - jnp.sum(fits.astype(jnp.int32))

    # rows [0, n_cells) are real cells; row n_cells is a sink that swallows
    # inactive points; row n_cells+1 is an always-empty guard that out-of-bounds
    # neighbourhood lookups land on (see `query`).
    table = jnp.full((n_cells + 2, hcfg.cell_capacity), EMPTY, dtype=jnp.int32)
    table = table.at[cells, jnp.clip(slot, 0, hcfg.cell_capacity - 1)].set(
        jnp.where(fits, idx.astype(jnp.int32), EMPTY)
    )
    return table.at[n_cells].set(EMPTY), overflow


def query(hcfg: SpatialHashConfig, tcfg: TerrainConfig, table: jax.Array, cell: jax.Array) -> jax.Array:
    """Candidate point indices in the 3x3 neighbourhood of `cell`. (9*capacity,)."""
    gx, gy = grid_dims(hcfg, tcfg)
    cx = cell % gx
    cy = cell // gx
    d = jnp.array([-1, 0, 1], dtype=jnp.int32)
    nx = cx + d[:, None]  # (3, 1)
    ny = cy + d[None, :]  # (1, 3)
    # Out-of-bounds neighbours are routed to the guard row rather than clamped:
    # clamping would list the same edge cell twice and hand the caller duplicate
    # candidate indices.
    inside = (nx >= 0) & (nx < gx) & (ny >= 0) & (ny < gy)
    flat = jnp.where(inside, ny * gx + nx, gx * gy + 1).reshape(-1)  # (9,)
    return table[flat].reshape(-1)


def topk_neighbours(
    hcfg: SpatialHashConfig,
    tcfg: TerrainConfig,
    query_xy: jax.Array,  # (B, 2)
    point_xy: jax.Array,  # (T, 2)
    valid: jax.Array,  # (T,) bool -- globally active points
    k: int,
    pair_valid: jax.Array | None = None,  # (B, T) bool -- per-query sensing mask
):
    """K nearest valid points within `hcfg.query_radius` of each query.

    Returns `(idx, mask)`; `idx` is (B, k) into `point_xy`, `mask` is (B, k)
    bool. Padded entries point at index 0 with mask False so downstream gathers
    stay in bounds.
    """
    table, _ = build(hcfg, tcfg, point_xy, valid)
    n_cells = grid_dims(hcfg, tcfg)[0] * grid_dims(hcfg, tcfg)[1]
    cells = cell_of(hcfg, tcfg, query_xy)  # (B,)
    cand = jax.vmap(lambda c: query(hcfg, tcfg, table, c))(cells)  # (B, 9*cap)

    ok = cand >= 0
    safe = jnp.where(ok, cand, 0)
    d2 = jnp.sum((point_xy[safe] - query_xy[:, None, :]) ** 2, axis=-1)
    within = d2 <= hcfg.query_radius**2
    keep = ok & valid[safe] & within
    if pair_valid is not None:
        keep = keep & jnp.take_along_axis(pair_valid, safe, axis=-1)
    d2 = jnp.where(keep, d2, jnp.inf)

    order = jnp.argsort(d2, axis=-1)[:, :k]
    idx = jnp.take_along_axis(safe, order, axis=-1)
    mask = jnp.isfinite(jnp.take_along_axis(d2, order, axis=-1))
    return jnp.where(mask, idx, 0), mask


def topk_bruteforce(query_xy, point_xy, valid, k: int, radius: float | None = None, pair_valid=None):
    """Reference implementation the grid is tested against."""
    d2 = jnp.sum((point_xy[None, :, :] - query_xy[:, None, :]) ** 2, axis=-1)
    keep = jnp.broadcast_to(valid[None, :], d2.shape)
    if pair_valid is not None:
        keep = keep & pair_valid
    if radius is not None:
        keep = keep & (d2 <= radius**2)
    d2 = jnp.where(keep, d2, jnp.inf)
    order = jnp.argsort(d2, axis=-1)[:, :k]
    mask = jnp.isfinite(jnp.take_along_axis(d2, order, axis=-1))
    return jnp.where(mask, order, 0), mask
