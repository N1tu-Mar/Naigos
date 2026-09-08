"""Terrain: bilinear DEM sampling and radar line-of-sight occlusion.

The DEM lives as a single `(ny, nx)` float32 heightmap in a local ENU frame.
`naigos.data.dem` produces it from a real elevation source (Copernicus / 3DEP);
`synthetic_terrain` here produces a deterministic ridge-and-valley stand-in so
the env, the tests and the training loop all run with zero network access.

The signature mechanic of Naigos is that flying low behind a ridge breaks the
radar ray and collapses detection probability. That happens in `los_clearance`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import DetectionConfig, TerrainConfig


def synthetic_terrain(key: jax.Array, cfg: TerrainConfig) -> jax.Array:
    """Deterministic ridged terrain in metres AMSL, shape (ny, nx).

    Sum of a few anisotropic sinusoids plus a smooth random field. Produces
    ridgelines with real valleys -- enough relief that LOS masking is a genuine
    tactical option rather than a rounding error.
    """
    ys = jnp.arange(cfg.ny, dtype=jnp.float32) * cfg.cell
    xs = jnp.arange(cfg.nx, dtype=jnp.float32) * cfg.cell
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")

    k1, k2 = jax.random.split(key)
    phase = jax.random.uniform(k1, (4,), minval=0.0, maxval=2 * jnp.pi)
    L = max(cfg.extent_x, cfg.extent_y)

    ridges = (
        900.0 * jnp.sin(2 * jnp.pi * (0.9 * X + 0.4 * Y) / L + phase[0])
        + 600.0 * jnp.sin(2 * jnp.pi * (-0.5 * X + 1.1 * Y) / L + phase[1])
        + 320.0 * jnp.sin(2 * jnp.pi * (2.3 * X + 1.7 * Y) / L + phase[2])
        + 140.0 * jnp.sin(2 * jnp.pi * (3.1 * X - 2.9 * Y) / L + phase[3])
    )
    # low-frequency random basin so every episode is not the same map
    noise = jax.random.normal(k2, (8, 8), dtype=jnp.float32)
    noise = jax.image.resize(noise, (cfg.ny, cfg.nx), method="cubic") * 250.0

    h = 700.0 + ridges + noise
    return jnp.maximum(h, 0.0).astype(jnp.float32)


def sample_height(hmap: jax.Array, cfg: TerrainConfig, x: jax.Array, y: jax.Array) -> jax.Array:
    """Bilinear DEM lookup at local ENU metres. Clamps at the grid edge."""
    fx = jnp.clip(x / cfg.cell, 0.0, cfg.nx - 1.0)
    fy = jnp.clip(y / cfg.cell, 0.0, cfg.ny - 1.0)
    x0 = jnp.floor(fx).astype(jnp.int32)
    y0 = jnp.floor(fy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, cfg.nx - 1)
    y1 = jnp.minimum(y0 + 1, cfg.ny - 1)
    tx = fx - x0
    ty = fy - y0

    h00 = hmap[y0, x0]
    h10 = hmap[y0, x1]
    h01 = hmap[y1, x0]
    h11 = hmap[y1, x1]
    top = h00 * (1 - tx) + h10 * tx
    bot = h01 * (1 - tx) + h11 * tx
    return top * (1 - ty) + bot * ty


def los_clearance(
    hmap: jax.Array,
    tcfg: TerrainConfig,
    dcfg: DetectionConfig,
    p_from: jax.Array,
    p_to: jax.Array,
) -> jax.Array:
    """Minimum ray-above-terrain clearance in metres along the segment.

    Marches `dcfg.los_samples` interior samples between the two points. At each
    sample the straight-line ray height is corrected downward for earth
    curvature (4/3-earth radar refraction model) before being compared with the
    DEM. A negative return means terrain cuts the ray: the target is masked.

    Shapes broadcast: `p_from` and `p_to` are (..., 3); the result is (...).
    """
    s = (jnp.arange(dcfg.los_samples, dtype=jnp.float32) + 0.5) / dcfg.los_samples
    s = s.reshape((1,) * (p_from.ndim - 1) + (dcfg.los_samples,))  # (..., S)

    seg = p_to - p_from  # (..., 3)
    px = p_from[..., None, 0] + s * seg[..., None, 0]
    py = p_from[..., None, 1] + s * seg[..., None, 1]
    pz = p_from[..., None, 2] + s * seg[..., None, 2]

    ground = sample_height(hmap, tcfg, px, py)

    # curvature drop of the chord relative to a straight line: d1*d2/(2*Re_eff)
    d_horiz = jnp.sqrt(jnp.sum(seg[..., :2] ** 2, axis=-1))[..., None]
    d1 = s * d_horiz
    d2 = (1.0 - s) * d_horiz
    drop = d1 * d2 / (2.0 * dcfg.earth_radius_eff)

    clearance = (pz - drop) - ground  # (..., S)
    return jnp.min(clearance, axis=-1)


def visibility(
    hmap: jax.Array,
    tcfg: TerrainConfig,
    dcfg: DetectionConfig,
    p_from: jax.Array,
    p_to: jax.Array,
) -> jax.Array:
    """Soft LOS factor in [0, 1]. 1 = clear ray, 0 = fully terrain-masked.

    Soft rather than boolean so the gradient of exposure w.r.t. altitude is
    informative for the reward shaping -- a hard step would give the policy no
    signal about *how close* it is to breaking the ray.
    """
    c = los_clearance(hmap, tcfg, dcfg, p_from, p_to)
    return jax.nn.sigmoid(c / dcfg.los_clearance_scale)


def terrain_gradient(hmap: jax.Array, cfg: TerrainConfig, x: jax.Array, y: jax.Array):
    """(dh/dx, dh/dy) in m/m via central differences on the bilinear surface."""
    e = cfg.cell
    hx1 = sample_height(hmap, cfg, x + e, y)
    hx0 = sample_height(hmap, cfg, x - e, y)
    hy1 = sample_height(hmap, cfg, x, y + e)
    hy0 = sample_height(hmap, cfg, x, y - e)
    return (hx1 - hx0) / (2 * e), (hy1 - hy0) / (2 * e)
