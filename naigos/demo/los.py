"""The refracted line-of-sight ray, as a drawable polyline.

``naigos.env.terrain.los_clearance`` marches ``dcfg.los_samples`` interior
samples along the segment, drops each one for 4/3-earth radar refraction, and
returns the *minimum* clearance. That single number is all the detection model
needs, and it is all the frame ever carried -- so the viewer drew the ray as a
straight two-point line between the emitter and the aircraft.

A straight line is the wrong curve. Over an 80 km ray the refraction drop
reaches ~100 m at the midpoint, so on grazing geometry the drawn line clears a
ridge the model says it does not: the picture contradicts the number printed
next to it, and the picture is what a reader believes. This module reconstructs
the same sampled ray the model marched, so the polyline on screen is the
geometry the clearance was computed from, and marks the sample where the
clearance is worst -- the point the ray is actually pinched at.

Pure NumPy, and it imports nothing from ``naigos.env`` or ``naigos.rl``.

That is deliberate, and it is the same rule ``naigos/rl/verifier.py`` follows:
a check that reuses the env's own functions cannot catch a bug in those
functions. Here the consequence is sharper still -- if this module called into
the env's sampler, "the drawing agrees with the model" would be true by
construction and would prove nothing. Instead the drop formula is re-derived
from the same physics, and ``tests/test_los_profile.py`` asserts the two agree
numerically. When they stop agreeing, that test fails, which is the point.

The ground heights are supplied by the caller rather than sampled here, because
sampling the DEM is exactly the operation this module must not own: the surface
belongs to the simulation, and the renderer's job is to draw what the
simulation used, not to look it up again.
"""

from __future__ import annotations

import numpy as np

#: Vertices actually sent to the browser per ray. The model marches
#: ``los_samples`` (96 by default) of them; sending all 96 for every aircraft on
#: every tick is ~10x the frame budget for a curve whose whole shape is one
#: shallow arc. 32 is enough to render that arc smoothly, and the pinch sample
#: is always forced into the selection (see :func:`draw_indices`), so the one
#: vertex that carries a claim is never the one dropped.
DRAW_SAMPLES = 32

#: Clearance bands, in metres, keyed to ``DetectionConfig.los_clearance_scale``.
#: ``visibility()`` is a sigmoid rather than a step, so "grazing" is a real
#: state and not a rounding artifact.
CLEARANCE_MASKED = -60.0
CLEARANCE_GRAZING = 60.0


def sample_fractions(los_samples: int) -> np.ndarray:
    """The interior sample positions along the segment, in [0, 1].

    Midpoint rule, matching ``naigos.env.terrain.los_clearance`` exactly:
    ``(i + 0.5) / n``. Endpoints are deliberately not sampled -- the emitter and
    the aircraft are both known to be above the ground they sit on, so including
    them would only ever dilute the minimum.
    """
    if los_samples < 1:
        raise ValueError(f"los_samples must be >= 1, got {los_samples}")
    return (np.arange(los_samples, dtype=np.float64) + 0.5) / los_samples


def curvature_drop(s: np.ndarray, d_horiz: np.ndarray, earth_radius_eff: float) -> np.ndarray:
    """4/3-earth refraction drop of the chord below the straight line, in metres.

    ``d1 * d2 / (2 * Re_eff)`` where ``d1``/``d2`` are the horizontal distances
    from each end to the sample. Zero at both ends, maximal at the midpoint, and
    quadratic in range -- which is why it is negligible on a 10 km look and
    ~100 m on an 80 km one.

    ``s`` is (..., S); ``d_horiz`` broadcasts against it as (..., 1).
    """
    d1 = s * d_horiz
    d2 = (1.0 - s) * d_horiz
    return d1 * d2 / (2.0 * earth_radius_eff)


def profile(
    p_from: np.ndarray,
    p_to: np.ndarray,
    ground: np.ndarray,
    *,
    los_samples: int,
    earth_radius_eff: float,
) -> dict:
    """Reconstruct the sampled, refracted ray the model marched.

    Parameters
    ----------
    p_from, p_to
        (..., 3) local ENU endpoints -- emitter and aircraft, in that order.
    ground
        (..., S) terrain heights already sampled by the caller at the S sample
        points this function lays out. ``sample_fractions`` defines where those
        are, so a caller that uses it cannot put them anywhere else.
    los_samples, earth_radius_eff
        From ``DetectionConfig``. Passed in rather than imported, so this module
        stays independent of the env package.

    Returns a dict of arrays:

    ``x``/``y``/``z``   (..., S) the ray's own path, with the drop applied to z.
                        Drawing these vertices draws the geometry the clearance
                        was computed from.
    ``ground``          (..., S) as given.
    ``clearance``       (..., S) ``(z - drop) - ground``, the per-sample version
                        of what the env reduces with ``min``.
    ``pinch``           (...) index of the worst sample.
    ``min_clearance``   (...) the value at that index. Equal to
                        ``naigos.env.terrain.los_clearance`` on the same inputs.
    """
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    ground = np.asarray(ground, dtype=np.float64)

    s = sample_fractions(los_samples)
    s = s.reshape((1,) * (p_from.ndim - 1) + (los_samples,))  # (..., S)

    seg = p_to - p_from
    x = p_from[..., None, 0] + s * seg[..., None, 0]
    y = p_from[..., None, 1] + s * seg[..., None, 1]
    z = p_from[..., None, 2] + s * seg[..., None, 2]

    d_horiz = np.sqrt(np.sum(seg[..., :2] ** 2, axis=-1))[..., None]
    drop = curvature_drop(s, d_horiz, earth_radius_eff)

    z_ray = z - drop
    clearance = z_ray - ground
    pinch = np.argmin(clearance, axis=-1)

    return {
        "x": x,
        "y": y,
        "z": z_ray,
        "drop": drop * np.ones_like(z),
        "ground": ground,
        "clearance": clearance,
        "pinch": pinch,
        "min_clearance": np.min(clearance, axis=-1),
    }


def draw_indices(los_samples: int, pinch: int, draw_samples: int = DRAW_SAMPLES) -> np.ndarray:
    """Which samples to send to the browser, always including the pinch.

    An evenly spaced subset, with the worst sample forced in. Dropping the pinch
    would leave a polyline that is smooth, plausible, and missing the single
    vertex the whole overlay exists to show; forcing it in costs one index.

    Returned sorted and unique, so the vertices stay in ray order -- a polyline
    whose points are out of order draws a bowtie, not a ray.
    """
    if draw_samples < 2:
        raise ValueError(f"draw_samples must be >= 2, got {draw_samples}")
    if draw_samples >= los_samples:
        idx = np.arange(los_samples)
    else:
        idx = np.unique(np.linspace(0, los_samples - 1, draw_samples).round().astype(int))
    return np.unique(np.concatenate([idx, [int(pinch)]]))


def band(min_clearance: float) -> str:
    """The three-state description of a ray: masked, grazing, or clear."""
    if min_clearance < CLEARANCE_MASKED:
        return "masked"
    if min_clearance < CLEARANCE_GRAZING:
        return "grazing"
    return "clear"
