"""Detection: radar range equation + aspect-dependent RCS + terrain masking.

Built from first principles (prompt.md s8 guardrail) rather than from any
product data sheet. The monostatic radar range equation gives received SNR

    SNR = P_t G^2 lambda^2 sigma / ((4 pi)^3 R^4 k T B F L)

Everything except `sigma` (target RCS) and `R` (slant range) is a constant of
the emitter, so we fold them into one calibration point per threat kind:
`snr_ref_db` is the SNR of a `rcs_head_on` target at `detect_range`. Then

    SNR(R, sigma) = snr_ref + 40 log10(detect_range / R) + 10 log10(sigma / sigma_ref)

and detection probability is a logistic in (SNR - threshold), which is a smooth
stand-in for the Swerling-model Pd(SNR, Pfa) curve. Terrain LOS multiplies the
result; an altitude-band gate multiplies it again.

See components/detection.json for the citation trail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import DetectionConfig, TerrainConfig
from . import terrain as terrain_mod

_LOG10 = jnp.log(10.0)


def _db(x: jax.Array) -> jax.Array:
    return 10.0 * jnp.log(jnp.maximum(x, 1e-12)) / _LOG10


def aspect_rcs(dcfg: DetectionConfig, blue_heading: jax.Array, bearing_to_threat: jax.Array):
    """RCS in m^2 as a function of aspect angle.

    Nose-on and tail-on present the small `rcs_head_on`; broadside presents the
    much larger `rcs_beam`. Modelled as a raised cosine in 2*aspect so it is
    smooth and symmetric front/back -- a deliberate simplification, not a
    measured signature.
    """
    aspect = bearing_to_threat - blue_heading
    w = 0.5 * (1.0 - jnp.cos(2.0 * aspect))  # 0 at nose/tail, 1 at beam
    return dcfg.rcs_head_on + w * (dcfg.rcs_beam - dcfg.rcs_head_on)


def snr_db(
    dcfg: DetectionConfig,
    slant_range: jax.Array,
    rcs: jax.Array,
    detect_range: jax.Array,
    snr_ref_db: jax.Array,
) -> jax.Array:
    """Received SNR in dB. R^-4 falloff, linear in RCS."""
    r = jnp.maximum(slant_range, 1.0)
    range_term = 40.0 * jnp.log(detect_range / r) / _LOG10
    rcs_term = _db(rcs / dcfg.rcs_head_on)
    return snr_ref_db + range_term + rcs_term


def altitude_gate(alt_agl: jax.Array, alt_amsl: jax.Array, alt_min: jax.Array, alt_max: jax.Array):
    """Soft [0,1] gate for the threat's engagement altitude band.

    Below `alt_min` (AGL) the target is in the ground clutter notch; above
    `alt_max` (AMSL) it is out of reach. Smooth edges keep the reward gradient
    alive near the boundary.
    """
    lo = jax.nn.sigmoid((alt_agl - alt_min) / jnp.maximum(0.25 * alt_min, 25.0))
    hi = jax.nn.sigmoid((alt_max - alt_amsl) / 500.0)
    return lo * hi


def detection_probability(
    hmap: jax.Array,
    tcfg: TerrainConfig,
    dcfg: DetectionConfig,
    blue_pos: jax.Array,  # (B, 3)
    blue_heading: jax.Array,  # (B,)
    threat_pos: jax.Array,  # (T, 3)
    kind_params: dict[str, jax.Array],  # each (T,)
    threat_active: jax.Array,  # (T,) bool
):
    """Per-step detection probability for every (threat, blue) pair.

    Returns a dict with `pd` (T, B), `slant` (T, B), `vis` (T, B) so callers can
    reuse the geometry for the reward and the verifier without recomputing it.
    """
    bp = blue_pos[None, :, :]  # (1, B, 3)
    tp = threat_pos[:, None, :]  # (T, 1, 3)
    d = bp - tp  # (T, B, 3)
    slant = jnp.sqrt(jnp.sum(d**2, axis=-1) + 1e-6)

    bearing = jnp.arctan2(-d[..., 1], -d[..., 0])  # from blue toward threat
    rcs = aspect_rcs(dcfg, blue_heading[None, :], bearing)

    s = snr_db(
        dcfg,
        slant,
        rcs,
        kind_params["detect_range"][:, None],
        kind_params["snr_ref_db"][:, None],
    )
    pd_snr = jax.nn.sigmoid(kind_params["snr_logistic_k"][:, None] * (s - kind_params["snr_threshold_db"][:, None]))

    # radar horizon / terrain masking: the emitter sits ~10 m above its ground
    emitter = threat_pos + jnp.array([0.0, 0.0, 10.0])
    vis = terrain_mod.visibility(hmap, tcfg, dcfg, emitter[:, None, :], jnp.broadcast_to(bp, d.shape))

    ground = terrain_mod.sample_height(hmap, tcfg, blue_pos[:, 0], blue_pos[:, 1])
    alt_agl = blue_pos[:, 2] - ground
    gate = altitude_gate(
        alt_agl[None, :],
        blue_pos[None, :, 2],
        kind_params["alt_min"][:, None],
        kind_params["alt_max"][:, None],
    )

    pd = pd_snr * vis * gate * threat_active[:, None].astype(jnp.float32)
    return {"pd": pd, "slant": slant, "vis": vis, "alt_agl": alt_agl, "snr_db": s}


def update_track(
    lock: jax.Array,  # (T, B) in [0, 1]
    pd: jax.Array,  # (T, B)
    kind_params: dict[str, jax.Array],
    dt: float,
) -> jax.Array:
    """Integrate per-step detections into a persistent track/lock quality.

    Rises at `lock_gain * pd`, decays at `lock_decay * (1 - pd)`. A single lucky
    detection does not produce a lock; sustained exposure does. This is what
    makes *timing* -- popping up briefly then re-masking -- a real tactic.
    """
    gain = kind_params["lock_gain"][:, None]
    decay = kind_params["lock_decay"][:, None]
    d_lock = gain * pd - decay * (1.0 - pd) * lock
    return jnp.clip(lock + dt * d_lock, 0.0, 1.0)


def engagement(
    dcfg: DetectionConfig,
    lock: jax.Array,  # (T, B)
    dwell: jax.Array,  # (T, B) seconds of continuous lock
    slant: jax.Array,  # (T, B)
    blue_alt_agl: jax.Array,  # (B,)
    blue_alt_amsl: jax.Array,  # (B,)
    kind_params: dict[str, jax.Array],
    dt: float = 1.0,
):
    """Which (threat, blue) pairs are inside a lethal envelope with a firing solution.

    Returns `in_envelope` (T,B) float, `firing` (T,B) float, and the per-blue
    per-step kill hazard `hazard` (B,).
    """
    in_range = (slant <= kind_params["lethal_range"][:, None]).astype(jnp.float32)
    band = (
        (blue_alt_agl[None, :] >= kind_params["alt_min"][:, None])
        & (blue_alt_amsl[None, :] <= kind_params["alt_max"][:, None])
    ).astype(jnp.float32)
    in_envelope = in_range * band

    locked = (lock >= dcfg.lock_threshold).astype(jnp.float32)
    ready = (dwell >= kind_params["reaction_latency"][:, None]).astype(jnp.float32)
    firing = in_envelope * locked * ready

    # p_kill is a per-SECOND hazard, so a step of length dt survives it with
    # probability (1 - p)^dt. Getting this wrong silently makes the whole world
    # dt-dependent -- the calibration bug prompt.md s5 warns about.
    p_step = 1.0 - (1.0 - jnp.clip(kind_params["p_kill"], 0.0, 0.999)) ** dt
    h = firing * p_step[:, None]
    hazard = 1.0 - jnp.prod(1.0 - jnp.clip(h, 0.0, 0.999), axis=0)
    return in_envelope, firing, hazard
