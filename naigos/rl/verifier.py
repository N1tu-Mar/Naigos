"""Deterministic CMDP verifier -- pure NumPy, zero JAX.

Nomos's rule, kept verbatim: **verify the trace, don't re-simulate it.** The
verifier is handed the logged states (positions, headings, threat positions, the
DEM) and independently recomputes every hard-constraint quantity: terrain
line-of-sight, detection probability, track build-up, lethal-envelope dwell,
terrain/ceiling/stall/bounds violations. It then compares its own numbers with
what the environment reported.

It shares no code with the JAX env by design. If a JAX rewrite quietly changes a
broadcast, a clamp or a unit, this catches it -- which a re-simulation using the
same code path never would.

Because it is pure NumPy it runs standalone: `pytest tests/test_verifier.py`
needs no accelerator, no jit warm-up and no env import beyond the config
dataclasses.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..env.config import DetectionConfig, EnvConfig, TerrainConfig, ThreatKindConfig

_LOG10 = np.log(10.0)


# --------------------------------------------------------------------- terrain
def sample_height(hmap: np.ndarray, tcfg: TerrainConfig, x, y) -> np.ndarray:
    fx = np.clip(np.asarray(x) / tcfg.cell, 0.0, tcfg.nx - 1.0)
    fy = np.clip(np.asarray(y) / tcfg.cell, 0.0, tcfg.ny - 1.0)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, tcfg.nx - 1)
    y1 = np.minimum(y0 + 1, tcfg.ny - 1)
    tx, ty = fx - x0, fy - y0
    top = hmap[y0, x0] * (1 - tx) + hmap[y0, x1] * tx
    bot = hmap[y1, x0] * (1 - tx) + hmap[y1, x1] * tx
    return top * (1 - ty) + bot * ty


def los_clearance(hmap, tcfg: TerrainConfig, dcfg: DetectionConfig, p_from, p_to) -> np.ndarray:
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    s = (np.arange(dcfg.los_samples) + 0.5) / dcfg.los_samples
    s = s.reshape((1,) * (p_from.ndim - 1) + (dcfg.los_samples,))
    seg = p_to - p_from
    px = p_from[..., None, 0] + s * seg[..., None, 0]
    py = p_from[..., None, 1] + s * seg[..., None, 1]
    pz = p_from[..., None, 2] + s * seg[..., None, 2]
    ground = sample_height(hmap, tcfg, px, py)
    d_h = np.sqrt(np.sum(seg[..., :2] ** 2, axis=-1))[..., None]
    drop = (s * d_h) * ((1.0 - s) * d_h) / (2.0 * dcfg.earth_radius_eff)
    return np.min((pz - drop) - ground, axis=-1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ------------------------------------------------------------------- detection
def detection_probability(hmap, cfg: EnvConfig, blue_pos, blue_psi, threat_pos, kp, active):
    """(T, B) detection probability, recomputed from first principles."""
    dcfg = cfg.detection
    bp = np.asarray(blue_pos, dtype=np.float64)[None, :, :]
    tp = np.asarray(threat_pos, dtype=np.float64)[:, None, :]
    d = bp - tp
    slant = np.sqrt(np.sum(d**2, axis=-1) + 1e-6)

    bearing = np.arctan2(-d[..., 1], -d[..., 0])
    aspect = bearing - np.asarray(blue_psi)[None, :]
    w = 0.5 * (1.0 - np.cos(2.0 * aspect))
    rcs = dcfg.rcs_head_on + w * (dcfg.rcs_beam - dcfg.rcs_head_on)

    snr = (
        kp["snr_ref_db"][:, None]
        + 40.0 * np.log(kp["detect_range"][:, None] / np.maximum(slant, 1.0)) / _LOG10
        + 10.0 * np.log(np.maximum(rcs / dcfg.rcs_head_on, 1e-12)) / _LOG10
    )
    pd_snr = _sigmoid(kp["snr_logistic_k"][:, None] * (snr - kp["snr_threshold_db"][:, None]))

    emitter = np.asarray(threat_pos, dtype=np.float64).copy()
    emitter[:, 2] += 10.0
    vis = _sigmoid(
        los_clearance(
            hmap, cfg.terrain, dcfg, emitter[:, None, :], np.broadcast_to(bp, d.shape)
        )
        / dcfg.los_clearance_scale
    )

    ground = sample_height(hmap, cfg.terrain, np.asarray(blue_pos)[:, 0], np.asarray(blue_pos)[:, 1])
    alt_agl = np.asarray(blue_pos)[:, 2] - ground
    lo = _sigmoid((alt_agl[None, :] - kp["alt_min"][:, None]) / np.maximum(0.25 * kp["alt_min"][:, None], 25.0))
    hi = _sigmoid((kp["alt_max"][:, None] - np.asarray(blue_pos)[None, :, 2]) / 500.0)

    pd = pd_snr * vis * lo * hi * np.asarray(active, dtype=np.float64)[:, None]
    return pd, slant, vis, alt_agl


def kind_params_numpy(cfg: EnvConfig, kind: np.ndarray) -> dict[str, np.ndarray]:
    # numeric fields only: `label` is a string and would poison the float table.
    fields = [f.name for f in dataclasses.fields(ThreatKindConfig) if f.type in ("float", "bool")]
    table = {f: np.array([float(getattr(k, f)) for k in cfg.threat_kinds], dtype=np.float64) for f in fields}
    table["detect_range"] *= cfg.red_detect_scale
    table["lethal_range"] *= cfg.red_lethal_scale
    table["reaction_latency"] *= cfg.red_latency_scale
    table["speed"] *= cfg.red_speed_scale
    return {f: table[f][np.asarray(kind)] for f in table}


# ------------------------------------------------------------------ the report
@dataclasses.dataclass
class VerificationReport:
    steps: int
    shootdowns: int
    terrain_violations: int
    bounds_violations: int
    ceiling_violations: int
    envelope_dwell_steps: int
    total_cost: float
    mean_exposure: float
    max_pd_error: float
    mismatches: list[str]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def __str__(self) -> str:
        head = (
            f"steps={self.steps} shootdowns={self.shootdowns} "
            f"terrain={self.terrain_violations} bounds={self.bounds_violations} "
            f"ceiling={self.ceiling_violations} envelope_dwell={self.envelope_dwell_steps} "
            f"cost={self.total_cost:.2f} mean_pd={self.mean_exposure:.4f} "
            f"max_pd_err={self.max_pd_error:.2e}"
        )
        if self.mismatches:
            return head + "\nMISMATCHES:\n  " + "\n  ".join(self.mismatches)
        return head + "\nOK"


def verify_trace(
    cfg: EnvConfig,
    hmap: np.ndarray,
    traj: dict[str, Any],
    threat_kind: np.ndarray,
    threat_active: np.ndarray,
    tol: float = 2e-3,
) -> VerificationReport:
    """Recheck one logged episode.

    `traj` needs, on a leading time axis: `pos` (S, B, 3), `psi` (S, B),
    `speed` (S, B), `alive` (S, B), `threat_pos` (S, T, 3), `lock` (S, T, B),
    and `terms` (the env's own RewardTerms) for cross-checking.
    """
    pos = np.asarray(traj["pos"], dtype=np.float64)
    psi = np.asarray(traj["psi"], dtype=np.float64)
    speed = np.asarray(traj["speed"], dtype=np.float64)
    alive = np.asarray(traj["alive"])
    tpos = np.asarray(traj["threat_pos"], dtype=np.float64)
    env_lock = np.asarray(traj["lock"], dtype=np.float64)
    terms = traj["terms"]

    S, B, _ = pos.shape
    kp = kind_params_numpy(cfg, threat_kind)
    af = cfg.airframe

    mismatches: list[str] = []
    shoot = terr = bounds = ceil = dwell_steps = 0
    pd_sum = 0.0
    pd_n = 0
    max_pd_err = 0.0
    lock = np.zeros((cfg.n_threat, B))
    dwell = np.zeros((cfg.n_threat, B))
    reached = np.zeros(B, dtype=bool)
    prev_alive = np.ones(B, dtype=bool)

    for s in range(S):
        live = prev_alive & ~reached
        pd, slant, _vis, alt_agl = detection_probability(
            hmap, cfg, pos[s], psi[s], tpos[s], kp, threat_active
        )
        pd = pd * live[None, :]

        lock = np.clip(
            lock + cfg.dt * (kp["lock_gain"][:, None] * pd - kp["lock_decay"][:, None] * (1 - pd) * lock),
            0.0,
            1.0,
        )
        locked = lock >= cfg.detection.lock_threshold
        dwell = np.where(locked, dwell + cfg.dt, 0.0)

        in_env = (slant <= kp["lethal_range"][:, None]) & (
            (alt_agl[None, :] >= kp["alt_min"][:, None]) & (pos[s][None, :, 2] <= kp["alt_max"][:, None])
        )
        firing = in_env & locked & (dwell >= kp["reaction_latency"][:, None])

        # --- independent recomputation of the hard constraints -----------------
        terr_v = (alt_agl < af.floor_agl) & live
        oob_v = (
            (pos[s][:, 0] < 0)
            | (pos[s][:, 0] > cfg.terrain.extent_x)
            | (pos[s][:, 1] < 0)
            | (pos[s][:, 1] > cfg.terrain.extent_y)
        ) & live
        ceil_v = (pos[s][:, 2] >= af.ceiling - 1e-3) & live
        dwell_v = np.any(in_env & (lock > 0.05), axis=0) & live

        terr += int(terr_v.sum())
        bounds += int(oob_v.sum())
        ceil += int(ceil_v.sum())
        dwell_steps += int(dwell_v.sum())

        now_alive = np.asarray(alive[s], dtype=bool)
        died = prev_alive & ~now_alive
        killed_by_fire = died & ~terr_v & ~oob_v
        shoot += int(killed_by_fire.sum())

        # A kill can only be logged where a threat actually had a firing
        # solution. This is the check that catches a phantom-shootdown bug.
        for b in np.nonzero(killed_by_fire)[0]:
            if not firing[:, b].any():
                mismatches.append(f"t={s} agent={b}: logged shootdown with no firing solution")

        # cross-check the env's own reported quantities
        err = np.abs(np.max(pd, axis=0) - np.asarray(terms.exposure[s], dtype=np.float64))
        err = np.where(live, err, 0.0)
        max_pd_err = max(max_pd_err, float(err.max()))
        if err.max() > tol:
            mismatches.append(f"t={s}: exposure mismatch, max |dpd|={err.max():.3e} > tol")

        lock_err = np.abs(lock - env_lock[s])[:, live].max() if live.any() else 0.0
        if lock_err > 10 * tol:
            mismatches.append(f"t={s}: lock mismatch, max |dlock|={lock_err:.3e}")

        stall_v = (speed[s] < af.v_stall - 1e-3) & live
        if stall_v.any():
            mismatches.append(f"t={s}: airspeed below stall on agents {np.nonzero(stall_v)[0].tolist()}")

        pd_sum += float(pd[:, live].max(axis=0).sum()) if live.any() else 0.0
        pd_n += int(live.sum())

        reached = reached | np.asarray(traj["reached"][s], dtype=bool)
        prev_alive = now_alive

    total_cost = float(shoot + terr + bounds + dwell_steps)
    return VerificationReport(
        steps=S,
        shootdowns=shoot,
        terrain_violations=terr,
        bounds_violations=bounds,
        ceiling_violations=ceil,
        envelope_dwell_steps=dwell_steps,
        total_cost=total_cost,
        mean_exposure=pd_sum / max(pd_n, 1),
        max_pd_error=max_pd_err,
        mismatches=mismatches,
    )
