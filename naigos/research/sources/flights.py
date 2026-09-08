"""Real aircraft kinematics (OpenSky Network ADS-B) -> airframe envelope calibration.

What this does and does not establish, stated plainly because it bounds every claim built on it:

  * ADS-B traffic is overwhelmingly civil transport and general aviation. It calibrates the
    point-mass model's *plausible operating envelope* -- cruise and manoeuvre speeds, sustained
    climb and descent rates, and the turn rates aircraft actually fly -- against measured
    behaviour instead of invented numbers.
  * It does NOT characterise tactical aircraft performance, and nothing here claims to. The
    g-limit in the env is a physics relationship (turn rate = g*tan(phi)/V), not a datum; what
    OpenSky supplies is the observed *speed and climb* envelope, plus a check that the observed
    turn rates are consistent with the bank angles that relationship predicts.

Sampling method: repeated ``/states/all`` snapshots over a fixed busy box. Consecutive snapshots
of the same ICAO24 address give a finite-difference turn rate and longitudinal acceleration that
the instantaneous state vector does not carry.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from .. import cache

STATES_URL = "https://opensky-network.org/api/states/all"

# A deliberately busy box (Southern California / Los Angeles basin), under 25 square degrees so
# it costs a single anonymous API credit per snapshot. The calibration box is not the AOI: we
# want traffic density for statistics, not traffic over the specific terrain.
CALIBRATION_BOX = {"lamin": 32.8, "lomin": -120.5, "lamax": 36.8, "lomax": -116.0}

# OpenSky serves anonymous clients at 10 s time resolution; sample slower than that so
# consecutive snapshots are genuinely independent updates rather than the same state repeated.
SNAPSHOT_INTERVAL_S = 12.0

# states/all vector layout, https://openskynetwork.github.io/opensky-api/rest.html
IDX = {
    "icao24": 0, "callsign": 1, "origin_country": 2, "time_position": 3, "last_contact": 4,
    "longitude": 5, "latitude": 6, "baro_altitude": 7, "on_ground": 8, "velocity": 9,
    "true_track": 10, "vertical_rate": 11, "sensors": 12, "geo_altitude": 13, "squawk": 14,
    "spi": 15, "position_source": 16,
}


def sample_states(
    n_snapshots: int = 24,
    interval_s: float = SNAPSHOT_INTERVAL_S,
    box: dict[str, float] | None = None,
    force: bool = False,
) -> cache.Artifact:
    """Collect and cache ``n_snapshots`` of live traffic over the calibration box."""
    box = box or CALIBRATION_BOX
    key = f"opensky/states/{n_snapshots}x{int(interval_s)}s"

    def build() -> bytes:
        import requests

        from ..allowlist import check_url

        check_url(STATES_URL, "opensky")
        snaps: list[dict[str, Any]] = []
        for i in range(n_snapshots):
            if i:
                time.sleep(interval_s)
            resp = requests.get(
                STATES_URL, params=box, timeout=45,
                headers={"User-Agent": cache.USER_AGENT},
            )
            if resp.status_code == 429:  # rate limited; keep what we have rather than fail
                break
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("states"):
                snaps.append(payload)
        return json.dumps({"box": box, "interval_s": interval_s, "snapshots": snaps}).encode()

    return cache.produce(
        key=key, source_key="opensky", url=STATES_URL,
        rel_path=f"flights/opensky_states_{n_snapshots}x{int(interval_s)}s.json",
        builder=build, force=force,
        note=f"{n_snapshots} live state snapshots over {box}, {interval_s}s apart.",
    )


def _wrap180(deg: np.ndarray) -> np.ndarray:
    return (deg + 180.0) % 360.0 - 180.0


def derive_envelope(art: cache.Artifact) -> dict[str, Any]:
    """Reduce the snapshots to the airframe envelope numbers the env is parameterized with."""
    doc = json.loads(art.abs_path.read_text())
    snaps = doc["snapshots"]

    # Airborne instantaneous states.
    speed, vrate, alt, track, tstamp, icao = [], [], [], [], [], []
    for snap in snaps:
        for s in snap["states"]:
            if s[IDX["on_ground"]]:
                continue
            v, vr, a, tr = (s[IDX[k]] for k in ("velocity", "vertical_rate", "baro_altitude", "true_track"))
            if None in (v, tr) or a is None:
                continue
            speed.append(v); vrate.append(vr if vr is not None else 0.0)
            alt.append(a); track.append(tr)
            tstamp.append(s[IDX["last_contact"]]); icao.append(s[IDX["icao24"]])

    speed = np.asarray(speed); vrate = np.asarray(vrate)
    alt = np.asarray(alt); track = np.asarray(track)
    tstamp = np.asarray(tstamp, dtype=float); icao = np.asarray(icao)

    # Finite-difference turn rate and acceleration per aircraft between consecutive snapshots.
    turn_rate, accel, bank_deg = [], [], []
    order = np.lexsort((tstamp, icao))
    ic, ts, tk, sp = icao[order], tstamp[order], track[order], speed[order]
    for i in range(len(ic) - 1):
        if ic[i] != ic[i + 1]:
            continue
        dt = ts[i + 1] - ts[i]
        if not (2.0 <= dt <= 60.0):
            continue
        dpsi = float(_wrap180(np.asarray(tk[i + 1] - tk[i])))
        rate = dpsi / dt
        if abs(rate) > 15.0:  # >15 deg/s is a decode artefact, not an airliner
            continue
        turn_rate.append(rate)
        accel.append((sp[i + 1] - sp[i]) / dt)
        v = 0.5 * (sp[i] + sp[i + 1])
        if v > 40.0:
            # Coordinated level turn: omega = g*tan(phi)/V  =>  phi = atan(omega*V/g)
            bank_deg.append(math.degrees(math.atan(abs(math.radians(rate)) * v / 9.80665)))

    def q(a: list | np.ndarray, ps=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return {}
        return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps} | {
            "n": int(a.size), "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
        }

    climb = vrate[vrate > 0.5]
    descent = vrate[vrate < -0.5]
    return {
        "n_snapshots": len(snaps),
        "n_airborne_states": int(speed.size),
        "n_tracked_pairs": len(turn_rate),
        "ground_speed_ms": q(speed),
        "baro_altitude_m": q(alt),
        "climb_rate_ms": q(climb),
        "descent_rate_ms": q(-descent),
        "turn_rate_deg_s": q(np.abs(turn_rate)),
        "long_accel_ms2": q(np.abs(accel)),
        "implied_bank_deg": q(bank_deg),
        "caveat": (
            "Civil ADS-B traffic. Bounds the plausible transport/GA envelope and validates the "
            "coordinated-turn relationship; it is not tactical aircraft performance data."
        ),
    }
