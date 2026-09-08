"""Static configuration for the Naigos airspace environment.

Everything here is a frozen dataclass so it can be closed over by `jax.jit` as a
static argument. Nothing in this module holds traced arrays.

Units are SI throughout: metres, seconds, radians, kilograms. Altitudes are
metres above mean sea level (AMSL) to match the DEM.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

G0 = 9.80665  # m/s^2

# --- threat kind enum (kept as plain ints so it survives jit) -----------------
THREAT_STATIC_SAM = 0  # fixed radar + SAM site
THREAT_MOBILE = 1  # tank / mobile SAM: drives, short range
THREAT_INTERCEPTOR = 2  # airborne pursuer, vectors onto detected blue
N_THREAT_KINDS = 3


@dataclass(frozen=True)
class AirframeConfig:
    """Point-mass 3D airframe limits.

    Defaults are placeholders in the right order of magnitude for a subsonic
    strike aircraft. They MUST be recalibrated from OpenSky-derived statistics
    before any published number is claimed -- see docs/DATA.md.
    """

    v_stall: float = 65.0  # m/s, minimum airspeed
    v_max: float = 300.0  # m/s
    v_init: float = 180.0  # m/s
    n_max: float = 7.0  # max load factor (g), caps bank angle
    gamma_max: float = 0.35  # rad, ~20 deg climb/descent path angle
    roc_max: float = 60.0  # m/s, max vertical rate (limits gamma at speed)
    ceiling: float = 12000.0  # m AMSL service ceiling
    floor_agl: float = 30.0  # m AGL, below this counts as a terrain violation

    # first-order actuator lags (time constants, seconds)
    tau_bank: float = 0.5
    tau_gamma: float = 1.0
    tau_speed: float = 4.0

    # fuel: kg, burn = base + throttle term + manoeuvre (load-factor) term
    fuel_init: float = 3000.0
    burn_base: float = 0.25  # kg/s
    burn_throttle: float = 0.90  # kg/s at full throttle
    burn_maneuver: float = 0.35  # kg/s per excess g


@dataclass(frozen=True)
class ThreatKindConfig:
    """Parameterised threat abstraction.

    GUARDRAIL (prompt.md s8): these are *shape* parameters for the exposure /
    survival tradeoff, not a capability database for any fielded system. Values
    are nominal, open-literature-order magnitudes chosen to make the RL problem
    interesting. See components/red_team.json.
    """

    detect_range: float = 80_000.0  # m, range at which pd crosses ~0.5 head-on, clear LOS
    lethal_range: float = 40_000.0  # m, engagement envelope radius
    alt_min: float = 50.0  # m AGL, lower edge of engagement envelope
    alt_max: float = 20_000.0  # m AMSL, upper edge
    reaction_latency: float = 4.0  # s of sustained lock before a shot is taken
    lock_gain: float = 1.0  # 1/s, how fast exposure builds into a track
    lock_decay: float = 0.35  # 1/s, how fast a track fades once undetected
    p_kill: float = 0.20  # per-second kill hazard once locked & inside envelope
    speed: float = 0.0  # m/s, ground/air speed of the platform
    turn_rate: float = 0.0  # rad/s
    # radar: reference SNR (dB) at `detect_range` for a 1 m^2 RCS target, and the
    # detection threshold. pd is a logistic in (snr - snr_threshold).
    snr_ref_db: float = 13.0
    snr_threshold_db: float = 13.0
    snr_logistic_k: float = 0.45  # 1/dB


def default_threat_kinds() -> tuple[ThreatKindConfig, ...]:
    """One config per THREAT_* kind, in enum order."""
    return (
        # static SAM/radar: long reach, patient, no mobility
        ThreatKindConfig(
            detect_range=80_000.0,
            lethal_range=40_000.0,
            alt_min=60.0,
            alt_max=20_000.0,
            reaction_latency=5.0,
            p_kill=0.25,
            speed=0.0,
            turn_rate=0.0,
        ),
        # mobile ground unit: short reach, relocates
        ThreatKindConfig(
            detect_range=25_000.0,
            lethal_range=8_000.0,
            alt_min=0.0,
            alt_max=4_500.0,
            reaction_latency=2.5,
            lock_gain=1.4,
            p_kill=0.30,
            speed=12.0,
            turn_rate=0.15,
            snr_ref_db=11.0,
        ),
        # interceptor drone: modest sensor, closes the distance itself
        ThreatKindConfig(
            detect_range=30_000.0,
            lethal_range=6_000.0,
            alt_min=0.0,
            alt_max=13_000.0,
            reaction_latency=2.0,
            lock_gain=1.2,
            lock_decay=0.5,
            p_kill=0.35,
            speed=150.0,
            turn_rate=0.20,
            snr_ref_db=10.0,
        ),
    )


@dataclass(frozen=True)
class DetectionConfig:
    """Terrain-masked detection model knobs."""

    los_samples: int = 24  # ray-march samples per (blue, threat) LOS test
    los_clearance_scale: float = 60.0  # m, softness of the terrain-mask sigmoid
    rcs_head_on: float = 1.0  # m^2, RCS at nose/tail aspect
    rcs_beam: float = 6.0  # m^2, RCS at beam aspect (broadside is bigger)
    # earth curvature: drop = d^2 / (2 * k * R_earth), k=4/3 for radar refraction
    earth_radius_eff: float = 8_495_000.0
    lock_threshold: float = 0.6  # track quality that counts as "locked"


@dataclass(frozen=True)
class TerrainConfig:
    """Local ENU grid the DEM is resampled onto.

    Origin is the (0,0) corner of the grid in local metres. Cell size is uniform.
    """

    nx: int = 128
    ny: int = 128
    cell: float = 1_500.0  # m per cell -> 192 km square at defaults

    @property
    def extent_x(self) -> float:
        return (self.nx - 1) * self.cell

    @property
    def extent_y(self) -> float:
        return (self.ny - 1) * self.cell


@dataclass(frozen=True)
class SpatialHashConfig:
    """Fixed-capacity uniform grid used for neighbour queries (O(N*C)).

    `query_radius` is both the cell side and the search cutoff: the 3x3 cell
    neighbourhood around a query provably covers every point within that radius.
    Set it to the largest sensor reach an agent should be able to perceive.
    """

    query_radius: float = 90_000.0  # m -- an aircraft's RWR/sensing horizon
    cell_capacity: int = 32  # must exceed the max points expected in one cell;
    # `build` returns an overflow count and tests assert it stays at 0.


@dataclass(frozen=True)
class EnvConfig:
    """Top-level env spec. Padded, fixed-size -- everything is jit/vmap safe."""

    n_blue: int = 4
    n_threat: int = 16
    k_threat_obs: int = 6  # K nearest *sensed* threats in the observation
    k_friend_obs: int = 3

    dt: float = 1.0  # s per env step
    max_steps: int = 400

    airframe: AirframeConfig = field(default_factory=AirframeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    hash: SpatialHashConfig = field(default_factory=SpatialHashConfig)
    threat_kinds: tuple[ThreatKindConfig, ...] = field(default_factory=default_threat_kinds)

    objective_radius: float = 3_000.0  # m, arrival tolerance (horizontal)

    # --- curriculum knobs (red team v2). Written by the curriculum scheduler. ---
    red_detect_scale: float = 1.0  # multiplies every kind's detect_range
    red_lethal_scale: float = 1.0
    red_latency_scale: float = 1.0  # >1 = slower red reaction = easier
    red_speed_scale: float = 1.0
    n_threat_active: int = 16  # <= n_threat; the rest spawn inactive (padding)

    # --- observation feature widths (derived; kept here so nets can import) ---
    @property
    def ego_dim(self) -> int:
        return 10

    @property
    def threat_feat_dim(self) -> int:
        return 9 + N_THREAT_KINDS  # see obs.py::threat_features

    @property
    def friend_feat_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 3  # bank, flight-path-angle, throttle -- NO weapon axis. Ever.

    def replace(self, **kw) -> "EnvConfig":
        return dataclasses.replace(self, **kw)


# --- The hard invariant, asserted in code so it cannot rot silently ----------
BLUE_ACTION_NAMES: tuple[str, ...] = ("bank_cmd", "gamma_cmd", "throttle_cmd")
"""Blue's complete action space. Flight controls only.

prompt.md s0: blue is purely evasive. There is no engage/fire/target action, not
now and not behind a flag. `tests/test_invariant.py` enforces the length and the
names of this tuple; adding an offensive axis fails the suite by construction.
"""
