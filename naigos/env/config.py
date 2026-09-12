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

# --- threat kinds -------------------------------------------------------------
# Kinds are DATA, not an enum: the count and the parameters come from
# `components/model.detection.json` via `naigos.env.theatre_bridge`, so the real
# radar calibration the research agent produced can define five classes without
# any code change. The indices below name the *default* synthetic set only.
THREAT_STATIC_SAM = 0
THREAT_MOBILE = 1
THREAT_INTERCEPTOR = 2


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
    lethal_range: float = 18_000.0  # m, engagement envelope radius
    # NOTE lethal_range << detect_range on purpose. If the two are comparable the
    # aircraft is inside the envelope by the time it is detected at all, there is
    # no standoff band to route through, and the task degenerates into a coin
    # flip. The gap between "seen" and "shootable" IS the problem being solved.
    alt_min: float = 50.0  # m AGL, lower edge of engagement envelope
    alt_max: float = 20_000.0  # m AMSL, upper edge
    reaction_latency: float = 4.0  # s of sustained lock before a shot is taken
    lock_gain: float = 0.25  # 1/s, how fast exposure builds into a track
    lock_decay: float = 0.40  # 1/s, how fast a track fades once undetected
    p_kill: float = 0.05  # per-SECOND kill hazard once locked & inside envelope
    speed: float = 0.0  # m/s, ground/air speed of the platform
    turn_rate: float = 0.0  # rad/s
    # radar: reference SNR (dB) at `detect_range` for a 1 m^2 RCS target, and the
    # detection threshold. pd is a logistic in (snr - snr_threshold).
    snr_ref_db: float = 13.0
    snr_threshold_db: float = 13.0
    snr_logistic_k: float = 0.45  # 1/dB
    airborne: bool = False  # flies a 3D track instead of sitting on the DEM
    spawn_weight: float = 1.0  # relative frequency when populating a theatre
    label: str = "generic"  # human-readable class name, carried into the demo

    # --- airborne flight limits (see docs/interceptor-3d-dynamics.md) --------
    # MODELLING ASSUMPTION, not a capability claim. These are the abstract
    # vehicle's OWN limits, deliberately kept apart from `alt_min` / `alt_max`,
    # which describe where its *weapon* can engage. A kind can be allowed to
    # engage up to 13 km and still be unable to fly there.
    #
    # They are read only when `airborne` is true, so a ground kind carrying the
    # defaults still never leaves the DEM, and a theatre-derived airborne class
    # from `theatre_bridge` inherits usable limits without a JSON change.
    climb_rate_max: float = 40.0  # m/s, best sustained rate of climb
    descent_rate_max: float = 60.0  # m/s, max rate of descent (positive magnitude)
    terrain_clearance: float = 200.0  # m AGL, hard floor the vehicle will not fly below
    vehicle_ceiling: float = 11_000.0  # m AMSL, hard flight ceiling of the vehicle


def default_threat_kinds() -> tuple[ThreatKindConfig, ...]:
    """One config per THREAT_* kind, in enum order."""
    return (
        # static SAM/radar: long reach, patient, no mobility
        ThreatKindConfig(
            detect_range=80_000.0,
            lethal_range=18_000.0,
            alt_min=60.0,
            alt_max=20_000.0,
            reaction_latency=6.0,
            p_kill=0.045,
            speed=0.0,
            turn_rate=0.0,
            spawn_weight=0.40,
            label="static_sam",
        ),
        # mobile ground unit: short reach, relocates
        ThreatKindConfig(
            detect_range=25_000.0,
            lethal_range=6_000.0,
            alt_min=0.0,
            alt_max=4_500.0,
            reaction_latency=3.0,
            lock_gain=0.35,
            p_kill=0.070,
            speed=12.0,
            turn_rate=0.15,
            snr_ref_db=11.0,
            spawn_weight=0.35,
            label="mobile_ground",
        ),
        # interceptor drone: modest sensor, closes the distance itself
        ThreatKindConfig(
            detect_range=30_000.0,
            lethal_range=4_500.0,
            alt_min=0.0,
            alt_max=13_000.0,
            reaction_latency=2.5,
            lock_gain=0.30,
            lock_decay=0.55,
            p_kill=0.090,
            speed=150.0,
            turn_rate=0.20,
            snr_ref_db=10.0,
            airborne=True,
            # asymmetric on purpose: gravity is free, thrust is not, so a dive
            # is faster than a climb and diving after blue is cheaper than
            # chasing it up. Ceiling sits BELOW alt_max (13 km): the vehicle
            # cannot fly to the top of its own engagement envelope.
            climb_rate_max=40.0,
            descent_rate_max=60.0,
            terrain_clearance=200.0,
            vehicle_ceiling=11_000.0,
            spawn_weight=0.25,
            label="interceptor",
        ),
    )


@dataclass(frozen=True)
class DetectionConfig:
    """Terrain-masked detection model knobs."""

    # Ray-march samples per (blue, threat) LOS test. Spacing is ray_length / S,
    # so at S=24 a 128 km diagonal ray is sampled every 5.3 km and walks straight
    # past ridges.
    #
    # MEASURED, 2000 rays over the Tehran DEM, emitter at ground+10 m and target
    # at 150 m AGL, fraction judged hard-visible:
    #
    #            S=24    S=48    S=96   S=192   S=384   S=768
    #   1500 m  0.1985  0.1875  0.1830  0.1800  0.1805  0.1800
    #    500 m  0.1440  0.1295  0.1225  0.1195  0.1180  0.1170
    #
    # Read both axes. At fixed S=96, refining the grid 1500 -> 500 m moves
    # visibility -6.0 pp (-33% relative) -- roughly 3x the -2.2 pp that S=24 -> 96
    # buys at 500 m. The coarse grid was smoothing ridges away and over-reporting
    # visibility by ~50% relative. So BOTH mattered, and the grid mattered more;
    # an earlier version of this comment claimed resolution bought nothing, which
    # was measured on high-altitude rollouts where it genuinely does not.
    #
    # S=96 is not converged -- it still carries ~+0.55 pp (4.7% relative) against
    # S=768 -- but it captures most of the correction at a 27% throughput cost
    # (85k -> 62k env-steps/s at the live config). That is the trade taken here.
    los_samples: int = 96
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

    dt: float = 2.0  # s per env step (decision rate)
    substeps: int = 2  # airframe integration substeps per env step
    max_steps: int = 500  # 1000 s of flight -- enough to cross the default map

    airframe: AirframeConfig = field(default_factory=AirframeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    hash: SpatialHashConfig = field(default_factory=SpatialHashConfig)
    threat_kinds: tuple[ThreatKindConfig, ...] = field(default_factory=default_threat_kinds)

    objective_radius: float = 3_000.0  # m, arrival tolerance (horizontal)

    # Boundary shaping and spawn geometry are EXTENT-RELATIVE, not absolute
    # metres. MEASURED BUG: a fixed 10 km ramp is narrower than the spawn inset
    # on the 190 km synthetic map but WIDER than it on the 97 km real theatre,
    # so every aircraft spawned inside the ramp and carried a constant penalty
    # it could not escape. Anything geometric has to scale with the map.
    spawn_inset_frac: float = 0.10  # start/objective distance from the x edges
    edge_margin_frac: float = 0.06  # boundary ramp width, fraction of min extent

    @property
    def edge_margin(self) -> float:
        return self.edge_margin_frac * min(self.terrain.extent_x, self.terrain.extent_y)

    # Per-world play area. The DEM grid is a static shape, so map-size variety is
    # a random rectangle INSIDE the grid, drawn at reset and carried as
    # EnvState.bounds; every geometric rule above is then relative to that box.
    # Off by default: every world plays on the full grid, exactly as before.
    map_randomize: bool = False
    map_min_extent_m: float = 30_000.0  # shortest allowed box side
    map_aspect_range: tuple[float, float] = (0.4, 1.5)  # box height / width (Dubai ~0.46, Owens ~1.35)

    # --- curriculum knobs (red team v2). Written by the curriculum scheduler. ---
    red_detect_scale: float = 1.0  # multiplies every kind's detect_range
    red_lethal_scale: float = 1.0
    red_latency_scale: float = 1.0  # >1 = slower red reaction = easier
    red_speed_scale: float = 1.0
    n_threat_active: int = 16  # <= n_threat; the rest spawn inactive (padding)

    # Append body-frame distances to the map edge (forward, left, right, back) to
    # the ego vector. Off by default: checkpoints trained before it existed have
    # a 10-wide ego input. See obs.py::edge_distances.
    obs_edge_features: bool = False

    # Route geometry (next-steps.md E-3; docs/route-generalization.md).
    # "legacy" is the original placement -- start on the west inset line,
    # objective on the east one -- kept verbatim so every existing checkpoint,
    # golden rollout and baseline is unchanged. "diverse" draws one route family
    # per world from `route_train_families`, with the bearing jittered inside
    # that family's sector. Held-out bearings are never drawn by any mode; they
    # are only reachable through `NaigosEnv.reset_route` from ROUTE_BANK.
    route_mode: str = "legacy"
    route_train_families: tuple[str, ...] = (
        "west_east",
        "east_west",
        "south_north",
        "north_south",
        "southwest_northeast",
        "northeast_southwest",
    )
    route_bearing_jitter_deg: float = 10.0  # half-width of each training sector
    # Cross-track offsets of starts/objectives, as a fraction of the half-side of
    # the inset box's inscribed square. <1 keeps every route line inside the box,
    # and fixes the guaranteed length (see route_min_length_frac).
    route_lateral_frac: float = 0.6
    # Every route's along-track length is at least this fraction of the play
    # box's short side. Checked analytically against the geometry at env
    # construction, and empirically by tests/test_route_generalization.py.
    route_min_length_frac: float = 0.5

    # --- observation feature widths (derived; kept here so nets can import) ---
    @property
    def ego_dim(self) -> int:
        return 14 if self.obs_edge_features else 10

    @property
    def n_threat_kinds(self) -> int:
        return len(self.threat_kinds)

    @property
    def threat_feat_dim(self) -> int:
        return 9 + self.n_threat_kinds  # see obs.py::threat_features

    @property
    def friend_feat_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 3  # bank, flight-path-angle, throttle -- NO weapon axis. Ever.

    def replace(self, **kw) -> "EnvConfig":
        return dataclasses.replace(self, **kw)


# --- route geometry: families, the frozen scenario bank, the disjointness rule ---
#
# Bearings are the direction of TRAVEL in the ENU grid, math convention (the one
# `psi` uses): 0 deg flies toward +x (east), 90 deg toward +y (north). A family
# is named origin_destination, so "west_east" starts in the west.
#
# The family index is what EnvState.route carries and what per-family metrics
# are keyed by. APPEND ONLY: reordering renumbers every recorded result.
ROUTE_FAMILY_NAMES: tuple[str, ...] = (
    "west_east",
    "east_west",
    "south_north",
    "north_south",
    "southwest_northeast",
    "northeast_southwest",
    "southeast_northwest",
    "northwest_southeast",
    "oblique",
)
ROUTE_MODES: tuple[str, ...] = ("legacy", "diverse")

# Families training may draw from, and the centre of each one's sector.
TRAIN_ROUTE_CENTERS_DEG: dict[str, float] = {
    "west_east": 0.0,
    "east_west": 180.0,
    "south_north": 90.0,
    "north_south": 270.0,
    "southwest_northeast": 45.0,
    "northeast_southwest": 225.0,
}

# Families that exist ONLY in the held-out bank. The second diagonal axis is an
# extrapolation test (a direction never flown in training); the obliques sit
# half-way between trained directions, an interpolation test.
HELDOUT_ROUTE_CENTERS_DEG: dict[str, tuple[float, ...]] = {
    "southeast_northwest": (135.0,),
    "northwest_southeast": (315.0,),
    "oblique": (22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5),
}

# Minimum angular clearance between the edge of any training sector and any
# held-out bearing. With the default 10 deg jitter the tightest pair (an oblique
# at 20.5 deg against the west_east sector ending at 10 deg) clears it by 5.5 deg.
ROUTE_SECTOR_GUARD_DEG = 5.0

ROUTE_BANK_VERSION = "route-bank-v1"
# Scenario seeds are reserved ranges, disjoint from each other, from the cloud
# pipeline's held-out seeds (900001+, naigos/pipeline/default_config.json) and
# from any training seed (`assert_route_bank_disjoint` refuses an overlap).
ROUTE_BANK_SEED_BASE: dict[str, int] = {"heldout": 710_000, "train_geometry": 720_000}
ROUTE_BANK_SEED_SPAN = 10_000
# sha256 of the canonical JSON of the bank. `route_bank()` refuses to return a
# bank that does not hash to this, so an edited bank cannot keep the v1 name.
ROUTE_BANK_V1_SHA256 = "d3b44303697da01666e2173b8d34547d90694b026c3ccd0bcbbcbfdcef86e7fa"


@dataclass(frozen=True)
class RouteScenario:
    """One frozen evaluation scenario: a route bearing plus the seed of everything else.

    The seed fixes terrain (if synthetic), play box (if map_randomize), start and
    objective cross-track offsets, the threat field and the red team's dice; the
    bearing fixes the route direction. Nothing else is free.
    """

    scenario_id: str
    split: str  # "heldout" | "train_geometry"
    family: str
    bearing_deg: float
    seed: int

    @property
    def family_id(self) -> int:
        return ROUTE_FAMILY_NAMES.index(self.family)


def _build_route_bank_v1() -> tuple[RouteScenario, ...]:
    """Enumerate route-bank-v1. Pure Python, no RNG: the bank is the loop below.

    96 scenarios:
      heldout         48  second diagonal axis, both directions, 4 bearings x 4 seeds
                          each (32); 8 obliques x 2 bearings (16)
      train_geometry  48  the six training families at bearings inside their
                          sectors, 4 bearings x 2 seeds each -- the in-distribution
                          reference, on seeds training never draws
    """
    rows: list[RouteScenario] = []
    count = {"heldout": 0, "train_geometry": 0}

    def add(split: str, family: str, bearing: float) -> None:
        i = count[split]
        count[split] += 1
        rows.append(
            RouteScenario(
                scenario_id=f"{ROUTE_BANK_VERSION}/{split}/{i:03d}",
                split=split,
                family=family,
                bearing_deg=float(bearing % 360.0),
                seed=ROUTE_BANK_SEED_BASE[split] + i,
            )
        )

    for family in ("southeast_northwest", "northwest_southeast"):
        (c,) = HELDOUT_ROUTE_CENTERS_DEG[family]
        for off in (-6.0, -2.0, 2.0, 6.0):
            for _ in range(4):
                add("heldout", family, c + off)
    for c in HELDOUT_ROUTE_CENTERS_DEG["oblique"]:
        for off in (-2.0, 2.0):
            add("heldout", "oblique", c + off)
    for family, c in TRAIN_ROUTE_CENTERS_DEG.items():
        for off in (-6.0, -2.0, 2.0, 6.0):
            for _ in range(2):
                add("train_geometry", family, c + off)
    return tuple(rows)


def route_bank_digest(rows) -> str:
    import hashlib
    import json

    blob = json.dumps([dataclasses.asdict(r) for r in rows], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def route_bank(split: str | None = None, version: str = ROUTE_BANK_VERSION) -> tuple[RouteScenario, ...]:
    """The frozen bank (optionally one split), verified against its pinned digest."""
    if version != ROUTE_BANK_VERSION:
        raise ValueError(f"unknown route bank {version!r}; this code defines {ROUTE_BANK_VERSION!r}")
    rows = _build_route_bank_v1()
    got = route_bank_digest(rows)
    if got != ROUTE_BANK_V1_SHA256:
        raise RuntimeError(
            f"{ROUTE_BANK_VERSION} hashes to {got}, not the pinned {ROUTE_BANK_V1_SHA256}. "
            "A frozen bank was edited; publish the change as a new version instead."
        )
    if split is None:
        return rows
    if split not in ROUTE_BANK_SEED_BASE:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(ROUTE_BANK_SEED_BASE)}")
    return tuple(r for r in rows if r.split == split)


def angle_between_deg(a: float, b: float) -> float:
    """Smallest absolute angle between two bearings, in [0, 180]."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def route_protocol_problems(cfg: EnvConfig) -> list[str]:
    """Everything wrong with `cfg`'s route settings; empty means usable.

    The rule that matters: no bearing training can draw may come within
    ROUTE_SECTOR_GUARD_DEG of any held-out bearing. A config that would let
    training see held-out geometry is refused, not warned about.
    """
    problems: list[str] = []
    if cfg.route_mode not in ROUTE_MODES:
        return [f"route_mode {cfg.route_mode!r} is not one of {ROUTE_MODES}"]
    if cfg.route_mode == "legacy":
        return problems  # geometry is the original west-to-east placement
    fams = tuple(cfg.route_train_families)
    if not fams:
        problems.append("route_train_families is empty")
    for f in fams:
        if f not in TRAIN_ROUTE_CENTERS_DEG:
            kind = "a held-out family" if f in HELDOUT_ROUTE_CENTERS_DEG else "not a route family"
            problems.append(f"route_train_families contains {f!r}, which is {kind}")
    if len(set(fams)) != len(fams):
        problems.append("route_train_families has duplicates")
    J = cfg.route_bearing_jitter_deg
    if not (0.0 <= J < 45.0):
        problems.append(f"route_bearing_jitter_deg {J} must be in [0, 45)")
    for row in _build_route_bank_v1():
        if row.split != "heldout":
            continue
        for f in fams:
            if f not in TRAIN_ROUTE_CENTERS_DEG:
                continue
            gap = angle_between_deg(row.bearing_deg, TRAIN_ROUTE_CENTERS_DEG[f]) - J
            if gap < ROUTE_SECTOR_GUARD_DEG:
                problems.append(
                    f"training sector {f} (+/-{J} deg) comes within {gap:.1f} deg of held-out "
                    f"{row.scenario_id} at {row.bearing_deg} deg (guard {ROUTE_SECTOR_GUARD_DEG} deg)"
                )
    f = cfg.route_lateral_frac
    if not (0.0 <= f < 1.0):
        problems.append(f"route_lateral_frac {f} must be in [0, 1)")
    elif route_min_length_guarantee(cfg) < cfg.route_min_length_frac:
        problems.append(
            f"route geometry only guarantees {route_min_length_guarantee(cfg):.3f} x the box's short "
            f"side, below route_min_length_frac {cfg.route_min_length_frac}"
        )
    if cfg.spawn_inset_frac <= cfg.edge_margin_frac:
        problems.append("spawn_inset_frac must exceed edge_margin_frac, or routes start inside the boundary ramp")
    return problems


def route_min_length_guarantee(cfg: EnvConfig) -> float:
    """Lower bound on along-track route length / short side of the play box.

    Starts and objectives sit on lines through the inset box at cross-track
    offset |s| <= f*m, where m is the half short side of the inset box. The disk
    of radius m about the centre lies inside that box, so each line runs at
    least sqrt(m^2 - s^2) >= m*sqrt(1 - f^2) either side of the centre's
    perpendicular. Both ends together: 2*m*sqrt(1 - f^2), with
    m = (0.5 - inset) * short side.
    """
    f = cfg.route_lateral_frac
    return 2.0 * (0.5 - cfg.spawn_inset_frac) * (1.0 - f * f) ** 0.5


# --- The hard invariant, asserted in code so it cannot rot silently ----------
BLUE_ACTION_NAMES: tuple[str, ...] = ("bank_cmd", "gamma_cmd", "throttle_cmd")
"""Blue's complete action space. Flight controls only.

prompt.md s0: blue is purely evasive. There is no engage/fire/target action, not
now and not behind a flag. `tests/test_invariant.py` enforces the length and the
names of this tuple; adding an offensive axis fails the suite by construction.
"""
