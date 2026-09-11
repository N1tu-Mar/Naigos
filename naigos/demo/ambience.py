"""Fictional conflict ambience: a deterministic, art-directed VFX stream. Presentation only.

``conflict_ambience`` is an opt-in presentation profile that puts recurring,
distant, non-graphic effects on the horizon of a replay or a live session --
a blast flash, a column of smoke rising and drifting, a dust burst, a spray of
sparks -- so the scene reads as a high-intensity simulated setting. It is an
art-directed VFX stream and nothing else:

* **Not a model.** No munition, projectile, strike, target or damage is
  simulated or implied. An effect has an origin, a start time and a look; it
  has no cause, no object and no outcome, and the page labels the whole layer
  ``FICTIONAL AMBIENCE -- not simulated events``.
* **Never an input.** This module imports nothing from ``naigos.env`` or
  ``naigos.rl`` and nothing there imports it. It is handed plain coordinates
  (copies) of where the aircraft and threats are, so it can keep away from
  them, and it hands back a list of dicts. There is no path from here into a
  state, an observation, a reward, LOS or detection;
  ``tests/test_city_presentation.py`` steps the env with and without a stream
  and compares every leaf.
* **Deterministic.** Every event is a pure function of (visual seed, setting,
  scene mask, time bucket). Time is cut into ``BUCKET_S`` buckets and each
  bucket draws from its own seeded generator, and the concurrency cap counts
  that raw stream rather than what it kept, so the stream over [0, 600) is
  exactly the concatenation of the streams over [0, 300) and [300, 600), a
  replay export and a re-export are identical, and a live session with the
  same seed (and the same entities) shows the same effects at the same times. There is no browser timer and
  no ``Math.random`` anywhere in the effect path.
* **Nowhere in particular.** Origins come from a synthetic scene mask: a
  hashed value-noise field thresholded inside the city's coarse, hand-drawn
  ``ambience`` regions (open land, far from the city core), minus the urban
  sub-box, minus every protected zone with the ``ambience`` policy (buffered,
  plus a margin), minus an edge margin. No map search, no named place, no
  building, facility or road is consulted, and no origin can be within
  ``entity_standoff_m`` of an aircraft or a threat at the moment it starts.
* **Bounded.** A setting's rate is capped at ``MAX_RATE_PER_MIN`` and each
  bucket at ``max_per_bucket``; overlapping effects are capped at
  ``max_concurrent``. The page drops to a lighter rendering under
  ``prefers-reduced-motion`` and in performance mode.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass

from .cities import CityConfig, distance_m, m_per_deg_lon, point_in_polygon, M_PER_DEG_LAT

PROFILE_KEY = "conflict_ambience"
#: Hard cap on any setting's mean rate. Tested.
MAX_RATE_PER_MIN = 10.0
#: Length of one independently seeded time bucket, in sim seconds.
BUCKET_S = 10.0
#: Scene-mask lattice spacing.
MASK_CELL_M = 500.0
#: Live generation happens at the start of a bucket from positions at that
#: moment, so the standoff is inflated by how far a fast aircraft can move in
#: one bucket.
LIVE_STANDOFF_PAD_M = 250.0 * BUCKET_S
#: Extra clearance around a protected zone's own buffer.
ZONE_MARGIN_M = 1500.0

#: Said wherever the layer is drawn. The effect is the picture; this is the claim.
AMBIENCE_NOTE = (
    "FICTIONAL AMBIENCE -- presentation-only VFX generated from a visual seed. Not simulated "
    "events: no munition, strike, target or damage is modelled or implied, nothing is hit, "
    "and nothing here reaches the simulation, LOS, detection or rewards."
)

KINDS = ("distant_flash", "smoke_column", "dust_burst", "spark_spray")


@dataclass(frozen=True)
class AmbienceSetting:
    key: str
    rate_per_min: float
    max_per_bucket: int
    max_concurrent: int
    #: relative weights over KINDS
    weights: tuple[float, float, float, float]
    #: how long smoke persists, seconds (min, max)
    smoke_s: tuple[float, float]
    #: plume top above ground, metres (min, max)
    plume_m: tuple[float, float]
    #: plume base radius, metres (min, max)
    scale_m: tuple[float, float]

    def __post_init__(self):
        if not (0.0 < self.rate_per_min <= MAX_RATE_PER_MIN):
            raise ValueError(f"{self.key}: rate {self.rate_per_min}/min outside (0, {MAX_RATE_PER_MIN}]")
        if self.max_per_bucket < 1 or self.max_concurrent < 1:
            raise ValueError(f"{self.key}: caps must be positive")


SETTINGS: dict[str, AmbienceSetting] = {
    "sparse": AmbienceSetting(
        key="sparse", rate_per_min=2.0, max_per_bucket=1, max_concurrent=5,
        weights=(0.35, 0.35, 0.2, 0.1), smoke_s=(45.0, 90.0), plume_m=(600.0, 1400.0),
        scale_m=(180.0, 320.0)),
    "sustained": AmbienceSetting(
        key="sustained", rate_per_min=6.0, max_per_bucket=3, max_concurrent=12,
        weights=(0.3, 0.35, 0.2, 0.15), smoke_s=(60.0, 140.0), plume_m=(800.0, 2000.0),
        scale_m=(220.0, 420.0)),
}
DEFAULT_SETTING = "sustained"


class AmbienceError(ValueError):
    """An ambience request that cannot be honoured."""


def get_setting(key: str) -> AmbienceSetting:
    if key not in SETTINGS:
        raise AmbienceError(f"unknown ambience setting {key!r}; known: {sorted(SETTINGS)}")
    return SETTINGS[key]


# --- the synthetic scene mask ---------------------------------------------------------


def _hash01(seed: int, i: int, j: int) -> float:
    h = hashlib.sha256(f"naigos-ambience-mask:{seed}:{i}:{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def _value_noise(seed: int, x: float, y: float) -> float:
    """Smooth [0, 1) noise on a unit lattice, bilinear with a smoothstep. Deterministic."""
    i, j = math.floor(x), math.floor(y)
    fx, fy = x - i, y - j
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash01(seed, i, j), _hash01(seed, i + 1, j)
    c, d = _hash01(seed, i, j + 1), _hash01(seed, i + 1, j + 1)
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


@dataclass(frozen=True)
class SceneMask:
    """Where effects may originate: a list of lattice-cell centres, and how it was made."""

    aoi: str
    seed: int
    cells: tuple[tuple[float, float], ...]
    cell_m: float
    digest: str

    def summary(self) -> dict:
        return {"aoi": self.aoi, "seed": self.seed, "n_cells": len(self.cells),
                "cell_m": self.cell_m, "digest": self.digest,
                "source": "synthetic value-noise inside hand-drawn ambience regions"}


def excluded(city: CityConfig, lon: float, lat: float) -> str | None:
    """Why an origin may not be here, or None. Used for the mask AND for every event."""
    a = city.aoi_def
    margin = city.ambience_edge_margin_m
    if not (a.west + margin / m_per_deg_lon(lat) <= lon <= a.east - margin / m_per_deg_lon(lat)
            and a.south + margin / M_PER_DEG_LAT <= lat <= a.north - margin / M_PER_DEG_LAT):
        return "edge_margin"
    w, s, e, n = city.urban_bounds
    if w <= lon <= e and s <= lat <= n:
        return "urban_core"
    for z in city.zones("ambience"):
        zw, zs, ze, zn = z.buffered()
        dlon, dlat = ZONE_MARGIN_M / m_per_deg_lon(lat), ZONE_MARGIN_M / M_PER_DEG_LAT
        if zw - dlon <= lon <= ze + dlon and zs - dlat <= lat <= zn + dlat:
            return f"protected_zone:{z.name}"
    if not any(point_in_polygon(lon, lat, r.polygon) for r in city.ambience_regions):
        return "outside_regions"
    return None


def build_mask(city: CityConfig, seed: int, threshold: float = 0.42,
               noise_scale_m: float = 4000.0) -> SceneMask:
    """The synthetic origin mask for ``city`` under ``seed``. Pure; cached by callers."""
    a = city.aoi_def
    lat_c = (a.south + a.north) / 2.0
    dlon, dlat = MASK_CELL_M / m_per_deg_lon(lat_c), MASK_CELL_M / M_PER_DEG_LAT
    cells = []
    nx = int((a.east - a.west) / dlon)
    ny = int((a.north - a.south) / dlat)
    for j in range(ny):
        lat = a.south + (j + 0.5) * dlat
        for i in range(nx):
            lon = a.west + (i + 0.5) * dlon
            if excluded(city, lon, lat):
                continue
            xm = (lon - a.west) * m_per_deg_lon(lat_c) / noise_scale_m
            ym = (lat - a.south) * M_PER_DEG_LAT / noise_scale_m
            if _value_noise(seed, xm, ym) < threshold:
                continue
            cells.append((round(lon, 6), round(lat, 6)))
    if not cells:
        raise AmbienceError(f"{city.aoi}: the ambience mask is empty -- widen the regions")
    digest = hashlib.sha256(repr(cells).encode()).hexdigest()[:16]
    return SceneMask(aoi=city.aoi, seed=seed, cells=tuple(cells), cell_m=MASK_CELL_M, digest=digest)


# --- the stream ----------------------------------------------------------------------------


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _bucket(city: CityConfig, setting: AmbienceSetting, mask: SceneMask, seed: int, k: int,
            positions_at, standoff_m: float) -> list[dict]:
    """The events whose start falls in bucket ``k``. Depends on nothing but its arguments."""
    rng = random.Random(f"naigos-ambience:{seed}:{setting.key}:{mask.digest}:{k}")
    n = min(_poisson(rng, setting.rate_per_min * BUCKET_S / 60.0), setting.max_per_bucket)
    out = []
    for j in range(n):
        t = round(k * BUCKET_S + rng.random() * BUCKET_S, 2)
        kind = rng.choices(KINDS, weights=setting.weights)[0]
        smoke = round(rng.uniform(*setting.smoke_s), 1)
        plume = round(rng.uniform(*setting.plume_m), 0)
        scale = round(rng.uniform(*setting.scale_m), 0)
        drift = round(rng.uniform(0.0, 360.0), 1)
        sub = rng.randrange(1 << 30)
        others = positions_at(t) if positions_at else ()
        placed = None
        for _attempt in range(4):   # a fixed number of draws: rejection stays deterministic
            clon, clat = mask.cells[rng.randrange(len(mask.cells))]
            lon = clon + (rng.random() - 0.5) * mask.cell_m / m_per_deg_lon(clat)
            lat = clat + (rng.random() - 0.5) * mask.cell_m / M_PER_DEG_LAT
            if excluded(city, lon, lat):
                continue
            if any(distance_m(lon, lat, olon, olat) < standoff_m for olon, olat in others):
                continue
            placed = (round(lon, 6), round(lat, 6))
            break
        if placed is None:
            continue
        out.append({
            "id": f"amb-{k}-{j}", "t_s": t, "kind": kind,
            "lon": placed[0], "lat": placed[1],
            "flash_s": 0.9 if kind in ("distant_flash", "spark_spray") else 0.0,
            "duration_s": smoke if kind != "spark_spray" else round(min(smoke, 12.0), 1),
            "plume_m": plume if kind in ("distant_flash", "smoke_column") else round(plume * 0.35),
            "scale_m": scale, "drift_deg": drift, "seed": sub,
            "fictional": True,
        })
    return out


def _life(e: dict) -> float:
    return max(e["duration_s"], e["flash_s"])


def lookback_buckets(setting: AmbienceSetting) -> int:
    """How many earlier buckets can still have an effect on screen."""
    return int(math.ceil(max(setting.smoke_s[1], 1.0) / BUCKET_S))


def _cap_concurrency(raw: list[dict], cap: int, t0: float, t1: float) -> list[dict]:
    """The events in [t0, t1) that start while fewer than ``cap`` others are drawn.

    Counted against the RAW stream, not against what was kept. The raw stream
    is a pure function of each bucket, so this rule gives the same answer for
    an event whatever window it is asked in -- which is what makes a live
    session, a replay export and any slice of either show the same effects.
    A greedy rule over kept events would not: dropping one event early lets a
    later one through, and the result would depend on where the window began.
    Since kept <= raw at every instant, at most ``cap`` effects are ever drawn.
    """
    order = sorted(raw, key=lambda e: (e["t_s"], e["id"]))
    kept = []
    for i, ev in enumerate(order):
        if not (t0 <= ev["t_s"] < t1):
            continue
        live = sum(1 for o in order[:i] if o["t_s"] + _life(o) > ev["t_s"])
        if live < cap:
            kept.append(ev)
    return kept


def generate(city: CityConfig, setting: AmbienceSetting, mask: SceneMask, seed: int,
             t0: float, t1: float, positions_at=None, standoff_m: float | None = None) -> list[dict]:
    """Every event starting in [t0, t1), for a replay or one live window.

    ``positions_at(t)`` returns [(lon, lat), ...] for the aircraft and threats at
    sim time ``t`` -- plain floats, read and never written.
    """
    standoff = city.ambience_standoff_m if standoff_m is None else standoff_m
    k0, k1 = int(math.floor(t0 / BUCKET_S)), int(math.ceil(t1 / BUCKET_S))
    raw = []
    for k in range(max(0, k0 - lookback_buckets(setting)), k1):
        raw += _bucket(city, setting, mask, seed, k, positions_at, standoff)
    return _cap_concurrency(raw, setting.max_concurrent, t0, t1)


@dataclass
class AmbienceConfig:
    """What was asked for, resolved: off, or a setting with a seed. Credential-free."""

    enabled: bool
    setting: str | None = None
    seed: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["profile"] = PROFILE_KEY if self.enabled else None
        d["note"] = AMBIENCE_NOTE if self.enabled else None
        if self.enabled:
            s = get_setting(self.setting)
            d.update(rate_per_min=s.rate_per_min, max_concurrent=s.max_concurrent,
                     bucket_s=BUCKET_S)
        return d


def stream_block(city: CityConfig, cfg: AmbienceConfig, duration_s: float,
                 positions_at=None) -> dict:
    """The whole stream for a recording, as exported with a static replay."""
    setting = get_setting(cfg.setting)
    mask = build_mask(city, cfg.seed)
    events = generate(city, setting, mask, cfg.seed, 0.0, duration_s, positions_at)
    return {**cfg.as_dict(), "mask": mask.summary(), "duration_s": duration_s,
            "events": events}


class LiveAmbience:
    """The same stream, generated one bucket ahead of a running simulation."""

    def __init__(self, city: CityConfig, cfg: AmbienceConfig):
        self.city, self.cfg = city, cfg
        self.setting = get_setting(cfg.setting)
        self.mask = build_mask(city, cfg.seed)
        self.next_bucket = 0
        #: raw events of recent buckets, for the concurrency rule's lookback
        self._raw: dict[int, list[dict]] = {}

    def advance(self, sim_t: float, positions: list[tuple[float, float]]) -> list[dict]:
        """Events for every bucket that has begun by ``sim_t``. ``positions`` is copied."""
        pts = [(float(a), float(b)) for a, b in positions]
        out: list[dict] = []
        look = lookback_buckets(self.setting)
        while self.next_bucket * BUCKET_S <= sim_t:
            k = self.next_bucket
            # each bucket is drawn once, from the positions at the moment it begins
            self._raw[k] = _bucket(self.city, self.setting, self.mask, self.cfg.seed, k,
                                   lambda t: pts,
                                   self.city.ambience_standoff_m + LIVE_STANDOFF_PAD_M)
            raw = [e for j in range(max(0, k - look), k + 1) for e in self._raw.get(j, [])]
            out += _cap_concurrency(raw, self.setting.max_concurrent, k * BUCKET_S,
                                    (k + 1) * BUCKET_S)
            for j in [j for j in self._raw if j < k - look]:
                del self._raw[j]
            self.next_bucket += 1
        return out
