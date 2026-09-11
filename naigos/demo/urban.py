"""The local city layer for ``--visual urban-presentation``: OSM buildings and roads.

    uv run python -m naigos.demo.urban --aoi tehran_basin

One bounded Overpass query, cached; a derived, versioned browser payload built
from it. Visual only. This module sits in ``naigos/demo`` and nothing under
``naigos/env`` or ``naigos/rl`` may import it (``tests/test_urban_layer.py``
asserts that): building geometry is never terrain, never radar cover and never
an input to LOS, detection or any simulation result. The page says so in the
HUD wherever the buildings are drawn.

What is fetched
---------------
Exactly one request, to the allowlisted Overpass host (``osm_urban_visual`` in
``naigos.research.allowlist``), for:

* ways tagged ``building`` inside a documented urban sub-box of the AOI
  (``URBAN_BOUNDS``), minus buildings tagged military/bunker, minus anything
  tagged ``military=*``, minus anything inside a ``landuse=military`` area;
* major-road centrelines (motorway, trunk, primary, secondary, tertiary and
  their links) in the same box.

Bounded three ways: the box itself, the server-side ``[timeout]`` and
``[maxsize]``, and a client-side byte cap that aborts the download. No names,
no other geography, no force dispositions, facilities or weapon data.

What is cached
--------------
Under ``data_cache/visual/urban/<aoi>/`` -- a namespace of its own, outside the
research ``manifest.json``, so the raw bytes cannot become a component
parameter by accident:

    overpass_<query sha12>.json   the raw response, byte for byte
    urban_<aoi>.json              the derived browser payload (SCHEMA)
    provenance.json               source, URL, query digest, fetch time,
                                  sha256 of both files, licence, attribution

A second run with the same query reads the raw file and makes no network call;
``--force`` refetches, ``--offline`` refuses to fetch at all.

What reaches the browser
------------------------
Only what the renderer needs: exterior rings in WGS84 (quantised to 1e-6 deg,
delta-encoded), one height per building, and road centrelines with a 3-way road
class. No tags, no OSM ids, no names. Heights follow one documented rule
(``building_height``): a valid ``height`` tag, else ``building:levels`` x
``FLOOR_HEIGHT_M``, else a deterministic ordinary-civilian fallback from the
building tag and footprint area. Malformed, open, self-intersecting,
degenerate and out-of-bounds footprints are rejected and counted; an empty
result is an error, not an empty layer.

Numpy-free and simulation-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "naigos.urban/1"
#: Bumped whenever the derivation below changes what a given raw file yields.
BUILDER_VERSION = 1

SOURCE_KEY = "osm_urban_visual"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "naigos-demo-urban/1 (visual-only city layer; bounded query; contact via repository)"

#: Server-side limits, sent in the query, and the client-side cap that backs them.
QUERY_TIMEOUT_S = 180
QUERY_MAXSIZE_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BYTES = 160 * 1024 * 1024

#: One storey, in metres, for ``building:levels`` and for the fallback. A
#: typical residential floor-to-floor height; documented, not measured.
FLOOR_HEIGHT_M = 3.2
#: A ``height`` outside this range is treated as a tagging error and ignored.
MIN_HEIGHT_M, MAX_HEIGHT_M = 2.0, 400.0
MAX_LEVELS = 120

#: Footprint sanity. Below the first is a drawing artefact; above the second is
#: a mis-tagged landuse polygon, not a building.
MIN_AREA_M2, MAX_AREA_M2 = 4.0, 250_000.0
MAX_RING_VERTICES = 400

#: Browser encoding: integer steps of this many degrees (~0.1 m), relative to
#: the box's south-west corner, and chunks of this many degrees on a side.
QUANTUM_DEG = 1e-6
CHUNK_DEG = 0.01
#: Roads are split into pieces of at most this many vertices so a long
#: motorway lands in the chunks it crosses, not all in the first one.
ROAD_PIECE_VERTICES = 16

ROAD_CLASSES = {
    "motorway": 0, "motorway_link": 0, "trunk": 0, "trunk_link": 0,
    "primary": 1, "primary_link": 1, "secondary": 1, "secondary_link": 1,
    "tertiary": 2, "tertiary_link": 2,
}

#: Fallback storeys by building tag, for footprints with neither a height nor
#: a level count. ``None`` means "scale with footprint area" (see
#: ``fallback_levels``). Ordinary civilian building classes only.
FALLBACK_LEVELS_BY_TAG: dict[str, int | None] = {
    # ancillary single-storey structures
    "garage": 1, "garages": 1, "shed": 1, "hut": 1, "carport": 1, "roof": 1, "kiosk": 1,
    "cabin": 1, "toilets": 1, "service": 1, "container": 1, "greenhouse": 1,
    # houses
    "house": 2, "detached": 2, "semidetached_house": 2, "terrace": 2, "bungalow": 1,
    # large low buildings
    "industrial": 2, "warehouse": 2, "retail": 2, "supermarket": 2,
    # everything else, including the common "yes": by footprint area
}
FALLBACK_AREA_LEVELS = (1, 6)  # clamp for the area rule


#: Building tags treated as places of worship when a city asks for them to be
#: left out (``UrbanBounds.exclude_religious``). Matched against ``building``;
#: ``amenity=place_of_worship`` is excluded alongside.
RELIGIOUS_BUILDING_TAGS = ("mosque", "church", "cathedral", "chapel", "temple", "shrine",
                           "synagogue", "religious", "monastery")


@dataclass(frozen=True)
class UrbanBounds:
    """A documented sub-box of an AOI. WGS84 degrees."""

    aoi: str
    west: float
    south: float
    east: float
    north: float
    rationale: str
    #: Buffered (west, south, east, north) boxes of the theatre's protected
    #: zones with the ``extraction`` policy. No footprint with a vertex inside
    #: one is kept, and road centrelines are cut where they enter one.
    exclusions: tuple = ()
    #: Leave places of worship out of the layer entirely -- query and derive.
    exclude_religious: bool = False
    #: The tallest plausible ``height`` tag for this city. Supertall towers are
    #: real in some theatres; the default keeps the original 400 m rule.
    max_height_m: float = 400.0

    def excluded(self, lon: float, lat: float) -> bool:
        return any(w <= lon <= e and s <= lat <= n for (w, s, e, n) in self.exclusions)

    @property
    def as_dict(self) -> dict:
        return {"west": self.west, "south": self.south, "east": self.east, "north": self.north}

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north

    @property
    def overpass_bbox(self) -> str:
        """Overpass order: south, west, north, east."""
        return f"{self.south:.4f},{self.west:.4f},{self.north:.4f},{self.east:.4f}"


#: Per-AOI city boxes. Smaller than the AOI on purpose: the whole 80 x 100 km
#: theatre is mostly Alborz slope and open basin, and a building query over it
#: would be large for no visual gain.
URBAN_BOUNDS: dict[str, UrbanBounds] = {
    "tehran_basin": UrbanBounds(
        aoi="tehran_basin",
        west=51.30, south=35.66, east=51.50, north=35.81,
        rationale=(
            "Central and northern Tehran, ~18 x 17 km, from the city centre up to the "
            "Alborz foothills: the densest civilian fabric in the AOI, with the range "
            "rising directly behind it. Entirely inside the tehran_basin AOI."
        ),
    ),
}


def _register_city_bounds() -> None:
    """Every city config's urban box (``naigos/demo/cities/*.json``), keyed by AOI.

    A theatre is added by adding its city config, never by editing the table
    above, so two theatres on two branches cannot collide here. An AOI already
    in the table keeps its entry; ``tests/test_city_presentation.py`` checks the
    two agree.
    """
    from .cities import CITIES

    for c in CITIES.values():
        if c.aoi in URBAN_BOUNDS:
            continue
        w, s, e, n = c.urban_bounds
        URBAN_BOUNDS[c.aoi] = UrbanBounds(
            aoi=c.aoi, west=w, south=s, east=e, north=n, rationale=c.urban_rationale,
            exclusions=tuple(tuple(round(v, 6) for v in z.buffered()) for z in c.zones("extraction")),
            exclude_religious=c.exclude_religious_buildings,
            max_height_m=c.max_building_height_m,
        )


_register_city_bounds()


class UrbanDataError(RuntimeError):
    """The local city layer cannot be built or loaded, with a reason to print."""


# --- the query ---------------------------------------------------------------------------


def overpass_query(b: UrbanBounds) -> str:
    """The one query this module ever sends. Deterministic, so it hashes stably."""
    bbox = b.overpass_bbox
    excluded = "military|bunker"
    worship = ""
    if b.exclude_religious:
        excluded += "|" + "|".join(RELIGIOUS_BUILDING_TAGS)
        worship = '["amenity"!="place_of_worship"]'
    return (
        f"[out:json][timeout:{QUERY_TIMEOUT_S}][maxsize:{QUERY_MAXSIZE_BYTES}];\n"
        # military land is found only in order to subtract what lies inside it
        f'(way["landuse"="military"]({bbox});relation["landuse"="military"]({bbox}););\n'
        "map_to_area->.mil;\n"
        f'way["building"]["building"!~"^({excluded})$"][!"military"]{worship}({bbox})->.b;\n'
        "way.b(area.mil)->.inmil;\n"
        "(.b; - .inmil;)->.civ;\n"
        f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"]({bbox})->.roads;\n'
        "(.civ; .roads;);\n"
        "out body geom qt;\n"
    )


def query_digest(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


# --- heights -----------------------------------------------------------------------------

_HEIGHT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(m|meter|meters|metre|metres|ft|feet|')?\s*$")


def parse_height(value, max_m: float = MAX_HEIGHT_M) -> float | None:
    """A ``height`` tag in metres, or None if it is absent, unparseable or implausible."""
    if value is None:
        return None
    m = _HEIGHT_RE.match(str(value).lower().replace(",", "."))
    if not m:
        return None
    h = float(m.group(1))
    if m.group(2) in ("ft", "feet", "'"):
        h *= 0.3048
    return h if MIN_HEIGHT_M <= h <= max_m else None


def parse_levels(value, max_m: float = MAX_HEIGHT_M) -> float | None:
    """``building:levels`` as a storey count, or None."""
    if value is None:
        return None
    try:
        n = float(str(value).strip().replace(",", "."))
    except ValueError:
        return None
    max_levels = max(MAX_LEVELS, int(max_m / FLOOR_HEIGHT_M))
    return n if 1.0 <= n <= max_levels and math.isfinite(n) else None


def fallback_levels(building_tag: str | None, area_m2: float) -> int:
    """Storeys for an untagged footprint: by tag, else by footprint area.

    The area rule is 1 + round(log2(area / 50 m^2)), clamped to 1-6 storeys:
    50 m^2 -> 1, 100 -> 2, 200 -> 3, 400 -> 4, 800 -> 5, >= 1600 -> 6. It is a
    presentation heuristic for ordinary civilian fabric, nothing more.
    """
    by_tag = FALLBACK_LEVELS_BY_TAG.get((building_tag or "").strip().lower())
    if by_tag is not None:
        return by_tag
    lo, hi = FALLBACK_AREA_LEVELS
    n = 1 + round(math.log2(max(area_m2, 25.0) / 50.0))
    return int(min(max(n, lo), hi))


def building_height(tags: dict, area_m2: float, max_m: float = MAX_HEIGHT_M) -> tuple[float, str]:
    """(height in metres, which rule produced it). Deterministic.

    Order: a valid ``height``; else ``building:levels`` x FLOOR_HEIGHT_M; else
    ``fallback_levels`` x FLOOR_HEIGHT_M. ``max_m`` is the city's plausibility
    cap (``UrbanBounds.max_height_m``).
    """
    h = parse_height(tags.get("height"), max_m)
    if h is not None:
        return round(h, 1), "height"
    lv = parse_levels(tags.get("building:levels"), max_m)
    if lv is not None:
        return round(min(lv * FLOOR_HEIGHT_M, max_m), 1), "levels"
    return round(fallback_levels(tags.get("building"), area_m2) * FLOOR_HEIGHT_M, 1), "fallback"


# --- geometry ----------------------------------------------------------------------------


def _local_xy(ring, lat0: float) -> list[tuple[float, float]]:
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_574.0
    lon0 = ring[0][0]
    return [((lon - lon0) * kx, (lat - ring[0][1]) * ky) for lon, lat in ring]


def ring_area_m2(ring) -> float:
    """Unsigned planar area of an OPEN ring of (lon, lat), in square metres."""
    pts = _local_xy(ring, ring[0][1])
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True if segments p1-p2 and p3-p4 intersect (including touching/collinear overlap)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_seg(p1, p2, p3)) or (o2 == 0 and on_seg(p1, p2, p4))
            or (o3 == 0 and on_seg(p3, p4, p1)) or (o4 == 0 and on_seg(p3, p4, p2)))


def self_intersects(ring) -> bool:
    """Whether an OPEN ring's edges cross anywhere other than at shared vertices."""
    pts = _local_xy(ring, ring[0][1])
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges share a vertex by construction
            if _segments_cross(*edges[i], *edges[j]):
                return True
    return False


def _coords(geometry) -> list[tuple[float, float]] | None:
    """Overpass `geometry` -> [(lon, lat)], or None if any vertex is malformed."""
    if not isinstance(geometry, list):
        return None
    out = []
    for p in geometry:
        try:
            lon, lat = float(p["lon"]), float(p["lat"])
        except (TypeError, KeyError, ValueError):
            return None
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return None
        out.append((lon, lat))
    return out


def clean_ring(geometry, bounds: UrbanBounds) -> tuple[list | None, str | None]:
    """A validated OPEN exterior ring, or (None, reason)."""
    pts = _coords(geometry)
    if pts is None:
        return None, "malformed"
    if len(pts) < 4 or pts[0] != pts[-1]:
        return None, "open_ring"
    ring = [pts[0]]
    for p in pts[1:-1]:
        if p != ring[-1]:
            ring.append(p)
    if len(ring) > 1 and ring[-1] == ring[0]:
        ring.pop()
    if len(ring) < 3:
        return None, "degenerate"
    if len(ring) > MAX_RING_VERTICES:
        return None, "too_complex"
    if not all(bounds.contains(lon, lat) for lon, lat in ring):
        return None, "out_of_bounds"
    if any(bounds.excluded(lon, lat) for lon, lat in ring):
        return None, "protected_zone"
    # before the area test: a bow tie's shoelace area cancels to ~0, and it
    # should be reported as what it is
    if self_intersects(ring):
        return None, "self_intersecting"
    area = ring_area_m2(ring)
    if area < MIN_AREA_M2:
        return None, "degenerate"
    if area > MAX_AREA_M2:
        return None, "implausible_area"
    return ring, None


def clip_road(geometry, bounds: UrbanBounds) -> tuple[list[list], str | None]:
    """In-bounds runs of a road centreline, split into pieces; or ([], reason)."""
    pts = _coords(geometry)
    if pts is None or len(pts) < 2:
        return [], "malformed"
    runs, cur = [], []
    for p in pts:
        if bounds.contains(*p) and not bounds.excluded(*p):
            if not cur or p != cur[-1]:
                cur.append(p)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    if not runs:
        return [], "out_of_bounds"
    pieces = []
    step = ROAD_PIECE_VERTICES - 1
    for run in runs:
        for i in range(0, len(run) - 1, step):
            piece = run[i:i + ROAD_PIECE_VERTICES]
            if len(piece) >= 2:
                pieces.append(piece)
    return pieces, None


# --- encoding ----------------------------------------------------------------------------


def encode_coords(pts, origin: tuple[float, float]) -> list[int]:
    """[(lon, lat)] -> [x0, y0, dx1, dy1, ...] in QUANTUM_DEG steps from `origin`."""
    out, px, py = [], 0, 0
    for i, (lon, lat) in enumerate(pts):
        x = round((lon - origin[0]) / QUANTUM_DEG)
        y = round((lat - origin[1]) / QUANTUM_DEG)
        if i == 0:
            out += [x, y]
        else:
            out += [x - px, y - py]
        px, py = x, y
    return out


def decode_coords(values, origin: tuple[float, float]) -> list[tuple[float, float]]:
    """Inverse of `encode_coords` (to within QUANTUM_DEG). The page's decodeCoords() is this."""
    pts, x, y = [], 0, 0
    for i in range(0, len(values), 2):
        if i == 0:
            x, y = values[0], values[1]
        else:
            x += values[i]
            y += values[i + 1]
        pts.append((origin[0] + x * QUANTUM_DEG, origin[1] + y * QUANTUM_DEG))
    return pts


def _chunk_key(lon: float, lat: float, b: UrbanBounds) -> tuple[int, int]:
    return (int((lon - b.west) // CHUNK_DEG), int((lat - b.south) // CHUNK_DEG))


# --- derivation --------------------------------------------------------------------------


@dataclass
class DeriveReport:
    buildings: int = 0
    roads: int = 0
    road_ways: int = 0
    rejected: dict = field(default_factory=dict)
    height_rule: dict = field(default_factory=lambda: {"height": 0, "levels": 0, "fallback": 0})
    ignored_elements: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def derive(raw: dict, bounds: UrbanBounds, source: dict) -> dict:
    """The browser payload from a raw Overpass response. Pure and deterministic."""
    if not isinstance(raw, dict) or not isinstance(raw.get("elements"), list):
        raise UrbanDataError("the Overpass response has no `elements` list")
    remark = str(raw.get("remark") or "")
    if "error" in remark.lower():
        # Overpass reports a timeout or an out-of-memory as HTTP 200 plus a
        # remark, with whatever it managed so far. Partial data is refused.
        raise UrbanDataError(f"Overpass returned a partial result: {remark.strip()[:200]}")

    origin = (bounds.west, bounds.south)
    rep = DeriveReport()
    chunks: dict[tuple[int, int], dict] = {}

    def chunk(key):
        if key not in chunks:
            cx = bounds.west + (key[0] + 0.5) * CHUNK_DEG
            cy = bounds.south + (key[1] + 0.5) * CHUNK_DEG
            chunks[key] = {"k": list(key), "c": [round(cx, 6), round(cy, 6)], "b": [], "r": [], "_v": 0.0}
        return chunks[key]

    # Overpass's qt order is by quadtile, not id; sort so the payload is a pure
    # function of the data, whatever order it arrived in.
    elements = sorted((e for e in raw["elements"] if isinstance(e, dict)),
                      key=lambda e: (str(e.get("type")), int(e.get("id", 0) or 0)))
    for el in elements:
        if el.get("type") != "way":
            rep.ignored_elements += 1
            continue
        tags = el.get("tags") or {}
        if "building" in tags:
            if tags.get("military") or str(tags.get("building")).lower() in ("military", "bunker"):
                rep.reject("excluded_tag")  # belt and braces: the query already excludes these
                continue
            if bounds.exclude_religious and (
                    str(tags.get("building")).lower() in RELIGIOUS_BUILDING_TAGS
                    or tags.get("amenity") == "place_of_worship"):
                rep.reject("excluded_tag")
                continue
            ring, why = clean_ring(el.get("geometry"), bounds)
            if ring is None:
                rep.reject(why)
                continue
            area = ring_area_m2(ring)
            h, rule = building_height(tags, area, bounds.max_height_m)
            rep.height_rule[rule] += 1
            cx = sum(p[0] for p in ring) / len(ring)
            cy = sum(p[1] for p in ring) / len(ring)
            ch = chunk(_chunk_key(cx, cy, bounds))
            ch["b"].append([int(round(h * 10))] + encode_coords(ring, origin))
            ch["_v"] += area * h
            rep.buildings += 1
        elif tags.get("highway") in ROAD_CLASSES:
            pieces, why = clip_road(el.get("geometry"), bounds)
            if not pieces:
                rep.reject(f"road_{why}")
                continue
            rep.road_ways += 1
            cls = ROAD_CLASSES[tags["highway"]]
            for piece in pieces:
                ch = chunk(_chunk_key(*piece[0], bounds))
                ch["r"].append([cls] + encode_coords(piece, origin))
                rep.roads += 1
        else:
            rep.ignored_elements += 1

    if rep.buildings == 0:
        raise UrbanDataError(
            f"no valid building footprints in {bounds.aoi} urban bounds {bounds.as_dict} "
            f"(rejected: {rep.rejected or 'none'}); refusing to write an empty layer")

    ordered = [chunks[k] for k in sorted(chunks)]
    # The densest chunk (by building volume) is where the city camera looks.
    dense = max(ordered, key=lambda c: c["_v"])
    for c in ordered:
        c.pop("_v")
    body = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "aoi": bounds.aoi,
        "presentation_only": True,
        "note": ("Presentation only. OpenStreetMap building footprints and road centrelines, "
                 "extruded for context; not used by terrain LOS, detection, or any simulation "
                 "result."),
        "bounds": bounds.as_dict,
        "origin": list(origin),
        "quantum_deg": QUANTUM_DEG,
        "chunk_deg": CHUNK_DEG,
        "focus": {"lon": dense["c"][0], "lat": dense["c"][1]},
        "height_rule": {
            "order": ["height tag (m)", f"building:levels x {FLOOR_HEIGHT_M} m",
                      "fallback storeys by tag, else 1+round(log2(area/50 m^2)) clamped 1-6, "
                      f"x {FLOOR_HEIGHT_M} m"],
            "floor_height_m": FLOOR_HEIGHT_M,
            "valid_height_m": [MIN_HEIGHT_M, bounds.max_height_m],
        },
        "counts": {
            "buildings": rep.buildings, "roads": rep.roads, "road_ways": rep.road_ways,
            "chunks": len(ordered), "rejected": dict(sorted(rep.rejected.items())),
            "height_rule": rep.height_rule,
        },
        "encoding": {
            "building": "[height_dm, x0, y0, dx1, dy1, ...] open exterior ring",
            "road": "[class, x0, y0, dx1, dy1, ...] class 0 motorway/trunk, 1 primary/secondary, "
                    "2 tertiary",
            "units": "integer steps of quantum_deg from origin (lon, lat); deltas after the first",
        },
        "source": source,
        "chunks": ordered,
    }
    if bounds.exclusions or bounds.exclude_religious:
        # What this city's layer deliberately leaves out, stated in the payload
        # the page draws -- by count and rule, never by listing what was dropped.
        body["exclusions"] = {
            "protected_zone_boxes": len(bounds.exclusions),
            "protected_zone_rejects": rep.rejected.get("protected_zone", 0),
            "places_of_worship_excluded": bounds.exclude_religious,
        }
    body["cache_id"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "source"},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return body


# --- cache -------------------------------------------------------------------------------


def cache_root(aoi: str) -> Path:
    """``<research cache>/visual/urban/<aoi>``. Honours NAIGOS_CACHE_DIR like the rest."""
    from ..research import cache as research_cache

    return research_cache.cache_dir() / "visual" / "urban" / aoi


def payload_path(aoi: str) -> Path:
    return cache_root(aoi) / f"urban_{aoi}.json"


def provenance_path(aoi: str) -> Path:
    return cache_root(aoi) / "provenance.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fetch_raw(query: str, timeout_s: int = QUERY_TIMEOUT_S + 30,
              max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    """POST the query to the allowlisted Overpass host, streaming under a byte cap."""
    from ..research.allowlist import check_url

    check_url(OVERPASS_URL, SOURCE_KEY)
    import requests

    resp = requests.post(OVERPASS_URL, data={"data": query}, stream=True,
                         timeout=(20, timeout_s), headers={"User-Agent": USER_AGENT})
    try:
        if resp.status_code != 200:
            raise UrbanDataError(f"Overpass answered HTTP {resp.status_code}: "
                                 f"{resp.text[:200] if resp.text else ''}")
        buf = bytearray()
        for piece in resp.iter_content(chunk_size=1 << 16):
            buf += piece
            if len(buf) > max_bytes:
                raise UrbanDataError(
                    f"Overpass response exceeded the {max_bytes // (1024 * 1024)} MB cap; "
                    "refusing to cache a truncated file")
        return bytes(buf)
    finally:
        resp.close()


def build(aoi: str, *, force: bool = False, offline: bool = False, fetcher=fetch_raw,
          log=print) -> dict:
    """Fetch (once) and derive the city layer for `aoi`. Returns the provenance record."""
    from ..research.allowlist import ALLOWLIST

    if aoi not in URBAN_BOUNDS:
        raise UrbanDataError(
            f"no documented urban sub-bound for AOI {aoi!r}; known: {sorted(URBAN_BOUNDS)}")
    bounds = URBAN_BOUNDS[aoi]
    src = ALLOWLIST[SOURCE_KEY]
    query = overpass_query(bounds)
    qsha = query_digest(query)
    root = cache_root(aoi)
    raw_path = root / f"overpass_{qsha[:12]}.json"
    prov_file = provenance_path(aoi)
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}

    raw_rec = prov.get("raw") if prov.get("raw", {}).get("query_sha256") == qsha else None
    if raw_path.exists() and raw_rec and not force:
        data = raw_path.read_bytes()
        if _sha256(data) != raw_rec.get("sha256"):
            raise UrbanDataError(f"{raw_path} does not match its recorded sha256; rerun with --force")
        log(f"raw: cached {raw_path} ({len(data) / 1e6:.1f} MB) -- no network call")
    else:
        if offline:
            raise UrbanDataError(f"--offline and no cached raw response at {raw_path}")
        log(f"raw: querying {OVERPASS_URL} for {aoi} urban bounds {bounds.as_dict} "
            f"(timeout {QUERY_TIMEOUT_S} s, cap {MAX_RESPONSE_BYTES // (1024 * 1024)} MB)")
        data = fetcher(query)
        root.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(data)
        raw_rec = {
            "path": raw_path.name, "url": OVERPASS_URL, "method": "POST data=<query>",
            "query": query, "query_sha256": qsha, "bytes": len(data), "sha256": _sha256(data),
            "fetched_at": _now(),
        }
        log(f"raw: cached {len(data) / 1e6:.1f} MB -> {raw_path}")

    try:
        raw = json.loads(data)
    except ValueError as e:
        raise UrbanDataError(f"cached Overpass response is not JSON: {e}") from e
    source = {
        "key": SOURCE_KEY, "name": src.name, "licence": src.license,
        "licence_url": src.license_url, "attribution": "(c) OpenStreetMap contributors (ODbL)",
        "url": OVERPASS_URL, "query_sha256": qsha, "raw_sha256": raw_rec["sha256"],
        "fetched_at": raw_rec["fetched_at"], "osm_timestamp": (raw.get("osm3s") or {}).get(
            "timestamp_osm_base"),
    }
    payload = derive(raw, bounds, source)
    body = json.dumps(payload, separators=(",", ":")).encode()
    out = payload_path(aoi)
    out.write_bytes(body)
    prov = {
        "schema": SCHEMA, "aoi": aoi, "bounds": bounds.as_dict, "rationale": bounds.rationale,
        "source_key": SOURCE_KEY, "licence": src.license, "licence_url": src.license_url,
        "citation": src.citation, "role": src.role,
        "raw": raw_rec,
        "derived": {"path": out.name, "bytes": len(body), "sha256": _sha256(body),
                    "cache_id": payload["cache_id"], "builder_version": BUILDER_VERSION,
                    "built_at": _now(), "counts": payload["counts"]},
    }
    prov_file.write_text(json.dumps(prov, indent=2) + "\n")
    return prov


# --- loading, for the server and the static export --------------------------------------


@dataclass
class UrbanStatus:
    """Whether a usable local city layer exists for an AOI, and why not if not."""

    aoi: str
    state: str                     # "available" | "unavailable" | "invalid"
    reason: str | None = None
    payload: dict | None = None
    bounds: dict | None = None
    focus: dict | None = None

    @property
    def available(self) -> bool:
        return self.state == "available"

    def summary(self) -> dict:
        """The credential-free, chunk-free block `/scene` carries as `scene["urban"]`."""
        p = self.payload or {}
        return {
            "state": self.state,
            "reason": self.reason,
            "aoi": self.aoi,
            "schema": p.get("schema"),
            "cache_id": p.get("cache_id"),
            "counts": p.get("counts"),
            "bounds": self.bounds,
            "focus": self.focus,
            "height_rule": p.get("height_rule"),
            "source": p.get("source"),
            "exclusions": p.get("exclusions"),
            "attribution": "Buildings and roads (c) OpenStreetMap contributors, ODbL",
            "presentation_only": True,
            "build_command": f"uv run python -m naigos.demo.urban --aoi {self.aoi}",
        }


def load(aoi: str) -> UrbanStatus:
    """Read the derived payload for `aoi`. Never touches the network."""
    b = URBAN_BOUNDS.get(aoi)
    if b is None:
        return UrbanStatus(aoi, "unavailable", reason=f"no documented urban sub-bound for {aoi!r}")
    centre = {"lon": round((b.west + b.east) / 2, 6), "lat": round((b.south + b.north) / 2, 6)}
    path = payload_path(aoi)
    if not path.exists():
        return UrbanStatus(aoi, "unavailable", bounds=b.as_dict, focus=centre, reason=(
            f"urban data unavailable: no local cache at {path}. Build it once with "
            f"`uv run python -m naigos.demo.urban --aoi {aoi}`"))
    try:
        payload = json.loads(path.read_text())
    except ValueError as e:
        return UrbanStatus(aoi, "invalid", bounds=b.as_dict, focus=centre,
                           reason=f"urban cache {path} is not JSON ({e})")
    problems = validate_payload(payload, aoi)
    if problems:
        return UrbanStatus(aoi, "invalid", bounds=b.as_dict, focus=centre,
                           reason="urban cache rejected: " + "; ".join(problems))
    return UrbanStatus(aoi, "available", payload=payload, bounds=payload["bounds"],
                       focus=payload.get("focus") or centre)


def validate_payload(p: dict, aoi: str) -> list[str]:
    """Why a payload cannot be drawn for `aoi`, or []."""
    errs = []
    if p.get("schema") != SCHEMA:
        errs.append(f"schema {p.get('schema')!r}, expected {SCHEMA!r}")
    if p.get("aoi") != aoi:
        errs.append(f"built for {p.get('aoi')!r}, not {aoi!r}")
    b = URBAN_BOUNDS.get(aoi)
    pb = p.get("bounds") or {}
    if b and pb != b.as_dict:
        errs.append("bounds differ from the documented urban sub-bound; rebuild")
    if not p.get("chunks") or not (p.get("counts") or {}).get("buildings"):
        errs.append("no buildings")
    if not p.get("presentation_only"):
        errs.append("not marked presentation_only")
    return errs


def page_payload(status: UrbanStatus) -> dict | None:
    """What `/urban` serves and the static export embeds: the payload, as built."""
    return status.payload if status.available else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="naigos.demo.urban", description=__doc__.split("\n\n")[0])
    ap.add_argument("--aoi", default="tehran_basin", choices=sorted(URBAN_BOUNDS))
    ap.add_argument("--force", action="store_true", help="refetch even if a raw response is cached")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    a = ap.parse_args(argv)
    try:
        prov = build(a.aoi, force=a.force, offline=a.offline)
    except UrbanDataError as e:
        print(f"naigos.demo.urban: {e}", file=sys.stderr)
        return 1
    d = prov["derived"]
    c = d["counts"]
    print(f"derived: {payload_path(a.aoi)}  ({d['bytes'] / 1e6:.1f} MB, cache id {d['cache_id']})")
    print(f"buildings {c['buildings']:,}  road pieces {c['roads']:,} ({c['road_ways']:,} ways)  "
          f"chunks {c['chunks']}")
    print(f"heights: {c['height_rule']['height']:,} from height tags, "
          f"{c['height_rule']['levels']:,} from levels, {c['height_rule']['fallback']:,} fallback")
    if c["rejected"]:
        print("rejected: " + ", ".join(f"{k} {v:,}" for k, v in c["rejected"].items()))
    print("licence: ODbL 1.0 -- (c) OpenStreetMap contributors. Presentation only; the "
          "simulation never reads this layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA", "URBAN_BOUNDS", "UrbanBounds", "UrbanDataError", "UrbanStatus",
    "building_height", "build", "derive", "load", "overpass_query",
]
