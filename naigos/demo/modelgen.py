"""Procedural, generic glTF 2.0 models for the Cesium viewer.

    uv run python scripts/build_models.py          # rewrite naigos/demo/assets/models/*.glb
    uv run python scripts/build_models.py --check  # fail if a committed file drifted

Why generated rather than downloaded. The viewer needs four shapes -- a
fixed-wing aircraft, a ground vehicle, a stationary sensor site and an airborne
drone -- and every one of them has to be (a) redistributable under a licence
this repository can carry, (b) obviously generic, and (c) available with no
network at runtime. Third-party model sites fail (a) or (b) often enough that
checking each candidate is its own project, and a realistic model "because the
thumbnail looked good" is exactly what the guardrail forbids. So the shapes are
built here from boxes, prisms and lofts, in about the detail of a board-game
piece: recognisable silhouettes, no real platform, no insignia. They are
repository-authored and dedicated to the public domain (CC0-1.0); see
`naigos/demo/assets/models/README.md`.

The generator is deterministic -- same code, same bytes -- so
`tests/test_model_assets.py` can rebuild every file and compare it to the
committed one. That is the provenance record: the GLB in the repository is
exactly what this file produces, nothing else was pasted in.

Conventions (glTF 2.0): metres, right-handed, **+Y up, +Z forward, +X left**.
CesiumJS's default axis correction for glTF 2.0 maps +Z to the model frame's +X
and +Y to +Z, which is the frame `HeadingPitchRoll` rotates -- so no per-model
axis fix is needed, and the registry records that as zero offsets rather than
leaving it implicit.

Numpy only. Nothing here imports the simulation.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "assets" / "models"

#: Generator version, written into each file's `asset.generator`. Bump when the
#: geometry changes on purpose so a drifted file is distinguishable from a
#: regenerated one.
GENERATOR = "naigos.demo.modelgen 1"

# --- palette ------------------------------------------------------------------
# Neutral, matte, unmarked. The viewer tints aircraft per id and threats red at
# draw time (ModelGraphics.color, MIX), so the base colours only have to read
# as "painted metal" and "dark glass" under the scene light.
MATERIALS = {
    "airframe": ((0.78, 0.80, 0.83, 1.0), 0.15, 0.55),
    "canopy": ((0.10, 0.13, 0.18, 1.0), 0.30, 0.25),
    "hull": ((0.46, 0.48, 0.42, 1.0), 0.10, 0.80),
    "dark": ((0.16, 0.17, 0.17, 1.0), 0.05, 0.90),
    "sensor": ((0.84, 0.85, 0.82, 1.0), 0.20, 0.50),
    "pad": ((0.55, 0.54, 0.50, 1.0), 0.00, 0.95),
    "drone": ((0.60, 0.63, 0.66, 1.0), 0.20, 0.50),
}


# --- triangle soup ------------------------------------------------------------


@dataclass
class Part:
    """Flat-shaded triangles for one material, before they become a primitive."""

    material: str
    tris: list = field(default_factory=list)  # each (3, 3)

    def add(self, tris: np.ndarray, inside) -> None:
        """Append triangles, flipping any whose normal points at `inside`.

        `inside` is a point, or a callable giving one per triangle, known to be
        inside the solid. Winding outward by construction is fragile across
        a dozen primitives; winding outward by test is not.
        """
        for t in np.asarray(tris, dtype=np.float64):
            n = np.cross(t[1] - t[0], t[2] - t[0])
            if np.linalg.norm(n) < 1e-12:
                continue
            c = t.mean(axis=0)
            ref = inside(c) if callable(inside) else np.asarray(inside, dtype=np.float64)
            if np.dot(n, c - ref) < 0:
                t = t[[0, 2, 1]]
            self.tris.append(t)


def _quad(a, b, c, d):
    return [np.array([a, b, c]), np.array([a, c, d])]


def box(part: Part, center, size) -> None:
    cx, cy, cz = center
    hx, hy, hz = (s / 2.0 for s in size)
    v = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # index = 4*ix + 2*iy + iz
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for f in faces:
        tris += _quad(*(v[i] for i in f))
    part.add(tris, center)


def prism(part: Part, poly, axis: str, lo: float, hi: float) -> None:
    """Extrude a convex 2D polygon along `axis` from `lo` to `hi`.

    `poly` is in the plane of the other two axes, in "xyz" order with `axis`
    removed: axis="x" means poly is (y, z), axis="y" means (x, z).
    """
    idx = {"x": 0, "y": 1, "z": 2}
    k = idx[axis]
    i, j = [a for a in (0, 1, 2) if a != k]
    poly = np.asarray(poly, dtype=np.float64)

    def lift(p, h):
        v = np.zeros(3)
        v[i], v[j], v[k] = p[0], p[1], h
        return v

    bottom = [lift(p, lo) for p in poly]
    top = [lift(p, hi) for p in poly]
    tris = []
    for m in range(1, len(poly) - 1):
        tris.append(np.array([bottom[0], bottom[m], bottom[m + 1]]))
        tris.append(np.array([top[0], top[m], top[m + 1]]))
    for m in range(len(poly)):
        n = (m + 1) % len(poly)
        tris += _quad(bottom[m], bottom[n], top[n], top[m])
    centroid = lift(poly.mean(axis=0), (lo + hi) / 2.0)
    part.add(tris, centroid)


def loft(part: Part, sections, sides: int = 10, axis: str = "z") -> None:
    """A body of revolution-ish solid: elliptical sections along `axis`.

    `sections` is a list of (s, rx, ry, cx, cy): position along the axis, the
    two semi-axes, and the section centre in the other two coordinates. A
    zero-radius section closes the end to a point (a nose or a tail cone).
    """
    idx = {"x": 0, "y": 1, "z": 2}
    k = idx[axis]
    i, j = [a for a in (0, 1, 2) if a != k]
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rings = []
    for s, rx, ry, cx, cy in sections:
        ring = np.zeros((sides, 3))
        ring[:, i] = cx + rx * np.cos(ang)
        ring[:, j] = cy + ry * np.sin(ang)
        ring[:, k] = s
        rings.append(ring)

    tris = []
    for a, b in zip(rings[:-1], rings[1:]):
        for m in range(sides):
            n = (m + 1) % sides
            tris += _quad(a[m], a[n], b[n], b[m])
    for ring in (rings[0], rings[-1]):
        c = ring.mean(axis=0)
        for m in range(sides):
            tris.append(np.array([c, ring[m], ring[(m + 1) % sides]]))

    # np.interp wants ascending positions; a nose-to-tail list is descending.
    order = np.argsort([sec[0] for sec in sections])
    s_vals = np.array([sections[o][0] for o in order])
    centres = np.array([[sections[o][3], sections[o][4]] for o in order])

    def inside(p):
        s = np.clip(p[k], s_vals.min(), s_vals.max())
        c0 = np.interp(s, s_vals, centres[:, 0])
        c1 = np.interp(s, s_vals, centres[:, 1])
        v = np.zeros(3)
        v[i], v[j], v[k] = c0, c1, s
        return v

    part.add(tris, inside)


def cylinder(part: Part, center, radius: float, length: float, axis: str, sides: int = 12) -> None:
    c = list(center)
    idx = {"x": 0, "y": 1, "z": 2}
    k = idx[axis]
    i, j = [a for a in (0, 1, 2) if a != k]
    lo, hi = c[k] - length / 2.0, c[k] + length / 2.0
    loft(part, [(lo, radius, radius, c[i], c[j]), (hi, radius, radius, c[i], c[j])],
         sides=sides, axis=axis)


# --- the four models ----------------------------------------------------------


def fixed_wing() -> dict:
    """Generic single-engine jet trainer proportions: 14 m long, 10 m span.

    Low swept wing, conventional tail, bubble canopy. No stores, no markings.
    Nose at +Z.
    """
    body, glass = Part("airframe"), Part("canopy")
    loft(body, [
        (7.0, 0.00, 0.00, 0.0, 0.00),
        (5.8, 0.45, 0.42, 0.0, 0.02),
        (4.0, 0.80, 0.78, 0.0, 0.08),
        (1.0, 0.92, 0.90, 0.0, 0.10),
        (-3.0, 0.78, 0.80, 0.0, 0.18),
        (-5.8, 0.42, 0.52, 0.0, 0.35),
        (-7.0, 0.22, 0.30, 0.0, 0.45),
    ], sides=12)
    loft(glass, [
        (4.6, 0.00, 0.00, 0.0, 0.80),
        (3.8, 0.42, 0.40, 0.0, 0.85),
        (2.2, 0.50, 0.50, 0.0, 0.88),
        (0.6, 0.30, 0.28, 0.0, 0.80),
        (0.2, 0.00, 0.00, 0.0, 0.78),
    ], sides=10)
    # wings: (x, z) planform, extruded in y. +X is left.
    for side in (1.0, -1.0):
        prism(body, [(side * 0.8, 1.8), (side * 5.0, -0.6), (side * 5.0, -1.6), (side * 0.8, -2.2)],
              "y", -0.30, -0.08)
        prism(body, [(side * 0.3, -4.9), (side * 2.6, -6.0), (side * 2.6, -6.6), (side * 0.3, -6.6)],
              "y", 0.25, 0.37)
    # fin: (y, z) profile extruded in x
    prism(body, [(0.6, -4.6), (2.9, -6.2), (2.9, -7.0), (0.6, -7.0)], "x", -0.08, 0.08)
    return {"name": "fixed_wing", "nodes": [{"name": "airframe", "parts": [body, glass]}]}


def ground_vehicle() -> dict:
    """Generic six-wheeled utility chassis with a rotating sensor turret.

    8 m long. The turret is its own glTF node, `turret`, so the viewer can yaw
    it toward the aircraft the simulation says this unit is tracking -- and
    only then. Front at +Z.
    """
    hull, dark = Part("hull"), Part("dark")
    # side profile (y, z) with a sloped nose, extruded across the width
    prism(hull, [(0.9, -4.0), (0.9, 3.1), (1.4, 4.0), (2.3, 3.6), (2.3, -4.0)], "x", -1.25, 1.25)
    box(hull, (0.0, 2.55, 2.4), (2.3, 0.5, 1.4))   # cab roof block
    for z in (-2.7, 0.0, 2.7):
        for x in (-1.35, 1.35):
            cylinder(dark, (x, 0.55, z), 0.55, 0.45, "x", sides=12)
    turret, sensor = Part("hull"), Part("sensor")
    cylinder(turret, (0.0, 0.25, 0.0), 0.95, 0.5, "y", sides=14)
    box(turret, (0.0, 0.75, -0.2), (1.4, 0.55, 1.6))
    # sensor panel, tilted back, facing +Z (turret forward)
    prism(sensor, [(0.7, 0.25), (2.3, 0.65), (2.35, 0.45), (0.75, 0.05)], "x", -0.9, 0.9)
    box(sensor, (0.0, 1.3, 1.2), (0.25, 0.25, 0.9))   # short boom
    return {"name": "ground_vehicle", "nodes": [
        {"name": "hull", "parts": [hull, dark]},
        {"name": "turret", "parts": [turret, sensor], "parent": "hull",
         "translation": [0.0, 2.3, -1.0]},
    ]}


def sensor_site() -> dict:
    """Generic fixed installation: pad, equipment shelter, mast, rotating head.

    18 m across. The head is the glTF node `sensor`; it is turned toward a
    tracked aircraft exactly as the vehicle turret is, and otherwise faces the
    site's simulated heading. Front at +Z.
    """
    pad, shelter, dark = Part("pad"), Part("hull"), Part("dark")
    oct_ = [(9.0 * np.cos(a), 9.0 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    prism(pad, oct_, "y", 0.0, 0.35)
    box(shelter, (-4.2, 1.6, -3.8), (3.2, 2.5, 2.4))
    box(shelter, (4.0, 1.4, -4.2), (2.6, 2.1, 2.2))
    for x in (-5.5, -3.0, 3.0, 5.5):   # generic sealed canisters on low frames
        box(dark, (x, 1.05, 4.6), (1.0, 1.0, 3.8))
    cylinder(dark, (0.0, 3.6, 0.0), 0.35, 6.5, "y", sides=10)
    head, panel = Part("hull"), Part("sensor")
    cylinder(head, (0.0, 0.35, 0.0), 0.9, 0.7, "y", sides=12)
    prism(panel, [(0.3, 0.3), (3.4, 0.9), (3.45, 0.55), (0.4, -0.05)], "x", -2.2, 2.2)
    return {"name": "sensor_site", "nodes": [
        {"name": "site", "parts": [pad, shelter, dark]},
        {"name": "sensor", "parts": [head, panel], "parent": "site",
         "translation": [0.0, 6.85, 0.0]},
    ]}


def interceptor_drone() -> dict:
    """Generic tailless delta drone, 6 m long, 4.4 m span, twin canted fins.

    No canopy (uncrewed) and no stores. Nose at +Z.
    """
    body = Part("drone")
    loft(body, [
        (3.0, 0.00, 0.00, 0.0, 0.00),
        (2.2, 0.28, 0.24, 0.0, 0.02),
        (0.5, 0.45, 0.36, 0.0, 0.05),
        (-2.2, 0.40, 0.32, 0.0, 0.05),
        (-3.0, 0.22, 0.20, 0.0, 0.05),
    ], sides=10)
    for side in (1.0, -1.0):
        prism(body, [(side * 0.3, 1.2), (side * 2.2, -2.2), (side * 2.2, -2.7), (side * 0.3, -2.7)],
              "y", -0.08, 0.04)
        prism(body, [(0.25, -1.6), (1.15, -2.6), (1.15, -3.0), (0.25, -3.0)], "x",
              side * 0.55 - 0.04, side * 0.55 + 0.04)
    return {"name": "interceptor_drone", "nodes": [{"name": "airframe", "parts": [body]}]}


BUILDERS = {
    "fixed_wing": fixed_wing,
    "ground_vehicle": ground_vehicle,
    "sensor_site": sensor_site,
    "interceptor_drone": interceptor_drone,
}


# --- GLB writer ---------------------------------------------------------------


def _pad4(b: bytes, fill: bytes) -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def to_glb(model: dict) -> bytes:
    """Serialise a model to a binary glTF 2.0 container, deterministically.

    Flat-shaded, non-indexed triangles: POSITION and NORMAL only, one primitive
    per material per node. Coordinates are rounded to 0.1 mm before packing so
    the bytes do not depend on the last ulp of numpy's trig on a given machine.
    """
    mat_names = sorted({p.material for n in model["nodes"] for p in n["parts"]})
    materials = [
        {"name": m, "pbrMetallicRoughness": {
            "baseColorFactor": list(MATERIALS[m][0]),
            "metallicFactor": MATERIALS[m][1],
            "roughnessFactor": MATERIALS[m][2]}}
        for m in mat_names
    ]

    blob = b""
    buffer_views, accessors, meshes, nodes = [], [], [], []
    name_to_index = {n["name"]: i for i, n in enumerate(model["nodes"])}

    for node in model["nodes"]:
        by_mat: dict[str, list] = {}
        for p in node["parts"]:
            by_mat.setdefault(p.material, []).extend(p.tris)
        prims = []
        for m in sorted(by_mat):
            tris = np.round(np.asarray(by_mat[m]), 4)                 # (N, 3, 3)
            pos = tris.reshape(-1, 3).astype("<f4")
            nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = np.round(np.repeat(nrm, 3, axis=0), 5).astype("<f4")
            attrs = {}
            for sem, arr, bounds in (("POSITION", pos, True), ("NORMAL", nrm, False)):
                off = len(blob)
                data = arr.tobytes()
                blob = _pad4(blob + data, b"\x00")
                buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data),
                                     "target": 34962})
                acc = {"bufferView": len(buffer_views) - 1, "componentType": 5126,
                       "count": int(arr.shape[0]), "type": "VEC3"}
                if bounds:
                    acc["min"] = [float(v) for v in arr.min(axis=0)]
                    acc["max"] = [float(v) for v in arr.max(axis=0)]
                accessors.append(acc)
                attrs[sem] = len(accessors) - 1
            prims.append({"attributes": attrs, "material": mat_names.index(m), "mode": 4})
        meshes.append({"name": node["name"], "primitives": prims})
        gnode = {"name": node["name"], "mesh": len(meshes) - 1}
        if node.get("translation"):
            gnode["translation"] = [float(v) for v in node["translation"]]
        nodes.append(gnode)

    for node in model["nodes"]:
        if node.get("parent"):
            parent = nodes[name_to_index[node["parent"]]]
            parent.setdefault("children", []).append(name_to_index[node["name"]])
    roots = [name_to_index[n["name"]] for n in model["nodes"] if not n.get("parent")]

    gltf = {
        "asset": {"version": "2.0", "generator": GENERATOR,
                  "copyright": "Naigos contributors, CC0-1.0 (public domain dedication)"},
        "scene": 0,
        "scenes": [{"name": model["name"], "nodes": roots}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<III", 0x46546C67, 2, total)
            + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


def build(name: str) -> bytes:
    return to_glb(BUILDERS[name]())


def write_all(out_dir: Path = MODELS_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in BUILDERS:
        p = out_dir / f"{name}.glb"
        p.write_bytes(build(name))
        written[name] = p
    return written
