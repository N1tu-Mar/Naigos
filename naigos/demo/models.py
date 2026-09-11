"""The model registry: which glTF draws what, and every number that places it.

One typed table instead of magic numbers in `assets/cesium.html`. The page
receives it through `/scene` (`scene["models"]`) and reads scale, altitude
offset, axis correction, turret node and fallback from there, so a change to how
a model is drawn is a change to this file and shows up in review as one.

Three visual classes cover every threat kind the simulation can hold, because
the mapping reads the kind's *fields* rather than its label:

    airborne                 -> interceptor_drone   (holds an altitude)
    not airborne, speed > 0  -> ground_vehicle      (pinned to the DEM, moves)
    not airborne, speed == 0 -> sensor_site         (pinned to the DEM, fixed)

Theatre kinds are data (`components/model.detection.json`), and a new class
needs no code change here as long as it sets those two fields -- which every
`ThreatKindConfig` does.

What a model is allowed to claim. The anchor point is simulation state: an
aircraft's logged position, or the simulation DEM under a ground threat. The
model's extent around that point is presentation -- models are drawn with a
minimum pixel size so they stay legible from an AOI-wide camera, which makes
them larger than life at distance. `tests/test_model_assets.py` pins the
registry; `tests/test_visual_fidelity.py` pins how the page uses it.

Numpy-free and simulation-free: this module imports nothing from `naigos.env`
or `naigos.rl`, so the page contract can be checked without a JIT.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "assets" / "models"
MODELS_README = MODELS_DIR / "README.md"

#: Where the live server serves the files. The static export inlines them as
#: data URIs instead, so the artifact still fetches nothing but CesiumJS.
MODEL_ROUTE = "/models/"

#: The one licence every shipped model carries. Changing it means changing the
#: README record too; the asset test holds the two together.
MODEL_LICENCE = "CC0-1.0"


@dataclass(frozen=True)
class ModelSpec:
    """How one glTF file is drawn. Frozen: the registry is config, not state."""

    key: str
    file: str
    #: What this model stands for, in words the HUD can print.
    role: str
    #: "position": the simulated 3D position is the anchor (aircraft, drones).
    #: "ground": the simulation DEM under the platform's (x, y) is the anchor.
    anchor: str
    #: Uniform model scale. Models are authored in metres at near-real size.
    scale: float
    #: CesiumJS ModelGraphics.minimumPixelSize -- legibility at distance.
    minimum_pixel_size: int
    #: CesiumJS ModelGraphics.maximumScale -- the cap on that enlargement, so a
    #: far-off model does not grow to the size of the valley it is in.
    maximum_scale: float
    #: Metres added to the anchor altitude AFTER vertical exaggeration, so a
    #: model's base sits on the drawn surface whatever the relief multiplier is.
    altitude_offset_m: float
    #: The file's own axes. glTF 2.0 is +Z forward, +Y up; CesiumJS's default
    #: correction handles exactly that, so the offsets below are all zero for
    #: the generated set. Recorded rather than assumed.
    forward_axis: str
    up_axis: str
    heading_offset_deg: float
    pitch_offset_deg: float
    roll_offset_deg: float
    #: glTF node yawed toward a tracked aircraft, or None.
    turret_node: str | None
    #: Nodes the viewer depends on. A file without them is treated as failed.
    required_nodes: tuple[str, ...]
    #: The marker drawn only if the file fails to load.
    fallback_pixel_size: int
    fallback_color: str
    licence: str = MODEL_LICENCE
    source: str = "naigos.demo.modelgen"

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file


REGISTRY: dict[str, ModelSpec] = {
    "fixed_wing": ModelSpec(
        key="fixed_wing", file="fixed_wing.glb", role="blue aircraft (evasive, unarmed)",
        anchor="position", scale=1.0, minimum_pixel_size=44, maximum_scale=160.0,
        altitude_offset_m=0.0, forward_axis="+Z", up_axis="+Y",
        heading_offset_deg=0.0, pitch_offset_deg=0.0, roll_offset_deg=0.0,
        turret_node=None, required_nodes=("airframe",),
        fallback_pixel_size=11, fallback_color="#4da3ff",
    ),
    "ground_vehicle": ModelSpec(
        key="ground_vehicle", file="ground_vehicle.glb", role="mobile ground threat (generic)",
        anchor="ground", scale=1.0, minimum_pixel_size=34, maximum_scale=140.0,
        altitude_offset_m=0.0, forward_axis="+Z", up_axis="+Y",
        heading_offset_deg=0.0, pitch_offset_deg=0.0, roll_offset_deg=0.0,
        turret_node="turret", required_nodes=("hull", "turret"),
        fallback_pixel_size=9, fallback_color="#ff5a5a",
    ),
    "sensor_site": ModelSpec(
        key="sensor_site", file="sensor_site.glb", role="stationary sensor site (generic)",
        anchor="ground", scale=1.0, minimum_pixel_size=38, maximum_scale=90.0,
        altitude_offset_m=0.0, forward_axis="+Z", up_axis="+Y",
        heading_offset_deg=0.0, pitch_offset_deg=0.0, roll_offset_deg=0.0,
        turret_node="sensor", required_nodes=("site", "sensor"),
        fallback_pixel_size=9, fallback_color="#ff5a5a",
    ),
    "interceptor_drone": ModelSpec(
        key="interceptor_drone", file="interceptor_drone.glb",
        role="airborne interceptor/drone (generic)",
        anchor="position", scale=1.0, minimum_pixel_size=34, maximum_scale=220.0,
        altitude_offset_m=0.0, forward_axis="+Z", up_axis="+Y",
        heading_offset_deg=0.0, pitch_offset_deg=0.0, roll_offset_deg=0.0,
        turret_node=None, required_nodes=("airframe",),
        fallback_pixel_size=9, fallback_color="#ff5a5a",
    ),
}

#: The aircraft model. Blue has exactly one visual class.
AIRCRAFT_MODEL = "fixed_wing"


def threat_model_key(airborne: bool, mobile: bool) -> str:
    """The visual class for a threat, from the two fields every kind carries.

    Total by construction: any (airborne, mobile) pair maps somewhere, so there
    is no threat kind -- default, theatre-derived or future -- that can reach
    the page without a model.
    """
    if airborne:
        return "interceptor_drone"
    return "ground_vehicle" if mobile else "sensor_site"


def model_for_kind(kind) -> str:
    """`threat_model_key` for a `ThreatKindConfig`-shaped object."""
    return threat_model_key(bool(getattr(kind, "airborne", False)),
                            float(getattr(kind, "speed", 0.0)) > 0.0)


def rendered_anchor_altitude(ground_m: float, spec: ModelSpec, exaggeration: float = 1.0) -> float:
    """Where the page puts a ground-anchored model, in metres, for a relief multiplier.

    The same formula as `anchorZ()` in `assets/cesium.html`: the DEM height goes
    through the exaggeration exactly as the mesh does, and the offset is added
    afterwards so the base stays on the drawn surface instead of floating
    `offset * k` above it.
    """
    return ground_m * exaggeration + spec.altitude_offset_m


# --- file validation -------------------------------------------------------------------


GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A


def validate_glb(data: bytes, required_nodes: tuple[str, ...] = ()) -> list[str]:
    """Why a GLB cannot be drawn, or [] if it can. The page runs the same checks.

    Structural only -- header, version, JSON chunk, the nodes the viewer
    animates. It does not claim a GPU could render the file; that needs a GPU.
    """
    errs: list[str] = []
    if len(data) < 20:
        return ["file shorter than a GLB header"]
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != GLB_MAGIC:
        return ["not a binary glTF (bad magic)"]
    if version != 2:
        errs.append(f"glTF container version {version}, expected 2")
    if length != len(data):
        errs.append(f"header length {length} != file length {len(data)}")
    jlen, jtype = struct.unpack("<II", data[12:20])
    if jtype != CHUNK_JSON:
        return errs + ["first chunk is not JSON"]
    try:
        gltf = json.loads(data[20:20 + jlen])
    except ValueError as e:
        return errs + [f"JSON chunk does not parse: {e}"]
    if gltf.get("asset", {}).get("version") != "2.0":
        errs.append("asset.version is not 2.0")
    if not gltf.get("meshes"):
        errs.append("no meshes")
    names = {n.get("name") for n in gltf.get("nodes", [])}
    for node in required_nodes:
        if node not in names:
            errs.append(f"missing node {node!r}")
    return errs


def readme_records() -> dict[str, bool]:
    """Which registry files the models README documents, with the licence named."""
    text = MODELS_README.read_text() if MODELS_README.exists() else ""
    return {spec.file: (f"`{spec.file}`" in text and MODEL_LICENCE in text)
            for spec in REGISTRY.values()}


def audit() -> list[dict]:
    """Per-model status, for `--smoke-render` and the tests. No network, no GPU."""
    docs = readme_records()
    out = []
    for spec in REGISTRY.values():
        exists = spec.path.exists()
        errs = validate_glb(spec.path.read_bytes(), spec.required_nodes) if exists else ["missing file"]
        out.append({
            "key": spec.key, "file": spec.file, "exists": exists,
            "bytes": spec.path.stat().st_size if exists else 0,
            "valid": not errs, "errors": errs,
            "documented": docs.get(spec.file, False), "licence": spec.licence,
            # the page draws the marker exactly when the file is unusable
            "fallback": bool(errs),
        })
    return out


# --- what the page receives ------------------------------------------------------------


def page_registry(inline: bool = False) -> dict:
    """The registry as the page reads it from `scene["models"]`.

    `inline=False` (the live server) points each model at `MODEL_ROUTE`.
    `inline=True` (the static export) embeds the bytes as a data URI, so the
    artifact stays a single file that fetches nothing but CesiumJS. A file that
    is missing gets `uri: None`, which the page treats as a load failure -- the
    fallback path, announced, rather than a silently empty scene.
    """
    models = {}
    for spec in REGISTRY.values():
        d = asdict(spec)
        d.pop("source")
        d["required_nodes"] = list(spec.required_nodes)
        if inline:
            d["uri"] = ("data:model/gltf-binary;base64,"
                        + base64.b64encode(spec.path.read_bytes()).decode("ascii")
                        if spec.path.exists() else None)
        else:
            d["uri"] = MODEL_ROUTE + spec.file
        models[spec.key] = d
    return {"aircraft": AIRCRAFT_MODEL, "models": models}


def served_file(path: str) -> Path | None:
    """The file behind a `/models/<name>` request, or None.

    Only registry filenames resolve. Anything else -- another extension, a
    subdirectory, `..` -- is refused without touching the filesystem, so the
    route cannot be used to read outside the models directory.
    """
    if not path.startswith(MODEL_ROUTE):
        return None
    name = path[len(MODEL_ROUTE):]
    allowed = {spec.file: spec.path for spec in REGISTRY.values()}
    return allowed.get(name)
