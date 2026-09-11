"""Emit cited component specs -- the research agent's actual deliverable.

Nothing enters the env directly from a fetch. Every dataset is distilled into one
``components/<id>.json`` carrying the schema below, and the env reads *those*. That indirection
is the point: a component file states what the data is, what decision it settles, which cached
bytes back it, and under what license -- so the design can be reasoned over as a graph, and no
number in the simulation is untraceable.

Schema (after Nomos ``components/README.md``):
  id, role, inputs, outputs, decision, rationale, parameters, evidence,
  license, sources[], cached_artifacts[], invariants[], generated_at
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .allowlist import ALLOWLIST
from .cache import REPO_ROOT, Artifact

#: The local default, unchanged: ``components/`` next to the package.
DEFAULT_COMPONENTS_DIR = REPO_ROOT / "components"
# Read at call time by every function below, for the same reason as
# ``cache.CACHE_DIR``: a snapshot worker points it at a snapshot-specific root.
COMPONENTS_DIR = Path(os.environ.get("NAIGOS_COMPONENTS_DIR") or DEFAULT_COMPONENTS_DIR)


def components_dir() -> Path:
    """The component root in effect right now."""
    return COMPONENTS_DIR


def set_components_dir(path: str | os.PathLike) -> Path:
    """Point component reads and writes at ``path``. Returns the previous root."""
    global COMPONENTS_DIR
    previous = COMPONENTS_DIR
    COMPONENTS_DIR = Path(path)
    return previous

REQUIRED_FIELDS = (
    "id", "role", "inputs", "outputs", "decision", "rationale",
    "license", "sources", "generated_at",
)


class UncitedComponent(ValueError):
    """Raised when a component would be written without a source. Nothing enters the env uncited."""


def _source_block(source_keys: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for key in source_keys:
        src = ALLOWLIST[key]
        out.append({
            "key": src.key,
            "name": src.name,
            "license": src.license,
            "license_url": src.license_url,
            "citation": src.citation,
            "docs": list(src.docs),
            "attribution_required": src.attribution_required,
            "notes": src.notes,
        })
    return out


def _artifact_block(artifacts: Iterable[Artifact]) -> list[dict[str, Any]]:
    return [
        {
            "key": a.key, "path": a.path, "bytes": a.bytes,
            "sha256": a.sha256, "fetched_at": a.fetched_at, "url": a.url,
        }
        for a in artifacts
    ]


def write_component(
    component_id: str,
    *,
    role: str,
    inputs: list[str],
    outputs: list[str],
    decision: str,
    rationale: str,
    source_keys: list[str],
    parameters: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    artifacts: list[Artifact] | None = None,
    invariants: list[str] | None = None,
    caveats: list[str] | None = None,
) -> Path:
    """Write one cited component spec and return its path."""
    if not source_keys:
        raise UncitedComponent(
            f"component {component_id!r} has no sources; nothing enters the env without a citation"
        )
    unknown = [k for k in source_keys if k not in ALLOWLIST]
    if unknown:
        raise UncitedComponent(f"component {component_id!r} cites non-allowlisted sources: {unknown}")

    doc = {
        "id": component_id,
        "role": role,
        "inputs": inputs,
        "outputs": outputs,
        "decision": decision,
        "rationale": rationale,
        "parameters": parameters or {},
        "evidence": evidence or {},
        "invariants": invariants or [],
        "caveats": caveats or [],
        "license": "; ".join(sorted({ALLOWLIST[k].license for k in source_keys})),
        "sources": _source_block(source_keys),
        "cached_artifacts": _artifact_block(artifacts or []),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    missing = [f for f in REQUIRED_FIELDS if not doc.get(f)]
    if missing:
        raise UncitedComponent(f"component {component_id!r} missing required fields: {missing}")

    COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = COMPONENTS_DIR / f"{component_id}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def component_dir(aoi: str | None = None) -> Path:
    """Where to read component specs from.

    ``components/*.json`` always holds the most recent research run. Each run
    also snapshots itself to ``components/aoi/<name>/``, because four of the six
    specs (env.aoi, data.terrain_dem, data.airfields, data.atmosphere) are
    AOI-scoped and a run for a second theatre would otherwise overwrite the
    first -- silently making a checkpoint trained on one theatre unreproducible
    while every file still looks valid.
    """
    if aoi:
        scoped = COMPONENTS_DIR / "aoi" / aoi
        if scoped.is_dir():
            return scoped
        raise FileNotFoundError(
            f"no component snapshot for AOI {aoi!r} at {scoped}. "
            f"Run `naigos-research --aoi {aoi}` to build it. "
            f"Available: {sorted(p.name for p in (COMPONENTS_DIR / 'aoi').glob('*')) or 'none'}"
        )
    return COMPONENTS_DIR


def load_component(component_id: str, aoi: str | None = None) -> dict[str, Any]:
    """Read a component spec. This is how ``naigos/data`` and the env consume research output."""
    path = component_dir(aoi) / f"{component_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `naigos-research` to build the data layer first."
        )
    return json.loads(path.read_text())


def snapshot_aoi(aoi: str) -> Path:
    """Copy the current component set into ``components/aoi/<aoi>/``."""
    import shutil

    dest = COMPONENTS_DIR / "aoi" / aoi
    dest.mkdir(parents=True, exist_ok=True)
    for src in COMPONENTS_DIR.glob("*.json"):
        shutil.copy2(src, dest / src.name)
    return dest
