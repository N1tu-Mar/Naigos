"""Immutable, provenance-checked input snapshots.

A snapshot is one research-pipeline output, frozen: the raw cache with its
sha256 manifest, the cited component specs, the AOI-scoped component copy the
env loads, and the ``DATA.md`` rendered from them. It is the only thing a
candidate is trained or evaluated on, and its ID is recorded in everything
downstream of it.

    /pipeline/snapshots/<snapshot-id>/
        cache/                 raw bytes + cache/manifest.json (sha256 per artifact)
        components/            *.json and components/aoi/<aoi>/*.json
        DATA.md                provenance log for this snapshot
        snapshot.json          the record; written last, inside staging

Build discipline:

 1. **Seed.** Copy the previous completed snapshot's cache and components into a
    fresh staging directory (a copy, so the parent stays immutable). Artifacts
    from sources listed in ``snapshot.refresh_sources`` are dropped from the
    seeded manifest, so the research layer re-fetches exactly those and reuses
    everything else offline -- the same idempotent-cache contract it has
    locally, pointed at a different root.
 2. **Build.** Run ``naigos.research.run.build`` in a subprocess with the cache
    and component roots pointed at staging and the egress guard installed.
 3. **Validate.** ``validate_tree`` below: every manifest entry's bytes exist
    and hash to what the manifest says, no file in the cache is uncited, every
    component cites allowlisted sources and names cached bytes that exist with
    the same hash, the AOI in the components is the AOI that was asked for, and
    the scope guardrail is present where it must be.
 4. **Publish.** Write ``snapshot.json`` (write-once, carrying a content hash
    over every file) into staging, then rename staging into place. The rename
    fails if the destination exists, so a snapshot directory, once it exists,
    is complete and is never written again.

``verify_completed`` is what every consumer calls before trusting a snapshot:
it re-hashes the whole tree against the record, so a byte that changed after
publication is caught at the point of use, not in a later audit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Callable

from ..research.allowlist import ALLOWLIST, GUARDRAIL, SourceNotAllowed, check_url
from ..research.aoi import get_aoi
from ..research.spec import REQUIRED_FIELDS
from ..rl import runmeta
from . import layout

SNAPSHOT_SCHEMA = 1
CACHE_DIRNAME = "cache"
COMPONENTS_DIRNAME = "components"
DATA_DOC = "DATA.md"

#: Everything ``naigos.data.theatre.load_theatre`` reads, plus the research
#: agent's own contract. A snapshot missing any one of them cannot be trained on.
REQUIRED_COMPONENTS = (
    "env.aoi", "data.terrain_dem", "data.airfields", "data.atmosphere",
    "data.flight_envelope", "model.detection", "research.agent",
)
#: Components whose caveats must carry the scope guardrail verbatim.
GUARDRAIL_COMPONENTS = ("model.detection", "research.agent")
BLUE_EVASIVE_INVARIANT = "Blue never acts on a threat"


#: GDAL (under rasterio/rioxarray/py3dep) does its own HTTP through libcurl,
#: in C, below the Python-level egress guard. py3dep 0.19 fetches 30 m 3DEP
#: through GDAL from prd-tnm.s3.amazonaws.com, which is not an allowlisted
#: host. So in the guarded child GDAL's HTTP goes to a closed local port: a
#: DEM re-fetch fails loudly instead of leaving the allowlist unnoticed. The
#: default refresh (open_meteo) never needs it; DEM bytes come from the seed.
NATIVE_NETWORK_OFF = {"GDAL_HTTP_PROXY": "127.0.0.1:9", "GDAL_HTTP_TIMEOUT": "5",
                      "GDAL_HTTP_MAX_RETRY": "0"}


class SnapshotError(RuntimeError):
    """A snapshot that must not be published or used."""


# --- validation ------------------------------------------------------------------


def _safe_rel(rel: object) -> str | None:
    if not isinstance(rel, str) or not rel:
        return None
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        return None
    return p.as_posix()


def validate_tree(root: str | os.PathLike, aoi_name: str) -> list[str]:
    """Every problem with a built snapshot tree. Empty means publishable."""
    root = Path(root)
    problems: list[str] = []
    cache_dir, comp_dir = root / CACHE_DIRNAME, root / COMPONENTS_DIRNAME
    try:
        aoi = get_aoi(aoi_name)
    except KeyError as e:
        return [f"unknown AOI {aoi_name!r}: {e}"]

    # -- the cache: every entry's bytes present and matching; nothing uncited --
    manifest: dict = {}
    mpath = cache_dir / "manifest.json"
    if not mpath.exists():
        problems.append("cache/manifest.json is missing: no provenance ledger")
    else:
        try:
            manifest = json.loads(mpath.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("not an object")
        except (ValueError, OSError) as e:
            problems.append(f"cache/manifest.json does not parse: {e}")
            manifest = {}
    cited_paths: set[str] = set()
    for key, entry in sorted(manifest.items()):
        if not isinstance(entry, dict):
            problems.append(f"manifest entry {key!r} is not an object")
            continue
        rel = _safe_rel(entry.get("path"))
        if rel is None:
            problems.append(f"manifest entry {key!r} has an unsafe path {entry.get('path')!r}")
            continue
        cited_paths.add(rel)
        src = ALLOWLIST.get(entry.get("source_key"))
        if src is None:
            problems.append(f"manifest entry {key!r} cites non-allowlisted source {entry.get('source_key')!r}")
        elif src.hosts:
            try:
                check_url(str(entry.get("url") or ""), src.key)
            except SourceNotAllowed as e:
                problems.append(f"manifest entry {key!r}: {e}")
        f = cache_dir / rel
        if not f.is_file():
            problems.append(f"manifest entry {key!r}: bytes missing at cache/{rel}")
            continue
        size = f.stat().st_size
        if entry.get("bytes") != size:
            problems.append(f"manifest entry {key!r}: size {size} != recorded {entry.get('bytes')}")
        actual = layout.sha256_file(f)
        if actual != entry.get("sha256"):
            problems.append(f"manifest entry {key!r}: sha256 mismatch at cache/{rel}")
    if cache_dir.is_dir():
        for rel in layout.tree_files(cache_dir):
            if rel != "manifest.json" and rel not in cited_paths:
                problems.append(f"cache/{rel} is not in the manifest: uncited bytes")

    # -- components: cited, allowlisted, hash-linked to the cache --
    docs: dict[str, dict] = {}
    for cid in REQUIRED_COMPONENTS:
        path = comp_dir / f"{cid}.json"
        if not path.exists():
            problems.append(f"required component {cid} is missing")
            continue
        try:
            docs[cid] = json.loads(path.read_text())
        except (ValueError, OSError) as e:
            problems.append(f"component {cid} does not parse: {e}")
    for path in sorted(comp_dir.glob("*.json")) if comp_dir.is_dir() else []:
        cid = path.stem
        if cid not in docs:
            try:
                docs[cid] = json.loads(path.read_text())
            except (ValueError, OSError) as e:
                problems.append(f"component {cid} does not parse: {e}")
    for cid, doc in sorted(docs.items()):
        missing = [f for f in REQUIRED_FIELDS if not doc.get(f)]
        if missing:
            problems.append(f"component {cid} missing required fields {missing}")
        keys = [s.get("key") for s in doc.get("sources") or [] if isinstance(s, dict)]
        if not keys:
            problems.append(f"component {cid} is uncited: no sources")
        bad = sorted(k for k in keys if k not in ALLOWLIST)
        if bad:
            problems.append(f"component {cid} cites non-allowlisted sources {bad}")
        for art in doc.get("cached_artifacts") or []:
            entry = manifest.get(art.get("key"))
            if entry is None:
                problems.append(f"component {cid} names artifact {art.get('key')!r} absent from the manifest")
            elif entry.get("sha256") != art.get("sha256") or entry.get("path") != art.get("path"):
                problems.append(f"component {cid}: artifact {art.get('key')!r} hash/path differs from the manifest")

    # -- AOI consistency: the theatre in the files is the theatre asked for --
    env_aoi = (docs.get("env.aoi") or {}).get("parameters") or {}
    if docs.get("env.aoi") is not None:
        if env_aoi.get("name") != aoi.name:
            problems.append(f"env.aoi names AOI {env_aoi.get('name')!r}, snapshot is for {aoi.name!r}")
        if env_aoi.get("fingerprint") != aoi.fingerprint:
            problems.append(f"env.aoi fingerprint {env_aoi.get('fingerprint')!r} != {aoi.fingerprint!r}")
        if [float(x) for x in env_aoi.get("bbox_wgs84") or []] != [float(x) for x in aoi.bbox]:
            problems.append("env.aoi bbox differs from the AOI definition")
    for cid in ("data.terrain_dem", "env.aoi"):
        for art in (docs.get(cid) or {}).get("cached_artifacts") or []:
            if aoi.fingerprint not in str(art.get("key")) or f"/{aoi.name}/" not in str(art.get("key")):
                problems.append(f"{cid} artifact {art.get('key')!r} is not scoped to {aoi.name}/{aoi.fingerprint}")
    scoped = comp_dir / "aoi" / aoi.name
    if not scoped.is_dir():
        problems.append(f"components/aoi/{aoi.name}/ is missing: the env loads the AOI-scoped copy")
    else:
        for cid in REQUIRED_COMPONENTS:
            a, b = comp_dir / f"{cid}.json", scoped / f"{cid}.json"
            if a.exists() and (not b.exists() or layout.sha256_file(a) != layout.sha256_file(b)):
                problems.append(f"components/aoi/{aoi.name}/{cid}.json differs from components/{cid}.json")

    # -- scope: the guardrail and blue's evasive-only invariant are present --
    guard = GUARDRAIL.strip()
    for cid in GUARDRAIL_COMPONENTS:
        if cid in docs and guard not in (docs[cid].get("caveats") or []):
            problems.append(f"component {cid} does not carry the scope GUARDRAIL")
    inv = (docs.get("model.detection") or {}).get("invariants") or []
    if "model.detection" in docs and not any(str(i).startswith(BLUE_EVASIVE_INVARIANT) for i in inv):
        problems.append("model.detection lost the invariant that blue never acts on a threat")
    return problems


def content_digest(root: str | os.PathLike) -> dict:
    """Hash of every data file in a snapshot, excluding the record itself."""
    root = Path(root)
    files = [f for f in layout.tree_files(root) if f != layout.SNAPSHOT_RECORD]
    return layout.tree_digest(root, files)


def verify_completed(lay: layout.Layout, sid: str, *, aoi_name: str | None = None) -> list[str]:
    """Problems with a published snapshot. Re-hashes every file against its record."""
    try:
        d = lay.snapshot_dir(sid)
    except layout.LayoutError as e:
        return [str(e)]
    try:
        rec = layout.read_json(d / layout.SNAPSHOT_RECORD)
    except (ValueError, OSError) as e:
        return [f"snapshot.json does not parse: {e}"]
    if rec is None:
        return [f"{sid}: no snapshot.json -- not a completed snapshot"]
    problems = []
    if rec.get("snapshot_id") != sid:
        problems.append(f"snapshot.json names {rec.get('snapshot_id')!r}, directory is {sid!r}")
    if rec.get("status") != "completed":
        problems.append(f"snapshot.json status is {rec.get('status')!r}")
    aoi = aoi_name or ((rec.get("aoi") or {}).get("name"))
    if aoi_name and (rec.get("aoi") or {}).get("name") != aoi_name:
        problems.append(f"snapshot is for AOI {(rec.get('aoi') or {}).get('name')!r}, expected {aoi_name!r}")
    now = content_digest(d)
    recorded = (rec.get("content") or {})
    if now["sha256"] != recorded.get("sha256"):
        changed = sorted(set(now["files"].items()) ^ set((recorded.get("files") or {}).items()))
        names = sorted({f for f, _ in changed})[:10]
        problems.append(f"content hash differs from snapshot.json: changed/missing/extra {names}")
    problems.extend(validate_tree(d, aoi or ""))
    return problems


def stage_local_copy(lay: layout.Layout, sid: str, dest: str | os.PathLike) -> Path:
    """Copy a published snapshot to container-local disk and prove the copy is exact.

    Training and evaluation read this copy, never the Volume directory: nothing a
    trainer does can then alter a published snapshot, and the copy is re-hashed
    against ``snapshot.json`` so a torn or stale Volume read is caught here.
    """
    problems = verify_completed(lay, sid)
    if problems:
        raise SnapshotError(f"snapshot {sid} does not verify: " + "; ".join(problems[:10]))
    rec = layout.read_json(lay.snapshot_record(sid))
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(lay.snapshot_dir(sid), dest, copy_function=shutil.copy2)
    if content_digest(dest)["sha256"] != (rec.get("content") or {}).get("sha256"):
        raise SnapshotError(f"local copy of {sid} does not hash to its record")
    return dest


def latest_completed(lay: layout.Layout, *, aoi_name: str) -> str | None:
    """The newest snapshot that verifies, for this AOI. Newest by ID (window-sortable)."""
    for sid in sorted(lay.snapshot_ids(), reverse=True):
        rec = layout.read_json(lay.snapshot_record(sid)) if lay.snapshot_record(sid).exists() else None
        if not rec or (rec.get("aoi") or {}).get("name") != aoi_name:
            continue
        if not verify_completed(lay, sid, aoi_name=aoi_name):
            return sid
    return None


# --- building -------------------------------------------------------------------


def seed_staging(staging: Path, parent: Path | None, refresh_sources: list[str]) -> list[str]:
    """Copy a parent snapshot's data into staging and drop the artifacts to refresh.

    Returns the manifest keys that were dropped. Copies, never links: a hard
    link would make the parent's bytes writable through the child.
    """
    cache_dir, comp_dir = staging / CACHE_DIRNAME, staging / COMPONENTS_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    if parent is None:
        return []
    shutil.copytree(parent / CACHE_DIRNAME, cache_dir, dirs_exist_ok=True, copy_function=shutil.copy2)
    shutil.copytree(parent / COMPONENTS_DIRNAME, comp_dir, dirs_exist_ok=True, copy_function=shutil.copy2)
    mpath = cache_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    dropped = []
    for key, entry in list(manifest.items()):
        if entry.get("source_key") in refresh_sources:
            rel = _safe_rel(entry.get("path"))
            if rel and (cache_dir / rel).exists():
                (cache_dir / rel).unlink()
            del manifest[key]
            dropped.append(key)
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sorted(dropped)


def run_research_subprocess(*, staging: Path, aoi: str, skip_flights: bool, flight_snapshots: int,
                            timeout_s: int, guard: bool = True,
                            python: str = sys.executable) -> dict:
    """Run the research build against staging roots in a guarded child process."""
    result = staging / ".build_result.json"
    cmd = [python, "-m", "naigos.pipeline.snapshot_build",
           "--staging", str(staging), "--aoi", aoi, "--flight-snapshots", str(flight_snapshots),
           "--result", str(result)]
    if skip_flights:
        cmd.append("--skip-flights")
    if not guard:
        cmd.append("--no-egress-guard")
    env = {k: v for k, v in os.environ.items()
           if k not in ("NAIGOS_CACHE_DIR", "NAIGOS_COMPONENTS_DIR")}
    if guard:
        env.update(NATIVE_NETWORK_OFF)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    tail = runmeta.redact((proc.stderr or "")[-4000:])
    if proc.returncode != 0:
        raise SnapshotError(f"research build exited {proc.returncode}: {tail}")
    try:
        payload = json.loads(result.read_text())
    finally:
        result.unlink(missing_ok=True)
    payload["log_tail"] = tail
    return payload


def build_record(*, sid: str, staging: Path, aoi_name: str, window: str, idempotency_key: str,
                 config_digest: str, stage_digest: str, code: dict, parent: str | None,
                 dropped: list[str], refresh_sources: list[str], started_utc: str,
                 research: dict, origin: str = "scheduled") -> dict:
    aoi = get_aoi(aoi_name)
    manifest = json.loads((staging / CACHE_DIRNAME / "manifest.json").read_text())
    comp_dir = staging / COMPONENTS_DIRNAME
    return {
        "schema": SNAPSHOT_SCHEMA,
        "snapshot_id": sid,
        "status": "completed",
        "origin": origin,
        "created_utc": started_utc,
        "finished_utc": layout.utc_stamp(),
        "window": window,
        "idempotency_key": idempotency_key,
        "config_digest": config_digest,
        "stage_digest": stage_digest,
        "code": code,
        "parents": {"snapshot_id": parent},
        "aoi": {"name": aoi.name, "fingerprint": aoi.fingerprint, "bbox": list(aoi.bbox)},
        "refresh": {"sources": list(refresh_sources), "dropped_artifacts": dropped},
        "artifacts": {k: {"sha256": v.get("sha256"), "source_key": v.get("source_key"),
                          "fetched_at": v.get("fetched_at")} for k, v in sorted(manifest.items())},
        "components": {p.stem: layout.sha256_file(p) for p in sorted(comp_dir.glob("*.json"))},
        "guardrail_sha256": layout.digest(GUARDRAIL.strip()),
        "allowlist_hosts": sorted({h for s in ALLOWLIST.values() for h in s.hosts}),
        "research": {k: v for k, v in research.items() if k in ("components", "artifacts", "egress_guard")},
        "content": content_digest(staging),
    }


def publish(lay: layout.Layout, sid: str, staging: Path, record: dict, *, aoi_name: str) -> Path:
    """Validate staging, write the record, and rename into place. All or nothing."""
    problems = validate_tree(staging, aoi_name)
    if problems:
        raise SnapshotError("snapshot failed validation: " + "; ".join(problems[:20]))
    final = lay.snapshot_dir(sid)
    if final.exists():
        raise layout.ImmutableRecordError(f"{final} already exists; snapshots are never rewritten")
    layout.write_once_json(staging / layout.SNAPSHOT_RECORD, record)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, final)  # same filesystem; fails if final appeared meanwhile
    return final


def build(lay: layout.Layout, *, sid: str, attempt: int, cfg: dict, window: str,
          idempotency_key: str, config_digest: str, stage_digest: str, code: dict,
          research_runner: Callable[..., dict] = run_research_subprocess,
          on_progress: Callable[[str], None] = lambda msg: None) -> dict:
    """Seed, build, validate and publish one snapshot. Returns its record."""
    aoi_name = cfg["aoi"]
    scfg = cfg["snapshot"]
    staging = lay.staging_dir(sid, attempt)
    if staging.exists():
        shutil.rmtree(staging)  # a previous attempt's leftovers; staging is never published as-is
    started = layout.utc_stamp()
    parent = latest_completed(lay, aoi_name=aoi_name)
    on_progress(f"seeding from {parent or 'nothing (first snapshot)'}")
    dropped = seed_staging(staging, lay.snapshot_dir(parent) if parent else None,
                           list(scfg["refresh_sources"]))
    on_progress(f"refreshing {scfg['refresh_sources']}; dropped {len(dropped)} artifact(s)")
    research = research_runner(staging=staging, aoi=aoi_name, skip_flights=scfg["skip_flights"],
                               flight_snapshots=scfg["flight_snapshots"], timeout_s=scfg["timeout_s"])
    record = build_record(sid=sid, staging=staging, aoi_name=aoi_name, window=window,
                          idempotency_key=idempotency_key, config_digest=config_digest,
                          stage_digest=stage_digest, code=code, parent=parent, dropped=dropped,
                          refresh_sources=list(scfg["refresh_sources"]), started_utc=started,
                          research=research)
    on_progress("validating")
    publish(lay, sid, staging, record, aoi_name=aoi_name)
    return record


# --- bootstrap: a first snapshot from an operator's already-cited local cache -------


def _other_aoi_scoped(key: str, aoi_name: str) -> bool:
    from ..research.aoi import AOIS

    return any(f"/{other}/" in key for other in AOIS if other != aoi_name)


def prepare_seed(local_cache: Path, local_components: Path, aoi_name: str, dest: Path) -> list[str]:
    """Assemble a snapshot tree from a local research cache, for one AOI only.

    The first scheduled snapshot has no parent to copy the DEM from, and a DEM
    re-fetch is blocked in the cloud (see ``NATIVE_NETWORK_OFF``). The operator's
    local cache was built by the same research agent under the same allowlist,
    so it can seed the chain -- but only through the same validation as any
    snapshot, run here first and again in the cloud before publication.
    Returns the validation problems (empty means uploadable).
    """
    local_cache, local_components, dest = Path(local_cache), Path(local_components), Path(dest)
    scoped = local_components / "aoi" / aoi_name
    if not scoped.is_dir():
        return [f"{scoped} does not exist: run `naigos-research --aoi {aoi_name}` first"]
    manifest = json.loads((local_cache / "manifest.json").read_text())
    keep = {k: v for k, v in manifest.items() if not _other_aoi_scoped(k, aoi_name)}
    (dest / CACHE_DIRNAME).mkdir(parents=True, exist_ok=True)
    for entry in keep.values():
        rel = _safe_rel(entry.get("path"))
        if rel is None:
            return [f"unsafe path in local manifest: {entry.get('path')!r}"]
        src = local_cache / rel
        if src.is_file():
            (dest / CACHE_DIRNAME / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / CACHE_DIRNAME / rel)
    (dest / CACHE_DIRNAME / "manifest.json").write_text(json.dumps(keep, indent=2, sort_keys=True) + "\n")
    # The AOI-scoped copy is the authoritative one for this theatre; it becomes
    # both the top-level set and the scoped set, as a research run would write.
    for target in (dest / COMPONENTS_DIRNAME, dest / COMPONENTS_DIRNAME / "aoi" / aoi_name):
        target.mkdir(parents=True, exist_ok=True)
        for f in sorted(scoped.glob("*.json")):
            shutil.copy2(f, target / f.name)
    (dest / DATA_DOC).write_text(
        f"# DATA.md - provenance\n\nOperator-seeded snapshot for {aoi_name}, assembled from a local "
        "research cache. Every artifact below is listed with its sha256 in cache/manifest.json.\n")
    return validate_tree(dest, aoi_name)


def seed_id(content_sha256: str, code: dict, now=None) -> str:
    from .schedule import manual_window

    key = layout.digest({"stage": "seed", "content": content_sha256, "commit": (code or {}).get("commit")})
    return layout.snapshot_id(manual_window(now or layout.utc_now()), key)


def publish_seed(lay: layout.Layout, sid: str, *, aoi_name: str, code: dict, config_digest: str,
                 stage_digest: str, allow_existing: bool = False) -> dict:
    """Validate an uploaded seed in staging (attempt 1) and publish it."""
    staging = lay.staging_dir(sid, 1)
    if not staging.is_dir():
        raise SnapshotError(f"no uploaded seed at {staging}")
    if not allow_existing and latest_completed(lay, aoi_name=aoi_name) is not None:
        raise SnapshotError("completed snapshots already exist; a seed is only for bootstrapping "
                            "(pass allow_existing to add one anyway)")
    record = build_record(
        sid=sid, staging=staging, aoi_name=aoi_name, window=sid.split("-")[1],
        idempotency_key=layout.digest({"seed": sid}), config_digest=config_digest,
        stage_digest=stage_digest, code=code, parent=None, dropped=[], refresh_sources=[],
        started_utc=layout.utc_stamp(), research={}, origin="operator-seed")
    publish(lay, sid, staging, record, aoi_name=aoi_name)
    return record
