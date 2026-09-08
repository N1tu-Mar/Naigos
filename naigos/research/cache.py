"""Idempotent, cited cache for every raw artifact the research agent pulls.

Contract (spec section 8):
  * fetch -> cache -> cite. Nothing enters the env without a cited source.
  * idempotent + offline-after-first-run: a second run re-reads the cache and makes no network
    calls unless ``force=True`` or the artifact is missing.
  * every artifact records its source key, URL, license, sha256 and fetch timestamp in
    ``data_cache/manifest.json``, which is the provenance ledger ``docs/DATA.md`` is rendered from.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .allowlist import ALLOWLIST, check_url

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("NAIGOS_CACHE_DIR", REPO_ROOT / "data_cache"))
MANIFEST_PATH = CACHE_DIR / "manifest.json"

USER_AGENT = "naigos-research/0.1 (open-data research agent; contact via repository)"


@dataclass
class Artifact:
    """One cached raw file plus the provenance needed to cite it."""

    key: str
    source_key: str
    url: str
    path: str
    bytes: int
    sha256: str
    fetched_at: str
    license: str
    citation: str
    note: str = ""

    @property
    def abs_path(self) -> Path:
        return CACHE_DIR / self.path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def get_artifact(key: str) -> Artifact | None:
    """Return the cached artifact for ``key`` if it exists on disk, else ``None``."""
    entry = load_manifest().get(key)
    if entry is None:
        return None
    art = Artifact(**entry)
    return art if art.abs_path.exists() else None


def record(
    key: str,
    source_key: str,
    url: str,
    rel_path: str,
    payload: bytes,
    note: str = "",
) -> Artifact:
    """Write ``payload`` to the cache and register it in the manifest."""
    src = ALLOWLIST[source_key]
    dest = CACHE_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    art = Artifact(
        key=key,
        source_key=source_key,
        url=url,
        path=rel_path,
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        fetched_at=_now(),
        license=src.license,
        citation=src.citation,
        note=note,
    )
    manifest = load_manifest()
    manifest[key] = asdict(art)
    save_manifest(manifest)
    return art


def fetch(
    key: str,
    source_key: str,
    url: str,
    rel_path: str,
    *,
    force: bool = False,
    note: str = "",
    params: dict[str, Any] | None = None,
    timeout: int = 120,
) -> Artifact:
    """Fetch ``url`` (allowlist-checked) into the cache, or return the cached artifact."""
    cached = get_artifact(key)
    if cached is not None and not force:
        return cached
    check_url(url, source_key)
    import requests  # imported lazily so `data`-only installs need no HTTP stack

    resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return record(key, source_key, resp.url, rel_path, resp.content, note=note)


def produce(
    key: str,
    source_key: str,
    url: str,
    rel_path: str,
    builder: Callable[[], bytes],
    *,
    force: bool = False,
    note: str = "",
) -> Artifact:
    """Cache an artifact produced by ``builder`` (used where a library, not HTTP, does the pull)."""
    cached = get_artifact(key)
    if cached is not None and not force:
        return cached
    return record(key, source_key, url, rel_path, builder(), note=note)
