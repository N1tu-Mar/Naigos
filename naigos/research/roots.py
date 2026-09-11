"""Where the research layer reads and writes: the cache root and the component root.

Both default to the repository (``data_cache/`` and ``components/``), exactly as
before. Two ways to point them somewhere else, for a caller that must keep one
set of inputs apart from another -- the cloud pipeline builds every snapshot in
its own directory so a completed snapshot is never touched by the next one:

  * environment, before import: ``NAIGOS_CACHE_DIR`` and ``NAIGOS_COMPONENTS_DIR``;
  * explicitly, scoped: ``with research_roots(cache_dir=..., components_dir=...)``.

The scoped form restores the previous roots on exit, including on an exception,
so a failed snapshot build cannot leave a process pointed at a half-built root.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import cache, spec


@dataclass(frozen=True)
class Roots:
    cache_dir: Path
    components_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"


def current_roots() -> Roots:
    return Roots(cache_dir=cache.cache_dir(), components_dir=spec.components_dir())


def default_roots() -> Roots:
    """The repository-relative defaults, ignoring any override in effect."""
    return Roots(cache_dir=cache.DEFAULT_CACHE_DIR, components_dir=spec.DEFAULT_COMPONENTS_DIR)


@contextlib.contextmanager
def research_roots(
    *,
    cache_dir: str | os.PathLike | None = None,
    components_dir: str | os.PathLike | None = None,
) -> Iterator[Roots]:
    """Run the research layer against explicit roots, then put the old ones back.

    A root left as ``None`` keeps whatever is in effect. Not thread-safe: the
    roots are process globals, and the pipeline runs one snapshot build per
    process for exactly that reason.
    """
    prev_cache = cache.set_cache_dir(cache_dir) if cache_dir is not None else None
    prev_comp = spec.set_components_dir(components_dir) if components_dir is not None else None
    try:
        yield current_roots()
    finally:
        if prev_cache is not None:
            cache.set_cache_dir(prev_cache)
        if prev_comp is not None:
            spec.set_components_dir(prev_comp)
