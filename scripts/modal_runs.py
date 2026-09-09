#!/usr/bin/env python
"""List, fetch and verify training runs stored on the Modal Volume.

    uv run python scripts/modal_runs.py list
    uv run python scripts/modal_runs.py fetch smoke-s0-20260909T101500Z
    uv run python scripts/modal_runs.py verify runs/smoke-s0-20260909T101500Z

`list` and `fetch` shell out to the `modal` CLI, so they need Modal installed
and authenticated (`modal token new`). `verify` does not: it reads a local
directory, so a run fetched by any means -- including `modal volume get` typed
by hand -- can be checked, and so the checking logic is unit tested offline.

Verification is not a formality. A run that hit the Modal timeout leaves a
directory shaped exactly like a completed one, and a run whose worker fell back
to CPU leaves timings that look like GPU timings. Both are reported here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naigos.rl import runmeta  # noqa: E402
from naigos.rl.modal_train import RUNS_ROOT, VOLUME_NAME  # noqa: E402


def _modal(*args: str) -> int:
    if shutil.which("modal") is None:
        raise SystemExit(
            "the `modal` CLI is not on PATH. `uv pip install modal` then `modal token new`; "
            "no credential is stored in this repository."
        )
    print("$ modal " + " ".join(args))
    return subprocess.run(["modal", *args]).returncode


def cmd_list(a) -> int:
    return _modal("volume", "ls", VOLUME_NAME, a.path)


def cmd_fetch(a) -> int:
    name = runmeta.validate_run_name(a.run_name)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    rc = _modal("volume", "get", VOLUME_NAME, f"{RUNS_ROOT}/{name}", str(dest), "--force")
    if rc != 0:
        return rc
    return _report(dest / name)


def cmd_verify(a) -> int:
    return _report(Path(a.path))


def _report(directory: Path) -> int:
    print(json.dumps(runmeta.summarize_run(directory), indent=2, default=str))
    problems = runmeta.verify_run_dir(directory)
    if not problems:
        print(f"\nOK: {directory} is a complete, self-describing run.")
        return 0
    print(f"\n{len(problems)} problem(s) with {directory}:")
    for p in problems:
        print(f"  - {p}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list runs on the Volume")
    p_list.add_argument("path", nargs="?", default="/", help="path within the Volume")
    p_list.set_defaults(fn=cmd_list)

    p_fetch = sub.add_parser("fetch", help="download one run, then verify it")
    p_fetch.add_argument("run_name")
    p_fetch.add_argument("--dest", default="runs", help="local directory to download into")
    p_fetch.set_defaults(fn=cmd_fetch)

    p_verify = sub.add_parser("verify", help="check an already-downloaded run directory")
    p_verify.add_argument("path")
    p_verify.set_defaults(fn=cmd_verify)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
