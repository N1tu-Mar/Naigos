#!/usr/bin/env python
"""Operate the cloud learning pipeline: status, inspect, promote, pause, resume.

    modal token new                                       # once, on any machine
    modal deploy naigos/rl/modal_pipeline.py              # the only step that needs this laptop
    uv run python scripts/pipeline.py status              # any time, from any clone
    uv run python scripts/pipeline.py list-candidates
    uv run python scripts/pipeline.py inspect <id>        # s-... | c-... | g<N> | champion
    uv run python scripts/pipeline.py promote <candidate-id> --approver <name>
    uv run python scripts/pipeline.py rollback --generation <N> --approver <name>
    uv run python scripts/pipeline.py pause --reason "budget review"
    uv run python scripts/pipeline.py resume
    uv run python scripts/pipeline.py retry <snapshot-or-candidate-id>
    uv run python scripts/pipeline.py run-now snapshot|nightly|weekly
    uv run python scripts/pipeline.py config show | config set --file changes.json
    uv run python scripts/pipeline.py prune [--apply]
    uv run python scripts/pipeline.py seed               # once: bootstrap the first snapshot

Every command is executed by the *deployed* app, next to the Volume, so it
reads the authoritative remote record and needs no local job index: a fresh
clone with Modal credentials sees exactly what the laptop that deployed it
sees. ``--volume-dir`` runs the read-only commands against a local copy instead
(for example one fetched with ``modal volume get naigos-runs /pipeline``), with
no Modal account at all.

``promote`` re-validates the candidate in the cloud -- provenance, hashes, the
recorded decision -- and re-runs its held-out evaluation against the current
champion before the pointer moves. It prints why if it refuses; the champion
is then unchanged.

No credential is read, printed or passed as an argument here. Modal
authenticates from ``~/.modal.toml`` or ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naigos.pipeline import admin as padmin  # noqa: E402
from naigos.rl import runmeta  # noqa: E402
from naigos.rl.modal_pipeline import APP_NAME  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEPLOY_HINT = "modal deploy naigos/rl/modal_pipeline.py"


# --- transport ---------------------------------------------------------------------


def _deployed(fn_name: str):
    try:
        import modal
    except ImportError:
        raise SystemExit("the `modal` client is not installed: `uv pip install modal` then "
                         "`modal token new`. Or use --volume-dir for a local copy.") from None
    last = None
    for getter in ("from_name", "lookup"):
        get = getattr(modal.Function, getter, None)
        if get is None:
            continue
        try:
            return get(APP_NAME, fn_name)
        except Exception as e:  # noqa: BLE001 - auth and not-deployed both land here
            last = e
    raise SystemExit(f"cannot reach the deployed {APP_NAME}/{fn_name}: {runmeta.redact(str(last))}\n"
                     f"Deploy it first:\n  {DEPLOY_HINT}")


def _offline_services(volume_dir: str):
    from naigos.pipeline import layout, leases, worker

    lay = layout.Layout(Path(volume_dir))
    if not lay.root.is_dir():
        raise SystemExit(f"{lay.root} does not exist: --volume-dir must contain pipeline/")
    return worker.Services(lay=lay, leases=leases.LeaseManager(leases.FileLeaseStore(lay.root / "locks")),
                           code=runmeta.git_info(REPO))


def call_admin(a, command: str, args: dict | None = None) -> dict:
    args = dict(args or {})
    if getattr(a, "volume_dir", None):
        if command not in padmin.READ_ONLY:
            raise SystemExit(f"`{command}` changes remote state and runs only in the deployed app; "
                             "--volume-dir is read-only")
        return padmin.handle(_offline_services(a.volume_dir), command, args)
    return _deployed("admin").remote(command, args)


# --- rendering ---------------------------------------------------------------------


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "  (none)"
    widths = {c: max(len(c), *(len(str(r.get(c) if r.get(c) is not None else "-")) for r in rows))
              for c in cols}
    head = "  " + "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "  " + "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        lines.append("  " + "  ".join(str(r.get(c) if r.get(c) is not None else "-").ljust(widths[c])
                                      for c in cols))
    return "\n".join(lines)


def render_status(report: dict) -> str:
    out = []
    cfg = report.get("config") or {}
    if cfg.get("error"):
        out.append(f"CONFIG INVALID: {cfg['error']}\n  {cfg.get('note')}")
    else:
        state = f"PAUSED ({cfg.get('pause_reason')})" if cfg.get("paused") else "active"
        out.append(f"pipeline: {state}   config v{cfg.get('version')} ({cfg.get('source')})   "
                   f"aoi {cfg.get('aoi')}   auto_promote {cfg.get('auto_promote')}")
    out.append("\nschedule (UTC):")
    out.append(_table([{"stage": k, **v} for k, v in (report.get("schedule") or {}).items()],
                      ["stage", "status", "cron", "next_utc", "note"]))
    last = report.get("coordinator") or {}
    if last:
        out.append(f"\nlast tick: {last.get('kind')} at {last.get('at_utc')}: "
                   f"{last.get('status')} -- {last.get('reason')}")
    ch = report.get("champion")
    out.append("\nchampion: " + (f"generation {ch['generation']} = {ch['candidate_id']} "
                                f"({ch['mode']} by {ch['promoted_by']} at {ch['promoted_utc']})"
                                if ch else "none yet"))
    out.append("\nsnapshots:")
    out.append(_table(report.get("snapshots") or [],
                      ["snapshot_id", "status", "created_utc", "parent", "failure_reason"]))
    out.append("\ncandidates:")
    out.append(_table(report.get("candidates") or [],
                      ["candidate_id", "status", "decision", "action", "snapshot_id", "failure_reason"]))
    trouble = [j for j in report.get("jobs") or []
               if j["status"] in ("failed", "unknown", "running", "queued")]
    out.append("\njobs needing attention or in flight:")
    out.append(_table(trouble, ["key", "status", "attempt", "heartbeat_age_s", "failure_reason"]))
    return "\n".join(out)


def _print(a, payload: dict, text: str | None = None) -> None:
    if a.json or text is None:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


# --- commands ----------------------------------------------------------------------


def cmd_status(a) -> int:
    report = call_admin(a, "status")
    _print(a, report, render_status(report))
    return 0


def cmd_list_snapshots(a) -> int:
    rows = call_admin(a, "list-snapshots")["snapshots"]
    _print(a, {"snapshots": rows}, _table(rows, ["snapshot_id", "status", "created_utc", "parent",
                                                 "content_sha256"]))
    return 0


def cmd_list_candidates(a) -> int:
    rows = call_admin(a, "list-candidates")["candidates"]
    _print(a, {"candidates": rows}, _table(rows, ["candidate_id", "kind", "status", "decision",
                                                  "action", "is_champion", "snapshot_id"]))
    return 0


def cmd_inspect(a) -> int:
    print(json.dumps(call_admin(a, "inspect", {"id": a.id}), indent=2, default=str))
    return 0


def cmd_promote(a) -> int:
    result = _deployed("promote_candidate").remote(a.candidate_id, a.approver,
                                                   a.expected_generation, a.note)
    if result.get("ok"):
        p = result["pointer"]
        print(f"promoted {p['candidate_id']} to champion generation {p['generation']} "
              f"(previous: {p.get('previous')})")
        return 0
    print(f"NOT promoted -- the champion is unchanged.\n{result.get('error')}")
    for problem in result.get("problems") or []:
        print(f"  - {problem}")
    return 1


def cmd_rollback(a) -> int:
    result = _deployed("rollback_champion").remote(a.generation, a.approver, a.note)
    if result.get("ok"):
        p = result["pointer"]
        print(f"champion generation {p['generation']} now points at {p['candidate_id']} "
              f"(rollback to g{a.generation})")
        return 0
    print(f"NOT rolled back -- the champion is unchanged.\n{result.get('error')}")
    for problem in result.get("problems") or []:
        print(f"  - {problem}")
    return 1


def cmd_pause(a) -> int:
    _print(a, call_admin(a, "pause", {"reason": a.reason, "actor": a.approver}))
    return 0


def cmd_resume(a) -> int:
    _print(a, call_admin(a, "resume", {"actor": a.approver}))
    return 0


def cmd_retry(a) -> int:
    _print(a, call_admin(a, "retry", {"id": a.id}))
    return 0


def cmd_unlock(a) -> int:
    if not a.yes:
        raise SystemExit(
            "unlock removes a lease so a stage can run again. Only do this after confirming, in "
            "the Modal dashboard or with `status`, that the job which held it is gone -- two "
            "writers in one candidate interleave checkpoints. Re-run with --yes to proceed.")
    _print(a, call_admin(a, "unlock", {"key": a.key, "force_live": a.force_live}))
    return 0


def cmd_run_now(a) -> int:
    _print(a, call_admin(a, "run-now", {"kind": a.kind}))
    return 0


def cmd_config(a) -> int:
    if a.action == "show":
        _print(a, call_admin(a, "config-show"))
        return 0
    if not a.file:
        raise SystemExit("config set needs --file with a JSON object of top-level sections to replace")
    changes = json.loads(Path(a.file).read_text())
    _print(a, call_admin(a, "config-set", {"changes": changes, "actor": a.approver}))
    return 0


def cmd_seed(a) -> int:
    """Bootstrap the first snapshot from this clone's cited research cache.

    Validated here first, so a broken cache is never uploaded; uploaded to a
    staging directory; then validated again and published by the deployed app,
    which trusts nothing the client says about the bytes.
    """
    import tempfile

    from naigos.pipeline import config as pcfg
    from naigos.pipeline import layout
    from naigos.pipeline import snapshot as psnap
    from naigos.rl.modal_pipeline import VOLUME_NAME

    aoi = a.aoi or pcfg.load_default()["aoi"]
    with tempfile.TemporaryDirectory(prefix="naigos-seed-") as td:
        tree = Path(td) / "seed"
        problems = psnap.prepare_seed(Path(a.cache_dir), Path(a.components_dir), aoi, tree)
        if problems:
            print(f"refusing to upload: the local cache does not validate for {aoi}:")
            for p in problems:
                print(f"  - {p}")
            return 1
        sid = psnap.seed_id(psnap.content_digest(tree)["sha256"], runmeta.git_info(REPO))
        remote = f"/{layout.PIPELINE_DIRNAME}/snapshots/.staging/{sid}.1"
        size = sum(f.stat().st_size for f in tree.rglob("*") if f.is_file())
        print(f"uploading {size / 1e6:.1f} MB for {aoi} to {VOLUME_NAME}:{remote}")
        import modal

        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(tree), remote)
    _print(a, call_admin(a, "publish-seed", {"snapshot_id": sid, "aoi": aoi,
                                             "allow_existing": a.allow_existing}))
    return 0


def cmd_prune(a) -> int:
    _print(a, call_admin(a, "prune" if a.apply else "prune-plan"))
    if not a.apply:
        print("\n(dry run; re-run with --apply to delete the listed directories)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=f"deploy: {DEPLOY_HINT}")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--volume-dir", default=None,
                    help="read-only: run against a local copy containing pipeline/, no Modal needed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        return p

    add("status", cmd_status, "scheduled/queued/running/completed/failed/skipped/rejected/"
                              "inconclusive/promoted, from the remote record")
    add("list-snapshots", cmd_list_snapshots, "every snapshot and its state")
    add("list-candidates", cmd_list_candidates, "every candidate, its decision and state")
    p = add("inspect", cmd_inspect, "full records for one snapshot, candidate or champion generation")
    p.add_argument("id")
    p = add("promote", cmd_promote, "re-validate one eligible candidate and make it champion")
    p.add_argument("candidate_id")
    p.add_argument("--approver", default=padmin.default_actor(), help="recorded in the pointer")
    p.add_argument("--expected-generation", type=int, default=None,
                   help="refuse unless the champion is still at this generation")
    p.add_argument("--note", default=None)
    p = add("rollback", cmd_rollback, "repoint the champion at an earlier generation's candidate")
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--approver", default=padmin.default_actor())
    p.add_argument("--note", default=None)
    p = add("pause", cmd_pause, "publish a config version that stops all costly work")
    p.add_argument("--reason", default=None)
    p.add_argument("--approver", default=padmin.default_actor())
    p = add("resume", cmd_resume, "publish a config version that lifts the pause")
    p.add_argument("--approver", default=padmin.default_actor())
    p = add("retry", cmd_retry, "re-run one failed stage (training resumes from its checkpoint)")
    p.add_argument("id")
    p = add("unlock", cmd_unlock, "clear a stale lease after confirming its holder is gone")
    p.add_argument("key", help="e.g. candidate:c-20260911-nightly-0123456789ab or champion")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--force-live", action="store_true", help="even if Modal says the holder is live")
    p = add("run-now", cmd_run_now, "run one coordinator tick immediately (manual window)")
    p.add_argument("kind", choices=["snapshot", "nightly", "weekly"])
    p = add("config", cmd_config, "show the effective config, or publish a new version")
    p.add_argument("action", choices=["show", "set"])
    p.add_argument("--file", default=None)
    p.add_argument("--approver", default=padmin.default_actor())
    p = add("prune", cmd_prune, "retention: list (or --apply) directories past the keep windows")
    p.add_argument("--apply", action="store_true")
    p = add("seed", cmd_seed, "bootstrap the first snapshot from this clone's cited research cache")
    p.add_argument("--aoi", default=None, help="default: the configured AOI")
    p.add_argument("--cache-dir", default=str(REPO / "data_cache"))
    p.add_argument("--components-dir", default=str(REPO / "components"))
    p.add_argument("--allow-existing", action="store_true",
                   help="publish even though completed snapshots already exist")

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
