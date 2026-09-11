"""Promotion gates, the decision record, and the champion pointer.

**The decision.** ``decide`` turns an evaluation record into one of three
outcomes by checking every gate and recording each check:

  eligible      every gate passed on measured numbers
  rejected      at least one gate failed on measured numbers
  inconclusive  no gate failed, but at least one could not be decided -- a
                metric that is missing, not finite or out of range, or a champion
                that could not be re-evaluated. Never promoted.

There is no waiver. A gate whose input is absent is ``unknown``, and one
``unknown`` makes the candidate inconclusive; nothing here fills in a number.

Gates (thresholds from the versioned config's ``gates`` section):

  provenance       the candidate's run, snapshot and checkpoint verify cleanly
  heldout_seeds    evaluation seeds are recorded, non-empty, disjoint from training
  verifier         the independent CMDP verifier found no mismatch on sampled episodes
  min_survival     survival_rate  >= gates.min_survival_rate
  min_objective    objective_rate >= gates.min_objective_rate
  vs_avoid_nap     survival/objective not below the avoid-plus-nap baseline by more
                   than gates.baseline_tolerance, on the same held-out episodes
  vs_champion      exposure_early, shootdown_rate, terrain_rate, bounds_rate not above
                   the current champion (re-evaluated on the same episodes) by more
                   than gates.champion_tolerance; survival/objective not below it

**The pointer.** ``/pipeline/champions/current.json`` is small and names one
candidate, its checkpoint hash and a monotonically increasing ``generation``.
It moves only through ``publish_champion``, which (1) refuses unless the caller
names the generation it expects to replace -- compare-and-swap, so two
promoters cannot both win -- (2) writes the new generation's history entry
write-once, then (3) replaces the pointer atomically. Every refusal path raises
before step 3, so the existing pointer is byte-for-byte unchanged; that is
tested for each path. The previous champion stays in place and addressable
until the instant the rename lands.

Rollback is a *new* generation that points at an older candidate, not a rewind:
the history stays append-only and says who rolled back to what.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from ..rl import runmeta
from . import layout

DECISION_SCHEMA = 1
POINTER_SCHEMA = 1

ELIGIBLE, REJECTED, INCONCLUSIVE = "eligible", "rejected", "inconclusive"
PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"

#: Lower is better for these; the rest of the champion tolerances are higher-is-better.
LOWER_IS_BETTER = ("exposure_early", "shootdown_rate", "terrain_rate", "bounds_rate")
HIGHER_IS_BETTER = ("survival_rate", "objective_rate")
RATE_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


class PromotionRefused(RuntimeError):
    """The pointer was not moved. ``problems`` says why."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = list(problems or [])


class ChampionConflict(PromotionRefused):
    """The pointer moved since the caller read it."""


# --- the decision ------------------------------------------------------------------


def _metric(block: dict | None, name: str) -> float | None:
    """A rate in [0, 1], or None if absent/non-finite/out of range. Never invented."""
    if not isinstance(block, dict):
        return None
    v = block.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        return None
    return v


def _check(checks: list, gate: str, status: str, detail: str) -> None:
    checks.append({"gate": gate, "status": status, "detail": detail})


def decide(evaluation: dict, gates: dict, *, provenance_problems: list[str]) -> dict:
    """``{"outcome", "checks"}`` for one evaluated candidate. Pure."""
    checks: list[dict] = []

    if provenance_problems:
        _check(checks, "provenance", FAIL, "; ".join(provenance_problems[:10]))
    else:
        _check(checks, "provenance", PASS, "run, snapshot and checkpoint verify cleanly")

    held = evaluation.get("heldout") or {}
    seeds, train = held.get("seeds"), held.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or not isinstance(train, list):
        _check(checks, "heldout_seeds", UNKNOWN, "held-out or training seeds not recorded")
    elif set(seeds) & set(train):
        _check(checks, "heldout_seeds", FAIL, f"overlap with training seeds {sorted(set(seeds) & set(train))}")
    else:
        _check(checks, "heldout_seeds", PASS,
               f"{len(seeds)} held-out seeds x {held.get('worlds_per_seed')} worlds, disjoint from {train}")

    ver = evaluation.get("verifier") or {}
    if not isinstance(ver.get("episodes"), int) or ver.get("episodes", 0) < 1 or "ok" not in ver:
        _check(checks, "verifier", UNKNOWN, "no verifier result recorded")
    elif not ver["ok"]:
        _check(checks, "verifier", FAIL,
               f"{len(ver.get('mismatches') or [])} mismatch(es): {(ver.get('mismatches') or [''])[:3]}")
    else:
        _check(checks, "verifier", PASS, f"{ver['episodes']} episode(s) re-checked, no mismatch")

    cand = evaluation.get("candidate") or {}
    for gate, metric, key in (("min_survival", "survival_rate", "min_survival_rate"),
                              ("min_objective", "objective_rate", "min_objective_rate")):
        v = _metric(cand, metric)
        if v is None:
            _check(checks, gate, UNKNOWN, f"{metric} missing or not a rate")
        else:
            ok = v >= gates[key]
            _check(checks, gate, PASS if ok else FAIL, f"{metric} {v:.4f} vs minimum {gates[key]:.4f}")

    base = ((evaluation.get("baselines") or {}).get("avoid_nap"))
    tol = gates["baseline_tolerance"]
    for metric in HIGHER_IS_BETTER:
        c, b = _metric(cand, metric), _metric(base, metric)
        gate = f"vs_avoid_nap.{metric}"
        if c is None or b is None:
            _check(checks, gate, UNKNOWN, f"{metric} missing for candidate or avoid-plus-nap baseline")
        else:
            ok = c >= b - tol[metric]
            _check(checks, gate, PASS if ok else FAIL,
                   f"{metric} {c:.4f} vs baseline {b:.4f} (tolerance {tol[metric]})")

    champ = evaluation.get("champion")
    tol = gates["champion_tolerance"]
    if champ is None:
        _check(checks, "vs_champion", PASS, "no current champion: nothing to regress against")
    elif not isinstance(champ, dict) or champ.get("error") or not isinstance(champ.get("metrics"), dict):
        _check(checks, "vs_champion", UNKNOWN,
               f"champion could not be re-evaluated: {runmeta.redact(str((champ or {}).get('error')))}")
    else:
        for metric in RATE_METRICS:
            c, h = _metric(cand, metric), _metric(champ["metrics"], metric)
            gate = f"vs_champion.{metric}"
            if c is None or h is None:
                _check(checks, gate, UNKNOWN, f"{metric} missing for candidate or champion")
                continue
            ok = c <= h + tol[metric] if metric in LOWER_IS_BETTER else c >= h - tol[metric]
            _check(checks, gate, PASS if ok else FAIL,
                   f"{metric} {c:.4f} vs champion {champ.get('candidate_id')} {h:.4f} "
                   f"(tolerance {tol[metric]})")

    statuses = {c["status"] for c in checks}
    outcome = REJECTED if FAIL in statuses else INCONCLUSIVE if UNKNOWN in statuses else ELIGIBLE
    return {"outcome": outcome, "checks": checks}


def build_decision(*, candidate_id: str, evaluation: dict, evaluation_sha256: str, gates: dict,
                   config_version: int, config_digest: str, auto_promote: bool, code: dict,
                   provenance_problems: list[str]) -> dict:
    verdict = decide(evaluation, gates, provenance_problems=provenance_problems)
    return {
        "schema": DECISION_SCHEMA,
        "candidate_id": layout.validate_candidate_id(candidate_id),
        "snapshot_id": evaluation.get("snapshot_id"),
        "created_utc": layout.utc_stamp(),
        "outcome": verdict["outcome"],
        "checks": verdict["checks"],
        "gates": gates,
        "gates_digest": layout.digest(gates),
        "evaluation_sha256": evaluation_sha256,
        "config_version": config_version,
        "config_digest": config_digest,
        "code": code,
        "parents": {"snapshot_id": evaluation.get("snapshot_id"),
                    "champion": (evaluation.get("champion") or {}).get("candidate_id")
                    if isinstance(evaluation.get("champion"), dict) else None},
        # What happens next is recorded, not implied. Shadow is the default:
        # an eligible decision changes nothing until an operator promotes it,
        # unless the versioned config explicitly says auto_promote: true.
        "action": ("auto_promote" if (auto_promote and verdict["outcome"] == ELIGIBLE)
                   else "shadow" if verdict["outcome"] == ELIGIBLE else "none"),
    }


# --- the pointer -------------------------------------------------------------------


def read_champion(lay: layout.Layout) -> dict | None:
    return layout.read_json(lay.champion_pointer)


def _history_generations(lay: layout.Layout) -> list[int]:
    d = lay.champion_pointer.parent / "history"
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("g*.json"):
        stem = p.stem[1:]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def champion_history(lay: layout.Layout) -> list[dict]:
    return [layout.read_json(lay.champion_generation(g)) for g in _history_generations(lay)]


def promoted_candidates(lay: layout.Layout) -> set[str]:
    return {h.get("candidate_id") for h in champion_history(lay) if h}


def publish_champion(lay: layout.Layout, *, candidate_id: str, snapshot_id: str,
                     checkpoint: dict, decision_sha256: str | None, evaluation_sha256: str | None,
                     mode: str, actor: str, expected_generation: int, code: dict,
                     note: str | None = None, rollback_of: int | None = None) -> dict:
    """Move the pointer to ``candidate_id``, or raise without touching it."""
    if mode not in ("auto", "manual", "rollback"):
        raise ValueError(f"unknown promotion mode {mode!r}")
    layout.validate_candidate_id(candidate_id)
    current = read_champion(lay)
    current_gen = int((current or {}).get("generation") or 0)
    if current_gen != expected_generation:
        raise ChampionConflict(
            f"champion is at generation {current_gen}, this promotion expected {expected_generation}",
            [f"generation moved to {current_gen}"])
    # An interrupted publish can leave a history entry that never became
    # current; step past it rather than colliding with it forever.
    new_gen = max([current_gen] + _history_generations(lay)) + 1
    pointer = {
        "schema": POINTER_SCHEMA,
        "generation": new_gen,
        "candidate_id": candidate_id,
        "snapshot_id": snapshot_id,
        "checkpoint": checkpoint,
        "decision_sha256": decision_sha256,
        "evaluation_sha256": evaluation_sha256,
        "mode": mode,
        "promoted_utc": layout.utc_stamp(),
        "promoted_by": runmeta.redact(actor)[:120],
        "note": runmeta.redact(note)[:500] if note else None,
        "rollback_of": rollback_of,
        "code": code,
        "previous": ({"generation": current_gen, "candidate_id": current.get("candidate_id")}
                     if current else None),
    }
    try:
        layout.write_once_json(lay.champion_generation(new_gen), pointer)
    except layout.ImmutableRecordError:
        raise ChampionConflict(f"generation {new_gen} was published concurrently",
                               [f"history g{new_gen:06d} exists"]) from None
    layout.atomic_write_json(lay.champion_pointer, pointer)
    return pointer


def promote(lay: layout.Layout, *, candidate_id: str, actor: str, mode: str,
            expected_generation: int | None, revalidate: Callable[[str], dict],
            note: str | None = None) -> dict:
    """Re-run validation for ``candidate_id`` and, only if all of it passes, publish.

    ``revalidate(candidate_id)`` must return ``{"problems": [...], "snapshot_id",
    "checkpoint", "decision_sha256", "evaluation_sha256", "code"}``. It is where
    the caller re-hashes the artifacts, re-checks the recorded decision, and
    re-runs the evaluation; this function only refuses or publishes.
    """
    before = read_champion(lay)
    gen = int((before or {}).get("generation") or 0)
    if expected_generation is not None and expected_generation != gen:
        raise ChampionConflict(f"champion is at generation {gen}, expected {expected_generation}")
    if before and before.get("candidate_id") == candidate_id:
        raise PromotionRefused(f"{candidate_id} is already the champion (generation {gen})")
    result = revalidate(candidate_id)
    problems = list(result.get("problems") or [])
    if problems:
        raise PromotionRefused(f"refusing to promote {candidate_id}: validation failed", problems)
    return publish_champion(
        lay, candidate_id=candidate_id, snapshot_id=result["snapshot_id"],
        checkpoint=result["checkpoint"], decision_sha256=result.get("decision_sha256"),
        evaluation_sha256=result.get("evaluation_sha256"), mode=mode, actor=actor,
        expected_generation=gen, code=result.get("code") or {}, note=note)


def rollback(lay: layout.Layout, *, to_generation: int, actor: str,
             revalidate_provenance: Callable[[str], dict], note: str | None = None) -> dict:
    """Publish a new generation pointing at the candidate of ``to_generation``.

    Rollback re-checks provenance (artifacts exist and hash as recorded) but not
    the regression gates: it is a deliberate move back to something that
    passed them once, usually because the newer champion misbehaved.
    """
    target = layout.read_json(lay.champion_generation(to_generation))
    if target is None:
        raise PromotionRefused(f"no champion generation {to_generation} in history")
    before = read_champion(lay)
    gen = int((before or {}).get("generation") or 0)
    if before and before.get("candidate_id") == target.get("candidate_id"):
        raise PromotionRefused(f"{target.get('candidate_id')} is already the champion")
    result = revalidate_provenance(target["candidate_id"])
    problems = list(result.get("problems") or [])
    if result.get("checkpoint", {}).get("sha256") != (target.get("checkpoint") or {}).get("sha256"):
        problems.append("checkpoint hash differs from the one generation "
                        f"{to_generation} promoted")
    if problems:
        raise PromotionRefused(f"refusing to roll back to generation {to_generation}", problems)
    return publish_champion(
        lay, candidate_id=target["candidate_id"], snapshot_id=target["snapshot_id"],
        checkpoint=target["checkpoint"], decision_sha256=target.get("decision_sha256"),
        evaluation_sha256=target.get("evaluation_sha256"), mode="rollback", actor=actor,
        expected_generation=gen, code=result.get("code") or target.get("code") or {},
        note=note, rollback_of=to_generation)


def champion_checkpoint_path(lay: layout.Layout, pointer: dict) -> Path:
    """The pointer's checkpoint as a path, validated to stay inside its candidate."""
    cid = layout.validate_candidate_id(pointer.get("candidate_id"))
    rel = (pointer.get("checkpoint") or {}).get("file")
    if not isinstance(rel, str) or not runmeta.CHECKPOINT_RE.match(rel):
        raise layout.LayoutError(f"champion pointer names an invalid checkpoint {rel!r}")
    return lay.candidate_dir(cid) / rel
