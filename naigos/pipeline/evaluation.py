"""Held-out evaluation: fixed recorded seeds, the project's own evaluator, the verifier.

What "held out" means here, precisely, because next-steps.md E-6 is right that
it is easy to overclaim: evaluation episodes are generated from integer seeds
listed in the versioned config (``evaluation.heldout_seeds``), every one of them
disjoint from every training seed, with the same scenario generator training
uses. That is held-out *seeds*, not a held-out *distribution*. It makes the
result reproducible and stops a candidate from being scored on the episodes it
trained on; it does not claim generalization to different route geometry.

Every policy in one evaluation -- the candidate, the avoid-plus-nap and direct
baselines, and the current champion -- is rolled out on the *same* keys and
measured by ``naigos.rl.train.rollout_metrics``, the arithmetic ``evaluate``
uses. The first ``verifier_episodes`` candidate episodes are then re-rolled
unbatched and handed to ``naigos.rl.verifier.verify_trace``, the pure-NumPy
CMDP verifier that shares no code with the env.

Evaluation always runs on CPU (the evaluator Function requests no GPU): the
same checkpoint on the same keys then produces the same numbers at promotion
time, which is what ``compare_reproduction`` checks before a pointer moves.

Only ``run_evaluation`` and the helpers below it import JAX, lazily, so the
rest of the package stays importable (and testable) without it.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

from . import config as pcfg
from . import layout

EVALUATION_SCHEMA = 1
POLICY_BLOCKS = ("candidate", "baselines.avoid_nap", "baselines.direct")


class HeldOutOverlap(ValueError):
    """Evaluation seeds share a seed with training. Refused, never waived."""


def heldout_plan(cfg: dict, kind: str) -> dict:
    ev = cfg["evaluation"]
    return {
        "seeds": list(ev["heldout_seeds"]),
        "worlds_per_seed": int(ev["weekly_worlds_per_seed"] if kind == "weekly" else ev["worlds_per_seed"]),
        "red_level": float(ev["red_level"]),
        "use_cbf": bool(ev["use_cbf"]),
        "verifier_episodes": int(ev["verifier_episodes"]),
        "training_seeds": sorted(set(pcfg.training_seeds(cfg))),
    }


def assert_heldout(plan: dict, run_seed: int | None) -> dict:
    """Refuse a plan whose seeds overlap training; return it with the run seed recorded."""
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise HeldOutOverlap("no held-out seeds recorded: an evaluation with no fixed seeds is not reproducible")
    train = set(plan.get("training_seeds") or [])
    if run_seed is not None:
        train.add(int(run_seed))
    overlap = sorted(set(seeds) & train)
    if overlap:
        raise HeldOutOverlap(
            f"held-out seeds {overlap} were used for training; the evaluation would score the "
            "candidate on episodes it trained on"
        )
    return {**plan, "training_seeds": sorted(train)}


def _dig(doc: dict, dotted: str):
    node = doc
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def compare_reproduction(recorded: dict, rerun: dict, max_err: float) -> list[str]:
    """Problems if a re-run of the same evaluation does not reproduce the record."""
    problems = []
    for block in POLICY_BLOCKS:
        a, b = _dig(recorded, block), _dig(rerun, block)
        if not isinstance(a, dict) or not isinstance(b, dict):
            problems.append(f"{block}: missing from {'record' if not isinstance(a, dict) else 're-run'}")
            continue
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
                problems.append(f"{block}.{k}: not comparable ({x!r} vs {y!r})")
            elif not (math.isfinite(x) and math.isfinite(y)) or abs(x - y) > max_err:
                problems.append(f"{block}.{k}: recorded {x!r}, re-run {y!r} (max error {max_err})")
    for key in ("seeds", "worlds_per_seed", "red_level"):
        if (recorded.get("heldout") or {}).get(key) != (rerun.get("heldout") or {}).get(key):
            problems.append(f"heldout.{key} differs between record and re-run")
    return problems


def checkpoint_entry(path: str | Path) -> dict:
    from ..rl import runmeta

    path = Path(path)
    return {"file": path.name, "sha256": layout.sha256_file(path),
            "iteration": runmeta.checkpoint_iteration(path)}


def load_actor_params(path: str | Path):
    """The actor parameters, where the demo has always read them."""
    with open(path, "rb") as f:
        return pickle.load(f)["actor"]


# --- the JAX part ------------------------------------------------------------------


def build_env(*, aoi: str, n_blue: int, n_threat: int, cell_m: float, red_level: float):
    """The evaluation env: the snapshot's theatre at a fixed, recorded red level."""
    from ..env.flight_env import NaigosEnv
    from ..env.theatre_bridge import env_from_theatre
    from ..rl.red_team import RedCurriculum

    cfg, hmap, _notes = env_from_theatre(aoi=aoi, n_blue=n_blue, n_threat=n_threat, cell_m=cell_m)
    cfg = RedCurriculum().apply(cfg, red_level)
    return NaigosEnv(cfg, hmap=hmap)


def _to_numpy_trace(traj):
    import numpy as np

    out = {k: np.asarray(v) for k, v in traj.items() if k != "terms"}
    out["terms"] = type(traj["terms"])(*[np.asarray(x) for x in traj["terms"]])
    return out


def run_evaluation(env, plan: dict, candidate_params, champion_params=None) -> dict:
    """Roll every policy on the same held-out keys. Returns metrics and verifier result."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from ..rl import verifier
    from ..rl.ppo import greedy_policy
    from ..rl.train import BASELINE_POLICIES, rollout_metrics

    keys = jnp.concatenate([jax.random.split(jax.random.PRNGKey(int(s)), plan["worlds_per_seed"])
                            for s in plan["seeds"]])
    n = int(keys.shape[0])
    afilter = None
    if plan["use_cbf"]:
        from ..rl.cbf import CBFConfig, make_policy_filter

        afilter = make_policy_filter(CBFConfig(), env.cfg)

    def measure(pol) -> dict:
        final, traj = jax.jit(jax.vmap(lambda k: env.rollout(k, pol, action_filter=afilter)))(keys)
        return rollout_metrics(final, traj, n, env.cfg.n_blue)

    cand_pol = greedy_policy(candidate_params, env.cfg)
    result = {
        "candidate": measure(cand_pol),
        "baselines": {name: measure(pol) for name, pol in BASELINE_POLICIES.items()},
        "n_episodes": n,
        "device": jax.default_backend(),
    }
    if champion_params is not None:
        try:
            result["champion_metrics"] = measure(greedy_policy(champion_params, env.cfg))
        except Exception as e:  # noqa: BLE001 - an incompatible champion is inconclusive, not fatal
            result["champion_error"] = f"{type(e).__name__}: {e}"

    reports, mismatches = [], []
    for i in range(min(plan["verifier_episodes"], n)):
        final, traj = env.rollout(keys[i], cand_pol, action_filter=afilter)
        rep = verifier.verify_trace(env.cfg, np.asarray(final.hmap), _to_numpy_trace(traj),
                                    np.asarray(final.threats.kind), np.asarray(final.threats.active))
        reports.append({"episode": i, "ok": rep.ok, "shootdowns": rep.shootdowns,
                        "terrain_violations": rep.terrain_violations,
                        "bounds_violations": rep.bounds_violations,
                        "max_pd_error": rep.max_pd_error})
        mismatches.extend(f"episode {i}: {m}" for m in rep.mismatches[:20])
    result["verifier"] = {"episodes": len(reports), "ok": not mismatches,
                          "mismatches": mismatches[:100], "reports": reports}
    return result


def build_record(*, candidate_id: str, snapshot_rec: dict, plan: dict, env_shape: dict,
                 checkpoint: dict, measured: dict, champion: dict | None, code: dict,
                 config_digest: str, config_version: int) -> dict:
    """The write-once ``evaluation.json``. ``champion`` is ``None`` when there is none."""
    champ_block = None
    if champion is not None:
        champ_block = {
            "candidate_id": champion.get("candidate_id"),
            "generation": champion.get("generation"),
            "checkpoint_sha256": (champion.get("checkpoint") or {}).get("sha256"),
            "metrics": measured.get("champion_metrics"),
            "error": champion.get("error") or measured.get("champion_error"),
        }
    return {
        "schema": EVALUATION_SCHEMA,
        "candidate_id": layout.validate_candidate_id(candidate_id),
        "snapshot_id": snapshot_rec.get("snapshot_id"),
        "snapshot_content_sha256": (snapshot_rec.get("content") or {}).get("sha256"),
        "created_utc": layout.utc_stamp(),
        "code": code,
        "config_digest": config_digest,
        "config_version": config_version,
        "aoi": (snapshot_rec.get("aoi") or {}).get("name"),
        "heldout": {**plan, "n_episodes": measured.get("n_episodes")},
        "env": env_shape,
        "device": measured.get("device"),
        "checkpoint": checkpoint,
        "candidate": measured.get("candidate"),
        "baselines": measured.get("baselines"),
        "champion": champ_block,
        "verifier": measured.get("verifier"),
    }
