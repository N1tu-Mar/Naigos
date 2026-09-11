"""Complete recovery state: what a training run needs to be picked up mid-flight.

The checkpoints this repository wrote before were *policy exports*. They carried
the actor and critic parameters, the Lagrange multiplier's pre-activation, the
red curriculum level and the iteration number -- everything the demo needs to
fly a route, and not enough to continue training. Restarting from one would have
silently reset:

  * **both optimizer states.** Adam's first and second moment estimates are the
    optimizer. Dropping them restarts from a zero-momentum, zero-variance state
    with a bias correction that thinks it is at step 1, so the first few updates
    after a "resume" are effectively a different learning rate.
  * **the multiplier's optimizer state.** Same problem on the constraint
    channel, where it matters more: the Lagrange multiplier is an integrator, so
    losing its Adam state loses the accumulated price of the cost constraint.
  * **the RNG stream.** Every rollout, every minibatch permutation and every
    evaluation draws from one key chain seeded once. Re-seeding at resume makes
    the second half of the run a different sample path, so the run is no longer
    reproducible from its seed and an interrupted run cannot be shown to equal
    an uninterrupted one.
  * **the reward curriculum.** Its dial is derived from the measured shootdown
    rate, which is only measured at an eval boundary. Without it a resumed run
    reverts to the survival-dominant weights it had at iteration 1.
  * **`history.json` continuity.** `run()` rewrites the file from its in-memory
    list, so a resume that started with an empty list would truncate the curve
    to the resumed segment.

So the schema-2 checkpoint carries all of it. The top-level `actor`, `critic`,
`lam_raw`, `red_level`, `iter` and `env_cfg` keys are kept exactly where they
were, because `naigos/demo/replay.py` and `naigos/demo/live.py` read
`pickle.load(f)["actor"]` and a checkpoint format change must not break the
demo. Everything new lives under `recovery`.

Restoring is grafting, not unpickling live objects: a `TrainState` holds an
`apply_fn` and an optax `GradientTransformation`, neither of which survives a
pickle round trip in a form worth trusting. The learner is rebuilt by
`init_learner` and the saved arrays are grafted onto that structure, with the
pytree structure, shapes and dtypes checked first. A checkpoint whose structure
does not match the code trying to load it is refused rather than silently
half-loaded.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import runmeta
from .ppo import Learner
from .red_team import RedCurriculum
from .reward import RewardCurriculum, RewardWeights

#: 1 = the policy-export format written before resume existed. 2 adds
#: `recovery`. `load` reads both; only 2 can be resumed from.
CHECKPOINT_SCHEMA = 2

REQUIRED_RECOVERY_KEYS = (
    "iteration",
    "rng_key",
    "actor_params",
    "actor_opt_state",
    "actor_step",
    "critic_params",
    "critic_opt_state",
    "critic_step",
    "lam_raw",
    "lam_opt_state",
    "red_level",
    "shootdown_rate",
    "survival_rate",
    "history",
    "red_curriculum",
    "reward_curriculum",
    "reward_weights_base",
)


class CheckpointError(RuntimeError):
    """A checkpoint cannot be read, or does not match the code reading it."""


# --- RNG --------------------------------------------------------------------


def _key_to_record(key) -> dict:
    """Serialise a PRNG key without assuming which JAX key representation is in use.

    JAX has two: a raw `uint32[2]` array and a typed key array. Storing the raw
    data plus the implementation name round-trips both, and a checkpoint written
    under one and read under the other is caught by the impl mismatch rather
    than by producing a different random stream.
    """
    try:
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
            return {
                "impl": str(jax.random.key_impl(key)),
                "data": np.asarray(jax.device_get(jax.random.key_data(key))),
            }
    except (AttributeError, TypeError):  # pragma: no cover - version dependent
        pass
    return {"impl": None, "data": np.asarray(jax.device_get(key))}


def _key_from_record(record: dict):
    if not isinstance(record, dict) or "data" not in record:
        raise CheckpointError("checkpoint has no usable RNG key record")
    data = jnp.asarray(record["data"], dtype=jnp.uint32)
    impl = record.get("impl")
    if impl:
        try:
            return jax.random.wrap_key_data(data, impl=impl)
        except (TypeError, ValueError) as e:  # pragma: no cover - version dependent
            raise CheckpointError(
                f"checkpoint used PRNG implementation {impl!r}, which this JAX cannot "
                f"reconstruct ({e}). Resuming would change the random stream."
            ) from e
    return data


# --- pytree grafting --------------------------------------------------------


def _graft(fresh, saved, what: str):
    """Put `saved`'s leaves into `fresh`'s structure, or refuse.

    The structure comes from the *code* and the values come from the
    *checkpoint*, so a network or optimizer that changed shape between the two
    is an error here instead of a confusing failure a thousand iterations later.
    """
    fresh_leaves, fresh_def = jax.tree.flatten(fresh)
    saved_leaves, saved_def = jax.tree.flatten(saved)
    if fresh_def != saved_def:
        raise CheckpointError(
            f"{what} in this checkpoint has a different structure to the one this code "
            f"builds. The checkpoint was written by different code; resume is refused. "
            f"(checkpoint: {saved_def}; this code: {fresh_def})"
        )
    out = []
    for i, (fl, sl) in enumerate(zip(fresh_leaves, saved_leaves)):
        arr = jnp.asarray(sl)
        want_shape, want_dtype = jnp.shape(fl), jnp.asarray(fl).dtype
        if arr.shape != want_shape:
            raise CheckpointError(
                f"{what} leaf {i}: checkpoint has shape {arr.shape}, this code expects "
                f"{want_shape}"
            )
        out.append(arr.astype(want_dtype))
    return jax.tree.unflatten(fresh_def, out)


def _to_host(tree):
    """Pull a pytree back to numpy so the pickle holds no device handles."""
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


# --- write ------------------------------------------------------------------


def save(
    path: str | os.PathLike,
    *,
    learner: Learner,
    key,
    iteration: int,
    red_level: float,
    shootdown_rate: float,
    survival_rate: float,
    survival_w: float,
    efficiency_w: float,
    history: list,
    env_cfg,
    ppo_cfg,
    red_curriculum: RedCurriculum,
    reward_curriculum: RewardCurriculum,
    reward_weights_base: RewardWeights,
    wall_s: float,
    run_name: str | None = None,
    code_commit: str | None = None,
    resumed_from: list | None = None,
) -> Path:
    """Write one checkpoint atomically.

    Atomic because the newest checkpoint is exactly the one a killed container
    was most likely mid-write on. Writing to a temporary name and renaming means
    a run that dies here loses the *new* checkpoint, not the last good one --
    the case `runmeta.select_checkpoint` walks backwards for.
    """
    path = Path(path)
    blob = {
        # --- unchanged policy-export surface: the demo reads these ---
        "schema": CHECKPOINT_SCHEMA,
        "actor": _to_host(learner.actor.params),
        "critic": _to_host(learner.critic.params),
        "lam_raw": float(learner.lam_raw),
        "red_level": float(red_level),
        "iter": int(iteration),
        "env_cfg": dataclasses.asdict(env_cfg),
        # --- everything needed to continue ---
        "recovery": {
            "iteration": int(iteration),
            "rng_key": _key_to_record(key),
            "actor_params": _to_host(learner.actor.params),
            "actor_opt_state": _to_host(learner.actor.opt_state),
            "actor_step": int(learner.actor.step),
            "critic_params": _to_host(learner.critic.params),
            "critic_opt_state": _to_host(learner.critic.opt_state),
            "critic_step": int(learner.critic.step),
            "lam_raw": np.asarray(jax.device_get(learner.lam_raw)),
            "lam_opt_state": _to_host(learner.lam_opt),
            "red_level": float(red_level),
            # Both curricula are driven by these two measured rates and they are
            # only refreshed at an eval boundary, so they are state, not derived.
            "shootdown_rate": float(shootdown_rate),
            "survival_rate": float(survival_rate),
            "survival_w": float(survival_w),
            "efficiency_w": float(efficiency_w),
            "history": list(history),
            # The curricula themselves are frozen dataclasses living in code.
            # Storing their fields makes a code change that moves a promotion
            # threshold a detectable incompatibility rather than an invisible
            # one, which is the whole point of refusing an incompatible resume.
            "red_curriculum": dataclasses.asdict(red_curriculum),
            "reward_curriculum": dataclasses.asdict(reward_curriculum),
            "reward_weights_base": dataclasses.asdict(reward_weights_base),
            "ppo": dataclasses.asdict(ppo_cfg),
            "wall_s": float(wall_s),
            "run_name": run_name,
            "code_commit": code_commit,
            "resumed_from": list(resumed_from or []),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# --- read -------------------------------------------------------------------


def load(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or "actor" not in blob:
        raise CheckpointError(f"{path} is not a Naigos checkpoint")
    return blob


def is_resumable(path: str | os.PathLike) -> bool:
    """True if this file is a complete, schema-2 checkpoint.

    Used as `runmeta.select_checkpoint(..., is_valid=is_resumable)`, so a
    truncated pickle or a pre-resume policy export is skipped over and the
    newest genuinely resumable checkpoint is chosen instead of being masked by
    it. Deliberately swallows every read error: the caller's next candidate is
    the right response to any of them.
    """
    try:
        blob = load(path)
    except Exception:
        return False
    rec = blob.get("recovery")
    if not isinstance(rec, dict):
        return False
    if blob.get("schema") != CHECKPOINT_SCHEMA:
        return False
    if any(k not in rec for k in REQUIRED_RECOVERY_KEYS):
        return False
    # the filename is what `select_checkpoint` sorts on, so it has to agree with
    # the contents or the "most recent" checkpoint is not the most recent one
    named = runmeta.checkpoint_iteration(path)
    return named is None or named == rec.get("iteration")


def latest_resumable(directory: str | os.PathLike) -> Path | None:
    return runmeta.select_checkpoint(directory, is_valid=is_resumable)


def curriculum_compatibility(
    blob: dict,
    *,
    red_curriculum: RedCurriculum,
    reward_curriculum: RewardCurriculum,
    reward_weights_base: RewardWeights,
) -> list[str]:
    """Differences between the curricula in the checkpoint and the ones in code.

    A resume across one of these produces a `history.json` whose two halves were
    trained against different promotion thresholds or different reward weights.
    Reported under the `curriculum` override key.
    """
    rec = blob.get("recovery") or {}
    problems = []
    for label, saved, current in (
        ("red curriculum", rec.get("red_curriculum"), dataclasses.asdict(red_curriculum)),
        ("reward curriculum", rec.get("reward_curriculum"), dataclasses.asdict(reward_curriculum)),
        ("reward weights", rec.get("reward_weights_base"), dataclasses.asdict(reward_weights_base)),
    ):
        if saved is None:
            problems.append(f"{label}: not recorded in this checkpoint")
            continue
        changed = sorted(
            k for k in set(saved) | set(current) if saved.get(k) != current.get(k)
        )
        if changed:
            problems.append(
                f"{label} changed since the checkpoint: {', '.join(changed)} "
                f"(checkpoint {[saved.get(k) for k in changed]}, "
                f"code {[current.get(k) for k in changed]})"
            )
    return problems


def restore_learner(template: Learner, blob: dict) -> Learner:
    """Graft a checkpoint's arrays onto a freshly initialised learner.

    `template` must come from `init_learner` with the same configuration, so the
    optimizer transformations, the network structure and the multiplier's Adam
    state are the ones this code builds. Only the values come from disk.
    """
    rec = blob.get("recovery")
    if not isinstance(rec, dict):
        raise CheckpointError(
            "this checkpoint predates resume support (no `recovery` block): it holds the "
            "policy but not the optimizer state, the RNG stream or the curriculum state. "
            "Training cannot be continued from it; it can only be replayed."
        )
    missing = [k for k in REQUIRED_RECOVERY_KEYS if k not in rec]
    if missing:
        raise CheckpointError(f"checkpoint recovery block is missing {missing}")

    actor = template.actor.replace(
        params=_graft(template.actor.params, rec["actor_params"], "actor parameters"),
        opt_state=_graft(template.actor.opt_state, rec["actor_opt_state"], "actor optimizer state"),
        step=jnp.asarray(rec["actor_step"], dtype=jnp.asarray(template.actor.step).dtype),
    )
    critic = template.critic.replace(
        params=_graft(template.critic.params, rec["critic_params"], "critic parameters"),
        opt_state=_graft(template.critic.opt_state, rec["critic_opt_state"], "critic optimizer state"),
        step=jnp.asarray(rec["critic_step"], dtype=jnp.asarray(template.critic.step).dtype),
    )
    return Learner(
        actor=actor,
        critic=critic,
        lam_raw=_graft(template.lam_raw, rec["lam_raw"], "lagrange multiplier"),
        lam_opt=_graft(template.lam_opt, rec["lam_opt_state"], "multiplier optimizer state"),
    )


def restore_key(blob: dict):
    return _key_from_record((blob.get("recovery") or {}).get("rng_key"))


# --- observation layout -----------------------------------------------------


def obs_config_from_blob(blob: dict) -> dict:
    """The observation flags the checkpoint's actor was trained with, as
    `EnvConfig.replace` overrides.

    A checkpoint written before a flag existed (checkpoints/theatre_1000.pkl)
    does not record it, and was trained without it.
    """
    env_cfg = blob.get("env_cfg") or {}
    return {"obs_edge_features": bool(env_cfg.get("obs_edge_features", False))}


def require_actor_ego_dim(actor_params, cfg, source) -> None:
    """Refuse an actor whose ego input width is not the one `cfg` builds.

    The width is read off the threat encoder's ego-query projection, whose
    input is the ego vector alone.
    """
    got = int(np.shape(actor_params["params"]["threat_encoder"]["Dense_2"]["kernel"])[0])
    if got != cfg.ego_dim:
        raise SystemExit(
            f"{source}: the actor takes a {got}-wide ego observation but this env builds "
            f"{cfg.ego_dim} (obs_edge_features={cfg.obs_edge_features}). The checkpoint's "
            f"env_cfg does not describe the observation its actor was trained on."
        )
