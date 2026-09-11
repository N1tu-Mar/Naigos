"""Checkpoint recovery state and interrupted/resumed training -- offline, on CPU.

The claims being tested are the ones the detached-execution path rests on:

  * up to the interruption, an interrupted run and an uninterrupted one are the
    same run, bit for bit, and the checkpoint carries all of it;
  * resuming is reproducible: the same checkpoint resumed twice reaches the same
    learner, bit for bit.

What a resume does NOT reproduce is the uninterrupted run's sample path after
the interruption. PPO rollouts persist across iterations (episodes run over
iteration boundaries) and the in-flight episodes are not checkpointed, so a
resumed run starts fresh episodes where the uninterrupted one was mid-sortie.
Everything else -- parameters, optimizer states, multiplier, RNG stream,
curricula, history -- comes off the checkpoint exactly.

Everything here runs on synthetic terrain with a two-agent, two-world
configuration so the whole file finishes in under a minute on a laptop CPU. The
sizes are chosen to be cheap and to still exercise the parts that are easy to
get wrong: an evaluation boundary and a curriculum tick both fall inside the
resumed segment, so the RNG splits they perform are part of what has to match.
"""
from __future__ import annotations

import dataclasses
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest

from naigos.env.config import EnvConfig
from naigos.rl import checkpoint as ck
from naigos.rl import runmeta as rm
from naigos.rl.ppo import PPOConfig, init_learner
from naigos.rl.red_team import RedCurriculum
from naigos.rl.reward import RewardCurriculum, RewardWeights
from naigos.rl.train import TrainConfig, run

# Deliberately tiny. Two blue agents, two worlds, eight steps: not a training
# run, a determinism fixture.
ENV = EnvConfig(n_blue=2, n_threat=3, n_threat_active=3)
PPO = PPOConfig(n_envs=2, n_steps=8, n_minibatches=2)
ITERATIONS = 4
INTERRUPT_AT = 2


def _train_cfg(out, iterations: int) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        out_dir=str(out),
        seed=3,
        eval_every=2,
        eval_worlds=2,
        checkpoint_every=2,
        curriculum_every=2,
    )


def _learner_leaves(learner):
    """Every number that defines the learner: parameters, both optimizer states,
    the multiplier and the multiplier's optimizer state."""
    return jax.tree.leaves((
        learner.actor.params, learner.actor.opt_state, learner.actor.step,
        learner.critic.params, learner.critic.opt_state, learner.critic.step,
        learner.lam_raw, learner.lam_opt,
    ))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One uninterrupted run, and one interrupted at `INTERRUPT_AT` then resumed.

    Module-scoped because each of these is a real (if tiny) training run and
    every test in this file asks a different question about the same pair.
    """
    root = tmp_path_factory.mktemp("resume")
    straight_dir, broken_dir, again_dir = root / "straight", root / "broken", root / "again"

    straight, straight_history = run(ENV, PPO, _train_cfg(straight_dir, ITERATIONS))
    # The interruption: a run that was only ever going to reach INTERRUPT_AT,
    # which is what a container killed at that point leaves behind.
    run(ENV, PPO, _train_cfg(broken_dir, INTERRUPT_AT))
    # the same interrupted run, copied before either copy is resumed
    shutil.copytree(broken_dir, again_dir)
    resumed_ckpt = ck.latest_resumable(broken_dir)
    resumed, resumed_history = run(
        ENV, PPO, _train_cfg(broken_dir, ITERATIONS), resume_from=resumed_ckpt
    )
    again, again_history = run(
        ENV, PPO, _train_cfg(again_dir, ITERATIONS), resume_from=ck.latest_resumable(again_dir)
    )
    return {
        "straight": straight,
        "straight_history": straight_history,
        "straight_dir": straight_dir,
        "resumed": resumed,
        "resumed_history": resumed_history,
        "broken_dir": broken_dir,
        "resumed_ckpt": resumed_ckpt,
        "again": again,
        "again_history": again_history,
    }


# --- the headline claim ------------------------------------------------------


def _worst_difference(a, b) -> float:
    assert len(a) == len(b) and a
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b))


def test_the_interrupted_run_checkpoints_the_uninterrupted_runs_state(runs):
    """Up to the interruption the two runs are the same run, and the checkpoint
    a resume starts from holds all of it: parameters, both optimizer states, the
    multiplier, its optimizer state and the RNG stream."""
    a = ck.load(runs["straight_dir"] / f"ckpt_{INTERRUPT_AT:06d}.pkl")["recovery"]
    b = ck.load(runs["resumed_ckpt"])["recovery"]
    for key in ("actor_params", "actor_opt_state", "actor_step", "critic_params",
                "critic_opt_state", "critic_step", "lam_raw", "lam_opt_state"):
        worst = _worst_difference(jax.tree.leaves(a[key]), jax.tree.leaves(b[key]))
        assert worst == 0.0, f"{key} differs at the interruption by {worst}"
    np.testing.assert_array_equal(np.asarray(a["rng_key"]["data"]), np.asarray(b["rng_key"]["data"]))


def test_resuming_the_same_checkpoint_twice_reaches_the_same_learner(runs):
    """The rollout is re-initialised on resume from a seed-derived key, so the
    continuation is reproducible even though it is not the uninterrupted run's
    sample path (see the module docstring)."""
    worst = _worst_difference(_learner_leaves(runs["resumed"]), _learner_leaves(runs["again"]))
    assert worst == 0.0, f"two resumes of one checkpoint differ by {worst}"


def test_a_resumed_run_reports_the_same_metrics_where_it_can(runs):
    """Rows up to the interruption come off the checkpoint and match the
    uninterrupted run exactly; rows after it match a second resume exactly."""
    def rows(history):
        return {r["iter"]: r for r in history if r.get("phase") != "baseline"}

    a, b, c = rows(runs["straight_history"]), rows(runs["resumed_history"]), rows(runs["again_history"])
    assert set(a) == set(b) == set(c)
    for it in a:
        ref = a[it] if it <= INTERRUPT_AT else c[it]
        for key in ("reward", "cost", "lambda", "survival_rate", "shootdown_rate",
                    "objective_rate", "red_level"):
            assert ref[key] == pytest.approx(b[it][key], abs=0.0, rel=0.0), (it, key)


def test_the_resumed_run_continues_from_the_next_iteration(runs):
    iters = [r["iter"] for r in runs["resumed_history"]]
    assert iters == sorted(iters), "a resumed history must stay monotonic"
    assert iters[-1] == ITERATIONS
    assert rm.checkpoint_iteration(runs["resumed_ckpt"]) == INTERRUPT_AT


def test_resuming_carries_the_earlier_history_rather_than_truncating_it(runs):
    """`history.json` is rewritten from memory at every eval boundary. A resume
    that started with an empty list would silently discard the first half of the
    curve while leaving a file that looks complete."""
    a = [r["iter"] for r in runs["straight_history"]]
    b = [r["iter"] for r in runs["resumed_history"]]
    assert a == b


# --- what the resumed run says about itself ----------------------------------


def test_resumed_output_is_marked_as_resumed(runs):
    perf = rm.summarize_run(runs["broken_dir"])
    assert perf["resumed"] is True
    assert perf["resumed_from_iteration"] == INTERRUPT_AT
    rows = [r for r in runs["resumed_history"] if r["iter"] > INTERRUPT_AT]
    assert rows and all(r["resumed_from_iteration"] == INTERRUPT_AT for r in rows)


def test_an_uninterrupted_run_is_not_marked_as_resumed(runs):
    assert rm.summarize_run(runs["straight_dir"])["resumed"] is False


def test_the_resume_chain_records_every_continuation(runs):
    blob = ck.load(runs["broken_dir"] / f"ckpt_{ITERATIONS:06d}.pkl")
    chain = blob["recovery"]["resumed_from"]
    assert len(chain) == 1
    assert chain[0]["from_iteration"] == INTERRUPT_AT
    assert chain[0]["checkpoint"] == f"ckpt_{INTERRUPT_AT:06d}.pkl"


# --- what is actually in a checkpoint ----------------------------------------


def test_a_checkpoint_carries_the_complete_recovery_state(runs):
    blob = ck.load(runs["resumed_ckpt"])
    rec = blob["recovery"]
    for key in ck.REQUIRED_RECOVERY_KEYS:
        assert key in rec, key
    # the specific things whose absence would silently change training
    assert rec["actor_opt_state"] is not None and rec["critic_opt_state"] is not None
    assert rec["lam_opt_state"] is not None
    assert rec["rng_key"]["data"].shape != ()
    assert rec["red_level"] is not None and rec["shootdown_rate"] is not None
    assert rec["survival_w"] is not None and rec["efficiency_w"] is not None
    assert rec["history"]


def test_the_checkpoint_still_holds_the_policy_where_the_demo_looks_for_it(runs):
    """`naigos/demo/replay.py` and `naigos/demo/live.py` read
    `pickle.load(f)["actor"]`. Adding recovery state must not move it."""
    blob = pickle.loads(Path(runs["resumed_ckpt"]).read_bytes())
    assert "actor" in blob and "critic" in blob
    assert blob["iter"] == INTERRUPT_AT
    assert isinstance(blob["env_cfg"], dict)
    assert blob["env_cfg"]["n_blue"] == ENV.n_blue


def test_the_checkpoint_holds_no_device_handles(runs):
    """Pickling a live JAX array ties the file to the machine that wrote it.
    Everything is pulled back to numpy first."""
    blob = ck.load(runs["resumed_ckpt"])
    for leaf in jax.tree.leaves(blob["recovery"]["actor_params"]):
        assert isinstance(leaf, np.ndarray)


# --- refusing what cannot be resumed -----------------------------------------


def test_a_pre_resume_checkpoint_is_refused_rather_than_half_loaded(tmp_path):
    """The old format holds the policy and nothing else. Loading it and calling
    the result a resume would restart Adam, the multiplier's optimizer and the
    RNG stream while reporting a continuous run."""
    key = jax.random.PRNGKey(0)
    from naigos.env.flight_env import NaigosEnv

    env = NaigosEnv(ENV)
    _, sample_obs = env.reset(key)
    learner = init_learner(key, ENV, PPO, sample_obs)
    legacy = tmp_path / "ckpt_000010.pkl"
    legacy.write_bytes(pickle.dumps({
        "actor": jax.device_get(learner.actor.params),
        "critic": jax.device_get(learner.critic.params),
        "lam_raw": float(learner.lam_raw),
        "red_level": 0.0,
        "iter": 10,
        "env_cfg": dataclasses.asdict(ENV),
    }))
    assert ck.is_resumable(legacy) is False
    assert ck.latest_resumable(tmp_path) is None
    with pytest.raises(ck.CheckpointError, match="predates resume support"):
        ck.restore_learner(learner, ck.load(legacy))


def test_a_truncated_checkpoint_is_skipped_for_the_last_good_one(runs, tmp_path):
    """A container killed mid-write leaves a newest checkpoint that is a partial
    pickle. Selecting it would fail the resume; selecting nothing would throw
    away a run that was perfectly recoverable."""
    good = Path(runs["resumed_ckpt"]).read_bytes()
    (tmp_path / "ckpt_000002.pkl").write_bytes(good)
    (tmp_path / "ckpt_000004.pkl").write_bytes(good[: len(good) // 2])
    assert ck.latest_resumable(tmp_path).name == "ckpt_000002.pkl"


def test_a_checkpoint_whose_filename_disagrees_with_its_contents_is_rejected(runs, tmp_path):
    """`select_checkpoint` sorts on the filename, so a file named for one
    iteration and holding another makes "most recent" a lie."""
    misnamed = tmp_path / "ckpt_009999.pkl"
    misnamed.write_bytes(Path(runs["resumed_ckpt"]).read_bytes())
    assert ck.is_resumable(misnamed) is False


def test_a_changed_curriculum_is_reported_as_incompatible(runs):
    """Only a code change can move a curriculum threshold, and a resume across
    one splices two different tasks into one history."""
    blob = ck.load(runs["resumed_ckpt"])
    assert ck.curriculum_compatibility(
        blob, red_curriculum=RedCurriculum(), reward_curriculum=RewardCurriculum(),
        reward_weights_base=RewardWeights(),
    ) == []
    problems = ck.curriculum_compatibility(
        blob,
        red_curriculum=dataclasses.replace(RedCurriculum(), promote_survival=0.5),
        reward_curriculum=RewardCurriculum(),
        reward_weights_base=dataclasses.replace(RewardWeights(), arrived=400.0),
    )
    assert any("promote_survival" in p for p in problems)
    assert any("arrived" in p for p in problems)


def test_a_checkpoint_from_a_differently_shaped_network_is_refused(runs):
    """Grafting is structure-checked and shape-checked, so a network that changed
    between the checkpoint and the code loading it fails here rather than a
    thousand iterations later.

    `n_blue` is the change used because it is one that actually moves a
    parameter shape: the centralised critic sees the whole scene, so its input
    width is a function of the team size. (`n_threat` does not -- the actor and
    critic both encode threats through a shared per-threat MLP, which is why the
    red curriculum can change how many are active mid-run without a reinit.)"""
    from naigos.env.flight_env import NaigosEnv

    wider = dataclasses.replace(ENV, n_blue=ENV.n_blue + 2)
    key = jax.random.PRNGKey(0)
    env = NaigosEnv(wider)
    _, sample_obs = env.reset(key)
    template = init_learner(key, wider, PPO, sample_obs)
    with pytest.raises(ck.CheckpointError):
        ck.restore_learner(template, ck.load(runs["resumed_ckpt"]))


def test_resuming_a_checkpoint_already_at_the_declared_end_is_a_no_op(runs, tmp_path):
    final = tmp_path / "ckpt_000004.pkl"
    final.write_bytes((Path(runs["broken_dir"]) / f"ckpt_{ITERATIONS:06d}.pkl").read_bytes())
    learner, history = run(ENV, PPO, _train_cfg(tmp_path, ITERATIONS), resume_from=final)
    assert [r["iter"] for r in history][-1] == ITERATIONS
    assert _learner_leaves(learner)[0] is not None
