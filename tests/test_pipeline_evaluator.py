"""The real held-out evaluator, on a tiny synthetic env: it runs, it reproduces, it verifies.

Uses JAX on CPU and no theatre data. Slow-ish (one compile per policy), so sized
to the minimum that still exercises every branch the pipeline relies on.
"""

from __future__ import annotations

import pickle

import jax
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.pipeline import evaluation, promotion
from naigos.pipeline import config as pcfg
from naigos.rl.ppo import PPOConfig, init_learner

CFG = EnvConfig(n_blue=2, n_threat=4, n_threat_active=4, max_steps=40)
PLAN = {"seeds": [900001, 900002], "worlds_per_seed": 2, "red_level": 0.0, "use_cbf": False,
        "verifier_episodes": 1, "training_seeds": [0, 1]}


@pytest.fixture(scope="module")
def env_and_params():
    env = NaigosEnv(CFG)
    key = jax.random.PRNGKey(0)
    _, obs = env.reset(key)
    learner = init_learner(key, CFG, PPOConfig(n_envs=2, n_steps=4), obs)
    return env, learner.actor.params


@pytest.fixture(scope="module")
def measured(env_and_params):
    env, params = env_and_params
    return evaluation.run_evaluation(env, PLAN, params, champion_params=params)


def test_every_policy_is_measured_on_the_same_episodes(measured):
    assert measured["n_episodes"] == 4
    for block in (measured["candidate"], measured["baselines"]["avoid_nap"],
                  measured["baselines"]["direct"], measured["champion_metrics"]):
        for k in ("survival_rate", "objective_rate", "exposure_early", "shootdown_rate",
                  "terrain_rate", "bounds_rate"):
            assert 0.0 <= block[k] <= 1.0, k
    # same params as candidate and champion on the same keys -> identical numbers
    assert measured["candidate"] == measured["champion_metrics"]


def test_the_verifier_ran_on_the_candidate(measured):
    ver = measured["verifier"]
    assert ver["episodes"] == 1 and ver["ok"] is True and ver["mismatches"] == []


def test_a_re_run_reproduces_the_record_exactly(env_and_params, measured):
    env, params = env_and_params
    again = evaluation.run_evaluation(env, PLAN, params)
    rec = {"candidate": measured["candidate"], "baselines": measured["baselines"],
           "heldout": PLAN}
    fresh = {"candidate": again["candidate"], "baselines": again["baselines"], "heldout": PLAN}
    assert evaluation.compare_reproduction(rec, fresh, 0.0) == []


def test_different_heldout_seeds_are_different_episodes(env_and_params, measured):
    env, params = env_and_params
    other = evaluation.run_evaluation(env, {**PLAN, "seeds": [900003, 900004]}, params)
    assert other["baselines"]["direct"] != measured["baselines"]["direct"] or \
        other["candidate"] != measured["candidate"]


def test_an_incompatible_champion_is_recorded_not_fatal(env_and_params):
    env, params = env_and_params
    out = evaluation.run_evaluation(env, PLAN, params, champion_params={"not": "params"})
    assert "champion_error" in out and "champion_metrics" not in out


def test_the_real_record_feeds_the_gates(tmp_path, measured):
    ckpt = tmp_path / "ckpt_000002.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump({"actor": {}}, f)
    rec = evaluation.build_record(
        candidate_id="c-20260911-nightly-0123456789ab",
        snapshot_rec={"snapshot_id": "s-20260911-0123456789ab", "content": {"sha256": "x"},
                      "aoi": {"name": "owens_valley"}},
        plan=PLAN, env_shape={"n_blue": 2}, checkpoint=evaluation.checkpoint_entry(ckpt),
        measured=measured, champion=None, code={}, config_digest="d", config_version=1)
    out = promotion.decide(rec, pcfg.load_default()["gates"], provenance_problems=[])
    # An untrained policy: the verdict is whatever the numbers say, but every
    # gate has a real input -- nothing is unknown.
    assert out["outcome"] in (promotion.ELIGIBLE, promotion.REJECTED)
    assert all(c["status"] != promotion.UNKNOWN for c in out["checks"]), out["checks"]
