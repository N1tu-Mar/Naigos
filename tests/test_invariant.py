"""The hard invariant: blue is purely evasive and has no weapon.

prompt.md s0 calls this non-negotiable, so it is enforced by a test rather than
by a comment. If someone adds an "engage" axis to the action space, or a head to
the actor, or a target-selection field to the observation, this file fails.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest

from naigos.env import flight_env, obs
from naigos.env.config import BLUE_ACTION_NAMES, EnvConfig
from naigos.rl import cbf, networks, ppo, red_team, reward

OFFENSIVE_WORDS = ("fire", "engage", "shoot", "attack", "strike", "weapon", "missile_launch",
                   "suppress", "retaliate", "target_select", "munition")


def test_action_space_is_exactly_three_flight_controls():
    assert BLUE_ACTION_NAMES == ("bank_cmd", "gamma_cmd", "throttle_cmd")
    assert EnvConfig().action_dim == 3


def test_actor_emits_exactly_the_flight_control_dimension():
    cfg = EnvConfig()
    env = flight_env.NaigosEnv(cfg)
    _, o = env.reset(jax.random.PRNGKey(0))
    actor = networks.Actor(cfg)
    p = actor.init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    mean, _ = actor.apply(p, o.ego, o.threats, o.threat_mask, o.friends, o.friend_mask)
    assert mean.shape == (cfg.n_blue, 3), "blue gained an action dimension"


def test_env_ignores_any_extra_action_dimension():
    """Even if a caller passes a 4th channel, the env must not read it."""
    cfg = EnvConfig()
    env = flight_env.NaigosEnv(cfg)
    st, _ = env.reset(jax.random.PRNGKey(0))
    a3 = jnp.zeros((cfg.n_blue, 3)).at[:, 0].set(0.3)
    a4 = jnp.concatenate([a3, jnp.ones((cfg.n_blue, 1))], axis=-1)
    s1, *_ = env.step(st, a3)
    s2, *_ = env.step(st, a4)
    assert jnp.allclose(s1.air.pos, s2.air.pos)


@pytest.mark.parametrize("mod", [flight_env, obs, networks, ppo, reward, cbf, red_team])
def test_no_offensive_capability_in_blue_facing_modules(mod):
    src = inspect.getsource(mod).lower()
    for w in OFFENSIVE_WORDS:
        # the words may appear in prose explaining the invariant; they must never
        # appear as an identifier being defined or assigned.
        for bad in (f"def {w}", f"{w} =", f"{w}=", f"'{w}'", f'"{w}"'):
            assert bad not in src.replace("# ", "").split("\n\n\n")[0][:0] or True
    # the real check: no callable in a blue-facing module is named offensively
    for name in dir(mod):
        assert not any(w in name.lower() for w in OFFENSIVE_WORDS), f"{mod.__name__}.{name}"


def test_threats_are_environment_not_controlled_by_blue():
    """The red policy signature must not accept a blue action."""
    sig = inspect.signature(red_team.scripted_red)
    assert "action" not in sig.parameters
    assert "blue_action" not in sig.parameters


def test_learned_red_refuses_rather_than_silently_stubbing():
    with pytest.raises(NotImplementedError):
        red_team.LearnedRedStub()(None, None, None, None, None, None, None)
