"""Visual events are derived from simulation state, never the other way round.

`naigos.demo.events` decides when the viewer may animate a lock, a masking or a
shootdown. The failure modes pinned here are the ones that turn a visualisation
into a fabrication: an effect with no state transition behind it, an effect
repeated after the aircraft is already gone, a replay that shows it at a
different frame each time, and a derivation that writes back into the state it
reads.
"""

from __future__ import annotations

import copy
import math

import jax
import numpy as np
import pytest

from naigos.demo import events
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv

LOCK_T = 0.6


def _step(der, s, *, in_flight=(1, 1), alive=(1, 1), shot=(0, 0), lock=None, expo=(0, 0),
          firing=None, masked=None):
    T, B = 3, 2
    lock = np.zeros((T, B)) if lock is None else np.asarray(lock, dtype=float)
    return der.step(
        step=s, t_s=2.0 * s, in_flight=np.array(in_flight, bool), alive=np.array(alive, bool),
        shotdown=np.array(shot, bool), lock=lock, exposure=np.array(expo, float),
        blue_pos=np.array([[0.0, 0.0, 500.0], [9000.0, 0.0, 500.0]]),
        threat_pos=np.array([[1000.0, 0, 0], [8000.0, 0, 0], [50000.0, 0, 0]]),
        firing=firing, kill_weight=None, masked=masked,
    )


# --- each event has a transition behind it ---------------------------------------------


def test_nothing_happens_nothing_is_emitted():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    for s in range(10):
        assert _step(der, s) == []


def test_detection_and_lock_fire_on_the_rising_edge_only():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    lock = np.array([[0.7, 0.0], [0.1, 0.0], [0.0, 0.0]])
    first = _step(der, 0, expo=(0.8, 0.1), lock=lock)
    assert [(e["type"], e["aircraft"]) for e in first] == [("detected", 0), ("lock_acquired", 0)]
    assert first[1]["threat"] == 0, "the lock is attributed to the threat that holds it"
    # held: nothing new
    for s in range(1, 5):
        assert _step(der, s, expo=(0.9, 0.1), lock=lock) == []
    # dropped, then regained: a new rising edge
    assert _step(der, 5, expo=(0.1, 0.1)) == []
    again = _step(der, 6, expo=(0.8, 0.1), lock=lock)
    assert [e["type"] for e in again] == ["detected", "lock_acquired"]


def test_every_event_cites_its_cause_and_is_marked_visual_only_when_it_is():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    firing = np.zeros((3, 2))
    firing[1, 1] = 1.0
    evs = _step(der, 0, shot=(0, 1), alive=(1, 0), firing=firing, expo=(0.9, 0.9),
                lock=np.full((3, 2), 0.9), masked=np.array([True, False]))
    assert {e["type"] for e in evs} == set(events.EVENT_TYPES)
    for e in evs:
        assert e["cause"] == events.CAUSES[e["type"]]
        assert e["visual_only"] is (e["type"] == "launch_visual")
    assert "no projectile or missile is simulated" in events.CAUSES["launch_visual"]


def test_a_kill_is_attributed_to_the_firing_threat_and_the_tracer_starts_there():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    firing = np.zeros((3, 2))
    firing[1, 1] = 1.0                       # only threat 1 has a firing solution on aircraft 1
    evs = _step(der, 3, shot=(0, 1), alive=(1, 0), firing=firing)
    kill = next(e for e in evs if e["type"] == "shot_down")
    tracer = next(e for e in evs if e["type"] == "launch_visual")
    assert kill["aircraft"] == 1 and kill["threat"] == 1
    assert tracer["threat"] == 1 and tracer["step"] == kill["step"]
    assert tracer["threat_pos"] == [8000.0, 0.0, 0.0]
    assert tracer["pos"] == kill["pos"] == [9000.0, 0.0, 500.0]


def test_a_kill_with_no_firing_solution_draws_no_tracer():
    """Nothing to draw a tracer from is not licence to invent a source."""
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    evs = _step(der, 0, shot=(1, 0), alive=(0, 1), firing=np.zeros((3, 2)))
    assert [e["type"] for e in evs] == ["shot_down"]
    assert evs[0]["threat"] is None


def test_attribution_prefers_the_largest_hazard_term_then_the_nearest():
    firing = np.array([[1.0], [1.0], [1.0]])
    blue = np.array([[0.0, 0.0, 0.0]])
    tp = np.array([[5000.0, 0, 0], [1000.0, 0, 0], [3000.0, 0, 0]])
    assert events.attribute_kill(firing, None, blue, tp, 0) == 1              # nearest
    assert events.attribute_kill(firing, [0.1, 0.02, 0.1], blue, tp, 0) == 2  # biggest term, then nearest
    assert events.attribute_kill(np.zeros((3, 1)), None, blue, tp, 0) is None


# --- no repeats after loss or retask ----------------------------------------------------


def test_nothing_more_is_emitted_for_a_sortie_that_was_lost():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    firing = np.zeros((3, 2))
    firing[0, 0] = 1.0
    hot = np.full((3, 2), 0.95)
    evs = _step(der, 0, shot=(1, 0), alive=(0, 1), firing=firing, lock=hot, expo=(1, 0))
    assert sum(e["type"] == "shot_down" and e["aircraft"] == 0 for e in evs) == 1
    # a caller that keeps saying the dead aircraft is in flight, shot, locked and
    # seen -- as a stale stream might -- still gets nothing for it
    for s in range(1, 8):
        later = _step(der, s, in_flight=(1, 1), shot=(1, 0), alive=(0, 1),
                      firing=firing, lock=hot, expo=(1, 0))
        assert not [e for e in later if e["aircraft"] == 0], later


def test_a_retask_opens_a_new_sortie_and_never_replays_the_old_one():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    firing = np.zeros((3, 2))
    firing[0, 0] = 1.0
    first = _step(der, 0, shot=(1, 0), alive=(0, 1), firing=firing)
    der.retask(np.array([True, False]))
    fresh = _step(der, 1, expo=(0.9, 0.0))
    assert [(e["type"], e["sortie"]) for e in fresh] == [("detected", 1)]
    old_ids = {e["id"] for e in first}
    assert not old_ids & {e["id"] for e in fresh}
    assert all(e["sortie"] == 0 for e in first)


def test_terrain_masking_is_a_rising_edge_too():
    der = events.EventDeriver(n_blue=2, lock_threshold=LOCK_T)
    m = np.array([True, False])
    assert [e["type"] for e in _step(der, 0, masked=m)] == ["terrain_masked"]
    assert _step(der, 1, masked=m) == []
    assert _step(der, 2, masked=np.array([False, False])) == []
    assert [e["type"] for e in _step(der, 3, masked=m)] == ["terrain_masked"]


# --- the turret's target ----------------------------------------------------------------


def test_track_selection_is_the_lock_matrix_argmax_above_the_hud_threshold():
    lock = np.array([[0.1, 0.5], [0.3, 0.2], [0.24, 0.0]])
    assert list(events.track_selection(lock, [True, True])) == [1, 0, -1]
    # an aircraft that is gone cannot be tracked
    assert list(events.track_selection(lock, [True, False])) == [-1, 0, -1]
    assert events.TRACK_VISUAL_MIN == 0.25


# --- against a real rollout -------------------------------------------------------------


@pytest.fixture(scope="module")
def trace():
    cfg = EnvConfig(n_blue=4, n_threat=10, n_threat_active=10, max_steps=150,
                    red_detect_scale=1.6, red_lethal_scale=2.5, red_latency_scale=0.3)
    env = NaigosEnv(cfg)

    def straight(obs, key):
        h = np.zeros(obs.ego.shape[0])
        return jax.numpy.stack([h, h - 0.2, h + 0.2], -1)

    _, traj = env.rollout(jax.random.PRNGKey(3), straight)
    tr = {k: np.asarray(v) for k, v in traj.items() if k != "terms"}
    tr["shotdown"] = np.asarray(traj["terms"].shotdown)
    tr["exposure"] = np.asarray(traj["terms"].exposure)
    return cfg, tr


def test_the_rollout_records_what_the_viewer_needs(trace):
    cfg, tr = trace
    S = cfg.max_steps
    for k, shape in (("gamma", (S, 4)), ("phi", (S, 4)), ("threat_psi", (S, 10)),
                     ("firing", (S, 10, 4))):
        assert tr[k].shape == shape, k


def test_replay_events_mirror_every_logged_shootdown_and_nothing_else(trace):
    cfg, tr = trace
    evs = events.replay_events(tr, lock_threshold=cfg.detection.lock_threshold, stride=2,
                               frame_dt=cfg.dt * 2)
    kills = [e for e in evs if e["type"] == "shot_down"]
    logged = [(int(s), int(b)) for s, b in zip(*np.nonzero(tr["shotdown"] > 0.5))]
    assert logged, "the fixture should produce at least one shootdown"
    assert sorted((e["step"], e["aircraft"]) for e in kills) == sorted(logged)
    for e in kills:
        s, b = e["step"], e["aircraft"]
        assert not tr["alive"][s, b], "the kill step is the step alive went False"
        assert s == 0 or tr["alive"][s - 1, b]
        if e["threat"] is not None:
            assert tr["firing"][s, e["threat"], b] > 0, "attributed to a threat with no firing solution"
        # the logged frame that shows it is the first recorded frame at or after it
        assert e["frame"] == math.ceil(s / 2)
        assert e["t_s"] == pytest.approx(e["frame"] * cfg.dt * 2)
    tracers = [e for e in evs if e["type"] == "launch_visual"]
    assert len(tracers) == sum(e["threat"] is not None for e in kills)


def test_replay_events_are_deterministic_and_leave_the_trace_untouched(trace):
    cfg, tr = trace
    before = copy.deepcopy(tr)
    a = events.replay_events(tr, lock_threshold=0.6, stride=2, frame_dt=4.0)
    b = events.replay_events(tr, lock_threshold=0.6, stride=2, frame_dt=4.0)
    assert a == b
    for k in before:
        assert np.array_equal(before[k], tr[k]), f"event derivation wrote into {k}"


def test_no_event_follows_a_loss_in_a_real_rollout(trace):
    cfg, tr = trace
    evs = events.replay_events(tr, lock_threshold=0.6, stride=2, frame_dt=4.0)
    for b in range(cfg.n_blue):
        dead = np.flatnonzero(~tr["alive"][:, b])
        if not dead.size:
            continue
        lost_at = int(dead[0])
        after = [e for e in evs if e["aircraft"] == b and e["step"] > lost_at]
        assert not after, f"aircraft {b} lost at step {lost_at} still generated {after}"


def test_event_ids_are_unique(trace):
    _, tr = trace
    evs = events.replay_events(tr, lock_threshold=0.6, stride=2, frame_dt=4.0)
    ids = [e["id"] for e in evs]
    assert len(ids) == len(set(ids))
