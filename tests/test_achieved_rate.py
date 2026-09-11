"""The live stream's measured rate: sim seconds per wall second, labelled "achieved".

`--speed` is what the operator asked for; `live.AchievedRate` is what the worker
actually managed, over a rolling wall-clock window. Offline, no browser:

* the calculation, on a fake clock -- on pace, falling behind, a regime change
  inside and after the window, and nothing claimed before there is data;
* the wiring -- a running live Simulation publishes it in every frame, beside
  the requested rate and never in place of it;
* the separation -- a recording's replay payload carries no stream rate at all.
"""

from __future__ import annotations

import json
import threading
import time

import jax
import numpy as np
import pytest

from naigos.data.geodetic import GeoRef
from naigos.demo import live
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.networks import Actor
from naigos.rl.ppo import greedy_policy


def _feed(meter: live.AchievedRate, steps: int, dt_sim: float, dt_wall: float,
          t0: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    sim, wall = t0
    for _ in range(steps):
        sim += dt_sim
        wall += dt_wall
        meter.record(sim, wall)
    return sim, wall


# --- the calculation ---------------------------------------------------------------------


def test_nothing_is_claimed_before_there_is_anything_to_measure():
    m = live.AchievedRate(window_s=5.0)
    assert m.rate() is None and m.span_s() == 0.0
    m.record(0.0, 100.0)
    assert m.rate() is None, "one sample is a point, not a rate"
    m.record(0.5, 100.0)
    assert m.rate() is None, "no wall time elapsed: no rate, not infinity"
    d = m.as_dict(10.0)
    assert d["label"] == "achieved"
    assert d["achieved_sim_s_per_wall_s"] is None and d["fraction_of_requested"] is None
    assert d["requested_sim_s_per_wall_s"] == 10.0


def test_on_pace_the_achieved_rate_equals_the_requested_one():
    m = live.AchievedRate(window_s=5.0)
    m.record(0.0, 0.0)
    _feed(m, 400, dt_sim=0.5, dt_wall=0.05)          # 0.5 sim s every 50 ms = 10x
    assert m.rate() == pytest.approx(10.0)
    d = m.as_dict(10.0)
    assert d["achieved_sim_s_per_wall_s"] == pytest.approx(10.0)
    assert d["fraction_of_requested"] == pytest.approx(1.0)


def test_falling_behind_is_measured_not_assumed():
    # asked for 10x, but each 0.5 s step takes 100 ms of wall clock: 5x
    m = live.AchievedRate(window_s=5.0)
    m.record(0.0, 0.0)
    _feed(m, 200, dt_sim=0.5, dt_wall=0.1)
    d = m.as_dict(10.0)
    assert d["achieved_sim_s_per_wall_s"] == pytest.approx(5.0)
    assert d["requested_sim_s_per_wall_s"] == 10.0
    assert d["fraction_of_requested"] == pytest.approx(0.5)


def test_the_window_rolls_and_the_old_regime_ages_out():
    m = live.AchievedRate(window_s=2.0)
    m.record(0.0, 0.0)
    t = _feed(m, 100, dt_sim=0.5, dt_wall=0.05)      # 5 s at 10x
    assert m.rate() == pytest.approx(10.0)
    # a stall: the next steps take four times as long (2.5x)
    t = _feed(m, 5, dt_sim=0.5, dt_wall=0.2, t0=t)    # 1 s into a 2 s window
    mixed = m.rate()
    assert 2.5 < mixed < 10.0, "half a window in, the rate is between the two regimes"
    _feed(m, 20, dt_sim=0.5, dt_wall=0.2, t0=t)       # well past the window
    assert m.rate() == pytest.approx(2.5), "the old regime has aged out entirely"


def test_a_warmed_up_meter_always_spans_the_whole_window_and_stays_bounded():
    m = live.AchievedRate(window_s=1.0)
    m.record(0.0, 0.0)
    _feed(m, 1000, dt_sim=0.5, dt_wall=0.01)
    assert 1.0 <= m.span_s() < 1.0 + 0.01 + 1e-9
    assert len(m._samples) <= 1.0 / 0.01 + 2, "memory is bounded by the window, not the run"


def test_a_long_single_step_still_yields_a_rate():
    # one step longer than the whole window: the meter keeps two samples and
    # reports that step's rate rather than going blank
    m = live.AchievedRate(window_s=1.0)
    m.record(0.0, 0.0)
    m.record(0.5, 3.0)
    assert m.rate() == pytest.approx(0.5 / 3.0)


# --- the wiring --------------------------------------------------------------------------

CFG = EnvConfig(n_blue=2, n_threat=4, n_threat_active=3, max_steps=60)
NOTES = {"georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0,
                    "origin_northing_m": 3_950_000.0}}


def _sim(speed: float) -> live.Simulation:
    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    return live.Simulation(env=env, policy=greedy_policy(params, CFG),
                           georef=GeoRef(**NOTES["georef"]), speed=speed, reroll_s=0.0)


def test_a_running_live_stream_publishes_the_achieved_rate_beside_the_requested_one():
    # Requested: effectively unbounded. Achieved: whatever the machine manages,
    # which is finite -- so the two must differ, and the frame must say which is which.
    sim = _sim(speed=1e9)
    first = sim.latest()["live_rate"]
    assert first["label"] == "achieved" and first["achieved_sim_s_per_wall_s"] is None

    worker = threading.Thread(target=sim.run, daemon=True)
    worker.start()
    deadline = time.monotonic() + 60.0
    while sim.tick < 15 and time.monotonic() < deadline:
        time.sleep(0.05)
    sim.stop()
    worker.join(timeout=30.0)
    assert sim.tick >= 15, "the worker did not step"

    frame = sim.latest()
    r = frame["live_rate"]
    assert r["label"] == "achieved"
    assert r["requested_sim_s_per_wall_s"] == 1e9
    assert 0.0 < r["achieved_sim_s_per_wall_s"] < 1e9
    assert r["fraction_of_requested"] < 1.0
    assert r["window_wall_s"] > 0.0
    # the meter measured the same sim clock the frame reports
    assert sim.achieved._samples[-1][0] == pytest.approx(sim.sim_t)
    json.dumps(frame)  # still a plain JSON frame for the SSE stream


def test_the_meter_is_fed_state_it_cannot_change():
    # AchievedRate holds floats copied out of the loop and exposes no path back
    # into the Simulation: the step, dt, sleep schedule and state are untouched.
    sim = _sim(speed=10.0)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(jax.device_get(sim.state))]
    sim.achieved.record(123.0, 456.0)
    sim.achieved.record(124.0, 456.5)
    sim.achieved.as_dict(sim.speed)
    after = jax.tree.leaves(jax.device_get(sim.state))
    assert all(np.array_equal(a, np.asarray(b)) for a, b in zip(before, after))
    assert sim.sim_t == 0.0 and sim.speed == 10.0


# --- the separation from replay ------------------------------------------------------------


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    from naigos.demo import replay

    env = NaigosEnv(CFG)
    _, o = env.reset(jax.random.PRNGKey(0))
    params = Actor(CFG).init(jax.random.PRNGKey(1), o.ego, o.threats, o.threat_mask,
                             o.friends, o.friend_mask)
    results = replay.run(env, params, n_worlds=2, seed=3)
    notes = {"theatre": "test_basin", **NOTES}
    georef = GeoRef(**notes["georef"])
    lon, lat = georef.to_wgs84(np.array([0.0, CFG.terrain.extent_x]),
                               np.array([0.0, CFG.terrain.extent_y]))
    notes["geo_bounds"] = {"west": float(lon[0]), "east": float(lon[1]),
                           "south": float(lat[0]), "north": float(lat[1])}
    out = tmp_path_factory.mktemp("ratedemo") / "demo.json"
    replay.to_json(results, CFG, out, env=env, notes=notes)
    return out


def test_a_replay_carries_no_stream_rate(recording):
    payload = live.static_payload(recording)
    scene = payload["/scene"]
    assert scene["mode"] == "replay" and scene["speed"] is None
    blob = json.dumps(payload)
    assert "live_rate" not in blob and "achieved_sim_s_per_wall_s" not in blob
