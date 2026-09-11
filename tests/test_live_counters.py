"""Per-layout tallies in the live viewer's counters.

The cumulative success rate is sortie-weighted across threat layouts: a bad
layout kills aircraft fast, launches more sorties and dominates the number.
These tests pin the per-layout bookkeeping that is published beside it --
unit-level, no server, no browser.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from naigos.data.geodetic import GeoRef
from naigos.demo import live
from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv

NOTES = {"georef": {"utm_epsg": 32639, "origin_easting_m": 500_000.0,
                    "origin_northing_m": 3_950_000.0}}


def _two_layouts() -> live.Counters:
    c = live.Counters()
    c.record(reached=30)                        # layout 1: 30 reached of 30 finished
    c.tick(600.0)
    c.close_layout()
    c.record(reached=1, shot_down=70, terrain=2, bounds=1)   # layout 2: 1 of 74
    c.tick(600.0)
    c.close_layout()
    return c


def test_layout_mean_is_layout_weighted_and_per_sortie_rate_is_unchanged():
    d = _two_layouts().as_dict()
    # the existing, sortie-weighted number: 31 / 104
    assert d["success_rate"] == pytest.approx(31 / 104, abs=1e-3)
    assert d["success_rate"] == pytest.approx(0.298, abs=1e-3)
    # the layout-weighted one: mean(30/30, 1/74)
    assert d["layout_mean_success"] == pytest.approx(0.507, abs=1e-3)
    assert d["layout_success_min"] == pytest.approx(0.0135, abs=1e-4)
    assert d["layout_success_max"] == pytest.approx(1.0)
    assert d["layouts_completed"] == 2
    assert d["layouts_in_mean"] == 2


def test_per_layout_tallies_are_published():
    d = _two_layouts().as_dict()
    first, second = d["per_layout"]
    assert first == {"layout": 1, "reached": 30, "shot_down": 0, "terrain": 0, "bounds": 0,
                     "fuel": 0, "finished": 30, "sim_seconds": 600.0, "success_rate": 1.0}
    assert second["layout"] == 2
    assert (second["reached"], second["shot_down"], second["terrain"], second["bounds"],
            second["finished"]) == (1, 70, 2, 1, 74)
    assert second["success_rate"] == pytest.approx(1 / 74, abs=1e-4)
    # a fresh tally is open for the third layout
    cur = d["current_layout"]
    assert cur["layout"] == 3 and cur["finished"] == 0 and cur["success_rate"] is None


def test_existing_keys_and_values_are_unchanged():
    c = live.Counters(sorties=6)
    c.record(reached=2, shot_down=3, terrain=1, bounds=1, fuel=1)
    d = c.as_dict()
    assert {k: d[k] for k in ("sorties_launched", "objectives_reached", "shot_down",
                              "terrain_losses", "bounds_losses", "fuel_losses",
                              "success_rate")} == {
        "sorties_launched": 6, "objectives_reached": 2, "shot_down": 3, "terrain_losses": 1,
        "bounds_losses": 1, "fuel_losses": 1, "success_rate": 0.25}
    assert (c.sorties, c.reached, c.shot_down, c.terrain, c.bounds, c.fuel) == (6, 2, 3, 1, 1, 1)


def test_a_layout_with_no_finished_sortie_is_excluded_from_the_mean():
    c = _two_layouts()
    c.tick(5.0)
    c.close_layout()                            # layout 3: nothing finished
    d = c.as_dict()
    assert d["layouts_completed"] == 3
    assert d["layouts_in_mean"] == 2
    assert d["per_layout"][-1]["finished"] == 0
    assert d["per_layout"][-1]["success_rate"] is None
    assert d["layout_mean_success"] == pytest.approx(0.507, abs=1e-3)
    assert d["layout_success_min"] == pytest.approx(0.0135, abs=1e-4)


def test_no_completed_layout_publishes_none():
    d = live.Counters().as_dict()
    assert d["layouts_completed"] == 0
    assert d["per_layout"] == []
    assert d["layout_mean_success"] is None
    assert d["layout_success_min"] is None and d["layout_success_max"] is None


def test_per_layout_keeps_only_the_last_20():
    c = live.Counters()
    for i in range(25):
        c.record(reached=i % 2, shot_down=1 - i % 2)
        c.close_layout()
    d = c.as_dict()
    assert d["layouts_completed"] == 25
    assert [t["layout"] for t in d["per_layout"]] == list(range(6, 26))
    # the mean still covers every completed layout, not just the last 20
    assert d["layout_mean_success"] == pytest.approx(12 / 25, abs=1e-4)


# --- Simulation.run: attribution on the reroll step -----------------------------------------


CFG = EnvConfig(n_blue=3, n_threat=8, n_threat_active=6, max_steps=120)


def _terms(arrived=0, shot=0):
    z = np.zeros(CFG.n_blue)
    a, s = z.copy(), z.copy()
    a[:arrived] = 1.0
    s[:shot] = 1.0
    return SimpleNamespace(shotdown=s, terrain_violation=z, bounds_violation=z,
                           out_of_fuel=z, arrived=a, exposure=z)


def test_an_outcome_on_the_reroll_step_belongs_to_the_layout_active_during_it():
    """Step 3 crosses the reroll time and has one arrival; step 4 has one shootdown.
    The arrival is layout 1's, even though layout 2 is drawn at the end of that step."""
    dt = CFG.dt
    sim = live.Simulation(env=NaigosEnv(CFG), policy=None, georef=GeoRef(**NOTES["georef"]),
                          speed=1e9, reroll_s=3 * dt)
    script = {3: _terms(arrived=1), 4: _terms(shot=1)}
    step_no = {"n": 0}

    def fake_step(state, obs, key):
        step_no["n"] += 1
        info = {"agent_done": np.zeros(CFG.n_blue, dtype=bool)}
        return (state, obs, script.get(step_no["n"], _terms()), info,
                np.ones(CFG.n_blue, dtype=bool), key)

    def fake_publish(feasible, exposure, retasked=None):
        if step_no["n"] >= 5:
            sim.stop()

    sim._step = fake_step
    sim._derive_events = lambda *a: None
    sim._publish = fake_publish
    sim.run()

    d = sim.counters.as_dict()
    assert sim.threat_draws == 2
    assert d["layouts_completed"] == 1
    first = d["per_layout"][0]
    assert first["layout"] == 1
    assert (first["reached"], first["shot_down"], first["finished"]) == (1, 0, 1)
    assert first["sim_seconds"] == pytest.approx(3 * dt)
    cur = d["current_layout"]
    assert cur["layout"] == 2
    assert (cur["reached"], cur["shot_down"], cur["finished"]) == (0, 1, 1)
    assert cur["sim_seconds"] == pytest.approx(2 * dt)
    # and the cumulative counters saw both, exactly as before
    assert (d["objectives_reached"], d["shot_down"]) == (1, 1)
