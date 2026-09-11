"""The sweep has to be fair before it can be useful.

Every test here is about one of two claims the result file makes: that the grid
it names is the grid it flew, and that every cell flew the SAME scenarios. A
sweep that quietly reseeded between cells would still produce a table, and the
table would be noise with a margin column next to it.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from naigos.env.config import EnvConfig
from naigos.env.flight_env import NaigosEnv
from naigos.rl.cbf import CBFConfig
from naigos.rl.cbf_sweep import (
    ARMS,
    DELTA_KEYS,
    METRIC_KEYS,
    SCHEMA_VERSION,
    GridCell,
    SweepConfigError,
    SweepGrid,
    canonical_json,
    cell_cbf_config,
    cell_env_config,
    env_config_from_blob,
    format_table,
    load_result,
    markdown_summary,
    require_actor_compatible,
    run_sweep,
    seed_keys,
    seed_plan,
    serialize,
    validate_seeds,
    write_result,
)
from naigos.rl.red_team import RedCurriculum

CKPT = Path("checkpoints/theatre_1000.pkl")

# A deliberately tiny world. The sweep's correctness is about pairing and
# bookkeeping, not about statistics, so the tests buy shape and identity checks
# rather than sample size.
TINY = EnvConfig(n_blue=2, n_threat=8, n_threat_active=4, max_steps=12)


def straight_policy(cfg):
    """A fixture policy: hold wings level and fly at the objective heading.

    Cheap, deterministic and free of learned parameters, so a full-grid sweep
    runs in seconds and the assertions are about the sweep rather than about a
    checkpoint being present.
    """

    def pol(obs, key):
        herr = jnp.arctan2(obs.ego[:, 4], obs.ego[:, 5])
        return jnp.stack(
            [jnp.clip(herr * 2.0, -1, 1), jnp.zeros_like(herr), jnp.full_like(herr, 0.6)], -1
        )

    return pol


FIXTURE_PROV = {"label": "straight_policy", "source": "tests/test_cbf_sweep.py"}


def _tiny_sweep(**kw):
    params = dict(
        grid=SweepGrid(margins=(0.0, 1_500.0), threat_counts=(4,), red_levels=(0.3,)),
        make_policy=straight_policy,
        policy_provenance=FIXTURE_PROV,
        base_env_cfg=TINY,
        seeds=(101, 102),
        n_steps=8,
    )
    params.update(kw)
    return run_sweep(**params)


# --- grid expansion and validation -------------------------------------------


def test_grid_expands_to_the_full_product_sorted_by_margin():
    grid = SweepGrid(margins=(1_500.0, 0.0), threat_counts=(12, 6), red_levels=(0.6, 0.2))
    cells = grid.cells()
    assert len(cells) == 8 == grid.as_dict()["n_cells"]
    assert len(set(c.key for c in cells)) == 8
    # sorted by margin first, so the report's ordering is a property of the grid
    assert [c.margin for c in cells] == sorted(c.margin for c in cells)
    assert cells[0] == GridCell(margin=0.0, n_threat_active=6, red_level=0.2)


def test_axis_order_does_not_change_the_expansion():
    a = SweepGrid(margins=(0.0, 1_500.0), threat_counts=(6, 12), red_levels=(0.2, 0.6)).cells()
    b = SweepGrid(margins=(1_500.0, 0.0), threat_counts=(12, 6), red_levels=(0.6, 0.2)).cells()
    assert a == b


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(margins=()), "margins is empty"),
        (dict(threat_counts=()), "threat_counts is empty"),
        (dict(red_levels=()), "red_levels is empty"),
        (dict(margins=(-1.0,)), "negative"),
        (dict(margins=(float("nan"),)), "not finite"),
        (dict(red_levels=(1.5,)), "outside [0, 1]"),
        (dict(red_levels=(-0.1,)), "outside [0, 1]"),
        (dict(threat_counts=(-2,)), "negative"),
        (dict(margins=(0.0, 0.0)), "duplicates"),
        (dict(threat_counts=(6, 6)), "duplicates"),
    ],
)
def test_invalid_grids_raise_explicitly(kw, needle):
    base = dict(margins=(0.0,), threat_counts=(4,), red_levels=(0.3,))
    base.update(kw)
    with pytest.raises(SweepConfigError) as e:
        SweepGrid(**base).validate()
    assert needle in str(e.value)


def test_threat_count_above_the_padding_capacity_is_refused():
    grid = SweepGrid(margins=(0.0,), threat_counts=(4, 99), red_levels=(0.3,))
    grid.validate()  # no cfg: the capacity is unknown, so nothing to check
    with pytest.raises(SweepConfigError) as e:
        grid.validate(TINY)
    assert "n_threat=8" in str(e.value)


def test_a_cell_over_capacity_is_refused_at_config_time():
    with pytest.raises(SweepConfigError):
        cell_env_config(TINY, GridCell(0.0, 99, 0.3), RedCurriculum())


def test_run_sweep_needs_exactly_one_policy_source():
    with pytest.raises(SweepConfigError) as e:
        run_sweep(base_env_cfg=TINY)
    assert "exactly one" in str(e.value)
    with pytest.raises(SweepConfigError):
        run_sweep(base_env_cfg=TINY, checkpoint=CKPT, make_policy=straight_policy)


def test_a_callable_policy_without_provenance_is_refused():
    with pytest.raises(SweepConfigError) as e:
        run_sweep(base_env_cfg=TINY, make_policy=straight_policy)
    assert "policy_provenance" in str(e.value)


def test_a_missing_checkpoint_is_refused_before_anything_is_flown():
    with pytest.raises(SweepConfigError) as e:
        run_sweep(base_env_cfg=TINY, checkpoint=Path("checkpoints/does-not-exist.pkl"))
    assert "not found" in str(e.value)


def test_density_is_applied_after_the_curriculum_not_before():
    """`RedCurriculum.apply` writes n_threat_active from the level. If the sweep
    applied it last the density axis would collapse into the red axis."""
    cur = RedCurriculum()
    for level in (0.0, 0.5, 1.0):
        cfg = cell_env_config(TINY, GridCell(0.0, 5, level), cur)
        assert cfg.n_threat_active == 5
        assert cfg.red_detect_scale == pytest.approx(cur.apply(TINY, level).red_detect_scale)


def test_only_the_margin_moves_between_cbf_configs():
    base = CBFConfig()
    got = cell_cbf_config(base, GridCell(4_200.0, 6, 0.2))
    assert got.margin == 4_200.0
    assert {k: v for k, v in vars(got).items() if k != "margin"} == {
        k: v for k, v in vars(base).items() if k != "margin"
    }


# --- the seed plan ------------------------------------------------------------


def test_seed_plan_is_contiguous_explicit_and_validated():
    assert seed_plan(20_000, 4) == (20_000, 20_001, 20_002, 20_003)
    with pytest.raises(SweepConfigError):
        seed_plan(0, 0)
    with pytest.raises(SweepConfigError):
        seed_plan(-1, 4)
    with pytest.raises(SweepConfigError):
        validate_seeds(())
    with pytest.raises(SweepConfigError):
        validate_seeds((7, 7))
    with pytest.raises(SweepConfigError):
        validate_seeds((-3,))


def test_seed_keys_are_a_pure_function_of_the_seed_list():
    a = seed_keys((1, 2, 3))
    b = seed_keys((1, 2, 3))
    assert a.shape == (3, 2)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a)[0], np.asarray(seed_keys((9,)))[0])


def test_every_cell_and_both_arms_declare_the_same_seed_set():
    payload = _tiny_sweep(
        grid=SweepGrid(margins=(0.0, 1_500.0), threat_counts=(3, 4), red_levels=(0.3,))
    )
    assert payload["seeds"]["values"] == [101, 102]
    assert payload["seeds"]["shared_across_cells"] is True
    assert payload["seeds"]["shared_across_arms"] is True
    assert len(payload["cells"]) == 4


def test_cells_that_differ_only_in_margin_are_flown_on_identical_worlds():
    """The strong form of the pairing claim: the margin is a filter parameter
    and never reaches `reset`, so two such cells must agree on the env config
    they recorded, down to every field."""
    payload = _tiny_sweep()
    by_margin = {r["cell"]["margin"]: r for r in payload["cells"]}
    assert set(by_margin) == {0.0, 1_500.0}
    assert by_margin[0.0]["env_config"] == by_margin[1_500.0]["env_config"]
    assert by_margin[0.0]["cbf_config"]["margin"] == 0.0
    assert by_margin[1_500.0]["cbf_config"]["margin"] == 1_500.0
    # and the unfiltered control is the same run in both, since the filter is
    # what the margin lives in
    assert by_margin[0.0]["arms"]["no_cbf"] == by_margin[1_500.0]["arms"]["no_cbf"]


def test_raising_the_density_adds_threats_without_moving_the_existing_ones():
    """Why a density column is still a paired comparison: `threats.spawn` draws
    all n_threat positions and masks with `arange(T) < n_threat_active`."""
    key = jax.random.PRNGKey(101)
    lo = NaigosEnv(TINY.replace(n_threat_active=3)).reset(key)[0]
    hi = NaigosEnv(TINY.replace(n_threat_active=6)).reset(key)[0]
    assert np.allclose(np.asarray(lo.threats.pos), np.asarray(hi.threats.pos))
    assert np.allclose(np.asarray(lo.objective), np.asarray(hi.objective))
    assert np.allclose(np.asarray(lo.hmap), np.asarray(hi.hmap))
    assert int(np.asarray(lo.threats.active).sum()) == 3
    assert int(np.asarray(hi.threats.active).sum()) == 6


# --- result schema ------------------------------------------------------------


def test_both_arms_report_the_full_metric_schema():
    payload = _tiny_sweep()
    assert payload["metric_keys"] == list(METRIC_KEYS)
    for row in payload["cells"]:
        assert set(row["arms"]) == set(ARMS)
        for arm in ARMS:
            assert set(row["arms"][arm]) == set(METRIC_KEYS)
            assert all(isinstance(v, float) for v in row["arms"][arm].values())
        assert set(row["delta"]) == set(DELTA_KEYS)
        for k in DELTA_KEYS:
            assert row["delta"][k] == pytest.approx(
                row["arms"]["cbf"][k] - row["arms"]["no_cbf"][k]
            )


def test_the_required_outcome_rates_are_all_present():
    """The brief names these by hand; assert them by hand so a rename upstream
    cannot quietly drop one."""
    required = {
        "survival_rate",
        "objective_rate",
        "shootdown_rate",
        "terrain_rate",
        "bounds_rate",
        "timeout_rate",
        "fuel_loss_rate",
        "cbf_infeasible_rate",
        "mean_exposure",
        "exposure_early",
        "exposure_successful",
        "cumulative_exposure_per_sortie",
    }
    assert required <= set(METRIC_KEYS)
    row = _tiny_sweep()["cells"][0]
    assert required <= set(row["arms"]["cbf"])


def test_the_no_cbf_arm_can_be_switched_off_and_the_schema_says_so():
    payload = _tiny_sweep(include_no_cbf=False)
    assert payload["rollout"]["arms"] == ["cbf"]
    for row in payload["cells"]:
        assert set(row["arms"]) == {"cbf"}
        assert "delta" not in row


def test_the_infeasible_rate_is_surfaced_not_hidden():
    """A margin wide enough that the half-spaces cannot all be satisfied must
    show up as a number in the row, not as a silently clipped action."""
    payload = _tiny_sweep(
        grid=SweepGrid(margins=(0.0, 400_000.0), threat_counts=(4,), red_levels=(1.0,))
    )
    rates = {r["cell"]["margin"]: r["arms"]["cbf"]["cbf_infeasible_rate"] for r in payload["cells"]}
    assert set(rates) == {0.0, 400_000.0}
    assert all(0.0 <= v <= 1.0 for v in rates.values())
    # an envelope inflated past the map cannot be flown out of
    assert rates[400_000.0] > 0.0
    # and the unfiltered arm reports 0.0 by construction, which is a fact about
    # there being no filter -- not evidence that the filter was feasible
    assert all(r["arms"]["no_cbf"]["cbf_infeasible_rate"] == 0.0 for r in payload["cells"])


def test_the_fuel_rate_is_a_per_sortie_fraction_not_a_time_sum():
    """Running dry does not kill the aircraft, so the flag stays raised for the
    rest of the episode. Summing it over steps would exceed 1.0."""
    for row in _tiny_sweep()["cells"]:
        for arm in ARMS:
            assert 0.0 <= row["arms"][arm]["fuel_loss_rate"] <= 1.0


# --- provenance ---------------------------------------------------------------


def test_the_payload_carries_grid_seeds_config_commit_and_timestamp():
    payload = _tiny_sweep(notes="unit test")
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["kind"] == "cbf_calibration_sweep"
    assert payload["notes"] == "unit test"
    assert payload["created_utc"].endswith("+00:00")
    assert payload["grid"]["margins"] == [0.0, 1_500.0]
    assert payload["seeds"]["values"] == [101, 102]
    assert set(payload["code"]) == {"commit", "dirty"}
    assert payload["policy"] == {"kind": "callable", **FIXTURE_PROV}
    # the COMPLETE configuration, not a summary of it
    assert payload["base_cbf_config"]["margin"] == CBFConfig().margin
    assert set(payload["base_cbf_config"]) == {f for f in vars(CBFConfig())}
    assert payload["base_env_config"]["n_threat"] == TINY.n_threat
    assert payload["base_env_config"]["airframe"]["v_stall"] == TINY.airframe.v_stall
    assert payload["red_curriculum"]["promote_survival"] == RedCurriculum().promote_survival
    assert payload["rollout"]["sorties_per_arm"] == 2 * TINY.n_blue


def test_every_cell_records_the_configuration_it_was_actually_flown_with():
    payload = _tiny_sweep(
        grid=SweepGrid(margins=(0.0,), threat_counts=(3, 5), red_levels=(0.1, 0.9))
    )
    for row in payload["cells"]:
        c = row["cell"]
        assert row["env_config"]["n_threat_active"] == c["n_threat_active"]
        assert row["cbf_config"]["margin"] == c["margin"]
        expected = RedCurriculum().apply(TINY, c["red_level"])
        assert row["env_config"]["red_detect_scale"] == pytest.approx(expected.red_detect_scale)
        assert row["env_config"]["red_lethal_scale"] == pytest.approx(expected.red_lethal_scale)


@pytest.mark.skipif(not CKPT.exists(), reason="no shipped checkpoint")
def test_checkpoint_provenance_identifies_the_artefact():
    payload = _tiny_sweep(
        checkpoint=CKPT,
        make_policy=None,
        policy_provenance=None,
        base_env_cfg=None,
        env_overrides={"n_blue": 2, "n_threat": 8, "max_steps": 12},
        grid=SweepGrid(margins=(1_500.0,), threat_counts=(4,), red_levels=(0.3,)),
    )
    prov = payload["policy"]
    assert prov["kind"] == "checkpoint"
    assert prov["path"] == str(CKPT)
    assert len(prov["sha256"]) == 64
    assert prov["bytes"] == CKPT.stat().st_size
    assert prov["iteration"] == pickle.loads(CKPT.read_bytes())["iter"]
    assert prov["trained_env_cfg"]["n_threat"] == 16
    assert prov["obs_config"] == {"obs_edge_features": False}
    # the env actually flown was rebuilt from the checkpoint, then resized
    assert payload["base_env_config"]["n_blue"] == 2
    assert len(payload["base_env_config"]["threat_kinds"]) == len(
        prov["trained_env_cfg"]["threat_kinds"]
    )


@pytest.mark.skipif(not CKPT.exists(), reason="no shipped checkpoint")
def test_a_tiny_real_sweep_completes_offline():
    payload = _tiny_sweep(
        checkpoint=CKPT,
        make_policy=None,
        policy_provenance=None,
        base_env_cfg=None,
        env_overrides={"n_blue": 2, "n_threat": 8, "max_steps": 12},
        grid=SweepGrid(margins=(0.0, 1_500.0), threat_counts=(4,), red_levels=(0.3,)),
    )
    assert len(payload["cells"]) == 2
    for row in payload["cells"]:
        for arm in ARMS:
            m = row["arms"][arm]
            assert 0.0 <= m["survival_rate"] <= 1.0
            assert 0.0 <= m["objective_rate"] <= 1.0
            assert 0.0 <= m["cbf_infeasible_rate"] <= 1.0
            assert all(np.isfinite(v) for v in m.values())


@pytest.mark.skipif(not CKPT.exists(), reason="no shipped checkpoint")
def test_an_actor_whose_observation_width_does_not_match_is_refused():
    """A theatre checkpoint has five threat classes. A default `EnvConfig` has
    three, so the threat slot is 12 wide instead of 14 -- a shape mismatch the
    sweep must name rather than measure."""
    actor = pickle.loads(CKPT.read_bytes())["actor"]
    blob = {"actor": actor, "env_cfg": pickle.loads(CKPT.read_bytes())["env_cfg"]}
    require_actor_compatible(actor, env_config_from_blob(blob), str(CKPT))  # fine
    with pytest.raises(SweepConfigError) as e:
        require_actor_compatible(actor, EnvConfig(), str(CKPT))
    assert "threat slot" in str(e.value)


@pytest.mark.skipif(not CKPT.exists(), reason="no shipped checkpoint")
def test_env_config_from_blob_round_trips_the_trained_configuration():
    blob = pickle.loads(CKPT.read_bytes())
    cfg = env_config_from_blob(blob)
    stored = blob["env_cfg"]
    assert cfg.n_blue == stored["n_blue"]
    assert cfg.terrain.cell == stored["terrain"]["cell"]
    assert cfg.airframe.v_max == stored["airframe"]["v_max"]
    assert len(cfg.threat_kinds) == len(stored["threat_kinds"])
    assert cfg.threat_kinds[0].label == stored["threat_kinds"][0]["label"]
    # overrides win, and nothing else moves
    resized = env_config_from_blob(blob, n_blue=2, max_steps=12)
    assert (resized.n_blue, resized.max_steps) == (2, 12)
    assert resized.threat_kinds == cfg.threat_kinds


# --- serialization ------------------------------------------------------------


def test_serialization_is_deterministic_for_the_same_sweep():
    a = _tiny_sweep()
    b = _tiny_sweep()
    assert canonical_json(a) == canonical_json(b)
    # and the volatile keys are exactly the ones excluded, not a silent superset
    assert json.loads(canonical_json(a)).keys() == a.keys() - {"created_utc", "runtime"}


def test_serialize_is_sorted_stable_and_newline_terminated():
    payload = _tiny_sweep()
    text = serialize(payload)
    assert text.endswith("\n")
    assert text == serialize(json.loads(text))
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_a_written_result_round_trips_and_a_wrong_schema_is_refused(tmp_path):
    payload = _tiny_sweep()
    p = write_result(tmp_path / "sweep.json", payload)
    assert load_result(p)["seeds"]["values"] == [101, 102]
    p.write_text(serialize({**payload, "schema": SCHEMA_VERSION + 99}))
    with pytest.raises(SweepConfigError) as e:
        load_result(p)
    assert "schema" in str(e.value)


# --- reporting ----------------------------------------------------------------


def test_the_table_and_the_markdown_are_sorted_by_margin():
    payload = _tiny_sweep(
        grid=SweepGrid(margins=(3_000.0, 0.0, 1_500.0), threat_counts=(4,), red_levels=(0.3,))
    )
    body = [ln for ln in format_table(payload).splitlines()[2:] if ln.strip()]
    assert [float(ln.split()[0]) for ln in body] == [0.0, 1_500.0, 3_000.0]

    md = markdown_summary(payload)
    assert "infeas" in md
    assert "CBF minus no-CBF" in md
    assert str(payload["seeds"]["n_seeds"]) in md
    # both tables -- the metric one and the delta one -- are sorted by margin
    metrics_md, _, delta_md = md.partition("## CBF minus no-CBF")
    for section in (metrics_md, delta_md):
        margins = [
            float(ln.split("|")[1])
            for ln in section.splitlines()
            if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()
        ]
        assert margins == [0.0, 1_500.0, 3_000.0]


def test_the_markdown_names_its_provenance():
    payload = _tiny_sweep()
    md = markdown_summary(payload)
    assert f"schema `{SCHEMA_VERSION}`" in md
    assert "held-out, identical in every cell" in md
    assert str(payload["code"]["commit"]) in md
