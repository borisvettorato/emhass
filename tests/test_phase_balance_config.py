"""
Per-phase (L1/L2/L3) load balancing - config-layer tests.

Covers the config-layer normalisation: number_of_phases
(validate_num_phases), the per-phase/per-battery array broadcast/validate
helper (check_phase_array_params), the historical-share computation
(compute_phase_power_shares) used to split the aggregate load/PV forecast
per phase (see command_line.py::prepare_forecast_and_weather_data), and the
load_phase/battery_phase tag parser (_resolve_phase_tag) that supports both
a single phase and a "+"-joined combination for a genuinely multi-phase
device. The optimization.py LP constraint itself (that this parser feeds
into) is covered end-to-end in tests/test_optimization.py's
test_phase_balance_* tests.

Mirrors tests/test_multi_battery_config.py's shape closely - this feature's
config layer follows the exact same broadcast-scalar-or-exact-length-list
pattern check_batt_params already established for batteries, deliberately
without check_batt_params' own numeric coercion (see check_phase_array_params's
own docstring for why: a phase label like "L1" is not a number).
"""

import asyncio
import json
import logging
import pathlib

import pandas as pd
import pytest

from emhass import utils
from emhass.optimization import _resolve_phase_tag

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)


def _default_config() -> dict:
    return json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))


async def _build_params(overrides: dict | None = None) -> dict:
    config = _default_config()
    if overrides:
        config.update(overrides)
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    assert params is not False, "build_params failed (see logged error)"
    return params


def build_params(overrides: dict | None = None) -> dict:
    return asyncio.run(_build_params(overrides))


# ─────────────────────────── number_of_phases plumbing ──────────────────────


def test_number_of_phases_defaults_to_one():
    params = build_params()
    assert params["plant_conf"].get("number_of_phases") == 1


def test_number_of_phases_overridable_via_config():
    params = build_params({"number_of_phases": 3})
    assert params["plant_conf"]["number_of_phases"] == 3


@pytest.mark.parametrize("raw,expected", [(0, 1), (-1, 1), (4, 3), (10, 3)])
def test_validate_num_phases_clamps_out_of_range(raw, expected):
    """Out-of-range values are clamped WITH A WARNING, not raised - a bad
    number_of_phases should degrade the safety feature, never abort the
    whole optimization run (unlike validate_num_batteries, which does
    raise - a phase count has a real physical bound of 3, a battery count
    does not)."""
    result = utils.validate_num_phases({"number_of_phases": raw}, logger)
    assert result == expected


def test_validate_num_phases_missing_key_defaults_to_one():
    assert utils.validate_num_phases({}, logger) == 1


def test_validate_num_phases_non_numeric_defaults_to_one():
    assert utils.validate_num_phases({"number_of_phases": "not-a-number"}, logger) == 1


# ──────────────── check_phase_array_params (battery_phase / grid-per-phase) ──


_PHASE_ARRAY_PARAMS = [
    ("battery_phase", ""),
    ("maximum_power_from_grid_per_phase", 4000),
    ("maximum_power_to_grid_per_phase", 4000),
]


@pytest.mark.parametrize("param_name,default", _PHASE_ARRAY_PARAMS)
def test_check_phase_array_params_scalar_broadcasts_to_n(param_name, default):
    parameter = {param_name: "X" if isinstance(default, str) else 42}
    result = utils.check_phase_array_params(3, parameter, default, param_name, logger)
    expected = ["X", "X", "X"] if isinstance(default, str) else [42, 42, 42]
    assert result == expected
    assert parameter[param_name] == expected


@pytest.mark.parametrize("param_name,default", _PHASE_ARRAY_PARAMS)
def test_check_phase_array_params_exact_length_list_passthrough(param_name, default):
    values = ["L1", "L2", "L3"] if isinstance(default, str) else [1, 2, 3]
    parameter = {param_name: list(values)}
    result = utils.check_phase_array_params(3, parameter, default, param_name, logger)
    assert result == values


@pytest.mark.parametrize("param_name,default", _PHASE_ARRAY_PARAMS)
def test_check_phase_array_params_wrong_length_raises(param_name, default):
    parameter = {param_name: ["a", "b"] if isinstance(default, str) else [1, 2]}
    with pytest.raises(ValueError) as excinfo:
        utils.check_phase_array_params(3, parameter, default, param_name, logger)
    message = str(excinfo.value)
    assert param_name in message
    assert "3" in message


@pytest.mark.parametrize("param_name,default", _PHASE_ARRAY_PARAMS)
def test_check_phase_array_params_missing_key_defaults_and_broadcasts(param_name, default):
    result = utils.check_phase_array_params(3, {}, default, param_name, logger)
    assert result == [default, default, default]


@pytest.mark.parametrize("param_name,default", _PHASE_ARRAY_PARAMS)
def test_check_phase_array_params_count1_is_true_noop(param_name, default):
    """count == 1 must be a true no-op: a scalar stays a scalar, never
    wrapped into a 1-element list."""
    value = "L1" if isinstance(default, str) else 123
    parameter = {param_name: value}
    result = utils.check_phase_array_params(1, parameter, default, param_name, logger)
    assert result == value
    assert not isinstance(result, list)


def test_check_phase_array_params_string_value_not_numerically_coerced():
    """The whole reason this is a SEPARATE helper from check_batt_params:
    _coerce_batt_element unconditionally attempts float() on any non-null
    string, which raises for a genuine phase label like 'L1'. Confirm
    check_phase_array_params does not do this."""
    parameter = {"battery_phase": "L2"}
    result = utils.check_phase_array_params(1, parameter, "", "battery_phase", logger)
    assert result == "L2"


# ───────────────────────── compute_phase_power_shares ────────────────────────


def _hist_index(n=48):
    return pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")


def test_compute_phase_power_shares_basic_ratio():
    idx = _hist_index()
    df = pd.DataFrame(
        {
            "sensor.l1": [100.0] * len(idx),
            "sensor.l2": [300.0] * len(idx),
            "sensor.l3": [600.0] * len(idx),
        },
        index=idx,
    )
    shares = utils.compute_phase_power_shares(
        df, ["sensor.l1", "sensor.l2", "sensor.l3"], logger
    )
    assert shares == pytest.approx([0.1, 0.3, 0.6])


def test_compute_phase_power_shares_unconfigured_returns_none():
    idx = _hist_index()
    df = pd.DataFrame({"sensor.l1": [100.0] * len(idx)}, index=idx)
    assert utils.compute_phase_power_shares(df, ["", "", ""], logger) is None
    assert utils.compute_phase_power_shares(df, [], logger) is None
    assert utils.compute_phase_power_shares(None, ["sensor.l1"], logger) is None


def test_compute_phase_power_shares_missing_column_returns_none():
    idx = _hist_index()
    df = pd.DataFrame({"sensor.other": [100.0] * len(idx)}, index=idx)
    assert utils.compute_phase_power_shares(df, ["sensor.l1", "sensor.l2"], logger) is None


def test_compute_phase_power_shares_all_zero_returns_none():
    idx = _hist_index()
    df = pd.DataFrame(
        {"sensor.l1": [0.0] * len(idx), "sensor.l2": [0.0] * len(idx)}, index=idx
    )
    assert utils.compute_phase_power_shares(df, ["sensor.l1", "sensor.l2"], logger) is None


def test_compute_phase_power_shares_partial_configuration():
    """One phase sensor blank ('') - that phase gets a 0 share, the other
    two split the rest of the ratio normally."""
    idx = _hist_index()
    df = pd.DataFrame(
        {"sensor.l1": [200.0] * len(idx), "sensor.l3": [200.0] * len(idx)}, index=idx
    )
    shares = utils.compute_phase_power_shares(df, ["sensor.l1", "", "sensor.l3"], logger)
    assert shares == pytest.approx([0.5, 0.0, 0.5])


# ───────────────────── _resolve_phase_tag (load_phase parser) ────────────────

_PHASE_LABELS_3 = ["L1", "L2", "L3"]
_PHASE_LABELS_2 = ["L1", "L2"]


def test_resolve_phase_tag_empty_is_unassigned():
    assert _resolve_phase_tag("", _PHASE_LABELS_3) is None
    assert _resolve_phase_tag("   ", _PHASE_LABELS_3) is None


def test_resolve_phase_tag_single_label():
    assert _resolve_phase_tag("L1", _PHASE_LABELS_3) == ["L1"]
    assert _resolve_phase_tag("L2", _PHASE_LABELS_3) == ["L2"]


def test_resolve_phase_tag_two_phase_combo():
    assert _resolve_phase_tag("L1+L2", _PHASE_LABELS_3) == ["L1", "L2"]


def test_resolve_phase_tag_three_phase_combo():
    assert _resolve_phase_tag("L1+L2+L3", _PHASE_LABELS_3) == ["L1", "L2", "L3"]


def test_resolve_phase_tag_combo_valid_for_a_2_phase_household():
    """A genuinely 2-phase household (number_of_phases=2) must accept
    'L1+L2' as its own "all phases" combination - the parser is phase-
    count-agnostic, not hardcoded to 3."""
    assert _resolve_phase_tag("L1+L2", _PHASE_LABELS_2) == ["L1", "L2"]


def test_resolve_phase_tag_unknown_label_returns_none():
    assert _resolve_phase_tag("L3", _PHASE_LABELS_2) is None


def test_resolve_phase_tag_partially_unknown_combo_returns_none():
    """One valid label alongside one invalid one must exclude the WHOLE
    combination, not silently keep just the valid part - a misconfigured
    tag should degrade coverage visibly, not guess a different split."""
    assert _resolve_phase_tag("L1+L3", _PHASE_LABELS_2) is None


def test_resolve_phase_tag_whitespace_around_labels_tolerated():
    assert _resolve_phase_tag(" L1 + L2 ", _PHASE_LABELS_3) == ["L1", "L2"]


def test_resolve_phase_tag_malformed_plus_only_returns_none():
    assert _resolve_phase_tag("+", _PHASE_LABELS_3) is None
    assert _resolve_phase_tag("L1+", _PHASE_LABELS_3) == ["L1"]
