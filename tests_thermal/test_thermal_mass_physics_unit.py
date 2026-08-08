"""Unit tests for the thermal-mass physics simulation core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emhass.thermal.thermal_mass_physics import (
    DEFAULT_X0,
    PARAM_NAMES,
    ThermalInputs,
    _infer_timestep_hours,
    _prepare_inputs,
    _simulate_open_loop,
)

pytestmark = pytest.mark.unit


def _weather_df(n: int = 48, outdoor_temp: float = 5.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "outdoor_temp": outdoor_temp,
            "wind_speed": 10.0,
            "ghi": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
            "heatpump_duty": 0.0,
        },
        index=idx,
    )


def test_infer_timestep_hours_half_hourly() -> None:
    idx = pd.date_range("2026-01-15", periods=10, freq="30min", tz="UTC")
    assert _infer_timestep_hours(idx) == pytest.approx(0.5)


def test_infer_timestep_hours_too_short_falls_back() -> None:
    idx = pd.date_range("2026-01-15", periods=1, freq="30min", tz="UTC")
    assert _infer_timestep_hours(idx) == pytest.approx(0.25)


def test_prepare_inputs_defaults_missing_columns() -> None:
    idx = pd.date_range("2026-01-15", periods=24, freq="30min", tz="UTC")
    df = pd.DataFrame(index=idx)  # no columns at all

    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )

    assert isinstance(inputs, ThermalInputs)
    assert len(inputs.room) == len(idx)
    assert np.all(inputs.duty == 0.0)
    assert np.all(inputs.outdoor == 10.0)  # documented default in _prepare_inputs


def test_prepare_inputs_reads_provided_columns() -> None:
    df = _weather_df(n=10, outdoor_temp=-2.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    assert np.all(inputs.outdoor == -2.0)
    assert np.all(inputs.wind_speed == 10.0)
    assert np.all(inputs.duty == 0.0)


def test_simulate_open_loop_holds_steady_with_all_terms_zeroed() -> None:
    """With every gain/loss coefficient at zero, nothing should move the
    temperature away from its initial value - a basic conservation check."""
    df = _weather_df(n=48, outdoor_temp=5.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    params = np.zeros(len(PARAM_NAMES), dtype=float)
    params[PARAM_NAMES.index("tau_emit_h")] = 1.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("mass_tau_h")] = 10.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert np.allclose(sim.room, 20.0)


def test_simulate_open_loop_cools_toward_colder_outdoor() -> None:
    """The scenario compute_heating_forecast actually runs: heating off
    (duty=0), a real envelope-loss coefficient, outdoor colder than indoor -
    room temperature must trend down, not up or flat."""
    df = _weather_df(n=48, outdoor_temp=-5.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_wind_sin_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("ua_wind_cos_per_h_per_speed")] = 0.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim.room[-1] < 20.0
    assert np.all(np.diff(sim.room) <= 1e-9)  # monotonically non-increasing


def test_simulate_open_loop_clips_to_physical_bounds() -> None:
    """d_air_dt is unbounded by construction; the stepper must still clip
    room temperature to the documented [5, 35] degC physical range."""
    df = _weather_df(n=200, outdoor_temp=-50.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_base_per_h")] = 0.25  # upper bound, aggressive loss

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim.room.min() >= 5.0
    assert sim.room.max() <= 35.0
