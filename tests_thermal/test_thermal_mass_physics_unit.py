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
    _simulate_segmented,
)

pytestmark = pytest.mark.unit


def _weather_df(n: int = 48, outdoor_temp: float = 5.0, ghi: float = 0.0, blind_position: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "outdoor_temp": outdoor_temp,
            "wind_speed": 10.0,
            "ghi": ghi,
            "dni": 0.0,
            "dhi": 0.0,
            "heatpump_duty": 0.0,
            "blind_position": blind_position,
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


# ----------------------------------------------------------------------
# Wall state + blind-gated window solar (see thermal_mass_physics.py's own
# module docstring for the physical rationale: window-transmitted gain is
# blind-blockable and feeds air directly; opaque-exterior-wall-absorbed
# gain is never blind-blockable and feeds a separate, laggy wall state that
# only reaches the room via wall_to_mass_weight's pull on T_mass).
# ----------------------------------------------------------------------


def test_wall_state_relaxes_toward_outdoor_plus_solar_when_isolated() -> None:
    """With every OTHER coupling zeroed (air/mass/emit terms all 0, so
    T_wall's own dynamics can be observed in isolation via sim.wall), a
    constant-sun scenario must pull T_wall toward outdoor + wall_solar_gain
    * q_solar - the "sun-exposed wall runs hotter than ambient" physics the
    new state exists to capture."""
    df = _weather_df(n=200, outdoor_temp=5.0, ghi=800.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93, facade_azimuth_deg=180.0, facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35, solar_facade_weight=0.65,
    )
    params = np.zeros(len(PARAM_NAMES), dtype=float)
    params[PARAM_NAMES.index("tau_emit_h")] = 1.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("mass_tau_h")] = 10.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("wall_tau_h")] = 0.5  # fast enough to equilibrate within the run
    params[PARAM_NAMES.index("wall_solar_gain_c")] = 8.0
    # wall_to_mass_weight stays 0 - isolates T_wall's OWN dynamics from
    # feeding back into T_air/T_mass (which are held fixed here anyway
    # since every other gain is 0).

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0, initial_wall=20.0)

    expected_wall_target = 5.0 + 8.0 * inputs.q_solar[-1]
    assert sim.wall[-1] == pytest.approx(expected_wall_target, abs=0.05)
    assert sim.room[-1] == pytest.approx(20.0)  # wall_to_mass_weight=0 -> zero effect on air


def test_blind_position_gates_window_solar_but_not_wall_solar() -> None:
    """Closing the blind must suppress the WINDOW/air-feeding solar pathway
    (room heats less with blind=1 than blind=0) while leaving the
    exterior-wall pathway completely unaffected (sim.wall identical either
    way) - the core physical distinction this feature exists to draw."""
    df_open = _weather_df(n=48, outdoor_temp=5.0, ghi=800.0, blind_position=0.0)
    df_closed = _weather_df(n=48, outdoor_temp=5.0, ghi=800.0, blind_position=1.0)
    kwargs = dict(
        latitude=51.65, longitude=4.93, facade_azimuth_deg=180.0, facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35, solar_facade_weight=0.65,
    )
    inputs_open = _prepare_inputs(df_open, **kwargs)
    inputs_closed = _prepare_inputs(df_closed, **kwargs)
    params = DEFAULT_X0.copy()  # solar_gain_c_per_h > 0, wall_solar_gain_c > 0 by default

    sim_open = _simulate_open_loop(inputs_open, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)
    sim_closed = _simulate_open_loop(inputs_closed, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim_open.room[-1] > sim_closed.room[-1]
    np.testing.assert_allclose(sim_open.wall, sim_closed.wall)


def test_wall_to_mass_weight_zero_reproduces_pre_wall_mass_formula() -> None:
    """At wall_to_mass_weight=0, T_mass must follow EXACTLY the pre-wall
    recurrence (mass = mass + mass_alpha*(air-mass)), regardless of how the
    (now-decoupled) wall state itself moves - the backward-compatibility
    guarantee the module docstring claims."""
    df = _weather_df(n=48, outdoor_temp=-5.0, ghi=500.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93, facade_azimuth_deg=180.0, facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35, solar_facade_weight=0.65,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_wind_sin_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("ua_wind_cos_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("wall_to_mass_weight")] = 0.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    mass_alpha = 0.5 / DEFAULT_X0[PARAM_NAMES.index("mass_tau_h")]
    expected_mass = np.zeros(len(sim.mass))
    mass = 20.0
    for i, air in enumerate(sim.air_before):
        mass = mass + mass_alpha * (air - mass)
        expected_mass[i] = mass
    np.testing.assert_allclose(sim.mass, expected_mass, atol=1e-9)


def test_simulate_segmented_matches_manual_per_segment_loop() -> None:
    """_simulate_segmented batches every FULL segment across a vectorized
    numpy loop instead of calling _simulate_open_loop once per segment (a
    performance rewrite, see its own docstring for why this is legal -
    segments never share state) - this is the numerical proof it wasn't
    also a behavior change. Manually reproduces the exact per-segment
    approach _simulate_segmented used before that rewrite (call
    _simulate_open_loop once per segment, each reseeded from the room's own
    actual history at that segment's start) and asserts bit-for-bit
    agreement. n=140 with segment_len=48 deliberately isn't a multiple (140
    = 2*48 + 44) so this also exercises the leftover-tail code path, not
    just the fully-batched one."""
    n, segment_len = 140, 48
    idx = pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "outdoor_temp": 5.0 + 3.0 * np.sin(np.linspace(0, 6, n)),
            "wind_speed": rng.uniform(0, 8, n),
            "wind_bearing": rng.uniform(0, 360, n),
            "ghi": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 600.0,
            "dni": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 400.0,
            "dhi": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 150.0,
            "heatpump_duty": rng.uniform(0, 1, n),
            "room_temp": 20.0 + 2.0 * np.sin(np.linspace(0, 10, n)),
            "blind_position": (np.arange(n) % 7 == 0).astype(float),
        },
        index=idx,
    )
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93, facade_azimuth_deg=180.0, facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35, solar_facade_weight=0.65,
    )
    params = DEFAULT_X0.copy()

    actual = _simulate_segmented(inputs, params, dt_h=0.5, segment_len=segment_len)

    expected = np.zeros(n, dtype=float)
    for start in range(0, n, segment_len):
        stop = min(n, start + segment_len)
        sub = ThermalInputs(
            index=inputs.index[start:stop],
            room=inputs.room[start:stop],
            electric=inputs.electric[start:stop],
            gas=inputs.gas[start:stop],
            duty=inputs.duty[start:stop],
            supply=inputs.supply[start:stop],
            outdoor=inputs.outdoor[start:stop],
            wind_speed=inputs.wind_speed[start:stop],
            wind_sin=inputs.wind_sin[start:stop],
            wind_cos=inputs.wind_cos[start:stop],
            q_solar=inputs.q_solar[start:stop],
            sun_alt_sin=inputs.sun_alt_sin[start:stop],
            sun_alt_cos=inputs.sun_alt_cos[start:stop],
            sun_az_sin=inputs.sun_az_sin[start:stop],
            sun_az_cos=inputs.sun_az_cos[start:stop],
            heatpump_duty=inputs.heatpump_duty[start:stop],
            blind_position=inputs.blind_position[start:stop],
        )
        initial_air = float(inputs.room[max(0, start - 1)])
        initial_q_emit = float(
            inputs.duty[max(0, start - 1)] * max(inputs.supply[max(0, start - 1)] - initial_air, 0.0)
        )
        sim = _simulate_open_loop(
            sub, params, dt_h=0.5, initial_air=initial_air, initial_mass=initial_air,
            initial_q_emit=initial_q_emit, initial_wall=initial_air,
        )
        expected[start:stop] = sim.room

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_simulate_segmented_handles_short_input_via_tail_path_only() -> None:
    """n < segment_len must skip the vectorized batch entirely (0 full
    segments) and fall through cleanly to the tail path alone."""
    df = _weather_df(n=10, outdoor_temp=5.0, ghi=200.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93, facade_azimuth_deg=180.0, facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35, solar_facade_weight=0.65,
    )
    pred = _simulate_segmented(inputs, DEFAULT_X0.copy(), dt_h=0.5, segment_len=48)
    assert len(pred) == 10
    assert np.all(np.isfinite(pred))
