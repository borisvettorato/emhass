"""Unit tests for the RC-model sensorless blind/door Kalman inference
(thermal_mass_physics_kalman.py) - synthetic data with a KNOWN injected
blind-closure/door-opening pattern, verifying the teacher-forced one-step
predictor and the algebraic blind-position inversion actually recover it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emhass.thermal.blind_kalman_detector import (
    BLIND_KALMAN_Q,
    blind_cold_start_state,
    kalman_forward_filter_with_persistence,
    smoothed_blind_position,
)
from emhass.thermal.opening_kalman_detector import (
    cold_start_state,
    kalman_forward_filter_array,
    kalman_rts_smooth,
    smoothed_opening_flags,
)
from emhass.thermal.thermal_mass_physics import DEFAULT_X0, PARAM_NAMES, ThermalInputs, _simulate_open_loop
from emhass.thermal.thermal_mass_physics_kalman import (
    invert_blind_position_from_residual,
    predict_one_step_history,
    resolve_measurement_noise,
)

pytestmark = pytest.mark.unit


def _synthetic_inputs(n: int, *, blind_position: np.ndarray, door_open: np.ndarray) -> ThermalInputs:
    """A synthetic "sunny midday" pattern - raw ghi/dni/dhi plus a plausible
    (if not astronomically exact) sun-position trace, matching the SHAPE of
    a real day (rises, peaks near midday, sets), which is all these tests
    need - the plane-of-array formula itself is separately verified against
    real pvlib solar positions in test_facade_poa_matches_pvlib_isotropic_model.
    """
    idx = pd.date_range("2026-06-01", periods=n, freq="30min", tz="UTC")
    t = np.arange(n)
    hour = idx.hour + idx.minute / 60.0
    envelope = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    outdoor = 15.0 + 8.0 * np.sin(2 * np.pi * t / 48.0)
    duty = np.zeros(n)
    supply = np.full(n, 25.0)
    zeros = np.zeros(n)
    return ThermalInputs(
        index=idx,
        room=zeros.copy(),
        electric=zeros.copy(),
        gas=zeros.copy(),
        duty=duty,
        supply=supply,
        outdoor=outdoor,
        wind_speed=zeros.copy(),
        wind_sin=zeros.copy(),
        wind_cos=zeros.copy(),
        sun_alt_sin=envelope * 0.8,
        sun_alt_cos=np.full(n, 0.5),
        sun_az_sin=zeros.copy(),
        sun_az_cos=np.full(n, -1.0),  # sun due south, matching the default south-facing facade
        heatpump_duty=duty,
        blind_position=blind_position,
        door_open=door_open,
        ghi=envelope * 700.0,
        dni=envelope * 900.0,
        dhi=envelope * 150.0,
    )


def test_predict_one_step_history_matches_open_loop_with_matching_forcing() -> None:
    """When force_blind_zero/force_door_zero match the ACTUAL trajectory's
    own blind/door values (both all-zero here), predict_one_step_history's
    one-step predictions must agree closely with what _simulate_open_loop
    itself produces one step ahead - the teacher-forcing shouldn't change
    anything when there's nothing to correct."""
    n = 60
    inputs = _synthetic_inputs(n, blind_position=np.zeros(n), door_open=np.zeros(n))
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("solar_gain_c_per_h")] = 0.3

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=18.0)
    inputs_observed = ThermalInputs(**{**inputs.__dict__, "room": sim.room})

    pred, _sensitivity, _q_solar = predict_one_step_history(
        inputs_observed, params, 0.5, force_blind_zero=True, force_door_zero=True
    )
    # pred[i] predicts room[i] from room[i-1] - compare against the room
    # trajectory itself (both start from the same true open-loop dynamics).
    np.testing.assert_allclose(pred[1:], sim.room[1:], atol=0.05)


def test_predict_one_step_history_sensitivity_is_nonnegative_and_zero_at_night() -> None:
    n = 48
    inputs = _synthetic_inputs(n, blind_position=np.zeros(n), door_open=np.zeros(n))
    inputs = ThermalInputs(**{**inputs.__dict__, "room": np.full(n, 20.0)})
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("solar_gain_c_per_h")] = 0.3

    _pred, sensitivity, _q_solar = predict_one_step_history(inputs, params, 0.5, force_blind_zero=True)

    assert np.all(sensitivity >= 0.0)
    # sensitivity[i] is driven by ghi/dni/dhi[max(0, i-1)] (see
    # predict_one_step_history's own "src" convention), so the night mask
    # must be shifted the same way to compare like with like. ghi alone is
    # a fine day/night proxy here (_synthetic_inputs scales ghi/dni/dhi by
    # the exact same envelope, so they're all zero together at night).
    ghi_src = np.concatenate([inputs.ghi[:1], inputs.ghi[:-1]])
    night_mask = ghi_src < 1e-6
    assert night_mask.any()
    np.testing.assert_allclose(sensitivity[night_mask], 0.0, atol=1e-12)
    assert sensitivity.max() > 0.0  # some daytime signal must exist


def test_door_detection_recovers_injected_night_ventilation_with_no_false_positives() -> None:
    n = 200
    idx = pd.date_range("2026-06-01", periods=n, freq="30min", tz="UTC")
    hour = idx.hour + idx.minute / 60.0
    true_door = ((hour >= 1) & (hour <= 4)).astype(float)
    zeros = np.zeros(n)
    inputs_true = _synthetic_inputs(n, blind_position=zeros, door_open=true_door)
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("door_open_extra_loss_per_h")] = 0.35
    sim = _simulate_open_loop(inputs_true, params, dt_h=0.5, initial_air=18.0)

    inputs_baseline = ThermalInputs(**{**inputs_true.__dict__, "room": sim.room, "door_open": zeros})
    pred, _s, _q_solar = predict_one_step_history(inputs_baseline, params, 0.5, force_door_zero=True)
    residual = inputs_baseline.room - pred
    finite = residual[np.isfinite(residual)]
    mad = float(np.median(np.abs(finite - np.median(finite))))
    r = max(0.0004, (1.4826 * mad) ** 2)
    q = 0.2 * r
    x0, p0 = cold_start_state(float(inputs_baseline.room[0]), r)
    trajectory = kalman_forward_filter_array(x0, p0, pred, inputs_baseline.room, q, r)
    _, p_smooth = kalman_rts_smooth(trajectory)
    detected = smoothed_opening_flags(trajectory, p_smooth, r)

    false_positives = int(np.sum(detected & (true_door < 0.5)))
    true_positives = int(np.sum(detected & (true_door >= 0.5)))
    assert false_positives == 0
    assert true_positives >= int(true_door.sum()) * 0.5  # conservative gate, at least half recovered


def test_blind_detection_recovers_injected_midday_closure() -> None:
    n = 200
    idx = pd.date_range("2026-06-01", periods=n, freq="30min", tz="UTC")
    t = np.arange(n)
    hour = idx.hour + idx.minute / 60.0
    q_solar_check = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 0.6
    true_blind = ((hour >= 11) & (hour <= 15) & (q_solar_check > 0.3)).astype(float)
    zeros = np.zeros(n)
    inputs_true = _synthetic_inputs(n, blind_position=true_blind, door_open=zeros)
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("solar_gain_c_per_h")] = 0.3
    sim = _simulate_open_loop(inputs_true, params, dt_h=0.5, initial_air=18.0)

    inputs_baseline = ThermalInputs(**{**inputs_true.__dict__, "room": sim.room, "blind_position": zeros})
    pred, sensitivity, _q_solar = predict_one_step_history(inputs_baseline, params, 0.5, force_blind_zero=True)
    residual = inputs_baseline.room - pred
    finite = residual[np.isfinite(residual)]
    residual_std = float(1.4826 * np.median(np.abs(finite - np.median(finite))))

    # q_solar_check (already computed above, same envelope that drove
    # ghi/dni/dhi in _synthetic_inputs) stands in for a real q_solar array
    # here - close enough for this test's own threshold-based informative
    # gate, without needing to reconstruct the plane-of-array formula.
    raw = invert_blind_position_from_residual(residual, sensitivity, q_solar_check)
    r = resolve_measurement_noise(residual_std, sensitivity)
    x0, p0 = blind_cold_start_state()
    trajectory = kalman_forward_filter_with_persistence(x0, p0, raw, BLIND_KALMAN_Q, r)
    x_smooth, _ = kalman_rts_smooth(trajectory)
    position = smoothed_blind_position(x_smooth)

    informative = q_solar_check > 0.3
    assert informative.any()
    # Closed periods should read meaningfully higher than open periods.
    assert position[informative & (true_blind >= 0.5)].mean() > position[informative & (true_blind < 0.5)].mean()


def test_invert_blind_position_sign_and_clip() -> None:
    residual = np.array([-1.0, 1.0, -1.0])
    sensitivity = np.array([2.0, 2.0, 2.0])
    q_solar = np.array([1.0, 1.0, 1.0])
    raw = invert_blind_position_from_residual(residual, sensitivity, q_solar)
    # residual<0 (cooler than "assumed open") -> inferred position toward closed (positive).
    assert raw[0] == pytest.approx(0.5)
    # residual>0 (warmer than "assumed open") -> clipped at 0 (can't be "more open than open").
    assert raw[1] == pytest.approx(0.0)


def test_invert_blind_position_gates_on_uninformative_q_solar() -> None:
    residual = np.array([-1.0, -1.0])
    sensitivity = np.array([2.0, 2.0])
    q_solar = np.array([0.001, 1.0])  # first below the informative floor
    raw = invert_blind_position_from_residual(residual, sensitivity, q_solar)
    assert np.isnan(raw[0])
    assert np.isfinite(raw[1])


def test_resolve_measurement_noise_floor_and_ceiling() -> None:
    # Huge sensitivity -> r would collapse toward 0 without the floor.
    r_huge = resolve_measurement_noise(0.3, np.array([1000.0]))
    assert r_huge[0] == pytest.approx(0.0025)
    # Zero sensitivity -> r must hit the ceiling, never divide-by-zero raise.
    r_zero = resolve_measurement_noise(0.3, np.array([0.0]))
    assert r_zero[0] == pytest.approx(1.0)
