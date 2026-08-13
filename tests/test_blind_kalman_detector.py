#!/usr/bin/env python3

"""
Tests for emhass.thermal.blind_kalman_detector - the sensorless, continuous
(0-1) blind/shading-position Kalman filter for self-learning-physics rooms.
Pure math, hand-computed expected values where practical, no I/O - mirrors
tests/test_opening_kalman_detector.py's style.
"""

import unittest

import numpy as np
import pandas as pd

from emhass.thermal.blind_kalman_detector import (
    BLIND_KALMAN_COLD_START_P,
    BLIND_KALMAN_R_CEILING,
    BLIND_KALMAN_R_FLOOR,
    _robust_normalize_to_unit_interval,
    blind_cold_start_state,
    bootstrap_raw_blind_signal_from_residual,
    invert_blind_position_from_residual,
    kalman_forward_filter_with_persistence,
    predict_room_temperature_blind_open_baseline,
    resolve_blind_measurement_noise,
    smoothed_blind_position,
)
from emhass.thermal.opening_kalman_detector import kalman_predict_update, kalman_rts_smooth


class TestBlindColdStartState(unittest.TestCase):
    def test_returns_open_with_wide_prior(self):
        x, p = blind_cold_start_state()
        self.assertEqual(x, 0.0)
        self.assertEqual(p, BLIND_KALMAN_COLD_START_P)


class TestResolveBlindMeasurementNoise(unittest.TestCase):
    def test_hand_computed_mid_range_value(self):
        # residual_std_c=0.5, |beta|=0.1, dni=50 -> denom=5 -> (0.5/5)**2 = 0.01
        r = resolve_blind_measurement_noise(residual_std_c=0.5, beta=-0.1, dni=50.0)
        self.assertAlmostEqual(r, 0.01)

    def test_scales_as_inverse_square_of_dni(self):
        r_50 = resolve_blind_measurement_noise(residual_std_c=0.5, beta=-0.1, dni=50.0)
        r_70 = resolve_blind_measurement_noise(residual_std_c=0.5, beta=-0.1, dni=70.0)
        self.assertAlmostEqual(r_50 / r_70, (70.0 / 50.0) ** 2, places=3)

    def test_scales_as_inverse_square_of_beta(self):
        r_beta_01 = resolve_blind_measurement_noise(residual_std_c=0.5, beta=0.1, dni=50.0)
        r_beta_015 = resolve_blind_measurement_noise(residual_std_c=0.5, beta=0.15, dni=50.0)
        self.assertAlmostEqual(r_beta_01 / r_beta_015, (0.15 / 0.1) ** 2, places=3)

    def test_floors_a_tiny_value(self):
        r = resolve_blind_measurement_noise(residual_std_c=0.01, beta=1.0, dni=1000.0)
        self.assertAlmostEqual(r, BLIND_KALMAN_R_FLOOR)

    def test_ceilings_when_beta_is_zero(self):
        r = resolve_blind_measurement_noise(residual_std_c=0.5, beta=0.0, dni=50.0)
        self.assertAlmostEqual(r, BLIND_KALMAN_R_CEILING)

    def test_ceilings_when_dni_is_zero(self):
        r = resolve_blind_measurement_noise(residual_std_c=0.5, beta=0.1, dni=0.0)
        self.assertAlmostEqual(r, BLIND_KALMAN_R_CEILING)

    def test_scalar_dni_returns_python_float(self):
        r = resolve_blind_measurement_noise(residual_std_c=0.5, beta=0.1, dni=50.0)
        self.assertIsInstance(r, float)

    def test_array_dni_returns_array(self):
        r = resolve_blind_measurement_noise(
            residual_std_c=0.5, beta=0.1, dni=np.array([50.0, 70.0])
        )
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(len(r), 2)


class _FakeModelRecordingBlindPosition:
    """Stand-in for SelfLearningPhysicsModel, just enough to exercise
    predict_next_room_temperature_self_learning (opening_kalman_detector.py)
    - records the blind_position column it was actually called with, so the
    test can prove the wrapper forces it to 0.0 regardless of input."""

    def __init__(self, room_name: str, predicted_temp: float = 21.0):
        self.room_models_ = {room_name: object()}
        self.predicted_temp = predicted_temp
        self.seen_blind_positions: list[list] = []

    def predict_recursive(self, df_house_fc, dfs_by_room_fc, initial_room_states):
        room_name = next(iter(dfs_by_room_fc))
        self.seen_blind_positions.append(dfs_by_room_fc[room_name]["blind_position"].tolist())
        n = len(df_house_fc)
        return {"room_temp": {room_name: np.full(n, self.predicted_temp)}}


class TestPredictRoomTemperatureBlindOpenBaseline(unittest.TestCase):
    def test_forces_blind_position_to_zero_regardless_of_input(self):
        fake_model = _FakeModelRecordingBlindPosition("Living Room", predicted_temp=21.0)
        idx = pd.date_range("2026-01-01", periods=1, freq="30min", tz="UTC")
        df_house_fc = pd.DataFrame({"outdoor_temp": [5.0]}, index=idx)
        df_room_fc_with_blind = pd.DataFrame({"blind_position": [0.8]}, index=idx)

        result = predict_room_temperature_blind_open_baseline(
            fake_model, "Living Room", df_house_fc, df_room_fc_with_blind, current_temp=20.0
        )

        self.assertEqual(result, 21.0)
        self.assertEqual(fake_model.seen_blind_positions, [[0.0]])

    def test_room_not_in_model_returns_none(self):
        fake_model = _FakeModelRecordingBlindPosition("Living Room")
        idx = pd.date_range("2026-01-01", periods=1, freq="30min", tz="UTC")
        df_house_fc = pd.DataFrame({"outdoor_temp": [5.0]}, index=idx)
        df_room_fc = pd.DataFrame({"blind_position": [0.5]}, index=idx)

        result = predict_room_temperature_blind_open_baseline(
            fake_model, "Bedroom", df_house_fc, df_room_fc, current_temp=20.0
        )

        self.assertIsNone(result)
        self.assertEqual(fake_model.seen_blind_positions, [])


class TestInvertBlindPositionFromResidual(unittest.TestCase):
    def test_hand_computed_sign_and_masking(self):
        # beta negative (closing the blind cuts solar gain, as it must).
        residual = np.array([-1.0, -1.0, 0.0])
        dni = np.array([100.0, 30.0, 100.0])  # index 1 below the default 50 floor
        beta = -0.02

        raw = invert_blind_position_from_residual(residual, dni, beta)

        # t=0: -1.0 / (-0.02*100) = -1.0/-2.0 = 0.5
        self.assertAlmostEqual(raw[0], 0.5)
        # t=1: masked (dni below floor)
        self.assertTrue(np.isnan(raw[1]))
        # t=2: 0.0 / (-0.02*100) = 0.0
        self.assertAlmostEqual(raw[2], 0.0)

    def test_clips_out_of_range_inversions(self):
        beta = -0.02
        dni = np.array([100.0, 100.0])
        residual = np.array([-5.0, 5.0])  # would invert to 2.5 and -2.5

        raw = invert_blind_position_from_residual(residual, dni, beta)

        self.assertAlmostEqual(raw[0], 1.0)
        self.assertAlmostEqual(raw[1], 0.0)


class TestBootstrapRawBlindSignalFromResidual(unittest.TestCase):
    def test_more_negative_residual_produces_higher_synthetic_position(self):
        dni = np.array([100.0, 100.0, 100.0, 100.0])
        residual = np.array([-2.0, -1.0, -0.5, 0.0])

        result = bootstrap_raw_blind_signal_from_residual(residual, dni)

        finite = result[np.isfinite(result)]
        self.assertTrue(np.all(np.diff(finite) <= 0))
        self.assertTrue(np.all((finite >= 0.0) & (finite <= 1.0)))

    def test_uninformative_timestep_is_nan(self):
        # At least 2 informative points are needed for the percentile
        # normalization step itself to produce a non-NaN result.
        dni = np.array([100.0, 10.0, 80.0])
        residual = np.array([-1.0, -1.0, -0.5])

        result = bootstrap_raw_blind_signal_from_residual(residual, dni)

        self.assertFalse(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertFalse(np.isnan(result[2]))


class TestRobustNormalizeToUnitInterval(unittest.TestCase):
    def test_outlier_does_not_dominate_scale(self):
        # 20 normal values plus 1 outlier (~4.8% of the sample, just under
        # the 5th/95th percentile clip window) - the outlier must not crush
        # the normal values' own scale toward 0.
        normal_values = np.linspace(0.0, 0.4, 20)
        raw = np.concatenate([normal_values, [100.0]])

        result = _robust_normalize_to_unit_interval(raw)

        # The highest NORMAL value (index 19, raw=0.4) should land at/near
        # the top of the scale, not be crushed near 0 by the outlier.
        self.assertGreater(result[19], 0.5)

    def test_degenerate_zero_variance_input_returns_all_nan(self):
        raw = np.array([0.5, 0.5, 0.5, 0.5])
        result = _robust_normalize_to_unit_interval(raw)
        self.assertTrue(np.all(np.isnan(result)))

    def test_fewer_than_two_finite_values_returns_all_nan(self):
        raw = np.array([np.nan, 0.5, np.nan])
        result = _robust_normalize_to_unit_interval(raw)
        self.assertTrue(np.all(np.isnan(result)))

    def test_nan_input_stays_nan(self):
        raw = np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = _robust_normalize_to_unit_interval(raw)
        self.assertTrue(np.isnan(result[1]))
        self.assertFalse(np.isnan(result[0]))


class TestKalmanForwardFilterWithPersistence(unittest.TestCase):
    def test_matches_manual_persistence_loop_when_fully_informative(self):
        """With no NaN gaps, this must be exactly equivalent to manually
        chaining kalman_predict_update calls with x_pred=x_prev each step -
        it's a convenience wrapper around the SAME per-step math, not a
        second implementation."""
        x0, p0 = 0.3, 0.1
        z = np.array([0.5, 0.6, 0.4])
        q, r = 0.01, 0.05

        traj = kalman_forward_filter_with_persistence(x0, p0, z, q, r)

        x_prev, p_prev = x0, p0
        for t in range(len(z)):
            expected = kalman_predict_update(
                x_prev=x_prev, p_prev=p_prev, x_pred=x_prev,
                z_measured=float(z[t]), q=q, r=r,
            )
            self.assertAlmostEqual(traj.x_pred[t], expected.x_pred)
            self.assertAlmostEqual(traj.p_pred[t], expected.p_pred)
            self.assertAlmostEqual(traj.x_filt[t], expected.x_new)
            self.assertAlmostEqual(traj.p_filt[t], expected.p_new)
            self.assertAlmostEqual(traj.innovation[t], expected.innovation)
            x_prev, p_prev = expected.x_new, expected.p_new

    def test_nan_gap_freezes_x_and_grows_p(self):
        x0, p0 = 0.3, 0.1
        z = np.array([0.5, np.nan, np.nan])
        q, r = 0.01, 0.05

        traj = kalman_forward_filter_with_persistence(x0, p0, z, q, r)

        # x is frozen at whatever it was after the last real update.
        self.assertAlmostEqual(traj.x_filt[1], traj.x_filt[0])
        self.assertAlmostEqual(traj.x_filt[2], traj.x_filt[0])
        # p grows monotonically through the gap (q added each step, nothing
        # to shrink it back down).
        self.assertGreater(traj.p_filt[1], traj.p_filt[0])
        self.assertGreater(traj.p_filt[2], traj.p_filt[1])
        self.assertAlmostEqual(traj.p_filt[1], traj.p_filt[0] + q)
        self.assertAlmostEqual(traj.p_filt[2], traj.p_filt[1] + q)

    def test_array_r_varies_per_timestep(self):
        x0, p0 = 0.3, 0.1
        z = np.array([0.5, 0.5])
        q = 0.01
        r = np.array([0.01, 0.5])  # far more confident measurement at t=0 than t=1

        traj = kalman_forward_filter_with_persistence(x0, p0, z, q, r)

        # A tighter r at t=0 should shrink p_filt more aggressively than the
        # much looser r at t=1.
        shrink_0 = traj.p_pred[0] - traj.p_filt[0]
        shrink_1 = traj.p_pred[1] - traj.p_filt[1]
        self.assertGreater(shrink_0, shrink_1)

    def test_kalman_rts_smooth_runs_unmodified_against_its_output(self):
        """Regression-proves opening_kalman_detector.kalman_rts_smooth is
        genuinely reusable unchanged against a persistence-style trajectory
        - it only reads x_pred/p_pred/x_filt/p_filt generically."""
        x0, p0 = 0.3, 0.1
        z = np.array([0.5, np.nan, 0.4, 0.6, np.nan])
        q, r = 0.01, 0.05

        traj = kalman_forward_filter_with_persistence(x0, p0, z, q, r)
        x_smooth, p_smooth = kalman_rts_smooth(traj)

        self.assertEqual(len(x_smooth), 5)
        self.assertEqual(len(p_smooth), 5)
        self.assertTrue(np.all(np.isfinite(x_smooth)))
        self.assertTrue(np.all(p_smooth >= 0.0))
        # Smoothing (using future info too) can only ever reduce or match
        # uncertainty relative to the forward-only pass, same property the
        # window/door smoother's own test suite already asserts.
        self.assertTrue(np.all(p_smooth <= traj.p_pred + 1e-9))


class TestSmoothedBlindPosition(unittest.TestCase):
    def test_clips_out_of_range_values(self):
        x_smooth = np.array([-0.2, 0.0, 0.5, 1.0, 1.3])
        result = smoothed_blind_position(x_smooth)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
