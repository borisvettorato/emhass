#!/usr/bin/env python3

"""
Tests for emhass.thermal.opening_kalman_detector - the sensorless
"window/door probably open" Kalman filter. Pure math, hand-computed
expected values, no I/O - mirrors tests/test_arx_model.py's
style.
"""

import unittest

import numpy as np
import pandas as pd

from emhass.thermal.opening_kalman_detector import (
    KALMAN_GATE_SIGMA,
    cold_start_state,
    kalman_forward_filter_array,
    kalman_predict_update,
    kalman_rts_smooth,
    predict_next_room_temperature_physics_family,
    predict_next_room_temperature_self_learning,
    smoothed_opening_flags,
)
from emhass.utils import simulate_physics_room_temperature_trajectory


class TestKalmanPredictUpdate(unittest.TestCase):
    def test_small_residual_never_flags_open(self):
        """Window closed: a small residual well within normal noise must
        never cross the gate, however many cycles it repeats."""
        x, p = 20.0, 0.09
        for _ in range(5):
            result = kalman_predict_update(
                x_prev=x, p_prev=p, x_pred=20.0, z_measured=20.05, q=0.01, r=0.09
            )
            self.assertFalse(result.is_open)
            x, p = result.x_new, result.p_new

    def test_hand_computed_single_step(self):
        """Exact hand-computed values for one step, matching the module's
        own documented equations."""
        result = kalman_predict_update(
            x_prev=20.0, p_prev=0.1, x_pred=20.0, z_measured=20.5, q=0.0, r=0.1
        )
        # p_pred = p_prev + q = 0.1
        # innovation = 20.5 - 20.0 = 0.5
        # s = p_pred + r = 0.2
        # k = p_pred / s = 0.5
        # x_new = x_pred + k*innovation = 20.0 + 0.5*0.5 = 20.25
        # p_new = (1-k)*p_pred = 0.5*0.1 = 0.05
        self.assertAlmostEqual(result.p_pred, 0.1)
        self.assertAlmostEqual(result.innovation, 0.5)
        self.assertAlmostEqual(result.s, 0.2)
        self.assertAlmostEqual(result.k, 0.5)
        self.assertAlmostEqual(result.x_new, 20.25)
        self.assertAlmostEqual(result.p_new, 0.05)
        # gate: |0.5| > 3*sqrt(0.2)=1.342 -> False
        self.assertFalse(result.is_open)

    def test_sustained_large_residual_flags_open_within_two_steps(self):
        """Window opens: a large, REPEATING residual (the room stays
        colder than predicted every cycle, not just one noisy reading) must
        cross the gate within 1-2 chained steps."""
        q, r = 0.01, 0.09
        x, p = 20.0, r  # start from a settled cold-start-like state
        # Predictor keeps assuming "closed" (predicts ~20.0 each cycle,
        # mirroring a real window/door staying open across several cycles),
        # but the room is actually sitting near 18.5 (a sustained ~1.5C gap).
        opened = False
        for _ in range(2):
            result = kalman_predict_update(
                x_prev=x, p_prev=p, x_pred=20.0, z_measured=18.5, q=q, r=r
            )
            x, p = result.x_new, result.p_new
            opened = opened or result.is_open
        self.assertTrue(opened, "A sustained ~1.5C gap should cross the gate within 2 cycles")

    def test_zero_variance_edge_case_does_not_divide_by_zero(self):
        """s == 0 (p_pred=0, r=0) must not raise - K falls back to 0.0."""
        result = kalman_predict_update(
            x_prev=20.0, p_prev=0.0, x_pred=20.0, z_measured=20.0, q=0.0, r=0.0
        )
        self.assertEqual(result.k, 0.0)
        self.assertEqual(result.s, 0.0)

    def test_gate_sigma_is_configurable(self):
        """A looser gate_sigma should let a residual through that the
        default gate already flags. s = p_pred+r = 0.11, sqrt(s) = 0.332;
        default gate (3.0) threshold = 0.995 < innovation(1.0) -> open;
        loose gate (10.0) threshold = 3.317 > innovation(1.0) -> not open."""
        kwargs = {"x_prev": 20.0, "p_prev": 0.1, "x_pred": 20.0, "z_measured": 21.0, "q": 0.0, "r": 0.01}
        result_default = kalman_predict_update(**kwargs, gate_sigma=KALMAN_GATE_SIGMA)
        result_loose = kalman_predict_update(**kwargs, gate_sigma=10.0)
        self.assertTrue(result_default.is_open)
        self.assertFalse(result_loose.is_open)


class TestColdStartState(unittest.TestCase):
    def test_returns_measurement_and_r_exactly(self):
        x0, p0 = cold_start_state(z_measured=19.3, r=0.09)
        self.assertEqual(x0, 19.3)
        self.assertEqual(p0, 0.09)


class TestKalmanForwardFilterArray(unittest.TestCase):
    def test_matches_manual_loop_of_kalman_predict_update(self):
        """kalman_forward_filter_array must be exactly equivalent to
        manually chaining kalman_predict_update calls, element by
        element - it's a convenience wrapper, not a second implementation."""
        x0, p0 = 20.0, 0.1
        x_pred = np.array([20.0, 19.8, 20.2, 20.0])
        z = np.array([20.1, 19.5, 21.0, 20.0])
        q, r = 0.02, 0.09

        traj = kalman_forward_filter_array(x0, p0, x_pred, z, q, r)

        x_prev, p_prev = x0, p0
        for t in range(len(x_pred)):
            expected = kalman_predict_update(
                x_prev=x_prev, p_prev=p_prev, x_pred=float(x_pred[t]),
                z_measured=float(z[t]), q=q, r=r,
            )
            self.assertAlmostEqual(traj.x_pred[t], expected.x_pred)
            self.assertAlmostEqual(traj.p_pred[t], expected.p_pred)
            self.assertAlmostEqual(traj.x_filt[t], expected.x_new)
            self.assertAlmostEqual(traj.p_filt[t], expected.p_new)
            self.assertAlmostEqual(traj.innovation[t], expected.innovation)
            x_prev, p_prev = expected.x_new, expected.p_new


class TestKalmanRtsSmooth(unittest.TestCase):
    def test_hand_computed_three_step_example_q_zero(self):
        """x0=20.0, p0=0.1, x_pred=[20,20,20], z=[20.0,20.5,20.0], q=0, r=0.1.
        Forward: p_pred=[0.1,0.05,0.033333], x_filt=[20.0,20.166667,20.0],
        p_filt=[0.05,0.033333,0.025]. Backward (q=0 collapses
        p_pred[t+1]=p_filt[t], so C[t]=1.0 for every t): x_smooth=
        [20.166667, 20.166667, 20.0], p_smooth=[0.025, 0.025, 0.025] (a
        tractable corner case where p_smooth collapses to a uniform value
        across the whole window - not the general case, see the q>0 test
        below for that)."""
        x0, p0 = 20.0, 0.1
        x_pred = np.array([20.0, 20.0, 20.0])
        z = np.array([20.0, 20.5, 20.0])
        traj = kalman_forward_filter_array(x0, p0, x_pred, z, q=0.0, r=0.1)

        x_smooth, p_smooth = kalman_rts_smooth(traj)

        np.testing.assert_allclose(x_smooth, [20.166667, 20.166667, 20.0], atol=1e-5)
        np.testing.assert_allclose(p_smooth, [0.025, 0.025, 0.025], atol=1e-9)

    def test_p_smooth_never_exceeds_p_pred(self):
        """Conditioning on strictly more data (past AND future) can only
        reduce uncertainty, never increase it - p_smooth[t] <= p_pred[t]
        must hold for every t, for a general q > 0 case."""
        rng = np.random.default_rng(42)
        n = 20
        x0, p0 = 20.0, 0.2
        x_pred = 20.0 + rng.normal(0, 0.3, n)
        z = x_pred + rng.normal(0, 0.2, n)
        traj = kalman_forward_filter_array(x0, p0, x_pred, z, q=0.05, r=0.1)

        _, p_smooth = kalman_rts_smooth(traj)

        self.assertTrue(np.all(p_smooth <= traj.p_pred + 1e-12))

    def test_base_case_matches_last_forward_filtered_value(self):
        x0, p0 = 20.0, 0.1
        x_pred = np.array([20.0, 20.0, 20.0])
        z = np.array([20.0, 20.5, 21.0])
        traj = kalman_forward_filter_array(x0, p0, x_pred, z, q=0.01, r=0.1)

        x_smooth, p_smooth = kalman_rts_smooth(traj)

        self.assertAlmostEqual(x_smooth[-1], traj.x_filt[-1])
        self.assertAlmostEqual(p_smooth[-1], traj.p_filt[-1])


class TestSmoothedOpeningFlags(unittest.TestCase):
    def test_sustained_anomaly_flagged_transient_blip_not(self):
        """A single noisy reading surrounded by otherwise-normal
        measurements should NOT flag open; a sustained, repeating gap
        (a real open window/door) should."""
        x0, p0 = 20.0, 0.09
        q, r = 0.01, 0.09

        # Transient: one noisy reading, then back to normal.
        x_pred_blip = np.full(5, 20.0)
        z_blip = np.array([20.05, 20.6, 20.02, 19.98, 20.01])
        traj_blip = kalman_forward_filter_array(x0, p0, x_pred_blip, z_blip, q, r)
        _, p_smooth_blip = kalman_rts_smooth(traj_blip)
        flags_blip = smoothed_opening_flags(traj_blip, p_smooth_blip, r)
        self.assertFalse(np.any(flags_blip), "A single noisy reading should not flag open")

        # Sustained: a real, repeating gap for several steps.
        x_pred_sustained = np.full(5, 20.0)
        z_sustained = np.array([20.0, 15.0, 15.0, 15.0, 20.0])
        traj_sustained = kalman_forward_filter_array(
            x0, p0, x_pred_sustained, z_sustained, q, r
        )
        _, p_smooth_sustained = kalman_rts_smooth(traj_sustained)
        flags_sustained = smoothed_opening_flags(traj_sustained, p_smooth_sustained, r)
        self.assertTrue(np.any(flags_sustained), "A sustained 5C gap should flag open")

    def test_smoothed_gate_is_never_less_sensitive_than_forward_only(self):
        """p_smooth <= p_pred (proven separately in TestKalmanRtsSmooth)
        implies sqrt(p_smooth+r) <= sqrt(p_pred+r), so the smoothed gate's
        threshold is never larger than the forward-only gate's own
        threshold - i.e. smoothing can only flag MORE, never fewer,
        anomalies than a live forward-only detector would have."""
        rng = np.random.default_rng(7)
        n = 15
        x0, p0 = 20.0, 0.15
        x_pred = 20.0 + rng.normal(0, 0.3, n)
        z = x_pred + rng.normal(0, 0.25, n)
        q, r = 0.02, 0.09
        traj = kalman_forward_filter_array(x0, p0, x_pred, z, q, r)
        _, p_smooth = kalman_rts_smooth(traj)

        forward_only_flags = np.abs(traj.innovation) > KALMAN_GATE_SIGMA * np.sqrt(
            traj.p_pred + r
        )
        smoothed_flags = smoothed_opening_flags(traj, p_smooth, r)

        # Every forward-only flag must also be a smoothed flag (smoothing
        # only ever adds confidence, never removes it).
        self.assertTrue(np.all(smoothed_flags | ~forward_only_flags))


class TestPredictNextRoomTemperaturePhysicsFamily(unittest.TestCase):
    def test_matches_direct_simulate_physics_trajectory_call(self):
        """The wrapper must be a faithful delegate to
        utils.simulate_physics_room_temperature_trajectory, not a second,
        potentially-drifted copy of the recurrence formula."""
        kwargs = {
            "duty": 0.6,
            "outdoor_temp": 5.0,
            "nominal_power_w": 1500.0,
            "dt_hours": 0.5,
            "volume": 15.0,
            "base_loss": 0.045,
            "supply_temperature": 35.0,
            "carnot_efficiency": 0.4,
            "heating_demand_kwh": 0.1,
            "sense": "heat",
        }
        wrapped = predict_next_room_temperature_physics_family(
            current_temp=20.0, **kwargs
        )
        direct_trajectory = simulate_physics_room_temperature_trajectory(
            initial_temp=20.0,
            duty=np.array([kwargs["duty"], kwargs["duty"]]),
            outdoor_temp=np.array([kwargs["outdoor_temp"], kwargs["outdoor_temp"]]),
            nominal_power_w=kwargs["nominal_power_w"],
            dt_hours=kwargs["dt_hours"],
            volume=kwargs["volume"],
            base_loss=kwargs["base_loss"],
            supply_temperature=kwargs["supply_temperature"],
            carnot_efficiency=kwargs["carnot_efficiency"],
            heating_demand_kwh=np.array(
                [kwargs["heating_demand_kwh"], kwargs["heating_demand_kwh"]]
            ),
            sense=kwargs["sense"],
        )
        self.assertAlmostEqual(wrapped, float(direct_trajectory[1]), places=9)

    def test_zero_duty_and_demand_predicts_pure_heat_loss(self):
        """With no heat added and outdoor colder than indoor, the predicted
        next temperature must be strictly lower than the starting one."""
        predicted = predict_next_room_temperature_physics_family(
            current_temp=20.0,
            duty=0.0,
            outdoor_temp=0.0,
            nominal_power_w=1500.0,
            dt_hours=0.5,
            heating_demand_kwh=0.0,
        )
        self.assertLess(predicted, 20.0)


class _FakeSelfLearningModelForKalman:
    """Minimal stand-in for ArxModel, matching the shape
    tests/test_command_line_utils.py's own _FakeArxModel
    uses - just enough for predict_next_room_temperature_self_learning to
    exercise its own contract (room coverage check + 1-row call shape)."""

    def __init__(self, room_names, predicted_value=21.0):
        self.room_models_ = dict.fromkeys(room_names, object())
        self.predicted_value = predicted_value
        self.seen_calls = []

    def predict_recursive(self, df_house_fc, dfs_by_room_fc, initial_room_states):
        self.seen_calls.append((df_house_fc, dfs_by_room_fc, initial_room_states))
        n = len(df_house_fc)
        room_temp = {
            name: np.full(n, self.predicted_value) for name in dfs_by_room_fc
        }
        return {"room_temp": room_temp, "electric_power": np.zeros(n), "gas_consumption": None}


class TestPredictNextRoomTemperatureSelfLearning(unittest.TestCase):
    def test_returns_none_for_uncovered_room(self):
        model = _FakeSelfLearningModelForKalman(room_names=["Bedroom"])
        idx = pd.date_range("2026-01-01", periods=1, freq="30min", tz="UTC")
        df_house_fc = pd.DataFrame({"outdoor_temp": [5.0]}, index=idx)
        result = predict_next_room_temperature_self_learning(
            model, "Living Room", df_house_fc, df_house_fc.copy(), current_temp=20.0
        )
        self.assertIsNone(result)
        self.assertEqual(model.seen_calls, [])

    def test_calls_predict_recursive_with_one_row_and_returns_first_value(self):
        model = _FakeSelfLearningModelForKalman(
            room_names=["Living Room"], predicted_value=21.5
        )
        idx = pd.date_range("2026-01-01", periods=1, freq="30min", tz="UTC")
        df_house_fc = pd.DataFrame({"outdoor_temp": [5.0]}, index=idx)
        df_room_fc = pd.DataFrame({"outdoor_temp": [5.0]}, index=idx)

        result = predict_next_room_temperature_self_learning(
            model, "Living Room", df_house_fc, df_room_fc, current_temp=20.0
        )

        self.assertEqual(result, 21.5)
        self.assertEqual(len(model.seen_calls), 1)
        seen_house_fc, seen_rooms, seen_states = model.seen_calls[0]
        self.assertEqual(len(seen_house_fc), 1)
        self.assertIn("Living Room", seen_rooms)
        self.assertEqual(len(seen_rooms["Living Room"]), 1)
        self.assertEqual(seen_states, {"Living Room": 20.0})


if __name__ == "__main__":
    unittest.main()
