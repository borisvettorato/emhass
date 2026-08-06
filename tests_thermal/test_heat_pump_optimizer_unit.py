"""
Unit tests for HeatPumpOptimizer.

These tests validate deterministic optimizer behavior with synthetic inputs.
"""

import logging

import numpy as np
import pytest

from emhass.thermal import HeatPumpOptimizer

logger = logging.getLogger(__name__)


class TestHeatPumpOptimizerUnit:
    """Fast unit tests that do not require real CSV history or model inference."""

    @pytest.fixture
    def optimizer(self):
        # Project target curve: y = 40 - x and max deviation 10C.
        return HeatPumpOptimizer(
            curve_intercept=40.0,
            curve_slope=-1.0,
            max_deviation=10.0,
            min_supply_temp=10.0,
            max_supply_temp=60.0,
        )

    def test_baseline_curve_calculation(self, optimizer):
        outdoor_temps = np.array([0, 10, 20, 30])
        expected = np.array([40, 30, 20, 10])

        result = optimizer.get_baseline_curve(outdoor_temps)
        assert np.allclose(result, expected)

    def test_baseline_curve_extreme_temps(self, optimizer):
        outdoor_temps = np.array([-10, 45])
        expected = np.array([50, -5])

        result = optimizer.get_baseline_curve(outdoor_temps)
        assert np.allclose(result, expected)

    def test_optimal_setpoint_output_shape(self, optimizer):
        room_temps = np.array([20.0] * 144)
        outdoor_temps = np.array([8.0] * 144)

        result = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)

        assert "setpoint_optimal" in result
        assert "setpoint_min" in result
        assert "setpoint_max" in result
        assert len(result["setpoint_optimal"]) == 144

    def test_optimal_setpoint_bounds(self, optimizer):
        room_temps = np.linspace(18, 23, 144)
        outdoor_temps = np.linspace(-2, 15, 144)

        result = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)
        setpoint = result["setpoint_optimal"]

        assert np.all(setpoint >= optimizer.min_supply_temp)
        assert np.all(setpoint <= optimizer.max_supply_temp)

    def test_optimal_setpoint_within_curve_deviation(self, optimizer):
        room_temps = np.array([20.0] * 144)
        outdoor_temps = np.linspace(0, 25, 144)

        result = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)
        setpoint = result["setpoint_optimal"]

        baseline = optimizer.get_baseline_curve(outdoor_temps)
        baseline_clamped = np.clip(baseline, optimizer.min_supply_temp, optimizer.max_supply_temp)

        max_allowed = optimizer.max_deviation + 0.2
        assert np.mean(np.abs(setpoint - baseline_clamped) <= max_allowed) > 0.9

    def test_price_aware_no_worse_than_neutral(self, optimizer):
        room_temps = np.linspace(19.0, 21.0, 144)
        outdoor_temps = np.linspace(3.0, 12.0, 144)
        prices = np.linspace(0.15, 0.45, 144)

        neutral = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)
        price_aware = optimizer.get_price_aware_setpoint(
            room_temp_forecast=room_temps,
            outdoor_temp_forecast=outdoor_temps,
            price_forecast=prices,
            price_weight=0.8,
        )

        neutral_cost_proxy = float(np.sum(neutral["setpoint_optimal"] * prices))
        price_aware_cost_proxy = float(np.sum(price_aware["setpoint_price_aware"] * prices))

        assert price_aware_cost_proxy <= neutral_cost_proxy + 1e-6

    def test_constraint_violations_detect_out_of_range(self, optimizer):
        outdoor_temps = np.array([10.0] * 144)
        invalid_setpoint = np.array([75.0] * 144)

        violations = optimizer.get_constraint_violations(invalid_setpoint, outdoor_temps)
        assert np.any(violations["too_high"])

    def test_empty_forecast_handling(self, optimizer):
        room_temps = np.array([])
        outdoor_temps = np.array([])

        result = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)
        assert len(result["setpoint_optimal"]) == 0

        logger.info("Unit tests validated with synthetic inputs")
