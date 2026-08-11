#!/usr/bin/env python3

"""
Tests for HybridHeatPumpLR's electric_only mode (pure-electric heat pump,
no gas boiler).

Class-level coverage only - the command_line.py refit/forecast action tests
already exercise electric_only end to end (tests/test_command_line_utils.py,
test_refit_hybrid_heatpump_model_not_gated_on_is_hybrid and friends). This
file specifically pins down the four branches that had to change: fit(),
predict(), coef_summary(), predict_gas_proba().
"""

import unittest

import numpy as np
import pandas as pd

from emhass.thermal.hybrid_heatpump_lr import HybridHeatPumpLR


def _make_training_frame(n: int = 300, with_gas: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    outdoor = 5.0 + 8.0 * np.sin(np.linspace(0, 8 * np.pi, n))  # dips below 0degC part of the cycle
    duty = np.clip(rng.uniform(0.0, 1.0, n), 0.0, 1.0)
    df = pd.DataFrame(
        {
            "outdoor_temp": outdoor,
            "supply_temp": 35.0 + rng.normal(0, 1, n),
            "heatpump_duty": duty,
            "room_temp": 20.0 + rng.normal(0, 0.3, n),
            "wind_speed": rng.uniform(0, 8, n),
            "ghi_norm": rng.uniform(0, 1, n),
            "sun_alt_sin": rng.uniform(0, 1, n),
        },
        index=idx,
    )
    y_elec = 300.0 + 800.0 * duty * np.clip(15.0 - outdoor, 0.0, None) / 15.0
    if with_gas:
        y_gas = np.where(outdoor < 0.0, rng.uniform(0.01, 0.05, n), 0.0)
    else:
        y_gas = np.zeros(n)
    return df, y_elec, y_gas


class TestHybridHeatPumpLRElectricOnly(unittest.TestCase):
    def test_fit_predict_hybrid_mode_has_gas_model(self):
        df, y_elec, y_gas = _make_training_frame(with_gas=True)
        model = HybridHeatPumpLR()
        model.fit(df, y_elec, y_gas)

        self.assertIsNotNone(model.gas_model_)
        elec_pred, gas_pred = model.predict(df)
        self.assertEqual(len(elec_pred), len(df))
        self.assertEqual(len(gas_pred), len(df))

    def test_fit_predict_electric_only_mode_has_no_gas_model(self):
        df, y_elec, _ = _make_training_frame(with_gas=True)
        model = HybridHeatPumpLR(electric_only=True)
        model.fit(df, y_elec)

        self.assertIsNone(model.gas_model_)
        elec_pred, gas_pred = model.predict(df)
        self.assertEqual(len(elec_pred), len(df))
        np.testing.assert_array_equal(gas_pred, np.zeros(len(df)))

    def test_hybrid_mode_crashes_on_all_zero_gas_target(self):
        # The bug electric_only exists to route around: a real all-zero gas
        # target is a single-class y, which sklearn's LogisticRegression
        # refuses to fit. Confirms the failure mode is real, not assumed.
        df, y_elec, y_gas_all_zero = _make_training_frame(with_gas=False)
        model = HybridHeatPumpLR()
        with self.assertRaises(ValueError):
            model.fit(df, y_elec, y_gas_all_zero)

    def test_electric_only_never_touches_gas_target(self):
        # Same all-zero-gas data that crashes hybrid mode above must fit
        # cleanly in electric_only mode, since y_gas is never passed at all.
        df, y_elec, _ = _make_training_frame(with_gas=False)
        model = HybridHeatPumpLR(electric_only=True)
        model.fit(df, y_elec)  # must not raise
        self.assertIsNone(model.gas_model_)

    def test_coef_summary_electric_only_has_no_gas_rows(self):
        df, y_elec, _ = _make_training_frame(with_gas=True)
        model = HybridHeatPumpLR(electric_only=True)
        model.fit(df, y_elec)

        summary = model.coef_summary()
        self.assertIn("electric", set(summary["model"]))
        self.assertNotIn("gas_magnitude", set(summary["model"]))

    def test_coef_summary_hybrid_mode_has_gas_rows(self):
        df, y_elec, y_gas = _make_training_frame(with_gas=True)
        model = HybridHeatPumpLR()
        model.fit(df, y_elec, y_gas)

        summary = model.coef_summary()
        self.assertIn("electric", set(summary["model"]))
        self.assertIn("gas_magnitude", set(summary["model"]))

    def test_predict_gas_proba_electric_only_returns_zeros(self):
        df, y_elec, _ = _make_training_frame(with_gas=True)
        model = HybridHeatPumpLR(electric_only=True)
        model.fit(df, y_elec)

        proba = model.predict_gas_proba(df)
        np.testing.assert_array_equal(proba, np.zeros(len(df)))


if __name__ == "__main__":
    unittest.main()
