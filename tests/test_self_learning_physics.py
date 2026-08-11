#!/usr/bin/env python3

"""
Tests for the multi-room self-learning-physics model
(emhass.thermal.self_learning_physics), an online-adaptive (RLS) model
ported and extended from scripts/compare_ensemble.py's single-zone
"SelfLearningPhysics" benchmark to N rooms with learned inter-room coupling.

Coverage: _physics_features's neighbor-diff/group_duty column handling,
_rls_fit_theta's convergence on a known synthetic linear relationship (the
ported RLS mechanics, independent of any physics-feature collinearity),
the recursive multi-room predictor's closed-loop (non-teacher-forced) and
inter-room-coupling behavior against hand-computed expected values, and
coupling_coefficients_kw_per_k's unit conversion.
"""

import unittest

import numpy as np
import pandas as pd

from emhass.thermal.self_learning_physics import (
    _BASE_FEATURE_NAMES,
    SelfLearningPhysicsModel,
    _physics_features,
    _rls_fit_theta,
    _RoomModel,
)


class TestPhysicsFeatures(unittest.TestCase):
    def test_appends_group_duty_and_neighbor_diff_columns_in_order(self):
        idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
        df = pd.DataFrame(
            {
                "heatpump_duty": [0.1, 0.2, 0.3, 0.4],
                "outdoor_temp": [5.0] * 4,
                "supply_temp": [35.0] * 4,
                "wind_speed": [1.0] * 4,
                "dni": [0.0] * 4,
                "dhi": [0.0] * 4,
                "sun_alt_sin": [0.0] * 4,
                "room_temp": [20.0, 20.1, 20.2, 20.3],
            },
            index=idx,
        )
        group_duty = pd.Series([0.5] * 4, index=idx)
        neighbor_diffs = {
            "kitchen": pd.Series([1.0, 1.1, 1.2, 1.3], index=idx),
            "bedroom": pd.Series([-0.5, -0.4, -0.3, -0.2], index=idx),
        }

        X, names = _physics_features(df, group_duty=group_duty, neighbor_diffs=neighbor_diffs)

        self.assertEqual(
            names,
            [*_BASE_FEATURE_NAMES, "group_duty", "neighbor_diff::kitchen", "neighbor_diff::bedroom"],
        )
        self.assertEqual(X.shape, (4, len(names)))
        np.testing.assert_allclose(X[:, names.index("group_duty")], group_duty.to_numpy())
        np.testing.assert_allclose(
            X[:, names.index("neighbor_diff::kitchen")], neighbor_diffs["kitchen"].to_numpy()
        )
        np.testing.assert_allclose(
            X[:, names.index("neighbor_diff::bedroom")], neighbor_diffs["bedroom"].to_numpy()
        )

    def test_no_neighbor_diffs_only_appends_group_duty(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        df = pd.DataFrame({"heatpump_duty": [0.0, 0.5, 1.0]}, index=idx)

        X, names = _physics_features(df)

        self.assertEqual(names, [*_BASE_FEATURE_NAMES, "group_duty"])
        self.assertEqual(X.shape, (3, len(_BASE_FEATURE_NAMES) + 1))


class TestRlsFitTheta(unittest.TestCase):
    def test_converges_to_known_linear_relationship(self):
        # Noiseless, uncorrelated regressors - isolates the ported RLS
        # recursion itself from the physics-feature collinearity risk
        # already identified (and accepted) for the real feature set.
        rng = np.random.default_rng(0)
        n, n_feat = 3000, 4
        X = rng.normal(size=(n, n_feat))
        X[:, 0] = 1.0  # bias column
        theta_true = np.array([3.0, 2.0, -1.5, 0.5])
        y = X @ theta_true

        # A small ridge (large initial P) keeps the prior weak so RLS
        # converges tightly to the true (noiseless) coefficients; a large
        # ridge is a stronger shrinkage-toward-zero prior and would bias
        # theta_hat toward 0 by design - see SelfLearningPhysicsModel's own
        # docstring for what `ridge` controls.
        theta_hat, diagnostics = _rls_fit_theta(X, y, forgetting=1.0, ridge=0.1)

        np.testing.assert_allclose(theta_hat, theta_true, atol=0.01)
        self.assertEqual(diagnostics["n_obs"], n)


def _make_multiroom_training_frames(n: int = 600):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    outdoor = 5.0 + 8.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    duty = np.clip(rng.uniform(0.0, 1.0, n), 0.0, 1.0)
    df_house = pd.DataFrame(
        {
            "outdoor_temp": outdoor,
            "supply_temp": 35.0 + rng.normal(0, 1, n),
            "heatpump_duty": duty,
            "group_duty": duty,
            "wind_speed": rng.uniform(0, 8, n),
            "dni": rng.uniform(0, 1, n),
            "dhi": rng.uniform(0, 1, n),
            "sun_alt_sin": rng.uniform(0, 1, n),
        },
        index=idx,
    )
    room_a = df_house.copy()
    room_a["room_temp"] = 20.0 + rng.normal(0, 0.3, n)
    room_b = df_house.copy()
    room_b["room_temp"] = 19.0 + rng.normal(0, 0.3, n)
    y_elec = 300.0 + 800.0 * duty * np.clip(15.0 - outdoor, 0.0, None) / 15.0
    return df_house, {"A": room_a, "B": room_b}, y_elec


class TestSelfLearningPhysicsModelFitPredict(unittest.TestCase):
    def test_fit_and_predict_recursive_multiroom_electric_only(self):
        df_house, dfs_by_room, y_elec = _make_multiroom_training_frames()
        neighbor_map = {"A": ["B"], "B": ["A"]}
        model = SelfLearningPhysicsModel(electric_only=True)

        model.fit(df_house, dfs_by_room, y_elec, None, neighbor_map)

        self.assertIsNone(model.theta_gas_)
        self.assertEqual(set(model.room_models_.keys()), {"A", "B"})

        fc_house = df_house.iloc[:10]
        fc_rooms = {name: df.iloc[:10] for name, df in dfs_by_room.items()}
        result = model.predict_recursive(fc_house, fc_rooms, {"A": 20.0, "B": 19.0})

        self.assertIsNone(result["gas_consumption"])
        self.assertEqual(len(result["electric_power"]), 10)
        self.assertEqual(set(result["room_temp"].keys()), {"A", "B"})
        for arr in result["room_temp"].values():
            self.assertEqual(len(arr), 10)
            self.assertFalse(np.isnan(arr).any())

    def test_coupling_disabled_yields_rooms_with_no_neighbors(self):
        df_house, dfs_by_room, y_elec = _make_multiroom_training_frames()
        model = SelfLearningPhysicsModel(electric_only=True)

        model.fit(df_house, dfs_by_room, y_elec, None, neighbor_map={})

        for room_model in model.room_models_.values():
            self.assertEqual(room_model.neighbors, [])

    def test_predict_before_fit_raises(self):
        model = SelfLearningPhysicsModel()
        with self.assertRaises(RuntimeError):
            model.predict_recursive(pd.DataFrame(), {}, {})


class TestRecursivePredictionIsClosedLoop(unittest.TestCase):
    """Hand-computed checks that predict_recursive (a) never reads a
    forecast frame's own room_temp column (non-teacher-forced) and (b)
    correctly threads a neighbor's *predicted* (not observed) previous
    temperature into the next step - by constructing model internals
    directly with known coefficients, bypassing .fit() entirely so the
    expected numbers can be computed by hand.
    """

    def _build_model(self) -> SelfLearningPhysicsModel:
        house_feature_names = [*_BASE_FEATURE_NAMES, "group_duty"]
        room_a_features = [*_BASE_FEATURE_NAMES, "group_duty", "neighbor_diff::B"]
        room_b_features = [*_BASE_FEATURE_NAMES, "group_duty"]

        theta_a = np.zeros(len(room_a_features))
        theta_a[room_a_features.index("bias")] = 15.0
        theta_a[room_a_features.index("neighbor_diff::B")] = 0.2

        theta_b = np.zeros(len(room_b_features))
        theta_b[room_b_features.index("bias")] = 10.0

        model = SelfLearningPhysicsModel(electric_only=True)
        model.theta_elec_ = np.zeros(len(house_feature_names))
        model.house_feature_names_ = house_feature_names
        model.room_models_ = {
            "A": _RoomModel(theta_temp=theta_a, feature_names=room_a_features, neighbors=["B"]),
            "B": _RoomModel(theta_temp=theta_b, feature_names=room_b_features, neighbors=[]),
        }
        model._is_fitted = True
        return model

    def _fc_frame(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
        return pd.DataFrame(
            {
                "heatpump_duty": [0.0] * n,
                "outdoor_temp": [10.0] * n,
                "wind_speed": [0.0] * n,
                "dni": [0.0] * n,
                "dhi": [0.0] * n,
                "sun_alt_sin": [0.0] * n,
                "group_duty": [0.0] * n,
            },
            index=idx,
        )

    def test_matches_hand_computed_two_step_trajectory(self):
        model = self._build_model()
        fc_house = self._fc_frame(2)
        fc_rooms = {"A": fc_house.copy(), "B": fc_house.copy()}

        result = model.predict_recursive(fc_house, fc_rooms, {"A": 18.0, "B": 22.0})

        # Step 0: neighbor_diff = B(22.0) - A(18.0) = 4.0
        #   A = 15.0 + 0.2*4.0 = 15.8 ; B = 10.0 (no neighbor term)
        # Step 1: neighbor_diff = B(10.0) - A(15.8) = -5.8
        #   A = 15.0 + 0.2*(-5.8) = 13.84 ; B = 10.0
        np.testing.assert_allclose(result["room_temp"]["A"], [15.8, 13.84], atol=1e-9)
        np.testing.assert_allclose(result["room_temp"]["B"], [10.0, 10.0], atol=1e-9)

    def test_ignores_room_temp_column_in_forecast_frame(self):
        # A forecast frame's own "room_temp" column (if present at all) must
        # never leak into the recursion - only initial_room_states plus the
        # model's own prior-step predictions may drive the next step.
        model = self._build_model()
        fc_house = self._fc_frame(3)
        rooms_with_bogus_column = {
            "A": fc_house.assign(room_temp=999.0),
            "B": fc_house.assign(room_temp=-999.0),
        }
        rooms_without_column = {"A": fc_house.copy(), "B": fc_house.copy()}

        result_bogus = model.predict_recursive(fc_house, rooms_with_bogus_column, {"A": 18.0, "B": 22.0})
        result_clean = model.predict_recursive(fc_house, rooms_without_column, {"A": 18.0, "B": 22.0})

        np.testing.assert_allclose(result_bogus["room_temp"]["A"], result_clean["room_temp"]["A"])
        np.testing.assert_allclose(result_bogus["room_temp"]["B"], result_clean["room_temp"]["B"])


class TestCouplingCoefficientsKwPerK(unittest.TestCase):
    def test_matches_hand_computed_unit_conversion(self):
        # theta_diff == conversion * g * dt_hours  =>  g = theta_diff / (conversion * dt_hours)
        # conversion = 3600 / mass_kj_per_k
        theta_diff = 0.2
        mass_kj_per_k = 500.0
        dt_hours = 0.5
        conversion = 3600.0 / mass_kj_per_k
        expected_g = theta_diff / (conversion * dt_hours)

        model = SelfLearningPhysicsModel(electric_only=True)
        model.theta_elec_ = np.zeros(1)
        model.room_models_ = {
            "A": _RoomModel(
                theta_temp=np.array([theta_diff]), feature_names=["neighbor_diff::B"], neighbors=["B"]
            ),
            "B": _RoomModel(theta_temp=np.array([0.0]), feature_names=["bias"], neighbors=[]),
        }
        model._is_fitted = True

        result = model.coupling_coefficients_kw_per_k(
            {"A": mass_kj_per_k, "B": mass_kj_per_k}, dt_hours=dt_hours
        )

        self.assertIn(("A", "B"), result)
        self.assertAlmostEqual(result[("A", "B")], expected_g, places=9)

    def test_zero_or_missing_mass_skips_room(self):
        model = SelfLearningPhysicsModel(electric_only=True)
        model.theta_elec_ = np.zeros(1)
        model.room_models_ = {
            "A": _RoomModel(
                theta_temp=np.array([0.2]), feature_names=["neighbor_diff::B"], neighbors=["B"]
            ),
        }
        model._is_fitted = True

        result = model.coupling_coefficients_kw_per_k({"A": 0.0}, dt_hours=0.5)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
