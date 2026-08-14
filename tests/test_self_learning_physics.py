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
    _RLS_P_NORM_CEILING,
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
            [
                *_BASE_FEATURE_NAMES,
                "group_duty",
                "neighbor_diff::kitchen",
                "door_x_neighbor_diff::kitchen",
                "neighbor_diff::bedroom",
                "door_x_neighbor_diff::bedroom",
            ],
        )
        self.assertEqual(X.shape, (4, len(names)))
        np.testing.assert_allclose(X[:, names.index("group_duty")], group_duty.to_numpy())
        np.testing.assert_allclose(
            X[:, names.index("neighbor_diff::kitchen")], neighbor_diffs["kitchen"].to_numpy()
        )
        np.testing.assert_allclose(
            X[:, names.index("neighbor_diff::bedroom")], neighbor_diffs["bedroom"].to_numpy()
        )
        # No door_open column provided -> door_x_neighbor_diff::* is all-zero
        # (df.get("door_open", ...) defaults to 0.0, see _physics_features).
        np.testing.assert_allclose(
            X[:, names.index("door_x_neighbor_diff::kitchen")], np.zeros(4)
        )
        np.testing.assert_allclose(
            X[:, names.index("door_x_neighbor_diff::bedroom")], np.zeros(4)
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

    def test_covariance_windup_stays_bounded_over_long_idle_stretch(self):
        # Reproduces a real refit failure (observed on a real 180-day/
        # half-hourly refit window): a feature (e.g. a duty-related
        # cross-term) sits at ~0 for a long stretch (e.g. a summer with the
        # heat pump off) - exponential forgetting (forgetting < 1) inflates
        # P in that unexcited direction every idle step with nothing to
        # correct it. Left unbounded this reaches float64 overflow and
        # corrupts theta to NaN/a meaningless magnitude, even though no
        # single (x, y) pair is itself extreme - confirmed by running this
        # exact scenario through the pre-fix update rule (no covariance
        # cap): P overflows to inf by step ~4000 and theta is all-NaN by
        # the end. forgetting=0.9 (more aggressive than the model's own
        # 0.995 default) is used only to reach that failure within a few
        # thousand steps instead of the tens of thousands the real default
        # would need - the mechanism under test (unbounded idle-direction
        # growth under forgetting < 1) is the same regardless of the exact
        # forgetting value.
        rng = np.random.default_rng(3)
        n, n_feat = 8000, 4
        X = np.zeros((n, n_feat), dtype=float)
        X[:, 0] = 1.0  # bias column: always excited
        X[:, 1] = rng.normal(size=n)  # always excited
        # Columns 2 and 3 (e.g. duty*delta_supply, duty*delta_env) are only
        # excited in a short early burst, then go to exactly 0 for the rest
        # of the window - the idle-direction windup scenario.
        burst = slice(0, 400)
        X[burst, 2] = rng.normal(size=400)
        X[burst, 3] = rng.normal(size=400)
        theta_true = np.array([1.0, 2.0, -3.0, 0.5])
        y = X @ theta_true + rng.normal(scale=0.01, size=n)

        with np.errstate(all="raise"):
            theta_hat, diagnostics = _rls_fit_theta(X, y, forgetting=0.9, ridge=10.0)

        self.assertTrue(np.all(np.isfinite(theta_hat)))
        # Bounded by construction: the covariance cap keeps ||P|| <=
        # _RLS_P_NORM_CEILING, and the largest sane coefficient magnitude
        # the RLS update can produce off a bounded P and a well-scaled
        # (small-magnitude) x/y pair like this test's is nowhere near the
        # NaN/inf blowup this guards against (reproduced above without the
        # cap).
        self.assertTrue(np.all(np.abs(theta_hat) < 1e3))
        self.assertGreater(_RLS_P_NORM_CEILING, 0.0)
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
        room_a_features = [
            *_BASE_FEATURE_NAMES,
            "group_duty",
            "neighbor_diff::B",
            "door_x_neighbor_diff::B",
        ]
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


class TestPredictOneStepHistoryIsTeacherForced(unittest.TestCase):
    """Hand-computed checks that predict_one_step_history (a) uses the
    TRUE previous room_temp at every step (teacher-forced), never its own
    prior prediction, unlike predict_recursive's closed-loop design, and
    (b) threads a neighbor's true previous temperature the same way."""

    def _build_model(self) -> SelfLearningPhysicsModel:
        house_feature_names = [*_BASE_FEATURE_NAMES, "group_duty"]
        # _physics_features appends BOTH "neighbor_diff::<name>" and
        # "door_x_neighbor_diff::<name>" per declared neighbor (self.py:194-204)
        # - door coefficient left at 0 here since these tests don't exercise it.
        room_a_features = [
            *_BASE_FEATURE_NAMES,
            "group_duty",
            "neighbor_diff::B",
            "door_x_neighbor_diff::B",
        ]
        room_b_features = [*_BASE_FEATURE_NAMES, "group_duty"]

        theta_a = np.zeros(len(room_a_features))
        theta_a[room_a_features.index("bias")] = 15.0
        theta_a[room_a_features.index("room_last")] = 0.5
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

    def _history_frame(self, room_temp: list[float]) -> pd.DataFrame:
        n = len(room_temp)
        idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
        return pd.DataFrame(
            {
                "room_temp": room_temp,
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

    def test_uses_true_previous_temperature_not_own_prior_prediction(self):
        """Room A: bias=15.0, room_last=0.5, no neighbor effect (B held
        flat so neighbor_diff stays 0 throughout). With a KNOWN, deliberately
        volatile room_temp history [18.0, 30.0, 5.0, 20.0], each one-step
        prediction must use the TRUE previous value:
            pred[0] = 15.0 + 0.5*20.0 (shift(1).ffill() default seed for the
                       very first row's undefined "previous" - see below)
            pred[1] = 15.0 + 0.5*18.0 = 24.0
            pred[2] = 15.0 + 0.5*30.0 = 30.0
            pred[3] = 15.0 + 0.5*5.0  = 17.5
        None of these depend on any OTHER predicted value - proving this
        is teacher-forced, not chained/compounding like predict_recursive."""
        model = self._build_model()
        # Zero the neighbor_diff::B coefficient: this test isolates room_last
        # alone (neighbor coupling has its own dedicated test below), and B's
        # own true-previous value still moves between rows even when its
        # observed history is flat, so leaving this nonzero would fold an
        # extra neighbor term into the hand-computed expectations below.
        model.room_models_["A"].theta_temp[
            model.room_models_["A"].feature_names.index("neighbor_diff::B")
        ] = 0.0
        df_room_a = self._history_frame([18.0, 30.0, 5.0, 20.0])
        df_room_b = self._history_frame([19.0, 19.0, 19.0, 19.0])

        predictions = model.predict_one_step_history(
            "A", df_room_a, dfs_by_room={"A": df_room_a, "B": df_room_b}
        )

        self.assertEqual(len(predictions), 4)
        # _physics_features's own room_temp.shift(1).ffill().fillna(20.0)
        # convention: row 0 has no real "previous" row, so shift(1) is NaN,
        # ffill() leaves it NaN (nothing before it to fill from), and it
        # finally falls back to the fixed 20.0 default.
        np.testing.assert_allclose(predictions, [15.0 + 0.5 * 20.0, 24.0, 30.0, 17.5], atol=1e-9)

    def test_neighbor_diff_uses_neighbors_true_previous_temperature(self):
        model = self._build_model()
        # A: room_last coefficient is 0 here (rebuild theta without it) so
        # only the neighbor_diff term drives the result - isolates the
        # neighbor-threading behavior from the room's own room_last term.
        model.room_models_["A"].theta_temp[
            model.room_models_["A"].feature_names.index("room_last")
        ] = 0.0
        df_room_a = self._history_frame([18.0, 18.0, 18.0])
        df_room_b = self._history_frame([22.0, 25.0, 20.0])

        predictions = model.predict_one_step_history(
            "A", df_room_a, dfs_by_room={"A": df_room_a, "B": df_room_b}
        )

        # neighbor_diff[t] = B_true_previous[t] - A_true_previous[t]
        # Row 0: both default to 20.0 (no real previous row) -> diff=0.
        # Row 1: B_prev=22.0, A_prev=18.0 -> diff=4.0 -> 15.0+0.2*4.0=15.8
        # Row 2: B_prev=25.0, A_prev=18.0 -> diff=7.0 -> 15.0+0.2*7.0=16.4
        np.testing.assert_allclose(predictions, [15.0, 15.8, 16.4], atol=1e-9)

    def test_room_with_no_neighbors_ignores_dfs_by_room_argument(self):
        model = self._build_model()
        df_room_b = self._history_frame([19.0, 21.0, 17.0])

        predictions_with_none = model.predict_one_step_history("B", df_room_b, dfs_by_room=None)
        predictions_with_dict = model.predict_one_step_history(
            "B", df_room_b, dfs_by_room={"B": df_room_b}
        )

        np.testing.assert_allclose(predictions_with_none, [10.0, 10.0, 10.0])
        np.testing.assert_allclose(predictions_with_dict, [10.0, 10.0, 10.0])

    def test_differs_from_predict_recursive_when_history_is_volatile(self):
        """The whole point of teacher-forcing: for a volatile true history,
        predict_one_step_history's per-step predictions must NOT match
        predict_recursive's own closed-loop trajectory (which drifts away
        from the true history since it feeds its own prior guess forward)."""
        model = self._build_model()
        df_room_a = self._history_frame([18.0, 30.0, 5.0, 20.0])
        df_room_b = self._history_frame([19.0, 19.0, 19.0, 19.0])

        one_step = model.predict_one_step_history(
            "A", df_room_a, dfs_by_room={"A": df_room_a, "B": df_room_b}
        )
        recursive = model.predict_recursive(
            df_room_a, {"A": df_room_a, "B": df_room_b}, {"A": 18.0, "B": 19.0}
        )["room_temp"]["A"]

        self.assertFalse(
            np.allclose(one_step, recursive),
            "Teacher-forced and closed-loop predictions must diverge for a "
            "volatile true history - if they match, teacher-forcing isn't "
            "actually happening.",
        )


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
