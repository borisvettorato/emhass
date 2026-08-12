"""
Self-Learning Physics — Multi-Room Adaptive RC Thermal Model
===============================================================

Ports and extends the "SelfLearningPhysics" model from
``scripts/compare_ensemble.py`` (``_physics_features``/``_rls_fit_theta``/
``_recursive_physics_predict``) from single-zone to N rooms.

Physics basis
-------------
Each room's own temperature is fit as a linear function (Recursive Least
Squares, i.e. online-adaptive linear regression with a forgetting factor) of
physics-motivated features: its own duty-weighted temperature lift, an
outdoor heat-loss driver, wind/solar terms, and - the extension this module
adds - a per-neighbor temperature-difference term:

    T_self[t] ~= theta @ [1, T_self[t-1], duty, duty*delta_supply,
                          delta_env, duty*delta_env, cold, wind, ...,
                          group_duty, (T_neighbor1[t-1] - T_self[t-1]), ...]

A fitted neighbor-diff coefficient *is* a learned thermal coupling
coefficient - it plays exactly the same role as the manually-entered
``heatpump_room_coupling_conductance`` (kW/K) that
``optimization.py::_add_room_thermal_coupling_constraints`` already folds
into the live dispatch, just expressed in the RLS model's own per-step-delta
units instead. ``coupling_coefficients_kw_per_k`` converts between the two
representations.

Whole-house electric power and (unless ``electric_only``) gas consumption
are fit the same way, on a *single* shared feature set that does not vary
per room, using an aggregate "group_duty" feature (this heat pump's combined
duty across every room/dispatch load it drives, not any one room's own duty)
so the model reflects multiple rooms drawing on one shared unit rather than
a single whole-house sensor reading.

N independent per-room fits, not one joint fit: different rooms have
different physical parameters (volume, envelope, losses), so a single
room-invariant coefficient vector wouldn't correspond to any real physics -
and each room's own neighbor-diff feature set is a different width anyway
(a room with 3 declared neighbors needs 3 extra columns, a room with 0 needs
none), so a joint design matrix isn't even naturally the same shape across
rooms.

Confound this design guards against: rooms sharing one heat pump have
correlated duty (they compete for the same capacity cap) even with zero real
wall conductance between them. Without ``group_duty`` as an explicit control
feature in every room's own regression, a learned "coupling" coefficient
risks re-detecting that shared-driver correlation rather than true thermal
conductance - this is why ``group_duty`` is a required input, not optional.

Recursive/closed-loop prediction is joint across all rooms even though
fitting is independent: each room's step-k prediction needs its neighbors'
step-(k-1) *predicted* temperature (not the ground truth), matching
``_recursive_physics_predict``'s existing non-teacher-forced design in
``scripts/compare_ensemble.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trimmed toward build_heatpump_features's curated column-count philosophy
# (src/emhass/thermal/hybrid_heatpump_lr.py) rather than porting compare_
# ensemble.py's raw 39-wide sun/wind cross-term set unchanged: many of those
# cross-terms are heavily collinear with each other, which ridge/RLS
# regularisation only partially masks, and a smaller feature set is easier
# to keep consistent with the already-shipped HybridHeatPumpLR sibling.
_BASE_FEATURE_NAMES = [
    "bias",
    "room_last",
    "duty",
    "delta_supply",
    "duty_x_delta_supply",
    "delta_env",
    "duty_x_delta_env",
    "cold_below_2c",
    "wind_speed",
    "wind_x_outdoor",
    "dni",
    "dhi",
    "sun_alt_sin",
    "blind_x_dni",
]


def _physics_features(
    df: pd.DataFrame,
    *,
    room_state: pd.Series | None = None,
    group_duty: pd.Series | None = None,
    neighbor_diffs: dict[str, pd.Series] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build physics-motivated features for the online RLS model.

    ``room_state`` must be the latest observed/predicted room temperature
    before each row's timestamp (non-leaky) - falls back to an in-frame
    shift when not provided, matching ``build_heatpump_features``'s ``_col``
    convention of defaulting rather than raising on a missing column.

    ``group_duty`` is the shared heat pump's aggregate duty (see
    ``utils.compute_aggregate_heatpump_duty``), a required confound-control
    feature - see module docstring.

    ``neighbor_diffs``, if given, appends one column per entry
    (``neighbor_name -> (T_neighbor_last - T_self_last)`` Series aligned to
    ``df.index``), in dict-iteration (insertion) order - callers must build
    this dict in the same fixed order at fit and predict time for a given
    room (``SelfLearningPhysicsModel`` does this via each room's own stored
    ``neighbors`` list).

    :return: (feature matrix, feature name list) - the name list lets
        ``coef_summary``/``coupling_coefficients_kw_per_k`` map a fitted
        coefficient back to what it means.
    """
    duty = df.get("heatpump_duty", pd.Series(0.0, index=df.index)).fillna(0.0)
    if room_state is not None:
        room_s = room_state.reindex(df.index).fillna(20.0)
    elif "room_temp" in df.columns:
        room_s = df["room_temp"].shift(1).ffill().fillna(20.0)
    else:
        room_s = pd.Series(20.0, index=df.index)
    outdoor_s = df.get("outdoor_temp", pd.Series(10.0, index=df.index)).fillna(10.0)
    supply_default = room_s + 5.0
    supply_s = df.get("supply_temp", supply_default).fillna(supply_default)
    wind_speed_s = df.get("wind_speed", pd.Series(0.0, index=df.index)).fillna(0.0)
    dni_s = df.get("dni", pd.Series(0.0, index=df.index)).fillna(0.0)
    dhi_s = df.get("dhi", pd.Series(0.0, index=df.index)).fillna(0.0)
    sun_alt_sin_s = df.get("sun_alt_sin", pd.Series(0.0, index=df.index)).fillna(0.0)
    # Room's own live blind/shading position (0=open, 1=fully closed - see
    # heatpump_room_blind_sensors), merged in per-room by the caller exactly
    # like room_temp/dni/dhi already are (utils.py::_resolve_room_blind_entity_map).
    # blind_x_dni lets the RLS fit empirically learn how much THIS room's own
    # shading device actually cuts direct solar gain - no hand-specified
    # blocking percentage or window compass orientation needed anywhere in
    # this model, unlike the physics-family's calculate_shaded_window_irradiance.
    blind_s = df.get("blind_position", pd.Series(0.0, index=df.index)).fillna(0.0)

    room = room_s.to_numpy(dtype=float)
    outdoor = outdoor_s.to_numpy(dtype=float)
    supply = supply_s.to_numpy(dtype=float)
    duty_arr = duty.to_numpy(dtype=float)
    wind_speed = wind_speed_s.to_numpy(dtype=float)

    delta_supply = np.clip(supply - room, a_min=0.0, a_max=None)
    delta_env = np.clip(room - outdoor, a_min=0.0, a_max=None)
    cold = (outdoor < 2.0).astype(float)
    wind_outdoor = wind_speed * outdoor

    columns = [
        np.ones(len(df), dtype=float),
        room,
        duty_arr,
        delta_supply,
        duty_arr * delta_supply,
        delta_env,
        duty_arr * delta_env,
        cold,
        wind_speed,
        wind_outdoor,
        dni_s.to_numpy(dtype=float),
        dhi_s.to_numpy(dtype=float),
        sun_alt_sin_s.to_numpy(dtype=float),
        blind_s.to_numpy(dtype=float) * dni_s.to_numpy(dtype=float),
    ]
    feature_names = list(_BASE_FEATURE_NAMES)

    group_duty_s = group_duty if group_duty is not None else pd.Series(0.0, index=df.index)
    columns.append(group_duty_s.reindex(df.index).fillna(0.0).to_numpy(dtype=float))
    feature_names.append("group_duty")

    for neighbor_name, diff_series in (neighbor_diffs or {}).items():
        columns.append(diff_series.reindex(df.index).fillna(0.0).to_numpy(dtype=float))
        feature_names.append(f"neighbor_diff::{neighbor_name}")

    return np.column_stack(columns), feature_names


def _rls_fit_theta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    forgetting: float,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    """Fit an RLS model on training data and return fixed coefficients.

    Ported unchanged from scripts/compare_ensemble.py::_rls_fit_theta, plus
    a small diagnostics dict (n_obs, in-sample MAE) the refit action's
    deploy-quality gate can use.
    """
    n_feat = X_train.shape[1]
    theta = np.zeros(n_feat, dtype=float)
    p_mat = (1.0 / max(1e-9, ridge)) * np.eye(n_feat, dtype=float)
    abs_errs = []

    def _update(x_row: np.ndarray, y_val: float) -> None:
        nonlocal theta, p_mat
        x = x_row.reshape(-1, 1)
        denom = float(forgetting + (x.T @ p_mat @ x)[0, 0])
        gain = (p_mat @ x) / max(1e-9, denom)
        err = float(y_val - (theta @ x_row))
        abs_errs.append(abs(err))
        theta = theta + (gain[:, 0] * err)
        p_mat = (p_mat - (gain @ x.T @ p_mat)) / max(1e-9, forgetting)

    for xr, yr in zip(X_train, y_train, strict=False):
        _update(xr, float(yr))

    diagnostics = {
        "n_obs": len(y_train),
        # Mean absolute one-step prediction error *during* adaptation (not a
        # clean held-out score - the caller is expected to additionally
        # score on a true holdout split, same as refit_hybrid_heatpump_model).
        "in_sample_mae": float(np.mean(abs_errs)) if abs_errs else float("nan"),
    }
    return theta, diagnostics


@dataclass
class _RoomModel:
    """A single room's fitted temperature model."""

    theta_temp: np.ndarray
    feature_names: list[str]
    neighbors: list[str]


@dataclass
class SelfLearningPhysicsModel:
    """Multi-room online-adaptive (RLS) physics model.

    Parameters
    ----------
    forgetting_factor : float
        RLS forgetting factor (0-1]. Lower values adapt faster to recent
        data but are noisier; 1.0 is a plain (non-forgetting) recursive
        least squares fit.
    ridge : float
        Initial RLS covariance-matrix scale (``1/ridge`` on the diagonal) -
        larger values start more conservatively (slower initial adaptation).
    electric_only : bool
        When True, never fit/predict a gas component - mirrors
        HybridHeatPumpLR's own electric_only flag, for a pure-electric heat
        pump with no gas boiler.
    """

    forgetting_factor: float = 0.995
    ridge: float = 10.0
    electric_only: bool = False

    theta_elec_: np.ndarray | None = field(default=None, init=False, repr=False)
    theta_gas_: np.ndarray | None = field(default=None, init=False, repr=False)
    house_feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    room_models_: dict[str, _RoomModel] = field(default_factory=dict, init=False, repr=False)
    neighbor_map_: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    fit_diagnostics_: dict = field(default_factory=dict, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        df_house: pd.DataFrame,
        dfs_by_room: dict[str, pd.DataFrame],
        y_elec: np.ndarray,
        y_gas: np.ndarray | None,
        neighbor_map: dict[str, list[str]],
    ) -> SelfLearningPhysicsModel:
        """Fit the whole-house electric/gas models and every room's own
        temperature model.

        :param df_house: Whole-house training frame (outdoor_temp,
            wind_speed, dni, dhi, supply_temp, heatpump_duty, plus a
            "group_duty" column - the aggregate duty across every room/
            dispatch load this heat pump drives, see
            utils.compute_aggregate_heatpump_duty).
        :param dfs_by_room: room_name -> per-room training frame, each
            sharing df_house's index, containing at least "room_temp"
            (that room's own temperature history) plus the same physics
            columns as df_house (weather is shared/broadcast; "heatpump_duty"
            here should be that room's OWN duty where available).
        :param y_elec: Whole-house electric power training target [W].
        :param y_gas: Whole-house gas consumption training target
            [m3/interval], or None when electric_only.
        :param neighbor_map: room_name -> list of neighbor room names this
            room is thermally coupled to (operates on room names, not
            optimization.py's absolute load indices - that translation
            happens in command_line.py).
        """
        self.neighbor_map_ = {k: list(v) for k, v in neighbor_map.items()}
        self.fit_diagnostics_ = {}

        group_duty = df_house.get("group_duty", pd.Series(0.0, index=df_house.index))
        X_house, house_feature_names = _physics_features(df_house, group_duty=group_duty)
        self.house_feature_names_ = house_feature_names

        self.theta_elec_, elec_diag = _rls_fit_theta(
            X_house, np.asarray(y_elec, dtype=float),
            forgetting=self.forgetting_factor, ridge=self.ridge,
        )
        self.fit_diagnostics_["electric_power"] = elec_diag
        logger.info(
            "SelfLearningPhysicsModel: electric model fitted (%d samples, %d features)",
            elec_diag["n_obs"], X_house.shape[1],
        )

        if self.electric_only or y_gas is None:
            self.theta_gas_ = None
        else:
            self.theta_gas_, gas_diag = _rls_fit_theta(
                X_house, np.asarray(y_gas, dtype=float),
                forgetting=self.forgetting_factor, ridge=self.ridge,
            )
            self.fit_diagnostics_["gas_consumption"] = gas_diag

        self.room_models_ = {}
        for room_name, df_room in dfs_by_room.items():
            if "room_temp" not in df_room.columns:
                logger.warning(
                    "SelfLearningPhysicsModel: room %s has no room_temp column, skipping", room_name
                )
                continue
            room_temp_last = df_room["room_temp"].shift(1).ffill().fillna(20.0)
            neighbor_diffs: dict[str, pd.Series] = {}
            for neighbor_name in neighbor_map.get(room_name, []):
                df_neighbor = dfs_by_room.get(neighbor_name)
                if df_neighbor is None or "room_temp" not in df_neighbor.columns:
                    logger.warning(
                        "SelfLearningPhysicsModel: room %s declares neighbor %s with no "
                        "data available - that pair is skipped for this fit.",
                        room_name, neighbor_name,
                    )
                    continue
                neighbor_last = (
                    df_neighbor["room_temp"].shift(1).ffill().fillna(20.0).reindex(df_room.index)
                )
                neighbor_diffs[neighbor_name] = (neighbor_last - room_temp_last).fillna(0.0)

            group_duty_room = df_room.get("group_duty", group_duty.reindex(df_room.index)).fillna(0.0)
            X_room, room_feature_names = _physics_features(
                df_room, group_duty=group_duty_room, neighbor_diffs=neighbor_diffs
            )
            y_temp = df_room["room_temp"].fillna(20.0).to_numpy(dtype=float)
            theta_temp, temp_diag = _rls_fit_theta(
                X_room, y_temp, forgetting=self.forgetting_factor, ridge=self.ridge
            )
            self.fit_diagnostics_[f"room_temp::{room_name}"] = temp_diag
            self.room_models_[room_name] = _RoomModel(
                theta_temp=theta_temp,
                feature_names=room_feature_names,
                neighbors=list(neighbor_diffs.keys()),
            )
            logger.info(
                "SelfLearningPhysicsModel: room %s temperature model fitted "
                "(%d samples, %d neighbors)",
                room_name, temp_diag["n_obs"], len(neighbor_diffs),
            )

        self._is_fitted = True
        return self

    def predict_recursive(
        self,
        df_house_fc: pd.DataFrame,
        dfs_by_room_fc: dict[str, pd.DataFrame],
        initial_room_states: dict[str, float],
        initial_house_elec: float = 0.0,
        initial_house_gas: float = 0.0,
    ) -> dict:
        """Closed-loop (non-teacher-forced) multi-room forecast.

        Every room's step-k prediction uses its neighbors' step-(k-1)
        *predicted* temperature (a shared pre-step snapshot every room
        reads from, then all rooms' states are advanced together) - not the
        ground truth, matching
        scripts/compare_ensemble.py::_recursive_physics_predict's existing
        design, extended across rooms instead of within just one.

        :return: {"room_temp": {room_name: np.ndarray}, "electric_power":
            np.ndarray, "gas_consumption": np.ndarray | None}
        """
        self._check_fitted()
        n = len(df_house_fc)
        room_names = list(self.room_models_.keys())
        room_states = {name: float(initial_room_states.get(name, 20.0)) for name in room_names}
        pred_temp = {name: np.zeros(n, dtype=float) for name in room_names}
        pred_elec = np.zeros(n, dtype=float)
        pred_gas = np.zeros(n, dtype=float) if self.theta_gas_ is not None else None

        for i in range(n):
            snapshot = dict(room_states)

            house_row = df_house_fc.iloc[[i]]
            group_duty_row = house_row.get(
                "group_duty", pd.Series([0.0], index=house_row.index)
            )
            x_house, _ = _physics_features(house_row, group_duty=group_duty_row)
            pred_elec[i] = max(0.0, float(self.theta_elec_ @ x_house[0]))
            if self.theta_gas_ is not None:
                pred_gas[i] = max(0.0, float(self.theta_gas_ @ x_house[0]))

            new_states = {}
            for name in room_names:
                room_model = self.room_models_[name]
                df_room = dfs_by_room_fc.get(name)
                if df_room is None or i >= len(df_room):
                    new_states[name] = snapshot[name]
                    continue
                row = df_room.iloc[[i]]
                state_series = pd.Series([snapshot[name]], index=row.index)
                neighbor_diffs = {}
                for neighbor_name in room_model.neighbors:
                    if neighbor_name not in snapshot:
                        continue
                    neighbor_diffs[neighbor_name] = pd.Series(
                        [snapshot[neighbor_name] - snapshot[name]], index=row.index
                    )
                group_duty_room_row = row.get("group_duty", group_duty_row.reindex(row.index))
                x_room, _ = _physics_features(
                    row,
                    room_state=state_series,
                    group_duty=group_duty_room_row,
                    neighbor_diffs=neighbor_diffs,
                )
                temp_val = float(room_model.theta_temp @ x_room[0])
                pred_temp[name][i] = temp_val
                new_states[name] = temp_val
            room_states = new_states

        return {"room_temp": pred_temp, "electric_power": pred_elec, "gas_consumption": pred_gas}

    def coef_summary(self) -> pd.DataFrame:
        """Return fitted coefficients for the whole-house and every room's
        model, for physics sanity-checks (e.g. a neighbor-diff coefficient
        should be small and positive for a lightly-coupled pair)."""
        self._check_fitted()
        rows = []
        for name, coef in zip(self.house_feature_names_, self.theta_elec_, strict=True):
            rows.append({"model": "electric", "room": "__house__", "feature": name, "coef": float(coef)})
        if self.theta_gas_ is not None:
            for name, coef in zip(self.house_feature_names_, self.theta_gas_, strict=True):
                rows.append({"model": "gas", "room": "__house__", "feature": name, "coef": float(coef)})
        for room_name, room_model in self.room_models_.items():
            for name, coef in zip(room_model.feature_names, room_model.theta_temp, strict=True):
                rows.append({"model": "room_temp", "room": room_name, "feature": name, "coef": float(coef)})
        return pd.DataFrame(rows)

    def coupling_coefficients_kw_per_k(
        self,
        room_thermal_mass_kj_per_k: dict[str, float],
        dt_hours: float,
    ) -> dict[tuple[str, str], float]:
        """Convert each room's fitted neighbor-diff coefficients into the
        same kW/K units as the manually-entered
        heatpump_room_coupling_conductance field.

        A room's one-step regression is
        ``T_self[t] = ... + theta_diff * (T_neighbor[t-1] - T_self[t-1])``.
        optimization.py's own physical recurrence is
        ``T_self[t+1] = T_self[t] + conversion*(... + g*dt*(T_neighbor[t] - T_self[t]))``
        (see _add_thermal_battery_constraints/_add_room_thermal_coupling_constraints),
        where ``conversion = 3600/(density*heat_capacity*volume)``. Matching
        the two forms: ``theta_diff == conversion * g * dt_hours``, so
        ``g = theta_diff / (conversion * dt_hours)``.

        :param room_thermal_mass_kj_per_k: room_name -> density*heat_capacity*volume
            (kJ/K), i.e. the reciprocal of optimization.py's own `conversion`
            factor (in the same density/heat_capacity/volume units used
            there - default density=2400 kg/m3, heat_capacity=0.88 kJ/(kg*K)
            if the caller has no better estimate for a room).
        :param dt_hours: The timestep this model was fit/predicted at, in hours.
        :return: {(room_a, room_b): conductance_kw_per_k} with room_a < room_b
            alphabetically, averaging both directions' estimates when both
            rooms declared the pair.
        """
        self._check_fitted()
        raw: dict[tuple[str, str], list[float]] = {}
        for room_name, room_model in self.room_models_.items():
            mass = room_thermal_mass_kj_per_k.get(room_name)
            if not mass or mass <= 0:
                continue
            conversion = 3600.0 / mass
            for neighbor_name in room_model.neighbors:
                feature_name = f"neighbor_diff::{neighbor_name}"
                if feature_name not in room_model.feature_names:
                    continue
                idx = room_model.feature_names.index(feature_name)
                theta_diff = float(room_model.theta_temp[idx])
                g = theta_diff / (conversion * dt_hours) if conversion * dt_hours else 0.0
                key = tuple(sorted((room_name, neighbor_name)))
                raw.setdefault(key, []).append(g)
        return {key: float(np.mean(values)) for key, values in raw.items()}

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("SelfLearningPhysicsModel is not fitted yet. Call .fit() first.")
