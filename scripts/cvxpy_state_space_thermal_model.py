"""CVXPY-friendly thermal state-space model benchmark.

The goal of this script is not to optimize yet. It first checks whether a
linear state-space model that can later be written as CVXPY constraints predicts
the house well enough.

States:
- T_air: indoor air temperature
- T_mass: slow thermal mass temperature
- Q_emit: delayed heat emitter state driven by a heat input proxy

The fitted temperature dynamics are affine in the states for fixed weather and
control inputs, so the same equations can be reused in a CVXPY optimizer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import thermal_mass_physics_model as tm  # noqa: E402
from compare_ensemble import _build_aligned_index, _load_raw_df  # noqa: E402
from emhass.thermal.forecast_gridsearch import (  # noqa: E402
    SearchOptions,
    _prepare_features,
    create_sequences,
    split_sequences,
)


TEMP_PARAM_NAMES = [
    "emit_gain_per_h",
    "ua_base_per_h",
    "ua_wind_per_h_per_speed",
    "ua_wind_sin_per_h_per_speed",
    "ua_wind_cos_per_h_per_speed",
    "mass_gain_per_h",
    "solar_gain_c_per_h",
    "solar_alt_sin_gain_c_per_h",
    "solar_alt_cos_gain_c_per_h",
    "solar_az_sin_gain_c_per_h",
    "solar_az_cos_gain_c_per_h",
    "humidity_gain_c_per_h",
    "bias_c_per_h",
]

TEMP_FEATURE_COLS = [
    "q_emit",
    "outdoor_minus_air",
    "wind_outdoor_minus_air",
    "wind_sin_outdoor_minus_air",
    "wind_cos_outdoor_minus_air",
    "mass_minus_air",
    "q_solar",
    "q_solar_alt_sin",
    "q_solar_alt_cos",
    "q_solar_az_sin",
    "q_solar_az_cos",
    "humidity_anomaly",
    "bias",
]


@dataclass
class CvxSimResult:
    room: np.ndarray
    air_before: np.ndarray
    mass: np.ndarray
    q_emit: np.ndarray
    q_hp: np.ndarray
    d_air_dt: np.ndarray
    loss_coeff: np.ndarray


@dataclass
class LinearHead:
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    feature_cols: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.feature_cols].to_numpy(dtype=float)
        Xs = (Xv - self.mean) / self.scale
        return Xs @ self.coef + self.intercept


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(np.mean(err)),
    }


def _parse_grid(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _slice(raw_df: pd.DataFrame, ts: np.ndarray) -> pd.DataFrame:
    return raw_df.reindex(pd.DatetimeIndex(ts)).dropna(how="all")


def _prepare_inputs(df: pd.DataFrame, config: dict) -> tm.ThermalInputs:
    return tm._prepare_inputs(
        df,
        latitude=float(config["latitude"]),
        longitude=float(config["longitude"]),
        facade_azimuth_deg=float(config["facade_azimuth_deg"]),
        facade_tilt_deg=float(config["facade_tilt_deg"]),
        solar_horizontal_weight=float(config["solar_horizontal_weight"]),
        solar_facade_weight=float(config["solar_facade_weight"]),
    )


def _humidity(df: pd.DataFrame) -> np.ndarray:
    if "humidity" in df.columns:
        return pd.to_numeric(df["humidity"], errors="coerce").ffill().bfill().fillna(70.0).to_numpy(dtype=float)
    return np.full(len(df), 70.0, dtype=float)


def _split_timestamps(data_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    opts = SearchOptions(
        lookahead=args.lookahead,
        feature_level=args.feature_level,
        target_cols=target_cols,
        latitude=args.latitude,
        longitude=args.longitude,
        seed=args.seed,
    )
    X_raw, y_raw, _, _, _ = _prepare_features(data_path, opts=opts)
    aligned_index = _build_aligned_index(data_path, opts)
    X_seq, y_seq = create_sequences(X_raw, y_raw, lookback=args.input_window, lookahead=args.lookahead)
    X_train, _, X_val, _, X_test, _ = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_train), len(X_val)
    n_seq = len(X_seq)
    seq_start_ts = aligned_index[args.input_window : args.input_window + n_seq].to_numpy()
    train_ts = seq_start_ts[:n_train]
    val_ts = seq_start_ts[n_train : n_train + n_val]
    train_val_ts = seq_start_ts[: n_train + n_val]
    test_ts = seq_start_ts[n_train + n_val :]
    return train_ts, val_ts, train_val_ts, test_ts


def _heat_input_from_controls(duty: float, supply: float, air: float) -> float:
    return float(np.clip(duty, 0.0, 1.0) * max(float(supply) - float(air), 0.0))


def _roll_observed_states(
    inputs: tm.ThermalInputs,
    *,
    dt_h: float,
    tau_emit_h: float,
    mass_tau_h: float,
) -> dict[str, np.ndarray]:
    n = len(inputs.room)
    air_before = np.zeros(n, dtype=float)
    mass_series = np.zeros(n, dtype=float)
    q_emit_series = np.zeros(n, dtype=float)
    q_hp_series = np.zeros(n, dtype=float)

    if n == 0:
        return {
            "air_before": air_before,
            "mass": mass_series,
            "q_emit": q_emit_series,
            "q_hp": q_hp_series,
        }

    mass = float(inputs.room[0])
    q_emit = 0.0
    emit_alpha = float(np.clip(dt_h / max(tau_emit_h, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau_h, 1e-6), 0.0, 1.0))

    for i in range(n):
        air = float(inputs.room[max(0, i - 1)])
        q_hp = _heat_input_from_controls(inputs.duty[i], inputs.supply[i], air)
        q_emit = q_emit + emit_alpha * (q_hp - q_emit)
        mass = mass + mass_alpha * (air - mass)
        air_before[i] = air
        mass_series[i] = mass
        q_emit_series[i] = q_emit
        q_hp_series[i] = q_hp

    return {
        "air_before": air_before,
        "mass": mass_series,
        "q_emit": q_emit_series,
        "q_hp": q_hp_series,
    }


def _estimate_initial_states(
    history_inputs: tm.ThermalInputs,
    *,
    dt_h: float,
    tau_emit_h: float,
    mass_tau_h: float,
    warmup_steps: int,
) -> tuple[float, float, float]:
    if len(history_inputs.room) == 0:
        return 20.0, 20.0, 0.0

    start = max(0, len(history_inputs.room) - warmup_steps)
    air = float(history_inputs.room[start])
    mass = air
    q_emit = 0.0
    emit_alpha = float(np.clip(dt_h / max(tau_emit_h, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau_h, 1e-6), 0.0, 1.0))

    for i in range(start, len(history_inputs.room)):
        observed_air = float(history_inputs.room[i])
        q_hp = _heat_input_from_controls(history_inputs.duty[i], history_inputs.supply[i], observed_air)
        q_emit = q_emit + emit_alpha * (q_hp - q_emit)
        mass = mass + mass_alpha * (observed_air - mass)
        air = observed_air
    return air, mass, q_emit


def _temperature_features(
    inputs: tm.ThermalInputs,
    *,
    air_before: np.ndarray,
    mass: np.ndarray,
    q_emit: np.ndarray,
    humidity: np.ndarray,
    humidity_center: float,
) -> np.ndarray:
    env_gap = inputs.outdoor - air_before
    wind_env_gap = inputs.wind_speed * env_gap
    return np.column_stack(
        [
            q_emit,
            env_gap,
            wind_env_gap,
            inputs.wind_speed * inputs.wind_sin * env_gap,
            inputs.wind_speed * inputs.wind_cos * env_gap,
            mass - air_before,
            inputs.q_solar,
            inputs.q_solar * inputs.sun_alt_sin,
            inputs.q_solar * inputs.sun_alt_cos,
            inputs.q_solar * inputs.sun_az_sin,
            inputs.q_solar * inputs.sun_az_cos,
            (humidity - humidity_center) / 100.0,
            np.ones(len(air_before), dtype=float),
        ]
    )


def _loss_coeff(theta: np.ndarray, inputs: tm.ThermalInputs) -> np.ndarray:
    return (
        theta[1]
        + theta[2] * inputs.wind_speed
        + theta[3] * inputs.wind_speed * inputs.wind_sin
        + theta[4] * inputs.wind_speed * inputs.wind_cos
    )


def _solve_temperature_theta(
    X: np.ndarray,
    y: np.ndarray,
    *,
    inputs: tm.ThermalInputs,
    constraint_stride: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    theta = cp.Variable(len(TEMP_PARAM_NAMES))
    residual = X @ theta - y
    objective = cp.Minimize(cp.sum_squares(residual) / max(len(y), 1) + ridge * cp.sum_squares(theta))

    constraints = [
        theta[0] >= 0.0,
        theta[0] <= 1.0,
        theta[1] >= 0.0,
        theta[1] <= 0.30,
        theta[2] >= 0.0,
        theta[2] <= 0.08,
        theta[3] >= -0.05,
        theta[3] <= 0.05,
        theta[4] >= -0.05,
        theta[4] <= 0.05,
        theta[5] >= 0.0,
        theta[5] <= 0.80,
        theta[6] >= 0.0,
        theta[6] <= 5.0,
        theta[7] >= -3.0,
        theta[7] <= 3.0,
        theta[8] >= -3.0,
        theta[8] <= 3.0,
        theta[9] >= -3.0,
        theta[9] <= 3.0,
        theta[10] >= -3.0,
        theta[10] <= 3.0,
        theta[11] >= -2.0,
        theta[11] <= 2.0,
        theta[12] >= -0.30,
        theta[12] <= 0.30,
    ]
    stride = max(1, int(constraint_stride))
    speed = inputs.wind_speed[::stride]
    wind_sin = inputs.wind_sin[::stride]
    wind_cos = inputs.wind_cos[::stride]
    sampled_loss = (
        theta[1]
        + cp.multiply(speed, theta[2])
        + cp.multiply(speed * wind_sin, theta[3])
        + cp.multiply(speed * wind_cos, theta[4])
    )
    constraints.extend([sampled_loss >= 0.0, sampled_loss <= 0.50])

    problem = cp.Problem(objective, constraints)
    last_error = ""
    for solver in ("HIGHS", "CLARABEL", "OSQP", "SCS"):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            last_error = f"{solver}: {exc}"
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and theta.value is not None:
            values = np.asarray(theta.value, dtype=float)
            pred = X @ values
            return values, {
                "solver": solver,
                "status": str(problem.status),
                "objective": float(problem.value),
                "fit_mae_c_per_h": float(np.mean(np.abs(pred - y))),
                "success": True,
            }
        last_error = f"{solver}: status={problem.status}"
    raise RuntimeError(f"Could not solve CVXPY temperature fit: {last_error}")


def _fit_temperature_model(
    df: pd.DataFrame,
    config: dict,
    *,
    dt_h: float,
    tau_emit_h: float,
    mass_tau_h: float,
    humidity_center: float,
    ridge: float,
    constraint_stride: int,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    inputs = _prepare_inputs(df, config)
    states = _roll_observed_states(inputs, dt_h=dt_h, tau_emit_h=tau_emit_h, mass_tau_h=mass_tau_h)
    humidity = _humidity(df)
    X_all = _temperature_features(
        inputs,
        air_before=states["air_before"],
        mass=states["mass"],
        q_emit=states["q_emit"],
        humidity=humidity,
        humidity_center=humidity_center,
    )
    y_all = (inputs.room - states["air_before"]) / dt_h
    mask = np.arange(len(y_all)) > 0
    mask &= np.isfinite(y_all)
    mask &= np.all(np.isfinite(X_all), axis=1)
    fit_inputs = tm.ThermalInputs(
        index=inputs.index[mask],
        room=inputs.room[mask],
        electric=inputs.electric[mask],
        gas=inputs.gas[mask],
        duty=inputs.duty[mask],
        supply=inputs.supply[mask],
        outdoor=inputs.outdoor[mask],
        wind_speed=inputs.wind_speed[mask],
        wind_sin=inputs.wind_sin[mask],
        wind_cos=inputs.wind_cos[mask],
        q_solar=inputs.q_solar[mask],
        sun_alt_sin=inputs.sun_alt_sin[mask],
        sun_alt_cos=inputs.sun_alt_cos[mask],
        sun_az_sin=inputs.sun_az_sin[mask],
        sun_az_cos=inputs.sun_az_cos[mask],
        heatpump_duty=inputs.heatpump_duty[mask],
    )
    return _solve_temperature_theta(
        X_all[mask],
        y_all[mask],
        inputs=fit_inputs,
        constraint_stride=constraint_stride,
        ridge=ridge,
    )


def _simulate_open_loop(
    inputs: tm.ThermalInputs,
    humidity: np.ndarray,
    theta: np.ndarray,
    *,
    dt_h: float,
    tau_emit_h: float,
    mass_tau_h: float,
    humidity_center: float,
    initial_air: float,
    initial_mass: float,
    initial_q_emit: float,
) -> CvxSimResult:
    n = len(inputs.room)
    pred_room = np.zeros(n, dtype=float)
    air_before = np.zeros(n, dtype=float)
    mass_series = np.zeros(n, dtype=float)
    q_emit_series = np.zeros(n, dtype=float)
    q_hp_series = np.zeros(n, dtype=float)
    d_air_dt_series = np.zeros(n, dtype=float)
    loss_coeff_series = np.zeros(n, dtype=float)

    air = float(initial_air)
    mass = float(initial_mass)
    q_emit = float(initial_q_emit)
    emit_alpha = float(np.clip(dt_h / max(tau_emit_h, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau_h, 1e-6), 0.0, 1.0))

    for i in range(n):
        q_hp = _heat_input_from_controls(inputs.duty[i], inputs.supply[i], air)
        q_emit = q_emit + emit_alpha * (q_hp - q_emit)
        mass = mass + mass_alpha * (air - mass)
        one_input = tm.ThermalInputs(
            index=inputs.index[i : i + 1],
            room=inputs.room[i : i + 1],
            electric=inputs.electric[i : i + 1],
            gas=inputs.gas[i : i + 1],
            duty=inputs.duty[i : i + 1],
            supply=inputs.supply[i : i + 1],
            outdoor=inputs.outdoor[i : i + 1],
            wind_speed=inputs.wind_speed[i : i + 1],
            wind_sin=inputs.wind_sin[i : i + 1],
            wind_cos=inputs.wind_cos[i : i + 1],
            q_solar=inputs.q_solar[i : i + 1],
            sun_alt_sin=inputs.sun_alt_sin[i : i + 1],
            sun_alt_cos=inputs.sun_alt_cos[i : i + 1],
            sun_az_sin=inputs.sun_az_sin[i : i + 1],
            sun_az_cos=inputs.sun_az_cos[i : i + 1],
            heatpump_duty=inputs.heatpump_duty[i : i + 1],
        )
        X = _temperature_features(
            one_input,
            air_before=np.array([air], dtype=float),
            mass=np.array([mass], dtype=float),
            q_emit=np.array([q_emit], dtype=float),
            humidity=np.array([humidity[i]], dtype=float),
            humidity_center=humidity_center,
        )
        d_air_dt = float(X[0] @ theta)
        loss_coeff = float(_loss_coeff(theta, one_input)[0])
        air_before[i] = air
        q_hp_series[i] = q_hp
        q_emit_series[i] = q_emit
        mass_series[i] = mass
        d_air_dt_series[i] = d_air_dt
        loss_coeff_series[i] = loss_coeff
        air = float(np.clip(air + dt_h * d_air_dt, 5.0, 35.0))
        pred_room[i] = air

    return CvxSimResult(
        room=pred_room,
        air_before=air_before,
        mass=mass_series,
        q_emit=q_emit_series,
        q_hp=q_hp_series,
        d_air_dt=d_air_dt_series,
        loss_coeff=loss_coeff_series,
    )


def _fit_linear_head(X: pd.DataFrame, y: np.ndarray, *, ridge: float) -> LinearHead:
    feature_cols = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    mean = np.nanmean(Xv, axis=0)
    scale = np.nanstd(Xv, axis=0)
    scale[scale < 1e-8] = 1.0
    Xs = np.nan_to_num((Xv - mean) / scale)
    yv = np.asarray(y, dtype=float)

    beta = cp.Variable(Xs.shape[1])
    intercept = cp.Variable()
    residual = Xs @ beta + intercept - yv
    problem = cp.Problem(cp.Minimize(cp.sum_squares(residual) / max(len(yv), 1) + ridge * cp.sum_squares(beta)))
    last_error = ""
    for solver in ("HIGHS", "CLARABEL", "OSQP", "SCS"):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:
            last_error = f"{solver}: {exc}"
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and beta.value is not None:
            return LinearHead(
                coef=np.asarray(beta.value, dtype=float),
                intercept=float(intercept.value),
                mean=mean,
                scale=scale,
                feature_cols=feature_cols,
            )
        last_error = f"{solver}: status={problem.status}"
    raise RuntimeError(f"Could not solve CVXPY linear head: {last_error}")


def _power_feature_frame(
    inputs: tm.ThermalInputs,
    *,
    air_before: np.ndarray,
    mass: np.ndarray,
    q_emit: np.ndarray,
    q_hp: np.ndarray,
    humidity: np.ndarray,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(inputs.index)
    local = index.tz_convert("Europe/Amsterdam") if index.tz is not None else index.tz_localize("UTC").tz_convert("Europe/Amsterdam")
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0
    hour_rad = 2.0 * np.pi * hour / 24.0
    lift = np.clip(inputs.supply - inputs.outdoor, 0.0, None)
    supply_delta = np.clip(inputs.supply - air_before, 0.0, None)
    delta_env = np.clip(air_before - inputs.outdoor, 0.0, None)
    return pd.DataFrame(
        {
            "duty": inputs.duty,
            "supply_temp": inputs.supply,
            "outdoor_temp": inputs.outdoor,
            "lift": lift,
            "duty_lift": inputs.duty * lift,
            "duty_lift_sq": inputs.duty * lift * lift,
            "duty_supply": inputs.duty * inputs.supply,
            "supply_delta": supply_delta,
            "q_hp": q_hp,
            "q_emit": q_emit,
            "t_mass": mass,
            "mass_minus_air": mass - air_before,
            "delta_env": delta_env,
            "wind_speed": inputs.wind_speed,
            "wind_loss": inputs.wind_speed * delta_env,
            "wind_sin": inputs.wind_sin,
            "wind_cos": inputs.wind_cos,
            "q_solar": inputs.q_solar,
            "humidity": humidity,
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def _select_tau_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    config: dict,
    *,
    dt_h: float,
    emit_grid: list[float],
    mass_grid: list[float],
    humidity_center: float,
    warmup_steps: int,
    ridge: float,
    constraint_stride: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    val_inputs = _prepare_inputs(val_df, config)
    val_humidity = _humidity(val_df)
    history_before_val = raw_df.loc[raw_df.index < val_df.index[0]]
    history_inputs = _prepare_inputs(history_before_val, config)

    for tau_emit_h in emit_grid:
        for mass_tau_h in mass_grid:
            try:
                theta, fit_info = _fit_temperature_model(
                    train_df,
                    config,
                    dt_h=dt_h,
                    tau_emit_h=tau_emit_h,
                    mass_tau_h=mass_tau_h,
                    humidity_center=humidity_center,
                    ridge=ridge,
                    constraint_stride=constraint_stride,
                )
                initial_air, initial_mass, initial_q_emit = _estimate_initial_states(
                    history_inputs,
                    dt_h=dt_h,
                    tau_emit_h=tau_emit_h,
                    mass_tau_h=mass_tau_h,
                    warmup_steps=warmup_steps,
                )
                initial_air = tm._latest_before(raw_df, val_df.index[0], "room_temp", initial_air)
                sim = _simulate_open_loop(
                    val_inputs,
                    val_humidity,
                    theta,
                    dt_h=dt_h,
                    tau_emit_h=tau_emit_h,
                    mass_tau_h=mass_tau_h,
                    humidity_center=humidity_center,
                    initial_air=initial_air,
                    initial_mass=initial_mass,
                    initial_q_emit=initial_q_emit,
                )
                val_metrics = _metrics(val_inputs.room, sim.room)
                rows.append(
                    {
                        "tau_emit_h": tau_emit_h,
                        "mass_tau_h": mass_tau_h,
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "val_bias": val_metrics["bias"],
                        "solver": fit_info["solver"],
                        "fit_mae_c_per_h": fit_info["fit_mae_c_per_h"],
                        "success": True,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "tau_emit_h": tau_emit_h,
                        "mass_tau_h": mass_tau_h,
                        "val_rmse": np.nan,
                        "val_mae": np.nan,
                        "val_bias": np.nan,
                        "solver": "",
                        "fit_mae_c_per_h": np.nan,
                        "success": False,
                        "error": str(exc)[-300:],
                    }
                )

    grid_df = pd.DataFrame(rows).sort_values(["val_mae", "val_rmse"], na_position="last")
    if grid_df.empty or not bool(grid_df.iloc[0].get("success", False)):
        raise RuntimeError("No successful CVXPY tau-grid fit")
    best = grid_df.iloc[0]
    return float(best["tau_emit_h"]), float(best["mass_tau_h"]), grid_df


def _write_plot(pred_df: pd.DataFrame, metrics: dict, params: dict, output_path: Path) -> None:
    df = pred_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    x = df.index.tz_convert("Europe/Amsterdam")

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            "Room temperature",
            "Room temperature error",
            "CVXPY-friendly states",
            "Electric power",
            "Gas, duty, solar and wind-loss coefficient",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=df["true_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_room_temp"], mode="lines", name="CVXPYStateSpace", line=dict(color="#2563eb", width=1.9)), row=1, col=1)
    if "reference_room_temp" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["reference_room_temp"], mode="lines", name="ThermalMassPhysics reference", line=dict(color="#0f766e", width=1.4, dash="dot")), row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#6b7280", row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_room_temp"] - df["true_room_temp"], mode="lines", name="CVXPY error", line=dict(color="#dc2626", width=1.3), showlegend=False), row=2, col=1)
    if "reference_room_temp" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["reference_room_temp"] - df["true_room_temp"], mode="lines", name="Reference error", line=dict(color="#0f766e", width=1.0, dash="dot"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["t_mass"], mode="lines", name="T_mass", line=dict(color="#7c3aed", width=1.2)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["q_emit"], mode="lines", name="Q_emit", line=dict(color="#ea580c", width=1.2)), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["true_electric_power"], mode="lines", name="Actual electric", line=dict(color="#111827", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_electric_power"], mode="lines", name="Pred electric", line=dict(color="#2563eb", width=1.1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["true_gas_consumption"], mode="lines", name="Actual gas", line=dict(color="#111827", width=1.1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["pred_gas_consumption"], mode="lines", name="Pred gas", line=dict(color="#16a34a", width=1.1)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["heatpump_duty"], mode="lines", name="Heatpump duty", line=dict(color="#64748b", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["q_solar"], mode="lines", name="Q_solar", line=dict(color="#eab308", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["loss_coeff"], mode="lines", name="Loss coeff", line=dict(color="#0284c7", width=1.0)), row=5, col=1, secondary_y=True)

    cvx = metrics["CVXPYStateSpace"]["room_temp"]
    ref = metrics.get("ThermalMassPhysics", {}).get("room_temp")
    summary = f"CVXPYStateSpace room: MAE {cvx['mae']:.3f}, RMSE {cvx['rmse']:.3f}, bias {cvx['bias']:+.3f}"
    if ref:
        summary += f"<br>ThermalMassPhysics room: MAE {ref['mae']:.3f}, RMSE {ref['rmse']:.3f}, bias {ref['bias']:+.3f}"
    param_summary = "<br>".join(f"{k}: {v:.4g}" for k, v in params.items())
    fig.add_annotation(
        text=summary + "<br><br>" + param_summary,
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.995,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        font=dict(size=11),
    )
    fig.update_layout(
        title=f"CVXPY-friendly state-space model ({x.min().strftime('%Y-%m-%d %H:%M')} - {x.max().strftime('%Y-%m-%d %H:%M')} Europe/Amsterdam)",
        template="plotly_white",
        height=1350,
        width=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="degC error", row=2, col=1)
    fig.update_yaxes(title_text="T_mass degC", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Q_emit", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="W", row=4, col=1)
    fig.update_yaxes(title_text="m3/interval", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="duty / proxy", row=5, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=5, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and test a CVXPY-friendly thermal state-space model")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/cvxpy_state_space_thermal")
    parser.add_argument("--reference-physics-report-dir", default="tests_thermal/reports/thermal_mass_physics_sundir_20260519_final")
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--feature-level", choices=["minimal", "standard", "full"], default="standard")
    parser.add_argument("--target-cols", default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--latitude", type=float, default=51.65)
    parser.add_argument("--longitude", type=float, default=4.93)
    parser.add_argument("--facade-azimuth-deg", type=float, default=180.0)
    parser.add_argument("--facade-tilt-deg", type=float, default=90.0)
    parser.add_argument("--solar-horizontal-weight", type=float, default=0.35)
    parser.add_argument("--solar-facade-weight", type=float, default=0.65)
    parser.add_argument("--holdout-split", choices=["test"], default="test")
    parser.add_argument("--emit-tau-grid", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--mass-tau-grid", default="4,8,16,32,64,128")
    parser.add_argument("--state-warmup-steps", type=int, default=672)
    parser.add_argument("--temp-ridge", type=float, default=1e-4)
    parser.add_argument("--power-ridge", type=float, default=0.05)
    parser.add_argument("--gas-ridge", type=float, default=0.05)
    parser.add_argument("--gas-cutoff-temp", type=float, default=5.0)
    parser.add_argument("--constraint-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    np.random.seed(args.seed)
    data_path = Path(args.data_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "facade_azimuth_deg": args.facade_azimuth_deg,
        "facade_tilt_deg": args.facade_tilt_deg,
        "solar_horizontal_weight": args.solar_horizontal_weight,
        "solar_facade_weight": args.solar_facade_weight,
    }
    train_ts, val_ts, train_val_ts, test_ts = _split_timestamps(data_path, args)
    raw_df = _load_raw_df(data_path)
    dt_h = tm._infer_timestep_hours(raw_df.index)

    df_train = _slice(raw_df, train_ts)
    df_val = _slice(raw_df, val_ts)
    df_train_val = _slice(raw_df, train_val_ts)
    df_test = _slice(raw_df, test_ts)
    df_history = raw_df.loc[raw_df.index < df_test.index[0]].copy()
    humidity_center = float(np.nanmedian(_humidity(df_train_val)))

    tau_emit_h, mass_tau_h, grid_df = _select_tau_grid(
        df_train,
        df_val,
        raw_df,
        config,
        dt_h=dt_h,
        emit_grid=_parse_grid(args.emit_tau_grid),
        mass_grid=_parse_grid(args.mass_tau_grid),
        humidity_center=humidity_center,
        warmup_steps=args.state_warmup_steps,
        ridge=args.temp_ridge,
        constraint_stride=args.constraint_stride,
    )
    grid_df.to_csv(report_dir / "cvxpy_state_space_tau_grid.csv", index=False)

    theta, fit_info = _fit_temperature_model(
        df_train_val,
        config,
        dt_h=dt_h,
        tau_emit_h=tau_emit_h,
        mass_tau_h=mass_tau_h,
        humidity_center=humidity_center,
        ridge=args.temp_ridge,
        constraint_stride=args.constraint_stride,
    )
    test_inputs = _prepare_inputs(df_test, config)
    history_inputs = _prepare_inputs(df_history, config)
    test_humidity = _humidity(df_test)
    initial_air, initial_mass, initial_q_emit = _estimate_initial_states(
        history_inputs,
        dt_h=dt_h,
        tau_emit_h=tau_emit_h,
        mass_tau_h=mass_tau_h,
        warmup_steps=args.state_warmup_steps,
    )
    initial_air = tm._latest_before(raw_df, df_test.index[0], "room_temp", initial_air)
    test_sim = _simulate_open_loop(
        test_inputs,
        test_humidity,
        theta,
        dt_h=dt_h,
        tau_emit_h=tau_emit_h,
        mass_tau_h=mass_tau_h,
        humidity_center=humidity_center,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
    )

    train_val_inputs = _prepare_inputs(df_train_val, config)
    train_states = _roll_observed_states(train_val_inputs, dt_h=dt_h, tau_emit_h=tau_emit_h, mass_tau_h=mass_tau_h)
    train_humidity = _humidity(df_train_val)
    power_X_train = _power_feature_frame(
        train_val_inputs,
        air_before=train_states["air_before"],
        mass=train_states["mass"],
        q_emit=train_states["q_emit"],
        q_hp=train_states["q_hp"],
        humidity=train_humidity,
    )
    power_X_test = _power_feature_frame(
        test_inputs,
        air_before=test_sim.air_before,
        mass=test_sim.mass,
        q_emit=test_sim.q_emit,
        q_hp=test_sim.q_hp,
        humidity=test_humidity,
    )
    elec_head = _fit_linear_head(power_X_train, train_val_inputs.electric, ridge=args.power_ridge)
    gas_head = _fit_linear_head(power_X_train, train_val_inputs.gas, ridge=args.gas_ridge)
    pred_electric = np.clip(elec_head.predict(power_X_test), 0.0, None)
    pred_gas = np.clip(gas_head.predict(power_X_test), 0.0, None)
    pred_gas[test_inputs.outdoor > args.gas_cutoff_temp] = 0.0

    metrics: dict[str, dict[str, dict[str, float]]] = {
        "CVXPYStateSpace": {
            "room_temp": _metrics(test_inputs.room, test_sim.room),
            "electric_power": _metrics(test_inputs.electric, pred_electric),
            "gas_consumption": _metrics(test_inputs.gas, pred_gas),
        }
    }
    reference_room = None
    reference_dir = Path(args.reference_physics_report_dir)
    if (reference_dir / "thermal_mass_physics_metrics.json").exists():
        ref_payload = json.loads((reference_dir / "thermal_mass_physics_metrics.json").read_text(encoding="utf-8"))
        metrics["ThermalMassPhysics"] = ref_payload.get("metrics", {})
    if (reference_dir / "thermal_mass_physics_predictions.csv").exists():
        ref_pred = pd.read_csv(reference_dir / "thermal_mass_physics_predictions.csv")
        ref_pred["timestamp"] = pd.to_datetime(ref_pred["timestamp"], utc=True)
        ref_pred = ref_pred.set_index("timestamp").sort_index()
        reference_room = ref_pred.reindex(pd.DatetimeIndex(test_inputs.index))["pred_room_temp"].to_numpy(dtype=float)

    params_dict = {name: float(value) for name, value in zip(TEMP_PARAM_NAMES, theta, strict=True)}
    params_dict["tau_emit_h"] = float(tau_emit_h)
    params_dict["mass_tau_h"] = float(mass_tau_h)
    params_dict["humidity_center"] = float(humidity_center)

    pred_df = pd.DataFrame(
        {
            "timestamp": test_inputs.index,
            "true_room_temp": test_inputs.room,
            "pred_room_temp": test_sim.room,
            "true_electric_power": test_inputs.electric,
            "pred_electric_power": pred_electric,
            "true_gas_consumption": test_inputs.gas,
            "pred_gas_consumption": pred_gas,
            "t_mass": test_sim.mass,
            "q_emit": test_sim.q_emit,
            "q_hp": test_sim.q_hp,
            "q_solar": test_sim.q_emit * 0.0 + test_inputs.q_solar,
            "loss_coeff": test_sim.loss_coeff,
            "d_air_dt": test_sim.d_air_dt,
            "heatpump_duty": test_inputs.heatpump_duty,
            "outdoor_temp": test_inputs.outdoor,
            "supply_temp": test_inputs.supply,
            "humidity": test_humidity,
            "wind_speed": test_inputs.wind_speed,
        }
    )
    if reference_room is not None:
        pred_df["reference_room_temp"] = reference_room
    pred_df.to_csv(report_dir / "cvxpy_state_space_predictions.csv", index=False)

    rows = []
    for model_name, by_target in metrics.items():
        for target, vals in by_target.items():
            rows.append({"model": model_name, "target": target, **vals})
    pd.DataFrame(rows).to_csv(report_dir / "cvxpy_state_space_metrics.csv", index=False)
    pd.DataFrame([{"parameter": key, "value": value} for key, value in params_dict.items()]).to_csv(
        report_dir / "cvxpy_state_space_params.csv",
        index=False,
    )

    metadata = {
        "model": "CVXPYStateSpace",
        "data_path": str(data_path),
        "report_dir": str(report_dir),
        "reference_physics_report_dir": str(reference_dir),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_eval": int(len(test_inputs.room)),
        "dt_h": float(dt_h),
        "selected_tau_emit_h": float(tau_emit_h),
        "selected_mass_tau_h": float(mass_tau_h),
        "initial_state": {
            "T_air": float(initial_air),
            "T_mass": float(initial_mass),
            "Q_emit": float(initial_q_emit),
        },
        "fit_info": fit_info,
        "params": params_dict,
        "temperature_feature_cols": TEMP_FEATURE_COLS,
        "power_feature_cols": elec_head.feature_cols,
        "config": vars(args),
        "runtime_s": float(time.perf_counter() - started),
    }
    (report_dir / "cvxpy_state_space_metrics.json").write_text(
        json.dumps({"metrics": metrics, **metadata}, indent=2),
        encoding="utf-8",
    )
    _write_plot(pred_df, metrics, params_dict, report_dir / "cvxpy_state_space_plot.html")
    print(json.dumps({"metrics": metrics, "params": params_dict, "fit_info": fit_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
