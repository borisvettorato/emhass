"""One-day CVXPY optimizer for the CVXPY-friendly thermal state-space model.

This is a dry-run planner. It keeps the thermal dynamics linear and optimizes a
weather-curve offset plus heat input proxy. The offset mirrors weather-compensated
heat pumps where the real control is "curve +/- X degC".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cvxpy_state_space_thermal_model as cvxm  # noqa: E402
import thermal_mass_physics_model as tm  # noqa: E402
from compare_ensemble import _load_raw_df  # noqa: E402


def _price_forecast(
    df: pd.DataFrame,
    default_price: float,
    *,
    data_path: Path | None = None,
) -> np.ndarray:
    col = "sensor.current_electricity_market_price"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(default_price).to_numpy(dtype=float)
    if data_path is not None and Path(data_path).exists():
        try:
            price_df = pd.read_csv(data_path, usecols=["timestamp", col])
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
            price_df = price_df.set_index("timestamp").sort_index()
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            price = pd.to_numeric(price_df[col], errors="coerce").reindex(idx).ffill().bfill()
            return price.fillna(default_price).to_numpy(dtype=float)
        except Exception:
            pass
    return np.full(len(df), default_price, dtype=float)


def _fit_weather_curve(history_df: pd.DataFrame, *, min_supply: float, max_supply: float) -> tuple[float, float]:
    if {"outdoor_temp", "supply_temp", "heatpump_duty"}.issubset(history_df.columns):
        outdoor = pd.to_numeric(history_df["outdoor_temp"], errors="coerce")
        supply = pd.to_numeric(history_df["supply_temp"], errors="coerce")
        duty = pd.to_numeric(history_df["heatpump_duty"], errors="coerce")
        mask = (duty > 0.2) & outdoor.notna() & supply.notna()
        sample = pd.DataFrame({"outdoor": outdoor[mask], "supply": supply[mask]}).tail(14 * 96)
        sample = sample[(sample["supply"] >= min_supply - 2.0) & (sample["supply"] <= max_supply + 5.0)]
        if len(sample) >= 24 and sample["outdoor"].std() > 0.2:
            slope, intercept = np.polyfit(sample["outdoor"].to_numpy(dtype=float), sample["supply"].to_numpy(dtype=float), 1)
            return float(intercept), float(slope)
    return 30.0, -0.45


def _estimate_power_per_q(history_df: pd.DataFrame) -> float:
    if not {"room_temp", "supply_temp", "heatpump_duty", "electric_power"}.issubset(history_df.columns):
        return 45.0
    room = pd.to_numeric(history_df["room_temp"], errors="coerce").ffill()
    room_prev = room.shift(1).ffill()
    supply = pd.to_numeric(history_df["supply_temp"], errors="coerce").ffill()
    duty = pd.to_numeric(history_df["heatpump_duty"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    electric = pd.to_numeric(history_df["electric_power"], errors="coerce").clip(lower=0.0)
    q_raw = duty * (supply - room_prev).clip(lower=0.0)
    mask = (duty > 0.2) & (q_raw > 0.5) & electric.notna()
    ratio = (electric[mask] / q_raw[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 45.0
    return float(np.clip(ratio.median(), 20.0, 180.0))


def _reference_room(
    report_dir: Path,
    horizon_index: pd.DatetimeIndex,
    default_temp: float,
) -> np.ndarray:
    pred_path = report_dir / "cvxpy_state_space_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
        pred = pred.set_index("timestamp").sort_index()
        ref = pred.reindex(horizon_index)["pred_room_temp"].ffill().bfill()
        if ref.notna().all():
            return ref.to_numpy(dtype=float)
    return np.full(len(horizon_index), default_temp, dtype=float)


def _thermal_arrays(inputs: tm.ThermalInputs, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loss_coeff = (
        theta[1]
        + theta[2] * inputs.wind_speed
        + theta[3] * inputs.wind_speed * inputs.wind_sin
        + theta[4] * inputs.wind_speed * inputs.wind_cos
    )
    solar_const = (
        theta[6] * inputs.q_solar
        + theta[7] * inputs.q_solar * inputs.sun_alt_sin
        + theta[8] * inputs.q_solar * inputs.sun_alt_cos
        + theta[9] * inputs.q_solar * inputs.sun_az_sin
        + theta[10] * inputs.q_solar * inputs.sun_az_cos
    )
    return loss_coeff, solar_const


def _objective_summary(
    room: np.ndarray,
    p_hp: np.ndarray,
    prices: np.ndarray,
    *,
    dt_h: float,
    min_room: float,
    max_room: float,
) -> dict[str, float]:
    below = np.clip(min_room - room, 0.0, None)
    above = np.clip(room - max_room, 0.0, None)
    electric_kwh = p_hp / 1000.0 * dt_h
    return {
        "electric_cost_eur": float(np.sum(electric_kwh * prices)),
        "electric_kwh": float(np.sum(electric_kwh)),
        "comfort_degree_h": float(np.sum(below + above) * dt_h),
        "comfort_sq_degree_h": float(np.sum(below * below + above * above) * dt_h),
        "min_room_temp": float(np.min(room)),
        "max_room_temp": float(np.max(room)),
        "mean_room_temp": float(np.mean(room)),
        "mean_p_hp_w": float(np.mean(p_hp)),
    }


def _write_plot(plan_df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    df = plan_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    x = df.index.tz_convert("Europe/Amsterdam")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("Room temperature", "Weather-curve offset and supply", "Power and heat proxy", "Price, outdoor and solar"),
    )
    fig.add_trace(go.Scatter(x=x, y=df["actual_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_room_temp"], mode="lines", name="CVXPY optimized room", line=dict(color="#2563eb", width=2.0)), row=1, col=1)
    fig.add_hline(y=summary["config"]["min_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=summary["config"]["max_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_curve_offset"], mode="lines", name="Offset", line=dict(color="#7c3aed", width=1.5)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_supply_temp"], mode="lines", name="Supply", line=dict(color="#dc2626", width=1.4)), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_p_hp"], mode="lines", name="P_hp", line=dict(color="#2563eb", width=1.3)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_q_hp"], mode="lines", name="Q_hp proxy", line=dict(color="#ea580c", width=1.2)), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["electricity_price"], mode="lines", name="Price", line=dict(color="#7c3aed", width=1.1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["outdoor_temp"], mode="lines", name="Outdoor", line=dict(color="#0284c7", width=1.0)), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["q_solar"], mode="lines", name="Q solar", line=dict(color="#eab308", width=1.0)), row=4, col=1, secondary_y=True)
    fig.update_layout(template="plotly_white", height=1050, width=1450, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(title_text="Local time", row=4, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-day CVXPY thermal optimizer dry-run")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--model-report-dir", default="tests_thermal/reports/cvxpy_state_space_thermal_20260520")
    parser.add_argument("--report-dir", default="tests_thermal/reports/cvxpy_state_space_optimizer")
    parser.add_argument("--start", default="2026-05-08 00:00:00+00:00")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--min-room-temp", type=float, default=20.0)
    parser.add_argument("--max-room-temp", type=float, default=22.0)
    parser.add_argument("--min-supply-temp", type=float, default=25.0)
    parser.add_argument("--max-supply-temp", type=float, default=45.0)
    parser.add_argument("--min-offset", type=float, default=-10.0)
    parser.add_argument("--max-offset", type=float, default=10.0)
    parser.add_argument("--max-q-hp", type=float, default=25.0)
    parser.add_argument("--max-p-hp", type=float, default=3500.0)
    parser.add_argument("--default-electricity-price", type=float, default=0.25)
    parser.add_argument("--comfort-weight", type=float, default=45.0)
    parser.add_argument("--offset-weight", type=float, default=0.0008)
    parser.add_argument("--smooth-weight", type=float, default=0.002)
    parser.add_argument("--q-smooth-weight", type=float, default=0.001)
    args = parser.parse_args()

    started = time.perf_counter()
    data_path = Path(args.data_path)
    model_report_dir = Path(args.model_report_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads((model_report_dir / "cvxpy_state_space_metrics.json").read_text(encoding="utf-8"))
    params = payload["params"]
    theta = np.array([params[name] for name in cvxm.TEMP_PARAM_NAMES], dtype=float)
    tau_emit_h = float(params["tau_emit_h"])
    mass_tau_h = float(params["mass_tau_h"])
    humidity_center = float(params["humidity_center"])
    config = payload["config"]

    raw_df = _load_raw_df(data_path)
    start_ts = pd.Timestamp(args.start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    start_ts = start_ts.tz_convert(raw_df.index.tz or "UTC")
    horizon_df = raw_df.loc[raw_df.index >= start_ts].head(args.horizon).copy()
    if len(horizon_df) < args.horizon:
        raise ValueError(f"Not enough rows for horizon={args.horizon} from {start_ts}")
    history_df = raw_df.loc[raw_df.index < horizon_df.index[0]].copy()
    dt_h = tm._infer_timestep_hours(raw_df.index)

    prepare_config = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "facade_azimuth_deg": config["facade_azimuth_deg"],
        "facade_tilt_deg": config["facade_tilt_deg"],
        "solar_horizontal_weight": config["solar_horizontal_weight"],
        "solar_facade_weight": config["solar_facade_weight"],
    }
    inputs = cvxm._prepare_inputs(horizon_df, prepare_config)
    history_inputs = cvxm._prepare_inputs(history_df, prepare_config)
    humidity = cvxm._humidity(horizon_df)
    initial_air, initial_mass, initial_q_emit = cvxm._estimate_initial_states(
        history_inputs,
        dt_h=dt_h,
        tau_emit_h=tau_emit_h,
        mass_tau_h=mass_tau_h,
        warmup_steps=int(payload["config"]["state_warmup_steps"]),
    )
    initial_air = tm._latest_before(raw_df, horizon_df.index[0], "room_temp", initial_air)
    prices = _price_forecast(horizon_df, args.default_electricity_price, data_path=data_path)
    curve_intercept, curve_slope = _fit_weather_curve(
        history_df,
        min_supply=args.min_supply_temp,
        max_supply=args.max_supply_temp,
    )
    base_supply = np.clip(curve_intercept + curve_slope * inputs.outdoor, args.min_supply_temp, args.max_supply_temp)
    t_ref = _reference_room(model_report_dir, inputs.index, initial_air)
    p_per_q = _estimate_power_per_q(history_df)
    loss_coeff, solar_const = _thermal_arrays(inputs, theta)
    hum_const = theta[11] * ((humidity - humidity_center) / 100.0) + theta[12]

    n = len(horizon_df)
    t_air = cp.Variable(n + 1)
    t_mass = cp.Variable(n + 1)
    q_emit = cp.Variable(n + 1)
    q_hp = cp.Variable(n, nonneg=True)
    p_hp = cp.Variable(n, nonneg=True)
    offset = cp.Variable(n)
    slack_low = cp.Variable(n + 1, nonneg=True)
    slack_high = cp.Variable(n + 1, nonneg=True)

    supply = base_supply + offset
    emit_alpha = float(np.clip(dt_h / max(tau_emit_h, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau_h, 1e-6), 0.0, 1.0))
    constraints = [
        t_air[0] == initial_air,
        t_mass[0] == initial_mass,
        q_emit[0] == initial_q_emit,
        offset >= args.min_offset,
        offset <= args.max_offset,
        supply >= args.min_supply_temp,
        supply <= args.max_supply_temp,
        q_hp <= args.max_q_hp,
        q_hp <= supply - t_ref,
        p_hp <= args.max_p_hp,
        p_hp >= p_per_q * q_hp,
        t_air >= args.min_room_temp - slack_low,
        t_air <= args.max_room_temp + slack_high,
    ]

    for t in range(n):
        d_air_dt = (
            theta[0] * q_emit[t]
            + loss_coeff[t] * (inputs.outdoor[t] - t_air[t])
            + theta[5] * (t_mass[t] - t_air[t])
            + solar_const[t]
            + hum_const[t]
        )
        constraints.extend(
            [
                t_air[t + 1] == t_air[t] + dt_h * d_air_dt,
                t_mass[t + 1] == t_mass[t] + mass_alpha * (t_air[t] - t_mass[t]),
                q_emit[t + 1] == q_emit[t] + emit_alpha * (q_hp[t] - q_emit[t]),
            ]
        )

    electric_cost = cp.sum(cp.multiply(prices, p_hp / 1000.0 * dt_h))
    comfort = args.comfort_weight * cp.sum_squares(slack_low[1:] + slack_high[1:])
    offset_cost = args.offset_weight * cp.sum_squares(offset)
    smooth_cost = args.smooth_weight * cp.sum_squares(offset[1:] - offset[:-1])
    q_smooth_cost = args.q_smooth_weight * cp.sum_squares(q_hp[1:] - q_hp[:-1])
    problem = cp.Problem(cp.Minimize(electric_cost + comfort + offset_cost + smooth_cost + q_smooth_cost), constraints)

    solve_started = time.perf_counter()
    solver = "OSQP"
    try:
        problem.solve(solver=solver, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000)
    except Exception:
        pass
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        solver = "CLARABEL"
        problem.solve(solver=solver, verbose=False)
    solve_runtime_s = float(time.perf_counter() - solve_started)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY optimizer failed: status={problem.status}")

    t_air_v = np.asarray(t_air.value, dtype=float)
    q_emit_v = np.asarray(q_emit.value, dtype=float)
    q_hp_v = np.asarray(q_hp.value, dtype=float)
    p_hp_v = np.asarray(p_hp.value, dtype=float)
    offset_v = np.asarray(offset.value, dtype=float)
    supply_v = base_supply + offset_v
    predicted_room = t_air_v[1:]
    actual_summary = _objective_summary(
        inputs.room,
        inputs.electric,
        prices,
        dt_h=dt_h,
        min_room=args.min_room_temp,
        max_room=args.max_room_temp,
    )
    optimized_summary = _objective_summary(
        predicted_room,
        p_hp_v,
        prices,
        dt_h=dt_h,
        min_room=args.min_room_temp,
        max_room=args.max_room_temp,
    )

    plan_df = pd.DataFrame(
        {
            "timestamp": inputs.index,
            "electricity_price": prices,
            "outdoor_temp": inputs.outdoor,
            "q_solar": inputs.q_solar,
            "actual_room_temp": inputs.room,
            "actual_electric_power": inputs.electric,
            "actual_heatpump_duty": inputs.heatpump_duty,
            "actual_supply_temp": inputs.supply,
            "weather_curve_supply_temp": base_supply,
            "reference_room_temp": t_ref,
            "optimized_room_temp": predicted_room,
            "optimized_t_mass": np.asarray(t_mass.value, dtype=float)[1:],
            "optimized_q_emit": q_emit_v[1:],
            "optimized_q_hp": q_hp_v,
            "optimized_p_hp": p_hp_v,
            "optimized_curve_offset": offset_v,
            "optimized_supply_temp": supply_v,
        }
    )
    plan_path = report_dir / "cvxpy_state_space_optimization_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    summary = {
        "model": "CVXPYStateSpaceOptimizer",
        "data_path": str(data_path),
        "model_report_dir": str(model_report_dir),
        "report_dir": str(report_dir),
        "plan_path": str(plan_path),
        "plot_path": str(report_dir / "cvxpy_state_space_optimization_plot.html"),
        "start": str(inputs.index[0]),
        "end": str(inputs.index[-1]),
        "n_steps": int(n),
        "dt_h": float(dt_h),
        "solver": solver,
        "status": problem.status,
        "objective": float(problem.value),
        "solve_runtime_s": solve_runtime_s,
        "runtime_s": float(time.perf_counter() - started),
        "power_relation": {"p_per_q_w_per_c": float(p_per_q)},
        "weather_curve": {"intercept": float(curve_intercept), "slope_per_outdoor_c": float(curve_slope)},
        "initial_state": {
            "T_air": float(initial_air),
            "T_mass": float(initial_mass),
            "Q_emit": float(initial_q_emit),
        },
        "optimized_model": optimized_summary,
        "measured_actual": actual_summary,
        "config": vars(args),
    }
    (report_dir / "cvxpy_state_space_optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"case": "measured_actual", **actual_summary},
            {"case": "optimized_model", **optimized_summary},
        ]
    ).to_csv(report_dir / "cvxpy_state_space_optimization_summary.csv", index=False)
    _write_plot(plan_df, summary, report_dir / "cvxpy_state_space_optimization_plot.html")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
