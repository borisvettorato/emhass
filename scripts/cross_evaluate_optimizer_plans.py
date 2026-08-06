"""Cross-evaluate optimizer plans in the other thermal model.

This answers: if optimizer A's plan is applied to model B, does the plan still
look good? It is intentionally a dry-run analysis, not a Home Assistant action.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cvxpy_state_space_thermal_model as cvxm  # noqa: E402
import thermal_mass_optimizer as tmo  # noqa: E402
import thermal_mass_physics_model as tm  # noqa: E402
from compare_ensemble import _load_raw_df  # noqa: E402


def _with_controls(inputs: tm.ThermalInputs, duty: np.ndarray, supply: np.ndarray) -> tm.ThermalInputs:
    duty = np.asarray(duty, dtype=float).clip(0.0, 1.0)
    supply = np.asarray(supply, dtype=float)
    return replace(inputs, duty=duty, heatpump_duty=duty, supply=supply)


def _price_forecast(data_path: Path, index: pd.DatetimeIndex, default_price: float) -> np.ndarray:
    col = "sensor.current_electricity_market_price"
    try:
        df = pd.read_csv(data_path, usecols=["timestamp", col])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        idx = pd.DatetimeIndex(index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return pd.to_numeric(df[col], errors="coerce").reindex(idx).ffill().bfill().fillna(default_price).to_numpy(dtype=float)
    except Exception:
        return np.full(len(index), default_price, dtype=float)


def _summary(
    room: np.ndarray,
    electric_w: np.ndarray,
    gas_m3: np.ndarray,
    prices: np.ndarray,
    *,
    dt_h: float,
    min_room_temp: float,
    max_room_temp: float,
    gas_price: float,
) -> dict[str, float]:
    room = np.asarray(room, dtype=float)
    electric_w = np.asarray(electric_w, dtype=float)
    gas_m3 = np.asarray(gas_m3, dtype=float)
    below = np.clip(min_room_temp - room, 0.0, None)
    above = np.clip(room - max_room_temp, 0.0, None)
    electric_kwh = electric_w / 1000.0 * dt_h
    electric_cost = float(np.sum(electric_kwh * prices))
    gas_cost = float(np.sum(gas_m3) * gas_price)
    return {
        "electric_kwh": float(np.sum(electric_kwh)),
        "electric_cost_eur": electric_cost,
        "gas_m3": float(np.sum(gas_m3)),
        "gas_cost_eur": gas_cost,
        "total_cost_eur": electric_cost + gas_cost,
        "comfort_degree_h": float(np.sum(below + above) * dt_h),
        "comfort_sq_degree_h": float(np.sum(below * below + above * above) * dt_h),
        "min_room_temp": float(np.min(room)),
        "max_room_temp": float(np.max(room)),
        "mean_room_temp": float(np.mean(room)),
        "mean_power_w": float(np.mean(electric_w)),
    }


def _thermal_prepare_inputs(df: pd.DataFrame, config: dict) -> tm.ThermalInputs:
    return tm._prepare_inputs(
        df,
        latitude=float(config["latitude"]),
        longitude=float(config["longitude"]),
        facade_azimuth_deg=float(config["facade_azimuth_deg"]),
        facade_tilt_deg=float(config["facade_tilt_deg"]),
        solar_horizontal_weight=float(config["solar_horizontal_weight"]),
        solar_facade_weight=float(config["solar_facade_weight"]),
    )


def _cvx_prepare_config(config: dict) -> dict:
    return {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "facade_azimuth_deg": config["facade_azimuth_deg"],
        "facade_tilt_deg": config["facade_tilt_deg"],
        "solar_horizontal_weight": config["solar_horizontal_weight"],
        "solar_facade_weight": config["solar_facade_weight"],
    }


def _fit_cvx_power_heads(
    data_path: Path,
    raw_df: pd.DataFrame,
    payload: dict,
    *,
    dt_h: float,
):
    config = payload["config"]
    args = SimpleNamespace(**config)
    _, _, train_val_ts, _ = cvxm._split_timestamps(data_path, args)
    df_train_val = cvxm._slice(raw_df, train_val_ts)
    prepare_config = _cvx_prepare_config(config)
    inputs = cvxm._prepare_inputs(df_train_val, prepare_config)
    tau_emit_h = float(payload["params"]["tau_emit_h"])
    mass_tau_h = float(payload["params"]["mass_tau_h"])
    states = cvxm._roll_observed_states(inputs, dt_h=dt_h, tau_emit_h=tau_emit_h, mass_tau_h=mass_tau_h)
    humidity = cvxm._humidity(df_train_val)
    X = cvxm._power_feature_frame(
        inputs,
        air_before=states["air_before"],
        mass=states["mass"],
        q_emit=states["q_emit"],
        q_hp=states["q_hp"],
        humidity=humidity,
    )
    elec_head = cvxm._fit_linear_head(X, inputs.electric, ridge=float(config["power_ridge"]))
    gas_head = cvxm._fit_linear_head(X, inputs.gas, ridge=float(config["gas_ridge"]))
    return elec_head, gas_head


def _simulate_thermal_with_qhp(
    inputs: tm.ThermalInputs,
    params: np.ndarray,
    q_hp: np.ndarray,
    *,
    dt_h: float,
    initial_air: float,
    initial_mass: float,
    initial_q_emit: float,
) -> tm.SimResult:
    (
        tau_emit,
        emit_gain,
        ua_base,
        ua_wind,
        ua_wind_sin,
        ua_wind_cos,
        mass_tau,
        mass_gain,
        solar_gain,
        solar_alt_sin_gain,
        solar_alt_cos_gain,
        solar_az_sin_gain,
        solar_az_cos_gain,
        bias,
    ) = params
    n = len(inputs.room)
    pred_room = np.zeros(n, dtype=float)
    air_before = np.zeros(n, dtype=float)
    mass_series = np.zeros(n, dtype=float)
    q_emit_series = np.zeros(n, dtype=float)
    loss_series = np.zeros(n, dtype=float)

    air = float(initial_air)
    mass = float(initial_mass)
    q_emit = float(initial_q_emit)
    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau, 1e-6), 0.0, 1.0))
    q_hp = np.asarray(q_hp, dtype=float).clip(0.0, None)

    for i in range(n):
        air_before[i] = air
        q_emit = q_emit + emit_alpha * (q_hp[i] - q_emit)
        mass = mass + mass_alpha * (air - mass)
        direction_loss = (
            ua_base
            + ua_wind * inputs.wind_speed[i]
            + ua_wind_sin * inputs.wind_speed[i] * inputs.wind_sin[i]
            + ua_wind_cos * inputs.wind_speed[i] * inputs.wind_cos[i]
        )
        loss_coeff = max(0.0, float(direction_loss))
        solar_direction_gain = (
            solar_gain
            + solar_alt_sin_gain * inputs.sun_alt_sin[i]
            + solar_alt_cos_gain * inputs.sun_alt_cos[i]
            + solar_az_sin_gain * inputs.sun_az_sin[i]
            + solar_az_cos_gain * inputs.sun_az_cos[i]
        )
        solar_heat = max(0.0, float(solar_direction_gain)) * inputs.q_solar[i]
        d_air_dt = (
            emit_gain * q_emit
            + solar_heat
            - loss_coeff * (air - inputs.outdoor[i])
            + mass_gain * (mass - air)
            + bias
        )
        air = float(np.clip(air + dt_h * d_air_dt, 5.0, 35.0))
        pred_room[i] = air
        mass_series[i] = mass
        q_emit_series[i] = q_emit
        loss_series[i] = loss_coeff

    return tm.SimResult(
        room=pred_room,
        air_before=air_before,
        mass=mass_series,
        q_emit=q_emit_series,
        q_solar=inputs.q_solar.copy(),
        loss_coeff=loss_series,
    )


def _write_plot(df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], utc=True)
    plot_df = plot_df.set_index("timestamp").sort_index()
    x = plot_df.index.tz_convert("Europe/Amsterdam")

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        specs=[[{}], [{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Room temperature: own model and cross-evaluation",
            "Room deltas against own-model result",
            "Electric power",
            "Cumulative cost and kWh",
            "Supply, q_hp/duty and price",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=plot_df["actual_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["thermal_plan_in_thermal_room"], mode="lines", name="TM plan in TM", line=dict(color="#0f766e", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["thermal_plan_in_cvx_room"], mode="lines", name="TM plan in CVX", line=dict(color="#14b8a6", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["cvx_plan_in_cvx_room"], mode="lines", name="CVX plan in CVX", line=dict(color="#2563eb", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["cvx_plan_in_thermal_room"], mode="lines", name="CVX plan in TM", line=dict(color="#7c3aed", width=1.7, dash="dot")), row=1, col=1)
    fig.add_hline(y=summary["config"]["min_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=summary["config"]["max_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)

    fig.add_hline(y=0, line_width=1, line_color="#94a3b8", row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["thermal_plan_in_cvx_room"] - plot_df["thermal_plan_in_thermal_room"], mode="lines", name="TM plan: CVX-TM room", line=dict(color="#0f766e", width=1.4), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=plot_df["cvx_plan_in_thermal_room"] - plot_df["cvx_plan_in_cvx_room"], mode="lines", name="CVX plan: TM-CVX room", line=dict(color="#7c3aed", width=1.4), showlegend=False), row=2, col=1)

    power_cols = [
        ("thermal_plan_in_thermal_power", "TM plan in TM", "#0f766e"),
        ("thermal_plan_in_cvx_power", "TM plan in CVX", "#14b8a6"),
        ("cvx_plan_in_cvx_power", "CVX plan in CVX", "#2563eb"),
        ("cvx_plan_in_thermal_power", "CVX plan in TM", "#7c3aed"),
    ]
    for col, name, color in power_cols:
        fig.add_trace(go.Scatter(x=x, y=plot_df[col], mode="lines", name=name, line=dict(color=color, width=1.2)), row=3, col=1)

    for col, name, color in [
        ("thermal_plan_in_thermal_cum_cost", "TM in TM cost", "#0f766e"),
        ("thermal_plan_in_cvx_cum_cost", "TM in CVX cost", "#14b8a6"),
        ("cvx_plan_in_cvx_cum_cost", "CVX in CVX cost", "#2563eb"),
        ("cvx_plan_in_thermal_cum_cost", "CVX in TM cost", "#7c3aed"),
    ]:
        fig.add_trace(go.Scatter(x=x, y=plot_df[col], mode="lines", name=name, line=dict(color=color, width=1.2)), row=4, col=1, secondary_y=False)
    for col, name, color in [
        ("thermal_plan_in_thermal_cum_kwh", "TM in TM kWh", "#0f766e"),
        ("cvx_plan_in_cvx_cum_kwh", "CVX in CVX kWh", "#2563eb"),
    ]:
        fig.add_trace(go.Scatter(x=x, y=plot_df[col], mode="lines", name=name, line=dict(color=color, width=1.0, dash="dot")), row=4, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=plot_df["thermal_supply_temp"], mode="lines", name="TM supply", line=dict(color="#ea580c", width=1.2)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=plot_df["cvx_supply_temp"], mode="lines", name="CVX supply", line=dict(color="#dc2626", width=1.2)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=plot_df["thermal_duty"], mode="lines", name="TM duty", line=dict(color="#64748b", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=plot_df["cvx_q_hp"], mode="lines", name="CVX q_hp", line=dict(color="#7c3aed", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=plot_df["electricity_price"], mode="lines", name="Price", line=dict(color="#111827", width=1.0, dash="dot")), row=5, col=1, secondary_y=True)

    fig.update_layout(
        title="Cross-evaluation of optimizer plans",
        template="plotly_white",
        height=1350,
        width=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="delta degC", row=2, col=1)
    fig.update_yaxes(title_text="W", row=3, col=1)
    fig.update_yaxes(title_text="EUR", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="kWh", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="supply degC", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="duty/q/price", row=5, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=5, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-evaluate ThermalMass and CVXPY optimizer plans")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--thermal-physics-report-dir", default="tests_thermal/reports/thermal_mass_physics_sundir_20260519_final")
    parser.add_argument("--cvx-model-report-dir", default="tests_thermal/reports/cvxpy_state_space_thermal_20260520")
    parser.add_argument("--thermal-plan", default="tests_thermal/reports/optimizer_coldday_20260514_pricefix/thermal_mass_mpc/thermal_mass_optimization_plan.csv")
    parser.add_argument("--cvx-plan", default="tests_thermal/reports/optimizer_coldday_20260514_pricefix/cvxpy_state_space/cvxpy_state_space_optimization_plan.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/optimizer_coldday_20260514_pricefix/cross_evaluation")
    parser.add_argument("--min-room-temp", type=float, default=20.0)
    parser.add_argument("--max-room-temp", type=float, default=22.0)
    parser.add_argument("--default-electricity-price", type=float, default=0.25)
    parser.add_argument("--gas-price", type=float, default=1.35)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    thermal_plan = pd.read_csv(args.thermal_plan)
    cvx_plan = pd.read_csv(args.cvx_plan)
    thermal_plan["timestamp"] = pd.to_datetime(thermal_plan["timestamp"], utc=True)
    cvx_plan["timestamp"] = pd.to_datetime(cvx_plan["timestamp"], utc=True)
    thermal_plan = thermal_plan.set_index("timestamp").sort_index()
    cvx_plan = cvx_plan.set_index("timestamp").sort_index()
    index = thermal_plan.index.intersection(cvx_plan.index)
    if index.empty:
        raise ValueError("No overlapping timestamps between plans")
    thermal_plan = thermal_plan.reindex(index)
    cvx_plan = cvx_plan.reindex(index)

    raw_df = _load_raw_df(data_path)
    horizon_df = raw_df.reindex(index)
    history_df = raw_df.loc[raw_df.index < index[0]].copy()
    dt_h = tm._infer_timestep_hours(index)
    prices = _price_forecast(data_path, index, args.default_electricity_price)

    physics_payload = json.loads((Path(args.thermal_physics_report_dir) / "thermal_mass_physics_metrics.json").read_text(encoding="utf-8"))
    physics_config = physics_payload["config"]
    physics_params = np.array([physics_payload["params"][name] for name in tm.PARAM_NAMES], dtype=float)
    fit_ts, _ = tmo._split_timestamps(data_path, physics_config, int(physics_config.get("seed", 42)))
    tm_elec_head, tm_gas_head = tmo._fit_power_heads(raw_df, fit_ts, physics_config, physics_params, dt_h)

    tm_inputs_base = _thermal_prepare_inputs(horizon_df, physics_config)
    tm_history_inputs = _thermal_prepare_inputs(history_df, physics_config)
    tm_initial_air, tm_initial_mass, tm_initial_q_emit = tm._estimate_initial_states(
        tm_history_inputs,
        physics_params,
        dt_h=dt_h,
        warmup_steps=int(physics_config["state_warmup_steps"]),
    )
    tm_initial_air = tm._latest_before(raw_df, index[0], "room_temp", tm_initial_air)

    cvx_payload = json.loads((Path(args.cvx_model_report_dir) / "cvxpy_state_space_metrics.json").read_text(encoding="utf-8"))
    cvx_config = cvx_payload["config"]
    cvx_prepare_config = _cvx_prepare_config(cvx_config)
    cvx_theta = np.array([cvx_payload["params"][name] for name in cvxm.TEMP_PARAM_NAMES], dtype=float)
    cvx_tau_emit_h = float(cvx_payload["params"]["tau_emit_h"])
    cvx_mass_tau_h = float(cvx_payload["params"]["mass_tau_h"])
    cvx_humidity_center = float(cvx_payload["params"]["humidity_center"])
    cvx_elec_head, cvx_gas_head = _fit_cvx_power_heads(data_path, raw_df, cvx_payload, dt_h=dt_h)
    cvx_inputs_base = cvxm._prepare_inputs(horizon_df, cvx_prepare_config)
    cvx_history_inputs = cvxm._prepare_inputs(history_df, cvx_prepare_config)
    cvx_initial_air, cvx_initial_mass, cvx_initial_q_emit = cvxm._estimate_initial_states(
        cvx_history_inputs,
        dt_h=dt_h,
        tau_emit_h=cvx_tau_emit_h,
        mass_tau_h=cvx_mass_tau_h,
        warmup_steps=int(cvx_config["state_warmup_steps"]),
    )
    cvx_initial_air = tm._latest_before(raw_df, index[0], "room_temp", cvx_initial_air)
    cvx_humidity = cvxm._humidity(horizon_df)

    # ThermalMass plan evaluated by CVXPY model.
    tm_plan_in_cvx_inputs = _with_controls(
        cvx_inputs_base,
        thermal_plan["optimized_heatpump_duty"].to_numpy(dtype=float),
        thermal_plan["optimized_supply_temp"].to_numpy(dtype=float),
    )
    tm_plan_in_cvx_sim = cvxm._simulate_open_loop(
        tm_plan_in_cvx_inputs,
        cvx_humidity,
        cvx_theta,
        dt_h=dt_h,
        tau_emit_h=cvx_tau_emit_h,
        mass_tau_h=cvx_mass_tau_h,
        humidity_center=cvx_humidity_center,
        initial_air=cvx_initial_air,
        initial_mass=cvx_initial_mass,
        initial_q_emit=cvx_initial_q_emit,
    )
    tm_plan_in_cvx_X = cvxm._power_feature_frame(
        tm_plan_in_cvx_inputs,
        air_before=tm_plan_in_cvx_sim.air_before,
        mass=tm_plan_in_cvx_sim.mass,
        q_emit=tm_plan_in_cvx_sim.q_emit,
        q_hp=tm_plan_in_cvx_sim.q_hp,
        humidity=cvx_humidity,
    )
    tm_plan_in_cvx_power = np.clip(cvx_elec_head.predict(tm_plan_in_cvx_X), 0.0, None)
    tm_plan_in_cvx_gas = np.clip(cvx_gas_head.predict(tm_plan_in_cvx_X), 0.0, None)
    tm_plan_in_cvx_gas[tm_plan_in_cvx_inputs.outdoor > float(cvx_config["gas_cutoff_temp"])] = 0.0

    # CVXPY plan evaluated by ThermalMass model using its planned q_hp proxy.
    cvx_supply = cvx_plan["optimized_supply_temp"].to_numpy(dtype=float)
    cvx_q_hp = cvx_plan["optimized_q_hp"].to_numpy(dtype=float).clip(0.0, None)
    cvx_power = cvx_plan["optimized_p_hp"].to_numpy(dtype=float).clip(0.0, None)
    cvx_inputs_for_tm = _with_controls(tm_inputs_base, np.ones(len(index), dtype=float), cvx_supply)
    cvx_plan_in_tm_sim = _simulate_thermal_with_qhp(
        cvx_inputs_for_tm,
        physics_params,
        cvx_q_hp,
        dt_h=dt_h,
        initial_air=tm_initial_air,
        initial_mass=tm_initial_mass,
        initial_q_emit=tm_initial_q_emit,
    )
    # A proxy duty is only used to ask the ThermalMass power heads what they would expect.
    cvx_duty_proxy = np.clip(
        cvx_q_hp / np.maximum(cvx_supply - cvx_plan_in_tm_sim.air_before, 0.1),
        0.0,
        1.0,
    )
    cvx_inputs_for_tm_power = _with_controls(tm_inputs_base, cvx_duty_proxy, cvx_supply)
    cvx_tm_power_features = tm._power_features(cvx_inputs_for_tm_power, cvx_plan_in_tm_sim.air_before, cvx_plan_in_tm_sim.q_emit)
    cvx_plan_in_tm_power_head = np.clip(tm_elec_head.predict(cvx_tm_power_features), 0.0, None)
    cvx_plan_in_tm_gas = np.clip(tm_gas_head.predict(cvx_tm_power_features), 0.0, None)
    cvx_plan_in_tm_gas[tm_inputs_base.outdoor > float(physics_config["gas_cutoff_temp"])] = 0.0

    thermal_self_power = thermal_plan["optimized_electric_power"].to_numpy(dtype=float)
    thermal_self_gas = thermal_plan.get("optimized_gas_consumption", pd.Series(0.0, index=index)).to_numpy(dtype=float)
    cvx_self_room = cvx_plan["optimized_room_temp"].to_numpy(dtype=float)
    cvx_self_gas = np.zeros(len(index), dtype=float)

    rows = {
        "Thermal plan in ThermalMass": _summary(
            thermal_plan["optimized_room_temp"].to_numpy(dtype=float),
            thermal_self_power,
            thermal_self_gas,
            prices,
            dt_h=dt_h,
            min_room_temp=args.min_room_temp,
            max_room_temp=args.max_room_temp,
            gas_price=args.gas_price,
        ),
        "Thermal plan in CVXPYStateSpace": _summary(
            tm_plan_in_cvx_sim.room,
            tm_plan_in_cvx_power,
            tm_plan_in_cvx_gas,
            prices,
            dt_h=dt_h,
            min_room_temp=args.min_room_temp,
            max_room_temp=args.max_room_temp,
            gas_price=args.gas_price,
        ),
        "CVXPY plan in CVXPYStateSpace": _summary(
            cvx_self_room,
            cvx_power,
            cvx_self_gas,
            prices,
            dt_h=dt_h,
            min_room_temp=args.min_room_temp,
            max_room_temp=args.max_room_temp,
            gas_price=args.gas_price,
        ),
        "CVXPY plan in ThermalMass": _summary(
            cvx_plan_in_tm_sim.room,
            cvx_power,
            cvx_plan_in_tm_gas,
            prices,
            dt_h=dt_h,
            min_room_temp=args.min_room_temp,
            max_room_temp=args.max_room_temp,
            gas_price=args.gas_price,
        ),
        "CVXPY plan in ThermalMass power-head check": _summary(
            cvx_plan_in_tm_sim.room,
            cvx_plan_in_tm_power_head,
            cvx_plan_in_tm_gas,
            prices,
            dt_h=dt_h,
            min_room_temp=args.min_room_temp,
            max_room_temp=args.max_room_temp,
            gas_price=args.gas_price,
        ),
    }

    summary_rows = [{"case": name, **vals} for name, vals in rows.items()]
    pd.DataFrame(summary_rows).to_csv(report_dir / "cross_evaluation_summary.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "timestamp": index,
            "electricity_price": prices,
            "actual_room_temp": thermal_plan["actual_room_temp"].to_numpy(dtype=float),
            "thermal_plan_in_thermal_room": thermal_plan["optimized_room_temp"].to_numpy(dtype=float),
            "thermal_plan_in_cvx_room": tm_plan_in_cvx_sim.room,
            "cvx_plan_in_cvx_room": cvx_self_room,
            "cvx_plan_in_thermal_room": cvx_plan_in_tm_sim.room,
            "thermal_plan_in_thermal_power": thermal_self_power,
            "thermal_plan_in_cvx_power": tm_plan_in_cvx_power,
            "cvx_plan_in_cvx_power": cvx_power,
            "cvx_plan_in_thermal_power": cvx_power,
            "cvx_plan_in_thermal_power_head": cvx_plan_in_tm_power_head,
            "thermal_plan_in_thermal_gas": thermal_self_gas,
            "thermal_plan_in_cvx_gas": tm_plan_in_cvx_gas,
            "cvx_plan_in_thermal_gas": cvx_plan_in_tm_gas,
            "thermal_supply_temp": thermal_plan["optimized_supply_temp"].to_numpy(dtype=float),
            "cvx_supply_temp": cvx_supply,
            "thermal_duty": thermal_plan["optimized_heatpump_duty"].to_numpy(dtype=float),
            "cvx_q_hp": cvx_q_hp,
            "cvx_duty_proxy_for_thermal_power_head": cvx_duty_proxy,
        }
    )
    for prefix in [
        "thermal_plan_in_thermal",
        "thermal_plan_in_cvx",
        "cvx_plan_in_cvx",
        "cvx_plan_in_thermal",
    ]:
        power = pred_df[f"{prefix}_power"].to_numpy(dtype=float)
        pred_df[f"{prefix}_cum_kwh"] = np.cumsum(power / 1000.0 * dt_h)
        pred_df[f"{prefix}_cum_cost"] = np.cumsum(power / 1000.0 * dt_h * prices)
    pred_df.to_csv(report_dir / "cross_evaluation_predictions.csv", index=False)

    summary = {
        "start": str(index[0]),
        "end": str(index[-1]),
        "n_steps": int(len(index)),
        "dt_h": float(dt_h),
        "summary": rows,
        "notes": {
            "cvx_plan_in_thermal": "ThermalMass temperature simulation uses the CVXPY plan's optimized_q_hp heat proxy; electricity cost uses the CVXPY optimized_p_hp plan.",
            "cvx_plan_in_thermal_power_head_check": "Diagnostic only: asks the ThermalMass power heads what electric power they would infer after translating q_hp to a proxy duty.",
        },
        "config": vars(args),
    }
    (report_dir / "cross_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plot(pred_df, summary, report_dir / "cross_evaluation_plot.html")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
