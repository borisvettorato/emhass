"""Server-testable optimizer for the thermal-mass physics model.

This script uses the fitted thermal-mass model as a small MPC-style planner.
The optimization horizon is simulated open-loop: after the first state, room
temperature is always the model prediction, not the measured future room temp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _subset_inputs(inputs: tm.ThermalInputs, start: int, stop: int) -> tm.ThermalInputs:
    return replace(
        inputs,
        index=inputs.index[start:stop],
        room=inputs.room[start:stop],
        electric=inputs.electric[start:stop],
        gas=inputs.gas[start:stop],
        duty=inputs.duty[start:stop],
        supply=inputs.supply[start:stop],
        outdoor=inputs.outdoor[start:stop],
        wind_speed=inputs.wind_speed[start:stop],
        wind_sin=inputs.wind_sin[start:stop],
        wind_cos=inputs.wind_cos[start:stop],
        q_solar=inputs.q_solar[start:stop],
        sun_alt_sin=inputs.sun_alt_sin[start:stop],
        sun_alt_cos=inputs.sun_alt_cos[start:stop],
        sun_az_sin=inputs.sun_az_sin[start:stop],
        sun_az_cos=inputs.sun_az_cos[start:stop],
        heatpump_duty=inputs.heatpump_duty[start:stop],
    )


def _with_controls(inputs: tm.ThermalInputs, duty: np.ndarray, supply: np.ndarray) -> tm.ThermalInputs:
    duty = np.asarray(duty, dtype=float).clip(0.0, 1.0)
    supply = np.asarray(supply, dtype=float)
    return replace(inputs, duty=duty, heatpump_duty=duty, supply=supply)


def _split_timestamps(data_path: Path, config: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    target_cols = [c.strip() for c in str(config["target_cols"]).split(",") if c.strip()]
    opts = SearchOptions(
        lookahead=int(config["lookahead"]),
        feature_level=str(config["feature_level"]),
        target_cols=target_cols,
        latitude=float(config["latitude"]),
        longitude=float(config["longitude"]),
        seed=int(config.get("seed", seed)),
    )
    X_raw, y_raw, _, _, _ = _prepare_features(data_path, opts=opts)
    aligned_index = _build_aligned_index(data_path, opts)
    X_seq, y_seq = create_sequences(
        X_raw,
        y_raw,
        lookback=int(config["input_window"]),
        lookahead=int(config["lookahead"]),
    )
    X_train, _, X_val, _, X_test, _ = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_train), len(X_val)
    n_seq = len(X_seq)
    seq_start_ts = aligned_index[int(config["input_window"]) : int(config["input_window"]) + n_seq].to_numpy()
    train_ts = seq_start_ts[:n_train]
    val_ts = seq_start_ts[n_train : n_train + n_val]
    train_val_ts = seq_start_ts[: n_train + n_val]
    test_ts = seq_start_ts[n_train + n_val :]

    if config.get("holdout_split") == "val":
        fit_ts = train_ts
        eval_ts = val_ts
    else:
        fit_ts = train_val_ts if config.get("fit_on") == "train_val" else train_ts
        eval_ts = test_ts
    return fit_ts, eval_ts


def _fit_power_heads(
    raw_df: pd.DataFrame,
    fit_ts: np.ndarray,
    config: dict,
    params: np.ndarray,
    dt_h: float,
):
    df_fit = _slice(raw_df, fit_ts)
    fit_inputs = _prepare_inputs(df_fit, config)
    train_room_state = tm._observed_previous_room(df_fit)
    train_q_state = tm._q_emit_from_room_state(fit_inputs, train_room_state, params, dt_h=dt_h)
    X_power_train = tm._power_features(fit_inputs, train_room_state, train_q_state)
    elec_model = tm._fit_power_head(
        X_power_train,
        fit_inputs.electric,
        alpha=float(config["power_ridge_alpha"]),
    )
    gas_model = tm._fit_power_head(
        X_power_train,
        fit_inputs.gas,
        alpha=float(config["gas_ridge_alpha"]),
    )
    return elec_model, gas_model


def _predict_power(
    inputs: tm.ThermalInputs,
    sim: tm.SimResult,
    elec_model,
    gas_model,
    gas_cutoff_temp: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_power = tm._power_features(inputs, sim.air_before, sim.q_emit)
    electric = np.clip(elec_model.predict(X_power), 0.0, None)
    gas = np.clip(gas_model.predict(X_power), 0.0, None)
    gas[inputs.outdoor > gas_cutoff_temp] = 0.0
    return electric, gas


def _simulate_plan(
    base_inputs: tm.ThermalInputs,
    duty: np.ndarray,
    supply: np.ndarray,
    params: np.ndarray,
    *,
    dt_h: float,
    initial_air: float,
    initial_mass: float,
    initial_q_emit: float,
    elec_model,
    gas_model,
    gas_cutoff_temp: float,
) -> tuple[tm.SimResult, np.ndarray, np.ndarray]:
    plan_inputs = _with_controls(base_inputs, duty, supply)
    sim = tm._simulate_open_loop(
        plan_inputs,
        params,
        dt_h=dt_h,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
    )
    electric, gas = _predict_power(plan_inputs, sim, elec_model, gas_model, gas_cutoff_temp)
    return sim, electric, gas


def _price_forecast(
    df: pd.DataFrame,
    column: str,
    default_price: float,
    *,
    data_path: Path | None = None,
) -> np.ndarray:
    if column in df.columns:
        price = pd.to_numeric(df[column], errors="coerce").ffill().bfill().fillna(default_price)
        return price.to_numpy(dtype=float)
    if data_path is not None and Path(data_path).exists():
        try:
            price_df = pd.read_csv(data_path, usecols=["timestamp", column])
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
            price_df = price_df.set_index("timestamp").sort_index()
            idx = pd.DatetimeIndex(df.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            price = pd.to_numeric(price_df[column], errors="coerce").reindex(idx).ffill().bfill()
            price = price.fillna(default_price)
            return price.to_numpy(dtype=float)
        except Exception:
            pass
    return np.full(len(df), default_price, dtype=float)


def _weather_curve(
    history_df: pd.DataFrame,
    outdoor: np.ndarray,
    *,
    min_supply: float,
    max_supply: float,
) -> np.ndarray:
    supply = pd.to_numeric(history_df.get("supply_temp", pd.Series(dtype=float)), errors="coerce")
    duty = pd.to_numeric(history_df.get("heatpump_duty", pd.Series(dtype=float)), errors="coerce")
    active_supply = supply[duty > 0.2].dropna()
    base_supply = float(active_supply.tail(14 * 96).median()) if not active_supply.empty else 30.0
    if not np.isfinite(base_supply):
        base_supply = 30.0
    curve = base_supply + 0.45 * (10.0 - np.asarray(outdoor, dtype=float))
    return np.clip(curve, min_supply, max_supply)


def _objective(
    room: np.ndarray,
    electric_w: np.ndarray,
    gas_m3: np.ndarray,
    duty: np.ndarray,
    supply: np.ndarray,
    prices: np.ndarray,
    *,
    dt_h: float,
    min_room_temp: float,
    max_room_temp: float,
    gas_price: float,
    comfort_weight: float,
    switch_penalty: float,
    supply_step_penalty: float,
) -> dict[str, float]:
    electric_kwh = np.asarray(electric_w, dtype=float) / 1000.0 * dt_h
    gas = np.asarray(gas_m3, dtype=float)
    price = np.asarray(prices, dtype=float)
    below = np.clip(min_room_temp - np.asarray(room, dtype=float), 0.0, None)
    above = np.clip(np.asarray(room, dtype=float) - max_room_temp, 0.0, None)
    comfort_degree_h = float(np.sum(below + above) * dt_h)
    comfort_sq_degree_h = float(np.sum((below * below) + (above * above)) * dt_h)
    electric_cost = float(np.sum(electric_kwh * price))
    gas_cost = float(np.sum(gas) * gas_price)
    switches = float(np.sum(np.abs(np.diff(np.asarray(duty, dtype=float))) > 0.25))
    supply_steps = float(np.sum(np.abs(np.diff(np.asarray(supply, dtype=float)))))
    objective = (
        electric_cost
        + gas_cost
        + comfort_weight * comfort_sq_degree_h
        + switch_penalty * switches
        + supply_step_penalty * supply_steps
    )
    return {
        "objective": float(objective),
        "electric_cost_eur": electric_cost,
        "gas_cost_eur": gas_cost,
        "total_cost_eur": electric_cost + gas_cost,
        "electric_kwh": float(np.sum(electric_kwh)),
        "gas_m3": float(np.sum(gas)),
        "comfort_degree_h": comfort_degree_h,
        "comfort_sq_degree_h": comfort_sq_degree_h,
        "min_room_temp": float(np.min(room)) if len(room) else float("nan"),
        "max_room_temp": float(np.max(room)) if len(room) else float("nan"),
        "mean_room_temp": float(np.mean(room)) if len(room) else float("nan"),
        "mean_duty": float(np.mean(duty)) if len(duty) else float("nan"),
        "switches": switches,
        "supply_step_sum": supply_steps,
    }


def _action_candidates(
    base_supply: float,
    *,
    min_supply: float,
    max_supply: float,
) -> list[tuple[float, float]]:
    supplies = np.clip(
        np.array([base_supply - 5.0, base_supply - 3.0, base_supply - 1.0, base_supply + 1.0, base_supply + 3.0, base_supply + 5.0]),
        min_supply,
        max_supply,
    )
    actions = [(0.0, min_supply)]
    for supply in sorted(set(float(round(s, 2)) for s in supplies)):
        actions.append((1.0, supply))
    return actions


def _optimize_mpc(
    inputs: tm.ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    initial_air: float,
    initial_mass: float,
    initial_q_emit: float,
    elec_model,
    gas_model,
    gas_cutoff_temp: float,
    prices: np.ndarray,
    base_curve: np.ndarray,
    lookahead_steps: int,
    min_room_temp: float,
    max_room_temp: float,
    min_supply: float,
    max_supply: float,
    gas_price: float,
    comfort_weight: float,
    switch_penalty: float,
    supply_step_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = len(inputs.room)
    duty_plan = np.zeros(n, dtype=float)
    supply_plan = np.full(n, min_supply, dtype=float)
    air = float(initial_air)
    mass = float(initial_mass)
    q_emit = float(initial_q_emit)
    prev_duty = 0.0
    prev_supply = min_supply
    decisions_evaluated = 0

    for t in range(n):
        stop = min(n, t + max(1, lookahead_steps))
        sub = _subset_inputs(inputs, t, stop)
        sub_prices = prices[t:stop]
        best_score = None
        best_action = (0.0, min_supply)
        best_first_sim = None

        for duty, supply in _action_candidates(
            float(base_curve[t]),
            min_supply=min_supply,
            max_supply=max_supply,
        ):
            cand_duty = np.full(len(sub.room), duty, dtype=float)
            cand_supply = np.full(len(sub.room), supply, dtype=float)
            sim, electric, gas = _simulate_plan(
                sub,
                cand_duty,
                cand_supply,
                params,
                dt_h=dt_h,
                initial_air=air,
                initial_mass=mass,
                initial_q_emit=q_emit,
                elec_model=elec_model,
                gas_model=gas_model,
                gas_cutoff_temp=gas_cutoff_temp,
            )
            obj = _objective(
                sim.room,
                electric,
                gas,
                cand_duty,
                cand_supply,
                sub_prices,
                dt_h=dt_h,
                min_room_temp=min_room_temp,
                max_room_temp=max_room_temp,
                gas_price=gas_price,
                comfort_weight=comfort_weight,
                switch_penalty=0.0,
                supply_step_penalty=0.0,
            )
            immediate_smooth = (
                switch_penalty * float(abs(duty - prev_duty) > 0.25)
                + supply_step_penalty * abs(supply - prev_supply)
            )
            score = obj["objective"] + immediate_smooth
            decisions_evaluated += 1
            if best_score is None or score < best_score:
                best_score = score
                best_action = (duty, supply)
                best_first_sim = sim

        duty_plan[t], supply_plan[t] = best_action
        assert best_first_sim is not None
        air = float(best_first_sim.room[0])
        mass = float(best_first_sim.mass[0])
        q_emit = float(best_first_sim.q_emit[0])
        prev_duty, prev_supply = best_action

    return duty_plan, supply_plan, {"decisions_evaluated": float(decisions_evaluated)}


def _measured_summary(
    inputs: tm.ThermalInputs,
    prices: np.ndarray,
    *,
    dt_h: float,
    min_room_temp: float,
    max_room_temp: float,
    gas_price: float,
) -> dict[str, float]:
    return _objective(
        inputs.room,
        inputs.electric,
        inputs.gas,
        inputs.duty,
        inputs.supply,
        prices,
        dt_h=dt_h,
        min_room_temp=min_room_temp,
        max_room_temp=max_room_temp,
        gas_price=gas_price,
        comfort_weight=0.0,
        switch_penalty=0.0,
        supply_step_penalty=0.0,
    )


def _write_plot(plan_df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    df = plan_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    x = df.index.tz_convert("Europe/Amsterdam")

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "Room temperature",
            "Heat pump controls",
            "Electric power and gas",
            "Price, solar and outdoor",
            "Thermal states",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=df["actual_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["baseline_room_temp"], mode="lines", name="Model with actual controls", line=dict(color="#0f766e", width=1.6, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_room_temp"], mode="lines", name="Optimized plan", line=dict(color="#2563eb", width=2.0)), row=1, col=1)
    fig.add_hline(y=summary["config"]["min_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=summary["config"]["max_room_temp"], line_width=1, line_color="#94a3b8", row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["actual_heatpump_duty"], mode="lines", name="Actual duty", line=dict(color="#64748b", width=1.0, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_heatpump_duty"], mode="lines", name="Optimized duty", line=dict(color="#2563eb", width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["actual_supply_temp"], mode="lines", name="Actual supply", line=dict(color="#f97316", width=1.0, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_supply_temp"], mode="lines", name="Optimized supply", line=dict(color="#dc2626", width=1.4)), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["baseline_electric_power"], mode="lines", name="Baseline electric", line=dict(color="#0f766e", width=1.0, dash="dot")), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_electric_power"], mode="lines", name="Optimized electric", line=dict(color="#2563eb", width=1.2)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_gas_consumption"], mode="lines", name="Optimized gas", line=dict(color="#16a34a", width=1.0)), row=3, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=df["electricity_price"], mode="lines", name="Price", line=dict(color="#7c3aed", width=1.2)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["q_solar"], mode="lines", name="Q solar", line=dict(color="#f59e0b", width=1.0)), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["outdoor_temp"], mode="lines", name="Outdoor", line=dict(color="#0284c7", width=1.0)), row=4, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=df["optimized_t_mass"], mode="lines", name="T_mass", line=dict(color="#7c3aed", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["optimized_q_emit"], mode="lines", name="Q_emit", line=dict(color="#ea580c", width=1.2)), row=5, col=1)

    baseline = summary["baseline_model"]
    optimized = summary["optimized_model"]
    note = (
        f"Baseline cost EUR {baseline['total_cost_eur']:.2f}, comfort {baseline['comfort_degree_h']:.2f} degCh"
        f"<br>Optimized cost EUR {optimized['total_cost_eur']:.2f}, comfort {optimized['comfort_degree_h']:.2f} degCh"
    )
    fig.add_annotation(text=note, xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(template="plotly_white", height=1250, width=1550, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="duty / degC", row=2, col=1)
    fig.update_yaxes(title_text="W", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="m3/interval", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="EUR/kWh", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="weather", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=5, col=1)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _resolve_start(
    raw_df: pd.DataFrame,
    physics_report_dir: Path,
    eval_ts: np.ndarray,
    start: str | None,
) -> pd.Timestamp:
    if start:
        ts = pd.Timestamp(start)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(raw_df.index.tz or "UTC")

    pred_path = physics_report_dir / "thermal_mass_physics_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, usecols=["timestamp"], nrows=1)
        if len(pred):
            return pd.Timestamp(pred["timestamp"].iloc[0]).tz_convert(raw_df.index.tz or "UTC")

    if len(eval_ts):
        return pd.Timestamp(eval_ts[0]).tz_convert(raw_df.index.tz or "UTC")
    return raw_df.index[-96]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize heat-pump controls with the thermal-mass physics model")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--physics-report-dir", default="tests_thermal/reports/thermal_mass_physics_sundir_20260519_final")
    parser.add_argument("--report-dir", default="tests_thermal/reports/thermal_mass_optimization")
    parser.add_argument("--start", default=None)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--mpc-lookahead", type=int, default=16)
    parser.add_argument("--min-room-temp", type=float, default=20.0)
    parser.add_argument("--max-room-temp", type=float, default=22.0)
    parser.add_argument("--min-supply-temp", type=float, default=25.0)
    parser.add_argument("--max-supply-temp", type=float, default=45.0)
    parser.add_argument("--default-electricity-price", type=float, default=0.25)
    parser.add_argument("--gas-price", type=float, default=1.35)
    parser.add_argument("--comfort-weight", type=float, default=35.0)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--supply-step-penalty", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    data_path = Path(args.data_path)
    physics_report_dir = Path(args.physics_report_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    physics_payload = json.loads((physics_report_dir / "thermal_mass_physics_metrics.json").read_text(encoding="utf-8"))
    config = physics_payload["config"]
    params = np.array([physics_payload["params"][name] for name in tm.PARAM_NAMES], dtype=float)
    raw_df = _load_raw_df(data_path)
    dt_h = tm._infer_timestep_hours(raw_df.index)
    fit_ts, eval_ts = _split_timestamps(data_path, config, args.seed)
    start_ts = _resolve_start(raw_df, physics_report_dir, eval_ts, args.start)

    horizon_df = raw_df.loc[raw_df.index >= start_ts].head(args.horizon).copy()
    if horizon_df.empty:
        raise ValueError(f"No horizon data found at or after {start_ts}")
    history_df = raw_df.loc[raw_df.index < horizon_df.index[0]].copy()
    if history_df.empty:
        raise ValueError("No pre-horizon history available to initialize thermal states")

    elec_model, gas_model = _fit_power_heads(raw_df, fit_ts, config, params, dt_h)
    horizon_inputs = _prepare_inputs(horizon_df, config)
    history_inputs = _prepare_inputs(history_df, config)
    initial_air, initial_mass, initial_q_emit = tm._estimate_initial_states(
        history_inputs,
        params,
        dt_h=dt_h,
        warmup_steps=int(config["state_warmup_steps"]),
    )
    initial_air = tm._latest_before(raw_df, horizon_df.index[0], "room_temp", initial_air)

    prices = _price_forecast(
        horizon_df,
        "sensor.current_electricity_market_price",
        args.default_electricity_price,
        data_path=data_path,
    )
    base_curve = _weather_curve(
        history_df,
        horizon_inputs.outdoor,
        min_supply=args.min_supply_temp,
        max_supply=args.max_supply_temp,
    )
    gas_cutoff_temp = float(config["gas_cutoff_temp"])

    baseline_sim, baseline_electric, baseline_gas = _simulate_plan(
        horizon_inputs,
        horizon_inputs.duty,
        horizon_inputs.supply,
        params,
        dt_h=dt_h,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
        elec_model=elec_model,
        gas_model=gas_model,
        gas_cutoff_temp=gas_cutoff_temp,
    )

    opt_duty, opt_supply, optimizer_info = _optimize_mpc(
        horizon_inputs,
        params,
        dt_h=dt_h,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
        elec_model=elec_model,
        gas_model=gas_model,
        gas_cutoff_temp=gas_cutoff_temp,
        prices=prices,
        base_curve=base_curve,
        lookahead_steps=args.mpc_lookahead,
        min_room_temp=args.min_room_temp,
        max_room_temp=args.max_room_temp,
        min_supply=args.min_supply_temp,
        max_supply=args.max_supply_temp,
        gas_price=args.gas_price,
        comfort_weight=args.comfort_weight,
        switch_penalty=args.switch_penalty,
        supply_step_penalty=args.supply_step_penalty,
    )
    opt_sim, opt_electric, opt_gas = _simulate_plan(
        horizon_inputs,
        opt_duty,
        opt_supply,
        params,
        dt_h=dt_h,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
        elec_model=elec_model,
        gas_model=gas_model,
        gas_cutoff_temp=gas_cutoff_temp,
    )

    common_obj_kwargs = dict(
        dt_h=dt_h,
        min_room_temp=args.min_room_temp,
        max_room_temp=args.max_room_temp,
        gas_price=args.gas_price,
        comfort_weight=args.comfort_weight,
        switch_penalty=args.switch_penalty,
        supply_step_penalty=args.supply_step_penalty,
    )
    baseline_summary = _objective(
        baseline_sim.room,
        baseline_electric,
        baseline_gas,
        horizon_inputs.duty,
        horizon_inputs.supply,
        prices,
        **common_obj_kwargs,
    )
    optimized_summary = _objective(
        opt_sim.room,
        opt_electric,
        opt_gas,
        opt_duty,
        opt_supply,
        prices,
        **common_obj_kwargs,
    )
    measured_summary = _measured_summary(
        horizon_inputs,
        prices,
        dt_h=dt_h,
        min_room_temp=args.min_room_temp,
        max_room_temp=args.max_room_temp,
        gas_price=args.gas_price,
    )

    plan_df = pd.DataFrame(
        {
            "timestamp": horizon_inputs.index,
            "electricity_price": prices,
            "outdoor_temp": horizon_inputs.outdoor,
            "q_solar": horizon_inputs.q_solar,
            "actual_room_temp": horizon_inputs.room,
            "actual_electric_power": horizon_inputs.electric,
            "actual_gas_consumption": horizon_inputs.gas,
            "actual_heatpump_duty": horizon_inputs.heatpump_duty,
            "actual_supply_temp": horizon_inputs.supply,
            "weather_curve_supply_temp": base_curve,
            "baseline_room_temp": baseline_sim.room,
            "baseline_electric_power": baseline_electric,
            "baseline_gas_consumption": baseline_gas,
            "baseline_t_mass": baseline_sim.mass,
            "baseline_q_emit": baseline_sim.q_emit,
            "optimized_room_temp": opt_sim.room,
            "optimized_electric_power": opt_electric,
            "optimized_gas_consumption": opt_gas,
            "optimized_heatpump_duty": opt_duty,
            "optimized_supply_temp": opt_supply,
            "optimized_t_mass": opt_sim.mass,
            "optimized_q_emit": opt_sim.q_emit,
        }
    )
    plan_path = report_dir / "thermal_mass_optimization_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    summary = {
        "model": "ThermalMassPhysicsMPC",
        "data_path": str(data_path),
        "physics_report_dir": str(physics_report_dir),
        "report_dir": str(report_dir),
        "plan_path": str(plan_path),
        "plot_path": str(report_dir / "thermal_mass_optimization_plot.html"),
        "start": str(horizon_inputs.index[0]),
        "end": str(horizon_inputs.index[-1]),
        "n_steps": int(len(plan_df)),
        "dt_h": float(dt_h),
        "initial_state": {
            "T_air": float(initial_air),
            "T_mass": float(initial_mass),
            "Q_emit": float(initial_q_emit),
        },
        "baseline_model": baseline_summary,
        "optimized_model": optimized_summary,
        "measured_actual": measured_summary,
        "delta_optimized_minus_baseline": {
            key: float(optimized_summary[key] - baseline_summary[key])
            for key in [
                "objective",
                "total_cost_eur",
                "electric_cost_eur",
                "gas_cost_eur",
                "electric_kwh",
                "gas_m3",
                "comfort_degree_h",
                "comfort_sq_degree_h",
            ]
        },
        "optimizer_info": optimizer_info,
        "config": {
            "horizon": int(args.horizon),
            "mpc_lookahead": int(args.mpc_lookahead),
            "min_room_temp": float(args.min_room_temp),
            "max_room_temp": float(args.max_room_temp),
            "min_supply_temp": float(args.min_supply_temp),
            "max_supply_temp": float(args.max_supply_temp),
            "gas_price": float(args.gas_price),
            "comfort_weight": float(args.comfort_weight),
            "switch_penalty": float(args.switch_penalty),
            "supply_step_penalty": float(args.supply_step_penalty),
        },
        "runtime_s": float(time.perf_counter() - started),
    }
    summary_path = report_dir / "thermal_mass_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"case": "measured_actual", **measured_summary},
            {"case": "baseline_model", **baseline_summary},
            {"case": "optimized_model", **optimized_summary},
        ]
    ).to_csv(report_dir / "thermal_mass_optimization_summary.csv", index=False)
    _write_plot(plan_df, summary, report_dir / "thermal_mass_optimization_plot.html")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
