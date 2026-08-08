"""Thermal-mass physics model for room temperature, electric power and gas.

This model is intentionally separate from compare_ensemble.py so it can be
tested against the existing open-loop physics model without changing the
current benchmark pipeline.

The simulation core AND the fitting routine (ThermalInputs, PARAM_NAMES,
_prepare_inputs, _simulate_open_loop, _fit_temperature_params, ...) live in
emhass.thermal.thermal_mass_physics so they can be shared with the live
EMHASS actions (command_line.compute_heating_forecast,
command_line.refit_heating_model) - this script owns only CSV loading,
train/val/test splitting, the report I/O, and plotting.

The test forecast is strict open-loop for room temperature: after the first
step, only the model's previous prediction is fed back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_ensemble import _build_aligned_index, _load_raw_df  # noqa: E402
from emhass.thermal.forecast_gridsearch import (  # noqa: E402
    SearchOptions,
    _prepare_features,
    create_sequences,
    split_sequences,
)
from emhass.persistence import save_json_blob  # noqa: E402
from emhass.thermal.thermal_mass_physics import (  # noqa: E402
    DEFAULT_X0,
    LOWER_BOUNDS,
    PARAM_NAMES,
    UPPER_BOUNDS,
    ThermalInputs,
    _fit_temperature_params,
    _infer_timestep_hours,
    _metrics,
    _prepare_inputs,
    _series,
    _simulate_open_loop,
    _simulate_segmented,
)


def _slice(raw_df: pd.DataFrame, ts: np.ndarray) -> pd.DataFrame:
    return raw_df.reindex(pd.DatetimeIndex(ts)).dropna(how="all")


def _latest_before(raw_df: pd.DataFrame, timestamp: pd.Timestamp, column: str, default: float) -> float:
    if column not in raw_df.columns:
        return default
    hist = pd.to_numeric(raw_df.loc[raw_df.index < timestamp, column], errors="coerce").dropna()
    if hist.empty:
        return default
    return float(hist.iloc[-1])


def _observed_previous_room(df: pd.DataFrame, default: float = 20.0) -> np.ndarray:
    room = _series(df, "room_temp", default)
    return room.shift(1).ffill().bfill().fillna(default).to_numpy(dtype=float)


def _q_emit_from_room_state(
    inputs: ThermalInputs,
    room_state: np.ndarray,
    params: np.ndarray,
    *,
    dt_h: float,
) -> np.ndarray:
    tau_emit = float(params[0])
    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    q_emit = 0.0
    out = np.zeros(len(room_state), dtype=float)
    for i in range(len(room_state)):
        emit_raw = inputs.duty[i] * max(inputs.supply[i] - room_state[i], 0.0)
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)
        out[i] = q_emit
    return out


def _estimate_initial_states(
    history_inputs: ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    warmup_steps: int,
) -> tuple[float, float, float]:
    if len(history_inputs.room) == 0:
        return 20.0, 20.0, 0.0

    start = max(0, len(history_inputs.room) - warmup_steps)
    air = float(history_inputs.room[start])
    mass = air
    q_emit = 0.0
    tau_emit = params[0]
    mass_tau = params[6]
    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau, 1e-6), 0.0, 1.0))

    for i in range(start, len(history_inputs.room)):
        observed_air = float(history_inputs.room[i])
        emit_raw = history_inputs.duty[i] * max(history_inputs.supply[i] - observed_air, 0.0)
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)
        mass = mass + mass_alpha * (observed_air - mass)
        air = observed_air
    return air, mass, q_emit


def _power_features(inputs: ThermalInputs, room_state: np.ndarray, q_emit: np.ndarray) -> np.ndarray:
    lift = np.clip(inputs.supply - inputs.outdoor, 0.0, None)
    emit_raw = inputs.duty * np.clip(inputs.supply - room_state, 0.0, None)
    delta_env = np.clip(room_state - inputs.outdoor, 0.0, None)
    wind_loss = inputs.wind_speed * delta_env
    cold = (inputs.outdoor < 5.0).astype(float)
    return np.column_stack([
        np.ones(len(room_state), dtype=float),
        inputs.duty,
        lift,
        inputs.duty * lift,
        inputs.duty * lift * lift,
        inputs.duty * inputs.supply,
        inputs.duty * inputs.outdoor,
        emit_raw,
        q_emit,
        delta_env,
        wind_loss,
        cold,
        cold * inputs.duty,
        inputs.q_solar,
    ])


def _fit_power_head(X: np.ndarray, y: np.ndarray, *, alpha: float) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(X, y)
    return model


def _write_plot(pred_df: pd.DataFrame, metrics: dict[str, dict[str, float]], params: dict[str, float], output_path: Path) -> None:
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
            "Thermal states",
            "Electric power",
            "Gas, duty, solar and wind-loss coefficient",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=df["true_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_room_temp"], mode="lines", name="Thermal mass physics", line=dict(color="#0f766e", width=2.0)), row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#6b7280", row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_room_temp"] - df["true_room_temp"], mode="lines", name="Room error", line=dict(color="#dc2626", width=1.4), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["t_mass"], mode="lines", name="T_mass", line=dict(color="#7c3aed", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["q_emit"], mode="lines", name="Q_emit", line=dict(color="#f97316", width=1.2)), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["true_electric_power"], mode="lines", name="Actual electric", line=dict(color="#111827", width=1.3)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_electric_power"], mode="lines", name="Pred electric", line=dict(color="#2563eb", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["true_gas_consumption"], mode="lines", name="Actual gas", line=dict(color="#111827", width=1.2)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["pred_gas_consumption"], mode="lines", name="Pred gas", line=dict(color="#16a34a", width=1.2)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["heatpump_duty"], mode="lines", name="Heatpump duty", line=dict(color="#64748b", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["q_solar"], mode="lines", name="Q_solar/1000", line=dict(color="#eab308", width=1.0)), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["loss_coeff"], mode="lines", name="UA wind coeff", line=dict(color="#0284c7", width=1.0)), row=5, col=1, secondary_y=True)

    summary = "<br>".join(
        f"{target}: MAE {vals['mae']:.3f}, RMSE {vals['rmse']:.3f}, bias {vals['bias']:+.3f}"
        for target, vals in metrics.items()
    )
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
        title=f"Thermal-mass physics model ({x.min().strftime('%Y-%m-%d %H:%M')} - {x.max().strftime('%Y-%m-%d %H:%M')} Europe/Amsterdam)",
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
    parser = argparse.ArgumentParser(description="Thermal-mass physics model with Q_emit, Q_solar and directional wind")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/thermal_mass_physics")
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
    parser.add_argument("--fit-segment-len", type=int, default=96)
    parser.add_argument("--holdout-split", choices=["val", "test"], default="test")
    parser.add_argument("--fit-on", choices=["train", "train_val"], default="train_val")
    parser.add_argument("--state-warmup-steps", type=int, default=672)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--power-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--gas-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gas-cutoff-temp", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deploy-path",
        default="data/thermal_physics_params.json",
        help=(
            "Where to additionally write the fitted params for the live EMHASS "
            "heating-need-forecast action to pick up. Set to an empty string "
            "to skip deployment (report-only run)."
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_path = Path(args.data_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

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

    if args.holdout_split == "val":
        fit_ts = train_ts
        eval_ts = val_ts
    else:
        fit_ts = train_val_ts if args.fit_on == "train_val" else train_ts
        eval_ts = test_ts

    raw_df = _load_raw_df(data_path)
    df_fit = _slice(raw_df, fit_ts)
    df_eval = _slice(raw_df, eval_ts)
    df_history = raw_df.loc[raw_df.index < df_eval.index[0]].copy()

    fit_inputs = _prepare_inputs(
        df_fit,
        latitude=args.latitude,
        longitude=args.longitude,
        facade_azimuth_deg=args.facade_azimuth_deg,
        facade_tilt_deg=args.facade_tilt_deg,
        solar_horizontal_weight=args.solar_horizontal_weight,
        solar_facade_weight=args.solar_facade_weight,
    )
    eval_inputs = _prepare_inputs(
        df_eval,
        latitude=args.latitude,
        longitude=args.longitude,
        facade_azimuth_deg=args.facade_azimuth_deg,
        facade_tilt_deg=args.facade_tilt_deg,
        solar_horizontal_weight=args.solar_horizontal_weight,
        solar_facade_weight=args.solar_facade_weight,
    )
    history_inputs = _prepare_inputs(
        df_history,
        latitude=args.latitude,
        longitude=args.longitude,
        facade_azimuth_deg=args.facade_azimuth_deg,
        facade_tilt_deg=args.facade_tilt_deg,
        solar_horizontal_weight=args.solar_horizontal_weight,
        solar_facade_weight=args.solar_facade_weight,
    )

    dt_h = _infer_timestep_hours(raw_df.index)
    t0 = time.perf_counter()
    params, fit_info = _fit_temperature_params(
        fit_inputs,
        dt_h=dt_h,
        segment_len=args.fit_segment_len,
        max_nfev=args.max_nfev,
    )
    train_runtime_s = float(time.perf_counter() - t0)

    initial_air, initial_mass, initial_q_emit = _estimate_initial_states(
        history_inputs,
        params,
        dt_h=dt_h,
        warmup_steps=args.state_warmup_steps,
    )
    if len(df_eval):
        initial_air = _latest_before(raw_df, df_eval.index[0], "room_temp", initial_air)

    test_sim = _simulate_open_loop(
        eval_inputs,
        params,
        dt_h=dt_h,
        initial_air=initial_air,
        initial_mass=initial_mass,
        initial_q_emit=initial_q_emit,
    )

    train_room_state = _observed_previous_room(df_fit)
    train_q_state = _q_emit_from_room_state(fit_inputs, train_room_state, params, dt_h=dt_h)
    train_features = _power_features(fit_inputs, train_room_state, train_q_state)
    test_features = _power_features(eval_inputs, test_sim.air_before, test_sim.q_emit)

    elec_model = _fit_power_head(train_features, fit_inputs.electric, alpha=args.power_ridge_alpha)
    gas_model = _fit_power_head(train_features, fit_inputs.gas, alpha=args.gas_ridge_alpha)
    pred_electric = np.clip(elec_model.predict(test_features), 0.0, None)
    pred_gas = np.clip(gas_model.predict(test_features), 0.0, None)
    pred_gas[eval_inputs.outdoor > args.gas_cutoff_temp] = 0.0

    metrics = {
        "room_temp": _metrics(eval_inputs.room, test_sim.room),
        "electric_power": _metrics(eval_inputs.electric, pred_electric),
        "gas_consumption": _metrics(eval_inputs.gas, pred_gas),
    }
    params_dict = {name: float(value) for name, value in zip(PARAM_NAMES, params, strict=True)}
    metadata = {
        "model": "ThermalMassPhysics",
        "data_path": str(data_path),
        "report_dir": str(report_dir),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_eval": int(len(eval_inputs.room)),
        "holdout_split": args.holdout_split,
        "fit_on": args.fit_on,
        "dt_h": float(dt_h),
        "train_runtime_s": train_runtime_s,
        "fit_info": fit_info,
        "params": params_dict,
        "config": vars(args),
    }

    (report_dir / "thermal_mass_physics_metrics.json").write_text(
        json.dumps({"metrics": metrics, **metadata}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([
        {"model": "ThermalMassPhysics", "target": target, **vals}
        for target, vals in metrics.items()
    ]).to_csv(report_dir / "thermal_mass_physics_metrics.csv", index=False)
    pd.DataFrame([
        {"parameter": key, "value": value}
        for key, value in params_dict.items()
    ]).to_csv(report_dir / "thermal_mass_physics_params.csv", index=False)

    pred_df = pd.DataFrame({
        "timestamp": eval_inputs.index,
        "true_room_temp": eval_inputs.room,
        "pred_room_temp": test_sim.room,
        "true_electric_power": eval_inputs.electric,
        "pred_electric_power": pred_electric,
        "true_gas_consumption": eval_inputs.gas,
        "pred_gas_consumption": pred_gas,
        "t_mass": test_sim.mass,
        "q_emit": test_sim.q_emit,
        "q_solar": test_sim.q_solar,
        "loss_coeff": test_sim.loss_coeff,
        "heatpump_duty": eval_inputs.heatpump_duty,
        "outdoor_temp": eval_inputs.outdoor,
        "supply_temp": eval_inputs.supply,
        "wind_speed": eval_inputs.wind_speed,
    })
    pred_df.to_csv(report_dir / "thermal_mass_physics_predictions.csv", index=False)
    _write_plot(pred_df, metrics, params_dict, report_dir / "thermal_mass_physics_plot.html")

    if args.deploy_path:
        deploy_path = Path(args.deploy_path)
        deploy_conf = {"data_path": deploy_path.parent}
        deploy_conf["data_path"].mkdir(parents=True, exist_ok=True)
        deploy_logger = logging.getLogger("thermal_mass_physics_deploy")
        deployed = asyncio.run(
            save_json_blob(
                deploy_conf,
                deploy_path.name,
                {"params": params_dict, "fit_info": fit_info, "metrics": metrics},
                deploy_logger,
            )
        )
        if deployed:
            print(f"Deployed fitted params to {deploy_path}", flush=True)
        else:
            print(f"WARNING: failed to deploy fitted params to {deploy_path}", flush=True)

    print(json.dumps({"metrics": metrics, "params": params_dict, "fit_info": fit_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
