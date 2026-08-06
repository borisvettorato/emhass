"""Train a no-leak RF residual corrector on top of the physics model.

The residual RF is fitted on train+validation residuals only:

    residual = actual - physics_prediction

During test, the physical model runs open-loop and the RF predicts only the
residual from non-target sensor/control features. No measured test room
temperature or measured test error is fed back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_ensemble import (  # noqa: E402
    _build_aligned_index,
    _build_baseline_features,
    _load_raw_df,
    _physics_features,
    _previous_observed_series,
    _recursive_physics_predict,
    _rls_fit_theta,
)
from emhass.thermal.forecast_gridsearch import (  # noqa: E402
    SearchOptions,
    _prepare_features,
    create_sequences,
    split_sequences,
)


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(np.mean(err)),
    }


def _slice(raw_df: pd.DataFrame, ts: np.ndarray) -> pd.DataFrame:
    return raw_df.reindex(pd.DatetimeIndex(ts)).dropna(how="all")


def _fit_physics(
    raw_df: pd.DataFrame,
    df_train: pd.DataFrame,
    *,
    forgetting: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx = pd.DatetimeIndex(df_train.index)
    room_train_state = _previous_observed_series(raw_df, train_idx, "room_temp", default=20.0)
    x_train = _physics_features(df_train, room_state=room_train_state)

    theta_elec, _ = _rls_fit_theta(
        x_train,
        df_train["electric_power"].fillna(0.0).to_numpy(dtype=float),
        forgetting=forgetting,
        ridge=ridge,
    )
    theta_gas, _ = _rls_fit_theta(
        x_train,
        df_train["gas_consumption"].fillna(0.0).to_numpy(dtype=float),
        forgetting=forgetting,
        ridge=ridge,
    )
    theta_temp, _ = _rls_fit_theta(
        x_train,
        df_train["room_temp"].fillna(20.0).to_numpy(dtype=float),
        forgetting=forgetting,
        ridge=ridge,
    )
    return theta_elec, theta_gas, theta_temp


def _predict_physics_open_loop(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    *,
    theta_elec: np.ndarray,
    theta_gas: np.ndarray,
    theta_temp: np.ndarray,
    gas_cutoff_temp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = pd.DatetimeIndex(df.index)
    room_state = _previous_observed_series(raw_df, idx, "room_temp", default=20.0)
    initial_room_state = float(room_state.iloc[0]) if len(room_state) else 20.0
    pred_elec, pred_gas, pred_temp, _ = _recursive_physics_predict(
        df,
        theta_elec=theta_elec,
        theta_gas=theta_gas,
        theta_temp=theta_temp,
        initial_room_state=initial_room_state,
    )
    outdoor = df.get("outdoor_temp", pd.Series(10.0, index=df.index)).fillna(10.0).to_numpy(dtype=float)
    pred_gas[outdoor > gas_cutoff_temp] = 0.0
    return pred_elec, pred_gas, pred_temp


def _fit_residual_rf(
    X: np.ndarray,
    residual: np.ndarray,
    *,
    seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, residual)
    return model


def _build_plot(pred_df: pd.DataFrame, metrics: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    df = pred_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    x = df.index.tz_convert("Europe/Amsterdam")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Room temperature: physics vs residual RF",
            "Room temperature error: prediction - actual",
            "Electric power",
            "Gas consumption and heatpump duty",
        ),
    )

    fig.add_trace(go.Scatter(x=x, y=df["true_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["physics_room_temp"], mode="lines", name="Physics open-loop", line=dict(color="#dc2626", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["residual_rf_room_temp"], mode="lines", name="Physics + residual RF", line=dict(color="#16a34a", width=2.2)), row=1, col=1)

    fig.add_hline(y=0, line_width=1, line_color="#6b7280", row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["physics_room_temp"] - df["true_room_temp"], mode="lines", name="Physics error", line=dict(color="#dc2626", width=1.2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["residual_rf_room_temp"] - df["true_room_temp"], mode="lines", name="Residual RF error", line=dict(color="#16a34a", width=1.4), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["true_electric_power"], mode="lines", name="Actual electric", line=dict(color="#111827", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["physics_electric_power"], mode="lines", name="Physics electric", line=dict(color="#dc2626", width=1.0, dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["residual_rf_electric_power"], mode="lines", name="Residual RF electric", line=dict(color="#16a34a", width=1.2)), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["true_gas_consumption"], mode="lines", name="Actual gas", line=dict(color="#111827", width=1.2)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["residual_rf_gas_consumption"], mode="lines", name="Residual RF gas", line=dict(color="#16a34a", width=1.1)), row=4, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["heatpump_duty"], mode="lines", name="Heatpump duty", line=dict(color="#2563eb", width=1.0)), row=4, col=1, secondary_y=True)

    summary = "<br>".join(
        f"{model} {target}: MAE {vals['mae']:.3f}, RMSE {vals['rmse']:.3f}, bias {vals['bias']:+.3f}"
        for model, by_target in metrics.items()
        for target, vals in by_target.items()
        if target == "room_temp"
    )
    fig.add_annotation(
        text=summary,
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        font=dict(size=11),
    )
    fig.update_layout(
        title=f"No-Leak Physics + RF Residual ({x.min().strftime('%Y-%m-%d %H:%M')} - {x.max().strftime('%Y-%m-%d %H:%M')} Europe/Amsterdam)",
        template="plotly_white",
        height=1200,
        width=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="C", row=1, col=1)
    fig.update_yaxes(title_text="C error", row=2, col=1)
    fig.update_yaxes(title_text="W", row=3, col=1)
    fig.update_yaxes(title_text="m3/interval", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="duty", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=4, col=1)
    fig.write_html(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="No-leak Physics + RF residual correction")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/phys_rf_residual_noleak")
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--feature-level", choices=["minimal", "standard", "full"], default="standard")
    parser.add_argument("--target-cols", default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--physics-forgetting-factor", type=float, default=0.995)
    parser.add_argument("--physics-ridge", type=float, default=1.0)
    parser.add_argument("--selflearn-gas-cutoff-temp", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    opts = SearchOptions(
        lookahead=args.lookahead,
        feature_level=args.feature_level,
        target_cols=target_cols,
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

    raw_df = _load_raw_df(data_path)
    df_train = _slice(raw_df, train_ts)
    df_val = _slice(raw_df, val_ts)
    df_train_val = _slice(raw_df, train_val_ts)
    df_test = _slice(raw_df, test_ts)

    theta_elec, theta_gas, theta_temp = _fit_physics(
        raw_df,
        df_train,
        forgetting=float(args.physics_forgetting_factor),
        ridge=float(args.physics_ridge),
    )

    train_val_phys = _predict_physics_open_loop(
        raw_df,
        df_train_val,
        theta_elec=theta_elec,
        theta_gas=theta_gas,
        theta_temp=theta_temp,
        gas_cutoff_temp=float(args.selflearn_gas_cutoff_temp),
    )
    test_phys = _predict_physics_open_loop(
        raw_df,
        df_test,
        theta_elec=theta_elec,
        theta_gas=theta_gas,
        theta_temp=theta_temp,
        gas_cutoff_temp=float(args.selflearn_gas_cutoff_temp),
    )

    X_res_train = _build_baseline_features(df_train_val).to_numpy(dtype=float)
    X_res_test = _build_baseline_features(df_test).to_numpy(dtype=float)

    y_train_val_e = df_train_val["electric_power"].fillna(0.0).to_numpy(dtype=float)
    y_train_val_g = df_train_val["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    y_train_val_t = df_train_val["room_temp"].fillna(20.0).to_numpy(dtype=float)

    residual_e = y_train_val_e - train_val_phys[0]
    residual_g = y_train_val_g - train_val_phys[1]
    residual_t = y_train_val_t - train_val_phys[2]

    rf_e = _fit_residual_rf(X_res_train, residual_e, seed=args.seed + 10)
    rf_g = _fit_residual_rf(X_res_train, residual_g, seed=args.seed + 11)
    rf_t = _fit_residual_rf(X_res_train, residual_t, seed=args.seed + 12)

    pred_e_phys, pred_g_phys, pred_t_phys = test_phys
    pred_e = np.clip(pred_e_phys + rf_e.predict(X_res_test), a_min=0.0, a_max=None)
    pred_g = np.clip(pred_g_phys + rf_g.predict(X_res_test), a_min=0.0, a_max=None)
    pred_t = pred_t_phys + rf_t.predict(X_res_test)

    true_e = df_test["electric_power"].fillna(0.0).to_numpy(dtype=float)
    true_g = df_test["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    true_t = df_test["room_temp"].fillna(20.0).to_numpy(dtype=float)

    metrics = {
        "PhysicsOpenLoop": {
            "room_temp": _metrics(true_t, pred_t_phys),
            "electric_power": _metrics(true_e, pred_e_phys),
            "gas_consumption": _metrics(true_g, pred_g_phys),
        },
        "PhysicsResidualRFNoLeak": {
            "room_temp": _metrics(true_t, pred_t),
            "electric_power": _metrics(true_e, pred_e),
            "gas_consumption": _metrics(true_g, pred_g),
        },
    }
    (report_dir / "phys_rf_residual_noleak_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = []
    for model, by_target in metrics.items():
        for target, vals in by_target.items():
            rows.append({"model": model, "target": target, **vals})
    pd.DataFrame(rows).to_csv(report_dir / "phys_rf_residual_noleak_metrics.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "timestamp": df_test.index,
            "true_room_temp": true_t,
            "physics_room_temp": pred_t_phys,
            "residual_rf_room_temp": pred_t,
            "true_electric_power": true_e,
            "physics_electric_power": pred_e_phys,
            "residual_rf_electric_power": pred_e,
            "true_gas_consumption": true_g,
            "physics_gas_consumption": pred_g_phys,
            "residual_rf_gas_consumption": pred_g,
            "heatpump_duty": df_test.get("heatpump_duty", pd.Series(0.0, index=df_test.index)).fillna(0.0).to_numpy(dtype=float),
        }
    )
    pred_df.to_csv(report_dir / "phys_rf_residual_noleak_predictions.csv", index=False)
    _build_plot(pred_df, metrics, report_dir / "phys_rf_residual_noleak_plot.html")

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
