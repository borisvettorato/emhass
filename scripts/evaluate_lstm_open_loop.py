"""Train the thermal PINN/LSTM and evaluate the test split open-loop.

Open-loop here means:
- the first test prediction may use the last observed history before test;
- after that, previous test rows for room_temp, electric_power and
  gas_consumption are replaced by the model's own previous predictions before
  constructing the next input window.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from emhass.thermal.feature_engineering import build_feature_matrix
from emhass.thermal.forecast_gridsearch import (
    SearchOptions,
    create_physics_context_sequences,
    create_sequences,
    split_sequences,
)
from emhass.thermal.pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM


LSTM_STATE_FEATURES = [
    # Previous state is safe in a one-step sequence model: input rows end before
    # the target timestamp. Open-loop evaluation overwrites these with forecasts.
    "room_temp",
    "electric_power",
    "gas_consumption",
    # Weather/control inputs.
    "outdoor_temp",
    "humidity",
    "wind_speed",
    "wind_bearing",
    "heatpump_duty",
    "supply_temp",
    "return_temp",
    "flow_rate",
    "pv_power",
    # Hydronic/control heat proxies that do not depend on measured room_temp.
    "supply_temp_x_duty",
    "water_delta_t",
    "thermal_power_proxy",
    "thermal_power_hp_proxy",
    "thermal_power_boiler_proxy",
    # Solar/irradiance signals. Day-of-year is intentionally excluded.
    "ghi",
    "dni",
    "dhi",
    "ghi_norm",
    "dni_norm",
    "dhi_norm",
    "solar_heat",
    "sun_alt_sin",
    "sun_alt_cos",
    "sun_az_sin",
    "sun_az_cos",
    "sun_position_sin",
    "sun_position_cos",
    # Short-cycle time and wind geometry. No doy_sin/doy_cos.
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    "wind_bearing_sin",
    "wind_bearing_cos",
    "wind_u",
    "wind_w",
    "wind_speed_sq",
]


def _load_raw(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df.drop(columns=["sensor.current_electricity_market_price"], errors="ignore")


def _build_features(
    raw_df: pd.DataFrame,
    *,
    opts: SearchOptions,
    feature_profile: str = "standard",
    drop_doy: bool = False,
    drop_rolling: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    include_target_history = feature_profile == "lstm_state"
    include_feature_cols = LSTM_STATE_FEATURES if feature_profile == "lstm_state" else None
    feature_df, feature_cols = build_feature_matrix(
        raw_df,
        feature_level=opts.feature_level,
        latitude=opts.latitude,
        longitude=opts.longitude,
        facade_azimuth_deg=opts.facade_azimuth_deg,
        target_col=opts.target_cols[0],
        exclude_feature_cols=[] if include_target_history else opts.target_cols,
        include_feature_cols=include_feature_cols,
        drop_na=not include_target_history,
    )
    if include_target_history:
        target_history = [c for c in opts.target_cols if c in feature_df.columns]
        feature_cols = target_history + [c for c in feature_cols if c not in target_history]

    filtered: list[str] = []
    for col in feature_cols:
        if drop_doy and col in {"doy_sin", "doy_cos"}:
            continue
        if drop_rolling and "_roll" in col:
            continue
        filtered.append(col)
    if include_target_history:
        required = [c for c in dict.fromkeys(filtered + opts.target_cols) if c in feature_df.columns]
        feature_df = feature_df.dropna(subset=required)
    return feature_df, filtered


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(np.mean(pred - true)),
    }


def _target_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    target_cols: list[str],
) -> dict[str, dict[str, float]]:
    return {name: _metrics(true[:, i], pred[:, i]) for i, name in enumerate(target_cols)}


def _batch_context(
    context: dict[str, np.ndarray] | None,
    start_idx: int,
    end_idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if context is None:
        return None
    return {
        "room_temp_prev": torch.tensor(context["room_temp_prev"][start_idx:end_idx], dtype=torch.float32, device=device),
        "outdoor_temp": torch.tensor(context["outdoor_temp"][start_idx:end_idx], dtype=torch.float32, device=device),
        "solar_heat": torch.tensor(context["solar_heat"][start_idx:end_idx], dtype=torch.float32, device=device),
    }


def _predict_one_step_batches(
    model: QuantilePhysicsInformedLSTM,
    X_split: np.ndarray,
    y_split: np.ndarray,
    *,
    scaler_y: StandardScaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_split), batch_size):
            j = min(i + batch_size, len(X_split))
            xb = torch.tensor(X_split[i:j], dtype=torch.float32, device=device)
            out = model(xb)
            preds.append(out["q50"][:, 0, :].cpu().numpy())
            targets.append(y_split[i:j, 0, :])

    pred_scaled = np.vstack(preds)
    true_scaled = np.vstack(targets)
    return scaler_y.inverse_transform(pred_scaled), scaler_y.inverse_transform(true_scaled)


def _predict_scaled_first_step(
    model: QuantilePhysicsInformedLSTM,
    X_split: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_split), batch_size):
            j = min(i + batch_size, len(X_split))
            xb = torch.tensor(X_split[i:j], dtype=torch.float32, device=device)
            preds.append(model(xb)["q50"][:, 0, :].cpu().numpy())
    return np.vstack(preds)


def _scheduled_sampling_probability(
    epoch: int,
    epochs: int,
    start_prob: float,
    end_prob: float,
) -> float:
    if epochs <= 1:
        return float(end_prob)
    frac = (epoch - 1) / float(epochs - 1)
    return float(start_prob + frac * (end_prob - start_prob))


def _apply_scheduled_sampling(
    model: QuantilePhysicsInformedLSTM,
    X_train: np.ndarray,
    *,
    target_feature_indices: dict[int, int],
    probability: float,
    steps: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace recent target-history feature rows with prior model predictions.

    For sequence k, the row at -lag corresponds to the target timestamp of
    sequence k-lag. Replacing it with that earlier forecast approximates the
    same state feedback used in strict open-loop evaluation.
    """
    if probability <= 0.0 or steps <= 0 or not target_feature_indices:
        return X_train

    max_steps = min(int(steps), X_train.shape[1], len(X_train) - 1)
    if max_steps <= 0:
        return X_train

    pred_scaled = _predict_scaled_first_step(
        model,
        X_train,
        batch_size=batch_size,
        device=device,
    )
    X_aug = X_train.copy()
    for lag in range(1, max_steps + 1):
        rows = np.arange(lag, len(X_train))
        mask = rng.random(len(rows)) < probability
        if not np.any(mask):
            continue
        dst_rows = rows[mask]
        src_rows = dst_rows - lag
        for target_idx, feature_idx in target_feature_indices.items():
            X_aug[dst_rows, -lag, feature_idx] = pred_scaled[src_rows, target_idx]
    return X_aug


def _open_loop_predict(
    model: QuantilePhysicsInformedLSTM,
    raw_df: pd.DataFrame,
    test_ts: np.ndarray,
    *,
    opts: SearchOptions,
    feature_profile: str,
    drop_doy: bool,
    drop_rolling: bool,
    feature_cols: list[str],
    target_cols: list[str],
    input_window: int,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    device: torch.device,
) -> pd.DataFrame:
    state_raw = raw_df.copy()
    rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for ts_raw in test_ts:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            feature_df, _ = _build_features(
                state_raw,
                opts=opts,
                feature_profile=feature_profile,
                drop_doy=drop_doy,
                drop_rolling=drop_rolling,
            )
            if ts not in feature_df.index:
                raise KeyError(f"Timestamp {ts} missing after feature engineering")
            pos = int(feature_df.index.get_loc(ts))
            start = pos - input_window
            if start < 0:
                raise RuntimeError(f"Not enough feature history before {ts}")

            x_window = feature_df.iloc[start:pos][feature_cols].fillna(0.0).to_numpy(dtype=float)
            x_scaled = scaler_x.transform(x_window)
            xb = torch.tensor(x_scaled[np.newaxis, :, :], dtype=torch.float32, device=device)
            pred_scaled = model(xb)["q50"][:, 0, :].cpu().numpy()
            pred = scaler_y.inverse_transform(pred_scaled)[0]

            # Keep state variables in a physically plausible range before they
            # are used as future lags/rolling statistics.
            pred_state = pred.copy()
            for i, col in enumerate(target_cols):
                if col in {"electric_power", "gas_consumption"}:
                    pred_state[i] = max(0.0, pred_state[i])
                state_raw.loc[ts, col] = pred_state[i]

            actual = raw_df.loc[ts, target_cols].to_numpy(dtype=float)
            row: dict[str, object] = {"timestamp": ts}
            for i, col in enumerate(target_cols):
                row[f"true_{col}"] = float(actual[i])
                row[f"pred_{col}"] = float(pred_state[i])
            rows.append(row)

    return pd.DataFrame(rows)


def _build_error_plot(
    pred_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    teacher_forced: pd.DataFrame,
    output_path: Path,
) -> None:
    df = pred_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    tf = teacher_forced.copy()
    tf["timestamp"] = pd.to_datetime(tf["timestamp"], utc=True)
    tf = tf.set_index("timestamp").sort_index()

    ctx_cols = [
        "heatpump_duty", "electric_power", "gas_consumption", "supply_temp",
        "outdoor_temp", "humidity", "wind_speed", "dhi", "dni", "ghi", "pv_power",
    ]
    ctx = raw_df[[c for c in ctx_cols if c in raw_df.columns]].reindex(df.index).ffill().bfill()
    df = df.join(ctx, rsuffix="_raw")
    df = df.join(tf[["teacher_forced_room_temp"]], how="left")
    df["open_loop_error"] = df["pred_room_temp"] - df["true_room_temp"]
    df["teacher_forced_error"] = df["teacher_forced_room_temp"] - df["true_room_temp"]

    x = df.index.tz_convert("Europe/Amsterdam")
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            "Room temperature: actual vs LSTM",
            "Room temperature error: prediction - actual",
            "Heat pump duty and electric power",
            "Supply and outdoor temperature",
            "Irradiance",
            "Gas consumption and wind speed",
        ),
    )

    fig.add_trace(go.Scatter(x=x, y=df["true_room_temp"], mode="lines", name="Actual room temp", line=dict(color="#374151", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["pred_room_temp"], mode="lines", name="LSTM open-loop", line=dict(color="#7c3aed", width=1.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["teacher_forced_room_temp"], mode="lines", name="LSTM one-step", line=dict(color="#0284c7", width=1.3, dash="dash")), row=1, col=1)

    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["open_loop_error"], mode="lines", name="Open-loop error", line=dict(color="#7c3aed", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["teacher_forced_error"], mode="lines", name="One-step error", line=dict(color="#0284c7", width=1.2, dash="dash")), row=2, col=1)

    if "heatpump_duty" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["heatpump_duty"], mode="lines", name="Heatpump duty", line=dict(color="#1d4ed8", width=1.4)), row=3, col=1, secondary_y=False)
    if "electric_power" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["electric_power"], mode="lines", name="Electric power", line=dict(color="#dc2626", width=1.1)), row=3, col=1, secondary_y=True)
    if "supply_temp" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["supply_temp"], mode="lines", name="Supply temp", line=dict(color="#b45309", width=1.2)), row=4, col=1, secondary_y=False)
    if "outdoor_temp" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["outdoor_temp"], mode="lines", name="Outdoor temp", line=dict(color="#0284c7", width=1.2)), row=4, col=1, secondary_y=True)
    for col, color in [("ghi", "#f59e0b"), ("dni", "#ca8a04"), ("dhi", "#a16207")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col.upper(), line=dict(color=color, width=1.0)), row=5, col=1)
    if "gas_consumption" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["gas_consumption"], mode="lines", name="Gas consumption", line=dict(color="#16a34a", width=1.0)), row=6, col=1, secondary_y=False)
    if "wind_speed" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df["wind_speed"], mode="lines", name="Wind speed", line=dict(color="#64748b", width=1.0)), row=6, col=1, secondary_y=True)

    ol = _metrics(df["true_room_temp"].to_numpy(dtype=float), df["pred_room_temp"].to_numpy(dtype=float))
    tfm = _metrics(df["true_room_temp"].to_numpy(dtype=float), df["teacher_forced_room_temp"].to_numpy(dtype=float))
    fig.add_annotation(
        text=(
            f"Open-loop: MAE {ol['mae']:.3f} C, RMSE {ol['rmse']:.3f} C, bias {ol['bias']:+.3f} C<br>"
            f"One-step: MAE {tfm['mae']:.3f} C, RMSE {tfm['rmse']:.3f} C, bias {tfm['bias']:+.3f} C"
        ),
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
        font=dict(size=12),
    )
    fig.update_layout(
        title=(
            "LSTM Open-Loop Error Analysis "
            f"({x.min().strftime('%Y-%m-%d %H:%M')} - {x.max().strftime('%Y-%m-%d %H:%M')} Europe/Amsterdam)"
        ),
        height=1450,
        width=1500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="C", row=1, col=1)
    fig.update_yaxes(title_text="C error", row=2, col=1)
    fig.update_yaxes(title_text="Duty", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="W", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Supply C", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Outdoor C", row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text="W/m2", row=5, col=1)
    fig.update_yaxes(title_text="m3/interval", row=6, col=1, secondary_y=False)
    fig.update_yaxes(title_text="wind", row=6, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=6, col=1)
    fig.write_html(str(output_path))


def main() -> None:
    logging.getLogger("emhass.thermal.feature_engineering").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Train LSTM/PINN and evaluate test split open-loop")
    parser.add_argument("--data-path", default="tests_thermal/data/test_data_prepared.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/lstm_prepared_gpu_openloop")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--feature-level", choices=["minimal", "standard", "full"], default="standard")
    parser.add_argument("--feature-profile", choices=["standard", "lstm_state"], default="standard")
    parser.add_argument("--drop-doy", action="store_true")
    parser.add_argument("--drop-rolling", action="store_true")
    parser.add_argument("--target-cols", default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--physics-loss-weight", type=float, default=0.10)
    parser.add_argument("--physics-balance-weight", type=float, default=0.05)
    parser.add_argument("--scheduled-sampling-start", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-end", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    opts = SearchOptions(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lookahead=args.lookahead,
        feature_level=args.feature_level,
        target_cols=target_cols,
        physics_loss_weight=args.physics_loss_weight,
        physics_balance_weight=args.physics_balance_weight,
        seed=args.seed,
    )

    raw_df = _load_raw(Path(args.data_path))
    feature_df, feature_cols = _build_features(
        raw_df,
        opts=opts,
        feature_profile=args.feature_profile,
        drop_doy=args.drop_doy,
        drop_rolling=args.drop_rolling,
    )
    target_cols = [c for c in target_cols if c in feature_df.columns]
    opts.target_cols = target_cols

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(feature_df[feature_cols].to_numpy(dtype=float))
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(feature_df[target_cols].to_numpy(dtype=float))

    X_seq, y_seq = create_sequences(X, y, lookback=args.input_window, lookahead=args.lookahead)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows to create sequences")
    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_train), len(X_val)
    seq_start_ts = feature_df.index[args.input_window : args.input_window + len(X_seq)].to_numpy()
    train_ts = seq_start_ts[:n_train]
    val_ts = seq_start_ts[n_train : n_train + n_val]
    test_ts = seq_start_ts[n_train + n_val :]

    physics_signals = None
    if {"room_temp", "electric_power", "gas_consumption"}.issubset(set(target_cols)):
        outdoor_scaled = X[:, feature_cols.index("outdoor_temp")] if "outdoor_temp" in feature_cols else np.zeros(len(feature_df))
        solar_scaled = X[:, feature_cols.index("solar_heat")] if "solar_heat" in feature_cols else np.zeros(len(feature_df))
        room_idx = target_cols.index("room_temp")
        physics_signals = {
            "room_temp": y[:, room_idx].astype(np.float32),
            "outdoor_temp": outdoor_scaled.astype(np.float32),
            "solar_heat": solar_scaled.astype(np.float32),
        }
    context_seq = (
        create_physics_context_sequences(
            room_temp_signal=physics_signals["room_temp"],
            outdoor_signal=physics_signals["outdoor_temp"],
            solar_signal=physics_signals["solar_heat"],
            lookback=args.input_window,
            lookahead=args.lookahead,
        )
        if physics_signals is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantilePhysicsInformedLSTM(
        input_size=X.shape[1],
        hidden=args.hidden_size,
        num_layers=args.num_layers,
        lookahead=args.lookahead,
        targets=y.shape[1],
        dropout=0.0 if args.num_layers == 1 else 0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = QuantileLoss(
        weight_physics=args.physics_loss_weight,
        weight_physics_balance=args.physics_balance_weight,
    )
    target_feature_indices = {
        target_idx: feature_cols.index(target)
        for target_idx, target in enumerate(target_cols)
        if target in feature_cols
    }
    ss_rng = np.random.default_rng(args.seed)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        ss_prob = _scheduled_sampling_probability(
            epoch,
            args.epochs,
            args.scheduled_sampling_start,
            args.scheduled_sampling_end,
        )
        X_train_epoch = _apply_scheduled_sampling(
            model,
            X_train,
            target_feature_indices=target_feature_indices,
            probability=ss_prob,
            steps=args.scheduled_sampling_steps,
            batch_size=args.batch_size,
            device=device,
            rng=ss_rng,
        )
        model.train()
        train_losses = []
        for i in range(0, len(X_train), args.batch_size):
            j = min(i + args.batch_size, len(X_train))
            xb = torch.tensor(X_train_epoch[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[i:j], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            out = model(xb)
            ctx = _batch_context(context_seq, i, j, device)
            loss = loss_fn(out, yb, physics_context=ctx)["total"]
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                j = min(i + args.batch_size, len(X_val))
                xb = torch.tensor(X_val[i:j], dtype=torch.float32, device=device)
                yb = torch.tensor(y_val[i:j], dtype=torch.float32, device=device)
                out = model(xb)
                ctx = _batch_context(context_seq, n_train + i, n_train + j, device)
                val_losses.append(float(loss_fn(out, yb, physics_context=ctx)["total"].item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"scheduled_sampling={ss_prob:.3f}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"early_stop_epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    teacher_pred, teacher_true = _predict_one_step_batches(
        model,
        X_test,
        y_test,
        scaler_y=scaler_y,
        batch_size=args.batch_size,
        device=device,
    )
    teacher_df = pd.DataFrame({"timestamp": pd.DatetimeIndex(test_ts)})
    for i, col in enumerate(target_cols):
        teacher_df[f"teacher_forced_{col}"] = teacher_pred[:, i]
        teacher_df[f"true_{col}"] = teacher_true[:, i]

    open_loop_df = _open_loop_predict(
        model,
        raw_df,
        test_ts,
        opts=opts,
        feature_profile=args.feature_profile,
        drop_doy=args.drop_doy,
        drop_rolling=args.drop_rolling,
        feature_cols=feature_cols,
        target_cols=target_cols,
        input_window=args.input_window,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        device=device,
    )

    open_true = open_loop_df[[f"true_{c}" for c in target_cols]].to_numpy(dtype=float)
    open_pred = open_loop_df[[f"pred_{c}" for c in target_cols]].to_numpy(dtype=float)
    teacher_true_arr = teacher_df[[f"true_{c}" for c in target_cols]].to_numpy(dtype=float)
    teacher_pred_arr = teacher_df[[f"teacher_forced_{c}" for c in target_cols]].to_numpy(dtype=float)

    metrics = {
        "config": {
            "device": str(device),
            "input_window": args.input_window,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "lookahead": args.lookahead,
            "feature_level": args.feature_level,
            "feature_profile": args.feature_profile,
            "drop_doy": args.drop_doy,
            "drop_rolling": args.drop_rolling,
            "n_features": len(feature_cols),
            "scheduled_sampling_start": args.scheduled_sampling_start,
            "scheduled_sampling_end": args.scheduled_sampling_end,
            "scheduled_sampling_steps": args.scheduled_sampling_steps,
            "epochs_requested": args.epochs,
            "epochs_ran": len(history["train_loss"]),
        },
        "teacher_forced_test": _target_metrics(teacher_true_arr, teacher_pred_arr, target_cols),
        "open_loop_test": _target_metrics(open_true, open_pred, target_cols),
    }

    (report_dir / "lstm_open_loop_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = []
    for mode in ["teacher_forced_test", "open_loop_test"]:
        for target, vals in metrics[mode].items():
            rows.append({"mode": mode, "target": target, **vals})
    pd.DataFrame(rows).to_csv(report_dir / "lstm_open_loop_metrics.csv", index=False)
    open_loop_df.to_csv(report_dir / "lstm_open_loop_predictions.csv", index=False)
    teacher_df.to_csv(report_dir / "lstm_teacher_forced_predictions.csv", index=False)
    (report_dir / "lstm_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    _build_error_plot(
        open_loop_df,
        raw_df,
        teacher_forced=teacher_df,
        output_path=report_dir / "lstm_open_loop_error_analysis.html",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "metrics": metrics,
        },
        report_dir / "lstm_open_loop_model.pt",
    )

    print(json.dumps(metrics["open_loop_test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
