"""Compare multi-horizon vs one-step thermal PINN/LSTM models.

This script trains two models on the same data:
- multi-horizon: direct forecast over a configurable horizon
- one-step: direct next-step forecast

The comparison is performed on the aligned first forecast step for each test
sequence so both models are evaluated on the same timestamps and targets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

from emhass.thermal.feature_engineering import build_feature_matrix
from emhass.thermal.forecast_gridsearch import (
    SearchOptions,
    _prepare_features,
    create_physics_context_sequences,
    create_sequences,
    split_sequences,
)
from emhass.thermal.pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM


@dataclass
class ExperimentResult:
    name: str
    history: dict[str, list[float]]
    timestamps: np.ndarray
    true_step: np.ndarray
    pred_step: np.ndarray
    metrics_by_target: dict[str, dict[str, float]]


TARGET_DEFAULTS = ["room_temp", "electric_power", "gas_consumption"]


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


def _compute_metrics_by_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(target_names):
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
        }
    return metrics


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def _build_aligned_index(data_path: Path, opts: SearchOptions) -> pd.DatetimeIndex:
    raw_df = pd.read_csv(data_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.set_index("timestamp")
    raw_df = raw_df.drop(columns=["sensor.current_electricity_market_price"], errors="ignore")
    feature_df, _ = build_feature_matrix(
        raw_df,
        feature_level=opts.feature_level,
        latitude=opts.latitude,
        longitude=opts.longitude,
        facade_azimuth_deg=opts.facade_azimuth_deg,
        target_col=opts.target_cols[0],
        exclude_feature_cols=opts.target_cols,
        drop_na=True,
    )
    return feature_df.index


def _train_and_eval(
    *,
    name: str,
    data_path: Path,
    input_window: int,
    lookahead: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    patience: int,
    batch_size: int,
    feature_level: str,
    target_cols: list[str],
    physics_loss_weight: float,
    physics_balance_weight: float,
    seed: int,
    device: torch.device,
) -> ExperimentResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    opts = SearchOptions(
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        lookahead=lookahead,
        feature_level=feature_level,
        target_cols=target_cols,
        physics_loss_weight=physics_loss_weight,
        physics_balance_weight=physics_balance_weight,
        seed=seed,
    )

    X, y, scaler_y, _, physics_signals = _prepare_features(data_path, opts=opts)
    aligned_index = _build_aligned_index(data_path, opts)
    X_seq, y_seq = create_sequences(X, y, lookback=input_window, lookahead=lookahead)
    if len(X_seq) == 0:
        raise RuntimeError(f"{name}: not enough rows to create sequences")

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(f"{name}: split too small")

    n_seq = len(X_seq)
    seq_start_ts = aligned_index[input_window : input_window + n_seq].to_numpy()

    context_seq = None
    if physics_signals is not None:
        context_seq = create_physics_context_sequences(
            room_temp_signal=physics_signals["room_temp"],
            outdoor_signal=physics_signals["outdoor_temp"],
            solar_signal=physics_signals["solar_heat"],
            lookback=input_window,
            lookahead=lookahead,
        )

    model = QuantilePhysicsInformedLSTM(
        input_size=X.shape[1],
        hidden=hidden_size,
        num_layers=num_layers,
        lookahead=lookahead,
        targets=y.shape[1],
        dropout=0.0 if num_layers == 1 else 0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = QuantileLoss(
        weight_physics=physics_loss_weight,
        weight_physics_balance=physics_balance_weight,
    )

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for i in range(0, len(X_train), batch_size):
            j = min(i + batch_size, len(X_train))
            xb = torch.tensor(X_train[i:j], dtype=torch.float32, device=device)
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
            for i in range(0, len(X_val), batch_size):
                j = min(i + batch_size, len(X_val))
                xb = torch.tensor(X_val[i:j], dtype=torch.float32, device=device)
                yb = torch.tensor(y_val[i:j], dtype=torch.float32, device=device)
                out = model(xb)
                ctx = _batch_context(context_seq, len(X_train) + i, len(X_train) + j, device)
                val_losses.append(float(loss_fn(out, yb, physics_context=ctx)["total"].item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    n_train = len(X_train)
    n_val = len(X_val)
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            j = min(i + batch_size, len(X_test))
            xb = torch.tensor(X_test[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_test[i:j], dtype=torch.float32, device=device)
            out = model(xb)
            test_preds.append(out["q50"].cpu().numpy())
            test_targets.append(yb.cpu().numpy())

    test_pred = np.vstack(test_preds).reshape(len(X_test), lookahead, y.shape[1])
    test_true = np.vstack(test_targets).reshape(len(X_test), lookahead, y.shape[1])
    test_pred_dn = scaler_y.inverse_transform(test_pred.reshape(-1, y.shape[1])).reshape(len(X_test), lookahead, y.shape[1])
    test_true_dn = scaler_y.inverse_transform(test_true.reshape(-1, y.shape[1])).reshape(len(X_test), lookahead, y.shape[1])

    test_ts = seq_start_ts[n_train + n_val :]
    pred_step = test_pred_dn[:, 0, :]
    true_step = test_true_dn[:, 0, :]
    metrics_by_target = _compute_metrics_by_target(true_step, pred_step, target_cols)

    return ExperimentResult(
        name=name,
        history=history,
        timestamps=test_ts,
        true_step=true_step,
        pred_step=pred_step,
        metrics_by_target=metrics_by_target,
    )


def _build_comparison_plot(
    multi_result: ExperimentResult,
    one_step_result: ExperimentResult,
    target_names: list[str],
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Training loss comparison",
            "Room temperature: actual vs forecasts",
            "Electric power: actual vs forecasts",
            "Gas consumption: actual vs forecasts",
        ),
    )

    epochs_multi = np.arange(1, len(multi_result.history["train_loss"]) + 1)
    epochs_one = np.arange(1, len(one_step_result.history["train_loss"]) + 1)
    fig.add_trace(go.Scatter(x=epochs_multi, y=multi_result.history["train_loss"], mode="lines", name="multi train_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_multi, y=multi_result.history["val_loss"], mode="lines", name="multi val_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_one, y=one_step_result.history["train_loss"], mode="lines", name="one-step train_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs_one, y=one_step_result.history["val_loss"], mode="lines", name="one-step val_loss"), row=1, col=1)

    for idx, target_name in enumerate(target_names):
        row = idx + 2
        x_actual, y_actual = _downsample(multi_result.timestamps, multi_result.true_step[:, idx])
        x_multi, y_multi = _downsample(multi_result.timestamps, multi_result.pred_step[:, idx])
        x_one, y_one = _downsample(one_step_result.timestamps, one_step_result.pred_step[:, idx])

        fig.add_trace(go.Scatter(x=x_actual, y=y_actual, mode="lines", name=f"{target_name} actual"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x_multi, y=y_multi, mode="lines", name=f"{target_name} multi-horizon"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x_one, y=y_one, mode="lines", name=f"{target_name} one-step"), row=row, col=1)

    fig.update_layout(
        title="Multi-horizon vs one-step thermal LSTM comparison",
        template="plotly_white",
        height=1400,
        width=1500,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (C)", row=2, col=1)
    fig.update_yaxes(title_text="Power", row=3, col=1)
    fig.update_yaxes(title_text="Gas", row=4, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Timestamp", row=4, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-horizon vs one-step thermal LSTM/PINN")
    parser.add_argument("--data-path", type=str, default="tests_thermal/data/test_data.csv")
    parser.add_argument("--report-dir", type=str, default="tests_thermal/reports/horizon_compare")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--multi-lookahead", type=int, default=24)
    parser.add_argument("--feature-level", type=str, default="standard", choices=["minimal", "standard", "full"])
    parser.add_argument("--target-cols", type=str, default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--physics-loss-weight", type=float, default=0.10)
    parser.add_argument("--physics-balance-weight", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()] or TARGET_DEFAULTS
    data_path = Path(args.data_path)

    multi_result = _train_and_eval(
        name="multi_horizon",
        data_path=data_path,
        input_window=args.input_window,
        lookahead=args.multi_lookahead,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        feature_level=args.feature_level,
        target_cols=target_cols,
        physics_loss_weight=args.physics_loss_weight,
        physics_balance_weight=args.physics_balance_weight,
        seed=args.seed,
        device=device,
    )

    one_step_result = _train_and_eval(
        name="one_step",
        data_path=data_path,
        input_window=args.input_window,
        lookahead=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        feature_level=args.feature_level,
        target_cols=target_cols,
        physics_loss_weight=args.physics_loss_weight,
        physics_balance_weight=args.physics_balance_weight,
        seed=args.seed,
        device=device,
    )

    metrics = {
        "multi_horizon": multi_result.metrics_by_target,
        "one_step": one_step_result.metrics_by_target,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "input_window": args.input_window,
            "multi_lookahead": args.multi_lookahead,
            "one_step_lookahead": 1,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        },
    }
    metrics_path = report_dir / "horizon_comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for model_name, model_metrics in [("multi_horizon", multi_result.metrics_by_target), ("one_step", one_step_result.metrics_by_target)]:
        for target_name, target_metrics in model_metrics.items():
            rows.append({"model": model_name, "target": target_name, **target_metrics})
    metrics_csv_path = report_dir / "horizon_comparison_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    plot_path = report_dir / "horizon_comparison_plot.html"
    _build_comparison_plot(multi_result, one_step_result, target_cols, plot_path)

    print(f"Saved comparison metrics JSON: {metrics_path}")
    print(f"Saved comparison metrics CSV: {metrics_csv_path}")
    print(f"Saved comparison plot HTML: {plot_path}")


if __name__ == "__main__":
    main()
