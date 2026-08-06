"""Train/val/test PINN evaluation with output plot.

This script trains a single PINN configuration, evaluates on chronological
train/val/test splits, and generates an HTML plot with predictions vs actuals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from emhass.thermal.forecast_gridsearch import (
    SearchOptions,
    create_physics_context_sequences,
    create_sequences,
    split_sequences,
    _prepare_features,
)
from emhass.thermal.feature_engineering import build_feature_matrix
from emhass.thermal.pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM


@dataclass
class TrainConfig:
    input_window: int = 192
    hidden_size: int = 128
    num_layers: int = 2


def _load_config(config_path: Path | None) -> TrainConfig:
    if config_path is None:
        return TrainConfig()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return TrainConfig(
        input_window=int(payload["input_window"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
    )


def _prepare_features_for_training(
    data_path: Path,
    opts: SearchOptions,
    *,
    drop_doy: bool = False,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str], dict[str, np.ndarray] | None]:
    if not drop_doy:
        return _prepare_features(data_path, opts=opts)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing input data: {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.drop(columns=["sensor.current_electricity_market_price"], errors="ignore")

    requested_target_cols = list(opts.target_cols) if opts.target_cols else ["room_temp"]
    feature_df, feature_cols = build_feature_matrix(
        df,
        feature_level=opts.feature_level,
        latitude=opts.latitude,
        longitude=opts.longitude,
        facade_azimuth_deg=opts.facade_azimuth_deg,
        target_col=requested_target_cols[0],
        exclude_feature_cols=requested_target_cols,
        drop_na=True,
    )
    feature_cols = [c for c in feature_cols if c not in {"doy_sin", "doy_cos"}]

    target_cols = [c for c in requested_target_cols if c in feature_df.columns]
    if not target_cols:
        raise KeyError(f"None of requested target columns exist: {requested_target_cols}")
    opts.target_cols = target_cols

    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(feature_df[feature_cols].values)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(feature_df[target_cols].values)

    physics_signals = None
    if {"room_temp", "electric_power", "gas_consumption"}.issubset(set(target_cols)):
        outdoor_scaled = X[:, feature_cols.index("outdoor_temp")] if "outdoor_temp" in feature_cols else np.zeros(len(feature_df), dtype=np.float32)
        solar_scaled = X[:, feature_cols.index("solar_heat")] if "solar_heat" in feature_cols else np.zeros(len(feature_df), dtype=np.float32)
        room_idx = target_cols.index("room_temp")
        physics_signals = {
            "room_temp": y[:, room_idx].astype(np.float32),
            "outdoor_temp": outdoor_scaled.astype(np.float32),
            "solar_heat": solar_scaled.astype(np.float32),
        }

    return X, y, scaler_y, feature_cols, physics_signals


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


def _predict_split(
    model: QuantilePhysicsInformedLSTM,
    X_split: np.ndarray,
    y_split: np.ndarray,
    split_offset: int,
    batch_size: int,
    context_seq: dict[str, np.ndarray] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(X_split), batch_size):
            j = min(i + batch_size, len(X_split))
            xb = torch.tensor(X_split[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_split[i:j], dtype=torch.float32, device=device)
            out = model(xb)

            _ = _batch_context(context_seq, split_offset + i, split_offset + j, device)
            preds.append(out["q50"].cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred_arr = np.vstack(preds).reshape(-1, y_split.shape[-1])
    true_arr = np.vstack(targets).reshape(-1, y_split.shape[-1])
    return pred_arr, true_arr


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_c": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_c": float(mean_absolute_error(y_true, y_pred)),
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


def _build_plot(
    history: dict[str, list[float]],
    train_ts: np.ndarray,
    train_true: np.ndarray,
    train_pred: np.ndarray,
    val_ts: np.ndarray,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_ts: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training history",
            "Train split: actual vs forecast",
            "Validation split: actual vs forecast",
            "Test split: actual vs forecast",
        ),
    )

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="train_loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="val_loss"), row=1, col=1)

    def add_split(
        row: int,
        col: int,
        ts_vals: np.ndarray,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        name: str,
    ) -> None:
        x_true, y_true = _downsample(ts_vals, true_vals)
        x_pred, y_pred = _downsample(ts_vals, pred_vals)
        fig.add_trace(
            go.Scatter(x=x_true, y=y_true, mode="lines", name=f"{name} actual"),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=x_pred, y=y_pred, mode="lines", name=f"{name} forecast q50"),
            row=row,
            col=col,
        )

    add_split(1, 2, train_ts, train_true, train_pred, "train")
    add_split(2, 1, val_ts, val_true, val_pred, "val")
    add_split(2, 2, test_ts, test_true, test_pred, "test")

    fig.update_layout(
        title="PINN train/val/test evaluation",
        template="plotly_white",
        height=900,
        width=1400,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Temperature (C)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (C)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (C)", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def _build_test_targets_plot(
    test_ts: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    n_targets = test_true.shape[1]
    subplot_titles = [f"Test split: {name} actual vs forecast" for name in target_names[:n_targets]]
    fig = make_subplots(rows=n_targets, cols=1, subplot_titles=tuple(subplot_titles), shared_xaxes=True)

    for idx in range(n_targets):
        x_true, y_true = _downsample(test_ts, test_true[:, idx])
        x_pred, y_pred = _downsample(test_ts, test_pred[:, idx])
        fig.add_trace(
            go.Scatter(x=x_true, y=y_true, mode="lines", name=f"{target_names[idx]} actual"),
            row=idx + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_pred, y=y_pred, mode="lines", name=f"{target_names[idx]} forecast q50"),
            row=idx + 1,
            col=1,
        )

    fig.update_layout(
        title="PINN test forecast by target",
        template="plotly_white",
        height=350 * n_targets,
        width=1400,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Timestamp", row=n_targets, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PINN and evaluate train/val/test with plot output")
    parser.add_argument("--data-path", type=str, default="tests_thermal/data/test_data.csv")
    parser.add_argument("--report-dir", type=str, default="tests_thermal/reports/train_val_test")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=144)
    parser.add_argument("--feature-level", type=str, default="standard", choices=["minimal", "standard", "full"])
    parser.add_argument("--drop-doy", action="store_true")
    parser.add_argument("--target-cols", type=str, default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--physics-loss-weight", type=float, default=0.10)
    parser.add_argument("--physics-balance-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_json) if args.config_json else None
    cfg = _load_config(config_path)

    opts = SearchOptions(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lookahead=args.lookahead,
        feature_level=args.feature_level,
        target_cols=[c.strip() for c in args.target_cols.split(",") if c.strip()],
        physics_loss_weight=args.physics_loss_weight,
        physics_balance_weight=args.physics_balance_weight,
        seed=args.seed,
    )

    X, y, scaler_y, _, physics_signals = _prepare_features_for_training(
        Path(args.data_path),
        opts=opts,
        drop_doy=args.drop_doy,
    )

    # Build an aligned timestamp index so the plots use real datetimes on x-axis.
    raw_df = pd.read_csv(args.data_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.set_index("timestamp")
    raw_df = raw_df.drop(columns=["sensor.current_electricity_market_price"], errors="ignore")
    feature_df_index, _ = build_feature_matrix(
        raw_df,
        feature_level=args.feature_level,
        latitude=opts.latitude,
        longitude=opts.longitude,
        facade_azimuth_deg=opts.facade_azimuth_deg,
        target_col=opts.target_cols[0],
        exclude_feature_cols=opts.target_cols,
        drop_na=True,
    )

    X_seq, y_seq = create_sequences(X, y, lookback=cfg.input_window, lookahead=args.lookahead)

    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows to create train/val/test sequences.")

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError("Split too small. Increase data size or reduce lookahead/input_window.")

    n_seq = len(X_seq)
    if len(feature_df_index.index) < cfg.input_window + n_seq:
        raise RuntimeError("Timestamp alignment failed: not enough indexed rows after feature engineering.")
    seq_start_ts = feature_df_index.index[cfg.input_window : cfg.input_window + n_seq].to_numpy()

    context_seq = None
    if physics_signals is not None:
        context_seq = create_physics_context_sequences(
            room_temp_signal=physics_signals["room_temp"],
            outdoor_signal=physics_signals["outdoor_temp"],
            solar_signal=physics_signals["solar_heat"],
            lookback=cfg.input_window,
            lookahead=args.lookahead,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantilePhysicsInformedLSTM(
        input_size=X.shape[1],
        hidden=cfg.hidden_size,
        num_layers=cfg.num_layers,
        lookahead=args.lookahead,
        targets=y.shape[1],
        dropout=0.0 if cfg.num_layers == 1 else 0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = QuantileLoss(
        weight_physics=args.physics_loss_weight,
        weight_physics_balance=args.physics_balance_weight,
    )

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for _epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for i in range(0, len(X_train), args.batch_size):
            j = min(i + args.batch_size, len(X_train))
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
            for i in range(0, len(X_val), args.batch_size):
                j = min(i + args.batch_size, len(X_val))
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
            if patience_counter >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    n_train = len(X_train)
    n_val = len(X_val)

    train_pred, train_true = _predict_split(model, X_train, y_train, 0, args.batch_size, context_seq, device)
    val_pred, val_true = _predict_split(model, X_val, y_val, n_train, args.batch_size, context_seq, device)
    test_pred, test_true = _predict_split(model, X_test, y_test, n_train + n_val, args.batch_size, context_seq, device)

    train_pred_dn = scaler_y.inverse_transform(train_pred)
    val_pred_dn = scaler_y.inverse_transform(val_pred)
    test_pred_dn = scaler_y.inverse_transform(test_pred)
    train_true_dn = scaler_y.inverse_transform(train_true)
    val_true_dn = scaler_y.inverse_transform(val_true)
    test_true_dn = scaler_y.inverse_transform(test_true)

    # Convert flattened arrays back to (n_seq, lookahead, targets) and use first-step
    # values per sequence to get monotonic, timestamp-correct train/val/test curves.
    train_pred_seq = train_pred_dn.reshape(len(X_train), args.lookahead, y.shape[1])
    val_pred_seq = val_pred_dn.reshape(len(X_val), args.lookahead, y.shape[1])
    test_pred_seq = test_pred_dn.reshape(len(X_test), args.lookahead, y.shape[1])
    train_true_seq = train_true_dn.reshape(len(X_train), args.lookahead, y.shape[1])
    val_true_seq = val_true_dn.reshape(len(X_val), args.lookahead, y.shape[1])
    test_true_seq = test_true_dn.reshape(len(X_test), args.lookahead, y.shape[1])

    train_ts = seq_start_ts[:n_train]
    val_ts = seq_start_ts[n_train : n_train + n_val]
    test_ts = seq_start_ts[n_train + n_val :]

    room_idx = 0
    if opts.target_cols and "room_temp" in opts.target_cols:
        room_idx = opts.target_cols.index("room_temp")

    target_names = opts.target_cols if opts.target_cols else ["room_temp", "electric_power", "gas_consumption"]

    metrics = {
        "config": {
            "input_window": cfg.input_window,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "lookahead": args.lookahead,
            "feature_level": args.feature_level,
            "drop_doy": args.drop_doy,
            "n_features": int(X.shape[1]),
        },
        "train": _compute_metrics(train_true_seq[:, 0, room_idx], train_pred_seq[:, 0, room_idx]),
        "val": _compute_metrics(val_true_seq[:, 0, room_idx], val_pred_seq[:, 0, room_idx]),
        "test": _compute_metrics(test_true_seq[:, 0, room_idx], test_pred_seq[:, 0, room_idx]),
        "per_target": {
            "train": _compute_metrics_by_target(train_true_seq[:, 0, :], train_pred_seq[:, 0, :], target_names),
            "val": _compute_metrics_by_target(val_true_seq[:, 0, :], val_pred_seq[:, 0, :], target_names),
            "test": _compute_metrics_by_target(test_true_seq[:, 0, :], test_pred_seq[:, 0, :], target_names),
        },
    }

    metrics_path = report_dir / "train_val_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_path = report_dir / "train_val_test_plot.html"
    _build_plot(
        history,
        train_ts,
        train_true_seq[:, 0, room_idx],
        train_pred_seq[:, 0, room_idx],
        val_ts,
        val_true_seq[:, 0, room_idx],
        val_pred_seq[:, 0, room_idx],
        test_ts,
        test_true_seq[:, 0, room_idx],
        test_pred_seq[:, 0, room_idx],
        plot_path,
    )

    summary = pd.DataFrame(
        [
            {"split": "train", **metrics["train"]},
            {"split": "val", **metrics["val"]},
            {"split": "test", **metrics["test"]},
        ]
    )
    summary_path = report_dir / "train_val_test_metrics.csv"
    summary.to_csv(summary_path, index=False)

    detailed_rows = []
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics["per_target"][split_name]
        for target in target_names:
            detailed_rows.append(
                {
                    "split": split_name,
                    "target": target,
                    "rmse": split_metrics[target]["rmse"],
                    "mae": split_metrics[target]["mae"],
                }
            )
    detailed_path = report_dir / "train_val_test_metrics_by_target.csv"
    pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)

    targets_plot_path = report_dir / "train_val_test_targets_plot.html"
    _build_test_targets_plot(
        test_ts,
        test_true_seq[:, 0, :],
        test_pred_seq[:, 0, :],
        opts.target_cols if opts.target_cols else ["room_temp", "electric_power", "gas_consumption"],
        targets_plot_path,
    )

    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved metrics CSV: {summary_path}")
    print(f"Saved per-target metrics CSV: {detailed_path}")
    print(f"Saved plot HTML: {plot_path}")
    print(f"Saved targets plot HTML: {targets_plot_path}")


if __name__ == "__main__":
    main()
