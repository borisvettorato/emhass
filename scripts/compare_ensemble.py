"""Compare thermal forecasting models for electric and gas consumption.

Trains two models on identical train/val/test splits:
  - QuantilePhysicsInformedLSTM (one-step, lookahead=1)
  - HybridHeatPumpLR (physics-based Ridge + hurdle gas model)

Also benchmarks classical baselines (KNN, RF) and an adaptive self-learning
physical model, then plots all predictions against actual values for
electric_power and gas_consumption on the test set.

Usage
-----
  python scripts/compare_ensemble.py \\
      --data-path tests_thermal/data/test_data.csv \\
      --report-dir tests_thermal/reports/ensemble_compare \\
      --epochs 50 --patience 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emhass.thermal.feature_engineering import build_feature_matrix
from emhass.thermal.feature_engineering import normalise_sensors
from emhass.thermal.forecast_gridsearch import (
    SearchOptions,
    _prepare_features,
    create_physics_context_sequences,
    create_sequences,
    split_sequences,
)
from emhass.thermal.hybrid_heatpump_lr import HybridHeatPumpLR
from emhass.thermal.pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelResult:
    name: str
    timestamps: np.ndarray
    true_elec: np.ndarray
    true_gas: np.ndarray
    pred_elec: np.ndarray
    pred_gas: np.ndarray
    true_temp: np.ndarray | None = None
    pred_temp: np.ndarray | None = None
    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_runtime_s: float = float("nan")
    test_runtime_s: float = float("nan")


# ============================================================================
# HELPERS
# ============================================================================

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _metrics(true: np.ndarray, pred: np.ndarray, name: str) -> dict[str, float]:
    return {"rmse": _rmse(true, pred), "mae": _mae(true, pred), "name": name}


def _batch_context(
    context: dict[str, np.ndarray] | None,
    start_idx: int,
    end_idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if context is None:
        return None
    return {
        "room_temp_prev": torch.tensor(
            context["room_temp_prev"][start_idx:end_idx], dtype=torch.float32, device=device
        ),
        "outdoor_temp": torch.tensor(
            context["outdoor_temp"][start_idx:end_idx], dtype=torch.float32, device=device
        ),
        "solar_heat": torch.tensor(
            context["solar_heat"][start_idx:end_idx], dtype=torch.float32, device=device
        ),
    }


def _downsample(
    x: np.ndarray, y: np.ndarray, max_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def _build_aligned_index(data_path: Path, opts: SearchOptions) -> pd.DatetimeIndex:
    raw = pd.read_csv(data_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index("timestamp").drop(
        columns=["sensor.current_electricity_market_price"], errors="ignore"
    )
    feature_df, _ = build_feature_matrix(
        raw,
        feature_level=opts.feature_level,
        latitude=opts.latitude,
        longitude=opts.longitude,
        target_col=opts.target_cols[0],
        exclude_feature_cols=opts.target_cols,
        drop_na=True,
    )
    return feature_df.index


# ============================================================================
# LSTM TRAINING
# ============================================================================

def _train_lstm(
    *,
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
) -> ModelResult:
    """Train a one-step LSTM and return test-split predictions."""
    t0 = time.perf_counter()
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

    n_seq = len(X_seq)
    seq_start_ts = aligned_index[input_window : input_window + n_seq].to_numpy()

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_train), len(X_val)

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

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
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
                ctx = _batch_context(context_seq, n_train + i, n_train + j, device)
                val_losses.append(float(loss_fn(out, yb, physics_context=ctx)["total"].item()))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("LSTM early stop at epoch %d/%d", epoch, epochs)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    t_trained = time.perf_counter()

    # Inference on test set
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            j = min(i + batch_size, len(X_test))
            xb = torch.tensor(X_test[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_test[i:j], dtype=torch.float32, device=device)
            out = model(xb)
            preds.append(out["q50"].cpu().numpy())
            targets.append(yb.cpu().numpy())

    test_pred = np.vstack(preds).reshape(len(X_test), lookahead, y.shape[1])
    test_true = np.vstack(targets).reshape(len(X_test), lookahead, y.shape[1])
    test_pred_dn = scaler_y.inverse_transform(
        test_pred.reshape(-1, y.shape[1])
    ).reshape(len(X_test), lookahead, y.shape[1])
    test_true_dn = scaler_y.inverse_transform(
        test_true.reshape(-1, y.shape[1])
    ).reshape(len(X_test), lookahead, y.shape[1])

    # First forecast step → aligned with test timestamps
    pred_step = test_pred_dn[:, 0, :]
    true_step = test_true_dn[:, 0, :]
    test_ts = seq_start_ts[n_train + n_val :]

    elec_idx = target_cols.index("electric_power")
    gas_idx  = target_cols.index("gas_consumption")
    room_idx = target_cols.index("room_temp") if "room_temp" in target_cols else None

    metrics = {
        "electric_power": _metrics(true_step[:, elec_idx], pred_step[:, elec_idx], "LSTM"),
        "gas_consumption": _metrics(true_step[:, gas_idx], pred_step[:, gas_idx], "LSTM"),
    }
    true_temp = None
    pred_temp = None
    if room_idx is not None:
        true_temp = true_step[:, room_idx]
        pred_temp = pred_step[:, room_idx]
        metrics["room_temp"] = _metrics(true_temp, pred_temp, "LSTM")

    return ModelResult(
        name="LSTM",
        timestamps=test_ts,
        true_elec=true_step[:, elec_idx],
        true_gas=true_step[:, gas_idx],
        pred_elec=pred_step[:, elec_idx],
        pred_gas=pred_step[:, gas_idx],
        true_temp=true_temp,
        pred_temp=pred_temp,
        history=history,
        metrics=metrics,
        train_runtime_s=float(t_trained - t0),
        test_runtime_s=float(time.perf_counter() - t_trained),
    )


class _GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.2,
        )
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def _train_gru(
    *,
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
    seed: int,
    device: torch.device,
    enable_quantized_inference: bool = False,
) -> tuple[ModelResult, ModelResult | None]:
    """Train a CPU-friendly GRU baseline on the same one-step setup as LSTM."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)

    opts = SearchOptions(
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        lookahead=lookahead,
        feature_level=feature_level,
        target_cols=target_cols,
        seed=seed,
    )

    X, y, scaler_y, _, _ = _prepare_features(data_path, opts=opts)
    aligned_index = _build_aligned_index(data_path, opts)
    X_seq, y_seq = create_sequences(X, y, lookback=input_window, lookahead=lookahead)

    n_seq = len(X_seq)
    seq_start_ts = aligned_index[input_window : input_window + n_seq].to_numpy()

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_train), len(X_val)

    output_dim = lookahead * y.shape[1]
    model = _GRURegressor(
        input_size=X.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for i in range(0, len(X_train), batch_size):
            j = min(i + batch_size, len(X_train))
            xb = torch.tensor(X_train[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[i:j], dtype=torch.float32, device=device).reshape(j - i, -1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                j = min(i + batch_size, len(X_val))
                xb = torch.tensor(X_val[i:j], dtype=torch.float32, device=device)
                yb = torch.tensor(y_val[i:j], dtype=torch.float32, device=device).reshape(j - i, -1)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))

        t_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        v_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("GRU early stop at epoch %d/%d", epoch, epochs)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    t_trained = time.perf_counter()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            j = min(i + batch_size, len(X_test))
            xb = torch.tensor(X_test[i:j], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())

    pred_flat = np.vstack(preds)
    pred_seq = pred_flat.reshape(len(X_test), lookahead, y.shape[1])
    true_seq = y_test.reshape(len(X_test), lookahead, y.shape[1])

    pred_dn = scaler_y.inverse_transform(pred_seq.reshape(-1, y.shape[1])).reshape(len(X_test), lookahead, y.shape[1])
    true_dn = scaler_y.inverse_transform(true_seq.reshape(-1, y.shape[1])).reshape(len(X_test), lookahead, y.shape[1])

    pred_step = pred_dn[:, 0, :]
    true_step = true_dn[:, 0, :]
    test_ts = seq_start_ts[n_train + n_val :]

    elec_idx = target_cols.index("electric_power")
    gas_idx  = target_cols.index("gas_consumption")
    room_idx = target_cols.index("room_temp") if "room_temp" in target_cols else None

    metrics = {
        "electric_power": _metrics(true_step[:, elec_idx], pred_step[:, elec_idx], "GRU"),
        "gas_consumption": _metrics(true_step[:, gas_idx], pred_step[:, gas_idx], "GRU"),
    }
    true_temp = None
    pred_temp = None
    if room_idx is not None:
        true_temp = true_step[:, room_idx]
        pred_temp = pred_step[:, room_idx]
        metrics["room_temp"] = _metrics(true_temp, pred_temp, "GRU")

    gru_result = ModelResult(
        name="GRU",
        timestamps=test_ts,
        true_elec=true_step[:, elec_idx],
        true_gas=true_step[:, gas_idx],
        pred_elec=pred_step[:, elec_idx],
        pred_gas=pred_step[:, gas_idx],
        true_temp=true_temp,
        pred_temp=pred_temp,
        history=history,
        metrics=metrics,
        train_runtime_s=float(t_trained - t0),
        test_runtime_s=float(time.perf_counter() - t_trained),
    )

    quantized_result: ModelResult | None = None
    if enable_quantized_inference:
        if device.type != "cpu":
            logger.info("Skipping GRU quantized inference benchmark: requires CPU device")
        else:
            try:
                q_model = torch.quantization.quantize_dynamic(
                    model.cpu(),
                    {nn.GRU, nn.Linear},
                    dtype=torch.qint8,
                )
                q_model.eval()
                q_preds = []
                q_t0 = time.perf_counter()
                with torch.no_grad():
                    for i in range(0, len(X_test), batch_size):
                        j = min(i + batch_size, len(X_test))
                        xb = torch.tensor(X_test[i:j], dtype=torch.float32, device=torch.device("cpu"))
                        q_preds.append(q_model(xb).cpu().numpy())
                q_test_s = float(time.perf_counter() - q_t0)

                q_pred_flat = np.vstack(q_preds)
                q_pred_seq = q_pred_flat.reshape(len(X_test), lookahead, y.shape[1])
                q_pred_dn = scaler_y.inverse_transform(q_pred_seq.reshape(-1, y.shape[1])).reshape(
                    len(X_test), lookahead, y.shape[1]
                )
                q_pred_step = q_pred_dn[:, 0, :]

                q_metrics = {
                    "electric_power": _metrics(true_step[:, elec_idx], q_pred_step[:, elec_idx], "GRUQuantizedInference"),
                    "gas_consumption": _metrics(true_step[:, gas_idx], q_pred_step[:, gas_idx], "GRUQuantizedInference"),
                }
                q_true_temp = None
                q_pred_temp = None
                if room_idx is not None:
                    q_true_temp = true_step[:, room_idx]
                    q_pred_temp = q_pred_step[:, room_idx]
                    q_metrics["room_temp"] = _metrics(q_true_temp, q_pred_temp, "GRUQuantizedInference")

                quantized_result = ModelResult(
                    name="GRUQuantizedInference",
                    timestamps=test_ts,
                    true_elec=true_step[:, elec_idx],
                    true_gas=true_step[:, gas_idx],
                    pred_elec=q_pred_step[:, elec_idx],
                    pred_gas=q_pred_step[:, gas_idx],
                    true_temp=q_true_temp,
                    pred_temp=q_pred_temp,
                    history={},
                    metrics=q_metrics,
                    train_runtime_s=gru_result.train_runtime_s,
                    test_runtime_s=q_test_s,
                )
            except Exception as exc:
                logger.warning("GRU quantized inference benchmark failed: %s", exc)

    return gru_result, quantized_result


# ============================================================================
# LR TRAINING
# ============================================================================

def _load_raw_df(data_path: Path) -> pd.DataFrame:
    """Load normalised raw sensor DataFrame (no feature engineering)."""
    raw = pd.read_csv(data_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index("timestamp").drop(
        columns=["sensor.current_electricity_market_price"], errors="ignore"
    )
    return normalise_sensors(raw)


def _previous_observed_series(
    raw_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    column: str,
    *,
    default: float,
) -> pd.Series:
    """Return the latest known value strictly before each target timestamp.

    This is used for online one-step evaluation. It avoids feeding the target
    row's measured room temperature back into a same-timestamp temperature
    prediction, while still allowing the model to use the known thermal state
    from the previous observation.
    """
    idx = pd.DatetimeIndex(target_index)
    if len(idx) == 0 or column not in raw_df.columns:
        return pd.Series(default, index=idx, dtype=float)

    source = pd.to_numeric(raw_df[column], errors="coerce").dropna().sort_index()
    if source.empty:
        return pd.Series(default, index=idx, dtype=float)

    left = pd.DataFrame({"timestamp": idx, "__order": np.arange(len(idx), dtype=int)})
    right = source.rename("__value").reset_index()
    right.columns = ["timestamp", "__value"]

    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("__order")

    values = pd.to_numeric(merged["__value"], errors="coerce").fillna(default).to_numpy(dtype=float)
    return pd.Series(values, index=idx, dtype=float)


def _train_lr(
    *,
    data_path: Path,
    input_window: int,
    target_cols: list[str],
    feature_level: str,
    bivalent_point: float,
    ridge_alpha: float,
    gas_ridge_alpha: float,
    gas_binary_C: float,
    seed: int,
    # Shared split info (from LSTM run so splits are identical)
    seq_start_ts: np.ndarray,
    n_train: int,
    n_val: int,
) -> ModelResult:
    """Train HybridHeatPumpLR on train split, evaluate on test split."""
    t0 = time.perf_counter()
    np.random.seed(seed)

    raw_df = _load_raw_df(data_path)

    # The LSTM uses seq_start_ts[i] = aligned_index[input_window + i]
    # → timestamp at the start of each prediction horizon.
    # We use the same timestamp positions to slice raw_df rows so that
    # the LR and LSTM evaluate on exactly the same timesteps.
    n_seq = len(seq_start_ts)
    train_ts = seq_start_ts[:n_train]
    val_ts   = seq_start_ts[n_train : n_train + n_val]
    test_ts  = seq_start_ts[n_train + n_val :]

    def _slice(ts: np.ndarray) -> pd.DataFrame:
        """Return raw_df rows whose index is in ts (matching timestamps)."""
        ts_idx = pd.DatetimeIndex(ts)
        return raw_df.reindex(ts_idx).dropna(how="all")

    df_train = _slice(train_ts)
    df_val   = _slice(val_ts)
    df_test  = _slice(test_ts)

    elec_col = "electric_power"
    gas_col  = "gas_consumption"

    # Fill missing (safety: should not occur with clean data)
    for df_part in [df_train, df_val, df_test]:
        for col in [elec_col, gas_col]:
            if col not in df_part.columns:
                df_part[col] = 0.0

    lr_model = HybridHeatPumpLR(
        bivalent_point=bivalent_point,
        ridge_alpha=ridge_alpha,
        gas_ridge_alpha=gas_ridge_alpha,
        gas_binary_C=gas_binary_C,
    )
    lr_model.fit(df_train, df_train[elec_col].to_numpy(), df_train[gas_col].to_numpy())
    t_trained = time.perf_counter()

    pred_elec_test, pred_gas_test = lr_model.predict(df_test)
    true_elec_test = df_test[elec_col].to_numpy()
    true_gas_test  = df_test[gas_col].to_numpy()

    # Align timestamps: reindex test predictions to actual df_test index
    actual_test_ts = df_test.index.to_numpy()

    return ModelResult(
        name="HybridLR",
        timestamps=actual_test_ts,
        true_elec=true_elec_test,
        true_gas=true_gas_test,
        pred_elec=pred_elec_test,
        pred_gas=pred_gas_test,
        true_temp=None,
        pred_temp=None,
        history={},
        metrics={
            "electric_power": _metrics(true_elec_test, pred_elec_test, "HybridLR"),
            "gas_consumption": _metrics(true_gas_test, pred_gas_test, "HybridLR"),
        },
        train_runtime_s=float(t_trained - t0),
        test_runtime_s=float(time.perf_counter() - t_trained),
    )


def _build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude direct targets to avoid leakage (especially room_temp -> room_temp).
    excluded = {"electric_power", "gas_consumption", "room_temp"}
    cols = [
        c for c in df.columns
        if (c not in excluded) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError("No numeric baseline feature columns available")
    feat = df[cols].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill().bfill().fillna(0.0)
    return feat


def _train_sklearn_baseline(
    *,
    model_name: str,
    data_path: Path,
    seq_start_ts: np.ndarray,
    n_train: int,
    n_val: int,
    seed: int,
) -> ModelResult:
    t0 = time.perf_counter()
    raw_df = _load_raw_df(data_path)

    train_ts = seq_start_ts[:n_train]
    test_ts = seq_start_ts[n_train + n_val :]

    def _slice(ts: np.ndarray) -> pd.DataFrame:
        ts_idx = pd.DatetimeIndex(ts)
        return raw_df.reindex(ts_idx).dropna(how="all")

    df_train = _slice(train_ts)
    df_test = _slice(test_ts)
    X_train = _build_baseline_features(df_train).to_numpy(dtype=float)
    X_test = _build_baseline_features(df_test).to_numpy(dtype=float)

    y_e_train = df_train["electric_power"].fillna(0.0).to_numpy(dtype=float)
    y_g_train = df_train["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    y_t_train = df_train["room_temp"].fillna(20.0).to_numpy(dtype=float)
    y_e_test = df_test["electric_power"].fillna(0.0).to_numpy(dtype=float)
    y_g_test = df_test["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    y_t_test = df_test["room_temp"].fillna(20.0).to_numpy(dtype=float)

    if model_name == "KNN":
        elec_model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=15, weights="distance")),
        ])
        gas_model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=15, weights="distance")),
        ])
        temp_model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=15, weights="distance")),
        ])
    elif model_name == "RF":
        elec_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        gas_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed + 1,
            n_jobs=-1,
        )
        temp_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed + 2,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    elec_model.fit(X_train, y_e_train)
    gas_model.fit(X_train, y_g_train)
    temp_model.fit(X_train, y_t_train)
    t_trained = time.perf_counter()
    pred_elec = np.clip(elec_model.predict(X_test), a_min=0.0, a_max=None)
    pred_gas = np.clip(gas_model.predict(X_test), a_min=0.0, a_max=None)
    pred_temp = temp_model.predict(X_test)

    return ModelResult(
        name=model_name,
        timestamps=df_test.index.to_numpy(),
        true_elec=y_e_test,
        true_gas=y_g_test,
        pred_elec=pred_elec,
        pred_gas=pred_gas,
        true_temp=y_t_test,
        pred_temp=pred_temp,
        history={},
        metrics={
            "electric_power": _metrics(y_e_test, pred_elec, model_name),
            "gas_consumption": _metrics(y_g_test, pred_gas, model_name),
            "room_temp": _metrics(y_t_test, pred_temp, model_name),
        },
        train_runtime_s=float(t_trained - t0),
        test_runtime_s=float(time.perf_counter() - t_trained),
    )


def _physics_features(
    df: pd.DataFrame,
    *,
    room_state: pd.Series | None = None,
) -> np.ndarray:
    """Build non-leaky physical features for the online RLS model.

    ``room_state`` must be the latest observed room temperature before each
    prediction timestamp. Falling back to an in-frame shift keeps direct calls
    from using the current target row's room temperature.
    """
    duty = df.get("heatpump_duty", pd.Series(0.0, index=df.index)).fillna(0.0).to_numpy(dtype=float)
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

    if "sun_altitude_sin" in df.columns and "sun_altitude_cos" in df.columns:
        sun_alt_sin_s = df["sun_altitude_sin"].fillna(0.0)
        sun_alt_cos_s = df["sun_altitude_cos"].fillna(0.0)
    elif "sun_altitude" in df.columns:
        alt_rad = np.radians(df["sun_altitude"].fillna(0.0).to_numpy(dtype=float))
        sun_alt_sin_s = pd.Series(np.sin(alt_rad), index=df.index)
        sun_alt_cos_s = pd.Series(np.cos(alt_rad), index=df.index)
    else:
        sun_alt_sin_s = pd.Series(0.0, index=df.index)
        sun_alt_cos_s = pd.Series(0.0, index=df.index)

    if "sun_azimuth_sin" in df.columns and "sun_azimuth_cos" in df.columns:
        sun_az_sin_s = df["sun_azimuth_sin"].fillna(0.0)
        sun_az_cos_s = df["sun_azimuth_cos"].fillna(0.0)
    elif "sun_azimuth" in df.columns:
        az_rad = np.radians(df["sun_azimuth"].fillna(0.0).to_numpy(dtype=float))
        sun_az_sin_s = pd.Series(np.sin(az_rad), index=df.index)
        sun_az_cos_s = pd.Series(np.cos(az_rad), index=df.index)
    else:
        sun_az_sin_s = pd.Series(0.0, index=df.index)
        sun_az_cos_s = pd.Series(0.0, index=df.index)

    if "wind_bearing_sin" in df.columns and "wind_bearing_cos" in df.columns:
        wind_sin_s = df["wind_bearing_sin"].fillna(0.0)
        wind_cos_s = df["wind_bearing_cos"].fillna(0.0)
    elif "wind_bearing" in df.columns:
        wb_rad = np.radians(df["wind_bearing"].fillna(0.0).to_numpy(dtype=float))
        wind_sin_s = pd.Series(np.sin(wb_rad), index=df.index)
        wind_cos_s = pd.Series(np.cos(wb_rad), index=df.index)
    else:
        wind_sin_s = pd.Series(0.0, index=df.index)
        wind_cos_s = pd.Series(0.0, index=df.index)

    room = room_s.to_numpy(dtype=float)
    outdoor = outdoor_s.to_numpy(dtype=float)
    supply = supply_s.to_numpy(dtype=float)
    wind_speed = wind_speed_s.to_numpy(dtype=float)
    dni = dni_s.to_numpy(dtype=float)
    dhi = dhi_s.to_numpy(dtype=float)
    sun_alt_sin = sun_alt_sin_s.to_numpy(dtype=float)
    sun_alt_cos = sun_alt_cos_s.to_numpy(dtype=float)
    sun_az_sin = sun_az_sin_s.to_numpy(dtype=float)
    sun_az_cos = sun_az_cos_s.to_numpy(dtype=float)
    wind_sin = wind_sin_s.to_numpy(dtype=float)
    wind_cos = wind_cos_s.to_numpy(dtype=float)

    delta_supply = np.clip(supply - room, a_min=0.0, a_max=None)
    delta_env = np.clip(room - outdoor, a_min=0.0, a_max=None)
    cold = (outdoor < 2.0).astype(float)

    wind_outdoor = wind_speed * outdoor
    wind2_outdoor = (wind_speed ** 2) * outdoor

    sun_pos = np.sqrt(sun_alt_sin**2 + sun_alt_cos**2 + sun_az_sin**2 + sun_az_cos**2)
    sun_pos_dni = sun_pos * dni

    wind_sin_outdoor = wind_sin * outdoor
    wind_cos_outdoor = wind_cos * outdoor
    wind_sin_dni = wind_sin * dni
    wind_cos_dni = wind_cos * dni
    wind_sin_dhi = wind_sin * dhi
    wind_cos_dhi = wind_cos * dhi

    sun_alt_sin_outdoor = sun_alt_sin * outdoor
    sun_alt_cos_outdoor = sun_alt_cos * outdoor
    sun_az_sin_outdoor = sun_az_sin * outdoor
    sun_az_cos_outdoor = sun_az_cos * outdoor
    sun_alt_sin_dni = sun_alt_sin * dni
    sun_alt_cos_dni = sun_alt_cos * dni
    sun_az_sin_dni = sun_az_sin * dni
    sun_az_cos_dni = sun_az_cos * dni
    sun_alt_sin_dhi = sun_alt_sin * dhi
    sun_alt_cos_dhi = sun_alt_cos * dhi
    sun_az_sin_dhi = sun_az_sin * dhi
    sun_az_cos_dhi = sun_az_cos * dhi

    return np.column_stack([
        np.ones(len(df), dtype=float),
        room,
        duty,
        delta_supply,
        duty * delta_supply,
        delta_env,
        duty * delta_env,
        cold,
        duty * cold,
        wind_speed,
        wind_outdoor,
        wind2_outdoor,
        dni,
        dhi,
        sun_pos_dni,
        wind_sin,
        wind_cos,
        sun_alt_sin,
        sun_alt_cos,
        sun_az_sin,
        sun_az_cos,
        wind_sin_outdoor,
        wind_cos_outdoor,
        wind_sin_dni,
        wind_cos_dni,
        wind_sin_dhi,
        wind_cos_dhi,
        sun_alt_sin_outdoor,
        sun_alt_cos_outdoor,
        sun_az_sin_outdoor,
        sun_az_cos_outdoor,
        sun_alt_sin_dni,
        sun_alt_cos_dni,
        sun_az_sin_dni,
        sun_az_cos_dni,
        sun_alt_sin_dhi,
        sun_alt_cos_dhi,
        sun_az_sin_dhi,
        sun_az_cos_dhi,
    ])


def _rls_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    forgetting: float,
    ridge: float,
) -> tuple[np.ndarray, float, float]:
    n_feat = X_train.shape[1]
    theta = np.zeros(n_feat, dtype=float)
    p_mat = (1.0 / max(1e-9, ridge)) * np.eye(n_feat, dtype=float)

    def _update(x_row: np.ndarray, y_val: float) -> None:
        nonlocal theta, p_mat
        x = x_row.reshape(-1, 1)
        denom = float(forgetting + (x.T @ p_mat @ x)[0, 0])
        gain = (p_mat @ x) / max(1e-9, denom)
        err = float(y_val - (theta @ x_row))
        theta = theta + (gain[:, 0] * err)
        p_mat = (p_mat - (gain @ x.T @ p_mat)) / max(1e-9, forgetting)

    t_fit = time.perf_counter()
    for xr, yr in zip(X_train, y_train, strict=False):
        _update(xr, float(yr))
    t_infer = time.perf_counter()

    preds = np.zeros(len(X_test), dtype=float)
    for i_row, (xr, yr) in enumerate(zip(X_test, y_test, strict=False)):
        preds[i_row] = max(0.0, float(theta @ xr))
        _update(xr, float(yr))
    return preds, float(t_infer - t_fit), float(time.perf_counter() - t_infer)


def _rls_fit_theta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    forgetting: float,
    ridge: float,
) -> tuple[np.ndarray, float]:
    """Fit an RLS model on training data and return fixed coefficients."""
    n_feat = X_train.shape[1]
    theta = np.zeros(n_feat, dtype=float)
    p_mat = (1.0 / max(1e-9, ridge)) * np.eye(n_feat, dtype=float)

    def _update(x_row: np.ndarray, y_val: float) -> None:
        nonlocal theta, p_mat
        x = x_row.reshape(-1, 1)
        denom = float(forgetting + (x.T @ p_mat @ x)[0, 0])
        gain = (p_mat @ x) / max(1e-9, denom)
        err = float(y_val - (theta @ x_row))
        theta = theta + (gain[:, 0] * err)
        p_mat = (p_mat - (gain @ x.T @ p_mat)) / max(1e-9, forgetting)

    t0 = time.perf_counter()
    for xr, yr in zip(X_train, y_train, strict=False):
        _update(xr, float(yr))

    return theta, float(time.perf_counter() - t0)


def _recursive_physics_predict(
    df_test: pd.DataFrame,
    *,
    theta_elec: np.ndarray,
    theta_gas: np.ndarray,
    theta_temp: np.ndarray,
    initial_room_state: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Open-loop physical prediction using previous predicted room temperature.

    The first row starts from the latest observed room temperature before the
    test horizon. Every following row uses the previous predicted room temp,
    which avoids teacher forcing from measured test temperatures.
    """
    t0 = time.perf_counter()
    pred_elec = np.zeros(len(df_test), dtype=float)
    pred_gas = np.zeros(len(df_test), dtype=float)
    pred_temp = np.zeros(len(df_test), dtype=float)

    room_state = float(initial_room_state)
    for i, (ts, row) in enumerate(df_test.iterrows()):
        row_df = row.to_frame().T
        row_df.index = pd.DatetimeIndex([ts])
        state = pd.Series([room_state], index=row_df.index, dtype=float)
        x_row = _physics_features(row_df, room_state=state)[0]

        pred_elec[i] = max(0.0, float(theta_elec @ x_row))
        pred_gas[i] = max(0.0, float(theta_gas @ x_row))
        pred_temp[i] = float(theta_temp @ x_row)
        room_state = pred_temp[i]

    return pred_elec, pred_gas, pred_temp, float(time.perf_counter() - t0)


def _train_self_learning_physics(
    *,
    data_path: Path,
    seq_start_ts: np.ndarray,
    n_train: int,
    n_val: int,
    forgetting_factor: float,
    ridge: float,
    gas_cutoff_temp: float,
) -> ModelResult:
    raw_df = _load_raw_df(data_path)
    train_ts = seq_start_ts[:n_train]
    test_ts = seq_start_ts[n_train + n_val :]

    def _slice(ts: np.ndarray) -> pd.DataFrame:
        ts_idx = pd.DatetimeIndex(ts)
        return raw_df.reindex(ts_idx).dropna(how="all")

    df_train = _slice(train_ts)
    df_test = _slice(test_ts)

    train_idx = pd.DatetimeIndex(df_train.index)
    test_idx = pd.DatetimeIndex(df_test.index)
    room_train_state = _previous_observed_series(raw_df, train_idx, "room_temp", default=20.0)
    room_test_state = _previous_observed_series(raw_df, test_idx, "room_temp", default=20.0)

    x_train = _physics_features(df_train, room_state=room_train_state)
    y_e_train = df_train["electric_power"].fillna(0.0).to_numpy(dtype=float)
    y_g_train = df_train["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    y_t_train = df_train["room_temp"].fillna(20.0).to_numpy(dtype=float)
    y_e_test = df_test["electric_power"].fillna(0.0).to_numpy(dtype=float)
    y_g_test = df_test["gas_consumption"].fillna(0.0).to_numpy(dtype=float)
    y_t_test = df_test["room_temp"].fillna(20.0).to_numpy(dtype=float)

    theta_elec, _e_train_s = _rls_fit_theta(
        x_train,
        y_e_train,
        forgetting=forgetting_factor,
        ridge=ridge,
    )
    theta_gas, _g_train_s = _rls_fit_theta(
        x_train,
        y_g_train,
        forgetting=forgetting_factor,
        ridge=ridge,
    )
    theta_temp, _t_train_s = _rls_fit_theta(
        x_train,
        y_t_train,
        forgetting=forgetting_factor,
        ridge=ridge,
    )

    initial_room_state = float(room_test_state.iloc[0]) if len(room_test_state) else 20.0
    pred_elec, pred_gas, pred_temp, _test_s = _recursive_physics_predict(
        df_test,
        theta_elec=theta_elec,
        theta_gas=theta_gas,
        theta_temp=theta_temp,
        initial_room_state=initial_room_state,
    )
    outdoor_test = df_test.get("outdoor_temp", pd.Series(10.0, index=df_test.index)).fillna(10.0).to_numpy(dtype=float)
    pred_gas[outdoor_test > gas_cutoff_temp] = 0.0

    return ModelResult(
        name="SelfLearningPhysics",
        timestamps=df_test.index.to_numpy(),
        true_elec=y_e_test,
        true_gas=y_g_test,
        pred_elec=pred_elec,
        pred_gas=pred_gas,
        true_temp=y_t_test,
        pred_temp=pred_temp,
        history={},
        metrics={
            "electric_power": _metrics(y_e_test, pred_elec, "SelfLearningPhysics"),
            "gas_consumption": _metrics(y_g_test, pred_gas, "SelfLearningPhysics"),
            "room_temp": _metrics(y_t_test, pred_temp, "SelfLearningPhysics"),
        },
        train_runtime_s=_e_train_s + _g_train_s + _t_train_s,
        test_runtime_s=_test_s,
    )


# ============================================================================
# ENSEMBLE
# ============================================================================

def _make_pair_ensemble(
    first: ModelResult,
    second: ModelResult,
    *,
    name: str,
) -> ModelResult:
    """Simple 50/50 ensemble for two already-aligned model predictions."""
    ens_elec = 0.5 * first.pred_elec + 0.5 * second.pred_elec
    ens_gas  = 0.5 * first.pred_gas  + 0.5 * second.pred_gas
    ens_temp = None
    true_temp = first.true_temp
    if (first.pred_temp is not None) and (second.pred_temp is not None):
        ens_temp = 0.5 * first.pred_temp + 0.5 * second.pred_temp

    metrics = {
        "electric_power": _metrics(first.true_elec, ens_elec, name),
        "gas_consumption": _metrics(first.true_gas, ens_gas, name),
    }
    if (true_temp is not None) and (ens_temp is not None):
        metrics["room_temp"] = _metrics(true_temp, ens_temp, name)

    return ModelResult(
        name=name,
        timestamps=first.timestamps,
        true_elec=first.true_elec,
        true_gas=first.true_gas,
        pred_elec=ens_elec,
        pred_gas=ens_gas,
        true_temp=true_temp,
        pred_temp=ens_temp,
        history={},
        metrics=metrics,
        train_runtime_s=float(np.nansum([first.train_runtime_s, second.train_runtime_s])),
        test_runtime_s=float(np.nansum([first.test_runtime_s, second.test_runtime_s])),
    )


def _online_residual_blend(
    base_pred: np.ndarray,
    aux_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    forgetting: float,
    ridge: float,
) -> np.ndarray:
    """Online residual correction on top of base model.

    At each step t, predicts:
      y_hat_t = base_t + [1, (aux_t-base_t)] @ theta_{t-1}
    and then updates theta with the observed residual (y_t-base_t).
    """
    x2 = aux_pred - base_pred
    x_mat = np.column_stack([np.ones(len(x2), dtype=float), x2.astype(float)])

    theta = np.zeros(2, dtype=float)
    p_mat = (1.0 / max(1e-9, ridge)) * np.eye(2, dtype=float)
    preds = np.zeros(len(base_pred), dtype=float)

    for i, x_row in enumerate(x_mat):
        preds[i] = float(base_pred[i] + np.dot(theta, x_row))

        x = x_row.reshape(-1, 1)
        denom = float(forgetting + (x.T @ p_mat @ x)[0, 0])
        gain = (p_mat @ x) / max(1e-9, denom)
        residual_true = float(y_true[i] - base_pred[i])
        residual_pred = float(np.dot(theta, x_row))
        err = residual_true - residual_pred
        theta = theta + gain[:, 0] * err
        p_mat = (p_mat - (gain @ x.T @ p_mat)) / max(1e-9, forgetting)

    return preds


def _make_phys_residual_ensemble(
    phys: ModelResult,
    aux: ModelResult,
    *,
    name: str,
    forgetting: float,
    ridge: float,
) -> ModelResult:
    """Residual-trained ensemble using physical model as baseline."""
    pred_elec = _online_residual_blend(
        base_pred=phys.pred_elec,
        aux_pred=aux.pred_elec,
        y_true=phys.true_elec,
        forgetting=forgetting,
        ridge=ridge,
    )
    pred_gas = _online_residual_blend(
        base_pred=phys.pred_gas,
        aux_pred=aux.pred_gas,
        y_true=phys.true_gas,
        forgetting=forgetting,
        ridge=ridge,
    )

    true_temp = phys.true_temp
    pred_temp = None
    if (phys.pred_temp is not None) and (aux.pred_temp is not None) and (true_temp is not None):
        pred_temp = _online_residual_blend(
            base_pred=phys.pred_temp,
            aux_pred=aux.pred_temp,
            y_true=true_temp,
            forgetting=forgetting,
            ridge=ridge,
        )

    metrics = {
        "electric_power": _metrics(phys.true_elec, pred_elec, name),
        "gas_consumption": _metrics(phys.true_gas, pred_gas, name),
    }
    if (true_temp is not None) and (pred_temp is not None):
        metrics["room_temp"] = _metrics(true_temp, pred_temp, name)

    return ModelResult(
        name=name,
        timestamps=phys.timestamps,
        true_elec=phys.true_elec,
        true_gas=phys.true_gas,
        pred_elec=pred_elec,
        pred_gas=pred_gas,
        true_temp=true_temp,
        pred_temp=pred_temp,
        history={},
        metrics=metrics,
        train_runtime_s=float(np.nansum([phys.train_runtime_s, aux.train_runtime_s])),
        test_runtime_s=float(np.nansum([phys.test_runtime_s, aux.test_runtime_s])),
    )


def _estimate_standby_power(
    raw_df: pd.DataFrame,
    train_timestamps: np.ndarray,
    duty_threshold: float,
) -> float:
    """Estimate standby electric power from low-duty training samples."""
    train_idx = pd.DatetimeIndex(train_timestamps)
    df_train = raw_df.reindex(train_idx).dropna(how="all")
    if "electric_power" not in df_train.columns:
        return 0.0

    duty = df_train["heatpump_duty"].fillna(0.0) if "heatpump_duty" in df_train.columns else pd.Series(0.0, index=df_train.index)
    elec = df_train["electric_power"].fillna(0.0)
    mask = duty <= duty_threshold
    if int(mask.sum()) >= 5:
        return float(np.median(np.clip(elec[mask].to_numpy(dtype=float), a_min=0.0, a_max=None)))
    return float(np.percentile(np.clip(elec.to_numpy(dtype=float), a_min=0.0, a_max=None), 5))


def _apply_standby_rule_to_ensemble(
    ensemble: ModelResult,
    raw_df: pd.DataFrame,
    standby_power_w: float,
    duty_threshold: float,
) -> ModelResult:
    """Apply hard standby logic on top of ensemble predictions.

    Rule: if heatpump_duty <= threshold, electric = standby_power and gas = 0.
    """
    idx = pd.DatetimeIndex(ensemble.timestamps)
    df_eval = raw_df.reindex(idx)
    duty = df_eval["heatpump_duty"].fillna(0.0).to_numpy(dtype=float) if "heatpump_duty" in df_eval.columns else np.zeros(len(idx), dtype=float)
    hp_off = duty <= duty_threshold

    pred_elec = ensemble.pred_elec.copy()
    pred_gas = ensemble.pred_gas.copy()
    pred_elec[hp_off] = standby_power_w
    pred_gas[hp_off] = 0.0

    return ModelResult(
        name="Ensemble + standby",
        timestamps=ensemble.timestamps,
        true_elec=ensemble.true_elec,
        true_gas=ensemble.true_gas,
        pred_elec=pred_elec,
        pred_gas=pred_gas,
        true_temp=ensemble.true_temp,
        pred_temp=ensemble.pred_temp,
        history={},
        metrics={
            "electric_power": _metrics(ensemble.true_elec, pred_elec, "Ensemble+standby"),
            "gas_consumption": _metrics(ensemble.true_gas, pred_gas, "Ensemble+standby"),
            **({"room_temp": _metrics(ensemble.true_temp, ensemble.pred_temp, "Ensemble+standby")}
               if (ensemble.true_temp is not None and ensemble.pred_temp is not None) else {}),
        },
        train_runtime_s=ensemble.train_runtime_s,
        test_runtime_s=ensemble.test_runtime_s,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def _build_plot(
    *,
    lstm_history: dict[str, list[float]],
    models: list[ModelResult],
    raw_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Training loss — LSTM",
            "Room temperature: actual vs models  [°C]",
            "Electric power: actual vs models  [W]",
            "Gas consumption: actual vs models  [m³/interval]",
            "Irradiance: DNI & DHI",
            "Outdoor vs supply temperature  [°C]",
        ),
    )

    # Row 1 — LSTM training history
    epochs = np.arange(1, len(lstm_history.get("train_loss", [])) + 1)
    if len(epochs):
        fig.add_trace(go.Scatter(x=epochs, y=lstm_history["train_loss"],
                                  mode="lines", name="LSTM train loss",
                                  line=dict(color="royalblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=lstm_history["val_loss"],
                                  mode="lines", name="LSTM val loss",
                                  line=dict(color="royalblue", dash="dash")), row=1, col=1)

    # Row 2 — Room temperature
    ref = models[0]
    if ref.true_temp is not None:
        x_act_t, y_act_t = _downsample(ref.timestamps, ref.true_temp)
        fig.add_trace(go.Scatter(x=x_act_t, y=y_act_t, mode="lines", name="Room temp actual",
                                  line=dict(color="gray", width=1)), row=2, col=1)

    x_act, y_act = _downsample(ref.timestamps, ref.true_elec)
    fig.add_trace(go.Scatter(x=x_act, y=y_act, mode="lines", name="Electric actual",
                              line=dict(color="gray", width=1)), row=3, col=1)
    style_map = {
        "LSTM": dict(color="royalblue"),
        "GRU": dict(color="deepskyblue", dash="dash"),
        "HybridLR": dict(color="darkorange"),
        "KNN": dict(color="firebrick", dash="dot"),
        "RF": dict(color="teal", dash="dash"),
        "SelfLearningPhysics": dict(color="black", dash="dashdot"),
        "Ensemble LSTM+HybridLR": dict(color="seagreen", dash="dot"),
        "Ensemble Phys+RF": dict(color="olive", dash="dot"),
        "Ensemble Phys+LSTM": dict(color="mediumvioletred", dash="dashdot"),
        "Ensemble Phys+GRU": dict(color="sienna", dash="dashdot"),
        "Ensemble + standby": dict(color="purple", dash="dashdot"),
    }
    for model in models:
        if model.pred_temp is not None:
            xs_t, ys_t = _downsample(model.timestamps, model.pred_temp)
            fig.add_trace(
                go.Scatter(
                    x=xs_t,
                    y=ys_t,
                    mode="lines",
                    name=f"{model.name} temp",
                    line=style_map.get(model.name, dict(color="slategray")),
                ),
                row=2,
                col=1,
            )

    # Row 3 — Electric power
    for model in models:
        xs, ys = _downsample(model.timestamps, model.pred_elec)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=model.name,
                line=style_map.get(model.name, dict(color="slategray")),
            ),
            row=3,
            col=1,
        )

    # Row 4 — Gas consumption
    x_act_g, y_act_g = _downsample(ref.timestamps, ref.true_gas)
    fig.add_trace(go.Scatter(x=x_act_g, y=y_act_g, mode="lines", name="Gas actual",
                              line=dict(color="gray", width=1)), row=4, col=1)
    for model in models:
        xs, ys = _downsample(model.timestamps, model.pred_gas)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{model.name} gas",
                line=style_map.get(model.name, dict(color="slategray")),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # Row 5/6 — Weather and thermal context from raw dataframe
    if raw_df is not None:
        idx = pd.DatetimeIndex(ref.timestamps)
        df_plot = raw_df.reindex(idx)

        if "dni" in df_plot.columns:
            x_dni, y_dni = _downsample(df_plot.index.to_numpy(), df_plot["dni"].fillna(0.0).to_numpy(dtype=float))
            fig.add_trace(
                go.Scatter(x=x_dni, y=y_dni, mode="lines", name="DNI", line=dict(color="goldenrod")),
                row=5,
                col=1,
            )
        if "dhi" in df_plot.columns:
            x_dhi, y_dhi = _downsample(df_plot.index.to_numpy(), df_plot["dhi"].fillna(0.0).to_numpy(dtype=float))
            fig.add_trace(
                go.Scatter(x=x_dhi, y=y_dhi, mode="lines", name="DHI", line=dict(color="darkgoldenrod", dash="dash")),
                row=5,
                col=1,
            )

        if "outdoor_temp" in df_plot.columns:
            x_out, y_out = _downsample(df_plot.index.to_numpy(), df_plot["outdoor_temp"].ffill().fillna(0.0).to_numpy(dtype=float))
            fig.add_trace(
                go.Scatter(x=x_out, y=y_out, mode="lines", name="Outdoor temp", line=dict(color="steelblue")),
                row=6,
                col=1,
            )
        if "supply_temp" in df_plot.columns:
            x_sup, y_sup = _downsample(df_plot.index.to_numpy(), df_plot["supply_temp"].ffill().fillna(0.0).to_numpy(dtype=float))
            fig.add_trace(
                go.Scatter(x=x_sup, y=y_sup, mode="lines", name="Supply temp", line=dict(color="tomato")),
                row=6,
                col=1,
            )

    fig.update_layout(
        title="Hybrid Heat Pump: LSTM vs LR vs Ensemble",
        template="plotly_white",
        height=2050,
        width=1500,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="W", row=3, col=1)
    fig.update_yaxes(title_text="m³", row=4, col=1)
    fig.update_yaxes(title_text="Irradiance", row=5, col=1)
    fig.update_yaxes(title_text="°C", row=6, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Timestamp", row=4, col=1)
    fig.update_xaxes(title_text="Timestamp", row=5, col=1)
    fig.update_xaxes(title_text="Timestamp", row=6, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved plot: %s", output_path)


def _build_coef_plot(lr_model: HybridHeatPumpLR, output_path: Path) -> None:
    """Horizontal bar chart of standardised coefficients for interpretability."""
    try:
        coef_df = lr_model.coef_summary()
    except Exception as exc:
        logger.warning("Could not build coef plot: %s", exc)
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Electric model — standardised coefficients",
                        "Gas magnitude model — standardised coefficients"),
    )
    for col_idx, model_name in enumerate(["electric", "gas_magnitude"], start=1):
        sub = coef_df[coef_df["model"] == model_name].sort_values("coef_standardised")
        if sub.empty:
            continue
        colors = ["crimson" if v < 0 else "steelblue" for v in sub["coef_standardised"]]
        fig.add_trace(
            go.Bar(
                x=sub["coef_standardised"],
                y=sub["feature"],
                orientation="h",
                marker_color=colors,
                name=model_name,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title="HybridHeatPumpLR — physics coefficient interpretation",
        template="plotly_white",
        height=600,
        width=1400,
        showlegend=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved coefficient plot: %s", output_path)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LSTM, HybridLR, KNN, RF, self-learning physics and ensemble models"
    )
    parser.add_argument("--data-path", default="tests_thermal/data/test_data.csv")
    parser.add_argument("--report-dir", default="tests_thermal/reports/ensemble_compare")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--feature-level", default="standard", choices=["minimal", "standard", "full"])
    parser.add_argument("--target-cols", default="room_temp,electric_power,gas_consumption")
    parser.add_argument("--physics-loss-weight", type=float, default=0.10)
    parser.add_argument("--physics-balance-weight", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bivalent-point", type=float, default=2.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gas-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--gas-binary-c", type=float, default=1.0)
    parser.add_argument("--standby-duty-threshold", type=float, default=0.02)
    parser.add_argument("--standby-power-w", type=float, default=None)
    parser.add_argument("--physics-forgetting-factor", type=float, default=0.995)
    parser.add_argument("--physics-ridge", type=float, default=1.0)
    parser.add_argument("--selflearn-gas-cutoff-temp", type=float, default=5.0)
    parser.add_argument("--phys-ensemble-mode", default="avg", choices=["avg", "residual"])
    parser.add_argument("--phys-ensemble-forgetting", type=float, default=0.995)
    parser.add_argument("--phys-ensemble-ridge", type=float, default=5.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--quantized-gru-inference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path  = Path(args.data_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_cols = args.target_cols.split(",")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "cuda":
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if device.type == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(int(args.cpu_threads))
        logger.info("Configured torch CPU threads: %d", torch.get_num_threads())

    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Train LSTM (one-step)
    # ------------------------------------------------------------------
    logger.info("=== Training LSTM (lookahead=%d, epochs=%d) ===", args.lookahead, args.epochs)
    opts_for_split = SearchOptions(
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
    X_raw, y_raw, _, _, _ = _prepare_features(data_path, opts=opts_for_split)
    aligned_index = _build_aligned_index(data_path, opts_for_split)
    X_seq, y_seq = create_sequences(X_raw, y_raw, lookback=args.input_window, lookahead=args.lookahead)
    n_seq = len(X_seq)
    seq_start_ts = aligned_index[args.input_window : args.input_window + n_seq].to_numpy()
    X_tr, _, X_va, _, X_te, _ = split_sequences(X_seq, y_seq)
    n_train, n_val = len(X_tr), len(X_va)

    lstm_result = _train_lstm(
        data_path=data_path,
        input_window=args.input_window,
        lookahead=args.lookahead,
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

    logger.info("=== Training CPU-friendly GRU ===")
    gru_hidden_size = max(32, args.hidden_size // 2)
    gru_result, gru_quant_result = _train_gru(
        data_path=data_path,
        input_window=args.input_window,
        lookahead=args.lookahead,
        hidden_size=gru_hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        feature_level=args.feature_level,
        target_cols=target_cols,
        seed=args.seed,
        device=device,
        enable_quantized_inference=bool(args.quantized_gru_inference),
    )

    # ------------------------------------------------------------------
    # 2. Train HybridHeatPumpLR on same split boundaries
    # ------------------------------------------------------------------
    logger.info("=== Training HybridHeatPumpLR ===")
    lr_result = _train_lr(
        data_path=data_path,
        input_window=args.input_window,
        target_cols=target_cols,
        feature_level=args.feature_level,
        bivalent_point=args.bivalent_point,
        ridge_alpha=args.ridge_alpha,
        gas_ridge_alpha=args.gas_ridge_alpha,
        gas_binary_C=args.gas_binary_c,
        seed=args.seed,
        seq_start_ts=seq_start_ts,
        n_train=n_train,
        n_val=n_val,
    )

    logger.info("=== Training KNN baseline ===")
    knn_result = _train_sklearn_baseline(
        model_name="KNN",
        data_path=data_path,
        seq_start_ts=seq_start_ts,
        n_train=n_train,
        n_val=n_val,
        seed=args.seed,
    )

    logger.info("=== Training RF baseline ===")
    rf_result = _train_sklearn_baseline(
        model_name="RF",
        data_path=data_path,
        seq_start_ts=seq_start_ts,
        n_train=n_train,
        n_val=n_val,
        seed=args.seed,
    )

    logger.info("=== Training self-learning physical model ===")
    phys_result = _train_self_learning_physics(
        data_path=data_path,
        seq_start_ts=seq_start_ts,
        n_train=n_train,
        n_val=n_val,
        forgetting_factor=float(args.physics_forgetting_factor),
        ridge=float(args.physics_ridge),
        gas_cutoff_temp=float(args.selflearn_gas_cutoff_temp),
    )

    # ------------------------------------------------------------------
    # 3. Build ensemble (align: LR only predicts where raw data exists)
    # ------------------------------------------------------------------
    # Both models may have slightly different timestamp arrays if some
    # raw rows had NaN targets.  Re-align to LSTM timestamps.
    common_ts = pd.DatetimeIndex(lstm_result.timestamps)
    model_results_to_align = [gru_result, lr_result, knn_result, rf_result, phys_result]
    if gru_quant_result is not None:
        model_results_to_align.append(gru_quant_result)

    for model_result in model_results_to_align:
        common_ts = common_ts.intersection(pd.DatetimeIndex(model_result.timestamps))

    def _reindex(result: ModelResult, ts: pd.DatetimeIndex) -> ModelResult:
        src_ts = pd.DatetimeIndex(result.timestamps)
        mask = src_ts.isin(ts)
        return ModelResult(
            name=result.name,
            timestamps=result.timestamps[mask],
            true_elec=result.true_elec[mask],
            true_gas=result.true_gas[mask],
            pred_elec=result.pred_elec[mask],
            pred_gas=result.pred_gas[mask],
            true_temp=(result.true_temp[mask] if result.true_temp is not None else None),
            pred_temp=(result.pred_temp[mask] if result.pred_temp is not None else None),
            history=result.history,
            metrics=result.metrics,
            train_runtime_s=result.train_runtime_s,
            test_runtime_s=result.test_runtime_s,
        )

    lstm_aligned = _reindex(lstm_result, common_ts)
    gru_aligned = _reindex(gru_result, common_ts)
    gru_quant_aligned = _reindex(gru_quant_result, common_ts) if gru_quant_result is not None else None
    lr_aligned = _reindex(lr_result, common_ts)
    knn_aligned = _reindex(knn_result, common_ts)
    rf_aligned = _reindex(rf_result, common_ts)
    phys_aligned = _reindex(phys_result, common_ts)
    ensemble_lstm_lr = _make_pair_ensemble(lstm_aligned, lr_aligned, name="Ensemble LSTM+HybridLR")
    if args.phys_ensemble_mode == "residual":
        logger.info(
            "Using residual-trained Phys ensembles (forgetting=%.4f, ridge=%.4f)",
            args.phys_ensemble_forgetting,
            args.phys_ensemble_ridge,
        )
        ensemble_phys_rf = _make_phys_residual_ensemble(
            phys_aligned,
            rf_aligned,
            name="Ensemble Phys+RF",
            forgetting=float(args.phys_ensemble_forgetting),
            ridge=float(args.phys_ensemble_ridge),
        )
        ensemble_phys_lstm = _make_phys_residual_ensemble(
            phys_aligned,
            lstm_aligned,
            name="Ensemble Phys+LSTM",
            forgetting=float(args.phys_ensemble_forgetting),
            ridge=float(args.phys_ensemble_ridge),
        )
        ensemble_phys_gru = _make_phys_residual_ensemble(
            phys_aligned,
            gru_aligned,
            name="Ensemble Phys+GRU",
            forgetting=float(args.phys_ensemble_forgetting),
            ridge=float(args.phys_ensemble_ridge),
        )
    else:
        ensemble_phys_rf = _make_pair_ensemble(phys_aligned, rf_aligned, name="Ensemble Phys+RF")
        ensemble_phys_lstm = _make_pair_ensemble(phys_aligned, lstm_aligned, name="Ensemble Phys+LSTM")
        ensemble_phys_gru = _make_pair_ensemble(phys_aligned, gru_aligned, name="Ensemble Phys+GRU")

    raw_df_full = _load_raw_df(data_path)
    train_ts_idx = pd.DatetimeIndex(seq_start_ts[:n_train])
    inferred_standby_power = _estimate_standby_power(
        raw_df=raw_df_full,
        train_timestamps=train_ts_idx.to_numpy(),
        duty_threshold=args.standby_duty_threshold,
    )
    standby_power_w = inferred_standby_power if args.standby_power_w is None else float(max(args.standby_power_w, 0.0))
    logger.info(
        "Applying ensemble standby rule: duty<=%.3f -> elec=%.2fW and gas=0",
        args.standby_duty_threshold,
        standby_power_w,
    )
    ensemble_standby = _apply_standby_rule_to_ensemble(
        ensemble=ensemble_lstm_lr,
        raw_df=raw_df_full,
        standby_power_w=standby_power_w,
        duty_threshold=args.standby_duty_threshold,
    )

    # ------------------------------------------------------------------
    # 4. Persist metrics
    # ------------------------------------------------------------------
    all_metrics = {
        "LSTM": lstm_result.metrics,
        "GRU": gru_result.metrics,
        "HybridLR": lr_result.metrics,
        "KNN": knn_result.metrics,
        "RF": rf_result.metrics,
        "SelfLearningPhysics": phys_result.metrics,
        "EnsembleLSTMHybridLR": ensemble_lstm_lr.metrics,
        "EnsemblePhysRF": ensemble_phys_rf.metrics,
        "EnsemblePhysLSTM": ensemble_phys_lstm.metrics,
        "EnsemblePhysGRU": ensemble_phys_gru.metrics,
        "EnsembleStandby": ensemble_standby.metrics,
    }
    if gru_quant_aligned is not None:
        all_metrics["GRUQuantizedInference"] = gru_quant_aligned.metrics

    model_train_runtime_s = {
        "LSTM": lstm_result.train_runtime_s,
        "GRU": gru_result.train_runtime_s,
        "HybridLR": lr_result.train_runtime_s,
        "KNN": knn_result.train_runtime_s,
        "RF": rf_result.train_runtime_s,
        "SelfLearningPhysics": phys_result.train_runtime_s,
        "EnsembleLSTMHybridLR": ensemble_lstm_lr.train_runtime_s,
        "EnsemblePhysRF": ensemble_phys_rf.train_runtime_s,
        "EnsemblePhysLSTM": ensemble_phys_lstm.train_runtime_s,
        "EnsemblePhysGRU": ensemble_phys_gru.train_runtime_s,
        "EnsembleStandby": ensemble_standby.train_runtime_s,
    }
    if gru_quant_aligned is not None:
        model_train_runtime_s["GRUQuantizedInference"] = gru_quant_aligned.train_runtime_s

    model_test_runtime_s = {
        "LSTM": lstm_result.test_runtime_s,
        "GRU": gru_result.test_runtime_s,
        "HybridLR": lr_result.test_runtime_s,
        "KNN": knn_result.test_runtime_s,
        "RF": rf_result.test_runtime_s,
        "SelfLearningPhysics": phys_result.test_runtime_s,
        "EnsembleLSTMHybridLR": ensemble_lstm_lr.test_runtime_s,
        "EnsemblePhysRF": ensemble_phys_rf.test_runtime_s,
        "EnsemblePhysLSTM": ensemble_phys_lstm.test_runtime_s,
        "EnsemblePhysGRU": ensemble_phys_gru.test_runtime_s,
        "EnsembleStandby": ensemble_standby.test_runtime_s,
    }
    if gru_quant_aligned is not None:
        model_test_runtime_s["GRUQuantizedInference"] = gru_quant_aligned.test_runtime_s
    metrics_path = report_dir / "ensemble_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Saved metrics: %s", metrics_path)

    # CSV summary
    rows = []
    for model_name, m in all_metrics.items():
        for target, vals in m.items():
            rows.append({"model": model_name, "target": target,
                         "rmse": vals["rmse"], "mae": vals["mae"],
                         "train_runtime_s": model_train_runtime_s.get(model_name, float("nan")),
                         "test_runtime_s": model_test_runtime_s.get(model_name, float("nan"))})
    pd.DataFrame(rows).to_csv(report_dir / "ensemble_metrics.csv", index=False)

    # Coefficient summary for LR interpretability
    try:
        # Re-train LR model just to get the fitted object for coef extraction
        df_train_coef = raw_df_full.reindex(train_ts_idx).dropna(how="all")
        lr_model_obj = HybridHeatPumpLR(
            bivalent_point=args.bivalent_point,
            ridge_alpha=args.ridge_alpha,
            gas_ridge_alpha=args.gas_ridge_alpha,
            gas_binary_C=args.gas_binary_c,
        )
        lr_model_obj.fit(
            df_train_coef,
            df_train_coef["electric_power"].to_numpy(),
            df_train_coef["gas_consumption"].to_numpy(),
        )
        coef_df = lr_model_obj.coef_summary()
        coef_df.to_csv(report_dir / "lr_coefficients.csv", index=False)
        logger.info("Saved LR coefficients: %s", report_dir / "lr_coefficients.csv")
        _build_coef_plot(lr_model_obj, report_dir / "lr_coefficients_plot.html")
    except Exception as exc:
        logger.warning("Could not save LR coefficients: %s", exc)
        lr_model_obj = None

    # ------------------------------------------------------------------
    # 5. Main comparison plot
    # ------------------------------------------------------------------
    _build_plot(
        lstm_history=lstm_result.history,
        models=[
            lstm_aligned,
            gru_aligned,
            lr_aligned,
            knn_aligned,
            rf_aligned,
            phys_aligned,
            ensemble_lstm_lr,
            ensemble_phys_rf,
            ensemble_phys_lstm,
            ensemble_phys_gru,
            ensemble_standby,
            *( [gru_quant_aligned] if gru_quant_aligned is not None else [] ),
        ],
        raw_df=raw_df_full,
        output_path=report_dir / "ensemble_comparison_plot.html",
    )

    # Print summary table
    print("\n=== Test set metrics ===")
    print(f"{'Model':<20} {'Target':<20} {'RMSE':>10} {'MAE':>10}")
    print("-" * 64)
    for row in rows:
        print(f"{row['model']:<20} {row['target']:<20} {row['rmse']:>10.4f} {row['mae']:>10.4f}")


if __name__ == "__main__":
    main()
