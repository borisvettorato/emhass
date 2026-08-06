"""Forecast hyperparameter grid search for EMHASS thermal models.

This module supports:
- Phase 1 baseline search
- Phase 2 local refinement around phase-1 best config
- Optional two-phase execution in one call
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM
from .feature_engineering import (
    build_feature_matrix,
    recommended_feature_level,
    time_based_split,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_gridsearch")


@dataclass
class RunConfig:
    input_window: int
    hidden_size: int
    num_layers: int


@dataclass
class SearchOptions:
    epochs: int = 20
    patience: int = 5
    batch_size: int = 16
    lookahead: int = 144
    learning_rate: float = 1e-3
    dropout: float = 0.2
    physics_loss_weight: float = 0.1
    physics_balance_weight: float = 0.05
    seed: int = 42
    max_runs: int = 0
    target_cols: list[str] | None = None
    # Feature engineering
    latitude: float = 52.1202
    longitude: float = 4.4899
    facade_azimuth_deg: float | None = None
    feature_level: str = "standard"  # "minimal" | "standard" | "full"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_phase1_grid() -> list[RunConfig]:
    return [
        RunConfig(iw, hs, nl)
        for iw, hs, nl in itertools.product([96, 192], [64, 128], [1, 2])
    ]


def load_phase1_best(report_dir: Path) -> RunConfig | None:
    candidates = [
        report_dir / "forecast_gridsearch_phase1_best.json",
        report_dir / "forecast_gridsearch_best.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return RunConfig(
                    input_window=int(data["input_window"]),
                    hidden_size=int(data["hidden_size"]),
                    num_layers=int(data["num_layers"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(f"Invalid best-config file ignored: {candidate}")
    return None


def build_phase2_grid_from_center(center: RunConfig) -> list[RunConfig]:
    window_step = 24
    hidden_step = 32

    grid_input_window = sorted(
        {
            max(24, center.input_window - window_step),
            center.input_window,
            center.input_window + window_step,
        }
    )
    grid_hidden_size = sorted(
        {
            max(32, center.hidden_size - hidden_step),
            center.hidden_size,
            center.hidden_size + hidden_step,
        }
    )
    grid_num_layers = sorted({max(1, center.num_layers), max(1, center.num_layers + 1)})

    return [
        RunConfig(iw, hs, nl)
        for iw, hs, nl in itertools.product(grid_input_window, grid_hidden_size, grid_num_layers)
    ]


def write_next_steps_report(report_dir: Path, phase2: bool, best: dict) -> Path:
    phase_name = "phase2" if phase2 else "phase1"
    report_path = report_dir / f"forecast_gridsearch_{phase_name}_next_steps.md"

    content = (
        f"# Forecast Gridsearch Next Steps ({phase_name})\n\n"
        "Best configuration selected with ranking: RMSE -> MAE -> runtime\n\n"
        f"- input_window: {best['input_window']}\n"
        f"- hidden_size: {best['hidden_size']}\n"
        f"- num_layers: {best['num_layers']}\n"
        f"- rmse_c: {best['rmse_c']:.6f}\n"
        f"- mae_c: {best['mae_c']:.6f}\n"
        f"- runtime_s: {best['runtime_s']:.3f}\n\n"
        "## Recommended follow-up\n"
        "1. Retrain this best config with more epochs (30-50) and early stopping.\n"
        "2. Save the trained artifact to thermal_model_solar.pt (or project checkpoint path).\n"
        "3. Re-run thermal integration tests and inspect:\n"
        "   - tests_thermal/plots/test_optimization.html\n"
        "   - tests_thermal/reports/daily_backtest_kpi.csv\n"
        "4. Proceed to shadow mode in Home Assistant before live control.\n"
    )
    report_path.write_text(content, encoding="utf-8")
    return report_path


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    max_i = len(X) - lookback - lookahead + 1
    if max_i <= 0:
        return np.empty((0, lookback, X.shape[1])), np.empty((0, lookahead, y.shape[1]))

    for i in range(max_i):
        X_seq.append(X[i : i + lookback])
        y_seq.append(y[i + lookback : i + lookback + lookahead])

    return np.array(X_seq), np.array(y_seq)


def create_physics_context_sequences(
    room_temp_signal: np.ndarray,
    outdoor_signal: np.ndarray,
    solar_signal: np.ndarray,
    lookback: int,
    lookahead: int,
) -> dict[str, np.ndarray]:
    """Build context windows aligned with create_sequences() output."""
    max_i = len(room_temp_signal) - lookback - lookahead + 1
    if max_i <= 0:
        return {
            "room_temp_prev": np.empty((0,), dtype=np.float32),
            "outdoor_temp": np.empty((0, lookahead), dtype=np.float32),
            "solar_heat": np.empty((0, lookahead), dtype=np.float32),
        }

    room_prev, outdoor_ctx, solar_ctx = [], [], []
    for i in range(max_i):
        start = i + lookback
        stop = start + lookahead
        room_prev.append(room_temp_signal[start - 1])
        outdoor_ctx.append(outdoor_signal[start:stop])
        solar_ctx.append(solar_signal[start:stop])

    return {
        "room_temp_prev": np.asarray(room_prev, dtype=np.float32),
        "outdoor_temp": np.asarray(outdoor_ctx, dtype=np.float32),
        "solar_heat": np.asarray(solar_ctx, dtype=np.float32),
    }


def split_sequences(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(X_seq)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train : n_train + n_val]
    y_val = y_seq[n_train : n_train + n_val]
    X_test = X_seq[n_train + n_val :]
    y_test = y_seq[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_single_config(
    cfg: RunConfig,
    X: np.ndarray,
    y: np.ndarray,
    scaler_y: StandardScaler,
    opts: SearchOptions,
    device: torch.device,
    physics_signals: dict[str, np.ndarray] | None = None,
) -> dict:
    start = time.perf_counter()

    X_seq, y_seq = create_sequences(X, y, lookback=cfg.input_window, lookahead=opts.lookahead)
    if len(X_seq) == 0:
        return {
            "input_window": cfg.input_window,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "status": "skipped_not_enough_data",
        }

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X_seq, y_seq)

    context_seq = None
    if physics_signals is not None:
        context_seq = create_physics_context_sequences(
            room_temp_signal=physics_signals["room_temp"],
            outdoor_signal=physics_signals["outdoor_temp"],
            solar_signal=physics_signals["solar_heat"],
            lookback=cfg.input_window,
            lookahead=opts.lookahead,
        )
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        return {
            "input_window": cfg.input_window,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "status": "skipped_split_too_small",
        }

    model = QuantilePhysicsInformedLSTM(
        input_size=X.shape[1],
        hidden=cfg.hidden_size,
        num_layers=cfg.num_layers,
        lookahead=opts.lookahead,
        targets=y.shape[1],
        dropout=0.0 if cfg.num_layers == 1 else opts.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    loss_fn = QuantileLoss(
        weight_physics=opts.physics_loss_weight,
        weight_physics_balance=opts.physics_balance_weight,
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    def _batch_context(
        context: dict[str, np.ndarray] | None,
        start_idx: int,
        end_idx: int,
    ) -> dict[str, torch.Tensor] | None:
        if context is None:
            return None
        return {
            "room_temp_prev": torch.tensor(context["room_temp_prev"][start_idx:end_idx], dtype=torch.float32, device=device),
            "outdoor_temp": torch.tensor(context["outdoor_temp"][start_idx:end_idx], dtype=torch.float32, device=device),
            "solar_heat": torch.tensor(context["solar_heat"][start_idx:end_idx], dtype=torch.float32, device=device),
        }

    for epoch in range(1, opts.epochs + 1):
        model.train()
        for i in range(0, len(X_train), opts.batch_size):
            j = min(i + opts.batch_size, len(X_train))
            xb = torch.tensor(X_train[i : i + opts.batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[i : i + opts.batch_size], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            out = model(xb)
            ctx = _batch_context(context_seq, i, j)
            loss = loss_fn(out, yb, physics_context=ctx)["total"]
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), opts.batch_size):
                j = min(i + opts.batch_size, len(X_val))
                xb = torch.tensor(X_val[i : i + opts.batch_size], dtype=torch.float32, device=device)
                yb = torch.tensor(y_val[i : i + opts.batch_size], dtype=torch.float32, device=device)
                out = model(xb)
                ctx = _batch_context(context_seq, len(X_train) + i, len(X_train) + j)
                val_losses.append(loss_fn(out, yb, physics_context=ctx)["total"].item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= opts.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for i in range(0, len(X_test), opts.batch_size):
            xb = torch.tensor(X_test[i : i + opts.batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y_test[i : i + opts.batch_size], dtype=torch.float32, device=device)
            out = model(xb)
            preds.append(out["q50"].cpu().numpy())
            targets.append(yb.cpu().numpy())

    pred_arr = np.vstack(preds).reshape(-1, y.shape[1])
    true_arr = np.vstack(targets).reshape(-1, y.shape[1])
    pred_denorm = scaler_y.inverse_transform(pred_arr)
    true_denorm = scaler_y.inverse_transform(true_arr)

    room_idx = 0
    if opts.target_cols and "room_temp" in opts.target_cols:
        room_idx = opts.target_cols.index("room_temp")

    rmse_room = float(np.sqrt(mean_squared_error(true_denorm[:, room_idx], pred_denorm[:, room_idx])))
    mae_room = float(mean_absolute_error(true_denorm[:, room_idx], pred_denorm[:, room_idx]))

    return {
        "input_window": cfg.input_window,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "status": "ok",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "rmse_c": rmse_room,
        "mae_c": mae_room,
        "runtime_s": float(time.perf_counter() - start),
        "n_train_seq": len(X_train),
        "n_val_seq": len(X_val),
        "n_test_seq": len(X_test),
    }


def _prepare_features(
    data_path: Path,
    opts: SearchOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str], dict[str, np.ndarray] | None]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing input data: {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.drop(columns=["sensor.current_electricity_market_price"], errors="ignore")

    lat  = opts.latitude            if opts else 52.1202
    lon  = opts.longitude           if opts else 4.4899
    faz  = opts.facade_azimuth_deg  if opts else None
    level = opts.feature_level      if opts else "standard"
    requested_target_cols = list(opts.target_cols) if opts and opts.target_cols else ["room_temp"]

    feature_df, feature_cols = build_feature_matrix(
        df,
        feature_level=level,
        latitude=lat,
        longitude=lon,
        facade_azimuth_deg=faz,
        target_col=requested_target_cols[0],
        exclude_feature_cols=requested_target_cols,
        drop_na=True,
    )

    target_cols = [c for c in requested_target_cols if c in feature_df.columns]
    if not target_cols:
        raise KeyError(f"None of requested target columns exist: {requested_target_cols}")

    if opts is not None:
        opts.target_cols = target_cols

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(feature_df[feature_cols].values)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(feature_df[target_cols].values)

    physics_signals = None
    required_targets = {"room_temp", "electric_power", "gas_consumption"}
    if required_targets.issubset(set(target_cols)):
        if "outdoor_temp" in feature_cols:
            outdoor_idx = feature_cols.index("outdoor_temp")
            outdoor_scaled = X[:, outdoor_idx]
        else:
            outdoor_scaled = np.zeros(len(feature_df), dtype=np.float32)

        if "solar_heat" in feature_cols:
            solar_idx = feature_cols.index("solar_heat")
            solar_scaled = X[:, solar_idx]
        else:
            solar_scaled = np.zeros(len(feature_df), dtype=np.float32)

        room_idx = target_cols.index("room_temp")
        physics_signals = {
            "room_temp": y[:, room_idx].astype(np.float32),
            "outdoor_temp": outdoor_scaled.astype(np.float32),
            "solar_heat": solar_scaled.astype(np.float32),
        }

    return X, y, scaler_y, feature_cols, physics_signals


def _save_phase_outputs(report_dir: Path, result_df: pd.DataFrame, phase_name: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    generic_csv = report_dir / "forecast_gridsearch_results.csv"
    result_df.to_csv(generic_csv, index=False)
    paths["results"] = generic_csv

    phase_csv = report_dir / f"forecast_gridsearch_{phase_name}_results.csv"
    result_df.to_csv(phase_csv, index=False)
    paths["phase_results"] = phase_csv

    ok_df = result_df[result_df["status"] == "ok"].copy()
    if not ok_df.empty:
        top5 = ok_df.head(5)
        generic_top5 = report_dir / "forecast_gridsearch_top5.csv"
        phase_top5 = report_dir / f"forecast_gridsearch_{phase_name}_top5.csv"
        top5.to_csv(generic_top5, index=False)
        top5.to_csv(phase_top5, index=False)
        paths["top5"] = generic_top5
        paths["phase_top5"] = phase_top5

        best = ok_df.iloc[0].to_dict()
        generic_best = report_dir / "forecast_gridsearch_best.json"
        phase_best = report_dir / f"forecast_gridsearch_{phase_name}_best.json"
        generic_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
        phase_best.write_text(json.dumps(best, indent=2), encoding="utf-8")
        paths["best"] = generic_best
        paths["phase_best"] = phase_best

        next_steps = write_next_steps_report(report_dir, phase2=(phase_name == "phase2"), best=best)
        paths["next_steps"] = next_steps

    return paths


def run_gridsearch(
    data_path: Path,
    report_dir: Path,
    *,
    phase2: bool,
    options: SearchOptions,
) -> dict:
    set_seed(options.seed)
    report_dir.mkdir(parents=True, exist_ok=True)

    X, y, scaler_y, feature_cols, physics_signals = _prepare_features(data_path, opts=options)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if phase2:
        center = load_phase1_best(report_dir)
        if center is None:
            raise FileNotFoundError(
                "Phase 2 requires a phase-1 best config. Run phase 1 first to generate "
                "forecast_gridsearch_phase1_best.json"
            )
        logger.info(
            "Phase 2 center from phase 1 best: "
            f"input_window={center.input_window}, hidden_size={center.hidden_size}, num_layers={center.num_layers}"
        )
        configs = build_phase2_grid_from_center(center)
        phase_name = "phase2"
    else:
        configs = build_phase1_grid()
        phase_name = "phase1"

    if options.max_runs and options.max_runs > 0:
        configs = configs[: options.max_runs]

    logger.info(f"Running {len(configs)} forecast grid-search configurations ({phase_name})")

    rows = []
    for idx, cfg in enumerate(configs, start=1):
        logger.info(
            f"[{idx}/{len(configs)}] input_window={cfg.input_window}, "
            f"hidden_size={cfg.hidden_size}, num_layers={cfg.num_layers}"
        )
        rows.append(run_single_config(cfg, X, y, scaler_y, options, device, physics_signals=physics_signals))

    result_df = pd.DataFrame(rows)
    ok_df = result_df[result_df["status"] == "ok"].copy()
    non_ok_df = result_df[result_df["status"] != "ok"].copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(by=["rmse_c", "mae_c", "runtime_s"], ascending=[True, True, True])
    result_df = pd.concat([ok_df, non_ok_df], ignore_index=True)

    saved_paths = _save_phase_outputs(report_dir, result_df, phase_name)
    logger.info(f"Saved {phase_name} report: {saved_paths.get('phase_results')}")

    return {
        "phase": phase_name,
        "rows": len(result_df),
        "ok_rows": int((result_df["status"] == "ok").sum()),
        "paths": saved_paths,
    }


def run_two_phase_gridsearch(
    data_path: Path,
    report_dir: Path,
    *,
    phase1_options: SearchOptions,
    phase2_options: SearchOptions,
) -> dict:
    phase1 = run_gridsearch(data_path, report_dir, phase2=False, options=phase1_options)
    phase2 = run_gridsearch(data_path, report_dir, phase2=True, options=phase2_options)
    return {"phase1": phase1, "phase2": phase2}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast grid search (input_window, hidden_size, num_layers)")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--report-dir", type=str, default="tests_thermal/reports")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=144)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--physics-loss-weight", type=float, default=0.1)
    parser.add_argument("--physics-balance-weight", type=float, default=0.05)
    parser.add_argument(
        "--target-cols",
        type=str,
        default="room_temp",
        help="Comma-separated targets, e.g. room_temp,electric_power,gas_consumption",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--latitude", type=float, default=52.1202)
    parser.add_argument("--longitude", type=float, default=4.4899)
    parser.add_argument("--facade-azimuth", type=float, default=None)
    parser.add_argument(
        "--feature-level",
        type=str,
        default="standard",
        choices=["minimal", "standard", "full"],
    )
    parser.add_argument("--phase2", action="store_true", help="Run only phase 2 (requires phase-1 best)")
    parser.add_argument("--run-all-phases", action="store_true", help="Run phase 1 and phase 2 in one command")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    options = SearchOptions(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lookahead=args.lookahead,
        learning_rate=args.lr,
        dropout=args.dropout,
        physics_loss_weight=args.physics_loss_weight,
        physics_balance_weight=args.physics_balance_weight,
        max_runs=args.max_runs,
        latitude=args.latitude,
        longitude=args.longitude,
        facade_azimuth_deg=args.facade_azimuth,
        feature_level=args.feature_level,
        target_cols=[c.strip() for c in args.target_cols.split(",") if c.strip()],
    )

    data_path = Path(args.data_path)
    report_dir = Path(args.report_dir)

    if args.run_all_phases:
        run_two_phase_gridsearch(
            data_path,
            report_dir,
            phase1_options=options,
            phase2_options=options,
        )
        return

    run_gridsearch(data_path, report_dir, phase2=args.phase2, options=options)

if __name__ == "__main__":
    main()
