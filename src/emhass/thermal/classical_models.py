"""
Classical ML models for thermal forecasting
=============================================
Trains and evaluates a suite of scikit-learn regressors against the
building thermal data, each with its appropriate feature level.

Feature levels per model
------------------------
  minimal  (KNN, SVR)            — core physics + time + solar, no lags
  standard (ElasticNet, MLP, …)  — + lags(1,2,4,8) + rolling(1h,4h)
  full     (RF, ExtraTrees, GB)  — + deep lags + rolling(24h) + interactions

Entry points
------------
  train_all_models(df, ...)       — train every model, return ModelRegistry
  ModelRegistry.best()            — model with lowest RMSE on test set
  ModelRegistry.predict(df, model_name)  — generate temperature forecast
  ModelRegistry.save(path) / .load(path)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .feature_engineering import (
    MODEL_FEATURE_LEVEL,
    build_feature_matrix,
    recommended_feature_level,
    time_based_split,
)

logger = logging.getLogger(__name__)

AUTOREGRESSIVE_TARGET_COLS = ["room_temp", "electric_power", "gas_consumption"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: All available regressors with their default hyperparameters.
#: These are deliberately conservative; grid-search tunes them further.
_BASE_MODELS: dict[str, object] = {
    "ElasticNet":                ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
    "Ridge":                     Ridge(alpha=1.0),
    "Lasso":                     Lasso(alpha=0.05, max_iter=2000),
    "KNeighborsRegressor":       KNeighborsRegressor(n_neighbors=10, weights="distance"),
    "SVR":                       SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "RandomForestRegressor":     RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "ExtraTreesRegressor":       ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "AdaBoostRegressor":         AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
    "MLPRegressor":              MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}


@dataclass
class TrainResult:
    model_name: str
    feature_level: str
    feature_cols: list[str]
    pipeline: Pipeline
    rmse_train: float
    rmse_val: float
    rmse_test: float
    mae_test: float
    runtime_s: float
    n_train: int
    n_val: int
    n_test: int
    corr_test: float = np.nan
    diff_corr_test: float = np.nan
    std_ratio_test: float = np.nan
    turn_acc_test: float = np.nan
    gas_event_f1_test: float = np.nan
    gas_event_recall_test: float = np.nan
    selection_score: float = np.nan
    status: str = "ok"


@dataclass
class ModelRegistry:
    """Container for all trained models and their metrics."""

    results: dict[str, TrainResult] = field(default_factory=dict)
    selection_metric: str = "rmse_test"

    def add(self, result: TrainResult) -> None:
        self.results[result.model_name] = result

    def best(self, metric: str | None = None) -> TrainResult | None:
        ok = [r for r in self.results.values() if r.status == "ok"]
        if not ok:
            return None
        metric_name = metric or self.selection_metric
        filtered = [r for r in ok if np.isfinite(_safe_result_attr(r, metric_name, np.nan))]
        if filtered:
            return min(filtered, key=lambda r: _safe_result_attr(r, metric_name, np.inf))
        return min(ok, key=lambda r: _safe_result_attr(r, "rmse_test", np.inf))

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.results.values():
            rows.append(
                {
                    "model": r.model_name,
                    "level": r.feature_level,
                    "n_features": len(r.feature_cols),
                    "rmse_train": round(r.rmse_train, 4),
                    "rmse_val": round(r.rmse_val, 4),
                    "rmse_test": round(r.rmse_test, 4),
                    "mae_test": round(r.mae_test, 4),
                    "corr_test": round(_safe_result_attr(r, "corr_test", np.nan), 4),
                    "diff_corr_test": round(_safe_result_attr(r, "diff_corr_test", np.nan), 4),
                    "std_ratio_test": round(_safe_result_attr(r, "std_ratio_test", np.nan), 4),
                    "turn_acc_test": round(_safe_result_attr(r, "turn_acc_test", np.nan), 4),
                    "gas_event_f1_test": round(_safe_result_attr(r, "gas_event_f1_test", np.nan), 4),
                    "gas_event_recall_test": round(_safe_result_attr(r, "gas_event_recall_test", np.nan), 4),
                    "selection_score": round(_safe_result_attr(r, "selection_score", np.nan), 4),
                    "runtime_s": round(r.runtime_s, 2),
                    "status": r.status,
                }
            )
        df = pd.DataFrame(rows)
        sort_col = "selection_score" if "selection_score" in df.columns and df["selection_score"].notna().any() else "rmse_test"
        df = df.sort_values(sort_col)
        return df

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str | None = None,
        latitude: float = 52.1202,
        longitude: float = 4.4899,
        facade_azimuth_deg: float | None = None,
        target_col: str = "room_temp",
    ) -> pd.Series:
        """Generate a temperature forecast for *df* using *model_name* (or best model)."""
        if model_name is None:
            best = self.best()
            if best is None:
                raise RuntimeError("No trained models available")
            model_name = best.model_name

        result = self.results.get(model_name)
        if result is None or result.status != "ok":
            raise KeyError(f"Model '{model_name}' not available or failed to train")

        feature_df, _ = build_feature_matrix(
            df,
            feature_level=result.feature_level,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            drop_na=False,
        )

        # Keep only columns seen during training (in order)
        avail = [c for c in result.feature_cols if c in feature_df.columns]
        missing = set(result.feature_cols) - set(avail)
        if missing:
            logger.warning("predict: %d feature columns missing from input: %s", len(missing), missing)
            for col in missing:
                feature_df[col] = 0.0

        X = feature_df[result.feature_cols].fillna(0.0).values
        y_pred = result.pipeline.predict(X)
        return pd.Series(y_pred, index=feature_df.index, name=f"{target_col}_pred_{model_name}")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, result in self.results.items():
            model_file = path / f"{name}.pkl"
            with model_file.open("wb") as fh:
                pickle.dump(result, fh)
        # Write summary JSON for quick inspection
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                "feature_level": result.feature_level,
                "rmse_test": result.rmse_test,
                "mae_test":  result.mae_test,
                "corr_test": _safe_result_attr(result, "corr_test", np.nan),
                "diff_corr_test": _safe_result_attr(result, "diff_corr_test", np.nan),
                "std_ratio_test": _safe_result_attr(result, "std_ratio_test", np.nan),
                "turn_acc_test": _safe_result_attr(result, "turn_acc_test", np.nan),
                "gas_event_f1_test": _safe_result_attr(result, "gas_event_f1_test", np.nan),
                "gas_event_recall_test": _safe_result_attr(result, "gas_event_recall_test", np.nan),
                "selection_score": _safe_result_attr(result, "selection_score", np.nan),
                "n_features": len(result.feature_cols),
                "status": result.status,
            }
        (path / "registry_summary.json").write_text(
            json.dumps({"selection_metric": self.selection_metric, "models": summary}, indent=2)
        )
        logger.info("ModelRegistry saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        path = Path(path)
        selection_metric = "rmse_test"
        summary_file = path / "registry_summary.json"
        if summary_file.exists():
            try:
                summary_payload = json.loads(summary_file.read_text())
                if isinstance(summary_payload, dict):
                    selection_metric = str(summary_payload.get("selection_metric", selection_metric))
            except Exception:
                selection_metric = "rmse_test"

        registry = cls(selection_metric=selection_metric)
        for pkl_file in sorted(path.glob("*.pkl")):
            with pkl_file.open("rb") as fh:
                result: TrainResult = pickle.load(fh)
            registry.add(result)
        logger.info("ModelRegistry loaded: %d models from %s", len(registry.results), path)
        return registry


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _safe_result_attr(result: TrainResult, name: str, default: float) -> float:
    value = getattr(result, name, default)
    return default if value is None else value


def _corr_or_nan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _pattern_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_col: str) -> dict[str, float]:
    corr = _corr_or_nan(y_true, y_pred)
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    std_ratio = np.nan if std_true == 0 else float(std_pred / std_true)
    diff_corr = _corr_or_nan(np.diff(y_true), np.diff(y_pred)) if len(y_true) > 2 else np.nan

    turn_acc = np.nan
    if len(y_true) > 3:
        turning_true = np.sign(np.diff(y_true)[1:]) != np.sign(np.diff(y_true)[:-1])
        turning_pred = np.sign(np.diff(y_pred)[1:]) != np.sign(np.diff(y_pred)[:-1])
        if len(turning_true):
            turn_acc = float(np.mean(turning_true == turning_pred))

    gas_event_f1 = np.nan
    gas_event_recall = np.nan
    if target_col == "gas_consumption":
        actual_on = y_true > 0.01
        pred_on = y_pred > 0.01
        tp = float(np.sum(actual_on & pred_on))
        fp = float(np.sum(~actual_on & pred_on))
        fn = float(np.sum(actual_on & ~pred_on))
        denom = (2 * tp) + fp + fn
        gas_event_f1 = 0.0 if denom == 0 else float((2 * tp) / denom)
        gas_event_recall = 0.0 if np.sum(actual_on) == 0 else float(tp / np.sum(actual_on))

    return {
        "corr_test": corr,
        "diff_corr_test": diff_corr,
        "std_ratio_test": std_ratio,
        "turn_acc_test": turn_acc,
        "gas_event_f1_test": gas_event_f1,
        "gas_event_recall_test": gas_event_recall,
    }


def _selection_score(target_col: str, rmse_test: float, mae_test: float, metrics: dict[str, float]) -> float:
    corr = metrics.get("corr_test", np.nan)
    diff_corr = metrics.get("diff_corr_test", np.nan)
    std_ratio = metrics.get("std_ratio_test", np.nan)
    turn_acc = metrics.get("turn_acc_test", np.nan)
    gas_event_f1 = metrics.get("gas_event_f1_test", np.nan)
    gas_event_recall = metrics.get("gas_event_recall_test", np.nan)

    def safe_penalty(value: float, neutral: float = 0.0) -> float:
        return neutral if not np.isfinite(value) else value

    if target_col == "room_temp":
        return float(
            (0.35 * rmse_test)
            + (0.10 * mae_test)
            + (0.25 * safe_penalty(1.0 - corr, 1.0))
            + (0.15 * abs(safe_penalty(std_ratio, 0.0) - 1.0))
            + (0.10 * safe_penalty(1.0 - diff_corr, 1.0))
            + (0.05 * safe_penalty(1.0 - turn_acc, 1.0))
        )
    if target_col == "gas_consumption":
        return float(
            (0.20 * rmse_test)
            + (0.10 * mae_test)
            + (0.30 * safe_penalty(1.0 - gas_event_f1, 1.0))
            + (0.20 * safe_penalty(1.0 - gas_event_recall, 1.0))
            + (0.10 * abs(safe_penalty(std_ratio, 0.0) - 1.0))
            + (0.10 * safe_penalty(1.0 - corr, 1.0))
        )
    return float(
        (0.45 * rmse_test)
        + (0.10 * mae_test)
        + (0.25 * safe_penalty(1.0 - corr, 1.0))
        + (0.10 * abs(safe_penalty(std_ratio, 0.0) - 1.0))
        + (0.10 * safe_penalty(1.0 - diff_corr, 1.0))
    )


def selection_metric_for_target(target_col: str) -> str:
    return "selection_score" if target_col in AUTOREGRESSIVE_TARGET_COLS else "rmse_test"


def _train_single(
    model_name: str,
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    facade_azimuth_deg: float | None,
    target_col: str,
    test_frac: float,
    val_frac: float,
    feature_level_override: str | None,
    target_shift_steps: int = 0,
    exclude_feature_cols: list[str] | None = None,
) -> TrainResult:
    t0 = time.perf_counter()

    level: str = feature_level_override or recommended_feature_level(model_name)

    try:
        feature_df, feature_cols = build_feature_matrix(
            df,
            feature_level=level,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            drop_na=True,
        )
        if exclude_feature_cols:
            excluded = set(exclude_feature_cols)
            feature_cols = [c for c in feature_cols if c not in excluded]
        if target_shift_steps:
            feature_df = feature_df.copy()
            feature_df["__target_shifted__"] = feature_df[target_col].shift(-target_shift_steps)
            feature_df = feature_df.dropna(subset=["__target_shifted__"])
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", model_name, exc)
        return TrainResult(
            model_name=model_name,
            feature_level=level,
            feature_cols=[],
            pipeline=Pipeline([("scaler", StandardScaler())]),
            rmse_train=np.nan,
            rmse_val=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            runtime_s=time.perf_counter() - t0,
            n_train=0, n_val=0, n_test=0,
            status=f"feature_error: {exc}",
        )

    train_df, val_df, test_df = time_based_split(feature_df, test_frac=test_frac, val_frac=val_frac)

    if len(train_df) < 10 or len(val_df) < 5 or len(test_df) < 5:
        return TrainResult(
            model_name=model_name,
            feature_level=level,
            feature_cols=feature_cols,
            pipeline=Pipeline([("scaler", StandardScaler())]),
            rmse_train=np.nan,
            rmse_val=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            runtime_s=time.perf_counter() - t0,
            n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
            status="too_few_rows",
        )

    X_train = train_df[feature_cols].fillna(0.0).values
    y_col = "__target_shifted__" if target_shift_steps else target_col
    y_train = train_df[y_col].values

    X_val   = val_df[feature_cols].fillna(0.0).values
    y_val   = val_df[y_col].values

    X_test  = test_df[feature_cols].fillna(0.0).values
    y_test  = test_df[y_col].values

    base_model = _BASE_MODELS.get(model_name)
    if base_model is None:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  clone(base_model)),
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        logger.error("Training failed for %s: %s", model_name, exc)
        return TrainResult(
            model_name=model_name,
            feature_level=level,
            feature_cols=feature_cols,
            pipeline=pipeline,
            rmse_train=np.nan,
            rmse_val=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            runtime_s=time.perf_counter() - t0,
            n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
            status=f"train_error: {exc}",
        )

    rmse_train = _rmse(y_train, pipeline.predict(X_train))
    rmse_val   = _rmse(y_val,   pipeline.predict(X_val))

    y_pred     = pipeline.predict(X_test)
    rmse_test  = _rmse(y_test, y_pred)
    mae_test   = _mae(y_test, y_pred)
    pattern_metrics = _pattern_metrics(y_test, y_pred, target_col)
    selection_score = _selection_score(target_col, rmse_test, mae_test, pattern_metrics)

    runtime = time.perf_counter() - t0
    logger.info(
        "%-30s level=%-8s  RMSE train=%.3f  val=%.3f  test=%.3f  MAE=%.3f  t=%.1fs",
        model_name, level, rmse_train, rmse_val, rmse_test, mae_test, runtime,
    )

    return TrainResult(
        model_name=model_name,
        feature_level=level,
        feature_cols=feature_cols,
        pipeline=pipeline,
        rmse_train=rmse_train,
        rmse_val=rmse_val,
        rmse_test=rmse_test,
        mae_test=mae_test,
        corr_test=pattern_metrics["corr_test"],
        diff_corr_test=pattern_metrics["diff_corr_test"],
        std_ratio_test=pattern_metrics["std_ratio_test"],
        turn_acc_test=pattern_metrics["turn_acc_test"],
        gas_event_f1_test=pattern_metrics["gas_event_f1_test"],
        gas_event_recall_test=pattern_metrics["gas_event_recall_test"],
        selection_score=selection_score,
        runtime_s=runtime,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        status="ok",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all_models(
    df: pd.DataFrame,
    *,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    models: list[str] | None = None,
    feature_level_override: str | None = None,
    target_shift_steps: int = 0,
    exclude_feature_cols: list[str] | None = None,
) -> ModelRegistry:
    """Train all (or a subset of) classical ML models on *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensor data with DatetimeIndex.
    latitude, longitude : float
        Building location for solar position.
    facade_azimuth_deg : float or None
        Facade orientation in degrees from North for facade solar gain feature.
    target_col : str
        Prediction target column name.
    test_frac, val_frac : float
        Chronological split fractions.
    models : list[str] or None
        Subset of model names to train.  None trains all of ``_BASE_MODELS``.
    feature_level_override : str or None
        Force a specific feature level for all models instead of per-model defaults.

    Returns
    -------
    ModelRegistry
        Registry with every trained model and metrics.
    """
    model_names = models if models is not None else list(_BASE_MODELS.keys())
    registry = ModelRegistry(selection_metric=selection_metric_for_target(target_col))

    logger.info("Training %d thermal ML models on %d rows", len(model_names), len(df))

    for name in model_names:
        logger.info("--- %s ---", name)
        result = _train_single(
            name,
            df,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            test_frac=test_frac,
            val_frac=val_frac,
            feature_level_override=feature_level_override,
            target_shift_steps=target_shift_steps,
            exclude_feature_cols=exclude_feature_cols,
        )
        registry.add(result)

    summary = registry.summary()
    logger.info("\n%s", summary.to_string(index=False))

    best = registry.best()
    if best:
        logger.info(
            "Best model: %s (%s=%.4f)",
            best.model_name,
            registry.selection_metric,
            _safe_result_attr(best, registry.selection_metric, best.rmse_test),
        )

    return registry


def train_single_model(
    model_name: str,
    df: pd.DataFrame,
    *,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    feature_level_override: str | None = None,
    target_shift_steps: int = 0,
    exclude_feature_cols: list[str] | None = None,
) -> TrainResult:
    """Train a single named model."""
    if model_name not in _BASE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_BASE_MODELS)}")
    return _train_single(
        model_name,
        df,
        latitude=latitude,
        longitude=longitude,
        facade_azimuth_deg=facade_azimuth_deg,
        target_col=target_col,
        test_frac=test_frac,
        val_frac=val_frac,
        feature_level_override=feature_level_override,
        target_shift_steps=target_shift_steps,
        exclude_feature_cols=exclude_feature_cols,
    )


def train_autoregressive_target_registries(
    df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    models: list[str] | None = None,
    feature_level_override: str | None = None,
) -> dict[str, ModelRegistry]:
    """Train one-step-ahead registries for multiple endogenous thermal targets."""
    available_cols = set(df.columns)
    target_cols = [c for c in (target_cols or AUTOREGRESSIVE_TARGET_COLS) if c in available_cols]
    if not target_cols:
        raise ValueError("No requested autoregressive target columns found in dataframe")

    registries: dict[str, ModelRegistry] = {}
    for target_col in target_cols:
        registries[target_col] = train_all_models(
            df,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            test_frac=test_frac,
            val_frac=val_frac,
            models=models,
            feature_level_override=feature_level_override,
            target_shift_steps=1,
            exclude_feature_cols=target_cols,
        )
    return registries


def save_target_registries(
    path: Path,
    registries: dict[str, ModelRegistry],
) -> None:
    """Persist multi-target registries under subdirectories of *path*."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for stale_pkl in path.glob("*.pkl"):
        stale_pkl.unlink()
    for stale_json in [path / "registry_summary.json", path / "multitarget_registry.json"]:
        if stale_json.exists():
            stale_json.unlink()
    metadata = {
        "format": "multi_target_autoregressive_v1",
        "targets": sorted(registries.keys()),
    }
    for target, registry in registries.items():
        target_path = path / target
        if target_path.exists():
            for child in target_path.glob("*"):
                if child.is_file():
                    child.unlink()
        registry.save(path / target)
    (path / "multitarget_registry.json").write_text(json.dumps(metadata, indent=2))


def load_target_registries(path: Path) -> dict[str, ModelRegistry]:
    """Load multi-target registries if *path* contains subdirectories per target."""
    path = Path(path)
    metadata_file = path / "multitarget_registry.json"
    targets: list[str] = []
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        targets = [str(t) for t in metadata.get("targets", [])]
    else:
        targets = [p.name for p in path.iterdir() if p.is_dir() and any(p.glob("*.pkl"))] if path.exists() else []

    registries: dict[str, ModelRegistry] = {}
    for target in targets:
        target_path = path / target
        if target_path.exists():
            registries[target] = ModelRegistry.load(target_path)
    return registries
