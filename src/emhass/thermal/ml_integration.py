"""Integration helpers for using trained thermal ML models with EMHASS.

The existing pure physics thermal model is embedded directly into the CVXPY
optimization problem in optimization.py. Classical ML and PINN models are not
linear constraints, so they should be used as an external forecast layer.

Recommended flow for ML models:
1. Build room-temperature forecast from historical + forecast features.
2. Feed that forecast and outdoor temperature into HeatPumpOptimizer.
3. Use resulting setpoints / comfort diagnostics in UI, shadow mode, or a later
   control integration path.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .classical_models import AUTOREGRESSIVE_TARGET_COLS, ModelRegistry
from .pinn_optimizer import HeatPumpOptimizer

logger = logging.getLogger(__name__)


def _best_available_target_registries(
    registries: dict[str, ModelRegistry],
    requested_targets: list[str] | None = None,
) -> dict[str, ModelRegistry]:
    targets = requested_targets or AUTOREGRESSIVE_TARGET_COLS
    selected = {target: registries[target] for target in targets if target in registries}
    if "room_temp" not in selected:
        raise RuntimeError("Autoregressive thermal planning requires a room_temp registry")
    return selected


def build_autoregressive_forecast_bundle(
    df: pd.DataFrame,
    registries: dict[str, ModelRegistry],
    *,
    model_names: dict[str, str | None] | None = None,
    target_cols: list[str] | None = None,
    horizon: int = 144,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    outdoor_temp_col: str = "outdoor_temp",
) -> dict[str, Any]:
    """Roll out multi-target classical forecasts autoregressively over the horizon."""
    selected_registries = _best_available_target_registries(registries, target_cols)
    if len(df) < 2:
        raise ValueError("Need at least two rows to run autoregressive rollout")

    horizon = int(max(1, min(horizon, len(df) - 1)))
    future_index = df.index[-horizon:]
    working = df.copy()
    actuals: dict[str, pd.Series | None] = {}
    predictions: dict[str, list[float]] = {target: [] for target in selected_registries}
    chosen_models: dict[str, str] = {}

    for target, registry in selected_registries.items():
        actuals[target] = df[target].reindex(future_index) if target in df.columns else None
        chosen_models[target] = model_names.get(target) if model_names else None  # type: ignore[assignment]
        if chosen_models[target] is None:
            best = registry.best()
            if best is None:
                raise RuntimeError(f"No successful model available for target '{target}'")
            chosen_models[target] = best.model_name
        if target in working.columns:
            working.loc[future_index, target] = np.nan

    for timestamp in future_index:
        step_df = working.loc[:timestamp]
        step_predictions: dict[str, float] = {}
        for target, registry in selected_registries.items():
            pred = registry.predict(
                step_df,
                model_name=chosen_models[target],
                latitude=latitude,
                longitude=longitude,
                facade_azimuth_deg=facade_azimuth_deg,
                target_col=target,
            )
            step_predictions[target] = float(pred.loc[timestamp])
        for target, value in step_predictions.items():
            working.at[timestamp, target] = value
            predictions[target].append(value)

    outdoor_tail = None
    if outdoor_temp_col in df.columns:
        outdoor_tail = df[outdoor_temp_col].reindex(future_index).ffill().bfill()

    return {
        "model_names": chosen_models,
        "index": future_index,
        "horizon": len(future_index),
        "predictions": {
            target: pd.Series(values, index=future_index, name=f"predicted_{target}")
            for target, values in predictions.items()
        },
        "actuals": actuals,
        "outdoor_temp": outdoor_tail,
        "working_df": working,
    }


def _tail_series(values: pd.Series, horizon: int) -> pd.Series:
    if len(values) >= horizon:
        return values.iloc[-horizon:]
    return values


def _coerce_price_forecast(
    price_forecast: pd.Series | np.ndarray | list[float] | None,
    index: pd.Index,
) -> np.ndarray | None:
    if price_forecast is None:
        return None
    if isinstance(price_forecast, pd.Series):
        aligned = price_forecast.reindex(index)
        aligned = aligned.ffill().bfill()
        return aligned.to_numpy(dtype=float)
    arr = np.asarray(price_forecast, dtype=float)
    if len(arr) != len(index):
        raise ValueError(
            f"price_forecast length mismatch: expected {len(index)}, got {len(arr)}"
        )
    return arr


def build_classical_forecast_bundle(
    df: pd.DataFrame,
    registry: ModelRegistry,
    *,
    model_name: str | None = None,
    horizon: int = 144,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    outdoor_temp_col: str = "outdoor_temp",
) -> dict[str, Any]:
    """Build a forecast bundle from a trained classical thermal model."""
    y_pred = registry.predict(
        df,
        model_name=model_name,
        latitude=latitude,
        longitude=longitude,
        facade_azimuth_deg=facade_azimuth_deg,
        target_col=target_col,
    )
    pred_tail = _tail_series(y_pred, horizon)
    idx = pred_tail.index

    actual_tail = None
    if target_col in df.columns:
        actual_tail = df[target_col].reindex(idx)

    outdoor_tail = None
    if outdoor_temp_col in df.columns:
        outdoor_tail = df[outdoor_temp_col].reindex(idx).ffill().bfill()

    selected_name = model_name or (registry.best().model_name if registry.best() else None)
    if selected_name is None:
        raise RuntimeError("No successful model available in registry")

    return {
        "model_name": selected_name,
        "index": idx,
        "predicted_room_temp": pred_tail,
        "actual_room_temp": actual_tail,
        "outdoor_temp": outdoor_tail,
        "horizon": len(idx),
    }


def build_classical_optimization_plan(
    df: pd.DataFrame,
    registry: ModelRegistry | dict[str, ModelRegistry],
    *,
    model_name: str | None = None,
    horizon: int = 144,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    outdoor_temp_col: str = "outdoor_temp",
    price_forecast: pd.Series | np.ndarray | list[float] | None = None,
    target_room_temp_min: float = 19.5,
    target_room_temp_max: float = 22.0,
    price_weight: float = 0.8,
) -> dict[str, Any]:
    """Create optimizer-ready setpoints from a classical ML thermal forecast."""
    if isinstance(registry, dict):
        return build_autoregressive_classical_optimization_plan(
            df,
            registry,
            room_temp_model_name=model_name,
            horizon=horizon,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            outdoor_temp_col=outdoor_temp_col,
            price_forecast=price_forecast,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            price_weight=price_weight,
        )

    bundle = build_classical_forecast_bundle(
        df,
        registry,
        model_name=model_name,
        horizon=horizon,
        latitude=latitude,
        longitude=longitude,
        facade_azimuth_deg=facade_azimuth_deg,
        target_col=target_col,
        outdoor_temp_col=outdoor_temp_col,
    )

    if bundle["outdoor_temp"] is None:
        raise KeyError(
            f"Input dataframe must contain '{outdoor_temp_col}' to build optimization plan"
        )

    optimizer = HeatPumpOptimizer()
    room_temp_forecast = bundle["predicted_room_temp"].to_numpy(dtype=float)
    outdoor_temp_forecast = bundle["outdoor_temp"].to_numpy(dtype=float)

    neutral = optimizer.get_optimal_setpoint(
        room_temp_forecast=room_temp_forecast,
        outdoor_temp_forecast=outdoor_temp_forecast,
        target_room_temp_min=target_room_temp_min,
        target_room_temp_max=target_room_temp_max,
    )

    result: dict[str, Any] = {
        "forecast": bundle,
        "neutral": neutral,
        "price_aware": None,
    }

    prices = _coerce_price_forecast(price_forecast, bundle["index"])
    if prices is not None:
        result["price_aware"] = optimizer.get_price_aware_setpoint(
            room_temp_forecast=room_temp_forecast,
            outdoor_temp_forecast=outdoor_temp_forecast,
            price_forecast=prices,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            price_weight=price_weight,
        )

    logger.info(
        "Built classical optimization plan using model=%s horizon=%d",
        bundle["model_name"],
        bundle["horizon"],
    )
    return result


def build_autoregressive_classical_optimization_plan(
    df: pd.DataFrame,
    registries: dict[str, ModelRegistry],
    *,
    room_temp_model_name: str | None = None,
    horizon: int = 144,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    outdoor_temp_col: str = "outdoor_temp",
    price_forecast: pd.Series | np.ndarray | list[float] | None = None,
    target_room_temp_min: float = 19.5,
    target_room_temp_max: float = 22.0,
    price_weight: float = 0.8,
) -> dict[str, Any]:
    model_names = {"room_temp": room_temp_model_name} if room_temp_model_name else None
    bundle = build_autoregressive_forecast_bundle(
        df,
        registries,
        model_names=model_names,
        horizon=horizon,
        latitude=latitude,
        longitude=longitude,
        facade_azimuth_deg=facade_azimuth_deg,
        outdoor_temp_col=outdoor_temp_col,
    )

    if bundle["outdoor_temp"] is None:
        raise KeyError(
            f"Input dataframe must contain '{outdoor_temp_col}' to build optimization plan"
        )

    optimizer = HeatPumpOptimizer()
    room_temp_forecast = bundle["predictions"]["room_temp"].to_numpy(dtype=float)
    outdoor_temp_forecast = bundle["outdoor_temp"].to_numpy(dtype=float)
    neutral = optimizer.get_optimal_setpoint(
        room_temp_forecast=room_temp_forecast,
        outdoor_temp_forecast=outdoor_temp_forecast,
        target_room_temp_min=target_room_temp_min,
        target_room_temp_max=target_room_temp_max,
    )

    forecast_bundle = {
        "model_name": bundle["model_names"]["room_temp"],
        "model_names": bundle["model_names"],
        "index": bundle["index"],
        "predicted_room_temp": bundle["predictions"]["room_temp"],
        "actual_room_temp": bundle["actuals"].get("room_temp"),
        "predicted_electric_power": bundle["predictions"].get("electric_power"),
        "actual_electric_power": bundle["actuals"].get("electric_power"),
        "predicted_gas_consumption": bundle["predictions"].get("gas_consumption"),
        "actual_gas_consumption": bundle["actuals"].get("gas_consumption"),
        "outdoor_temp": bundle["outdoor_temp"],
        "horizon": bundle["horizon"],
    }

    result: dict[str, Any] = {
        "forecast": forecast_bundle,
        "neutral": neutral,
        "price_aware": None,
    }

    prices = _coerce_price_forecast(price_forecast, bundle["index"])
    if prices is not None:
        result["price_aware"] = optimizer.get_price_aware_setpoint(
            room_temp_forecast=room_temp_forecast,
            outdoor_temp_forecast=outdoor_temp_forecast,
            price_forecast=prices,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            price_weight=price_weight,
        )

    logger.info(
        "Built autoregressive classical optimization plan using room_temp model=%s horizon=%d",
        forecast_bundle["model_name"],
        forecast_bundle["horizon"],
    )
    return result


def _comfort_violation(
    room_temp_forecast: np.ndarray,
    target_room_temp_min: float,
    target_room_temp_max: float,
) -> float:
    """Compute mean comfort bound violation magnitude."""
    below = np.maximum(0.0, target_room_temp_min - room_temp_forecast)
    above = np.maximum(0.0, room_temp_forecast - target_room_temp_max)
    return float(np.mean(below + above))


def _energy_proxy(setpoints: np.ndarray, outdoor: np.ndarray) -> float:
    """Simple heat demand proxy based on supply-to-outdoor lift."""
    return float(np.mean(np.maximum(0.0, setpoints - outdoor)))


def _score_plan(
    plan: dict[str, Any],
    *,
    target_room_temp_min: float,
    target_room_temp_max: float,
    comfort_weight: float,
    energy_weight: float,
) -> float:
    forecast = plan["forecast"]
    neutral = plan["neutral"]

    room_temp = np.asarray(forecast["predicted_room_temp"], dtype=float)
    outdoor = np.asarray(forecast["outdoor_temp"], dtype=float)
    setpoints = np.asarray(neutral["setpoint_optimal"], dtype=float)

    comfort = _comfort_violation(room_temp, target_room_temp_min, target_room_temp_max)
    energy = _energy_proxy(setpoints, outdoor)
    return comfort_weight * comfort + energy_weight * energy


def build_two_stage_optimization_plan(
    df: pd.DataFrame,
    registry: ModelRegistry | dict[str, ModelRegistry],
    *,
    coarse_models: list[str] | None = None,
    fine_models: list[str] | None = None,
    top_k: int = 3,
    horizon: int = 144,
    latitude: float = 52.1202,
    longitude: float = 4.4899,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    outdoor_temp_col: str = "outdoor_temp",
    price_forecast: pd.Series | np.ndarray | list[float] | None = None,
    target_room_temp_min: float = 19.5,
    target_room_temp_max: float = 22.0,
    price_weight: float = 0.8,
    comfort_weight: float = 5.0,
    energy_weight: float = 1.0,
) -> dict[str, Any]:
    """Two-stage thermal planning: coarse model ranking then fine-model selection.

    Stage 1 (coarse): rank candidate models with a fast comfort+energy proxy score.
    Stage 2 (fine): evaluate fine models on top-k coarse candidates and return best.
    """
    room_temp_registry = registry["room_temp"] if isinstance(registry, dict) else registry
    if not room_temp_registry.results:
        raise RuntimeError("ModelRegistry has no models")

    available = [
        name for name, res in room_temp_registry.results.items() if getattr(res, "status", "") == "ok"
    ]
    if not available:
        raise RuntimeError("No successful model available in registry")

    if coarse_models is None:
        coarse_models = list(available)
    coarse_models = [m for m in coarse_models if m in available]
    if not coarse_models:
        coarse_models = list(available)

    coarse_rows: list[dict[str, Any]] = []
    for model_name in coarse_models:
        plan = build_classical_optimization_plan(
            df,
            registry,
            model_name=model_name,
            horizon=horizon,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            outdoor_temp_col=outdoor_temp_col,
            price_forecast=price_forecast,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            price_weight=price_weight,
        )
        score = _score_plan(
            plan,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            comfort_weight=comfort_weight,
            energy_weight=energy_weight,
        )
        coarse_rows.append(
            {
                "model": model_name,
                "score": score,
                "plan": plan,
            }
        )

    coarse_rows.sort(key=lambda row: row["score"])
    top_k = max(1, int(top_k))
    coarse_top = coarse_rows[: min(top_k, len(coarse_rows))]

    if fine_models is None:
        fine_candidates = [row["model"] for row in coarse_top]
    else:
        fine_models_set = {m for m in fine_models if m in available}
        fine_candidates = [row["model"] for row in coarse_top if row["model"] in fine_models_set]
        if not fine_candidates:
            fine_candidates = [row["model"] for row in coarse_top]

    fine_rows: list[dict[str, Any]] = []
    for model_name in fine_candidates:
        plan = build_classical_optimization_plan(
            df,
            registry,
            model_name=model_name,
            horizon=horizon,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=target_col,
            outdoor_temp_col=outdoor_temp_col,
            price_forecast=price_forecast,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            price_weight=price_weight,
        )
        neutral = np.asarray(plan["neutral"]["setpoint_optimal"], dtype=float)
        smoothness = float(np.mean(np.abs(np.diff(neutral)))) if len(neutral) > 1 else 0.0
        base_score = _score_plan(
            plan,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
            comfort_weight=comfort_weight,
            energy_weight=energy_weight,
        )
        final_score = base_score + 0.25 * smoothness
        fine_rows.append(
            {
                "model": model_name,
                "score": final_score,
                "base_score": base_score,
                "smoothness": smoothness,
                "plan": plan,
            }
        )

    fine_rows.sort(key=lambda row: row["score"])
    best = fine_rows[0]

    logger.info(
        "Built two-stage optimization plan: best_model=%s coarse_candidates=%d fine_candidates=%d",
        best["model"],
        len(coarse_rows),
        len(fine_rows),
    )
    return {
        "best_model": best["model"],
        "best_plan": best["plan"],
        "coarse_ranking": [
            {"model": row["model"], "score": float(row["score"])} for row in coarse_rows
        ],
        "fine_ranking": [
            {
                "model": row["model"],
                "score": float(row["score"]),
                "base_score": float(row["base_score"]),
                "smoothness": float(row["smoothness"]),
            }
            for row in fine_rows
        ],
    }
