"""Unit tests for ML-to-optimizer integration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emhass.thermal.classical_models import train_all_models
from emhass.thermal.ml_integration import (
    build_classical_forecast_bundle,
    build_classical_optimization_plan,
)


pytestmark = pytest.mark.unit


def _sample_df(n: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    t = np.linspace(0.0, 8.0, n)
    return pd.DataFrame(
        {
            "room_temp": 20 + 0.7 * np.sin(t),
            "outdoor_temp": 8 + 2.5 * np.cos(t / 2),
            "humidity": 60 + 3 * np.sin(t / 3),
            "wind_speed": 3 + np.abs(np.sin(t)),
            "wind_bearing": (180 + 60 * np.sin(t / 2)) % 360,
            "heatpump_duty": np.clip(0.5 + 0.3 * np.sin(t * 0.8), 0, 1),
            "supply_temp": 32 + 1.5 * np.cos(t),
            "electric_power": 650 + 90 * np.sin(t * 1.4),
            "ghi": np.clip(450 + 350 * np.sin(t * 1.3), 0, None),
            "dni": np.clip(300 + 250 * np.sin(t * 1.1), 0, None),
            "dhi": np.clip(120 + 50 * np.sin(t * 1.0), 0, None),
        },
        index=idx,
    )


def test_build_forecast_bundle_from_registry() -> None:
    df = _sample_df()
    registry = train_all_models(df, models=["ElasticNet"], test_frac=0.2, val_frac=0.1)

    bundle = build_classical_forecast_bundle(df, registry, horizon=48)

    assert bundle["model_name"] == "ElasticNet"
    assert len(bundle["predicted_room_temp"]) == 48
    assert bundle["actual_room_temp"] is not None
    assert bundle["outdoor_temp"] is not None


def test_build_optimization_plan_includes_setpoints() -> None:
    df = _sample_df()
    registry = train_all_models(df, models=["ElasticNet"], test_frac=0.2, val_frac=0.1)
    price = pd.Series(np.linspace(0.2, 0.4, len(df)), index=df.index)

    plan = build_classical_optimization_plan(df, registry, horizon=48, price_forecast=price)

    assert len(plan["neutral"]["setpoint_optimal"]) == 48
    assert plan["price_aware"] is not None
    assert len(plan["price_aware"]["setpoint_price_aware"]) == 48
