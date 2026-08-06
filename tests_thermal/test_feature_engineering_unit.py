"""Unit tests for thermal feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emhass.thermal.feature_engineering import (
    build_feature_matrix,
    build_sequence_matrix,
    recommended_feature_level,
    time_based_split,
)


pytestmark = pytest.mark.unit


def _sample_df(n: int = 220) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    base = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "room_temp": 20.0 + 0.8 * np.sin(base * 10),
            "outdoor_temp": 8.0 + 3.0 * np.sin(base * 4),
            "humidity": 60.0 + 5.0 * np.cos(base * 6),
            "wind_speed": 3.0 + np.abs(np.sin(base * 8)),
            "wind_bearing": (180 + 90 * np.sin(base * 2)) % 360,
            "heatpump_duty": np.clip(0.5 + 0.4 * np.sin(base * 5), 0.0, 1.0),
            "supply_temp": 32.0 + 2.0 * np.cos(base * 7),
            "electric_power": 700.0 + 100.0 * np.sin(base * 9),
            "ghi": np.clip(500 + 400 * np.sin(base * 12), 0.0, None),
            "dni": np.clip(350 + 300 * np.sin(base * 11), 0.0, None),
            "dhi": np.clip(120 + 80 * np.sin(base * 10), 0.0, None),
            "blind_position": np.clip(0.3 + 0.3 * np.sin(base * 13), 0.0, 1.0),
        },
        index=idx,
    )


def test_build_feature_matrix_minimal_contains_core_features() -> None:
    df, features = build_feature_matrix(_sample_df(), feature_level="minimal")

    assert len(df) > 0
    assert len(features) > 0
    assert "room_temp" in df.columns
    assert "hour_sin" in features
    assert "hour_cos" in features
    assert "sun_alt_sin" in features
    assert "solar_heat" in features
    assert "wind_u" in features
    assert "dT" in features


def test_build_feature_matrix_full_contains_lags_and_interactions() -> None:
    df, features = build_feature_matrix(_sample_df(), feature_level="full")

    assert len(df) > 0
    assert any(c.startswith("room_temp_lag") for c in features)
    assert any(c.startswith("room_temp_roll") for c in features)
    assert "dT_x_wind" in features
    assert "solar_heat_x_dT" in features


def test_sequence_matrix_shape() -> None:
    df, features = build_feature_matrix(_sample_df(), feature_level="standard")
    X, y = build_sequence_matrix(df, lookback=48, feature_cols=features, target_cols=["room_temp"])

    assert X.ndim == 3
    assert X.shape[1] == 48
    assert X.shape[2] == len(features)
    assert y is not None
    assert y.ndim == 2
    assert y.shape[1] == 1


def test_time_based_split_is_chronological() -> None:
    df, _ = build_feature_matrix(_sample_df(), feature_level="minimal")
    train, val, test = time_based_split(df, test_frac=0.2, val_frac=0.1)

    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_recommended_feature_level_mapping() -> None:
    assert recommended_feature_level("KNeighborsRegressor") == "minimal"
    assert recommended_feature_level("ElasticNet") == "standard"
    assert recommended_feature_level("RandomForestRegressor") == "full"
    assert recommended_feature_level("UnknownModel") == "standard"
