"""
Thermal Feature Engineering
============================
Single source of truth for all thermal ML model feature construction.

Used by:
- Classical ML models  : build_feature_matrix()  → flat (n_samples, n_features)
- LSTM / PINN          : build_sequence_matrix()  → 3D  (n_samples, lookback, n_features)

Feature pipeline
----------------
  1.  Sensor column harmonisation  (normalise_sensors)
  2.  Cyclic time features          (add_cyclic_time_features)
  3.  Solar position + irradiance   (add_solar_features)
  4.  Wind vector + speed²          (add_wind_features)
  5.  Physics-derived features      (add_physics_features)
  6.  Lag features                  (add_lag_features)       ← classical ML
  7.  Rolling statistics            (add_rolling_features)   ← classical ML
  8.  Interaction features          (add_interaction_features)

Feature levels
--------------
  minimal  – core physics + time + solar, no lags
             Best for KNN, SVR (distance-metric sensitive)
  standard – + lags(1,2,4,8) + rolling(1h,4h)
             Default for ElasticNet, MLP, LSTM
  full     – + deep lags(up to 24h) + rolling(24h) + interactions
             Best for RF, ExtraTrees, GradientBoosting
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition

logger = logging.getLogger(__name__)


# ============================================================================
# SENSOR COLUMN MAPPING
# ============================================================================

# Maps canonical names → list of alternative source column names.
# First alternative that exists in the DataFrame wins.
DEFAULT_SENSOR_MAP: dict[str, list[str]] = {
    "room_temp":       ["room_temp", "sensor.indoor_temperature", "indoor_temp", "temperature_indoor"],
    "outdoor_temp":    ["outdoor_temp", "sensor.openweathermap_temperature", "outdoor_temperature", "temperature_outdoor"],
    "humidity":        ["humidity", "sensor.openweathermap_humidity", "relative_humidity"],
    "wind_speed":      ["wind_speed", "sensor.openweathermap_wind_speed",  "windspeed"],
    "wind_bearing":    ["wind_bearing", "sensor.openweathermap_wind_bearing", "wind_direction", "wind_dir"],
    "heatpump_duty":   ["heatpump_duty", "sensor.climate_control_duty", "hp_duty", "hp_on", "hp_running"],
    "supply_temp":     ["supply_temp", "sensor.leaving_water_temperature", "flow_temp", "leavingwater_temp"],
    "return_temp":     ["return_temp", "sensor.return_water_temperature", "returnwater_temp", "retour_temp"],
    "flow_rate":       ["flow_rate", "sensor.heatpump_flow_rate", "water_flow_lpm", "water_flow_m3h"],
    "electric_power":  ["electric_power", "sensor.kwh_meter_vermogen", "hp_power_w", "power_w"],
    "gas_consumption": ["gas_consumption", "gas_meter_diff", "gas_m3h"],
    "ghi":             ["ghi", "sensor.solar_ghi", "solar_ghi", "global_horizontal_irradiance"],
    "dni":             ["dni", "sensor.solar_dni", "solar_dni", "direct_normal_irradiance"],
    "dhi":             ["dhi", "sensor.solar_dhi", "solar_dhi", "diffuse_horizontal_irradiance"],
    "blind_position":  ["blind_position", "heatpump_room_blind_sensors", "blind_pct"],
    "valve_position":  ["valve_position", "heatpump_room_valve_sensors", "trv_position"],
    "window_open":     ["window_open", "heatpump_room_window_sensors",  "window_binary"],
}


# ============================================================================
# FEATURE SET DEFINITIONS
# ============================================================================

_CORE_SENSORS = [
    "room_temp", "outdoor_temp", "humidity",
    "wind_speed", "electric_power", "gas_consumption", "heatpump_duty", "supply_temp",
]

_TIME_CYCLIC = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "doy_sin",  "doy_cos",
    "minute_sin", "minute_cos",
]

_SOLAR = [
    "sun_alt_sin", "sun_alt_cos",
    "sun_az_sin",  "sun_az_cos",
    "sun_position_sin", "sun_position_cos",
    "ghi_norm", "dni_norm", "dhi_norm",
    "solar_heat",
]

_WIND_VECTOR = ["wind_bearing_sin", "wind_bearing_cos", "wind_u", "wind_w", "wind_speed_sq"]

_PHYSICS = [
    "dT", "dT_sq", "dT_dt",
    "hp_effectiveness",
    "supply_dT",
    "supply_temp_x_duty",
    "water_delta_t",
    "thermal_power_proxy",
    "thermal_power_hp_proxy",
    "thermal_power_boiler_proxy",
    "heating_demand_proxy",
]

_FACADE = ["solar_on_facade"]

_INTERACTIONS = [
    "dT_x_wind", "solar_heat_x_dT", "hp_x_outdoor", "ghi_x_dT",
]

# Weather/control-only profile for sequence models where temporality is handled
# by the LSTM window itself (no explicit time/deep lag/dT features).
LSTM_WEATHER_CONTROL_FEATURES = [
    "room_temp_lag1",
    "outdoor_temp",
    "humidity",
    "wind_speed",
    "heatpump_duty",
    "supply_temp",
    "supply_temp_x_duty",
    "water_delta_t",
    "thermal_power_proxy",
    "thermal_power_hp_proxy",
    "thermal_power_boiler_proxy",
    "heating_demand_proxy",
    "sun_alt_sin",
    "sun_alt_cos",
    "sun_az_sin",
    "sun_az_cos",
    "sun_position_sin",
    "sun_position_cos",
    "ghi_norm",
    "dni_norm",
    "dhi_norm",
    "solar_heat",
    "wind_bearing_sin",
    "wind_bearing_cos",
    "wind_u",
    "wind_w",
    "wind_speed_sq",
]

# Columns used for lag / rolling computation
_STANDARD_LAG_COLS = ["room_temp", "outdoor_temp", "electric_power", "gas_consumption", "heatpump_duty", "ghi_norm"]
_STANDARD_LAGS     = [1, 2, 4, 8]               # in timesteps

_FULL_LAG_COLS = _STANDARD_LAG_COLS + ["dT", "supply_temp"]
_FULL_LAGS     = [1, 2, 4, 8, 16, 32, 48, 96]  # up to 96 × 15 min = 24 h

_ROLLING_COLS    = ["room_temp", "outdoor_temp", "electric_power", "gas_consumption", "ghi_norm"]
_ROLLING_WINDOWS = [4, 16, 96]                   # 1 h, 4 h, 24 h at 15-min resolution


FEATURE_SETS: dict[str, dict] = {
    "minimal": {
        "base":            _CORE_SENSORS + _TIME_CYCLIC + _SOLAR + _WIND_VECTOR + _PHYSICS,
        "lags":            [],
        "lag_cols":        [],
        "rolling_windows": [],
        "rolling_cols":    [],
        "interactions":    False,
    },
    "standard": {
        "base":            _CORE_SENSORS + _TIME_CYCLIC + _SOLAR + _WIND_VECTOR + _PHYSICS,
        "lags":            _STANDARD_LAGS,
        "lag_cols":        _STANDARD_LAG_COLS,
        "rolling_windows": [4, 16],
        "rolling_cols":    _ROLLING_COLS,
        "interactions":    False,
    },
    "full": {
        "base":            _CORE_SENSORS + _TIME_CYCLIC + _SOLAR + _WIND_VECTOR + _PHYSICS + _FACADE,
        "lags":            _FULL_LAGS,
        "lag_cols":        _FULL_LAG_COLS,
        "rolling_windows": _ROLLING_WINDOWS,
        "rolling_cols":    _ROLLING_COLS,
        "interactions":    True,
    },
}

# Recommended level per model
MODEL_FEATURE_LEVEL: dict[str, Literal["minimal", "standard", "full"]] = {
    "ElasticNet":                "standard",
    "KNeighborsRegressor":       "minimal",   # compact feature set → better distances
    "SVR":                       "minimal",   # kernel trick compensates for fewer features
    "RandomForestRegressor":     "full",
    "ExtraTreesRegressor":       "full",
    "GradientBoostingRegressor": "full",
    "MLPRegressor":              "standard",
    "LSTM":                      "standard",  # temporality from sequence, fewer lags needed
    "AdaBoostRegressor":         "standard",
}


# ============================================================================
# SOLAR POSITION (vectorised)
# ============================================================================

def _solar_position_vectorized(
    timestamps: pd.DatetimeIndex,
    latitude: float = 51.6588074,
    longitude: float = 4.942094,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (altitude_deg, azimuth_deg) arrays for all timestamps.

    Uses pvlib.solarposition.get_solarposition for consistent solar geometry.
    """
    solar = get_solarposition(
        time=timestamps,
        latitude=latitude,
        longitude=longitude,
    )
    altitude_deg = solar["apparent_elevation"].to_numpy(dtype=float)
    azimuth_deg = solar["azimuth"].to_numpy(dtype=float)

    return altitude_deg, azimuth_deg


# ============================================================================
# INDIVIDUAL FEATURE BUILDERS
# ============================================================================

def normalise_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alternative sensor column names to canonical names (in-place copy)."""
    df = df.copy()
    rename_map: dict[str, str] = {}
    for canonical, alternatives in DEFAULT_SENSOR_MAP.items():
        if canonical in df.columns:
            continue
        for alt in alternatives:
            if alt in df.columns:
                rename_map[alt] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclic encodings for hour, minute, day-of-week, day-of-year."""
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("add_cyclic_time_features: index is not DatetimeIndex, skipping")
        return df

    hour_frac = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)

    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    minute = df.index.minute
    df["minute_sin"] = np.sin(2 * np.pi * minute / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60.0)

    return df


def add_solar_features(
    df: pd.DataFrame,
    latitude: float = 51.6588074,
    longitude: float = 4.942094,
    facade_azimuth_deg: float | None = None,
) -> pd.DataFrame:
    """Add solar position (sin/cos) and irradiance features.

    Parameters
    ----------
    facade_azimuth_deg : float or None
        Facade orientation in degrees from North (0=N, 90=E, 180=S, 270=W).
        If given, computes facade-projected direct solar gain as an extra feature.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("add_solar_features: index is not DatetimeIndex, skipping")
        return df

    alt_deg, az_deg = _solar_position_vectorized(df.index, latitude, longitude)

    # Keep raw degrees for debugging / downstream use; do NOT use as ML features
    df["sun_altitude"] = alt_deg
    df["sun_azimuth"]  = az_deg

    alt_rad = np.radians(alt_deg)
    az_rad  = np.radians(az_deg)

    df["sun_alt_sin"] = np.sin(alt_rad)
    df["sun_alt_cos"] = np.cos(alt_rad)
    df["sun_az_sin"]  = np.sin(az_rad)
    df["sun_az_cos"]  = np.cos(az_rad)
    # Alias for downstream pipelines expecting a generic sun_position feature.
    df["sun_position_sin"] = df["sun_az_sin"]
    df["sun_position_cos"] = df["sun_az_cos"]

    # Normalise irradiance (clip 0-1, use historical max or physical maximum)
    for col, physical_max in [("ghi", 1000.0), ("dni", 900.0), ("dhi", 150.0)]:
        col_norm = f"{col}_norm"
        if col in df.columns:
            ref = float(df[col].max())
            if ref <= 0:
                ref = physical_max
            df[col_norm] = (df[col] / ref).clip(0.0, 1.0)
        else:
            df[col_norm] = 0.0

    # Solar heat gain: GHI × sin(altitude) — effective irradiance when sun is up
    df["solar_heat"] = df["ghi_norm"] * np.maximum(df["sun_alt_sin"], 0.0)

    # Facade-projected solar gain: DNI × cos(Δazimuth) × sin(altitude)
    # Captures how much direct sun hits a specific wall/window orientation.
    if facade_azimuth_deg is not None:
        facade_rad = np.radians(facade_azimuth_deg)
        df["solar_on_facade"] = (
            df["dni_norm"]
            * np.cos(facade_rad - az_rad)
            * np.maximum(np.sin(alt_rad), 0.0)
        ).clip(0.0, None)

    return df


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind vector components (u, w) and wind speed squared.

    wind_u : East component  (speed × sin(bearing))
    wind_w : North component (speed × cos(bearing))
    These encode both direction AND magnitude in two continuous features,
    so cyclic direction discontinuity at 360°/0° is fully resolved.
    """
    speed = df["wind_speed"].fillna(0.0) if "wind_speed" in df.columns else pd.Series(0.0, index=df.index)
    df["wind_speed_sq"] = speed ** 2

    if "wind_bearing" in df.columns:
        bearing_rad = np.radians(df["wind_bearing"].fillna(0.0))
        df["wind_bearing_sin"] = np.sin(bearing_rad)
        df["wind_bearing_cos"] = np.cos(bearing_rad)
        df["wind_u"] = speed * np.sin(bearing_rad)
        df["wind_w"] = speed * np.cos(bearing_rad)
    else:
        df["wind_bearing_sin"] = 0.0
        df["wind_bearing_cos"] = 0.0
        df["wind_u"] = 0.0
        df["wind_w"] = 0.0

    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-derived thermal features.

    dT          : indoor – outdoor temperature difference [°C]
    dT_sq       : dT²   (nonlinear heat loss proxy)
    dT_dt       : rate of indoor temp change [°C/h]
    hp_effectiveness : heatpump_duty × dT   (useful heat delivered proxy)
    supply_dT   : supply_temp – outdoor_temp  (supply lift above ambient)
    supply_temp_x_duty : supply_temp × heatpump_duty (delivered heat proxy)
    water_delta_t : supply_temp - return_temp [°C]
    thermal_power_proxy : flow_rate × water_delta_t (hydronic heat proxy)
    thermal_power_hp_proxy : thermal_power_proxy × heatpump_duty
    thermal_power_boiler_proxy : thermal_power_proxy × (1 - heatpump_duty)
    heating_demand_proxy : max(dT, 0) × (1 + wind_speed/10)
    room_temp_lag1 : previous-step room temperature (state anchor for sequence models)
    """
    if "room_temp" in df.columns and "outdoor_temp" in df.columns:
        df["dT"]    = df["room_temp"] - df["outdoor_temp"]
        df["dT_sq"] = df["dT"] ** 2
        df["room_temp_lag1"] = df["room_temp"].shift(1)

    if "room_temp" in df.columns:
        # Infer timestep from index; fall back to 15-min default
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            freq_h = (df.index[1] - df.index[0]).total_seconds() / 3600.0
        else:
            freq_h = 0.25
        df["dT_dt"] = df["room_temp"].diff() / max(freq_h, 1e-6)

    if "heatpump_duty" in df.columns and "dT" in df.columns:
        df["hp_effectiveness"] = df["heatpump_duty"] * df["dT"]

    if "supply_temp" in df.columns and "outdoor_temp" in df.columns:
        df["supply_dT"] = df["supply_temp"] - df["outdoor_temp"]

    if "supply_temp" in df.columns and "heatpump_duty" in df.columns:
        df["supply_temp_x_duty"] = df["supply_temp"] * df["heatpump_duty"]

    if "supply_temp" in df.columns and "return_temp" in df.columns:
        df["water_delta_t"] = (df["supply_temp"] - df["return_temp"]).clip(lower=0.0)

    if "flow_rate" in df.columns:
        # Normalized hydronic thermal-power proxy; calibrated scaling can be learned downstream.
        flow = df["flow_rate"].astype(float)
        if "water_delta_t" in df.columns:
            df["thermal_power_proxy"] = flow * df["water_delta_t"]

    if "thermal_power_proxy" in df.columns and "heatpump_duty" in df.columns:
        duty = df["heatpump_duty"].clip(lower=0.0, upper=1.0)
        df["thermal_power_hp_proxy"] = df["thermal_power_proxy"] * duty
        df["thermal_power_boiler_proxy"] = df["thermal_power_proxy"] * (1.0 - duty)

    if "dT" in df.columns:
        demand = df["dT"].clip(lower=0.0)
        if "wind_speed" in df.columns:
            demand = demand * (1.0 + df["wind_speed"].clip(lower=0.0) / 10.0)
        df["heating_demand_proxy"] = demand

    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add shifted (lag) versions of the specified columns.

    A lag of 1 means "the value 1 timestep ago".
    Classical ML models need explicit lags because they only see one row at inference.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean, std, min, max over given windows (in timesteps).

    These capture slowly-evolving trends that a single lag cannot represent.
    E.g. a 24-step rolling mean at 15-min resolution = 6-hour moving average.
    """
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"]  = df[col].rolling(w, min_periods=2).std().fillna(0.0)
            df[f"{col}_roll{w}_min"]  = df[col].rolling(w, min_periods=1).min()
            df[f"{col}_roll{w}_max"]  = df[col].rolling(w, min_periods=1).max()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit cross-product interaction features.

    These help linear models (ElasticNet) capture non-linear physics:
    - Heat loss is proportional to both ΔT and wind (convective coefficient)
    - Solar gain interacts with the temperature driving force
    """
    if "dT" in df.columns and "wind_speed" in df.columns:
        df["dT_x_wind"] = df["dT"] * df["wind_speed"]

    if "solar_heat" in df.columns and "dT" in df.columns:
        df["solar_heat_x_dT"] = df["solar_heat"] * df["dT"]

    if "heatpump_duty" in df.columns and "outdoor_temp" in df.columns:
        df["hp_x_outdoor"] = df["heatpump_duty"] * df["outdoor_temp"]

    if "ghi_norm" in df.columns and "dT" in df.columns:
        df["ghi_x_dT"] = df["ghi_norm"] * df["dT"]

    # Blind × solar: shading reduces effective solar gain
    if "blind_position" in df.columns and "solar_heat" in df.columns:
        # blind_position in [0,1] where 1 = fully closed
        df["shaded_solar"] = df["solar_heat"] * (1.0 - df["blind_position"])

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    feature_level: Literal["minimal", "standard", "full"] = "standard",
    latitude: float = 51.6588074,
    longitude: float = 4.942094,
    facade_azimuth_deg: float | None = None,
    target_col: str = "room_temp",
    drop_source_irradiance: bool = False,
    drop_na: bool = True,
    exclude_feature_cols: list[str] | None = None,
    include_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a flat feature matrix for classical ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sensor DataFrame with DatetimeIndex.  Column names may be HA entity
        IDs — they will be normalised to canonical names automatically.
    feature_level : {"minimal", "standard", "full"}
        Complexity of the feature set. Use `recommended_feature_level(model_name)`
        to pick the right level per model.
    latitude, longitude : float
        Building location for solar position calculation.
    facade_azimuth_deg : float or None
        If given, adds `solar_on_facade` feature (degrees from North).
    target_col : str
        Name of the prediction target after normalisation. Excluded from
        feature_cols in the returned list.
    drop_source_irradiance : bool
        Drop raw ghi/dni/dhi columns after normalising to ghi_norm etc.
    drop_na : bool
        Drop rows that contain any NaN after all feature engineering steps.
        Set False if you want to handle NaNs yourself (e.g. for imputation).

    Returns
    -------
    feature_df : pd.DataFrame
        Full DataFrame with all computed features and the target column.
    feature_cols : list[str]
        Names of selected feature columns for the chosen feature level.
    """
    fset = FEATURE_SETS[feature_level]
    df = df.copy()

    # Step 1: harmonise column names
    df = normalise_sensors(df)

    # Step 2: cyclic time features
    df = add_cyclic_time_features(df)

    # Step 3: solar position and irradiance
    df = add_solar_features(df, latitude, longitude, facade_azimuth_deg)

    # Step 4: wind vector
    df = add_wind_features(df)

    # Step 5: physics-derived features
    df = add_physics_features(df)

    # Step 6: lag features (classical ML only — empty list for minimal)
    if fset["lags"]:
        df = add_lag_features(df, fset["lag_cols"], fset["lags"])

    # Step 7: rolling statistics
    if fset["rolling_windows"]:
        df = add_rolling_features(df, fset["rolling_cols"], fset["rolling_windows"])

    # Step 8: interaction features (full level only)
    if fset.get("interactions"):
        df = add_interaction_features(df)

    # Drop raw sun position degrees (sin/cos encodes them without discontinuity)
    df = df.drop(columns=["sun_altitude", "sun_azimuth"], errors="ignore")

    if drop_source_irradiance:
        df = df.drop(columns=["ghi", "dni", "dhi"], errors="ignore")

    if drop_na:
        df = df.dropna()

    if exclude_feature_cols is None:
        exclude_feature_cols = [
            "sensor.current_electricity_market_price",
            "current_electricity_market_price",
            "electricity_price",
        ]

    selected: list[str] = []

    # Base level-specific features.
    selected.extend([c for c in fset["base"] if c in df.columns and c != target_col])

    # Lag features created above for the selected lag columns/windows.
    for col in fset["lag_cols"]:
        for lag in fset["lags"]:
            lag_col = f"{col}_lag{lag}"
            if lag_col in df.columns:
                selected.append(lag_col)

    # Rolling statistics generated by add_rolling_features.
    for col in fset["rolling_cols"]:
        for w in fset["rolling_windows"]:
            for stat in ("mean", "std", "min", "max"):
                roll_col = f"{col}_roll{w}_{stat}"
                if roll_col in df.columns:
                    selected.append(roll_col)

    # Optional interaction features for the full feature level.
    if fset.get("interactions"):
        for col in _INTERACTIONS + ["shaded_solar"]:
            if col in df.columns:
                selected.append(col)

    # Deduplicate while preserving order and apply explicit exclusions.
    excluded = set(exclude_feature_cols)
    seen: set[str] = set()
    feature_cols: list[str] = []
    for col in selected:
        if col == target_col or col in excluded or col in seen:
            continue
        seen.add(col)
        feature_cols.append(col)

    # Optional explicit allow-list for model-specific compact profiles.
    if include_feature_cols is not None:
        allowed = [c for c in include_feature_cols if c in df.columns]
        missing_allowed = [c for c in include_feature_cols if c not in df.columns]
        if missing_allowed:
            logger.warning(
                "build_feature_matrix: %d include_feature_cols missing from dataframe: %s",
                len(missing_allowed),
                missing_allowed,
            )
        feature_cols = [c for c in allowed if c not in excluded and c != target_col]

    logger.info(
        "build_feature_matrix: level=%s, n_rows=%d, n_features=%d",
        feature_level, len(df), len(feature_cols),
    )
    return df, feature_cols


def build_sequence_matrix(
    df: pd.DataFrame,
    lookback: int,
    feature_cols: list[str],
    target_cols: list[str] | None = None,
    lookahead: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reshape a flat feature DataFrame into 3D sequences for LSTM.

    Calls build_feature_matrix first if you want full feature engineering;
    this function only performs the windowing step.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (already computed by build_feature_matrix).
    lookback : int
        Number of historical timesteps per sequence window.
    feature_cols : list[str]
        Features to include in X.
    target_cols : list[str] or None
        Target columns. If given, y contains the value(s) at t + lookahead.
    lookahead : int
        How many steps ahead to predict. Default 1 (next step).

    Returns
    -------
    X : np.ndarray  shape (n_samples, lookback, n_features)
    y : np.ndarray  shape (n_samples, n_targets) or None
    """
    avail = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(avail)
    if missing:
        logger.warning("build_sequence_matrix: %d requested features missing: %s", len(missing), missing)

    X_data = df[avail].values.astype(np.float32)
    n = len(X_data)

    X_seqs, y_seqs = [], []
    for i in range(lookback, n - lookahead + 1):
        X_seqs.append(X_data[i - lookback : i])
        if target_cols:
            y_seqs.append([
                float(df[c].iloc[i + lookahead - 1]) if c in df.columns else np.nan
                for c in target_cols
            ])

    if not X_seqs:
        n_feat = len(avail)
        X = np.empty((0, lookback, n_feat), dtype=np.float32)
        return X, (np.empty((0, len(target_cols or [])), dtype=np.float32) if target_cols else None)

    X = np.stack(X_seqs).astype(np.float32)
    y = np.array(y_seqs, dtype=np.float32) if target_cols else None
    return X, y


def time_based_split(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    Never shuffles — temporal leakage would otherwise make results meaningless.
    train  : oldest  (1 - val_frac - test_frac)
    val    : middle  (val_frac)
    test   : newest  (test_frac)
    """
    n      = len(df)
    test_n = int(n * test_frac)
    val_n  = int(n * val_frac)
    train_n = n - val_n - test_n

    train = df.iloc[:train_n]
    val   = df.iloc[train_n : train_n + val_n]
    test  = df.iloc[train_n + val_n :]

    logger.info(
        "time_based_split: train=%d, val=%d, test=%d rows",
        len(train), len(val), len(test),
    )
    return train, val, test


# ============================================================================
# HELPERS
# ============================================================================

def recommended_feature_level(model_name: str) -> Literal["minimal", "standard", "full"]:
    """Return the recommended feature level for the given model name."""
    return MODEL_FEATURE_LEVEL.get(model_name, "standard")


def feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Log a concise feature summary (present, missing, stats)."""
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    logger.info("Feature summary: %d present, %d missing", len(present), len(missing))
    if missing:
        logger.debug("Missing features: %s", missing)
    logger.info("Dataset: %d rows, date range %s → %s",
                len(df),
                df.index[0] if isinstance(df.index, pd.DatetimeIndex) else "?",
                df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else "?")
