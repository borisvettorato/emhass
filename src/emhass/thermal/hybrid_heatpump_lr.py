"""
Hybrid Heat Pump — Physics-Informed Linear Regression Models
=============================================================

Models electric power consumption and gas consumption of a hybrid heat pump
system using physics-motivated features derived directly from thermodynamics.

Physics basis
-------------
A hybrid heat pump has two heat sources that can operate simultaneously or
in sequence:

  1. **Electric heat pump** (compression cycle)
     COP ≈ η · T_supply_K / (T_supply_K – T_outdoor_K)   (Carnot approximation)
     P_elec = Q_heat / COP
     → Power increases with temperature lift (supply – outdoor)
     → Power decreases with solar gain and milder weather

  2. **Gas boiler** (backup / bivalent operation)
     Fires when outdoor temp falls below the ``bivalent_point`` (°C) OR when
     the heat pump alone cannot cover peak demand.
     gas_m3 = Q_boiler_kWh / (η_boiler × gas_kWh_per_m3)

Gas model — hurdle (two-stage) approach
----------------------------------------
Gas consumption is sparse (~0 for >90 % of timesteps in mild weather).
A standard linear model cannot represent this well.  We use a hurdle model:

  Stage 1 — LogisticRegression → P(gas > 0)
  Stage 2 — Ridge              → E[gas | gas > 0]
  Output  — gas_pred = clip(P(gas > 0), 0, 1) × Ridge_pred

Physics features derived for each model
-----------------------------------------
Electric
  hp_cop_inverse       : (supply_temp – outdoor_temp) / supply_temp_K
                         ∝ 1/COP, directly scales with electric power
  hp_cop_x_demand      : hp_cop_inverse × max(0, dT)
                         Combined measure: high when cold AND COP is low
  hp_temp_lift         : heatpump_duty × (supply_temp – outdoor_temp)
                         Temperature lift while pump is active
  heat_demand_wind     : max(0, dT) × (1 + wind_speed/10)
                         Wind-enhanced convective heat loss
  solar_offset         : ghi_norm × sin(sun_altitude)
                         Solar gain that offsets heating demand
  heatpump_duty        : Duty cycle (part-load factor)
    electric_power_lag1  : previous-step electric power (autoregressive, lag-1 only)
    gas_consumption_lag1 : previous-step gas usage (autoregressive, lag-1 only)

Gas
  cold_below_bivalent  : max(0, bivalent_T – outdoor_temp)
                         Degrees below the bivalent point
  boiler_duty          : (1 – heatpump_duty)
                         Fraction of cycle not covered by heat pump
  boiler_demand        : boiler_duty × max(0, dT)
                         Heat demand assigned to the gas boiler
  peak_demand_signal   : max(0, heat_demand_wind – hp_rated_norm)
                         Excess demand beyond normalised HP capacity
    electric_power_lag1  : previous-step electric power (autoregressive, lag-1 only)
    gas_consumption_lag1 : previous-step gas usage (autoregressive, lag-1 only)

Usage
-----
  from emhass.thermal.hybrid_heatpump_lr import HybridHeatPumpLR

  model = HybridHeatPumpLR(bivalent_point=2.0)
  model.fit(df_train, y_elec_train, y_gas_train)
  elec_pred, gas_pred = model.predict(df_test)
  print(model.coef_summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Physical constants
_GAS_KWH_PER_M3: float = 10.5   # gross calorific value (Wobbe, typical NL)
_BOILER_EFF: float = 0.90        # condensing boiler efficiency
_KELVIN_OFFSET: float = 273.15


# ============================================================================
# PHYSICS FEATURE BUILDER
# ============================================================================

def build_heatpump_features(
    df: pd.DataFrame,
    bivalent_point: float = 2.0,
    hp_rated_norm: float = 3.0,          # normalised COP‑weighted capacity
    elec_feature_cols: list[str] | None = None,
    gas_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute physics-motivated features for a hybrid heat pump.

    Parameters
    ----------
    df : DataFrame
        Must contain (at minimum) canonical columns after sensor harmonisation:
        ``outdoor_temp``, ``supply_temp``, ``heatpump_duty``, ``room_temp``,
        ``wind_speed``, ``ghi_norm``, ``sun_alt_sin``.
        Missing optional columns are filled with sensible defaults.
    bivalent_point : float
        Outdoor temperature [°C] below which the gas boiler is expected
        to contribute.  Typical range: −5 °C (monovalent) to +2 °C (parallel).
    hp_rated_norm : float
        Heuristic normalised HP capacity used to compute the
        ``peak_demand_signal`` feature.  Increase if HP is oversized.
    elec_feature_cols : list[str] or None
        Override the default electric feature columns to use.
    gas_feature_cols : list[str] or None
        Override the default gas feature columns to use.

    Returns
    -------
    X_elec, X_gas : tuple[DataFrame, DataFrame]
        Feature matrices with matching index.
    """
    out = pd.DataFrame(index=df.index)

    # ---------- Helper: safely get a column or default to zeros ----------
    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(default).astype(float)
        logger.debug("build_heatpump_features: column '%s' missing, using %.3g", name, default)
        return pd.Series(default, index=df.index, dtype=float)

    outdoor = _col("outdoor_temp")
    supply  = _col("supply_temp", default=outdoor + 10.0)   # fallback: lift of 10 K
    duty    = _col("heatpump_duty").clip(0.0, 1.0)
    room    = _col("room_temp")
    wind    = _col("wind_speed").clip(lower=0.0)
    ghi     = _col("ghi_norm")
    sun_sin = _col("sun_alt_sin").clip(lower=0.0)

    # Supply temperature in Kelvin (avoid zero-division)
    supply_K = (supply + _KELVIN_OFFSET).clip(lower=1.0)

    # Temperature differences
    dT = (room - outdoor).clip(lower=0.0)               # heating demand driver [K]
    supply_lift = (supply - outdoor).clip(lower=0.5)     # temperature lift [K]

    # ----------------------------------------
    # COP-related features (electric model)
    # ----------------------------------------
    # Inverse COP proxy: the higher this is, the more electricity per kW heat
    hp_cop_inverse = (supply_lift / supply_K).clip(lower=0.0, upper=1.0)

    # Combined: high when cold AND COP is low
    hp_cop_x_demand = (hp_cop_inverse * dT).clip(lower=0.0)

    # Temperature lift while pump is active (part-load weighted)
    hp_temp_lift = (duty * supply_lift).clip(lower=0.0)

    # Wind-enhanced convective heat loss
    heat_demand_wind = (dT * (1.0 + wind / 10.0)).clip(lower=0.0)

    # Solar gain proxy
    solar_offset = (ghi * sun_sin).clip(lower=0.0)

    # Quadratic supply lift – captures nonlinear COP degradation
    supply_lift_sq = supply_lift ** 2

    # Interaction: duty × heat demand (total heat to deliver)
    hp_heat_delivered = (duty * heat_demand_wind).clip(lower=0.0)

    # ----------------------------------------
    # Gas boiler features
    # ----------------------------------------
    # Degrees below bivalent point
    cold_below = (bivalent_point - outdoor).clip(lower=0.0)

    # Boiler fraction of duty
    boiler_duty = (1.0 - duty).clip(lower=0.0, upper=1.0)

    # Heat demand assigned to boiler
    boiler_demand = (boiler_duty * dT).clip(lower=0.0)

    # Peak demand beyond rated HP capacity (proxy for needed gas backup)
    peak_demand = (heat_demand_wind - hp_rated_norm).clip(lower=0.0)

    # Cold × boiler duty – interaction term
    cold_x_boiler = (cold_below * boiler_duty).clip(lower=0.0)

    # ----------------------------------------
    # Time features (setpoint schedule proxy)
    # ----------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        hour_frac = df.index.hour + df.index.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
        out["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
        out["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    else:
        for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            out[c] = 0.0

    # ----------------------------------------
    # Collect named features
    # ----------------------------------------
    # Only lag-1 autoregressive terms are included by design.
    elec_lag1 = _col("electric_power").shift(1).ffill().fillna(0.0)
    gas_lag1 = _col("gas_consumption").shift(1).ffill().fillna(0.0)

    out["outdoor_temp"]        = outdoor
    out["supply_lift"]         = supply_lift
    out["supply_lift_sq"]      = supply_lift_sq
    out["hp_cop_inverse"]      = hp_cop_inverse
    out["hp_cop_x_demand"]     = hp_cop_x_demand
    out["hp_temp_lift"]        = hp_temp_lift
    out["heat_demand_wind"]    = heat_demand_wind
    out["hp_heat_delivered"]   = hp_heat_delivered
    out["solar_offset"]        = solar_offset
    out["heatpump_duty"]       = duty

    out["cold_below_bivalent"] = cold_below
    out["boiler_duty"]         = boiler_duty
    out["boiler_demand"]       = boiler_demand
    out["peak_demand_signal"]  = peak_demand
    out["cold_x_boiler"]       = cold_x_boiler
    out["electric_power_lag1"] = elec_lag1
    out["gas_consumption_lag1"] = gas_lag1

    # ----------------------------------------
    # Split into per-model feature sets
    # ----------------------------------------
    _DEFAULT_ELEC_COLS = [
        "hp_cop_inverse",
        "hp_cop_x_demand",
        "hp_temp_lift",
        "heat_demand_wind",
        "hp_heat_delivered",
        "solar_offset",
        "supply_lift",
        "supply_lift_sq",
        "heatpump_duty",
        "electric_power_lag1",
        "gas_consumption_lag1",
        "outdoor_temp",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    _DEFAULT_GAS_COLS = [
        "cold_below_bivalent",
        "boiler_duty",
        "boiler_demand",
        "peak_demand_signal",
        "cold_x_boiler",
        "electric_power_lag1",
        "gas_consumption_lag1",
        "outdoor_temp",
        "heat_demand_wind",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    e_cols = elec_feature_cols or _DEFAULT_ELEC_COLS
    g_cols = gas_feature_cols or _DEFAULT_GAS_COLS

    X_elec = out[[c for c in e_cols if c in out.columns]].copy()
    X_gas  = out[[c for c in g_cols if c in out.columns]].copy()

    return X_elec, X_gas


# ============================================================================
# HURDLE GAS MODEL
# ============================================================================

@dataclass
class _HurdleGasModel:
    """Two-stage gas consumption model.

    Stage 1  LogisticRegression → P(gas > 0)
    Stage 2  Ridge              → E[gas | gas > 0]
    """
    binary_threshold: float = 0.0
    binary_C: float = 1.0
    ridge_alpha: float = 1.0
    # Populated after fit()
    stage1_: Pipeline | None = field(default=None, repr=False)
    stage2_: Pipeline | None = field(default=None, repr=False)
    n_positive_: int = 0
    _is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_HurdleGasModel":
        y_bin = (y > self.binary_threshold).astype(int)
        self.n_positive_ = int(y_bin.sum())

        # Stage 1 — binary
        self.stage1_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.binary_C,
                max_iter=1000,
                class_weight="balanced",   # handles imbalance (~90% zeros)
                solver="lbfgs",
            )),
        ])
        self.stage1_.fit(X, y_bin)

        # Stage 2 — magnitude on positive samples only
        mask = y > self.binary_threshold
        if mask.sum() >= 5:
            self.stage2_ = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=self.ridge_alpha)),
            ])
            self.stage2_.fit(X[mask], y[mask])
        else:
            logger.warning(
                "_HurdleGasModel: only %d positive gas samples; "
                "skipping Stage 2, magnitude will default to mean.",
                mask.sum(),
            )
            self.stage2_ = None

        self._is_fitted = True
        logger.info(
            "HurdleGasModel fitted: %d total, %d positive (%.1f%%)",
            len(y), self.n_positive_, 100 * self.n_positive_ / max(len(y), 1),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("HurdleGasModel is not fitted yet.")

        prob_on = self.stage1_.predict_proba(X)[:, 1]   # P(gas > 0)

        if self.stage2_ is not None:
            magnitude = self.stage2_.predict(X).clip(min=0.0)
        else:
            magnitude = np.full(len(X), self.binary_threshold, dtype=float)

        return prob_on * magnitude

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(gas > 0) for each sample."""
        if not self._is_fitted:
            raise RuntimeError("HurdleGasModel is not fitted yet.")
        return self.stage1_.predict_proba(X)[:, 1]


# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

@dataclass
class HybridHeatPumpLR:
    """Physics-informed linear regression for hybrid heat pump consumption.

    Predicts ``electric_power`` (W) and ``gas_consumption`` (m³/interval)
    from raw sensor readings using physics-motivated features.

    Parameters
    ----------
    bivalent_point : float
        Outdoor temperature (°C) below which the gas boiler is expected to
        fire.  Adjust to match the actual system bivalent setting.
    hp_rated_norm : float
        Normalised HP capacity threshold used for the ``peak_demand_signal``
        gas feature.  Increase this for oversized heat pumps.
    ridge_alpha : float
        Regularisation strength for the Ridge electric model.
    gas_ridge_alpha : float
        Regularisation strength for the Stage-2 gas Ridge model.
    gas_binary_C : float
        Inverse regularisation parameter for the Stage-1 LogisticRegression.
    gas_binary_threshold : float
        Values at or below this are treated as "gas off" for Stage 1.
    standby_power_w : float or None
        Standby electric power [W] when the heat pump duty is effectively off.
        If None, this is estimated from training samples with low duty.
    standby_duty_threshold : float
        Duty threshold below which the heat pump is considered off/standby.
    boiler_cutoff_temp : float
        Boiler hard cut-off temperature [°C]. For outdoor temperatures above
        this value, gas is forced to 0.
    max_boiler_cutoff_temp : float
        Safety cap for boiler_cutoff_temp. Defaults to 5 °C for this setup.
    elec_feature_cols : list[str] or None
        Override default electric feature columns.
    gas_feature_cols : list[str] or None
        Override default gas feature columns.

    Attributes (populated after .fit())
    ------------------------------------
    elec_model_ : sklearn Pipeline
    gas_model_  : _HurdleGasModel
    elec_feature_names_ : list[str]
    gas_feature_names_  : list[str]
    """

    bivalent_point: float = 2.0
    hp_rated_norm: float = 3.0
    ridge_alpha: float = 1.0
    gas_ridge_alpha: float = 1.0
    gas_binary_C: float = 1.0
    gas_binary_threshold: float = 0.0
    standby_power_w: float | None = None
    standby_duty_threshold: float = 0.02
    boiler_cutoff_temp: float = 5.0
    max_boiler_cutoff_temp: float = 5.0
    elec_feature_cols: list[str] | None = None
    gas_feature_cols: list[str] | None = None

    # Populated after fit()
    elec_model_: Pipeline | None = field(default=None, repr=False, init=False)
    gas_model_: _HurdleGasModel | None = field(default=None, repr=False, init=False)
    elec_feature_names_: list[str] = field(default_factory=list, init=False)
    gas_feature_names_: list[str] = field(default_factory=list, init=False)
    standby_power_w_: float = field(default=0.0, init=False)
    _is_fitted: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        y_elec: pd.Series | np.ndarray,
        y_gas: pd.Series | np.ndarray,
    ) -> "HybridHeatPumpLR":
        """Fit the electric and gas models.

        Parameters
        ----------
        df : DataFrame
            Training sensor data with a DatetimeIndex.
        y_elec : array-like
            Electric power targets [W].
        y_gas : array-like
            Gas consumption targets [m³/interval].
        """
        X_elec_df, X_gas_df = self._build_features(df)
        X_elec = X_elec_df.to_numpy(dtype=float)
        X_gas  = X_gas_df.to_numpy(dtype=float)
        y_e = np.asarray(y_elec, dtype=float)
        y_g = np.asarray(y_gas, dtype=float)

        self.elec_feature_names_ = list(X_elec_df.columns)
        self.gas_feature_names_  = list(X_gas_df.columns)

        duty_signal = self._get_signal(df, "heatpump_duty").clip(0.0, 1.0)
        self.standby_power_w_ = self._resolve_standby_power(duty_signal, y_e)

        # ---------- Electric model ----------
        self.elec_model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.ridge_alpha)),
        ])
        self.elec_model_.fit(X_elec, y_e)
        logger.info("HybridHeatPumpLR: electric model fitted (%d samples, %d features)",
                    len(y_e), X_elec.shape[1])
        logger.info(
            "HybridHeatPumpLR: standby_power_w=%.2f W (duty_threshold=%.3f)",
            self.standby_power_w_,
            self.standby_duty_threshold,
        )

        # ---------- Gas hurdle model ----------
        self.gas_model_ = _HurdleGasModel(
            binary_threshold=self.gas_binary_threshold,
            binary_C=self.gas_binary_C,
            ridge_alpha=self.gas_ridge_alpha,
        )
        self.gas_model_.fit(X_gas, y_g)

        self._is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict electric power and gas consumption.

        Returns
        -------
        elec_pred : np.ndarray, shape (n,)  [W]
        gas_pred  : np.ndarray, shape (n,)  [m³/interval]
        """
        self._check_fitted()
        X_elec_df, X_gas_df = self._build_features(df)
        elec_pred = self.elec_model_.predict(X_elec_df.to_numpy(dtype=float)).clip(min=0.0)
        gas_pred  = self.gas_model_.predict(X_gas_df.to_numpy(dtype=float))

        duty_signal = self._get_signal(df, "heatpump_duty").clip(0.0, 1.0)
        outdoor_signal = self._get_signal(df, "outdoor_temp")
        hp_off_mask = duty_signal <= self.standby_duty_threshold
        cutoff = self._effective_boiler_cutoff()
        above_cutoff_mask = outdoor_signal > cutoff

        # Rule 1: when HP duty is off, electric falls back to standby and gas is off.
        elec_pred[hp_off_mask] = self.standby_power_w_
        gas_pred[hp_off_mask] = 0.0

        # Rule 2: above boiler cut-off temperature, boiler cannot fire.
        gas_pred[above_cutoff_mask] = 0.0

        return elec_pred, gas_pred

    def predict_gas_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(gas > 0) for each row in ``df``."""
        self._check_fitted()
        _, X_gas_df = self._build_features(df)
        gas_proba = self.gas_model_.predict_proba(X_gas_df.to_numpy(dtype=float))

        duty_signal = self._get_signal(df, "heatpump_duty").clip(0.0, 1.0)
        outdoor_signal = self._get_signal(df, "outdoor_temp")
        hp_off_mask = duty_signal <= self.standby_duty_threshold
        above_cutoff_mask = outdoor_signal > self._effective_boiler_cutoff()
        gas_proba[hp_off_mask | above_cutoff_mask] = 0.0
        return gas_proba

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coef_summary(self) -> pd.DataFrame:
        """Return a DataFrame of feature coefficients for both models.

        Useful for physics sanity-checks (e.g. ``hp_cop_inverse`` should
        have a large positive coefficient in the electric model).
        """
        self._check_fitted()
        rows = []

        elec_reg = self.elec_model_.named_steps["reg"]
        elec_scaler = self.elec_model_.named_steps["scaler"]
        for name, raw_coef, scale in zip(
            self.elec_feature_names_,
            elec_reg.coef_,
            elec_scaler.scale_,
        ):
            rows.append({
                "model": "electric",
                "feature": name,
                "coef_raw": float(raw_coef),
                # Standardised coefficient = coef × std  → comparable magnitudes
                "coef_standardised": float(raw_coef * scale),
            })

        gas_reg = self.gas_model_.stage2_
        if gas_reg is not None:
            gas_inner = gas_reg.named_steps["reg"]
            gas_scaler = gas_reg.named_steps["scaler"]
            for name, raw_coef, scale in zip(
                self.gas_feature_names_,
                gas_inner.coef_,
                gas_scaler.scale_,
            ):
                rows.append({
                    "model": "gas_magnitude",
                    "feature": name,
                    "coef_raw": float(raw_coef),
                    "coef_standardised": float(raw_coef * scale),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return build_heatpump_features(
            df,
            bivalent_point=self.bivalent_point,
            hp_rated_norm=self.hp_rated_norm,
            elec_feature_cols=self.elec_feature_cols,
            gas_feature_cols=self.gas_feature_cols,
        )

    def _get_signal(self, df: pd.DataFrame, name: str) -> np.ndarray:
        if name in df.columns:
            return df[name].fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(df), dtype=float)

    def _effective_boiler_cutoff(self) -> float:
        if self.boiler_cutoff_temp > self.max_boiler_cutoff_temp:
            logger.warning(
                "HybridHeatPumpLR: boiler_cutoff_temp=%.2f exceeds max %.2f; clamped.",
                self.boiler_cutoff_temp,
                self.max_boiler_cutoff_temp,
            )
        return float(min(self.boiler_cutoff_temp, self.max_boiler_cutoff_temp))

    def _resolve_standby_power(
        self,
        duty_signal: np.ndarray,
        y_electric: np.ndarray,
    ) -> float:
        if self.standby_power_w is not None:
            return float(max(self.standby_power_w, 0.0))

        standby_mask = duty_signal <= self.standby_duty_threshold
        if int(np.sum(standby_mask)) >= 5:
            # Robust to spikes in standby windows.
            return float(np.median(np.clip(y_electric[standby_mask], a_min=0.0, a_max=None)))

        # Fallback if there are too few explicit standby samples.
        return float(np.percentile(np.clip(y_electric, a_min=0.0, a_max=None), 5))

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "HybridHeatPumpLR is not fitted yet. Call .fit() first."
            )
