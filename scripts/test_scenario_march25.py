"""
Test scenario: Boris comes home March 25 at 16:00 CET
- EV: charge 30 kWh at 11 kW (fast mode) before 09:00 CET March 26
- Dishwasher: eco cycle, done by 12:00 CET March 26
- Washing machine: eco 40°C, done by 12:00 CET March 26
- Heat pump schedule: setpoints per period (visualized, not in optimizer)
- PV: 10 kWp, azimuth 185°, tilt 35°, inverter 7.4 kW
- Base load: 200 W constant
- Prices: from test CSV
"""

import asyncio
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pvlib
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

from emhass.optimization import Optimization
from emhass.utils import build_config, build_params, get_logger, get_root, get_yaml_parse
from emhass.thermal.hybrid_heatpump_lr import HybridHeatPumpLR
from emhass.thermal.feature_engineering import normalise_sensors, build_feature_matrix
from emhass.thermal.forecast_gridsearch import (
    SearchOptions,
    create_sequences,
    create_physics_context_sequences,
)
from emhass.thermal.pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM
import torch
import torch.optim as _torch_optim
import time as _time

# ── Paths ─────────────────────────────────────────────────────────────────────
root = pathlib.Path(str(get_root(__file__, num_parent=2)))
OUTPUT_HTML = root / "logs" / "test_scenario_march25_latest.html"
OUTPUT_COMPARISON_CSV = root / "logs" / "test_scenario_march25_costs_comparison.csv"
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "config_path": root / "config.json",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = get_logger(__name__, emhass_conf, save_to_file=False)
logging.getLogger("emhass.thermal.feature_engineering").setLevel(logging.WARNING)

# ── Scenario constants ─────────────────────────────────────────────────────────
TZ_NL = "Europe/Amsterdam"
START_UTC = pd.Timestamp("2026-03-25 15:00:00", tz="UTC")   # 16:00 CET

PV_KWP          = 10.0     # kWp
INVERTER_MAX_W  = 7400.0   # W  (7.4 kW inverter)
PV_TILT         = 35       # degrees
PV_AZIMUTH      = 185      # degrees from North (185° ≈ slightly SW of South)
LAT, LON        = 52.37, 4.90  # Amsterdam (approximate)

BASE_LOAD_W     = 200.0    # W constant household background
EV_POWER_W      = 11000.0  # W (11 kW fast charge)
EV_NEEDED_KWH   = 30.0     # kWh to add
TS_MIN          = 15       # optimization time step in minutes

# Power curves (W per 15-min slot) for eco cycles
DISHWASHER_CURVE = [900, 900, 60, 60, 60, 60, 900, 900, 60, 60, 60, 60, 60, 60]  # 14 slots = 3h30
WASHINGMACHINE_CURVE = [1500, 400, 100, 100, 100, 100, 100, 100, 200, 400, 600, 200]  # 12 slots = 3h

# Window ends (slot index from start):
#   EV: done by 09:00 CET March 26 = 08:00 UTC = start + 17h = 68 slots
#   Appliances: done by 16:00 CET March 26 = 15:00 UTC = start + 24h = 96 slots
EV_WINDOW_END   = 68
APPL_WINDOW_END = 96

GAS_PRICE_EUR_M3 = 1.10   # typical Dutch gas price (€/m³); gas_consumption in CSV is m³/kwartier

TEMP_ONLY_TARGET_C = 20.5
TEMP_ONLY_MAX_DEV_C = 0.25
TEMP_ONLY_MIN_C = TEMP_ONLY_TARGET_C - TEMP_ONLY_MAX_DEV_C
TEMP_ONLY_MAX_C = TEMP_ONLY_TARGET_C + TEMP_ONLY_MAX_DEV_C
TEMP_ONLY_TARGET_WEIGHT = 1800.0
TEMP_ONLY_SWITCH_PENALTY = 6.0
TEMP_ONLY_HARD_BAND_PENALTY = 1_000_000.0
MAX_ROOM_TEMP_STEP_C = 0.35
TEMP_ONLY_COOLDOWN_DELTA_C = 10.0
PHASE_SMOOTH_WINDOW = 3

# Forecast backend: "lstm_lr" (default) or "physics_only"
FORECAST_BACKEND = os.getenv("MARCH25_FORECAST_BACKEND", "lstm_lr").strip().lower()
# Optional Phase 3: run standard EMHASS optimizer again after lookahead stage
RUN_POST_STANDARD_OPT = os.getenv("MARCH25_RUN_POST_STANDARD_OPT", "1").strip().lower() in {"1", "true", "yes"}

# Heat pump setpoint schedule (local CET/CEST time boundaries, no DST issue before March 29)
# Format: list of (start_local, end_local, temp_min, temp_max)
# Times are local ISO strings. DST switchover on March 29 at 02:00 → 03:00.
HP_SCHEDULE = [
    # Constant 20-22°C throughout for simplified testing
    ("2026-03-25 16:00", "2026-03-29 12:00", 20.0, 22.0),
]


def compute_pv(df: pd.DataFrame) -> pd.Series:
    """Compute AC PV power (W) from irradiance columns using pvlib."""
    location = pvlib.location.Location(latitude=LAT, longitude=LON, tz="UTC", altitude=5)
    solpos = location.get_solarposition(df.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PV_TILT,
        surface_azimuth=PV_AZIMUTH,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=df["dni"],
        ghi=df["ghi"],
        dhi=df["dhi"],
    )
    poa_global = poa["poa_global"].fillna(0).clip(lower=0)
    # DC power: simple single-diode approximation
    # P_DC = P_STC * (G_poa / G_stc) * (1 + gamma * (T_cell - T_stc))
    g_stc = 1000.0             # W/m²
    t_stc = 25.0               # °C
    gamma = -0.004             # /°C temperature coefficient
    t_cell = df["outdoor_temp"] + 25.0  # rough NOCT approximation
    p_dc = PV_KWP * 1000 * (poa_global / g_stc) * (1 + gamma * (t_cell - t_stc))
    p_dc = p_dc.clip(lower=0)
    # Apply inverter limit and typical inverter efficiency
    eta_inv = 0.97
    p_ac = (p_dc * eta_inv).clip(upper=INVERTER_MAX_W, lower=0)
    return p_ac.rename("p_pv_forecast")


def build_hp_setpoint_series(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Return temp_min and temp_max series aligned to the optimization index (UTC)."""
    temp_min = pd.Series(np.nan, index=index)
    temp_max = pd.Series(np.nan, index=index)
    for start_str, end_str, t_min, t_max in HP_SCHEDULE:
        start_utc = pd.Timestamp(start_str, tz=TZ_NL).tz_convert("UTC")
        end_utc   = pd.Timestamp(end_str,   tz=TZ_NL).tz_convert("UTC")
        mask = (index >= start_utc) & (index < end_utc)
        temp_min[mask] = t_min
        temp_max[mask] = t_max
    # Forward/backward fill edges
    temp_min = temp_min.ffill().bfill()
    temp_max = temp_max.ffill().bfill()
    return temp_min, temp_max


def build_constant_temp_band(index: pd.DatetimeIndex, temp_min: float, temp_max: float) -> tuple[pd.Series, pd.Series]:
    """Return a constant comfort band aligned to the optimization index."""
    return (
        pd.Series(temp_min, index=index, dtype=float),
        pd.Series(temp_max, index=index, dtype=float),
    )


def _physics_features_rls(df: pd.DataFrame) -> np.ndarray:
    duty = df.get("heatpump_duty", pd.Series(0.0, index=df.index)).fillna(0.0).to_numpy(dtype=float)
    room = df.get("room_temp", pd.Series(20.0, index=df.index)).fillna(20.0).to_numpy(dtype=float)
    outdoor = df.get("outdoor_temp", pd.Series(10.0, index=df.index)).fillna(10.0).to_numpy(dtype=float)
    supply_default = pd.Series(room + 5.0, index=df.index)
    supply = df.get("supply_temp", supply_default).fillna(supply_default).to_numpy(dtype=float)

    delta_supply = np.clip(supply - room, a_min=0.0, a_max=None)
    delta_env = np.clip(room - outdoor, a_min=0.0, a_max=None)
    cold = (outdoor < 2.0).astype(float)

    return np.column_stack([
        np.ones(len(df), dtype=float),
        duty,
        delta_supply,
        duty * delta_supply,
        delta_env,
        duty * delta_env,
        cold,
        duty * cold,
    ])


def _rls_fit_predict_series(
    train_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    target_col: str,
    *,
    forgetting: float = 0.995,
    ridge: float = 1.0,
) -> np.ndarray:
    x_train = _physics_features_rls(train_df)
    x_scen = _physics_features_rls(scen_df)
    y_train = train_df[target_col].fillna(0.0).to_numpy(dtype=float)
    y_scen = scen_df[target_col].fillna(0.0).to_numpy(dtype=float)

    n_feat = x_train.shape[1]
    theta = np.zeros(n_feat, dtype=float)
    p_mat = (1.0 / max(1e-9, ridge)) * np.eye(n_feat, dtype=float)

    def _update(x_row: np.ndarray, y_val: float) -> None:
        nonlocal theta, p_mat
        x_col = x_row.reshape(-1, 1)
        denom = float(forgetting + (x_col.T @ p_mat @ x_col)[0, 0])
        gain = (p_mat @ x_col) / max(1e-9, denom)
        err = float(y_val - (theta @ x_row))
        theta = theta + gain[:, 0] * err
        p_mat = (p_mat - (gain @ x_col.T @ p_mat)) / max(1e-9, forgetting)

    for xr, yr in zip(x_train, y_train, strict=False):
        _update(xr, float(yr))

    preds = np.zeros(len(x_scen), dtype=float)
    for i, (xr, yr) in enumerate(zip(x_scen, y_scen, strict=False)):
        preds[i] = max(0.0, float(theta @ xr))
        _update(xr, float(yr))
    return preds


async def main():
    # ── Load CSV data ──────────────────────────────────────────────────────────
    csv_path = root / "tests_thermal/data/test_data.csv"
    df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_raw.index = df_raw.index.tz_convert("UTC")  # ensure UTC
    df = df_raw[df_raw.index >= START_UTC].copy()
    df = df.resample(f"{TS_MIN}min").interpolate(method="time")
    df = df.ffill()
    N = len(df)
    print(f"Optimization horizon: {N} slots x {TS_MIN} min = {N * TS_MIN / 60:.1f} h")
    print(f"  From: {df.index[0].tz_convert(TZ_NL)}")
    print(f"  To:   {df.index[-1].tz_convert(TZ_NL)}")
    print(f"  Forecast backend: {FORECAST_BACKEND}")
    print(f"  Run post-lookahead standard optimization: {RUN_POST_STANDARD_OPT}")

    # ── Train HybridHeatPumpLR on historical data before scenario ─────────────
    # Use all data before scenario start as training set, then predict for scenario
    price_drop = ["sensor.current_electricity_market_price"]
    df_train_raw = normalise_sensors(
        df_raw[df_raw.index < START_UTC].drop(columns=price_drop, errors="ignore")
    )
    df_scen_raw = normalise_sensors(
        df.drop(columns=price_drop, errors="ignore")
    )
    lr_model = HybridHeatPumpLR(bivalent_point=2.0, ridge_alpha=1.0, gas_ridge_alpha=1.0)
    lr_model.fit(
        df_train_raw,
        df_train_raw["electric_power"].to_numpy(),
        df_train_raw["gas_consumption"].to_numpy(),
    )
    pred_elec_w, pred_gas_m3 = lr_model.predict(df_scen_raw)
    pred_elec_w  = pred_elec_w.clip(min=0.0)
    pred_gas_m3  = pred_gas_m3.clip(min=0.0)
    print(f"\nHybridLR trained on {len(df_train_raw)} samples (before {START_UTC.tz_convert(TZ_NL)})")
    print(f"  Predicted elec (WP): mean {pred_elec_w.mean():.0f} W, max {pred_elec_w.max():.0f} W")
    print(f"  Predicted gas:       total {pred_gas_m3.sum():.2f} m3")

    # ── Train PINN-LSTM and build 50/50 ensemble with LR ─────────────────────
    print("\nTraining PINN-LSTM for ensemble with LR...")
    _t0 = _time.time()
    LSTM_INPUT_WINDOW = 96        # 24h lookback (captures daily patterns; faster on CPU than 192)
    LSTM_HIDDEN       = 64        # smaller hidden is faster on CPU
    LSTM_LAYERS       = 2
    LSTM_EPOCHS       = 0 if FORECAST_BACKEND == "physics_only" else 20
    LSTM_PATIENCE     = 10
    LSTM_BATCH        = 32        # bigger batches = much faster on CPU
    LSTM_TARGET_COLS  = ["room_temp", "electric_power", "gas_consumption"]

    opts_lstm = SearchOptions(
        epochs=LSTM_EPOCHS, patience=LSTM_PATIENCE,
        batch_size=LSTM_BATCH, lookahead=1,
        feature_level="standard",
        target_cols=LSTM_TARGET_COLS,
        physics_loss_weight=0.1, physics_balance_weight=0.05,
        seed=42, latitude=LAT, longitude=LON,
    )

    # Full feature/target matrices with explicit scalers so we can reuse them
    # during autoregressive control simulation under modified duty/setpoints.
    _raw_for_idx = pd.read_csv(csv_path)
    _raw_for_idx["timestamp"] = pd.to_datetime(_raw_for_idx["timestamp"])
    _raw_for_idx = _raw_for_idx.set_index("timestamp").drop(
        columns=["sensor.current_electricity_market_price"], errors="ignore"
    )
    _feat_for_idx, feature_cols_lstm = build_feature_matrix(
        _raw_for_idx,
        feature_level="standard",
        latitude=LAT, longitude=LON,
        target_col=LSTM_TARGET_COLS[0],
        exclude_feature_cols=LSTM_TARGET_COLS,
        drop_na=True,
    )
    aligned_idx = _feat_for_idx.index  # UTC-aware DatetimeIndex after NaN-drop

    scaler_X_lstm = StandardScaler()
    scaler_y_lstm = StandardScaler()
    X_full = scaler_X_lstm.fit_transform(_feat_for_idx[feature_cols_lstm].values)
    y_full = scaler_y_lstm.fit_transform(_feat_for_idx[LSTM_TARGET_COLS].values)

    if "outdoor_temp" in feature_cols_lstm:
        outdoor_scaled = X_full[:, feature_cols_lstm.index("outdoor_temp")]
    else:
        outdoor_scaled = np.zeros(len(_feat_for_idx), dtype=np.float32)
    if "solar_heat" in feature_cols_lstm:
        solar_scaled = X_full[:, feature_cols_lstm.index("solar_heat")]
    else:
        solar_scaled = np.zeros(len(_feat_for_idx), dtype=np.float32)
    room_idx_lstm = LSTM_TARGET_COLS.index("room_temp")
    phys_signals = {
        "room_temp": y_full[:, room_idx_lstm].astype(np.float32),
        "outdoor_temp": outdoor_scaled.astype(np.float32),
        "solar_heat": solar_scaled.astype(np.float32),
    }

    # Create sequences and date-based train / scenario split
    X_seq_full, y_seq_full = create_sequences(X_full, y_full, lookback=LSTM_INPUT_WINDOW, lookahead=1)
    n_seq_full   = len(X_seq_full)
    seq_start_ts = aligned_idx[LSTM_INPUT_WINDOW : LSTM_INPUT_WINDOW + n_seq_full]

    train_mask = seq_start_ts < START_UTC
    scen_mask  = seq_start_ts >= START_UTC
    print(f"  Sequences: {int(train_mask.sum())} train, {int(scen_mask.sum())} scenario")

    # Physics context sequences for physics-informed training loss
    ctx_seq = None
    if phys_signals is not None:
        ctx_seq = create_physics_context_sequences(
            room_temp_signal=phys_signals["room_temp"],
            outdoor_signal=phys_signals["outdoor_temp"],
            solar_signal=phys_signals["solar_heat"],
            lookback=LSTM_INPUT_WINDOW, lookahead=1,
        )

    # Split pre-scenario sequences into train (85%) and validation (15%)
    X_tr_all = X_seq_full[train_mask]
    y_tr_all = y_seq_full[train_mask]
    n_trval  = len(X_tr_all)
    n_val_l  = max(1, int(0.15 * n_trval))
    n_tr_l   = n_trval - n_val_l
    X_tr, y_tr = X_tr_all[:n_tr_l], y_tr_all[:n_tr_l]
    X_va, y_va = X_tr_all[n_tr_l:], y_tr_all[n_tr_l:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_lstm = QuantilePhysicsInformedLSTM(
        input_size=X_full.shape[1],
        hidden=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        lookahead=1,
        targets=y_full.shape[1],
        dropout=0.2,
    ).to(device)

    opt_lstm     = _torch_optim.Adam(model_lstm.parameters(), lr=1e-3)
    loss_fn_lstm = QuantileLoss(weight_physics=0.1, weight_physics_balance=0.05)

    def _bctx(start_idx: int, end_idx: int):
        if ctx_seq is None:
            return None
        return {
            "room_temp_prev": torch.tensor(ctx_seq["room_temp_prev"][start_idx:end_idx], dtype=torch.float32, device=device),
            "outdoor_temp":   torch.tensor(ctx_seq["outdoor_temp"][start_idx:end_idx],   dtype=torch.float32, device=device),
            "solar_heat":     torch.tensor(ctx_seq["solar_heat"][start_idx:end_idx],     dtype=torch.float32, device=device),
        }

    best_state_lstm = None
    best_val_lstm   = float("inf")
    patience_ctr    = 0

    for epoch in range(1, LSTM_EPOCHS + 1):
        model_lstm.train()
        for i in range(0, len(X_tr), LSTM_BATCH):
            j  = min(i + LSTM_BATCH, len(X_tr))
            xb = torch.tensor(X_tr[i:j], dtype=torch.float32, device=device)
            yb = torch.tensor(y_tr[i:j], dtype=torch.float32, device=device)
            opt_lstm.zero_grad()
            loss_fn_lstm(model_lstm(xb), yb, physics_context=_bctx(i, j))["total"].backward()
            opt_lstm.step()

        model_lstm.eval()
        val_losses_l = []
        with torch.no_grad():
            for i in range(0, len(X_va), LSTM_BATCH):
                j  = min(i + LSTM_BATCH, len(X_va))
                xb = torch.tensor(X_va[i:j], dtype=torch.float32, device=device)
                yb = torch.tensor(y_va[i:j], dtype=torch.float32, device=device)
                out = model_lstm(xb)
                val_losses_l.append(
                    float(loss_fn_lstm(out, yb, physics_context=_bctx(n_tr_l + i, n_tr_l + j))["total"].item())
                )

        v_loss = float(np.mean(val_losses_l))
        if epoch % 5 == 0:
            print(f"  LSTM epoch {epoch:2d}/{LSTM_EPOCHS} val_loss={v_loss:.4f}")
        if v_loss < best_val_lstm:
            best_val_lstm   = v_loss
            best_state_lstm = {k: v.detach().cpu().clone() for k, v in model_lstm.state_dict().items()}
            patience_ctr    = 0
        else:
            patience_ctr += 1
            if patience_ctr >= LSTM_PATIENCE:
                print(f"  LSTM early stop at epoch {epoch}/{LSTM_EPOCHS}")
                break

    if best_state_lstm:
        model_lstm.load_state_dict(best_state_lstm)

    # LSTM inference on scenario sequences (teacher-forced, 1-step ahead)
    X_scen  = X_seq_full[scen_mask]
    scen_ts = seq_start_ts[scen_mask]   # UTC-aware DatetimeIndex

    model_lstm.eval()
    _lstm_preds_lst = []
    with torch.no_grad():
        for i in range(0, len(X_scen), LSTM_BATCH):
            j  = min(i + LSTM_BATCH, len(X_scen))
            xb = torch.tensor(X_scen[i:j], dtype=torch.float32, device=device)
            _lstm_preds_lst.append(model_lstm(xb)["q50"].cpu().numpy())  # (batch, 1, n_targets)

    _lstm_preds_arr = np.vstack(_lstm_preds_lst)                          # (N_scen, 1, n_targets)
    _lstm_preds_dn  = scaler_y_lstm.inverse_transform(_lstm_preds_arr[:, 0, :])  # (N_scen, n_targets)

    _ri = LSTM_TARGET_COLS.index("room_temp")
    _ei = LSTM_TARGET_COLS.index("electric_power")
    _gi = LSTM_TARGET_COLS.index("gas_consumption")

    lstm_room_df = pd.Series(_lstm_preds_dn[:, _ri],         index=scen_ts).reindex(df.index, method="nearest").fillna(20.0)
    lstm_elec_df = pd.Series(_lstm_preds_dn[:, _ei].clip(0), index=scen_ts).reindex(df.index, method="nearest").fillna(0.0)
    lstm_gas_df  = pd.Series(_lstm_preds_dn[:, _gi].clip(0), index=scen_ts).reindex(df.index, method="nearest").fillna(0.0)

    # Forecast backend selection for lookahead simulation anchors.
    if FORECAST_BACKEND == "physics_only":
        print("\nUsing physics_only backend for lookahead stage (no DL anchors).")
        phys_elec = _rls_fit_predict_series(df_train_raw, df_scen_raw, "electric_power", forgetting=0.995, ridge=1.0)
        phys_gas = _rls_fit_predict_series(df_train_raw, df_scen_raw, "gas_consumption", forgetting=0.995, ridge=1.0)
        phys_room = _rls_fit_predict_series(df_train_raw, df_scen_raw, "room_temp", forgetting=0.995, ridge=1.0)

        lstm_elec_df = pd.Series(phys_elec, index=df_scen_raw.index).reindex(df.index, method="nearest").fillna(0.0)
        lstm_gas_df = pd.Series(phys_gas, index=df_scen_raw.index).reindex(df.index, method="nearest").fillna(0.0)
        lstm_room_df = pd.Series(phys_room, index=df_scen_raw.index).reindex(df.index, method="nearest").fillna(20.0)

        ens_pred_elec_w = np.clip(phys_elec, a_min=0.0, a_max=None)
        ens_pred_gas_m3 = np.clip(phys_gas, a_min=0.0, a_max=None)
        print(
            f"  Physics room_temp: mean {lstm_room_df.mean():.1f}C "
            f"[{lstm_room_df.min():.1f}, {lstm_room_df.max():.1f}]"
        )
        print(f"  Physics WP elec: mean {np.mean(ens_pred_elec_w):.0f} W, max {np.max(ens_pred_elec_w):.0f} W")
        print(f"  Physics gas:     total {np.sum(ens_pred_gas_m3):.2f} m3")
    else:
        # Ensemble: 50/50 average of LR + LSTM (elec and gas); LSTM-only for room_temp
        ens_pred_elec_w = (pred_elec_w + lstm_elec_df.values) / 2.0
        ens_pred_gas_m3 = (pred_gas_m3 + lstm_gas_df.values) / 2.0

    print(f"  LSTM stage done in {_time.time() - _t0:.1f}s")
    print(f"  Anchor room_temp: mean {lstm_room_df.mean():.1f}C [{lstm_room_df.min():.1f}, {lstm_room_df.max():.1f}]")
    print(f"  Anchor WP elec: mean {ens_pred_elec_w.mean():.0f} W, max {ens_pred_elec_w.max():.0f} W")
    print(f"  Anchor gas:     total {ens_pred_gas_m3.sum():.2f} m3")

    # ── PV forecast ────────────────────────────────────────────────────────────
    p_pv = compute_pv(df)
    print(f"\nPV: peak = {p_pv.max()/1000:.2f} kW, total = {p_pv.sum()*TS_MIN/60/1000:.1f} kWh")

    # ── Load forecast (constant base load) ────────────────────────────────────
    p_load = pd.Series(BASE_LOAD_W, index=df.index, name="p_load_forecast")

    # ── Electricity prices ─────────────────────────────────────────────────────
    price_col = "sensor.current_electricity_market_price"
    unit_load_cost  = df[price_col].fillna(0).clip(lower=0)  # EUR/kWh, no negative buying price
    unit_prod_price = unit_load_cost * 0.9                    # sell at 90% of buy price
    # Small buffer to guarantee non-zero prices
    unit_load_cost  = unit_load_cost.where(unit_load_cost > 0.001, 0.05)

    # ── Build df_input_data ───────────────────────────────────────────────────
    df_input = pd.DataFrame({
        "p_pv_forecast":    p_pv.values,
        "p_load_forecast":  p_load.values,
        "unit_load_cost":   unit_load_cost.values,
        "unit_prod_price":  unit_prod_price.values,
    }, index=df.index)

    # ── Build EMHASS config ───────────────────────────────────────────────────
    config = await build_config(emhass_conf, logger, str(emhass_conf["defaults_path"]))
    config["optimization_time_step"] = TS_MIN
    config["delta_forecast_daily"]   = 5      # multi-day horizon
    config["time_zone"]              = TZ_NL

    params = await build_params(emhass_conf, {}, config, logger)
    retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params, logger)

    # EV operating hours: need 30 kWh at 11 kW
    ev_hours = EV_NEEDED_KWH / (EV_POWER_W / 1000)  # = 2.727 h

    # Override optimizer config
    optim_conf.update({
        "number_of_deferrable_loads": 3,
        # Load 0: EV (scalar → standard/single-block load)
        # Load 1: Dishwasher (list → sequence load, power curve placed optimally)
        # Load 2: Washing machine (list → sequence load)
        "nominal_power_of_deferrable_loads": [
            EV_POWER_W,
            DISHWASHER_CURVE,
            WASHINGMACHINE_CURVE,
        ],
        "operating_hours_of_each_deferrable_load": [
            ev_hours,
            len(DISHWASHER_CURVE) * TS_MIN / 60,
            len(WASHINGMACHINE_CURVE) * TS_MIN / 60,
        ],
        # EV: fixed window, single contiguous block at full power
        "start_timesteps_of_each_deferrable_load": [0, 0, 0],
        "end_timesteps_of_each_deferrable_load":   [EV_WINDOW_END, APPL_WINDOW_END, APPL_WINDOW_END],
        # EV: semi-continuous (full power or off) + single start (one block)
        # Sequence loads handle their own continuity via the convolution constraint
        "treat_deferrable_load_as_semi_cont":      [True, False, False],
        "set_deferrable_load_single_constant":     [True, False, False],
        "set_deferrable_startup_penalty":          [0, 0, 0],
        "minimum_power_of_deferrable_loads":       [0, 0, 0],
        "load_dispatch_mode":                      ["hours", "hours", "hours"],
        "def_load_config":                         [{}, {}, {}],
        # Cost function: maximize profit (minimize net electricity cost)
        "costfun":           "profit",
        "set_use_battery":   False,
        "set_use_pv":        True,
        "lp_solver":         "PULP_CBC_CMD",
        "lp_solver_path":    "empty",
        "lp_solver_timeout": 120,
    })

    # ── Build and run optimizer ───────────────────────────────────────────────
    print("\nBuilding EMHASS Optimization object...")
    opt = Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        var_load_cost="unit_load_cost",
        var_prod_price="unit_prod_price",
        costfun="profit",
        emhass_conf=emhass_conf,
        logger=logger,
    )

    print("Running day-ahead optimization...")
    opt_res = opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load)

    if opt_res is None or opt_res.empty:
        print("ERROR: Optimization returned no results.")
        return

    status = opt_res["optim_status"].iloc[0]
    total_cost = -opt_res["cost_profit"].sum()  # negative of profit = net cost
    print(f"\n[OK] Optimization status: {status}")
    print(f"  Total net cost: EUR {total_cost:.4f}")

    # ── Extract results ───────────────────────────────────────────────────────
    idx_local = opt_res.index.tz_convert(TZ_NL)

    p_ev   = opt_res.get("P_deferrable0", pd.Series(0, index=opt_res.index))
    p_dish = opt_res.get("P_deferrable1", pd.Series(0, index=opt_res.index))
    p_wash = opt_res.get("P_deferrable2", pd.Series(0, index=opt_res.index))
    p_grid = opt_res.get("P_grid",        pd.Series(0, index=opt_res.index))
    p_pv_r = opt_res.get("P_PV",          p_pv.values)

    ev_start_slot = int((p_ev > 100).idxmax().timestamp() - df.index[0].timestamp()) // (TS_MIN * 60) \
        if (p_ev > 100).any() else None
    ev_total_kwh = float(p_ev.sum()) * TS_MIN / 60 / 1000
    dish_total_kwh = float(p_dish.sum()) * TS_MIN / 60 / 1000
    wash_total_kwh = float(p_wash.sum()) * TS_MIN / 60 / 1000

    print(f"\n  EV charged:         {ev_total_kwh:.2f} kWh  (target {EV_NEEDED_KWH:.0f} kWh)")
    print(f"  Dishwasher energy:  {dish_total_kwh:.3f} kWh")
    print(f"  Washing machine:    {wash_total_kwh:.3f} kWh")
    if (p_ev > 100).any():
        first = opt_res.index[p_ev > 100][0].tz_convert(TZ_NL)
        last  = opt_res.index[p_ev > 100][-1].tz_convert(TZ_NL)
        print(f"  EV charges:         {first.strftime('%H:%M')} - {last.strftime('%H:%M')} ({first.date()})")
    if (p_dish > 10).any():
        first = opt_res.index[p_dish > 10][0].tz_convert(TZ_NL)
        last  = opt_res.index[p_dish > 10][-1].tz_convert(TZ_NL)
        print(f"  Dishwasher runs:    {first.strftime('%d/%m %H:%M')} - {last.strftime('%H:%M')}")
    if (p_wash > 10).any():
        first = opt_res.index[p_wash > 10][0].tz_convert(TZ_NL)
        last  = opt_res.index[p_wash > 10][-1].tz_convert(TZ_NL)
        print(f"  Washing machine:    {first.strftime('%d/%m %H:%M')} - {last.strftime('%H:%M')}")

    # ── Pre-compute temperatures and comfort schedule ─────────────────────────
    outdoor_t  = df["outdoor_temp"].reindex(opt_res.index, method="nearest").fillna(0)
    supply_t   = df["supply_temp"].reindex(opt_res.index, method="nearest").fillna(0)
    gas_vals   = df["gas_consumption"].reindex(opt_res.index, method="nearest").fillna(0)
    elec_hp    = df["electric_power"].reindex(opt_res.index, method="nearest").fillna(0)
    temp_min_hp, temp_max_hp = build_hp_setpoint_series(opt_res.index)
    temp_min_temp_only, temp_max_temp_only = build_constant_temp_band(
        opt_res.index,
        TEMP_ONLY_MIN_C,
        TEMP_ONLY_MAX_C,
    )

    price_for_hp = df_input["unit_load_cost"].reindex(opt_res.index, method="nearest")
    sell_arr = df_input["unit_prod_price"].values
    price_arr = price_for_hp.values
    scen_exog = df_scen_raw.reindex(opt_res.index).copy()

    MIN_SWITCH_GAP_SLOTS = int(120 / TS_MIN)  # at most one duty switch every 2 hours
    MAX_SWITCHES_PER_DAY = 4
    MAX_SUPPLY_STEP_C = 2.0  # smooth supply trajectory (degC per 15-min step)

    def _curve_bounds(outdoor_temp_value: float) -> tuple[float, float]:
        # Weather curve: y = 40 - x, with hard +-10 degC band.
        curve = 40.0 - outdoor_temp_value
        low = max(25.0, curve - 10.0)
        high = min(55.0, curve + 10.0)
        if low > high:
            low = high
        return float(low), float(high)

    def _clip_supply(temp_value: float, outdoor_temp_value: float) -> float:
        low, high = _curve_bounds(outdoor_temp_value)
        return float(np.clip(temp_value, low, high))

    def _baseline_supply(outdoor_temp_value: float) -> float:
        return _clip_supply(40.0 - outdoor_temp_value, outdoor_temp_value)

    def _predict_lr_step(history_df: pd.DataFrame, candidate_row: pd.Series) -> tuple[float, float]:
        temp_df = pd.concat([history_df.tail(1), candidate_row.to_frame().T])
        elec_pred_step, gas_pred_step = lr_model.predict(temp_df)
        return float(elec_pred_step[-1]), float(gas_pred_step[-1])

    LOOKAHEAD_HOURS = (6.0, 12.0, 24.0)
    LOOKAHEAD_QUANTILES = (0.30, 0.70)
    NUDGE_DEADBAND = 0.10
    NUDGE_GAIN_MIN_C = 2.5
    NUDGE_GAIN_MAX_C = 6.0
    NUDGE_GAIN_STEP_C = 0.5
    MAX_NUDGE_C = 7.0
    WEIGHT_GRID_STEP = 0.25

    lstm_room_s = lstm_room_df.reindex(opt_res.index, method="nearest").fillna(20.0)
    lstm_elec_s = lstm_elec_df.reindex(opt_res.index, method="nearest").fillna(0.0)
    lstm_gas_s = lstm_gas_df.reindex(opt_res.index, method="nearest").fillna(0.0)

    def _predict_next_room(
        room_temp_value: float,
        duty_value: int,
        supply_value: float,
        outdoor_value: float,
        lstm_anchor_room: float,
    ) -> float:
        heat_gain = 0.11 * int(duty_value) * max(0.0, float(supply_value) - room_temp_value)
        heat_loss = 0.045 * max(0.0, room_temp_value - outdoor_value)
        room_model_next = room_temp_value + heat_gain - heat_loss
        blended_next = float(0.65 * room_model_next + 0.35 * lstm_anchor_room)
        bounded_delta = float(np.clip(blended_next - room_temp_value, -MAX_ROOM_TEMP_STEP_C, MAX_ROOM_TEMP_STEP_C))
        return float(room_temp_value + bounded_delta)

    def _comfort_penalty(next_room: float, target_min_value: float, target_max_value: float) -> float:
        # Too cold is worse than too hot.
        below = max(0.0, target_min_value - next_room)
        above = max(0.0, next_room - target_max_value)
        return 1200.0 * (below ** 2) + 350.0 * (above ** 2)

    def _comfort_penalty_temp_only(next_room: float, target_value: float, target_min_value: float, target_max_value: float) -> float:
        below = max(0.0, target_min_value - next_room)
        above = max(0.0, next_room - target_max_value)
        center_error = next_room - target_value
        hard_penalty = TEMP_ONLY_HARD_BAND_PENALTY if (below > 0.0 or above > 0.0) else 0.0
        return hard_penalty + (1600.0 * (below ** 2)) + (1600.0 * (above ** 2)) + (TEMP_ONLY_TARGET_WEIGHT * (center_error ** 2))

    def _compute_price_signal(price_values: np.ndarray, lookahead_slots: int) -> np.ndarray:
        q_low, q_high = LOOKAHEAD_QUANTILES
        n_local = len(price_values)
        out = np.zeros(n_local, dtype=float)
        for i_local in range(n_local):
            j_local = min(n_local, i_local + lookahead_slots)
            window = price_values[i_local:j_local]
            if window.size == 0:
                continue
            low = float(np.nanquantile(window, q_low))
            high = float(np.nanquantile(window, q_high))
            center = 0.5 * (low + high)
            spread = max(1e-6, high - low)
            signal = (center - float(price_values[i_local])) / spread
            out[i_local] = float(np.clip(signal, -1.5, 1.5))
        return out

    def _smooth_control_series(series: pd.Series, window: int = PHASE_SMOOTH_WINDOW) -> pd.Series:
        if window <= 1:
            return series.astype(float).copy()
        return series.astype(float).rolling(window=window, center=True, min_periods=1).mean()

    def _enforce_supply_limits(supply_series: pd.Series) -> pd.Series:
        out = supply_series.astype(float).copy()
        if len(out) == 0:
            return out
        prev_supply = float(out.iloc[0])
        prev_supply = _clip_supply(prev_supply, float(outdoor_t.iloc[0]))
        out.iloc[0] = prev_supply
        for pos in range(1, len(out)):
            outdoor_value = float(outdoor_t.iloc[pos])
            raw_supply = float(out.iloc[pos])
            clipped_step = float(np.clip(raw_supply, prev_supply - MAX_SUPPLY_STEP_C, prev_supply + MAX_SUPPLY_STEP_C))
            clipped_curve = _clip_supply(clipped_step, outdoor_value)
            out.iloc[pos] = clipped_curve
            prev_supply = clipped_curve
        return out

    def _simulate_plan(
        duty_plan: pd.Series,
        supply_plan: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        local_history = df_train_raw.copy()
        room_state = float(scen_exog["room_temp"].iloc[0])
        room_path: list[float] = []
        elec_path: list[float] = []
        gas_path: list[float] = []
        for pos, ts_step in enumerate(opt_res.index):
            outdoor_value = float(outdoor_t.iloc[pos])
            duty_value = int(duty_plan.iloc[pos])
            supply_value = float(supply_plan.iloc[pos])

            room_path.append(room_state)

            row = scen_exog.loc[ts_step].copy()
            row["room_temp"] = room_state
            row["heatpump_duty"] = duty_value
            row["supply_temp"] = supply_value
            row["electric_power"] = 0.0
            row["gas_consumption"] = 0.0

            lr_elec_step, lr_gas_step = _predict_lr_step(local_history, row)
            elec_step = 0.70 * lr_elec_step + 0.30 * float(lstm_elec_s.iloc[pos])
            gas_step = 0.70 * lr_gas_step + 0.30 * float(lstm_gas_s.iloc[pos])
            row["electric_power"] = float(max(0.0, elec_step))
            row["gas_consumption"] = float(max(0.0, gas_step))
            row.name = ts_step
            local_history = pd.concat([local_history, row.to_frame().T])

            room_state = _predict_next_room(
                room_temp_value=room_state,
                duty_value=duty_value,
                supply_value=supply_value,
                outdoor_value=outdoor_value,
                lstm_anchor_room=float(lstm_room_s.iloc[pos]),
            )

            elec_path.append(float(max(0.0, elec_step)))
            gas_path.append(float(max(0.0, gas_step)))

        return (
            pd.Series(room_path, index=opt_res.index),
            pd.Series(elec_path, index=opt_res.index),
            pd.Series(gas_path, index=opt_res.index),
        )

    def _compute_plan_costs(pred_elec_plan: pd.Series, pred_gas_plan: pd.Series) -> tuple[float, float, float]:
        p_grid_plan = p_grid + pred_elec_plan.values
        elec_buy = np.where(p_grid_plan > 0, price_arr * p_grid_plan / 1000 * (TS_MIN / 60), 0.0)
        elec_sell = np.where(p_grid_plan < 0, sell_arr * p_grid_plan / 1000 * (TS_MIN / 60), 0.0)
        elec_cost = float(elec_buy.sum() + elec_sell.sum())
        gas_cost = float((pred_gas_plan.values * GAS_PRICE_EUR_M3).sum())
        return elec_cost + gas_cost, elec_cost, gas_cost

    def _optimize_hp_duty_for_supply(
        nudged_supply: pd.Series,
        target_min_s: pd.Series,
        target_max_s: pd.Series,
        use_temp_only_penalty: bool = False,
    ) -> pd.Series:
        """Re-optimize binary heat pump duty based on a fixed supply profile and comfort targets."""
        hp_duty = pd.Series(1, index=opt_res.index, dtype=int)
        room_state = float(scen_exog["room_temp"].iloc[0])
        last_switch_pos = -10_000
        switches_per_day: dict[pd.Timestamp, int] = {}
        prev_duty = 1

        for pos, ts_step in enumerate(opt_res.index):
            target_min_value = float(target_min_s.iloc[pos])
            target_max_value = float(target_max_s.iloc[pos])
            outdoor_value = float(outdoor_t.iloc[pos])
            local_day = ts_step.tz_convert(TZ_NL).normalize()
            day_switches = switches_per_day.get(local_day, 0)
            
            # Respect the nudged supply as target (or use baseline if no nudge)
            target_supply = float(nudged_supply.iloc[pos])

            # Evaluate ON state with target supply
            on_next_room = _predict_next_room(
                room_temp_value=room_state,
                duty_value=1,
                supply_value=target_supply,
                outdoor_value=outdoor_value,
                lstm_anchor_room=float(lstm_room_s.iloc[pos]),
            )
            if use_temp_only_penalty:
                on_penalty = _comfort_penalty_temp_only(
                    on_next_room,
                    TEMP_ONLY_TARGET_C,
                    target_min_value,
                    target_max_value,
                )
            else:
                on_penalty = _comfort_penalty(on_next_room, target_min_value, target_max_value)

            # Evaluate OFF state
            allows_switch = ((pos - last_switch_pos) >= MIN_SWITCH_GAP_SLOTS) and (day_switches < MAX_SWITCHES_PER_DAY)
            off_supply = _clip_supply(max(25.0, outdoor_value + 2.0), outdoor_value)
            off_next_room = _predict_next_room(
                room_temp_value=room_state,
                duty_value=0,
                supply_value=off_supply,
                outdoor_value=outdoor_value,
                lstm_anchor_room=float(lstm_room_s.iloc[pos]),
            )
            if use_temp_only_penalty:
                off_penalty = _comfort_penalty_temp_only(
                    off_next_room,
                    TEMP_ONLY_TARGET_C,
                    target_min_value,
                    target_max_value,
                )
                off_penalty += TEMP_ONLY_SWITCH_PENALTY if prev_duty != 0 else 0.0
            else:
                off_penalty = _comfort_penalty(off_next_room, target_min_value, target_max_value)
                off_penalty += 30.0

            # Choose best state
            duty_value = 1
            next_room = on_next_room
            if allows_switch and (off_penalty + 0.5 < on_penalty):
                duty_value = 0
                next_room = off_next_room
                if prev_duty != duty_value:
                    switches_per_day[local_day] = day_switches + 1
                    last_switch_pos = pos

            hp_duty.iloc[pos] = int(duty_value)
            room_state = float(next_room)
            prev_duty = int(duty_value)

        return hp_duty

    lookahead_slots_map: dict[float, int] = {
        lh: max(1, int(round(lh * 60.0 / TS_MIN))) for lh in LOOKAHEAD_HOURS
    }
    lookahead_signals: dict[float, np.ndarray] = {
        lh: _compute_price_signal(price_arr, lookahead_slots_map[lh]) for lh in LOOKAHEAD_HOURS
    }

    def _build_price_supply(
        weights: tuple[float, float, float],
        nudge_gain_c: float,
    ) -> tuple[pd.Series, pd.Series]:
        w6, w12, w24 = weights
        price_supply = comfort_supply_s.copy()
        weighted_signal_path: list[float] = []
        for pos, _ts_step in enumerate(opt_res.index):
            duty_value = int(comfort_duty_s.iloc[pos])
            s6 = float(lookahead_signals[6.0][pos])
            s12 = float(lookahead_signals[12.0][pos])
            s24 = float(lookahead_signals[24.0][pos])
            raw_weighted_signal = (w6 * s6) + (w12 * s12) + (w24 * s24)

            # Reduce nudging when lookaheads disagree strongly (signal conflict).
            abs_sum = abs(w6 * s6) + abs(w12 * s12) + abs(w24 * s24)
            alignment = abs(raw_weighted_signal) / max(1e-6, abs_sum)
            weighted_signal = float(raw_weighted_signal * alignment)
            weighted_signal_path.append(weighted_signal)

            if duty_value == 0:
                continue

            outdoor_value = float(outdoor_t.iloc[pos])
            room_value = float(lstm_room_s.iloc[pos])
            target_min_value = float(temp_min_hp.iloc[pos])
            target_max_value = float(temp_max_hp.iloc[pos])
            current_supply = float(price_supply.iloc[pos])

            if abs(weighted_signal) < NUDGE_DEADBAND:
                nudge = 0.0
            else:
                nudge = float(np.clip(nudge_gain_c * weighted_signal, -MAX_NUDGE_C, MAX_NUDGE_C))

            # Preserve comfort margin symmetrically around the band.
            if nudge < 0.0 and room_value <= target_min_value + 0.15:
                nudge = 0.0
            if nudge > 0.0 and room_value >= target_max_value - 0.15:
                nudge = 0.0

            new_supply = _clip_supply(current_supply + nudge, outdoor_value)
            if pos > 0:
                prev_supply = float(price_supply.iloc[pos - 1])
                new_supply = float(
                    np.clip(new_supply, prev_supply - MAX_SUPPLY_STEP_C, prev_supply + MAX_SUPPLY_STEP_C)
                )
            price_supply.iloc[pos] = _clip_supply(new_supply, outdoor_value)
        return price_supply, pd.Series(weighted_signal_path, index=opt_res.index)

    # Phase 1: comfort-first optimization.
    # Start from duty always ON and optimize only supply_temp first.
    print("\nPhase 1: temperature optimization with tight 20.25-20.75C band...")
    comfort_duty_s = pd.Series(1, index=opt_res.index, dtype=int)
    comfort_supply_s = pd.Series(index=opt_res.index, dtype=float)
    last_switch_pos = -10_000
    switches_per_day: dict[pd.Timestamp, int] = {}
    room_state = float(scen_exog["room_temp"].iloc[0])
    prev_duty = 1
    prev_supply = _baseline_supply(float(outdoor_t.iloc[0]))

    for pos, ts_step in enumerate(opt_res.index):
        target_min_value = float(temp_min_temp_only.iloc[pos])
        target_max_value = float(temp_max_temp_only.iloc[pos])
        outdoor_value = float(outdoor_t.iloc[pos])
        local_day = ts_step.tz_convert(TZ_NL).normalize()
        day_switches = switches_per_day.get(local_day, 0)

        baseline = _baseline_supply(outdoor_value)
        candidates = [
            _clip_supply(baseline + delta, outdoor_value)
            for delta in (-TEMP_ONLY_COOLDOWN_DELTA_C, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
        ]
        candidates = sorted(set(float(v) for v in candidates))
        cooldown_supply = _clip_supply(baseline - TEMP_ONLY_COOLDOWN_DELTA_C, outdoor_value)
        cooldown_next_room = _predict_next_room(
            room_temp_value=room_state,
            duty_value=1,
            supply_value=cooldown_supply,
            outdoor_value=outdoor_value,
            lstm_anchor_room=float(lstm_room_s.iloc[pos]),
        )
        cooldown_penalty = _comfort_penalty_temp_only(
            cooldown_next_room,
            TEMP_ONLY_TARGET_C,
            target_min_value,
            target_max_value,
        ) + 4.0 * abs(cooldown_supply - baseline)

        best_supply = baseline
        best_next_room = room_state
        best_penalty = float("inf")
        for supply_candidate in candidates:
            room_candidate = _predict_next_room(
                room_temp_value=room_state,
                duty_value=1,
                supply_value=supply_candidate,
                outdoor_value=outdoor_value,
                lstm_anchor_room=float(lstm_room_s.iloc[pos]),
            )
            penalty = _comfort_penalty_temp_only(
                room_candidate,
                TEMP_ONLY_TARGET_C,
                target_min_value,
                target_max_value,
            )
            penalty += 4.0 * abs(supply_candidate - baseline)
            if penalty < best_penalty:
                best_penalty = penalty
                best_supply = supply_candidate
                best_next_room = room_candidate

        # If supply-only cannot find an acceptable response, allow duty OFF fallback.
        allows_switch = ((pos - last_switch_pos) >= MIN_SWITCH_GAP_SLOTS) and (day_switches < MAX_SWITCHES_PER_DAY)
        off_supply = _clip_supply(max(25.0, outdoor_value + 2.0), outdoor_value)
        off_next_room = _predict_next_room(
            room_temp_value=room_state,
            duty_value=0,
            supply_value=off_supply,
            outdoor_value=outdoor_value,
            lstm_anchor_room=float(lstm_room_s.iloc[pos]),
        )
        off_penalty = _comfort_penalty_temp_only(
            off_next_room,
            TEMP_ONLY_TARGET_C,
            target_min_value,
            target_max_value,
        )
        off_penalty += TEMP_ONLY_SWITCH_PENALTY if prev_duty != 0 else 0.0

        duty_value = 1
        supply_value = best_supply
        next_room = best_next_room
        is_overheating = room_state > target_max_value
        if is_overheating:
            # First try to cool down with weather-curve minus 10C before switching OFF.
            duty_value = 1
            supply_value = cooldown_supply
            next_room = cooldown_next_room
            if allows_switch and (cooldown_next_room > target_max_value) and (off_penalty + 0.1 < cooldown_penalty):
                duty_value = 0
                supply_value = off_supply
                next_room = off_next_room
                if prev_duty != duty_value:
                    switches_per_day[local_day] = day_switches + 1
                    last_switch_pos = pos
        elif allows_switch and (off_penalty <= best_penalty):
            duty_value = 0
            supply_value = off_supply
            next_room = off_next_room
            if prev_duty != duty_value:
                switches_per_day[local_day] = day_switches + 1
                last_switch_pos = pos

        # Smooth final supply trajectory while preserving weather-curve bounds.
        supply_value = float(np.clip(supply_value, prev_supply - MAX_SUPPLY_STEP_C, prev_supply + MAX_SUPPLY_STEP_C))
        supply_value = _clip_supply(supply_value, outdoor_value)

        # Keep room-state propagation consistent with the final selected control.
        next_room = _predict_next_room(
            room_temp_value=room_state,
            duty_value=duty_value,
            supply_value=supply_value,
            outdoor_value=outdoor_value,
            lstm_anchor_room=float(lstm_room_s.iloc[pos]),
        )

        comfort_duty_s.iloc[pos] = int(duty_value)
        comfort_supply_s.iloc[pos] = float(supply_value)
        room_state = float(next_room)
        prev_duty = int(duty_value)
        prev_supply = float(supply_value)

    # Phase 1 smoothing + second temperature re-optimization pass.
    comfort_supply_s = _enforce_supply_limits(_smooth_control_series(comfort_supply_s))
    comfort_duty_s = _optimize_hp_duty_for_supply(
        comfort_supply_s,
        temp_min_temp_only,
        temp_max_temp_only,
        use_temp_only_penalty=True,
    )
    print("Phase 1 complete.")

    # Phase 2: price optimization with weighted multi-lookahead nudging (6h/12h/24h).
    # We tune the lookahead weights to maximize forecasted savings versus temperature-only.
    print("Phase 2: price optimization with 20-22C band...")
    room_temp_only_s, pred_elec_temp_s, pred_gas_temp_s = _simulate_plan(comfort_duty_s, comfort_supply_s)
    temp_total_cost, temp_elec_cost, temp_gas_cost = _compute_plan_costs(pred_elec_temp_s, pred_gas_temp_s)

    weight_candidates: list[tuple[float, float, float]] = []
    grid_vals = np.arange(0.0, 1.0 + 1e-9, WEIGHT_GRID_STEP)
    for w6 in grid_vals:
        for w12 in grid_vals:
            w24 = 1.0 - w6 - w12
            if w24 < -1e-9:
                continue
            if w24 > 1.0 + 1e-9:
                continue
            w24 = float(max(0.0, min(1.0, w24)))
            if abs((w6 + w12 + w24) - 1.0) > 1e-6:
                continue
            weight_candidates.append((float(w6), float(w12), float(w24)))

    gain_candidates = np.arange(NUDGE_GAIN_MIN_C, NUDGE_GAIN_MAX_C + 1e-9, NUDGE_GAIN_STEP_C)
    print(
        f"Phase 2 tuning candidates: {len(weight_candidates)} weight sets x {len(gain_candidates)} gains = "
        f"{len(weight_candidates) * len(gain_candidates)} combinations"
    )
    tuning_rows: list[tuple[float, float, float, float, float, float, float]] = []
    best_payload: tuple[
        float,
        tuple[float, float, float],
        float,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
        pd.Series,
    ] | None = None

    for weights in weight_candidates:
        for gain_c in gain_candidates:
            supply_candidate, weighted_signal_candidate = _build_price_supply(weights, float(gain_c))
            room_candidate, pred_elec_candidate, pred_gas_candidate = _simulate_plan(comfort_duty_s, supply_candidate)
            cand_total_cost, cand_elec_cost, cand_gas_cost = _compute_plan_costs(pred_elec_candidate, pred_gas_candidate)
            saving_vs_temp = temp_total_cost - cand_total_cost
            tuning_rows.append(
                (weights[0], weights[1], weights[2], float(gain_c), saving_vs_temp, cand_elec_cost, cand_gas_cost)
            )
            if (best_payload is None) or (saving_vs_temp > best_payload[0]):
                best_payload = (
                    saving_vs_temp,
                    weights,
                    float(gain_c),
                    supply_candidate,
                    weighted_signal_candidate,
                    room_candidate,
                    pred_elec_candidate,
                    pred_gas_candidate,
                )

    assert best_payload is not None
    (
        best_saving,
        best_weights,
        best_gain_c,
        price_supply_s,
        best_weighted_signal_s,
        room_opt_s,
        pred_elec_s,
        pred_gas_s,
    ) = best_payload
    w6_best, w12_best, w24_best = best_weights

    # Light post-lookahead pass: keep the tuned supply path, but re-evaluate
    # the binary HP duty once so the final plan can react to the nudged profile.
    hp_duty_s = _optimize_hp_duty_for_supply(price_supply_s, temp_min_hp, temp_max_hp)
    room_opt_s, pred_elec_s, pred_gas_s = _simulate_plan(hp_duty_s, price_supply_s)
    final_total_cost, final_elec_cost, final_gas_cost = _compute_plan_costs(pred_elec_s, pred_gas_s)
    final_saving_vs_temp = temp_total_cost - final_total_cost
    # Phase 2 smoothing + second temperature re-optimization pass.
    opt_supply_s = _enforce_supply_limits(_smooth_control_series(price_supply_s))
    hp_duty_s = _optimize_hp_duty_for_supply(opt_supply_s, temp_min_hp, temp_max_hp)
    room_opt_s, pred_elec_s, pred_gas_s = _simulate_plan(hp_duty_s, opt_supply_s)
    final_total_cost, final_elec_cost, final_gas_cost = _compute_plan_costs(pred_elec_s, pred_gas_s)
    final_saving_vs_temp = temp_total_cost - final_total_cost
    print(
        "\nTuning multi-lookahead nudging (6h/12h/24h): "
        f"beste gewichten = ({w6_best:.2f}, {w12_best:.2f}, {w24_best:.2f}), gain={best_gain_c:.2f}C"
    )
    print(
        "  Verwachte besparing t.o.v. temp-only: "
        f"EUR {best_saving:.2f} (elek+gas)"
    )
    tuning_df = pd.DataFrame(
        tuning_rows,
        columns=["w6", "w12", "w24", "gain_c", "saving_vs_temp_eur", "elec_cost_eur", "gas_cost_eur"],
    ).sort_values("saving_vs_temp_eur", ascending=False)
    print("  Top 5 gewichten:")
    for _, row in tuning_df.head(5).iterrows():
        print(
            "   - "
            f"w6={row['w6']:.2f}, w12={row['w12']:.2f}, w24={row['w24']:.2f}, gain={row['gain_c']:.2f}C "
            f"=> saving EUR {row['saving_vs_temp_eur']:.2f}"
        )
    print(
        "  Finale stap met duty-heroptimalisatie: "
        f"saving EUR {final_saving_vs_temp:.2f} (elek+gas)"
    )
    print("Phase 2 complete.")

    # Optional Phase 3: rerun standard optimizer on top of lookahead-informed HP electric load.
    post_opt_res = None
    if RUN_POST_STANDARD_OPT:
        print("\nPhase 3: standaard optimalisatie bovenop lookahead-HP-profiel...")
        p_load_post = (p_load.values + pred_elec_s.values).astype(float)
        df_input_post = df_input.copy()
        df_input_post["p_load_forecast"] = p_load_post
        post_opt_res_first = opt.perform_dayahead_forecast_optim(
            df_input_post,
            p_pv,
            pd.Series(p_load_post, index=df.index, name="p_load_forecast"),
        )
        # Smooth + re-optimize Phase 3 once more.
        p_load_post_smooth = _smooth_control_series(pd.Series(p_load_post, index=df.index)).values.astype(float)
        df_input_post_smooth = df_input.copy()
        df_input_post_smooth["p_load_forecast"] = p_load_post_smooth
        post_opt_res = opt.perform_dayahead_forecast_optim(
            df_input_post_smooth,
            p_pv,
            pd.Series(p_load_post_smooth, index=df.index, name="p_load_forecast"),
        )
        if post_opt_res is not None and not post_opt_res.empty:
            post_status = post_opt_res["optim_status"].iloc[0]
            post_total_cost = -post_opt_res["cost_profit"].sum()
            print(f"  Post-lookahead optim status: {post_status}")
            print(f"  Post-lookahead net cost (elec-only objective): EUR {post_total_cost:.2f}")
            if post_opt_res_first is not None and not post_opt_res_first.empty:
                post_total_cost_first = -post_opt_res_first["cost_profit"].sum()
                print(
                    "  Phase 3 smoothing impact: "
                    f"EUR {post_total_cost - post_total_cost_first:+.2f} vs first post-run"
                )
        else:
            print("  Post-lookahead standaard optimalisatie gaf geen resultaat terug.")

    print("Building final plot...")

    comfort_status = np.where(
        room_opt_s.values < temp_min_hp.values,
        "TOO_COLD",
        np.where(room_opt_s.values > temp_max_hp.values, "TOO_HOT", "COMFORTABLE"),
    )
    n_cold = int(np.sum(comfort_status == "TOO_COLD"))
    n_hot = int(np.sum(comfort_status == "TOO_HOT"))
    n_ok = int(np.sum(comfort_status == "COMFORTABLE"))
    n_switches = int(np.sum(np.abs(np.diff(hp_duty_s.values)) > 0))
    n_large_supply_jumps = int(np.sum(np.abs(np.diff(opt_supply_s.values)) > 2.0))
    print(f"\nBinary HP optimizer: {n_ok} slots COMFORTABLE, {n_cold} TOO_COLD, {n_hot} TOO_HOT")
    print(f"  heatpump_duty ON in {int(hp_duty_s.sum())} / {len(hp_duty_s)} slots")
    print(f"  duty switches: {n_switches} (max 1 per 2h, max 4/day)")
    print(f"  supply jumps >2C/slot: {n_large_supply_jumps}")
    print(f"  Opt supply_temp: mean {opt_supply_s.mean():.1f}C, range [{opt_supply_s.min():.1f}, {opt_supply_s.max():.1f}]")

    # Cost comparison: temperature-only vs temperature+price stages.
    p_grid_total_temp = p_grid + pred_elec_temp_s.values
    elec_buy_cost_temp = np.where(p_grid_total_temp > 0, price_arr * p_grid_total_temp / 1000 * (TS_MIN / 60), 0.0)
    elec_sell_inc_temp = np.where(p_grid_total_temp < 0, sell_arr * p_grid_total_temp / 1000 * (TS_MIN / 60), 0.0)
    gas_cost_temp_per_slot = pred_gas_temp_s.values * GAS_PRICE_EUR_M3
    elec_total_cost_temp = float(elec_buy_cost_temp.sum() + elec_sell_inc_temp.sum())
    gas_total_m3_temp = float(pred_gas_temp_s.sum())
    gas_total_cost_temp = float(gas_cost_temp_per_slot.sum())

    # Total net grid including the temperature+price optimized heat pump load.
    p_grid_total = p_grid + pred_elec_s.values
    elec_buy_cost = np.where(p_grid_total > 0, price_arr * p_grid_total / 1000 * (TS_MIN / 60), 0.0)
    elec_sell_inc = np.where(p_grid_total < 0, sell_arr * p_grid_total / 1000 * (TS_MIN / 60), 0.0)

    gas_cost_per_slot = gas_vals.values * GAS_PRICE_EUR_M3
    pred_gas_cost_per_slot = pred_gas_s.values * GAS_PRICE_EUR_M3
    gas_total_m3 = float(pred_gas_s.sum())
    gas_total_cost = float(pred_gas_cost_per_slot.sum())
    elec_total_cost = float(elec_buy_cost.sum() + elec_sell_inc.sum())

    def _add_subplot_legend(
        row_number: int,
        items: list[tuple[str, str, str]],
    ) -> None:
        subplot_ref = fig.get_subplot(row_number, 1)
        if subplot_ref is None or subplot_ref.yaxis is None or subplot_ref.yaxis.domain is None:
            return
        domain = subplot_ref.yaxis.domain
        y_top = float(domain[1]) - 0.004
        y_bottom = float(domain[0]) + 0.004
        gap = min(0.020, max(0.012, (y_top - y_bottom) / max(2.0, len(items) + 0.75)))

        fig.add_annotation(
            x=1.01,
            y=y_top,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            align="left",
            text="<b>Legenda</b>",
            font=dict(size=10, color="#111111"),
        )

        for idx_item, (sample, color, label) in enumerate(items, start=1):
            fig.add_annotation(
                x=1.01,
                y=y_top - (idx_item * gap),
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text=f"<span style='color:{color}'><b>{sample}</b></span> {label}",
                font=dict(size=9, color="#222222"),
            )

    print("\nVergelijking tussenstap vs prijsstap:")
    print(f"  Temp-only:   Elek EUR {elec_total_cost_temp:.2f} | Gas {gas_total_m3_temp:.2f} m3 (EUR {gas_total_cost_temp:.2f})")
    print(f"  Temp+prijs:  Elek EUR {elec_total_cost:.2f} | Gas {gas_total_m3:.2f} m3 (EUR {gas_total_cost:.2f})")
    print(f"  Delta prijsstap: Elek EUR {elec_total_cost - elec_total_cost_temp:+.2f} | Gas EUR {gas_total_cost - gas_total_cost_temp:+.2f}")

    temp_hp_elec_kwh = float((pred_elec_temp_s.values / 1000.0 * (TS_MIN / 60.0)).sum())
    final_hp_elec_kwh = float((pred_elec_s.values / 1000.0 * (TS_MIN / 60.0)).sum())
    comparison_df = pd.DataFrame(
        [
            {
                "scenario": "temp_only",
                "elec_cost_eur": elec_total_cost_temp,
                "gas_m3": gas_total_m3_temp,
                "gas_cost_eur": gas_total_cost_temp,
                "total_cost_eur": elec_total_cost_temp + gas_total_cost_temp,
                "hp_elec_kwh": temp_hp_elec_kwh,
            },
            {
                "scenario": "temp_plus_price",
                "elec_cost_eur": elec_total_cost,
                "gas_m3": gas_total_m3,
                "gas_cost_eur": gas_total_cost,
                "total_cost_eur": elec_total_cost + gas_total_cost,
                "hp_elec_kwh": final_hp_elec_kwh,
            },
            {
                "scenario": "delta_temp_plus_price_minus_temp_only",
                "elec_cost_eur": elec_total_cost - elec_total_cost_temp,
                "gas_m3": gas_total_m3 - gas_total_m3_temp,
                "gas_cost_eur": gas_total_cost - gas_total_cost_temp,
                "total_cost_eur": (elec_total_cost + gas_total_cost) - (elec_total_cost_temp + gas_total_cost_temp),
                "hp_elec_kwh": final_hp_elec_kwh - temp_hp_elec_kwh,
            },
        ]
    )
    OUTPUT_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_COMPARISON_CSV, index=False)
    print(f"  Vergelijkings-CSV: {OUTPUT_COMPARISON_CSV}")

    # Compute supply temperature bounds for visualization
    supply_min_bound = pd.Series(dtype=float, index=opt_res.index)
    supply_max_bound = pd.Series(dtype=float, index=opt_res.index)
    for pos, ts_step in enumerate(opt_res.index):
        outdoor_val = float(outdoor_t.iloc[pos])
        low, high = _curve_bounds(outdoor_val)
        supply_min_bound.iloc[pos] = low
        supply_max_bound.iloc[pos] = high

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=True,
        specs=[
            [{"secondary_y": False}],   # row 1: power + deferrable
            [{"secondary_y": False}],   # row 2: indoor comfort and room temperature
            [{"secondary_y": False}],   # row 3: outdoor + supply temperatures
            [{"secondary_y": False}],   # row 4: duty timelines
            [{"secondary_y": True}],    # row 5: price + cost/slot
            [{"secondary_y": False}],   # row 6: lookahead state timelines + weighted signal
            [{"secondary_y": True}],    # row 7: gas volume + gas cost/slot
        ],
        row_heights=[0.14, 0.14, 0.14, 0.11, 0.13, 0.13, 0.21],
        subplot_titles=[
            "Vermogen & uitgestelde verbruikers",
            "Binnentemperatuur & comfortband",
            "Aanvoer- en buitentemperatuur",
            "WP duty-staten (temp-only en temp+prijs)",
            "Elektriciteitsprijs & kosten per kwartier",
            "Lookahead staten (6h/12h/24h) + gewogen signaal",
            f"Gasverbruik & gaskosten (gasprijs EUR {GAS_PRICE_EUR_M3:.2f}/m3)",
        ],
        vertical_spacing=0.045,
    )

    # --- Row 1: Power balance + deferrable loads (merged) ---
    fig.add_trace(go.Scatter(
        x=idx_local, y=p_pv_r / 1000, name="PV productie (kW)",
        fill="tozeroy", line=dict(color="#f4b400", width=1.5),
        fillcolor="rgba(244,180,0,0.20)",
    ), row=1, col=1)

    net_grid_kw = p_grid_total / 1000
    fig.add_trace(go.Scatter(
        x=idx_local, y=net_grid_kw, name="Net afname incl. WP (kW)",
        line=dict(color="#e53935", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=p_load.values / 1000, name="Basisverbruik (kW)",
        line=dict(color="#78909c", width=1, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=p_ev / 1000, name="Auto EV (kW)",
        fill="tozeroy", line=dict(color="#1565c0", width=1.5),
        fillcolor="rgba(21,101,192,0.25)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=p_dish / 1000, name="Vaatwasser (kW)",
        fill="tozeroy", line=dict(color="#2e7d32", width=1.5),
        fillcolor="rgba(46,125,50,0.25)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=p_wash / 1000, name="Wasmachine (kW)",
        fill="tozeroy", line=dict(color="#6a1b9a", width=1.5),
        fillcolor="rgba(106,27,154,0.25)",
    ), row=1, col=1)

    # HP actual electric power (measured, from CSV)
    hp_elec_local = elec_hp.copy()
    hp_elec_local.index = hp_elec_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=hp_elec_local.values / 1000, name="WP elektriciteit gemeten (kW)",
        line=dict(color="#00897b", width=1.5),
    ), row=1, col=1)

    # HP optimized electric power from the binary duty planner
    pred_elec_temp_local = pred_elec_temp_s.copy()
    pred_elec_temp_local.index = pred_elec_temp_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=pred_elec_temp_local.values / 1000, name="WP elektriciteit temp-only (kW)",
        line=dict(color="#26a69a", width=1.2, dash="dash"),
    ), row=1, col=1)

    pred_elec_local = pred_elec_s.copy()
    pred_elec_local.index = pred_elec_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=pred_elec_local.values / 1000, name="WP elektriciteit temp+prijs (kW)",
        line=dict(color="#00897b", width=2, dash="dot"),
    ), row=1, col=1)

    # Deadline markers in row 1 only
    ev_deadline   = pd.Timestamp("2026-03-26 09:00", tz=TZ_NL)
    appl_deadline = pd.Timestamp("2026-03-26 16:00", tz=TZ_NL)
    for deadline, label, color in [
        (ev_deadline,   "EV deadline 09:00",        "#1565c0"),
        (appl_deadline, "Apparaten deadline 16:00", "#2e7d32"),
    ]:
        fig.add_vline(
            x=deadline.timestamp() * 1000, line_dash="dash",
            line_color=color, line_width=1.2, row=1, col=1,
        )

    # --- Row 2: indoor comfort and room temperature ---
    fig.add_trace(go.Scatter(
        x=idx_local, y=temp_min_temp_only.values, name="Temp-stap min 20.25°C",
        line=dict(color="#e65100", width=2),
        fill=None,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=temp_max_temp_only.values, name="Temp-stap max 20.75°C",
        line=dict(color="#ef6c00", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(230,81,0,0.12)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=temp_min_hp.values, name="Prijs-stap min 20.0°C",
        line=dict(color="#8e24aa", width=1.5, dash="dash"),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=temp_max_hp.values, name="Prijs-stap max 22.0°C",
        line=dict(color="#ab47bc", width=1.5, dash="dash"),
    ), row=2, col=1)

    room_temp_only_local = room_temp_only_s.copy()
    room_temp_only_local.index = room_temp_only_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=room_temp_only_local.values, name="Kamertemp. temp-only (°C)",
        line=dict(color="#1976d2", width=1.5, dash="dot"),
    ), row=2, col=1)

    room_opt_local = room_opt_s.copy()
    room_opt_local.index = room_opt_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=room_opt_local.values, name="Kamertemp. temp+prijs (°C)",
        line=dict(color="#5c6bc0", width=2, dash="dot"),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local,
        y=temp_min_hp.values + hp_duty_s.values * 0.15,
        name="WP duty aan (marker)",
        mode="markers",
        marker=dict(color="#1b5e20", size=5, symbol="square"),
    ), row=2, col=1)

    # --- Row 3: outdoor and supply temperatures with bounds ---
    outdoor_t_local = outdoor_t.copy()
    outdoor_t_local.index = outdoor_t_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=outdoor_t_local.values, name="Buitentemperatuur (°C)",
        line=dict(color="#0288d1", width=1.5),
    ), row=3, col=1)

    supply_min_local = supply_min_bound.copy()
    supply_min_local.index = supply_min_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=supply_min_local.values, name="Aanvoer min-limiet (°C)",
        line=dict(color="#757575", width=1, dash="dash"),
    ), row=3, col=1)

    supply_max_local = supply_max_bound.copy()
    supply_max_local.index = supply_max_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=supply_max_local.values, name="Aanvoer max-limiet (°C)",
        line=dict(color="#9e9e9e", width=1, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(200,200,200,0.08)",
    ), row=3, col=1)

    supply_t_local = supply_t.copy()
    supply_t_local.index = supply_t_local.index.tz_convert(TZ_NL)
    fig.add_trace(go.Scatter(
        x=idx_local, y=supply_t_local.values, name="Aanvoertemp. gemeten (°C)",
        line=dict(color="#00796b", width=1.5, dash="dash"),
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=comfort_supply_s.values, name="Aanvoertemp. temp-only (°C)",
        line=dict(color="#ff8f00", width=1.2, dash="dot"),
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=idx_local, y=opt_supply_s.values, name="Aanvoertemp. geoptimaliseerd (°C)",
        line=dict(color="#e65100", width=2.5),
    ), row=3, col=1)

    # --- Row 5: Electricity price (left y) + cost per slot (right y) ---
    fig.add_trace(go.Scatter(
        x=idx_local, y=df_input["unit_load_cost"].values,
        name="Inkoopprijs (€/kWh)",
        fill="tozeroy", line=dict(color="#880e4f", width=1.5),
        fillcolor="rgba(136,14,79,0.12)",
    ), row=5, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=idx_local, y=df_input["unit_prod_price"].values,
        name="Verkoopprijs (€/kWh)",
        line=dict(color="#ad1457", width=1, dash="dot"),
    ), row=5, col=1, secondary_y=False)

    bar_w_ms = 13 * 60 * 1000  # 13 min in milliseconds (bar width for 15-min slots)
    fig.add_trace(go.Bar(
        x=idx_local, y=elec_buy_cost,
        name="Inkoop €/kwartier",
        marker_color="rgba(183,28,28,0.50)",
        width=bar_w_ms,
    ), row=5, col=1, secondary_y=True)

    fig.add_trace(go.Bar(
        x=idx_local, y=elec_sell_inc,
        name="Verkoop €/kwartier",
        marker_color="rgba(27,94,32,0.55)",
        width=bar_w_ms,
    ), row=5, col=1, secondary_y=True)

    # --- Row 4: duty timelines ---
    sig_6h = lookahead_signals[6.0]
    sig_12h = lookahead_signals[12.0]
    sig_24h = lookahead_signals[24.0]

    def _duty_state(duty_val: float) -> str:
        return "aan" if float(duty_val) >= 0.5 else "uit"

    def _duty_color(duty_val: float) -> str:
        return "rgba(27,94,32,0.95)" if float(duty_val) >= 0.5 else "rgba(97,97,97,0.95)"

    def _signal_state(signal_val: float) -> str:
        if signal_val > NUDGE_DEADBAND:
            return "goedkoop"
        if signal_val < -NUDGE_DEADBAND:
            return "duur"
        return "neutraal"

    def _state_color(signal_val: float) -> str:
        state = _signal_state(float(signal_val))
        if state == "goedkoop":
            return "rgba(46,125,50,0.95)"
        if state == "duur":
            return "rgba(198,40,40,0.95)"
        return "rgba(255,152,0,0.95)"

    duty_timeline_specs = [
        ("WP duty temp-only", comfort_duty_s.values, 0.15, _duty_color, _duty_state, "Duty"),
        ("WP duty temp+prijs", hp_duty_s.values, 1.15, _duty_color, _duty_state, "Duty"),
    ]
    for name, signal_arr, base, color_fn, text_fn, value_label in duty_timeline_specs:
        fig.add_trace(go.Bar(
            x=idx_local,
            y=np.full(len(signal_arr), 0.70),
            base=base,
            name=name,
            marker_color=[color_fn(v) for v in signal_arr],
            width=bar_w_ms,
            customdata=signal_arr,
            hovertemplate=(
                f"<b>{name}</b><br>State: %{{text}}<br>{value_label}: %{{customdata:.2f}}<extra></extra>"
            ),
            text=[text_fn(v) for v in signal_arr],
        ), row=4, col=1)

    # --- Row 6: lookahead state timelines + weighted signal line ---
    lookahead_timeline_specs = [
        ("Lookahead 6h", sig_6h, 0.15, _state_color, _signal_state, "Signaal"),
        ("Lookahead 12h", sig_12h, 1.15, _state_color, _signal_state, "Signaal"),
        ("Lookahead 24h", sig_24h, 2.15, _state_color, _signal_state, "Signaal"),
    ]
    for name, signal_arr, base, color_fn, text_fn, value_label in lookahead_timeline_specs:
        fig.add_trace(go.Bar(
            x=idx_local,
            y=np.full(len(signal_arr), 0.70),
            base=base,
            name=name,
            marker_color=[color_fn(v) for v in signal_arr],
            width=bar_w_ms,
            customdata=signal_arr,
            hovertemplate=(
                f"<b>{name}</b><br>State: %{{text}}<br>{value_label}: %{{customdata:.2f}}<extra></extra>"
            ),
            text=[text_fn(v) for v in signal_arr],
        ), row=6, col=1)

    weighted_signal_plot = best_weighted_signal_s.reindex(opt_res.index).fillna(0.0).values
    signal_line_y = 3.35 + 0.35 * weighted_signal_plot
    fig.add_trace(go.Scatter(
        x=idx_local,
        y=signal_line_y,
        name="Gewogen signaal (tuned)",
        line=dict(color="#1e3a8a", width=2),
        mode="lines",
        customdata=weighted_signal_plot,
        hovertemplate="<b>Gewogen signaal</b>: %{customdata:.2f}<extra></extra>",
    ), row=6, col=1)

    # --- Row 7: Gas consumption (left y) + gas cost per slot (right y) ---
    fig.add_trace(go.Bar(
        x=idx_local, y=gas_vals.values,
        name="Gasverbruik gemeten (m3/kw.)",
        marker_color="rgba(255,143,0,0.65)",
        width=bar_w_ms,
    ), row=7, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=idx_local, y=pred_gas_temp_s.values,
        name="Gasverbruik temp-only (m3/kw.)",
        line=dict(color="#ff7043", width=1.2, dash="dash"),
    ), row=7, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=idx_local, y=pred_gas_s.values,
        name="Gasverbruik temp+prijs (m3/kw.)",
        line=dict(color="#bf360c", width=1.5, dash="dot"),
    ), row=7, col=1, secondary_y=False)

    fig.add_trace(go.Bar(
        x=idx_local, y=gas_cost_per_slot,
        name="Gaskosten gemeten (EUR/kw.)",
        marker_color="rgba(230,81,0,0.40)",
        width=bar_w_ms,
    ), row=7, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=idx_local, y=pred_gas_cost_per_slot,
        name="Gaskosten geoptimaliseerd (EUR/kw.)",
        line=dict(color="#bf360c", width=1, dash="dash"),
    ), row=7, col=1, secondary_y=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"EMHASS Optimalisatie — scenario 25-29 maart 2026<br>"
                f"<sup>EV: {ev_total_kwh:.1f} kWh | "
                f"Vaatwasser: {dish_total_kwh:.3f} kWh | "
                f"Wasmachine: {wash_total_kwh:.3f} kWh | "
                f"WP duty aan: {int(hp_duty_s.sum())} slots | "
                f"Beste nudge gain: {best_gain_c:.2f}C | "
                f"Elek temp-only: EUR {elec_total_cost_temp:.2f} | "
                f"Elek temp+prijs: EUR {elec_total_cost:.2f} | "
                f"Gasverbruik geoptimaliseerd: {gas_total_m3:.2f} m3 (EUR {gas_total_cost:.2f})</sup>"
            ),
            font=dict(size=16),
        ),
        hovermode="x unified",
        showlegend=True,
        height=1400,
        width=1550,
        template="plotly_white",
        barmode="overlay",
        margin=dict(r=120),
        legend=dict(
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=9),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="toggleitem",
            traceorder="grouped",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    fig.update_yaxes(title_text="kW",        row=1, col=1)
    fig.update_yaxes(title_text="Kamertemp. (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Temperatuur (degC)", row=3, col=1)
    fig.update_yaxes(
        title_text="Duty state",
        row=4,
        col=1,
        range=[0.0, 2.1],
        tickvals=[0.5, 1.5],
        ticktext=["Temp duty", "Prijs duty"],
    )
    fig.update_yaxes(title_text="EUR/kWh",   row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EUR/kw.",   row=5, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Lookahead state",
        row=6,
        col=1,
        range=[0.0, 4.1],
        tickvals=[0.5, 1.5, 2.5, 3.35],
        ticktext=["6h", "12h", "24h", "Signaal"],
    )
    fig.update_yaxes(title_text="m3/kw.",    row=7, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EUR/kw.",   row=7, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Lokale tijd (Amsterdam)", row=7, col=1)

    # Group clickable legend entries by subplot section.
    row_group_map = {
        "y": "R1 Vermogen",
        "y2": "R2 Binnencomfort",
        "y3": "R3 Aanvoer/Buiten",
        "y4": "R4 Duty",
        "y5": "R5 Prijs/Kosten",
        "y6": "R5 Prijs/Kosten",
        "y7": "R6 Lookahead",
        "y8": "R7 Gas",
        "y9": "R7 Gas",
    }
    seen_groups: set[str] = set()
    for trace in fig.data:
        yaxis_name = getattr(trace, "yaxis", "y")
        group_name = row_group_map.get(yaxis_name, "Overig")
        trace.legendgroup = group_name
        if group_name not in seen_groups:
            trace.legendgrouptitle = {"text": group_name}
            seen_groups.add(group_name)

    # Keep native Plotly legend enabled so traces remain clickable.

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs="cdn")
    print("HTML written.")
    if os.environ.get("EMHASS_SHOW_PLOT", "1") == "1":
        fig.show()
        print("\nGrafiek geopend in browser.")
    else:
        print("\nGrafiek opgeslagen zonder browser te openen.")
    print(f"  HTML rapport: {OUTPUT_HTML}")
    print(f"  Gas totaal: {gas_total_m3:.2f} m3 -> EUR {gas_total_cost:.2f} gaskosten")
    print(f"  Elektriciteit netto: EUR {elec_total_cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
