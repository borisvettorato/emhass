"""
Integration tests for thermal optimization using real CSV data and model inference.
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch
from plotly import graph_objects as go

from emhass.thermal import HeatPumpOptimizer

logger = logging.getLogger(__name__)

# Acceptance criteria for rollout readiness checks.
ACCEPT_MAX_VIOLATION_RATE = 0.02
ACCEPT_MAX_DAILY_NEUTRAL_COST_DEGRADATION = 1e-6
ACCEPT_MIN_EVAL_DAYS = 1


class TestHeatPumpOptimizerIntegration:
    """Integration coverage with real data blocks and KPI-based acceptance checks."""

    @staticmethod
    def _last_horizon(series: np.ndarray, horizon: int = 144, fill_value: float = 0.0) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        if len(values) >= horizon:
            return values[-horizon:]
        padded = np.full(horizon, fill_value, dtype=float)
        padded[-len(values):] = values
        return padded

    @staticmethod
    def _daily_windows(df: pd.DataFrame, min_rows: int = 144) -> list[tuple[str, pd.DataFrame]]:
        # Build one 144-row rolling window per day, anchored at each day end.
        # This avoids false skips with 15-min data (typically 96 rows/day).
        windows: list[tuple[str, pd.DataFrame]] = []
        if len(df) < min_rows:
            return windows

        df_sorted = df.sort_index()
        for day, grp in df_sorted.groupby(df_sorted.index.date):
            end_ts = grp.index[-1]
            end_pos = df_sorted.index.get_loc(end_ts)
            if isinstance(end_pos, slice):
                end_pos = end_pos.stop - 1
            if end_pos + 1 >= min_rows:
                window = df_sorted.iloc[end_pos - min_rows + 1 : end_pos + 1]
                windows.append((str(day), window))
        return windows

    @staticmethod
    def _write_report_csv(df_report: pd.DataFrame, report_path: Path) -> Path:
        """Write report CSV and fall back to timestamped filename if file is locked."""
        try:
            df_report.to_csv(report_path, index=False)
            return report_path
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = report_path.with_name(f"{report_path.stem}_{ts}.csv")
            df_report.to_csv(fallback_path, index=False)
            return fallback_path

    @staticmethod
    def _room_temp_to_celsius(
        room_temp_raw: np.ndarray,
        room_temp_mean: float,
        room_temp_scale: float,
    ) -> np.ndarray:
        """
        Convert model room temperature output to Celsius when needed.

        Model checkpoints can output normalized values around 0.
        If values already look like Celsius, keep them unchanged.
        """
        vals = np.asarray(room_temp_raw, dtype=float)
        mean_val = float(np.nanmean(vals))
        std_val = float(np.nanstd(vals))

        looks_like_celsius = 5.0 <= mean_val <= 35.0 and std_val < 15.0
        if looks_like_celsius:
            return vals

        return vals * room_temp_scale + room_temp_mean

    @pytest.fixture
    def setup(self, load_real_data, pinn_model, sample_input_data_real):
        input_size = pinn_model.lstm.input_size
        df = load_real_data["df"]
        scaler_y = load_real_data["scaler_y"]

        outdoor_real = self._last_horizon(df["outdoor_temp"].values, 144, fill_value=10.0)
        supply_old_real = self._last_horizon(df["supply_temp"].values, 144, fill_value=30.0)
        indoor_old_real = self._last_horizon(df["room_temp"].values, 144, fill_value=20.0)

        price_col = "sensor.current_electricity_market_price"
        if price_col in df.columns:
            price_real = self._last_horizon(df[price_col].values, 144, fill_value=0.0)
        else:
            price_real = np.zeros(144, dtype=float)

        optimizer = HeatPumpOptimizer(
            curve_intercept=40.0,
            curve_slope=-1.0,
            max_deviation=10.0,
            min_supply_temp=10.0,
            max_supply_temp=60.0,
        )

        return {
            "data": load_real_data,
            "model": pinn_model,
            "sample_input": sample_input_data_real[:, :input_size],
            "optimizer": optimizer,
            "outdoor_real": outdoor_real,
            "supply_old_real": supply_old_real,
            "indoor_old_real": indoor_old_real,
            "price_real": price_real,
            "price_col": price_col,
            "room_temp_mean": float(scaler_y.mean_[0]),
            "room_temp_scale": float(scaler_y.scale_[0]),
        }

    def test_integration_single_horizon(self, setup):
        X_test = torch.tensor(setup["sample_input"][None, :, :], dtype=torch.float32)
        with torch.no_grad():
            output = setup["model"](X_test)

        room_temps = self._room_temp_to_celsius(
            output["q50"].numpy()[0, :, 0],
            setup["room_temp_mean"],
            setup["room_temp_scale"],
        )
        outdoor_temps = setup["outdoor_real"]

        result = setup["optimizer"].get_optimal_setpoint(room_temps, outdoor_temps)

        assert result["setpoint_optimal"].shape == (144,)
        assert np.all(result["setpoint_optimal"] >= 10.0)
        assert np.all(result["setpoint_optimal"] <= 60.0)

    def test_plot_contains_old_new_outdoor_price(self, setup):
        X_test = torch.tensor(setup["sample_input"][None, :, :], dtype=torch.float32)
        with torch.no_grad():
            output = setup["model"](X_test)

        room_temps = self._room_temp_to_celsius(
            output["q50"].numpy()[0, :, 0],
            setup["room_temp_mean"],
            setup["room_temp_scale"],
        )
        outdoor_temps = setup["outdoor_real"]
        supply_old = setup["supply_old_real"]
        indoor_old = setup["indoor_old_real"]
        prices = setup["price_real"]
        hours = np.arange(144) * 0.25

        neutral = setup["optimizer"].get_optimal_setpoint(room_temps, outdoor_temps)
        price_aware = setup["optimizer"].get_price_aware_setpoint(
            room_temp_forecast=room_temps,
            outdoor_temp_forecast=outdoor_temps,
            price_forecast=prices,
            price_weight=0.8,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=supply_old, mode="lines", name="Old HP supply temp"))
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=neutral["setpoint_optimal"],
                mode="lines",
                name="New HP setpoint (neutral)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=price_aware["setpoint_price_aware"],
                mode="lines",
                name="New HP setpoint (price-aware)",
            )
        )
        fig.add_trace(go.Scatter(x=hours, y=outdoor_temps, mode="lines", name="Outdoor temp"))
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=indoor_old,
                mode="lines",
                name="Indoor temp (old, measured)",
                line=dict(color="orange", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=room_temps,
                mode="lines",
                name="Indoor temp (new, LSTM forecast)",
                line=dict(color="darkorange", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=prices,
                mode="lines",
                name="Electricity price",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Heat Pump Optimization - old vs new with outdoor temp and price",
            yaxis_title="Temperature (C)",
            yaxis2=dict(title="Electricity price", overlaying="y", side="right", showgrid=False),
            template="plotly_white",
            height=650,
            width=1300,
        )

        output_path = Path(__file__).parent / "plots" / "test_optimization.html"
        output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(output_path))

        assert output_path.exists()

    def test_daily_backtest_acceptance_criteria(self, setup):
        """
        Step 4 implementation: evaluate per-day windows (real data blocks)
        instead of validating only one horizon.
        """
        df = setup["data"]["df"]
        report_path = Path(__file__).parent / "reports" / "daily_backtest_kpi.csv"
        report_path.parent.mkdir(exist_ok=True)
        windows = self._daily_windows(df, min_rows=144)

        if not windows:
            saved_path = self._write_report_csv(
                pd.DataFrame(
                [
                    {
                        "status": "skipped_not_enough_history",
                        "required_rows_per_window": 144,
                        "available_rows_total": len(df),
                    }
                ]
                ),
                report_path,
            )
            logger.info(f"Saved daily KPI report: {saved_path}")
            pytest.skip("Not enough history yet: need at least one full 144-row daily window")

        evaluated_days = 0
        violation_rates: list[float] = []
        daily_cost_deltas: list[float] = []

        daily_rows: list[dict[str, float | str]] = []

        for day_label, day_df in windows:
            # Use the most recent model input if we cannot fully reconstruct
            # each day's engineered input shape from raw data.
            X_test = torch.tensor(setup["sample_input"][None, :, :], dtype=torch.float32)
            with torch.no_grad():
                output = setup["model"](X_test)

            room_temps = self._room_temp_to_celsius(
                output["q50"].numpy()[0, :, 0],
                setup["room_temp_mean"],
                setup["room_temp_scale"],
            )
            outdoor = day_df["outdoor_temp"].values[-144:]

            if setup["price_col"] in day_df.columns:
                prices = day_df[setup["price_col"]].values[-144:]
            else:
                prices = np.zeros(144, dtype=float)

            neutral = setup["optimizer"].get_optimal_setpoint(room_temps, outdoor)
            price_aware = setup["optimizer"].get_price_aware_setpoint(
                room_temp_forecast=room_temps,
                outdoor_temp_forecast=outdoor,
                price_forecast=prices,
                price_weight=0.8,
            )

            # Acceptance criterion #1: constraints remain safe.
            violations = setup["optimizer"].get_constraint_violations(
                price_aware["setpoint_price_aware"],
                outdoor,
            )
            total_violations = np.zeros(144, dtype=bool)
            for arr in violations.values():
                total_violations = total_violations | arr
            violation_rate = float(np.mean(total_violations))
            violation_rates.append(violation_rate)

            # Acceptance criterion #2: price-aware should not be worse than neutral.
            neutral_cost_proxy = float(np.sum(neutral["setpoint_optimal"] * prices))
            price_aware_cost_proxy = float(np.sum(price_aware["setpoint_price_aware"] * prices))
            cost_delta = price_aware_cost_proxy - neutral_cost_proxy
            daily_cost_deltas.append(cost_delta)

            daily_rows.append(
                {
                    "day": day_label,
                    "violation_rate": violation_rate,
                    "neutral_cost_proxy": neutral_cost_proxy,
                    "price_aware_cost_proxy": price_aware_cost_proxy,
                    "cost_delta": cost_delta,
                }
            )

            evaluated_days += 1

        saved_path = self._write_report_csv(pd.DataFrame(daily_rows), report_path)

        assert evaluated_days >= ACCEPT_MIN_EVAL_DAYS
        assert float(np.mean(violation_rates)) <= ACCEPT_MAX_VIOLATION_RATE
        assert float(np.mean(daily_cost_deltas)) <= ACCEPT_MAX_DAILY_NEUTRAL_COST_DEGRADATION
        assert saved_path.exists()

        logger.info("Daily backtest acceptance criteria passed")
        logger.info(f"Days evaluated: {evaluated_days}")
        logger.info(f"Mean violation rate: {float(np.mean(violation_rates)):.4f}")
        logger.info(f"Mean cost delta vs neutral: {float(np.mean(daily_cost_deltas)):.6f}")
        logger.info(f"Saved daily KPI report: {saved_path}")
