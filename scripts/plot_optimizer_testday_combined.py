"""Combine thermal-mass MPC and CVXPY optimizer dry-run plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _comfort(room: np.ndarray, *, min_temp: float, max_temp: float, dt_h: float) -> dict[str, float]:
    below = np.clip(min_temp - room, 0.0, None)
    above = np.clip(room - max_temp, 0.0, None)
    return {
        "comfort_degree_h": float(np.sum(below + above) * dt_h),
        "comfort_sq_degree_h": float(np.sum(below * below + above * above) * dt_h),
        "min_room_temp": float(np.min(room)),
        "max_room_temp": float(np.max(room)),
        "mean_room_temp": float(np.mean(room)),
    }


def _series_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
    }


def _read_plan(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df.add_prefix(prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot both optimizer testday plans in one figure")
    parser.add_argument("--thermal-plan", default="tests_thermal/reports/optimizer_testday_20260520/thermal_mass_mpc/thermal_mass_optimization_plan.csv")
    parser.add_argument("--cvxpy-plan", default="tests_thermal/reports/optimizer_testday_20260520/cvxpy_state_space/cvxpy_state_space_optimization_plan.csv")
    parser.add_argument("--output-dir", default="tests_thermal/reports/optimizer_testday_20260520")
    parser.add_argument("--min-room-temp", type=float, default=20.0)
    parser.add_argument("--max-room-temp", type=float, default=22.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thermal = _read_plan(Path(args.thermal_plan), "thermal_")
    cvx = _read_plan(Path(args.cvxpy_plan), "cvx_")
    df = thermal.join(cvx, how="inner")
    if df.empty:
        raise ValueError("No overlapping timestamps between optimizer plans")

    if len(df) > 1:
        dt_h = float(df.index.to_series().diff().dropna().dt.total_seconds().median() / 3600.0)
    else:
        dt_h = 0.25

    price = df["thermal_electricity_price"].to_numpy(dtype=float)
    thermal_power = df["thermal_optimized_electric_power"].to_numpy(dtype=float)
    cvx_power = df["cvx_optimized_p_hp"].to_numpy(dtype=float)
    thermal_kwh_step = thermal_power / 1000.0 * dt_h
    cvx_kwh_step = cvx_power / 1000.0 * dt_h
    thermal_cost_step = thermal_kwh_step * price
    cvx_cost_step = cvx_kwh_step * price

    diff_df = pd.DataFrame(
        {
            "timestamp": df.index,
            "actual_room_temp": df["thermal_actual_room_temp"].to_numpy(dtype=float),
            "thermal_room_temp": df["thermal_optimized_room_temp"].to_numpy(dtype=float),
            "cvxpy_room_temp": df["cvx_optimized_room_temp"].to_numpy(dtype=float),
            "room_diff_cvx_minus_thermal": df["cvx_optimized_room_temp"].to_numpy(dtype=float) - df["thermal_optimized_room_temp"].to_numpy(dtype=float),
            "thermal_supply_temp": df["thermal_optimized_supply_temp"].to_numpy(dtype=float),
            "cvxpy_supply_temp": df["cvx_optimized_supply_temp"].to_numpy(dtype=float),
            "supply_diff_cvx_minus_thermal": df["cvx_optimized_supply_temp"].to_numpy(dtype=float) - df["thermal_optimized_supply_temp"].to_numpy(dtype=float),
            "thermal_power_w": thermal_power,
            "cvxpy_power_w": cvx_power,
            "power_diff_cvx_minus_thermal_w": cvx_power - thermal_power,
            "thermal_heatpump_duty": df["thermal_optimized_heatpump_duty"].to_numpy(dtype=float),
            "cvxpy_q_hp": df["cvx_optimized_q_hp"].to_numpy(dtype=float),
            "cvxpy_curve_offset": df["cvx_optimized_curve_offset"].to_numpy(dtype=float),
            "electricity_price": price,
            "thermal_cum_kwh": np.cumsum(thermal_kwh_step),
            "cvxpy_cum_kwh": np.cumsum(cvx_kwh_step),
            "thermal_cum_cost_eur": np.cumsum(thermal_cost_step),
            "cvxpy_cum_cost_eur": np.cumsum(cvx_cost_step),
        }
    )
    diff_path = output_dir / "optimizer_testday_combined_diff.csv"
    diff_df.to_csv(diff_path, index=False)

    thermal_room = df["thermal_optimized_room_temp"].to_numpy(dtype=float)
    cvx_room = df["cvx_optimized_room_temp"].to_numpy(dtype=float)
    summary = {
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "n_steps": int(len(df)),
        "dt_h": dt_h,
        "thermal_mass_mpc": {
            **_comfort(thermal_room, min_temp=args.min_room_temp, max_temp=args.max_room_temp, dt_h=dt_h),
            "electric_kwh": float(np.sum(thermal_kwh_step)),
            "electric_cost_eur": float(np.sum(thermal_cost_step)),
            "mean_power_w": float(np.mean(thermal_power)),
            "mean_supply_temp": float(df["thermal_optimized_supply_temp"].mean()),
            "mean_duty": float(df["thermal_optimized_heatpump_duty"].mean()),
        },
        "cvxpy_state_space": {
            **_comfort(cvx_room, min_temp=args.min_room_temp, max_temp=args.max_room_temp, dt_h=dt_h),
            "electric_kwh": float(np.sum(cvx_kwh_step)),
            "electric_cost_eur": float(np.sum(cvx_cost_step)),
            "mean_power_w": float(np.mean(cvx_power)),
            "mean_supply_temp": float(df["cvx_optimized_supply_temp"].mean()),
            "mean_curve_offset": float(df["cvx_optimized_curve_offset"].mean()),
        },
        "differences_cvx_minus_thermal": {
            "room_temp_c": _series_summary(cvx_room - thermal_room),
            "supply_temp_c": _series_summary(df["cvx_optimized_supply_temp"].to_numpy(dtype=float) - df["thermal_optimized_supply_temp"].to_numpy(dtype=float)),
            "power_w": _series_summary(cvx_power - thermal_power),
            "electric_kwh": float(np.sum(cvx_kwh_step) - np.sum(thermal_kwh_step)),
            "electric_cost_eur": float(np.sum(cvx_cost_step) - np.sum(thermal_cost_step)),
            "comfort_degree_h": float(
                _comfort(cvx_room, min_temp=args.min_room_temp, max_temp=args.max_room_temp, dt_h=dt_h)["comfort_degree_h"]
                - _comfort(thermal_room, min_temp=args.min_room_temp, max_temp=args.max_room_temp, dt_h=dt_h)["comfort_degree_h"]
            ),
        },
    }
    summary_path = output_dir / "optimizer_testday_combined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = df.index.tz_convert("Europe/Amsterdam")
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Room temperature",
            "Room difference: CVXPY minus ThermalMass",
            "Supply, curve offset and duty",
            "Heat pump electric power",
            "Cumulative kWh and cost",
            "Price, outdoor and solar",
        ),
    )
    fig.add_trace(go.Scatter(x=x, y=df["thermal_actual_room_temp"], mode="lines", name="Actual room", line=dict(color="#111827", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["thermal_optimized_room_temp"], mode="lines", name="ThermalMass + MPC room", line=dict(color="#0f766e", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["cvx_optimized_room_temp"], mode="lines", name="CVXPY room", line=dict(color="#2563eb", width=2.0)), row=1, col=1)
    fig.add_hline(y=args.min_room_temp, line_width=1, line_color="#94a3b8", row=1, col=1)
    fig.add_hline(y=args.max_room_temp, line_width=1, line_color="#94a3b8", row=1, col=1)

    room_diff = df["cvx_optimized_room_temp"] - df["thermal_optimized_room_temp"]
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8", row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=room_diff, mode="lines", name="Room diff", line=dict(color="#dc2626", width=1.6), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["thermal_optimized_supply_temp"], mode="lines", name="ThermalMass supply", line=dict(color="#ea580c", width=1.5)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["cvx_optimized_supply_temp"], mode="lines", name="CVXPY supply", line=dict(color="#dc2626", width=1.5)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["cvx_optimized_curve_offset"], mode="lines", name="CVXPY curve offset", line=dict(color="#7c3aed", width=1.2)), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["thermal_optimized_heatpump_duty"], mode="lines", name="ThermalMass duty", line=dict(color="#64748b", width=1.0, dash="dot")), row=3, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=thermal_power, mode="lines", name="ThermalMass electric", line=dict(color="#0f766e", width=1.4)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=cvx_power, mode="lines", name="CVXPY electric", line=dict(color="#2563eb", width=1.4)), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=cvx_power - thermal_power, name="Power diff CVX-TM", marker_color="rgba(220,38,38,0.28)"), row=4, col=1)

    fig.add_trace(go.Scatter(x=x, y=np.cumsum(thermal_kwh_step), mode="lines", name="ThermalMass cum kWh", line=dict(color="#0f766e", width=1.5)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(cvx_kwh_step), mode="lines", name="CVXPY cum kWh", line=dict(color="#2563eb", width=1.5)), row=5, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(thermal_cost_step), mode="lines", name="ThermalMass cum EUR", line=dict(color="#0f766e", width=1.2, dash="dot")), row=5, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=np.cumsum(cvx_cost_step), mode="lines", name="CVXPY cum EUR", line=dict(color="#2563eb", width=1.2, dash="dot")), row=5, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=price, mode="lines", name="Price", line=dict(color="#7c3aed", width=1.2)), row=6, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df["thermal_outdoor_temp"], mode="lines", name="Outdoor", line=dict(color="#0284c7", width=1.0)), row=6, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=df["thermal_q_solar"], mode="lines", name="Q solar", line=dict(color="#eab308", width=1.0)), row=6, col=1, secondary_y=True)

    diff = summary["differences_cvx_minus_thermal"]
    annotation = (
        f"CVXPY vs ThermalMass: room mean abs diff {diff['room_temp_c']['mean_abs']:.3f} C, "
        f"max {diff['room_temp_c']['max_abs']:.3f} C<br>"
        f"supply mean abs diff {diff['supply_temp_c']['mean_abs']:.2f} C; "
        f"energy {diff['electric_kwh']:+.2f} kWh; cost {diff['electric_cost_eur']:+.2f} EUR"
    )
    fig.add_annotation(text=annotation, xref="paper", yref="paper", x=0.01, y=0.995, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.88)")
    fig.update_layout(
        title=f"Optimizer testday comparison ({x.min().strftime('%Y-%m-%d %H:%M')} - {x.max().strftime('%Y-%m-%d %H:%M')} Europe/Amsterdam)",
        template="plotly_white",
        height=1500,
        width=1550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        barmode="relative",
    )
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="degC", row=2, col=1)
    fig.update_yaxes(title_text="supply degC", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="offset / duty", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="W", row=4, col=1)
    fig.update_yaxes(title_text="kWh", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="EUR", row=5, col=1, secondary_y=True)
    fig.update_yaxes(title_text="EUR/kWh", row=6, col=1, secondary_y=False)
    fig.update_yaxes(title_text="weather", row=6, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Local time", row=6, col=1)
    output_path = output_dir / "optimizer_testday_combined_plot.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(json.dumps({"plot_path": str(output_path), "diff_path": str(diff_path), "summary_path": str(summary_path), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
