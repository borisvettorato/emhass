from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PRICE_CANDIDATE_COLUMNS = [
    "sensor.current_electricity_market_price",
    "unit_load_cost",
    "price",
    "electricity_price",
]


@dataclass
class PriceWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    label: str


def _find_price_column(df: pd.DataFrame) -> str:
    for col in PRICE_CANDIDATE_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "No known price column found. Checked: " + ", ".join(PRICE_CANDIDATE_COLUMNS)
    )


def _infer_timestep_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 15
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return 15
    return int(max(1, round(float(diffs.median()))))


def _label_prices(
    price_series: pd.Series,
    lookahead_slots: int,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    values = price_series.to_numpy(dtype=float)
    n = len(values)
    low_thr = np.full(n, np.nan)
    high_thr = np.full(n, np.nan)
    labels = np.full(n, "neutral", dtype=object)

    for i in range(n):
        j = min(n, i + lookahead_slots)
        window = values[i:j]
        if window.size == 0 or np.all(np.isnan(window)):
            labels[i] = "neutral"
            continue
        low = float(np.nanquantile(window, q_low))
        high = float(np.nanquantile(window, q_high))
        low_thr[i] = low
        high_thr[i] = high
        val = values[i]
        if np.isnan(val):
            labels[i] = "neutral"
        elif val <= low:
            labels[i] = "cheap"
        elif val >= high:
            labels[i] = "expensive"
        else:
            labels[i] = "neutral"

    return pd.DataFrame(
        {
            "price": values,
            "low_threshold": low_thr,
            "high_threshold": high_thr,
            "label": labels,
        },
        index=price_series.index,
    )


def _extract_windows(
    labels: pd.Series,
    min_slots: int,
) -> list[PriceWindow]:
    windows: list[PriceWindow] = []
    idx = labels.index
    current_label: str | None = None
    start_pos = 0

    for pos, label in enumerate(labels.to_numpy(dtype=object)):
        if label != current_label:
            if current_label in {"cheap", "expensive"}:
                length = pos - start_pos
                if length >= min_slots:
                    windows.append(
                        PriceWindow(
                            start=idx[start_pos],
                            end=idx[pos - 1],
                            label=str(current_label),
                        )
                    )
            current_label = str(label)
            start_pos = pos

    if current_label in {"cheap", "expensive"}:
        length = len(labels) - start_pos
        if length >= min_slots:
            windows.append(
                PriceWindow(
                    start=idx[start_pos],
                    end=idx[-1],
                    label=str(current_label),
                )
            )

    return windows


def _build_figure(
    df_labels: pd.DataFrame,
    windows: list[PriceWindow],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_labels.index,
            y=df_labels["price"],
            mode="lines",
            name="Price",
            line={"width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_labels.index,
            y=df_labels["low_threshold"],
            mode="lines",
            name="Low threshold",
            line={"dash": "dot", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_labels.index,
            y=df_labels["high_threshold"],
            mode="lines",
            name="High threshold",
            line={"dash": "dot", "width": 1},
        )
    )

    cheap = df_labels[df_labels["label"] == "cheap"]
    expensive = df_labels[df_labels["label"] == "expensive"]
    if not cheap.empty:
        fig.add_trace(
            go.Scatter(
                x=cheap.index,
                y=cheap["price"],
                mode="markers",
                name="Cheap points",
                marker={"symbol": "circle", "size": 6, "color": "green"},
            )
        )
    if not expensive.empty:
        fig.add_trace(
            go.Scatter(
                x=expensive.index,
                y=expensive["price"],
                mode="markers",
                name="Expensive points",
                marker={"symbol": "x", "size": 7, "color": "red"},
            )
        )

    for window in windows:
        if window.label == "cheap":
            color = "rgba(0, 180, 0, 0.12)"
        else:
            color = "rgba(220, 0, 0, 0.12)"
        fig.add_vrect(
            x0=window.start,
            x1=window.end,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Electricity price",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def _flags_from_windows(index: pd.DatetimeIndex, windows: list[PriceWindow]) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["is_cheap_window"] = False
    out["is_expensive_window"] = False
    for window in windows:
        mask = (out.index >= window.start) & (out.index <= window.end)
        if window.label == "cheap":
            out.loc[mask, "is_cheap_window"] = True
        elif window.label == "expensive":
            out.loc[mask, "is_expensive_window"] = True
    return out


def _build_compare_figure(
    compare_results: list[tuple[float, pd.DataFrame, list[PriceWindow]]],
    q_low: float,
    q_high: float,
) -> go.Figure:
    fig = make_subplots(
        rows=len(compare_results),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"lookahead={lookahead:.1f}h" for lookahead, _, _ in compare_results],
    )

    for row, (lookahead_h, df_labels, windows) in enumerate(compare_results, start=1):
        fig.add_trace(
            go.Scatter(
                x=df_labels.index,
                y=df_labels["price"],
                mode="lines",
                name="Price" if row == 1 else None,
                showlegend=(row == 1),
                line={"width": 2, "color": "#3b6db1"},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_labels.index,
                y=df_labels["low_threshold"],
                mode="lines",
                name="Low threshold" if row == 1 else None,
                showlegend=(row == 1),
                line={"dash": "dot", "width": 1, "color": "#2f9e44"},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_labels.index,
                y=df_labels["high_threshold"],
                mode="lines",
                name="High threshold" if row == 1 else None,
                showlegend=(row == 1),
                line={"dash": "dot", "width": 1, "color": "#d9480f"},
            ),
            row=row,
            col=1,
        )

        for window in windows:
            fill = "rgba(0,180,0,0.12)" if window.label == "cheap" else "rgba(220,0,0,0.12)"
            fig.add_vrect(
                x0=window.start,
                x1=window.end,
                fillcolor=fill,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )

    fig.update_layout(
        title=(
            "PoC price windows comparison"
            f" | q_low={q_low:.2f}"
            f" | q_high={q_high:.2f}"
        ),
        template="plotly_white",
        height=max(700, 320 * len(compare_results)),
        legend={"orientation": "h", "y": 1.02},
    )
    fig.update_xaxes(title_text="Time", row=len(compare_results), col=1)
    for row in range(1, len(compare_results) + 1):
        fig.update_yaxes(title_text="Price", row=row, col=1)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="PoC for cheap/expensive price windows")
    parser.add_argument(
        "--data-path",
        default="tests_thermal/data/test_data.csv",
        help="CSV with timestamp and price column",
    )
    parser.add_argument(
        "--output-html",
        default="logs/poc_price_windows.html",
        help="Output HTML chart path",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/poc_price_windows_labels.csv",
        help="Output labels CSV path",
    )
    parser.add_argument(
        "--lookahead-hours",
        type=float,
        default=24.0,
        help="Forward window length in hours for quantile thresholds",
    )
    parser.add_argument(
        "--cheap-quantile",
        type=float,
        default=0.30,
        help="Lower quantile used for cheap labeling",
    )
    parser.add_argument(
        "--expensive-quantile",
        type=float,
        default=0.70,
        help="Upper quantile used for expensive labeling",
    )
    parser.add_argument(
        "--min-window-slots",
        type=int,
        default=2,
        help="Minimum contiguous slots to keep a cheap/expensive window",
    )
    parser.add_argument(
        "--compare-lookaheads",
        default="",
        help="Comma-separated lookahead hours for comparison mode, e.g. '6,12,24'",
    )
    parser.add_argument(
        "--compare-output-html",
        default="logs/poc_price_windows_compare.html",
        help="Output HTML path for comparison figure",
    )
    parser.add_argument(
        "--compare-output-csv",
        default="logs/poc_price_windows_compare_labels.csv",
        help="Output CSV path for comparison labels",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    price_col = _find_price_column(df)
    price_raw = pd.to_numeric(df[price_col], errors="coerce")

    price_for_threshold = price_raw.interpolate(limit_direction="both")
    timestep_min = _infer_timestep_minutes(price_for_threshold.index)
    if args.compare_lookaheads.strip():
        compare_hours = [float(x.strip()) for x in args.compare_lookaheads.split(",") if x.strip()]
        compare_results: list[tuple[float, pd.DataFrame, list[PriceWindow]]] = []
        merged_export = pd.DataFrame(index=price_for_threshold.index)
        merged_export["price"] = price_raw

        for lookahead_h in compare_hours:
            lookahead_slots = max(1, int(round(lookahead_h * 60.0 / timestep_min)))
            df_labels = _label_prices(
                price_series=price_for_threshold,
                lookahead_slots=lookahead_slots,
                q_low=args.cheap_quantile,
                q_high=args.expensive_quantile,
            )
            df_labels["price"] = price_raw
            windows = _extract_windows(df_labels["label"], min_slots=max(1, args.min_window_slots))
            compare_results.append((lookahead_h, df_labels, windows))

            suffix = str(int(lookahead_h)) if float(lookahead_h).is_integer() else str(lookahead_h).replace(".", "p")
            merged_export[f"label_{suffix}h"] = df_labels["label"]
            merged_export[f"low_threshold_{suffix}h"] = df_labels["low_threshold"]
            merged_export[f"high_threshold_{suffix}h"] = df_labels["high_threshold"]
            flags = _flags_from_windows(df_labels.index, windows)
            merged_export[f"is_cheap_window_{suffix}h"] = flags["is_cheap_window"]
            merged_export[f"is_expensive_window_{suffix}h"] = flags["is_expensive_window"]

            cheap_windows = sum(1 for w in windows if w.label == "cheap")
            expensive_windows = sum(1 for w in windows if w.label == "expensive")
            print(
                f"lookahead={lookahead_h:.1f}h | slots={lookahead_slots} | "
                f"cheap_windows={cheap_windows} | expensive_windows={expensive_windows}"
            )

        out_html = Path(args.compare_output_html)
        out_csv = Path(args.compare_output_csv)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fig = _build_compare_figure(
            compare_results=compare_results,
            q_low=args.cheap_quantile,
            q_high=args.expensive_quantile,
        )
        fig.write_html(out_html, include_plotlyjs="cdn")
        merged_export.to_csv(out_csv, index_label="timestamp")

        print(f"Price column: {price_col}")
        print(f"Timestep inferred: {timestep_min} min")
        print(f"Compare labels CSV: {out_csv}")
        print(f"Compare chart HTML: {out_html}")
        return

    lookahead_slots = max(1, int(round(args.lookahead_hours * 60.0 / timestep_min)))

    df_labels = _label_prices(
        price_series=price_for_threshold,
        lookahead_slots=lookahead_slots,
        q_low=args.cheap_quantile,
        q_high=args.expensive_quantile,
    )

    # Keep original prices for plotting and export; thresholds and labels come from filled data.
    df_labels["price"] = price_raw
    windows = _extract_windows(df_labels["label"], min_slots=max(1, args.min_window_slots))

    out_html = Path(args.output_html)
    out_csv = Path(args.output_csv)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fig = _build_figure(
        df_labels=df_labels,
        windows=windows,
        title=(
            "PoC price windows"
            f" | lookahead={args.lookahead_hours}h"
            f" | q_low={args.cheap_quantile:.2f}"
            f" | q_high={args.expensive_quantile:.2f}"
        ),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")

    export_df = df_labels.copy()
    flags = _flags_from_windows(export_df.index, windows)
    export_df["is_cheap_window"] = flags["is_cheap_window"]
    export_df["is_expensive_window"] = flags["is_expensive_window"]
    export_df.to_csv(out_csv, index_label="timestamp")

    cheap_windows = sum(1 for w in windows if w.label == "cheap")
    expensive_windows = sum(1 for w in windows if w.label == "expensive")
    print(f"Price column: {price_col}")
    print(f"Timestep inferred: {timestep_min} min")
    print(f"Lookahead slots: {lookahead_slots}")
    print(f"Cheap windows: {cheap_windows}")
    print(f"Expensive windows: {expensive_windows}")
    print(f"Labels CSV: {out_csv}")
    print(f"Chart HTML: {out_html}")


if __name__ == "__main__":
    main()
