from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = (
    r"C:\Users\Boris\OneDrive\Documenten\Klussen\Huizen\Moeders - Hoefslag 17"
    r"\Verbruiksdata\Open meteo opvuldata.csv"
)
DEFAULT_MEASUREMENT = "W/m\u00b2"
SOLAR_ENTITY_COLUMNS = {
    "solar_dhi": "dhi",
    "solar_dni": "dni",
    "solar_ghi": "ghi",
}


def load_ha_import_module():
    module_path = PROJECT_ROOT / "home assistant data import.py"
    spec = importlib.util.spec_from_file_location("ha_import", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_column_name(name: str) -> str:
    return (
        name.lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\u00b2", "2")
        .replace("\u00c2", "")
    )


def find_column(df: pd.DataFrame, required_tokens: tuple[str, ...]) -> str:
    for column in df.columns:
        normalized = normalize_column_name(column)
        if all(token in normalized for token in required_tokens):
            return column
    raise ValueError(f"Could not find column containing tokens: {required_tokens}")


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def parse_time_column(series: pd.Series, timezone: str) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Could not parse {int(timestamps.isna().sum())} timestamps")

    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(timezone)
    return timestamps.dt.tz_convert("UTC")


def direct_horizontal_to_dni(
    direct_horizontal: pd.Series,
    timestamps_utc: pd.Series,
    latitude: float,
    longitude: float,
    min_cos_zenith: float,
) -> pd.Series:
    location = Location(latitude, longitude, tz="UTC")
    solar_position = location.get_solarposition(pd.DatetimeIndex(timestamps_utc))
    cos_zenith = np.cos(np.deg2rad(solar_position["zenith"].to_numpy(dtype=float)))

    values = direct_horizontal.to_numpy(dtype=float)
    dni = np.where(cos_zenith > min_cos_zenith, values / cos_zenith, 0.0)
    dni = np.clip(dni, 0.0, 1200.0)
    return pd.Series(dni, index=direct_horizontal.index)


def build_solar_frame(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv_path, sep=None, engine="python")

    time_col = find_column(df, ("time",))
    dhi_col = find_column(df, ("diffuse", "radiation"))
    ghi_col = find_column(df, ("shortwave", "radiation"))

    timestamps = parse_time_column(df[time_col], args.timezone)
    out = pd.DataFrame(index=timestamps)
    out.index.name = "time"
    out["dhi"] = parse_numeric(df[dhi_col]).clip(lower=0.0).to_numpy()
    out["ghi"] = parse_numeric(df[ghi_col]).clip(lower=0.0).to_numpy()

    try:
        dni_col = find_column(df, ("direct", "normal"))
        out["dni"] = parse_numeric(df[dni_col]).clip(lower=0.0).to_numpy()
        print(f"Using direct-normal column as DNI: {dni_col}")
    except ValueError:
        direct_col = find_column(df, ("direct", "radiation"))
        direct_horizontal = parse_numeric(df[direct_col]).clip(lower=0.0)
        if args.no_convert_direct_horizontal:
            out["dni"] = direct_horizontal.to_numpy()
            print(f"Using direct-radiation column as-is for DNI: {direct_col}")
        else:
            out["dni"] = direct_horizontal_to_dni(
                direct_horizontal,
                timestamps,
                args.latitude,
                args.longitude,
                args.min_cos_zenith,
            ).to_numpy()
            print(f"Converted direct horizontal radiation to DNI: {direct_col}")

    out = out[["dhi", "dni", "ghi"]]
    out = out.dropna(how="any")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def existing_interval_starts(
    client,
    module,
    entity_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> set[int]:
    start_s = start.isoformat().replace("+00:00", "Z")
    end_s = end.isoformat().replace("+00:00", "Z")
    query = (
        f'SELECT "value" '
        f'FROM "{module.INFLUX_CONFIG["database"]}"."{module.RETENTION_POLICY}"."{DEFAULT_MEASUREMENT}" '
        f"WHERE time >= '{start_s}' AND time <= '{end_s}' "
        f'AND "entity_id" = \'{entity_id}\''
    )
    result = client.query(query)
    points = list(result.get_points())
    if not points:
        return set()
    timestamps = pd.to_datetime([point["time"] for point in points], utc=True, format="mixed")
    interval_starts = timestamps.floor(interval)
    return set(interval_starts.view("int64"))


def build_points(df: pd.DataFrame, skip_existing: dict[str, set[int]]) -> list[dict]:
    points: list[dict] = []
    for entity_id, column in SOLAR_ENTITY_COLUMNS.items():
        existing = skip_existing.get(entity_id, set())
        for timestamp, value in df[column].items():
            if timestamp.value in existing:
                continue
            points.append(
                {
                    "measurement": DEFAULT_MEASUREMENT,
                    "tags": {"entity_id": entity_id},
                    "time": timestamp.isoformat().replace("+00:00", "Z"),
                    "fields": {"value": float(value)},
                }
            )
    return points


def write_in_batches(client, points: list[dict], batch_size: int) -> None:
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.write_points(batch, time_precision="s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Open-Meteo solar fill data to InfluxDB.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--latitude", type=float, default=51.65)
    parser.add_argument("--longitude", type=float, default=4.93)
    parser.add_argument("--interval", default="15min")
    parser.add_argument("--min-cos-zenith", type=float, default=0.065)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--no-convert-direct-horizontal", action="store_true")
    parser.add_argument("--write", action="store_true", help="Actually write to InfluxDB. Without this, dry-run only.")
    args = parser.parse_args()

    module = load_ha_import_module()
    df = build_solar_frame(args)
    print(f"Loaded {len(df)} rows from {args.csv_path}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")
    print(df.describe().round(2).to_string())

    client = module.get_client()
    skip_existing: dict[str, set[int]] = {}
    if not args.overwrite_existing:
        for entity_id in SOLAR_ENTITY_COLUMNS:
            existing = existing_interval_starts(
                client,
                module,
                entity_id,
                df.index.min(),
                df.index.max(),
                args.interval,
            )
            skip_existing[entity_id] = existing
            print(f"{entity_id}: {len(existing)} existing {args.interval} intervals will be skipped")

    points = build_points(df, skip_existing)
    print(f"Prepared {len(points)} points for measurement {DEFAULT_MEASUREMENT}")
    if points[:3]:
        print("First points:")
        for point in points[:3]:
            print(point)

    if not args.write:
        print("Dry-run only. Re-run with --write to upload.")
        return

    write_in_batches(client, points, args.batch_size)
    print(f"Uploaded {len(points)} points to InfluxDB.")


if __name__ == "__main__":
    main()
